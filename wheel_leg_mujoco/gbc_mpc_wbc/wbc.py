"""GBC486 的 WBC 任务集与接触块。

**控制方式照搬 circleg ``APP/mujoco_mpc_wbc.py``**：任务集、任务顺序、``hip_tasks``
归入 balance / motion 的方式、求解器工厂、以及 ``SupervisedControl`` 的调用方式都一致。

只做"换模型必须改"的三处：

1. ``height`` 任务作用在腿长共模 ``L_c`` 上（circleg 作用在机身高度 ``qpos[2]``：曲柄
   构型的机身高度由曲柄角直接决定，本构型的机身高度由腿长决定）；
2. ``hip_tracking`` / ``hip_differential`` 改名 ``leg_tracking`` / ``leg_differential``，
   作用在共模 / 差模腿角 ``φ_c`` / ``φ_d`` 上（circleg 作用在髋关节上）；
3. 权限因子 ``permission``：本腿构型没有曲柄死点，恒取 1。circleg 两个 activation
   下限的默认值也是 1.0，所以 ``max(排程, floor)`` 恒为 1，等价于不做排程，因此不需要
   移植 ``schedule`` 的裸函数。

6 状态模型（held_leg）对应 circleg 的 ``held_hip``：``hip_tasks`` 只有一项"保持标称腿角"，
它按 circleg 的写法被移到 balance 里。8 状态模型（leg_input）对应 ``hip_input``：
``hip_tasks = (leg_tracking, leg_differential)``，差模进 balance、共模留在 motion。

增益与权重都是 circleg 默认值的直接搬用，**未调参**。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _circleg  # noqa: F401
from .dynamics import Gbc486Dynamics, RowSet, leg_combination
from Algorithm.geometry import wrap_angle  # noqa: E402
from Algorithm.mpc import ChassisState  # noqa: E402
from Algorithm.supervision import ContactBlock  # noqa: E402
from Algorithm.wbc import TaskLevel  # noqa: E402
from Algorithm.wbc.schedule import height_task_activation_from_factor, roll_gain_scale_from_factor  # noqa: E402
from Algorithm.wbc.task_qp import task_qp_resolver_factory  # noqa: E402
from Algorithm.wbc.weighted import WeightedSaturationResolver  # noqa: E402
from Algorithm.wbc.sns import NullSpaceHierarchy, SaturationResolver  # noqa: E402
from .wbc_limited import LimitedWeightedResolver  # noqa: E402

# circleg ``_TASK_QP_WEIGHTS``（用于 WLS / QP 后端）：按用户 2026-10-04 指定的优先级重设
#   pitch > 腿角 > 位移/速度 > 腿长 > yaw > roll
TASK_WEIGHTS = {
    "pitch": 100.0,
    "angle_R": 60.0, "angle_L": 60.0,
    "forward_velocity": 40.0,
    "length_R": 25.0, "length_L": 25.0,
    "yaw": 15.0,
    "roll": 10.0,
    # E2/WP3（2026-10-09）：跳跃期 COMPRESS/THRUST 移除腿长任务后，com_vertical 取
    # 代其作为竖直方向主任务（分析文档 C2）。权重来源：沿用被替代的 length_R/length_L
    # 在既有具名权重表中的名次 25；未做任何试数搜索，待问题排查agent复查。
    "com_vertical": 25.0,
}

BACKENDS = ("nullspace", "wls", "qp")


@dataclass(frozen=True)
class WbcGains:
    """circleg ``APP/mujoco_mpc_wbc.py::SimulationConfig`` 里的增益字段（同名、同默认值）。"""

    attitude_gains: tuple[float, float] = (80.0, 10.0)
    height_gains: tuple[float, float] = (100.0, 12.0)
    regularize_gains: tuple[float, float] = (60.0, 6.0)
    tracking_velocity_gain: float = 14.0      # forward_mode = "mpc"（2026-09-29 默认，ω = 7）
    tracking_position_gain: float = 49.0
    acceleration_limit: float = 50.0
    height_activation_floor: float = 1.0
    roll_activation_floor: float = 1.0


@dataclass(frozen=True)
class Nominal:
    """站立参考：腿长共模、腿角共模、偏航（机身俯仰 / 横滚 / 差模参考为 0）。"""

    leg_length: float
    leg_angle: float
    yaw: float = 0.0


def resolver_factory(backend: str):
    """circleg 的求解器注入方式：默认零空间法（``_nullspace_resolver``）；
    qp 用 ``task_qp_resolver_factory``；wls 用加权最小二乘。"""
    if backend == "nullspace":
        # 与 circleg Algorithm/supervision.py::_nullspace_resolver 相同
        return lambda mass, elimination: SaturationResolver(NullSpaceHierarchy(mass), elimination)
    if backend == "qp":
        return task_qp_resolver_factory(TASK_WEIGHTS)
    if backend == "wls":
        # 2026-10-05：改用限幅自洽版本。原 WeightedSaturationResolver 越限后直接
        # clamp，使返回的 τ 与 q̈/接触力不再动力学一致，且不产出 outcomes（反馈回路
        # 失效）。详见 gbc_mpc_wbc/wbc_limited.py 的模块说明。
        return lambda mass, elimination: LimitedWeightedResolver(elimination, TASK_WEIGHTS)
    raise ValueError(f"未知 WBC 后端 {backend}")


class Gbc486Tasks:
    """circleg ``MuJoCoMpcWbcSimulation.tasks`` / ``MuJoCoDynamics.matrices`` 的等价实现。"""

    def __init__(self, dynamics: Gbc486Dynamics, nominal: Nominal, gains: WbcGains = WbcGains(),
                 leg_input: bool = False, state_dim: int = 6):
        self.dyn = dynamics
        self.nominal = nominal
        self.gains = gains
        self.leg_input = leg_input
        self.state_dim = int(state_dim)

    # ------------------------------------------------------------ 接触块
    def contact_blocks(self, rows: RowSet, contact: tuple[bool, bool]) -> tuple[ContactBlock, ...]:
        """circleg blocks 部分 + 闭链 8 行。

        接触与 circleg 一致：每轮 ``[切向, 法向]`` 一块，再加一块共模横向（只含着地轮）。
        闭链行是本构型独有的，与接触状态无关，必须始终参与硬约束：块的 ``wheels`` 声明为
        “当前着地的轮子”，这样单支撑时不会被 circleg 的块筛选
        （``Algorithm/supervision.py`` 只保留 wheels 全在接触中的块）丢掉；腾空时不求解
        WBC，声明 ``(0, 1)`` 只作占位。
        """
        blocks = []
        lateral_rows, lateral_bias = [], []
        for side, active in enumerate(contact):
            if not active:
                continue
            wheel_rows, wheel_bias = rows.wheel(side)
            blocks.append(ContactBlock(wheel_rows[:2], wheel_bias[:2], (side,)))
            lateral_rows.append(wheel_rows[2])
            lateral_bias.append(wheel_bias[2])
        standing = tuple(i for i, a in enumerate(contact) if a)
        if lateral_rows:
            blocks.append(ContactBlock(np.mean(lateral_rows, axis=0)[None, :],
                                       np.array([np.mean(lateral_bias)]), standing))
        loop_rows, loop_bias = rows.loop
        blocks.append(ContactBlock(loop_rows, loop_bias, standing or (0, 1)))
        return tuple(blocks)

    @staticmethod
    def normal_indices(contact: tuple[bool, bool]) -> tuple[int, ...]:
        """堆叠后的约束力向量里法向分量的下标（每轮 ``[切, 法]`` 在前）。"""
        return tuple(range(1, 2 * sum(bool(c) for c in contact), 2))

    def contact_forces(self, recovered: np.ndarray, contact: tuple[bool, bool]) -> np.ndarray:
        """从反解结果里取出逐轮 ``[切向, 法向]``，形状 ``(着地轮数, 2)``。

        行排布的知识只在本类里维护：``contact_blocks`` 决定行顺序（每轮两块 ``[切, 法]`` 排在最前），
        采样级与 WBC 级都调这里，避免两处各写一份映射（第 15.1 条：一份逻辑一个持有者）。
        注意：横向（y）在本构型是**共模**一块（``contact_blocks`` 取两轮横向行的均值），
        不按轮分开，所以这里只给切向与法向。
        """
        count = int(sum(bool(c) for c in contact))
        return np.asarray(recovered, dtype=float)[:2 * count].reshape(count, 2)

    def contact_normals(self, recovered: np.ndarray, contact: tuple[bool, bool]) -> np.ndarray:
        """逐轮法向力（``contact_forces`` 的第 2 列）。"""
        return self.contact_forces(recovered, contact)[:, 1]

    # ------------------------------------------------------------ 任务
    def tasks(self, state: ChassisState, rows: RowSet, sample=None,
              desired_com_acceleration_z: float | None = None,
              pitch_reference: float | None = None, roll_reference: float | None = None,
              angle_R_reference: float | None = None, angle_L_reference: float | None = None):
        """返回 ``(balance_tasks, motion_tasks)``。顺序与 circleg 完全一致：

        ``(pitch, hip_tasks[-1], roll)`` + ``(yaw, height, forward_velocity) + hip_tasks[:-1]``，
        其中 6 状态模型的 ``hip_tasks = (leg_tracking标称,)``、8 状态模型为
        ``(leg_tracking跟踪MPC, leg_differential)``。

        ``desired_com_acceleration_z``（E2/WP2）为 None 时保持原任务列表不变（普通 GROUND
        路径）；给定时额外追加唯一名为 ``com_vertical`` 的质心竖直加速度任务
        ``J_com_z q̈ + bias_com_z = desired_com_acceleration_z``，Jacobian 与 bias 来自
        E1 通过的 ``Gbc486Dynamics.com_vertical_row()``。目标是显式诊断/跳跃参考，不从
        MPC ``sample`` 隐式借用。

        新部署（2026-10-09）：``pitch_reference`` / ``roll_reference`` /
        ``angle_R_reference`` / ``angle_L_reference`` 为非 None 时，对应任务目标改为保持该
        显式参考（由调用方在跳跃开始时捕获一次），用于冻结压缩期姿态/腿角参考的单变量实验。
        默认 None 时保持原有参考逻辑不变。
        """
        data = self.dyn.data
        nv = self.dyn.model.nv
        kp, kd = self.gains.attitude_gains
        kp_h, kd_h = self.gains.height_gains
        kp_r, kd_r = self.gains.regularize_gains
        task_rows, task_bias = rows.task
        roll, pitch, yaw = self.dyn.attitude()
        leg = self.dyn.leg_state()
        common_rows, common_bias = leg_combination(rows, +1.0)
        diff_rows, diff_bias = leg_combination(rows, -1.0)
        common = 0.5 * (leg[[0, 1]] + leg[[2, 3]])
        common_rate = common_rows @ data.qvel
        diff = 0.5 * (leg[[0, 1]] - leg[[2, 3]])
        diff_rate = diff_rows @ data.qvel
        n = self.nominal

        reference = (state.vector(self.state_dim) if sample is None else sample.state)
        acceleration = (np.zeros(self.state_dim) if sample is None else sample.acceleration)
        if sample is None:
            reference = reference.copy()
            reference[2:6] = 0.0
        pitch_target = acceleration[1] + kp * (reference[2] - state.pitch) + kd * (reference[3] - state.pitch_rate)
        roll_target = -kp * roll - kd * (task_rows[1] @ data.qvel)
        # 显式冻结参考（非 None）覆盖上面两个默认目标（本轮不改变默认 None 的行为）。
        if pitch_reference is not None:
            pitch_target = kp * (pitch_reference - state.pitch) - kd * state.pitch_rate
        if roll_reference is not None:
            roll_target = kp * (roll_reference - roll) - kd * (task_rows[1] @ data.qvel)
        yaw_target = acceleration[2] + kp * wrap_angle(reference[4] - state.yaw) + kd * (reference[5] - state.yaw_rate)
        if sample is None:
            # 非 GROUND：没有 MPC 轨迹，前进速度只做阻尼（circleg 的 sample is None 分支同式）
            velocity_target = -self.gains.tracking_velocity_gain * state.velocity
        else:
            velocity_target = (acceleration[0] + self.gains.tracking_velocity_gain * (reference[1] - state.velocity)
                               + self.gains.tracking_position_gain * (reference[0] - state.position))
        velocity_target = float(np.clip(velocity_target, -self.gains.acceleration_limit,
                                        self.gains.acceleration_limit))

        # 权限因子：本构型没有死点，permission = 1（与 circleg 默认 floor = 1.0 一致，激活恒为 1）
        authority = 1.0
        balance = (
            TaskLevel(task_rows[[2]], np.array([pitch_target]), task_bias[[2]], name="pitch"),
        )
        # 每条腿单独建模（用户 2026-10-04）：L_R、φ_R、L_L、φ_L 各自一个任务，
        # 不再用共模/差模组合。两条腿的腿长各自回 L0，对称性由任务自己保证。
        leg_rows, leg_bias = rows.leg
        leg_tasks = []
        for side, (length_index, angle_index) in enumerate(((0, 1), (2, 3))):
            suffix = "R" if side == 0 else "L"
            length_rate = float(leg_rows[length_index] @ data.qvel)
            # 符号与 MPC 状态一致：δφ_c := −(极角 − φ0)（见 model_builder.set_legs）。
            # 因此腿角任务的**行、实测位置、实测速率**三者同时取负；只翻其中一项会让任务自相矛盾。
            angle_row = -leg_rows[[angle_index]]
            angle_bias = -leg_bias[[angle_index]]
            angle_measured = n.leg_angle - float(leg[angle_index])
            angle_rate = -float(leg_rows[angle_index] @ data.qvel)
            length_target = kp_h * (n.leg_length - float(leg[length_index])) - kd_h * length_rate
            angle_reference = angle_R_reference if side == 0 else angle_L_reference
            if angle_reference is not None:
                angle_target = kp_r * (angle_reference - angle_measured) - kd_r * angle_rate
            elif self.leg_input and sample is not None:
                angle_target = (acceleration[3]
                                + kp_r * (reference[6] - angle_measured)
                                + kd_r * (reference[7] - angle_rate))
            else:
                angle_target = kp_r * (0.0 - angle_measured) - kd_r * angle_rate
            leg_tasks.append((
                TaskLevel(leg_rows[[length_index]], np.array([length_target]),
                          leg_bias[[length_index]], name=f"length_{suffix}",
                          activation=max(height_task_activation_from_factor(authority),
                                         self.gains.height_activation_floor)),
                TaskLevel(angle_row, np.array([angle_target]),
                          angle_bias, name=f"angle_{suffix}"),
            ))
        # 层级顺序（用户 2026-10-04）：pitch > 腿角 > 位移/速度 > 腿长 > yaw > roll
        balance = (balance[0],)
        motion = (
            leg_tasks[0][1], leg_tasks[1][1],
            TaskLevel(task_rows[[0]], np.array([velocity_target]), task_bias[[0]], name="forward_velocity"),
            leg_tasks[0][0], leg_tasks[1][0],
            TaskLevel(task_rows[[3]], np.array([yaw_target]), task_bias[[3]], name="yaw"),
            TaskLevel(task_rows[[1]], np.array([roll_target]), task_bias[[1]], name="roll",
                      activation=max(roll_gain_scale_from_factor(authority),
                                     self.gains.roll_activation_floor)),
        )
        if desired_com_acceleration_z is not None:
            com_row, com_bias, _com_z, _com_vz = self.dyn.com_vertical_row()
            motion = motion + (
                TaskLevel(np.atleast_2d(np.asarray(com_row, dtype=float)),
                          np.array([float(desired_com_acceleration_z)]),
                          np.array([float(com_bias)]), name="com_vertical"),
            )
        return balance, motion
