"""采样级（每个物理步一次）：把 MuJoCo 状态读成控制层可用的反馈。

这一级做什么
------------
- 刷新运动学（``mj_forward``）、取逐轮法向力
- 离地判定：逐轮法向力交给**算法层** ``GroundContactEstimator``（滞回 + 确认时间）；
  不在这里用裸阈值——几十微米间隙下 MuJoCo 会关接触，裸阈值会把瞬时卸载误判成离地
- 组装降阶模型状态 ``ChassisState`` 与任务行集
- 监督层模式（接地 / 恢复，是否允许机动）

这一级不做什么
--------------
- 不出力矩、不推进物理、不碰执行器
- 不持有 MPC / WBC（那是 ``control.py``）；本模块被它依赖，不反向依赖
- 不配置日志（见 ``logging_setup.py``）

节拍：每物理步调用一次（高频）。``dt`` 必须给**真实物理步长**，否则估计器与监督层的
秒级滞回会被放大。
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import _circleg  # noqa: F401
from .dynamics import leg_combination  # noqa: E402
from .logging_setup import get_logger  # noqa: E402
from Algorithm.contact_state import JumpPhaseHint  # noqa: E402
from Algorithm.mpc import ChassisState  # noqa: E402

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class Feedback:
    """一拍的反馈快照。字段全是读出来的量，不含任何控制决策。"""

    contact: tuple          # 逐轮接地判定（算法层估计器给出）
    state: object           # ChassisState：降阶模型状态（含里程计）
    rows: object            # 当前接触下的任务行集
    decision: object        # 监督层模式（含 motion_enabled / contact）
    forces: np.ndarray      # 逐轮法向力（诊断与记录用）
    other: object           # 非轮体接触等异常信息（停机判据用）


def chassis_state(*, dynamics, data, contact, forces, distance: float,
                  equilibrium_pitch: float, nominal_angle: float):
    """由接触判定与法向力组装 ``ChassisState``，同时给出任务行集。"""
    rows = dynamics.rowset()
    rates = rows.task[0] @ data.qvel
    roll, pitch, yaw = dynamics.attitude()
    leg = dynamics.leg_state()
    leg_rows, _ = leg_combination(rows, +1.0)
    return ChassisState(
        position=distance, velocity=float(rates[0]),
        pitch=float(pitch) - equilibrium_pitch,
        pitch_rate=float(rates[2]), yaw=float(yaw), yaw_rate=float(rates[3]),
        # 降阶模型第 6 个状态是共模腿角偏差 phi_c，反馈也取共模 0.5*(phi_R+phi_L)-phi0。
        # 用单腿（leg[1]）会让 MPC 与 WBC 的腿角反馈带上一份 phi_d 的系统误差。
        # 符号见 model_builder.LegInputModelBuilder.set_legs：delta_phi_c := -(极角 - phi0)。
        # 算法层依赖 d(v_dot)/d(phi_c) > 0，所以这里也要取负，否则状态与参考不同号。
        hip_angle=float(nominal_angle - 0.5 * (leg[1] + leg[3])),
        hip_rate=-float(leg_rows[1] @ data.qvel),
        normal_forces=tuple(float(v) for v in forces[:, 2]), contact=contact), rows


class FeedbackSampler:
    """采样级的执行者：每次 ``update`` 取一拍反馈，内部不缓存控制量。"""

    def __init__(self, *, model, data, dynamics, tasks, contact_estimator, supervisor,
                 reduced, nominal_angle: float, pitch_limit: float) -> None:
        self.model = model
        self.data = data
        self.dynamics = dynamics
        # 接触行排布的知识归任务集持有（contact_blocks / contact_normals），采样级只消费
        self.tasks = tasks
        self.contact_estimator = contact_estimator
        self.supervisor = supervisor
        self.last_forces: np.ndarray | None = None    # 上一拍可信的逐轮力（反解不可信时沿用）
        self.dt = float(model.opt.timestep)
        self.equilibrium_pitch = float(reduced.metadata["equilibrium_pitch"])
        self.nominal_angle = float(nominal_angle)
        self.pitch_limit = float(pitch_limit)

    def update(self, distance: float, phase_hint: JumpPhaseHint = JumpPhaseHint.NONE) -> Feedback:
        """取一拍反馈：运动学 -> 接触几何 -> 力雅可比反解接触力 -> 接触判定 -> 状态/监督层。

        接触力的口径（用户 2026-10-07）：**不用 MuJoCo 的 ``mj_contactForce``**，而是由动力学
        方程反解**实际状态**（``data.qacc`` + 实际施加的 ``data.ctrl``）：
        ``f = R⁻¹ Q_cᵀ (M q̈ + h − Sᵀ τ − τ_ext)``。WBC 级用的是同一入口，只是输入换成它自己解出的
        ``result.qacc``/力矩（那是**预测**，不能用于离地判定）。
        """
        mujoco.mj_forward(self.model, self.data)
        active, counts, other = self.dynamics.contact_geometry()
        rows = self.dynamics.rowset()
        forces = np.zeros((2, 3))
        flags = tuple(bool(n) for n in counts)
        if any(flags):
            blocks = self.tasks.contact_blocks(rows, flags)
            recovered = self.dynamics.recover_contact_force(
                blocks, self.data.qacc, self.data.ctrl, external=self.data.qfrc_applied)
            values = self.tasks.contact_forces(recovered, flags)
            for pair, side in zip(values, [i for i, f in enumerate(flags) if f]):
                # [切向, 法向]：切向给 x 分量，法向给 z（横向 y 在本构型是共模量，不按轮拆分）
                forces[side, 0] = float(pair[0])
                forces[side, 2] = float(pair[1])
        if self.dynamics.contact_untrusted():
            # 关节限位等非接触约束顶住时反解会把限位约束力算进接触力：本拍沿用上一拍可信值，
            # 不让估计器按假证据翻转（计数在 control 级记 negative_normal/torque，这里只保状态）
            forces = self.last_forces if self.last_forces is not None else forces
        else:
            self.last_forces = forces
        contact = self.contact_estimator.update(
            tuple(forces[:, 2]), self.dt, phase=phase_hint,
            geometry_contact=tuple(bool(v) for v in self.dynamics.contact_geometry()[0])).contact
        state, rows = chassis_state(
            dynamics=self.dynamics, data=self.data, contact=contact, forces=forces,
            distance=distance, equilibrium_pitch=self.equilibrium_pitch,
            nominal_angle=self.nominal_angle)
        roll, pitch, _ = self.dynamics.attitude()
        decision = self.supervisor.update(
            timestamp=float(self.data.time), dt=self.dt, contact=contact, enabled=True,
            command_fresh=True, stable=max(abs(roll), abs(pitch)) < self.pitch_limit,
            slipping=False)
        return Feedback(contact=contact, state=state, rows=rows, decision=decision,
                        forces=forces, other=other)
