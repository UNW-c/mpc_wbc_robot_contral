"""GBC486 的 WBC 适配层（M1）：选择矩阵、轮接触行、闭链行、腿空间行与偏置。

对应 circleg 的 ``APP/mujoco_mpc_wbc.py::MuJoCoDynamics``，差异：

- 选择矩阵乘上 ``gainprm[0]``（左轮执行器 gain = −1），因此 WBC 的力矩变量就是
  ``data.ctrl``，限值直接取 ``actuator_ctrlrange``；
- 增加闭链约束行：每个 ``connect`` 取 site 差在机身 x / z 轴上的投影（共 8 行）。
  平面外 y 行近似冗余（奇异值约 1e-4），不进入 WBC；
- 增加腿空间行 ``[L_R, φ_R, L_L, φ_L]``（定义见 ``robot.py``），用于 height / roll /
  leg_tracking / leg_differential 任务。

所有行对 ``qvel`` 线性：``ẏ = J q̇``；加速度层 ``ÿ = J q̈ + bias``，bias 由
``mj_integratePos`` 中心差分得到，与 circleg 的做法相同。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import _circleg  # noqa: F401  （sys.path / 字节码设置）
from . import robot as R
from Algorithm.wbc import ActuatorLimits  # noqa: E402

_DIFFERENCE_DT = 1.0e-6

# 行在堆叠矩阵中的位置
TASK = slice(0, 4)          # [heading 速度, roll 率, pitch 率, yaw 率]
CONTACT = slice(4, 10)      # 每轮 [heading, normal, lateral]，右轮在前
LOOP = slice(10, 18)        # 右腿 2 个 connect × (x, z)，再左腿
LEG = slice(18, 22)         # [L_R, φ_R, L_L, φ_L]
ROW_COUNT = 22


@dataclass(frozen=True)
class RowSet:
    """一次评估得到的全部行（``ROW_COUNT × nv``）及其加速度偏置。"""

    rows: np.ndarray
    bias: np.ndarray

    @property
    def task(self):
        return self.rows[TASK], self.bias[TASK]

    @property
    def loop(self):
        return self.rows[LOOP], self.bias[LOOP]

    @property
    def leg(self):
        return self.rows[LEG], self.bias[LEG]

    def wheel(self, side: int):
        """``side`` 0 = 右、1 = 左；返回 3 行 [heading, normal, lateral]。"""
        index = slice(CONTACT.start + 3 * side, CONTACT.start + 3 * side + 3)
        return self.rows[index], self.bias[index]

    def constraints(self, contact: tuple[bool, bool]):
        """WBC 硬约束行：接触在前、闭链在后。

        接触排列与 circleg 一致：``[切R, 法R, 切L, 法L, 共模横]``（只含着地轮），
        所以 ``TaskQpResolver`` 的 ``normal_indices`` 在双支撑时仍是 ``(1, 3)``。
        闭链行与接触状态无关，始终存在。
        """
        rows, bias, lateral_rows, lateral_bias = [], [], [], []
        for side, active in enumerate(contact):
            if not active:
                continue
            j, b = self.wheel(side)
            rows.append(j[:2])
            bias.append(b[:2])
            lateral_rows.append(j[2])
            lateral_bias.append(b[2])
        if lateral_rows:
            rows.append(np.mean(lateral_rows, axis=0)[None, :])
            bias.append(np.array([np.mean(lateral_bias)]))
        loop_rows, loop_bias = self.loop
        rows.append(loop_rows)
        bias.append(loop_bias)
        return np.vstack(rows), np.concatenate(bias)

    def contact_row_count(self, contact: tuple[bool, bool]) -> int:
        active = sum(bool(c) for c in contact)
        return 2 * active + (1 if active else 0)


def leg_combination(rows: RowSet, sign: float):
    """腿空间行的共模 / 差模组合。

    ``sign = +1``：共模 ``[L_c, φ_c] = 0.5·[(L_R + L_L), (φ_R + φ_L)]``；
    ``sign = -1``：差模 ``[L_d, φ_d] = 0.5·[(L_R − L_L), (φ_R − φ_L)]``。
    返回 ``(rows, bias)``，各 2 行。
    """
    leg, bias = rows.leg
    return (0.5 * (leg[[0, 1]] + sign * leg[[2, 3]]),
            0.5 * (bias[[0, 1]] + sign * bias[[2, 3]]))


def euler_zyx(quaternion) -> np.ndarray:
    """``[roll, pitch, yaw]``，与 ``caculation.orientation2euler`` 同一公式。"""
    w, x, y, z = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class Gbc486Dynamics:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.scratch = mujoco.MjData(model)
        self.floor = int(model.geom("floor").id)
        self.wheel_geoms = tuple(int(model.geom(leg.wheel_geom).id) for leg in R.LEGS)
        self.wheel_bodies = tuple(int(model.body(leg.wheel_body).id) for leg in R.LEGS)
        self.hip_bodies = tuple(int(model.body(leg.hip_body).id) for leg in R.LEGS)
        self.base = int(model.body("base").id)
        self.loop_sites = tuple((int(model.site(a).id), int(model.site(b).id))
                                for leg in R.LEGS for a, b in leg.loops)
        self.selection = self._selection()
        self.limits = ActuatorLimits(model.actuator_ctrlrange[:, 0].copy(),
                                     model.actuator_ctrlrange[:, 1].copy())
        self.leg_actuators = tuple(int(model.actuator(n).id) for n in (
            "Right_front_joint_act", "Right_rear_joint_act", "Left_front_joint_act", "Left_rear_joint_act"))
        self.wheel_actuators = tuple(int(model.actuator(n).id) for n in ("Right_Wheel_act", "Left_Wheel_act"))

    # ------------------------------------------------------------------ model
    def _selection(self) -> np.ndarray:
        m = self.model
        selection = np.zeros((m.nu, m.nv))
        for a in range(m.nu):
            if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(f"执行器 {m.actuator(a).name} 不是关节传动")
            if m.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED or \
                    m.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_NONE:
                raise ValueError(f"执行器 {m.actuator(a).name} 不是纯力矩源，选择矩阵不成立")
            joint = m.actuator_trnid[a, 0]
            selection[a, m.jnt_dofadr[joint]] = m.actuator_gear[a, 0] * m.actuator_gainprm[a, 0]
        return selection

    def mass(self) -> np.ndarray:
        matrix = np.empty((self.model.nv, self.model.nv))
        if tuple(map(int, mujoco.__version__.split(".")[:2])) >= (3, 13):
            mujoco.mj_fullM(self.model, self.data, matrix)
        else:
            mujoco.mj_fullM(self.model, matrix, self.data.qM)
        return matrix

    def dynamics_bias(self) -> np.ndarray:
        """``h = qfrc_bias − qfrc_passive``（被动力含流体力，在 MuJoCo 方程右端）。"""
        return self.data.qfrc_bias - self.data.qfrc_passive

    # ------------------------------------------------------------------- rows
    def _evaluate(self, data) -> np.ndarray:
        """在 ``data`` 当前位形上计算全部行（需要 kinematics + comPos 已更新）。"""
        m = self.model
        nv = m.nv
        roll, pitch, yaw = euler_zyx(data.qpos[3:7])
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        lateral = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        out = np.zeros((ROW_COUNT, nv))
        out[0, :3] = heading
        # free joint 的角速度 qvel[3:6] 在机体系
        sr, cr, tp, cp = math.sin(roll), math.cos(roll), math.tan(pitch), math.cos(pitch)
        out[1:4, 3:6] = [[1.0, sr * tp, cr * tp], [0.0, cr, -sr], [0.0, sr / cp, cr / cp]]
        jac = np.zeros((3, nv))
        down = np.array([0.0, 0.0, -1.0])
        for side, geom in enumerate(self.wheel_geoms):
            axle = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            radial = down - axle * (axle @ down)
            radial /= np.linalg.norm(radial)
            point = data.geom_xpos[geom] + m.geom_size[geom, 0] * radial
            mujoco.mj_jac(m, data, jac, None, point, m.geom_bodyid[geom])
            row = CONTACT.start + 3 * side
            out[row] = heading @ jac
            out[row + 1] = jac[2]
            out[row + 2] = lateral @ jac
        rot = data.xmat[self.base].reshape(3, 3)
        j1, j2 = np.zeros((3, nv)), np.zeros((3, nv))
        for k, (s1, s2) in enumerate(self.loop_sites):
            mujoco.mj_jacSite(m, data, j1, None, s1)
            mujoco.mj_jacSite(m, data, j2, None, s2)
            diff = j1 - j2
            out[LOOP.start + 2 * k] = rot[:, 0] @ diff
            out[LOOP.start + 2 * k + 1] = rot[:, 2] @ diff
        jw, jh = np.zeros((3, nv)), np.zeros((3, nv))
        for side, (wheel, hip) in enumerate(zip(self.wheel_bodies, self.hip_bodies)):
            mujoco.mj_jacBody(m, data, jw, None, wheel)
            mujoco.mj_jacBody(m, data, jh, None, hip)
            relative = rot.T @ (jw - jh)
            # 机身系相对位置只依赖关节角：机身角速度的 ω×r 项与坐标系旋转项恰好抵消
            relative[:, 3:6] = 0.0
            r = rot.T @ (data.xpos[wheel] - data.xpos[hip])
            length2 = r[0] ** 2 + r[2] ** 2
            length = math.sqrt(length2)
            out[LEG.start + 2 * side] = (r[0] * relative[0] + r[2] * relative[2]) / length
            out[LEG.start + 2 * side + 1] = (-r[2] * relative[0] + r[0] * relative[2]) / length2
        return out

    def _evaluate_at(self, qpos) -> np.ndarray:
        s = self.scratch
        s.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_comPos(self.model, s)
        return self._evaluate(s)

    def rowset(self) -> RowSet:
        """当前 ``data`` 状态的全部行与偏置。调用前 ``data`` 需已 ``mj_forward``。"""
        data = self.data
        rows = self._evaluate(data)
        plus, minus = data.qpos.copy(), data.qpos.copy()
        mujoco.mj_integratePos(self.model, plus, data.qvel, _DIFFERENCE_DT)
        mujoco.mj_integratePos(self.model, minus, data.qvel, -_DIFFERENCE_DT)
        bias = (self._evaluate_at(plus) - self._evaluate_at(minus)) @ data.qvel / (2 * _DIFFERENCE_DT)
        return RowSet(rows, bias)

    # ---------------------------------------------------------------- sensing
    def attitude(self) -> np.ndarray:
        return euler_zyx(self.data.qpos[3:7])

    def leg_state(self) -> np.ndarray:
        """[L_R, phi_R, L_L, phi_L]：腿长与腿摆角，由关节角**解析推导**。

        几何常数在首次调用时从模型一次性读出（``LegKinematics``），之后每拍只做解析
        链式组合，不再调用 MuJoCo 的位置接口（``data.xpos``）。旧实现保留在
        ``robot.leg_polar``，仅作对照，不进运行链路。
        """
        if getattr(self, "_leg_kinematics", None) is None:
            from .leg_kinematics import LegKinematics
            self._leg_kinematics = {leg.name: LegKinematics(self.model, leg) for leg in R.LEGS}
        values: list[float] = []
        for leg in R.LEGS:
            angles = {name: float(self.data.qpos[self.model.joint(name).qposadr[0]])
                      for name in (leg.front_joint, leg.rear_joint) + leg.passive_joints}
            values.extend(self._leg_kinematics[leg.name].lengths(angles))
        return np.asarray(values)

    def sensed_contact(self):
        """逐轮汇总轮–地接触力（作用在轮上，世界系），以及非轮–地接触数。"""
        m, data = self.model, self.data
        forces = np.zeros((2, 3))
        counts = np.zeros(2, dtype=int)
        other = 0
        wrench = np.zeros(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = (int(contact.geom1), int(contact.geom2))
            if self.floor not in pair:
                other += 1
                continue
            geom = pair[1] if pair[0] == self.floor else pair[0]
            if geom not in self.wheel_geoms:
                other += 1
                continue
            side = self.wheel_geoms.index(geom)
            mujoco.mj_contactForce(m, data, index, wrench)
            sign = 1.0 if pair[1] == geom else -1.0
            forces[side] += sign * contact.frame.reshape(3, 3).T @ wrench[:3]
            counts[side] += 1
        active = tuple(bool(n > 0 and f[2] > 0) for n, f in zip(counts, forces))
        return active, forces, counts, other

    def joint_limit_active(self) -> list[str]:
        """当前被 MuJoCo 关节限位约束顶住的关节名。"""
        data = self.data
        names = []
        for i in range(data.nefc):
            if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
                names.append(self.model.joint(int(data.efc_id[i])).name)
        return names
