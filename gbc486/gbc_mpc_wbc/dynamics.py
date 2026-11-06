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

from . import _paths  # noqa: F401
from . import robot as R
from Algorithm.wbc import ActuatorLimits  # noqa: E402
from Algorithm.wbc.projection import ContactElimination  # noqa: E402
from .logging_setup import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)

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


@dataclass(frozen=True)
class LegKinematicObservation:
    """独立的腿部运动学观测，顺序固定为 R 长度、R 摆角、L 长度、L 摆角。

    位置量来自解析闭环运动学，速度量来自同一当前状态的腿部 Jacobian
    与 ``qvel``。跳跃状态机、WBC 任务和记录器都应通过该接口读取腿部状态，
    不应各自重新从不同的 MuJoCo 字段计算。
    """

    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float).reshape(-1)
        velocity = np.asarray(self.velocity, dtype=float).reshape(-1)
        if position.shape != (4,) or velocity.shape != (4,):
            raise ValueError("腿部运动学观测必须是 [L_R, phi_R, L_L, phi_L] 的四维向量")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("腿部运动学观测不能包含 NaN 或 Inf")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)


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
        # PERF（第 22 节）：按物理步复用同一次计算的结果。缓存的两条硬约束：
        #   1) 生命周期 = 一个物理步（``begin_step`` 清空，``run_loop`` 每拍开头调用）；
        #   2) 复用判据是 (qpos, qvel) 的字节快照完全一致 —— 任何状态改动（mj_step /
        #      积分 / 直接写 qpos）都会自动 miss，不需要依赖调用方记得失效。
        # 缓存返回的是首次算出的原对象，不重排运算、不改 dtype 与形状，所以逐位一致。
        self._cache_signature: tuple[bytes, bytes] | None = None
        self._cache_rowset: RowSet | None = None
        self._cache_joint_limit: tuple[tuple, list[str]] | None = None
        self._cache_lengths: dict[tuple, tuple[float, float]] = {}
        self.cache_stats = {"steps": 0, "rowset_recompute": 0, "rowset_hit": 0,
                            "joint_limit_recompute": 0, "joint_limit_hit": 0,
                            "lengths_recompute": 0, "lengths_hit": 0}

    # ------------------------------------------------------------------ cache
    def begin_step(self) -> None:
        """一个物理步开始：清空本拍缓存（缓存生命周期 = 一个物理步）。"""
        self._cache_signature = None
        self._cache_rowset = None
        self._cache_joint_limit = None
        self._cache_lengths.clear()
        self.cache_stats["steps"] += 1

    def _signature(self) -> tuple[bytes, bytes]:
        """当前 (qpos, qvel) 的字节快照，用作"状态是否变过"的判据。"""
        return (self.data.qpos.tobytes(), self.data.qvel.tobytes())

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
        """当前 ``data`` 状态的全部行与偏置。调用前 ``data`` 需已 ``mj_forward``。

        PERF：同一物理步内重复调用直接返回首次结果（多个采样入口共用，行集是纯函数）。
        """
        signature = self._signature()
        if self._cache_rowset is not None and signature == self._cache_signature:
            self.cache_stats["rowset_hit"] += 1
            return self._cache_rowset
        data = self.data
        rows = self._evaluate(data)
        plus, minus = data.qpos.copy(), data.qpos.copy()
        mujoco.mj_integratePos(self.model, plus, data.qvel, _DIFFERENCE_DT)
        mujoco.mj_integratePos(self.model, minus, data.qvel, -_DIFFERENCE_DT)
        bias = (self._evaluate_at(plus) - self._evaluate_at(minus)) @ data.qvel / (2 * _DIFFERENCE_DT)
        result = RowSet(rows, bias)
        self._cache_rowset = result
        self._cache_signature = signature
        self.cache_stats["rowset_recompute"] += 1
        return result



    def com_vertical_row(self) -> tuple[np.ndarray, float, float, float]:
        """Return the base-subtree center-of-mass observation in world z."""
        data = self.data
        model = self.model
        mujoco.mj_comPos(model, data)
        body_id = int(model.body("base").id)
        jacobian = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSubtreeCom(model, data, jacobian, body_id)

        plus = data.qpos.copy()
        minus = data.qpos.copy()
        mujoco.mj_integratePos(model, plus, data.qvel, _DIFFERENCE_DT)
        mujoco.mj_integratePos(model, minus, data.qvel, -_DIFFERENCE_DT)
        plus_com = self._evaluate_com_position(plus)
        minus_com = self._evaluate_com_position(minus)
        com_position = np.asarray(data.subtree_com[body_id], dtype=float)
        center_com = np.asarray(data.subtree_com[body_id], dtype=float)
        bias = float((plus_com[2] - 2.0 * center_com[2] + minus_com[2])
                     / (_DIFFERENCE_DT * _DIFFERENCE_DT))
        row = np.asarray(jacobian[2], dtype=float).copy()
        com_velocity = float(row @ data.qvel)
        return row, bias, float(com_position[2]), com_velocity

    # ------------------------------------------------------------------ 角动量
    def pitch_momentum_row(self) -> tuple[np.ndarray, float, float]:
        """整机绕总质心的俯仰角动量观测 ``(A_H, bias, H)``，``H = A_H @ qvel``。

        文档 10.48 A0 的口径：
          ``A_H = sum_i [ (I_i*Jrot_i)_y + m_i*(r_z*Jpos_i,x - r_x*Jpos_i,z) ]``，
          ``r = xipos_i - 整机质心``，``Jpos/Jrot`` 用 ``mj_jacBodyCom``。
        ``bias = (A_H(q+eps*qvel) - A_H(q-eps*qvel)) @ qvel / (2*eps)``，与
        ``rowset()`` 里其它行的 bias 写法一致（同一 ``_DIFFERENCE_DT``）。
        """
        row = self._momentum_row_from(self.data.qpos)
        minus = self.data.qpos.copy()
        plus = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, plus, self.data.qvel, _DIFFERENCE_DT)
        mujoco.mj_integratePos(self.model, minus, self.data.qvel, -_DIFFERENCE_DT)
        row_plus = self._momentum_row_from(plus)
        row_minus = self._momentum_row_from(minus)
        bias = float(((row_plus - row_minus) @ self.data.qvel) / (2.0 * _DIFFERENCE_DT))
        return row, bias, float(row @ self.data.qvel)

    def _momentum_row_from(self, qpos: np.ndarray) -> np.ndarray:
        """在 scratch data 上算 CMM 的俯仰行（不改动 live data）。"""
        model, scratch = self.model, self.scratch
        scratch.qpos[:] = qpos
        scratch.qvel[:] = 0.0
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        centre = np.asarray(scratch.subtree_com[0], dtype=float)
        row = np.zeros(model.nv, dtype=float)
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        for index in range(1, model.nbody):
            jacp[:] = 0.0
            jacr[:] = 0.0
            mujoco.mj_jacBodyCom(model, scratch, jacp, jacr, index)
            rot = scratch.ximat[index].reshape(3, 3)
            inertia = rot @ np.diag(model.body_inertia[index]) @ rot.T
            offset = np.asarray(scratch.xipos[index], dtype=float) - centre
            row += inertia[1, :] @ jacr
            row += model.body_mass[index] * (offset[2] * jacp[0] - offset[0] * jacp[2])
        return row

    def _evaluate_com_position(self, qpos: np.ndarray) -> np.ndarray:
        """Evaluate the same base-subtree COM definition on scratch data."""
        scratch = self.scratch
        scratch.qpos[:] = qpos
        scratch.qvel[:] = self.data.qvel
        mujoco.mj_kinematics(self.model, scratch)
        mujoco.mj_comPos(self.model, scratch)
        body_id = int(self.model.body("base").id)
        return np.asarray(scratch.subtree_com[body_id], dtype=float).copy()

    # ---------------------------------------------------------------- sensing
    def attitude(self) -> np.ndarray:
        return euler_zyx(self.data.qpos[3:7])

    def _lengths(self, leg, angles: dict[str, float]) -> tuple[float, float]:
        """``LegKinematics.lengths`` 的按拍缓存（同一拍角度未变即命中）。

        PERF：``_chain`` 占 7%，每步 3 次调用；同一拍内 ``leg_state`` 与 ``thrust_torque``
        读的是同一组 qpos，所以键取 ``tuple(angles)`` 即足以判等。
        """
        key = (leg.name, tuple(angles.items()))
        cached = self._cache_lengths.get(key)
        if cached is not None:
            self.cache_stats["lengths_hit"] += 1
            return cached
        value = self._leg_kinematics[leg.name].lengths(angles)
        self._cache_lengths[key] = value
        self.cache_stats["lengths_recompute"] += 1
        return value

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
            values.extend(self._lengths(leg, angles))
        return np.asarray(values)

    def leg_velocity(self, rows: RowSet | None = None) -> np.ndarray:
        """返回 ``[Ldot_R, phidot_R, Ldot_L, phidot_L]``，单位为 m/s、rad/s。

        速度定义为当前腿部 Jacobian 乘以完整广义速度：

        ``[Ldot, phidot] = J_leg(q) @ qvel``。

        ``rows`` 可传入本周期已经计算好的 ``RowSet``，保证速度与同一次
        反馈采样使用完全相同的 Jacobian；不传时才重新计算运动学行。
        """
        current_rows = self.rowset() if rows is None else rows
        return np.asarray(current_rows.leg[0] @ self.data.qvel, dtype=float)

    def leg_kinematic_observation(self, rows: RowSet | None = None) -> LegKinematicObservation:
        """一次性返回独立腿部运动学位置和速度，避免调用方混用计算口径。"""
        return LegKinematicObservation(self.leg_state(), self.leg_velocity(rows))

    def thrust_torque(self, force_per_leg) -> tuple[np.ndarray, float]:
        """竖直等效力（施加在机体上，逐腿 [N]）-> 执行器力矩 + 峰值关节力矩。

        约定（与 ``LegKinematics`` 的极坐标 ``(L, phi)`` 功率对偶）：竖直力 ``F_z`` 在腿极坐标系下
        的分量为 ``F_L = F_z cos(phi)``、``F_phi = -F_z L sin(phi)``，于是 ``tau = J^T @ [F_L, F_phi]``，
        ``J = d(L, phi)/d(q_act)``（已消被动关节速率）。J **逐拍用实测位形算**：该系数随腿长变化
        21%~34%（方案第 2.1 节），不能取常数。

        执行器顺序与腿的驱动关节顺序一致：``leg_actuators`` 为
        (Right_front, Right_rear, Left_front, Left_rear)，对应 (jAB, jAG) 与 (jIJ, jIO)。
        """
        forces = np.asarray(force_per_leg, dtype=float).reshape(-1)
        if forces.size != len(R.LEGS):
            raise ValueError("force_per_leg 需要逐腿给一个竖向力")
        self.leg_state()          # 惰性建立解析运动学（几何常数只从模型读一次）
        torque = np.zeros(self.model.nu)
        peak = 0.0
        for side, leg in enumerate(R.LEGS):
            kin = self._leg_kinematics[leg.name]
            angles = {name: float(self.data.qpos[self.model.joint(name).qposadr[0]])
                      for name in (leg.front_joint, leg.rear_joint) + leg.passive_joints}
            length, phi = self._lengths(leg, angles)
            wrench = np.array([float(forces[side]) * np.cos(phi),
                               -float(forces[side]) * float(length) * np.sin(phi)])
            tau = kin.jacobian(angles).T @ wrench
            peak = max(peak, float(np.max(np.abs(tau))))
            for offset, actuator in enumerate(self.leg_actuators[2 * side:2 * side + 2]):
                torque[actuator] = float(tau[offset])
        return torque, peak

    def contact_geometry(self):
        """轮-地接触的**几何排查**（不涉及力）：返回 ``(逐轮接触, 逐轮接触点数, 非轮-地接触数)``。

        只看 MuJoCo 的接触对列表（几何信息），力一律由 ``recover_contact_force`` 反解——
        原来这里的 ``mj_contactForce`` 已删除（用户 2026-10-07：不要直接调用 MuJoCo 的力 API）。
        """
        data = self.data
        counts = np.zeros(2, dtype=int)
        other = 0
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
            counts[self.wheel_geoms.index(geom)] += 1
        return tuple(bool(n > 0) for n in counts), counts, other

    def contact_untrusted(self) -> bool:
        """本拍接触力反解是否**不可信**。

        有非接触类约束（当前只有关节限位）顶住时，其约束力不在 ``dynamics_bias`` 里，
        反解会把那份力算进接触力里，所以这一拍的法向力不能用于离地判定。
        """
        return bool(self.joint_limit_active())

    def recover_contact_force(self, blocks, qacc, torque, external=None) -> np.ndarray:
        """接触力的**唯一推导入口**：``f = R⁻¹ Q_cᵀ (M q̈ + h − Sᵀ τ − τ_ext)``。

        两个消费者共用本函数，只是输入不同：
        - WBC 级（预测）：``qacc = result.qacc``（控制器解出来的指令加速度）、``torque = 解``；
        - 采样级（实测，离地判定用）：``qacc = data.qacc``、``torque = data.ctrl``（实际施加）。
        ``external`` 传 ``data.qfrc_applied``：鼠标拖拽/推扰等外部施加力必须从残差里减掉，
        否则会被算进接触力（法向力虚高、离地判不出来）。
        """
        jacobian = np.vstack([b.jacobian for b in blocks])
        bias = self.dynamics_bias()
        if external is not None:
            bias = bias - np.asarray(external, dtype=float)
        elimination = ContactElimination.from_matrices(jacobian, self.selection, self.mass())
        return elimination.recover_contact_force(qacc, bias, torque)

    def joint_limit_active(self) -> list[str]:
        """当前被 MuJoCo 关节限位约束顶住的关节名。"""
        data = self.data
        # PERF：纯 numpy 循环，每步被 2.25 次调用。限位激活只由 qpos 决定（mj_forward 重建
        # efc 表是同值再推导），所以键用状态快照即可；缓存本身按物理步清空，不会跨步复用。
        count = int(data.nefc)
        key = self._signature()
        if self._cache_joint_limit is not None and self._cache_joint_limit[0] == key:
            self.cache_stats["joint_limit_hit"] += 1
            return self._cache_joint_limit[1]
        names = []
        for i in range(count):
            if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
                names.append(self.model.joint(int(data.efc_id[i])).name)
        self._cache_joint_limit = (key, names)
        self.cache_stats["joint_limit_recompute"] += 1
        return names
