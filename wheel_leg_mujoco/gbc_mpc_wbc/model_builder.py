"""GBC486 的 6 状态 held-leg 降阶线性化（M3），对应 circleg ``HeldHipModelBuilder``。

状态 ``x = [s, v, θ, θ̇, ψ, ψ̇]``，输入 ``u = [f_R, f_L]``（两轮纵向地面力 [N]）。

约束集合（理想"握住"）：轮接触 5 行 + 闭链 8 行 + 腿空间 4 行（L_R, φ_R, L_L, φ_L 的速度为 0）。
nv = 20，约束 17 行，剩 3 个自由速度坐标，正好由 (v, θ̇, ψ̇) 唯一确定。

方法与 circleg 相同：在 MuJoCo 当前位形上用 KKT 求约束加速度，再中心差分得到 A、B；
不拟合任何增益。轮力 → 力矩换算：右轮 ``ctrl = R·f_R``，左轮 ``ctrl = R·f_L``
（左轮执行器 gainprm = −1 已经抵消了左轮关节轴反向，见 M1 选择矩阵核验）；
本模块用 MuJoCo 实测 ∂v̇/∂f 校验符号，错号会直接报错。
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import brentq, fsolve, least_squares

from . import _circleg  # noqa: F401
from .logging_setup import get_logger  # noqa: E402
from . import robot as R
from .dynamics import Gbc486Dynamics, leg_combination  # noqa: E402
from Algorithm.mpc import ReducedModel  # noqa: E402

_LOGGER = get_logger(__name__)
_EPSILON = 1.0e-5

GRAVITY = 9.81
# A1：稳态加速度能力表的取值范围与步长（照 circleg 口径取 0.1 m/s^2 步长）。
# 上限取 4.0：A1 用 **drive_hip_angle_limit** 反查能力（`interp(drive_limit, q_c, a)`），
# 所以表必须覆盖到驱动预算对应的能力，否则 interp 被截断成表尾值。
# 本机 drive 预算 0.50 rad 对应约 3.8 m/s^2（≈ g·L_eff·sin(0.50)/z），取 4.0 留余量。
ACCEL_CAPACITY_STEP = 0.1
ACCEL_CAPACITY_MAX = 4.0
_CAPACITY_FILE = "capacity_gbc486.json"


def capacity_path() -> Path:
    """离线能力表（``tools/make_capacity.py`` 的产物）的路径。"""
    return Path(__file__).with_name(_CAPACITY_FILE)


def load_capacity_tables(leg_length: float | None = None) -> dict:
    """读取离线生成的能力表；不存在或条件不符时返回空 dict。

    表随标称腿长变化，用错表的后果比没表更糟，所以腿长不符时**不加载**并记 WARNING。
    """
    path = capacity_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _LOGGER.warning("能力表 %s 读取失败（%s），按无能力表处理", path, error)
        return {}
    recorded = payload.get("leg_length")
    if leg_length is not None and recorded is not None and abs(float(recorded) - float(leg_length)) > 1e-9:
        _LOGGER.warning("能力表腿长 %.3f 与当前标称腿长 %.3f 不符，不加载", float(recorded), leg_length)
        return {}
    return {key: value for key, value in payload.items() if key.startswith("accel_capacity")}


def wheel_radius(model) -> float:
    radii = {float(model.geom_size[model.geom(leg.wheel_geom).id, 0]) for leg in R.LEGS}
    if len(radii) != 1:
        raise ValueError(f"左右轮半径不一致：{radii}")
    return radii.pop()


def half_track(model, data) -> float:
    right = data.geom_xpos[model.geom(R.RIGHT.wheel_geom).id]
    left = data.geom_xpos[model.geom(R.LEFT.wheel_geom).id]
    return float(abs(left[1] - right[1]) / 2.0)


def yaw_inertia(model, data, radius: float, track: float) -> float:
    """整机绕质心竖直轴的复合惯量 + 原地转向时两轮自旋的反射惯量（与 circleg 同一方法）。"""
    ids = range(1, model.nbody)
    mass = np.array([model.body_mass[i] for i in ids])
    pos = np.array([data.xipos[i] for i in ids])
    center = (mass[:, None] * pos).sum(0) / mass.sum()
    inertia = 0.0
    for i, m_i, p in zip(ids, mass, pos):
        rot = data.ximat[i].reshape(3, 3)
        inertia += float((rot @ np.diag(model.body_inertia[i]) @ rot.T)[2, 2])
        inertia += float(m_i) * float((p - center)[0] ** 2 + (p - center)[1] ** 2)
    for leg in R.LEGS:
        body = model.body(leg.wheel_body).id
        rot = data.ximat[body].reshape(3, 3)
        spin_axis = data.xmat[body].reshape(3, 3)[:, 2]          # 轮关节轴 = body z
        spin = float(spin_axis @ (rot @ np.diag(model.body_inertia[body]) @ rot.T) @ spin_axis)
        inertia += spin * (track / radius) ** 2
    return float(inertia)


class HeldLegModelBuilder:
    STATE_NAMES = ("s", "v", "theta", "theta_dot", "psi", "psi_dot")
    INPUT_NAMES = ("f_right", "f_left")

    def __init__(self, dynamics: Gbc486Dynamics):
        self.dyn = dynamics
        self.model = dynamics.model
        self.data = dynamics.data
        self.pose = self.data.qpos.copy()
        self.radius = wheel_radius(self.model)
        self._torque_map = np.zeros((self.model.nu, 2))
        self._torque_map[dynamics.wheel_actuators[0], 0] = self.radius
        self._torque_map[dynamics.wheel_actuators[1], 1] = self.radius

    def torque_from_force(self, force) -> np.ndarray:
        return self._torque_map @ np.asarray(force, dtype=float)

    def _constraints(self, rows):
        contact, contact_bias = rows.constraints((True, True))
        leg, leg_bias = rows.leg
        return np.vstack((contact, leg)), np.r_[contact_bias, leg_bias]

    def derivative(self, state, force) -> np.ndarray:
        """连续时间导数。位形 = 名义位形 + 前进 s + 绕机身 y 转 θ + 绕 z 转 ψ（与 circleg 一致）。"""
        m, d = self.model, self.data
        state = np.asarray(state, dtype=float)
        d.qpos[:] = self.pose
        d.qpos[0] += state[0]
        mujoco.mju_quatIntegrate(d.qpos[3:7], np.array([0.0, state[2], state[4]]), 1.0)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        rows = self.dyn.rowset()
        constraints, _ = self._constraints(rows)
        task_rows, _ = rows.task
        mapping = np.vstack((constraints, task_rows[[0, 2, 3]]))
        d.qvel[:] = np.linalg.solve(mapping, np.r_[np.zeros(len(constraints)), state[[1, 3, 5]]])
        mujoco.mj_forward(m, d)
        rows = self.dyn.rowset()
        constraints, bias = self._constraints(rows)
        task_rows, task_bias = rows.task
        mass = self.dyn.mass()
        torque = self.torque_from_force(force)
        kkt = np.block([[mass, -constraints.T], [constraints, np.zeros((len(constraints),) * 2)]])
        rhs = np.r_[self.dyn.selection.T @ torque - self.dyn.dynamics_bias(), -bias]
        acceleration = np.linalg.solve(kkt, rhs)[:m.nv]
        out = np.empty(6)
        out[[0, 2, 4]] = state[[1, 3, 5]]
        out[[1, 3, 5]] = task_rows[[0, 2, 3]] @ acceleration + task_bias[[0, 2, 3]]
        return out

    def build(self) -> ReducedModel:
        m, d = self.model, self.data
        saved = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()
        try:
            mapping_rank = self._mapping_rank()

            def pitch_acceleration(pitch):
                state = np.zeros(6)
                state[2] = pitch
                return self.derivative(state, np.zeros(2))[3]

            equilibrium = brentq(pitch_acceleration, -0.02, 0.02, xtol=1e-14)
            mujoco.mju_quatIntegrate(self.pose[3:7], np.array([0.0, equilibrium, 0.0]), 1.0)
            A = np.column_stack([(self.derivative(np.eye(6)[i] * _EPSILON, np.zeros(2))
                                  - self.derivative(-np.eye(6)[i] * _EPSILON, np.zeros(2))) / (2 * _EPSILON)
                                 for i in range(6)])
            B = np.column_stack([(self.derivative(np.zeros(6), np.eye(2)[i] * _EPSILON)
                                  - self.derivative(np.zeros(6), -np.eye(2)[i] * _EPSILON)) / (2 * _EPSILON)
                                 for i in range(2)])
            affine = self.derivative(np.zeros(6), np.zeros(2))
            if not (B[1, 0] > 0.0 and B[1, 1] > 0.0):
                raise ValueError(f"轮力符号错误：∂v̇/∂f = {B[1]}（应为正）")
            if not (B[5, 1] < 0.0 < B[5, 0]):
                # 右轮在 −y 侧：向前推的力绕 +z 的力矩 τz = −y·Fx > 0，左轮相反
                raise ValueError(f"偏航符号与几何不符：∂ψ̈/∂f = {B[5]}（应为 [+, −]）")
            mujoco.mj_forward(m, d)
            track = half_track(m, d)
            inertia = yaw_inertia(m, d, self.radius, track)
            com = R.robot_com(m, d)
            contact_z = float(np.mean(R.wheel_contact_points(m, d)[:, 2]))
            metadata = dict(
                equilibrium_pitch=float(equilibrium),
                nominal_pitch_acceleration=float(affine[3]),
                nominal_forward_acceleration=float(affine[1]),
                mapping_rank=mapping_rank,
                com_height=float(com[2] - contact_z),
                total_mass=float(sum(m.body_mass)),
                leg_state=self.dyn.leg_state().tolist(),
            )
            return ReducedModel(A, B, self.radius, track, inertia, source="gbc486-held-leg", metadata=metadata)
        finally:
            d.qpos[:], d.qvel[:], d.ctrl[:] = saved
            mujoco.mj_forward(m, d)

    def _mapping_rank(self) -> int:
        """速度映射 [约束; v, θ̇, ψ̇ 行] 必须满秩 nv，否则 3 个坐标不能唯一确定速度。"""
        d = self.data
        d.qpos[:] = self.pose
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)
        rows = self.dyn.rowset()
        constraints, _ = self._constraints(rows)
        mapping = np.vstack((constraints, rows.task[0][[0, 2, 3]]))
        s = np.linalg.svd(mapping, compute_uv=False)
        rank = int(np.sum(s > 1e-9 * s[0]))
        if mapping.shape[0] != self.model.nv or rank != self.model.nv:
            raise ValueError(f"速度映射 {mapping.shape} 秩 {rank}，不能唯一确定 qvel")
        return rank


def torque_from_force(model, dynamics: Gbc486Dynamics, force) -> np.ndarray:
    """``[f_R, f_L]`` → 6 维 ctrl（只填两个轮通道）。"""
    out = np.zeros(model.nu)
    radius = wheel_radius(model)
    out[dynamics.wheel_actuators[0]] = radius * float(force[0])
    out[dynamics.wheel_actuators[1]] = radius * float(force[1])
    return out


class LegInputModelBuilder:
    """8 状态 leg-input 降阶模型（M5），对应 circleg ``HipInputModelBuilder``。

    状态 ``x = [s, v, θ, θ̇, ψ, ψ̇, δφ_c, δφ̇_c]``，输入 ``u = [f_R, f_L, τ_c]``。
    ``δφ_c = φ_c − φ0`` 是共模腿摆角相对标称值的偏差（circleg 的 ``q_c`` 标称正好为 0，
    本机 φ0 = −0.0502，所以这里用偏差，使参考量仍然为 0）。

    约束：轮接触 5 + 闭链 8 = 13 行硬约束，再“握住” ``L_R``、``L_L``、``φ_d`` 三行（共 16 行），
    剩 4 个自由速度坐标 ``(s, θ, ψ, φ_c)``；nv = 20，16 + 4 = 20，满秩。

    ``τ_c`` 的施加方式：先算出共模腿角的广义力方向 ``g = ∂φ_c/∂q``，再用 4 个腿电机的最小范数解
    ``τ_leg`` 去匹配它（在约束零空间内匹配，误差只落在闭链内力方向）。这样输入仍然是真实电机力矩，
    ``τ_c`` 的上下界可以直接取腿电机限值。
    """

    STATE_NAMES = ("s", "v", "theta", "theta_dot", "psi", "psi_dot", "phi_c", "phi_c_dot")
    INPUT_NAMES = ("f_right", "f_left", "tau_c")

    def __init__(self, dynamics: Gbc486Dynamics, leg_length: float):
        self.dyn = dynamics
        self.model = dynamics.model
        self.data = dynamics.data
        self.pose = self.data.qpos.copy()
        self.leg_length = float(leg_length)
        self.radius = wheel_radius(self.model)
        self._angle0 = dynamics.leg_state()[1]          # 标称共模腿角 φ0
        self._guess = None                              # IK 热启动
        self._torque_map = np.zeros((self.model.nu, 2))
        self._torque_map[dynamics.wheel_actuators[0], 0] = self.radius
        self._torque_map[dynamics.wheel_actuators[1], 1] = self.radius

    @property
    def nominal_leg_angle(self) -> float:
        return float(self._angle0)

    # ------------------------------------------------------------- 位形与约束
    def set_legs(self, phi_c: float) -> None:
        """用逆运动学把两条腿设成 ``L = L0, φ_polar = φ0 − phi_c``，差模为 0。

        **符号约定（2026-10-05，用户选定方案“甲”：改定义，不在消费端打补丁）**：
        状态里的 ``δφ_c`` 定义为 ``δφ_c := −(φ_polar − φ0)``。

        理由：Algorithm 层（ZMP 曲柄 ``sign(a)·asin(...)``、摆腿前馈 ``sign(a)·interp(...)``、
        能力表 ``interp(hip_limit, q_c, a)`` 的递增方向、``_apply_zmp_reference`` 的裁剪区间）
        全部建立在 **``∂v̇/∂φ_c > 0``** 之上。而本机极角增大是把质心推向接触点**后方**
        （实测 ±0.15 rad → ∓31.9 mm，``∂v̇/∂φ_polar = −6.12``），符号与 circleg 相反。
        取负后本机约定与 Algorithm 层一致，四处消费点无需任何补偿因子。

        物理含义：**正 δφ_c = 质心前移 = 前进加速；负 δφ_c = 质心后移 = 减速**。
        机械可达范围不变（幅度相同，只换方向）：δφ_c ∈ [−0.674, +0.784] rad。
        """
        self.data.qpos[:] = self.pose
        angle = self.nominal_leg_angle - float(phi_c)
        if self._guess is not None:
            self.data.qpos[R.qpos_addresses(self.model, R.RIGHT.solve_joints)] = self._guess[0]
            self.data.qpos[R.qpos_addresses(self.model, R.LEFT.solve_joints)] = self._guess[1]
        else:
            mujoco.mj_kinematics(self.model, self.data)
        for leg in R.LEGS:
            R.solve_leg(self.model, self.data, leg, self.leg_length, angle)
        self._guess = tuple(self.data.qpos[R.qpos_addresses(self.model, leg.solve_joints)].copy()
                            for leg in R.LEGS)

    def _constraints(self, rows):
        """硬约束 13 行 + 握住 L_R / L_L / φ_d 三行，共 16 行。"""
        contact, contact_bias = rows.constraints((True, True))
        leg, leg_bias = rows.leg
        differential_rows, differential_bias = leg_combination(rows, -1.0)
        held_rows = np.vstack((leg[[0]], leg[[2]], differential_rows[[1]]))
        held_bias = np.r_[leg_bias[[0]], leg_bias[[2]], differential_bias[1]]
        return (np.vstack((contact, held_rows)), np.r_[contact_bias, held_bias])

    def _free_rows(self, rows):
        """自由坐标行：前进速度、俯仰率、偏航率、共模腿角率。

        共模腿角行必须与 ``set_legs`` 的符号约定一致（δφ_c := −(极角 − φ0)）：
        位形用 ``nominal − phi_c``，则速度映射与加速度读数也必须走同一个坐标系，
        即取 **极角行的负号**。只翻位形不翻这一行，会让同一个状态 6
        在“位形”和“速度”上符号相反，线性化直接错掉
        （实测：极点 11.614 → 2.309，∂φ̈_c/∂φ_c +105.9 → −105.9）。
        """
        task_rows, task_bias = rows.task
        common_rows, common_bias = leg_combination(rows, +1.0)
        return (np.vstack((task_rows[[0, 2, 3]], -common_rows[[1]])),
                np.r_[task_bias[[0, 2, 3]], -common_bias[1]])

    def _leg_torque(self, rows, tau_c: float) -> np.ndarray:
        """把共模腿角方向的广义力折算成 4 个腿电机力矩（约束零空间内的最小范数解）。"""
        constraint, _ = self._constraints(rows)
        _, singular, vt = np.linalg.svd(constraint)
        null = vt[int(np.sum(singular > 1e-9 * singular[0])):].T        # nv × 4
        direction = leg_combination(rows, +1.0)[0][1]                   # ∂φ_c/∂q，nv
        selection = self.dyn.selection[list(self.dyn.leg_actuators)]    # 4 × nv
        matrix = (null.T @ selection.T)                                 # 4 × 4
        target = null.T @ direction
        solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = float(np.linalg.norm(matrix @ solution - target))
        if residual > 1e-6 * max(np.linalg.norm(target), 1e-9):
            raise ValueError(f"共模腿角方向的广义力不能用腿电机表示，残差 {residual:.3e}")
        torque = np.zeros(self.model.nu)
        torque[list(self.dyn.leg_actuators)] = solution * float(tau_c)
        return torque

    # ------------------------------------------------------------------ 导数
    def derivative(self, state, force) -> np.ndarray:
        m, d = self.model, self.data
        state = np.asarray(state, dtype=float)
        self.set_legs(state[6])
        d.qpos[0] += state[0]
        mujoco.mju_quatIntegrate(d.qpos[3:7], np.array([0.0, state[2], state[4]]), 1.0)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        rows = self.dyn.rowset()
        constraint, _ = self._constraints(rows)
        free_rows, _ = self._free_rows(rows)
        mapping = np.vstack((constraint, free_rows))
        d.qvel[:] = np.linalg.solve(mapping, np.r_[np.zeros(len(constraint)), state[[1, 3, 5, 7]]])
        mujoco.mj_forward(m, d)
        rows = self.dyn.rowset()
        constraint, bias = self._constraints(rows)
        free_rows, free_bias = self._free_rows(rows)
        torque = self.torque_from_force(force[:2]) + self._leg_torque(rows, force[2])
        kkt = np.block([[self.dyn.mass(), -constraint.T],
                        [constraint, np.zeros((len(constraint),) * 2)]])
        acceleration = np.linalg.solve(kkt, np.r_[self.dyn.selection.T @ torque - self.dyn.dynamics_bias(),
                                                  -bias])[:m.nv]
        out = np.empty(8)
        out[[0, 2, 4, 6]] = state[[1, 3, 5, 7]]
        out[[1, 3, 5, 7]] = free_rows @ acceleration + free_bias
        return out

    def torque_from_force(self, force) -> np.ndarray:
        return self._torque_map @ np.asarray(force, dtype=float)

    # -------------------------------------------------------------------- 构建
    def trim(self, phi_c: float):
        """工作点 ``(θ_trim, τ_c_trim)``：零轮力下 θ̈ = 0 且 φ̈_c = 0。"""
        def residual(values):
            theta, tau_c = values
            state = np.zeros(8)
            state[2], state[6] = theta, phi_c
            result = self.derivative(state, np.array([0.0, 0.0, tau_c]))
            return np.array([result[3], result[7]])

        solution, _, converged, message = fsolve(residual, np.array([0.0, 0.0]), full_output=True)
        if not converged:
            raise RuntimeError(f"leg_input 工作点求解未收敛：{message}")
        return float(solution[0]), float(solution[1])

    def build(self, phi_c: float = 0.0) -> ReducedModel:
        m, d = self.model, self.data
        saved = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()
        try:
            self.set_legs(phi_c)
            constraint, _ = self._constraints(self.dyn.rowset())
            free_rows, _ = self._free_rows(self.dyn.rowset())
            mapping = np.vstack((constraint, free_rows))
            rank = int(np.linalg.matrix_rank(mapping, tol=1e-9))
            if mapping.shape[0] != m.nv or rank != m.nv:
                raise ValueError(f"速度映射 {mapping.shape} 秩 {rank}，不能唯一确定 qvel")
            theta_trim, tau_trim = self.trim(phi_c)
            trim_state = np.zeros(8)
            trim_state[2], trim_state[6] = theta_trim, phi_c
            trim_force = np.array([0.0, 0.0, tau_trim])
            epsilon = _EPSILON
            A = np.column_stack([(self.derivative(trim_state + epsilon * np.eye(8)[i], trim_force)
                                  - self.derivative(trim_state - epsilon * np.eye(8)[i], trim_force)) / (2 * epsilon)
                                 for i in range(8)])
            B = np.column_stack([(self.derivative(trim_state, trim_force + epsilon * np.eye(3)[i])
                                  - self.derivative(trim_state, trim_force - epsilon * np.eye(3)[i])) / (2 * epsilon)
                                 for i in range(3)])
            affine = self.derivative(trim_state, trim_force)
            # 与 6 状态模型不同：φ_c 自由时轮力对 v̇ 的直接权限很小且符号为负（腿会自由摆动弱化了耦合），
            # 所以这里不断言 ∂v̇/∂f > 0，只断言轮力矩对 φ̈_c 有权限、以及偏航几何符号。
            if not (B[5, 1] < 0.0 < B[5, 0]):
                raise ValueError(f"偏航符号与几何不符：∂ψ̈/∂f = {B[5]}")
            if abs(B[7, 2]) < 1e-6:
                raise ValueError(f"共模腿力矩对 φ̈_c 无权限：∂φ̈_c/∂τ_c = {B[7, 2]:.3e}")
            if A[1, 6] <= 0.0:
                # Algorithm 层依赖的约定，之前这里没断言，所以符号反了也静默通过
                raise ValueError(
                    f"共模腿角符号与 Algorithm 层约定不符：∂v̇/∂φ_c = {A[1, 6]:.4f}（应为正）。"
                    "ZMP 曲柄、摆腿前馈与能力表插值方向都依赖这条约定，"
                    "详见 LegInputModelBuilder.set_legs 的符号约定说明。")
            # 配平点处非零漂移应当只是残差量级
            mujoco.mj_forward(m, d)
            track = half_track(m, d)
            inertia = yaw_inertia(m, d, self.radius, track)
            com = R.robot_com(m, d)
            contact_z = float(np.mean(R.wheel_contact_points(m, d)[:, 2]))
            # 摆腿的等效质心偏移半径 r_crank = d(CoM_x − 接触点_x)/d(δφ_c)。
            # **不能拿标称腿长 L0 代替**：质心不在腿顶端，本机实测 L0=0.28 时真实值约 0.213。
            # ZMP 公式 Δ* = z·a/g 里的 r_crank 就用这个量，所以直接从模型量。
            offset_step = 1.0e-4

            def leg_offset(phi_c: float) -> float:
                self.set_legs(phi_c)
                mujoco.mj_forward(m, d)
                center = R.robot_com(m, d)
                return float(center[0] - np.mean(R.wheel_contact_points(m, d)[:, 0]))

            leg_offset_radius = ((leg_offset(offset_step) - leg_offset(-offset_step))
                                 / (2.0 * offset_step))
            if not (leg_offset_radius > 0.0):
                # 与 A[1,6] 同源：r_crank > 0 等价于“正 δφ_c → 质心前移”，两条断言互相印证
                raise ValueError(f"摆腿等效半径非正：r_crank = {leg_offset_radius:.4f} m")
            metadata = dict(
                equilibrium_pitch=float(theta_trim), tau_trim=float(tau_trim), phi_c=float(phi_c),
                nominal_leg_angle=self.nominal_leg_angle,
                nominal_pitch_acceleration=float(affine[3]), nominal_forward_acceleration=float(affine[1]),
                nominal_leg_acceleration=float(affine[7]), mapping_rank=rank,
                com_height=float(com[2] - contact_z), total_mass=float(sum(m.body_mass)),
                leg_offset_radius=float(leg_offset_radius),
                leg_state=self.dyn.leg_state().tolist(),
            )
            metadata.update(load_capacity_tables(self.leg_length))
            return ReducedModel(A, B, self.radius, track, inertia, source="gbc486-leg-input", metadata=metadata)
        finally:
            d.qpos[:], d.qvel[:], d.ctrl[:] = saved
            self._guess = None
            mujoco.mj_forward(m, d)

    # ------------------------------------------------- 稳态能力表（A1 / A3）
    # 与 circleg ``HipInputModelBuilder.steady_state / accel_capacity_table /
    # tilted_capacity_table`` 同一算法：在非线性降阶 ODE 上求稳态族，不拟合增益。
    # 差异（已注明）：本实现额外用**执行器力矩限 + 摩擦锥**截断表，
    # circleg 原版只靠“q_c 单调性饱和”截断；"能力表"不尊重力矩/摩擦就不能当能力用。

    def steady_state(self, acceleration: float, theta_weight: float = 1.0e6) -> dict:
        """稳态工作点 ``(theta, q_c, F, tau_c)``：θ̈ = 0、φ̈_c = 0、v̇ = a（两轮等力）。

        3 个方程 4 个未知量，欠定方向用带权正则 ``√w·θ`` 压向 θ = 0，
        得到的正是“姿态倾角为零的稳态族”。
        """
        weight = math.sqrt(max(float(theta_weight), 0.0))
        target = float(acceleration)
        z, length = self._capacity_com_height, self.leg_length
        # 初值取解析关系 a = g·L·sin(q_c)/z（θ=0）与牛顿定律，比经验比例更稳
        q_guess = math.asin(min(abs(target) * z / (GRAVITY * length), 0.999))
        guess = np.array([0.0, math.copysign(q_guess, target),
                          self._capacity_mass * target / 2.0, 0.0])

        def residual(values):
            theta, q_c, force_each, tau_c = (float(v) for v in values)
            state = np.zeros(8)
            state[2], state[6] = theta, q_c
            derivative = self.derivative(state, np.array([force_each, force_each, tau_c]))
            return np.array([derivative[3], derivative[7],
                             derivative[1] - target, weight * theta])

        solution = least_squares(residual, guess, xtol=1e-13, ftol=1e-13, gtol=1e-13)
        theta, q_c, force_each, tau_c = (float(v) for v in solution.x)
        return dict(theta=theta, q_c=q_c, force=2.0 * force_each, tau_c=tau_c,
                    force_per_wheel=force_each, cost=float(solution.cost),
                    optimality=float(solution.optimality), success=bool(solution.success),
                    residual=float(np.linalg.norm(residual(solution.x))))

    def accel_capacity_table(self, step: float = ACCEL_CAPACITY_STEP,
                             maximum: float = ACCEL_CAPACITY_MAX) -> dict[str, list[float]]:
        """``(a, q_c)`` 稳态能力表。遇到第一个不可行点就停在上一有效点并记 INFO。"""
        accelerations = np.arange(0.0, float(maximum) + 0.5 * float(step), float(step))
        a_values: list[float] = []
        q_values: list[float] = []
        for acceleration in accelerations:
            solution = self.steady_state(float(acceleration))
            if not solution["success"] or solution["residual"] > 1e-6:
                _LOGGER.info("能力表在 a=%.2f 处求解失败（残差 %.2e），截断到 a=%.2f",
                             acceleration, solution["residual"], a_values[-1] if a_values else 0.0)
                break
            reason = self._capacity_infeasible(solution)
            if reason is not None:
                _LOGGER.info("能力表在 a=%.2f 处不可行（%s），截断到 a=%.2f",
                             acceleration, reason, a_values[-1] if a_values else 0.0)
                break
            q_c = float(solution["q_c"])
            if q_values and q_c <= q_values[-1]:      # 插值要求 q_c 严格递增
                _LOGGER.info("能力表在 a=%.2f 处饱和（q_c=%.4f 不再增加），截断到 a=%.2f",
                             acceleration, q_c, a_values[-1])
                break
            a_values.append(float(acceleration))
            q_values.append(q_c)
        return {"a": a_values, "q_c": q_values}

    def tilted_capacity_table(self, theta: float, leg_angles=None) -> dict:
        """机身固定前倾 ``theta`` 时的持续加速度能力表 ``(q_c -> a)``。

        固定 ``(theta, q_c)`` 解两轮力 ``f`` 与共模腿力矩 ``tau_c`` 使 θ̈ = 0、φ̈_c = 0，
        再读出 ``v̇``。腿角网格按本机构共模可达范围取。
        """
        if leg_angles is None:
            leg_angles = tuple(round(0.05 * i, 2) for i in range(0, 14))    # 0 .. 0.65 rad
        q_values: list[float] = []
        a_values: list[float] = []
        for q_c in leg_angles:
            def residual(values, q_c=float(q_c)):
                force_each, tau_c = (float(v) for v in values)
                state = np.zeros(8)
                state[2], state[6] = float(theta), q_c
                d = self.derivative(state, np.array([force_each, force_each, tau_c]))
                return [d[3], d[7]]

            force_guess = self._capacity_mass * GRAVITY * math.sin(max(abs(float(theta)), 1e-3)) / 2.0
            solution, _, converged, message = fsolve(residual, [force_guess, 0.0], full_output=True)
            if converged != 1:
                _LOGGER.info("倾角能力表在 q_c=%.2f 处不收敛（%s），截断", q_c, message)
                break
            state = np.zeros(8)
            state[2], state[6] = float(theta), float(q_c)
            force_each, tau_c = (float(v) for v in solution)
            acceleration = float(self.derivative(state, np.array([force_each, force_each, tau_c]))[1])
            q_values.append(float(q_c))
            a_values.append(acceleration)
        return {"theta": float(theta), "q_c": q_values, "a": a_values}

    def capacity_tables(self, pitch_max: float, brake_max: float | None = None,
                        step: float = ACCEL_CAPACITY_STEP,
                        maximum: float = ACCEL_CAPACITY_MAX) -> dict:
        """生成三张表，与 circleg ``APP/mujoco_balance_model.py`` 的键名一致。"""
        tables = {"accel_capacity": self.accel_capacity_table(step=step, maximum=maximum),
                  "accel_capacity_tilted": self.tilted_capacity_table(float(pitch_max))}
        if brake_max is not None:
            tables["accel_capacity_tilted_brake"] = self.tilted_capacity_table(float(brake_max))
        return tables

    @property
    def _capacity_mass(self) -> float:
        return float(sum(self.model.body_mass))

    @property
    def _capacity_friction(self) -> float:
        """轮地滑动摩擦系数（取 MJCF 里轮 geom 的定义，不另写常数）。"""
        geom = int(self.model.geom(R.LEGS[0].wheel_geom).id)
        return float(self.model.geom_friction[geom, 0])

    @property
    def _capacity_com_height(self) -> float:
        """接触点以上的质心高 [m]（ZMP 公式的 z_com）。

        性能：首次调用时在标称位形上现算并缓存。**不能直接缓存 data 状态**：
        ``derivative`` 会改写 ``data`` 的位形（set_legs 做逆运动学），
        在受污染的状态上算出的质心高会与 metadata 不一致。
        """
        if getattr(self, "_com_height_cache", None) is None:
            saved_qpos, saved_qvel = self.data.qpos.copy(), self.data.qvel.copy()
            self.data.qpos[:] = self.pose
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            com = R.robot_com(self.model, self.data)
            contact = float(np.mean(R.wheel_contact_points(self.model, self.data)[:, 2]))
            self._com_height_cache = float(com[2] - contact)
            self.data.qpos[:], self.data.qvel[:] = saved_qpos, saved_qvel
            mujoco.mj_forward(self.model, self.data)
        return float(self._com_height_cache)

    def _capacity_infeasible(self, solution: dict) -> str | None:
        """稳态点的力矩限 / 摩擦锥检查（可行返回 None，否则返回原因）。"""
        leg_limit = float(self.model.actuator_ctrlrange[self.dyn.leg_actuators[0], 1])
        if abs(float(solution["tau_c"])) > leg_limit:
            return f"腿力矩 {abs(solution['tau_c']):.1f} > {leg_limit:.1f} N*m"
        friction = self._capacity_friction * self._capacity_mass * GRAVITY / 2.0
        force = abs(float(solution["force_per_wheel"]))
        if force > friction:
            return f"单轮切向力 {force:.1f} > 摩擦限 {friction:.1f} N"
        wheel_limit = float(self.model.actuator_ctrlrange[self.dyn.wheel_actuators[0], 1])
        wheel_torque = force * self.radius
        if wheel_torque > wheel_limit:
            return f"轮力矩 {wheel_torque:.2f} > {wheel_limit:.2f} N*m"
        return None


def pendulum_pole(height: float, gravity: float = 9.81) -> float:
    return math.sqrt(gravity / height)
