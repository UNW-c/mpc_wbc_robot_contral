"""离线跳跃测试平台（J-M，第 17 次排查）：平面混合动力学模型，不依赖 MuJoCo。

广义坐标 ``g = [x, z, θ, q, φ]``：
- ``x, z``：机身质心的水平 / 竖直位置 [m]；
- ``θ``：机身俯仰 [rad]（正 = 前倾，与 MPC 状态同号约定）；
- ``q``：共模曲柄角 [rad]（0 = 下死点，机身最低；π = 上死点），两腿视为一条合并腿；
- ``φ``：轮子转角 [rad]（两轮合并）。

几何（数值取自 URDF_Model_plane.xml，手工抄录）：
- 髋轴在机身系中的位置 ``hip_offset``（相对机身质心，x 前 z 上）；
- 曲柄从髋轴指向轮心，长度 ``r = 0.06``，在机身系中与"竖直向上"夹角为 q（与 MuJoCo 核对）：
  q = 0 时轮心在髋轴正上方（机身最低），q 增大时轮心向后并下降，q = π 时在正下方（机身最高）；
- 曲柄质心位于距髋轴 ``crank_com`` 处。

动力学：用拉格朗日方程 ``M(g) g̈ + h(g, ġ) = Q``，M 由各刚体的质心速度雅可比解析组装，
科氏 / 离心项用 ``h = Ṁ ġ − ½ ∂(ġᵀ M ġ)/∂g`` 的数值差分计算（步长 1e-6），避免手推出错。

接触：轮子与地面是单边接触，法向用罚函数弹簧阻尼（只压不拉），切向用库仑摩擦的连续近似
（``F_t = −μ N tanh(v_slip / v_eps)``）。积分用 RK4，默认步长 0.2 ms。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_G = 9.81


@dataclass(frozen=True)
class JumpModelParams:
    body_mass: float = 8.0119
    body_inertia: float = 0.0535          # 机身绕 y 轴（XML base_link inertia 中间项）
    crank_mass: float = 2 * 0.3785
    crank_inertia: float = 2 * 0.00315    # 绕自身质心、y 轴
    crank_com: float = 0.0481             # 质心距髋轴
    wheel_mass: float = 2 * 0.3663
    wheel_inertia: float = 2 * 0.00825    # 轮子自转（XML joint2_Link inertia 第一项）
    crank_length: float = 0.06
    wheel_radius: float = 0.16
    # 髋轴相对机身质心（机身系，x 前、z 上）：XML hip pos (-0.01994, *, -0.093022) 减去 base ipos (-0.0199, *, -0.0133)
    hip_offset: tuple = (-0.00004, -0.07972)
    contact_stiffness: float = 5.0e4
    contact_damping: float = 500.0
    friction: float = 1.0
    slip_eps: float = 0.01
    hip_torque_limit: float = 10.0        # 每髋 [N·m]；合并腿的力矩上限 = 2 倍
    wheel_torque_limit: float = 5.0       # 每轮 [N·m]


@dataclass
class JumpModelState:
    g: np.ndarray                          # [x, z, θ, q, φ]
    gd: np.ndarray
    t: float = 0.0


class PlanarJumpModel:
    """平面合并腿模型。力矩输入 ``u = [τ_hip_total, τ_wheel_total]``（两侧之和）。"""

    def __init__(self, params: JumpModelParams | None = None):
        self.p = params or JumpModelParams()

    # ---- 运动学 ------------------------------------------------------------
    def _rot(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        # 机身系 (x 前, z 上) -> 世界系；θ 正 = 前倾（机头向下），即绕 +y 旋转
        return np.array([[c, s], [-s, c]])

    def points(self, g):
        """返回 (机身质心, 髋轴, 曲柄质心, 轮心) 的世界坐标，以及曲柄在世界系中的方向角。"""
        x, z, theta, q = g[0], g[1], g[2], g[3]
        body = np.array([x, z])
        R = self._rot(theta)
        hip = body + R @ np.array(self.p.hip_offset)
        # 曲柄在机身系中（与 MuJoCo 核对：q = 0 时轮心在髋轴正上方 0.06 m，机身最低；
        # q = π/2 时轮心在髋轴后方 0.06 m；q = π 时在正下方，机身最高）
        d_body = np.array([-np.sin(q), np.cos(q)])
        d = R @ d_body
        crank = hip + self.p.crank_com * d
        wheel = hip + self.p.crank_length * d
        return body, hip, crank, wheel

    def _jacobians(self, g, eps=1e-7):
        """各质心对广义坐标的雅可比（数值差分，2x5）。"""
        base = self.points(g)
        jac = [np.zeros((2, 5)) for _ in range(4)]
        for i in range(5):
            dg = np.zeros(5)
            dg[i] = eps
            plus = self.points(g + dg)
            minus = self.points(g - dg)
            for k in range(4):
                jac[k][:, i] = (plus[k] - minus[k]) / (2 * eps)
        return base, jac

    def mass_matrix(self, g):
        p = self.p
        _, (Jb, _, Jc, Jw) = self._jacobians(g)
        M = p.body_mass * Jb.T @ Jb + p.crank_mass * Jc.T @ Jc + p.wheel_mass * Jw.T @ Jw
        # 转动：机身角速度 = θ̇；曲柄角速度 = θ̇ + q̇（同一平面，方向约定一致）；轮子 = θ̇ + q̇ + φ̇（相对曲柄转）
        e_theta = np.array([0, 0, 1, 0, 0.0])
        # 角速度（绕 +y，数值核对：机身 x 轴与曲柄方向的世界角对 θ、q 的导数都是 −1，
        # 即 θ 与 q 的正向都对应绕 +y 转）：机身 = θ̇，曲柄 = θ̇ + q̇，轮子 = θ̇ + q̇ + φ̇。
        e_crank = np.array([0, 0, 1, 1, 0.0])
        e_wheel = np.array([0, 0, 1, 1, 1.0])
        M += p.body_inertia * np.outer(e_theta, e_theta)
        M += p.crank_inertia * np.outer(e_crank, e_crank)
        M += p.wheel_inertia * np.outer(e_wheel, e_wheel)
        return M

    def potential_gradient(self, g, eps=1e-7):
        def potential(gg):
            b, _, c, w = self.points(gg)
            p = self.p
            return _G * (p.body_mass * b[1] + p.crank_mass * c[1] + p.wheel_mass * w[1])
        grad = np.zeros(5)
        for i in range(5):
            dg = np.zeros(5)
            dg[i] = eps
            grad[i] = (potential(g + dg) - potential(g - dg)) / (2 * eps)
        return grad

    def coriolis(self, g, gd, eps=1e-6):
        """h = Ṁ ġ − ½ ∂(ġᵀ M ġ)/∂g（数值差分）。"""
        M_dot = (self.mass_matrix(g + eps * gd) - self.mass_matrix(g - eps * gd)) / (2 * eps)
        grad = np.zeros(5)
        for i in range(5):
            dg = np.zeros(5)
            dg[i] = eps
            grad[i] = (gd @ self.mass_matrix(g + dg) @ gd - gd @ self.mass_matrix(g - dg) @ gd) / (2 * eps)
        return M_dot @ gd - 0.5 * grad

    # ---- 接触 --------------------------------------------------------------
    def contact_force(self, g, gd):
        """返回 (广义接触力 5 维, 法向力 N, 轮缘滑移速度)。"""
        p = self.p
        _, (_, _, _, Jw) = self._jacobians(g)
        wheel = self.points(g)[3]
        penetration = p.wheel_radius - wheel[1]
        wheel_vel = Jw @ gd
        if penetration <= 0.0:
            return np.zeros(5), 0.0, 0.0
        normal = max(p.contact_stiffness * penetration - p.contact_damping * wheel_vel[1], 0.0)
        # 接触点（轮心正下方）的水平速度 = 轮心水平速度 + R·ω_wheel（ω 为世界系角速度，正 = 前倾方向转动）
        omega = gd[2] + gd[3] + gd[4]
        # 接触点（轮心正下方 R）速度 = v_center + ω×r = v_x − R·ω（ω 绕 +y；与 MuJoCo 核对：纯滚动前进时 ω = +v/R）
        slip = wheel_vel[0] - p.wheel_radius * omega
        tangential = -p.friction * normal * np.tanh(slip / p.slip_eps)
        # 力作用在接触点：水平力对轮心产生力矩 −R·F_t（绕轮心），竖直力过轮心
        force = np.array([tangential, normal])
        generalized = Jw.T @ force
        # 切向力对轮子转动的力矩：接触点在轮心下方 R，对应泛函力沿轮子世界角速度的雅可比
        e_wheel = np.array([0, 0, 1, 1, 1.0])
        # 切向力 F_t 作用在接触点 r = (0, −R)（相对轮心，x 前 z 上）。平面内绕 +y 的力矩
        # τ_y = r_z·F_x − r_x·F_z = −R·F_t（x 前、z 上、y 向左时 τ_y = z·F_x − x·F_z）。
        # 轮子世界角速度对广义坐标的雅可比为 e_wheel，所以广义力 = τ_y·e_wheel。
        generalized += -p.wheel_radius * tangential * e_wheel
        return generalized, normal, slip

    # ---- 积分 --------------------------------------------------------------
    def acceleration(self, g, gd, u):
        p = self.p
        tau_hip = float(np.clip(u[0], -2 * p.hip_torque_limit, 2 * p.hip_torque_limit))
        tau_wheel = float(np.clip(u[1], -2 * p.wheel_torque_limit, 2 * p.wheel_torque_limit))
        # 髋力矩：作用在 q 上（机身受反力矩，由 q 与 θ 的耦合自动体现）；轮力矩：作用在 φ 上
        Q = np.array([0.0, 0.0, 0.0, tau_hip, tau_wheel])
        contact, normal, slip = self.contact_force(g, gd)
        rhs = Q + contact - self.coriolis(g, gd) - self.potential_gradient(g)
        gdd = np.linalg.solve(self.mass_matrix(g), rhs)
        return gdd, normal, slip

    def step(self, state: JumpModelState, u, dt: float = 2e-4) -> tuple[JumpModelState, float, float]:
        def f(g, gd):
            gdd, n, s = self.acceleration(g, gd, u)
            return gd, gdd, n, s
        g, gd = state.g, state.gd
        k1g, k1v, n, s = f(g, gd)
        k2g, k2v, _, _ = f(g + 0.5 * dt * k1g, gd + 0.5 * dt * k1v)
        k3g, k3v, _, _ = f(g + 0.5 * dt * k2g, gd + 0.5 * dt * k2v)
        k4g, k4v, _, _ = f(g + dt * k3g, gd + dt * k3v)
        g_new = g + dt / 6 * (k1g + 2 * k2g + 2 * k3g + k4g)
        gd_new = gd + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        return JumpModelState(g_new, gd_new, state.t + dt), n, s

    # ---- 初始化 ------------------------------------------------------------
    def standing_state(self, q: float = 0.0, theta: float = 0.0) -> JumpModelState:
        """轮子刚好触地（静压缩量 = 总重 / 刚度）的站立状态。"""
        p = self.p
        g = np.array([0.0, 0.0, theta, q, 0.0])
        wheel = self.points(g)[3]
        total = p.body_mass + p.crank_mass + p.wheel_mass
        static = total * _G / p.contact_stiffness
        g[1] = p.wheel_radius - wheel[1] - static
        return JumpModelState(g, np.zeros(5))

    def body_height(self, state: JumpModelState) -> float:
        return float(state.g[1])

    def wheel_bottom(self, state: JumpModelState) -> float:
        return float(self.points(state.g)[3][1] - self.p.wheel_radius)
