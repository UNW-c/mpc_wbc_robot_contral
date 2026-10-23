"""两块（下块 GH+OP 子树 / 上块其余）状态与起跳段、飞行段用的离线表。

口径来源：部署文档 10.47 A（两块质量、p1/p2/J1/J2、H 分解）与 10.54 B（初始化表）。
本模块只做"读模型 + 算表"，不推进物理；所有 scratch 运算在独立的 MjData 上做，
结束前把 live data 用 mj_forward 恢复。
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.optimize import brentq

from . import robot as R


class TwoMassModel:
    """两块质量、质心与雅可比（10.47 A 口径）。"""

    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data
        right, left = R.LEGS
        self.lower_bodies = (right.passive_joints[0], left.passive_joints[0])   # "jGH" / "jOP"
        self.lower_names = tuple(str(model.body(int(model.joint(name).bodyid[0])).name)
                                 for name in self.lower_bodies)                  # GH / OP
        # 下块 = GH/OP **子树**（含各自的轮）：必须用 body_subtreemass，不是 body_mass
        self.mass_lower = float(sum(self._subtree_mass(name) for name in self.lower_names))
        self.mass_total = float(model.body_mass.sum())
        self.mass_upper = self.mass_total - self.mass_lower
        self.mu = self.mass_lower * self.mass_upper / self.mass_total

    # ---------------------------------------------------------------- 观测
    def _subtree_mass(self, name: str) -> float:
        body = self.model.body(name).id
        return float(self.model.body_subtreemass[body])

    def block_state(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (p1, p2)：两块质心的 (x, z)，单位 m。"""
        data, model = self.data, self.model
        com_lower = np.zeros(2)
        for name in self.lower_names:
            body = model.body(name).id
            com_lower += self._subtree_mass(name) * np.asarray(data.subtree_com[body], dtype=float)[[0, 2]]
        p1 = com_lower / self.mass_lower
        p2 = (self.mass_total * np.asarray(data.subtree_com[model.body("base").id], dtype=float)[[0, 2]]
              - self.mass_lower * p1) / self.mass_upper
        return p1, p2

    def block_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """两块质心速度 (v1, v2)，单位 m/s。"""
        j1, j2 = self.block_jacobians()
        qvel = np.asarray(self.data.qvel, dtype=float)
        return j1[[0, 2]] @ qvel, j2[[0, 2]] @ qvel

    def block_jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        """(J1, J2)：3 x nv 的两块质心雅可比（mj_jacSubtreeCom）。"""
        model, data = self.model, self.data
        nv = model.nv
        jac_lower = np.zeros((3, nv))
        for name in self.lower_names:
            block = np.zeros((3, nv))
            mujoco.mj_jacSubtreeCom(model, data, block, model.body(name).id)
            jac_lower += self._subtree_mass(name) * block
        j1 = jac_lower / self.mass_lower
        base = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(model, data, base, model.body("base").id)
        j2 = (self.mass_total * base - self.mass_lower * j1) / self.mass_upper
        return j1, j2

    def angular_momentum(self) -> float:
        """整机绕总质心的俯仰角动量 H1 + H2 + Horb（10.47 A）。"""
        j1, j2 = self.block_jacobians()
        v1 = j1[[0, 2]] @ self.data.qvel
        v2 = j2[[0, 2]] @ self.data.qvel
        p1, p2 = self.block_state()
        i1 = self._inertia_rows(j1)
        i2 = self._inertia_rows(j2)
        h1 = self.mass_lower * float(i1 @ self.data.qvel) + self.mass_lower * float(
            p1[1] * v1[0] - p1[0] * v1[1])
        h2 = self.mass_upper * float(i2 @ self.data.qvel) + self.mass_upper * float(
            p2[1] * v2[0] - p2[0] * v2[1])
        del h1, h2
        return float(self._momentum_row() @ self.data.qvel)

    def _inertia_rows(self, jac: np.ndarray) -> np.ndarray:
        del jac
        return np.zeros(self.model.nv)

    def _momentum_row(self) -> np.ndarray:
        """整机绕总质心的俯仰角动量行（与 dynamics.pitch_momentum_row 同一口径）。"""
        from .dynamics import Gbc486Dynamics  # 延迟导入避免循环
        del Gbc486Dynamics
        model, data = self.model, self.data
        centre = np.asarray(data.subtree_com[0], dtype=float)
        row = np.zeros(model.nv)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        for index in range(1, model.nbody):
            jacp[:] = 0.0
            jacr[:] = 0.0
            mujoco.mj_jacBodyCom(model, data, jacp, jacr, index)
            rot = data.ximat[index].reshape(3, 3)
            inertia = rot @ np.diag(model.body_inertia[index]) @ rot.T
            offset = np.asarray(data.xipos[index], dtype=float) - centre
            row += inertia[1, :] @ jacr
            row += model.body_mass[index] * (offset[2] * jacp[0] - offset[0] * jacp[2])
        return row


def _pose_quantities(model, scratch, length: float, angle: float | None) -> dict:
    """在 scratch 上摆一个 standing pose，返回两块口径的量。"""
    R.standing_pose(model, scratch, float(length), None if angle is None else float(angle))
    two = TwoMassModel(model, scratch)
    p1, p2 = two.block_state()
    com = np.asarray(scratch.subtree_com[model.body("base").id], dtype=float)
    lowest = float(np.min(R.wheel_contact_points(model, scratch)[:, 2]))
    theta = math.atan2(float(p1[0] - p2[0]), -float(p1[1] - p2[1]))
    return dict(p1=p1, p2=p2, dx=float(p2[0] - p1[0]), com_z=float(com[2]),
                D=float(com[2] - lowest), theta=theta, lowest=lowest)


def build_tables(model, data, length_nominal: float, length_min: float, length_max: float,
                 points: int = 47) -> dict:
    """按 10.54 B2/B3 建初始化表（在 scratch 上，live data 结束后 mj_forward 恢复）。"""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0
    end = _pose_quantities(model, scratch, length_nominal, None)
    theta_eq = float(end["theta"])
    grid = np.linspace(float(length_min), float(length_max), int(points))
    phi_grid = np.zeros_like(grid)
    dx_grid = np.zeros_like(grid)
    d_grid = np.zeros_like(grid)
    lowest_grid = np.zeros_like(grid)
    for index, length in enumerate(grid):
        def residual(phi: float) -> float:
            return _pose_quantities(model, scratch, float(length), float(phi))["theta"] - theta_eq
        lo, hi = -0.9, 0.9
        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo * f_hi > 0.0:
            raise RuntimeError("流形表：L=%.3f 处腿角区间内找不到 theta_eq" % length)
        phi = brentq(residual, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=100)
        row = _pose_quantities(model, scratch, float(length), float(phi))
        phi_grid[index] = float(phi)
        d_grid[index] = float(row["D"])
        stand = _pose_quantities(model, scratch, float(length), None)   # 10.68：静平衡构型的 Δx
        dx_grid[index] = float(stand["dx"])
        lowest_grid[index] = float(row["lowest"])
    # 端点雅可比（在标称 pose 上独立扰动 L 与 phi）
    step = 1.0e-5
    phi_nominal = float(np.interp(length_nominal, grid, phi_grid))
    plus_l = _pose_quantities(model, scratch, length_nominal + step, phi_nominal)
    minus_l = _pose_quantities(model, scratch, length_nominal - step, phi_nominal)
    plus_p = _pose_quantities(model, scratch, length_nominal, phi_nominal + step)
    minus_p = _pose_quantities(model, scratch, length_nominal, phi_nominal - step)
    dzc_dl = (plus_l["com_z"] - minus_l["com_z"]) / (2.0 * step)
    dzc_dp = (plus_p["com_z"] - minus_p["com_z"]) / (2.0 * step)
    ddx_dl = (plus_l["dx"] - minus_l["dx"]) / (2.0 * step)
    ddx_dp = (plus_p["dx"] - minus_p["dx"]) / (2.0 * step)
    dtheta_dphi = (plus_p["theta"] - minus_p["theta"]) / (2.0 * step)   # 10.68 切向系数
    # H 对 L' / phi' 的偏导（10.56 D1）：速度场取"贴地重放的两张 standing_pose 之差"，
    # 闭链与接触点贴地由 standing_pose 保证；不再用 _leg_rows 最小二乘
    # （那组速度只落在 4 个主动关节上，闭链残差 2.35、接触点横滑，不是可实现的两块运动）。
    mujoco.mj_forward(model, scratch)
    two = TwoMassModel(model, scratch)
    directions = {}
    for name, dL, dphi in (("Ldot", step, 0.0), ("phidot", 0.0, step)):
        plus = R.standing_pose(model, scratch, length_nominal + dL, phi_nominal + dphi)
        minus = R.standing_pose(model, scratch, length_nominal - dL, phi_nominal - dphi)
        velocity = np.zeros(model.nv)
        # dt = 2*step 使该速度场对应 L'=1（或 phi'=1）
        mujoco.mj_differentiatePos(model, velocity, 2.0 * step, minus.qpos, plus.qpos)
        scratch.qpos[:] = 0.5 * (np.asarray(plus.qpos) + np.asarray(minus.qpos))
        scratch.qpos[3:7] = plus.qpos[3:7]
        scratch.qvel[:] = velocity
        mujoco.mj_forward(model, scratch)
        directions[name] = two.angular_momentum()
        # D2 自检量：接触点竖速、闭链残差
        points = R.wheel_contact_points(model, scratch)
        del points
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)
    mujoco.mj_forward(model, data)          # 恢复 live data
    return dict(
        m1=two.mass_lower, m2=two.mass_upper, M=two.mass_total, mu=two.mu,
        dx_eq=float(end["dx"]), D_td=float(end["D"]), theta_eq=theta_eq,
        dzc_dL=float(dzc_dl), dzc_dphi=float(dzc_dp),
        ddx_dL=float(ddx_dl), ddx_dphi=float(ddx_dp),
        dH_dLdot=float(directions["Ldot"]), dH_dphidot=float(directions["phidot"]),
        length_grid=grid, D_grid=d_grid, phi_grid=phi_grid, lowest_grid=lowest_grid,
        dx_grid=dx_grid, dtheta_dphi=float(dtheta_dphi),
        length_nominal=float(length_nominal),
    )


def _leg_rows(model, data) -> np.ndarray:
    """4 x nv：d(L_R, phi_R, L_L, phi_L)/d qvel（与 rowset().leg[0] 同口径）。"""
    rows = np.zeros((4, model.nv))
    for side, leg in enumerate(R.LEGS):
        jac, _ = R.leg_jacobian(model, data, leg)
        for joint_index, name in enumerate((leg.front_joint, leg.rear_joint)):
            address = model.joint(name).dofadr[0]
            rows[2 * side:2 * side + 2, address] += jac[:, joint_index]
    return rows


def L_of_D(table: dict, D: float) -> float:
    """由 (质心 z - 轮最低点 z) 反查腿长（表单调，线性插值）。"""
    d = np.asarray(table["D_grid"], dtype=float)
    l = np.asarray(table["length_grid"], dtype=float)
    order = np.argsort(d)
    return float(np.interp(float(D), d[order], l[order]))


def D_of_L(table: dict, length: float) -> float:
    """由腿长查 (质心 z - 轮最低点 z)（流形表，线性插值）。10.68。"""
    return float(np.interp(float(length), table["length_grid"], table["D_grid"]))


def phi_theta(table: dict, length: float) -> float:
    """流形表插值：给定腿长，返回"两块连线世界角 = theta_eq"对应的腿角。"""
    return float(np.interp(float(length), table["length_grid"], table["phi_grid"]))


class TakeoffPlan:
    """THRUST 段两块计划（10.54 C1，竖向按 10.62 B 改五次 + 分离端点）。"""

    def __init__(self, table: dict, zc0: float, vcz0: float, acc0: float,
                 dx0: float, dx0_rate: float, mass_total: float, liftoff_speed: float,
                 t_lo: float = 0.085, t_hi: float = 0.120) -> None:
        self.table = table
        self.zc0, self.vcz0, self.acc0 = float(zc0), float(vcz0), float(acc0)
        self.zc_end, self.vcz_end, self.acc_end = (float(table["D_td"]), float(liftoff_speed),
                                                   -9.81)
        self.mass_total, self.mass_lower = float(mass_total), float(table["m1"])
        # T 不是设定值：在 [t_lo, t_hi] 上取"N 峰最小且全程 N>=0、zc'>=0"的 T
        best = None
        for T in np.arange(t_lo, t_hi + 1e-12, 0.001):
            coeff = self._vertical_quintic(float(T))
            grid = np.linspace(0.0, float(T), 201)
            acc = np.array([self._poly(coeff, 2, t) for t in grid])
            vel = np.array([self._poly(coeff, 1, t) for t in grid])
            if acc.min() < -9.81 - 1e-6 or vel.min() < -1e-9:
                continue
            peak = self.mass_total * (9.81 + acc.max())
            if best is None or peak < best[1]:
                best = (float(T), float(peak), coeff)
        if best is None:
            raise ValueError("竖向计划无可行 T（[%.3f, %.3f] 内 N>=0 与 zc'>=0 不可兼得）"
                             % (t_lo, t_hi))
        self.T, self.n_peak, self._zc_coeff = best
        # 端点速度：解 [dzc/dL dzc/dphi; dH/dL' dH/dphi'] [L'; phi'] = [zc'(T); H_t]
        matrix = np.array([[table["dzc_dL"], table["dzc_dphi"]],
                           [table["dH_dLdot"], table["dH_dphidot"]]], dtype=float)
        rhs = np.array([self.vcz_end, -0.20], dtype=float)
        self.ldot_end, self.phidot_end = (float(v) for v in np.linalg.solve(matrix, rhs))
        self.dx_end = float(table["dx_eq"])
        self.dx_rate_end = float(table["ddx_dL"] * self.ldot_end
                                 + table["ddx_dphi"] * self.phidot_end)
        self.dx0, self.dx0_rate = float(dx0), float(dx0_rate)
        self._dx_coeff = self._horizontal_quintic()
        self._dx_coeff_rate = np.array([n * self._dx_coeff[n] if n >= 1 else 0.0
                                        for n in range(6)])
        self._dx_coeff_acc = np.array([n * (n - 1) * self._dx_coeff[n] if n >= 2 else 0.0
                                       for n in range(6)])

    # ---------------------------------------------------------------- 五次多项式
    @staticmethod
    def _poly(coeff: np.ndarray, order: int, t: float) -> float:
        value = 0.0
        for n in range(order, 6):
            factor = 1.0
            for k in range(order):
                factor *= (n - k)
            value += factor * coeff[n] * t ** (n - order)
        return float(value)

    @staticmethod
    def _solve_quintic(known: tuple[float, float, float], end: tuple[float, float, float],
                       T: float) -> np.ndarray:
        """起点给 位置/速度/加速度，终点给同样三量，解 6 个系数。"""
        matrix = np.array([[T ** n for n in (3, 4, 5)],
                           [n * T ** (n - 1) for n in (3, 4, 5)],
                           [n * (n - 1) * T ** (n - 2) for n in (3, 4, 5)]], dtype=float)
        base = np.array([known[0] + known[1] * T + 0.5 * known[2] * T * T,
                         known[1] + known[2] * T,
                         known[2]], dtype=float)
        tail = np.linalg.solve(matrix, np.array(end, dtype=float) - base)
        return np.array([known[0], known[1], 0.5 * known[2], tail[0], tail[1], tail[2]], dtype=float)

    def _vertical_quintic(self, T: float) -> np.ndarray:
        return self._solve_quintic((self.zc0, self.vcz0, self.acc0),
                                   (self.zc_end, self.vcz_end, self.acc_end), T)

    def _horizontal_quintic(self) -> np.ndarray:
        return self._solve_quintic((self.dx0, self.dx0_rate, 0.0),
                                   (self.dx_end, self.dx_rate_end, 0.0), self.T)

    # ---------------------------------------------------------------- 采样
    def zc(self, t: float) -> float:
        return self._poly(self._zc_coeff, 0, t)

    def vcz(self, t: float) -> float:
        return self._poly(self._zc_coeff, 1, t)

    def acc(self, t: float) -> float:
        return self._poly(self._zc_coeff, 2, t)

    def dx(self, t: float) -> float:
        powers = np.array([t ** n for n in range(6)], dtype=float)
        return float(powers @ self._dx_coeff)

    def dx_rate(self, t: float) -> float:
        powers = np.array([t ** (n - 1) if n >= 1 else 0.0 for n in range(6)], dtype=float)
        return float(powers @ self._dx_coeff_rate)

    def dx_acc(self, t: float) -> float:
        powers = np.array([t ** (n - 2) if n >= 2 else 0.0 for n in range(6)], dtype=float)
        return float(powers @ self._dx_coeff_acc)


class FlightPlan:
    """飞行段方案 (b)：接触点鼓包 + 质心弹道 -> 腿长由 D(L) 反查（10.54 D）。"""

    def __init__(self, table: dict, vc: float, D_lo: float, floor: float,
                 D_td_flight: float | None = None) -> None:
        self.table = table
        self.vc, self.D_lo, self.floor = float(vc), float(D_lo), float(floor)
        # 10.68：触地构型 = 计划触地腿长 L_td 对应的 D；None = 原行为（站立 D_td）
        self.D_td = float(table["D_td"] if D_td_flight is None else D_td_flight)
        disc = self.vc * self.vc - 2.0 * 9.81 * (self.D_td - self.D_lo)
        if disc < 0.0:
            raise ValueError("飞行计划无解（判别式<0）")
        self.T = float((self.vc + math.sqrt(disc)) / 9.81)

        def clearance(h: float) -> float:
            t = np.linspace(0.0, self.T, 201)
            u = t / self.T
            f = 64.0 * u ** 3 * (1.0 - u) ** 3
            zc = self.vc * t - 0.5 * 9.81 * t * t
            lref = np.array([L_of_D(self.table, self.D_lo + zc[i] - h * f[i]) for i in range(t.size)])
            return float(lref.min()) - self.floor

        lo, hi = 0.01, 0.60
        if clearance(lo) * clearance(hi) > 0.0:
            raise ValueError("飞行计划：找不到满足最小腿长约束的幅值 h")
        self.h = float(brentq(clearance, lo, hi, xtol=1e-6, maxiter=100))

    def sample(self, t: float) -> dict:
        u = 0.0 if self.T <= 0.0 else min(1.0, t / self.T)
        f = 64.0 * u ** 3 * (1.0 - u) ** 3
        f_d = 64.0 * 3.0 * u * u * (1.0 - u) ** 2 * (1.0 - 2.0 * u)
        f_dd = 64.0 * 6.0 * u * (1.0 - u) * ((1.0 - 2.0 * u) ** 2 - u * (1.0 - u))
        # 10.66：计划终点后保持终点构型（两块：下块触地构型、L = L_td），在空中等待触地，不外推弹道
        t_plan = min(float(t), self.T)
        zc = self.vc * t_plan - 0.5 * 9.81 * t_plan * t_plan
        zw = self.h * f
        l_ref = L_of_D(self.table, self.D_lo + zc - zw)
        dt = 1.0e-4
        if t >= self.T:
            l_next = l_ref
        else:
            l_next = L_of_D(self.table, self.D_lo + self.vc * (t + dt) - 0.5 * 9.81 * (t + dt) ** 2
                            - self.h * (f + f_d * dt / max(self.T, 1e-9)))
        return dict(u=u, f=f, zw=zw, l_ref=float(l_ref),
                    v_ref=float((l_next - l_ref) / dt),
                    z1_dd=float(self.h * f_dd / (self.T * self.T)),
                    phi_theta=float(phi_theta(self.table, l_ref)))


def dx_rel_bias(dynamics) -> float:
    """Δx 行（J2[x] - J1[x]）的速度积项：q ± eps*qvel 差分（与 rowset() 同口径）。"""
    model, data, scratch = dynamics.model, dynamics.data, dynamics.scratch
    qvel = np.asarray(data.qvel, dtype=float)
    eps = 1.0e-6
    values = []
    for sign in (1.0, -1.0):
        scratch.qpos[:] = data.qpos
        mujoco.mj_integratePos(model, scratch.qpos, qvel, sign * eps)
        scratch.qvel[:] = 0.0
        mujoco.mj_forward(model, scratch)
        two = TwoMassModel(model, scratch)
        j1, j2 = two.block_jacobians()
        values.append(float((j2[0] - j1[0]) @ qvel))
    del scratch
    mujoco.mj_forward(model, data)
    return float((values[0] - values[1]) / (2.0 * eps))


class LandingPlan:
    """落地段两块计划（10.68）：质心竖向五次，触地 N=0 起步（zc''(0)=-g），
    终点静止于 D(L_end)（zc'=zc''=0）；T 在 [t_lo, t_hi] 上取 N 峰最小且 N>=0、zc'<=0、zc>=终点。"""

    def __init__(self, table: dict, zc0: float, vcz0: float, length_end: float,
                 mass_total: float, t_lo: float = 0.060, t_hi: float = 0.250) -> None:
        self.table = table
        self.zc0, self.vcz0, self.acc0 = float(zc0), float(vcz0), -9.81
        self.zc_end = D_of_L(table, length_end)
        self.mass_total = float(mass_total)
        self.length_end = float(length_end)
        best = None
        for T in np.arange(t_lo, t_hi + 1e-12, 0.001):
            coeff = TakeoffPlan._solve_quintic((self.zc0, self.vcz0, self.acc0),
                                               (self.zc_end, 0.0, 0.0), float(T))
            grid = np.linspace(0.0, float(T), 201)
            acc = np.array([TakeoffPlan._poly(coeff, 2, t) for t in grid])
            vel = np.array([TakeoffPlan._poly(coeff, 1, t) for t in grid])
            pos = np.array([TakeoffPlan._poly(coeff, 0, t) for t in grid])
            if acc.min() < -9.81 - 1e-6 or vel.max() > 1e-9 or pos.min() < self.zc_end - 1e-9:
                continue
            peak = self.mass_total * (9.81 + acc.max())
            if best is None or peak < best[1]:
                best = (float(T), float(peak), coeff)
        if best is None:
            raise ValueError("落地计划无可行 T（[%.3f, %.3f] 内 N>=0、zc'<=0、zc>=终点 不可兼得）"
                             % (t_lo, t_hi))
        self.T, self.n_peak, self._zc_coeff = best

    def zc(self, t: float) -> float:
        return TakeoffPlan._poly(self._zc_coeff, 0, min(float(t), self.T))

    def vcz(self, t: float) -> float:
        return 0.0 if t >= self.T else TakeoffPlan._poly(self._zc_coeff, 1, float(t))

    def acc(self, t: float) -> float:
        return 0.0 if t >= self.T else TakeoffPlan._poly(self._zc_coeff, 2, float(t))

    def dx_eq_at(self, t: float) -> float:
        """导出量：计划质心高度对应的静平衡构型两块相对水平位置（dx_rel 计划）。"""
        return float(np.interp(L_of_D(self.table, self.zc(t)),
                               self.table["length_grid"], self.table["dx_grid"]))
