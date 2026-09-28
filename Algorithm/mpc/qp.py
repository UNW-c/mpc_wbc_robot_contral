"""凝聚型（condensed）MPC 二次规划：把状态消掉，只留 ``2N`` 个力变量。

预测方程::

    X = Phi @ x0 + Gamma @ U + Psi        # X = [x_1; ...; x_N]

``Psi`` 对模型自带的常值漂移 ``w`` 是线性的，所以每个模型只需算一次映射
``Psi = D_map @ w``；每拍可变的已知输入（髋力矩，方案 §1.5）走同一个映射，见
:meth:`CondensedMPCQP.drift_response`。

代价::

    J = (X - X*)^T Qbar (X - X*) + U^T Rbar U + (D U - E u_prev)^T RbarD (...)

约束（全部写成对 ``U`` 的线性不等式，所以 QP 只有 ``5N`` 行）::

    1) 逐轮箱型   |f_i,k| <= min(tau_max / R, mu * f_z,i,k)   <- 力矩限 + 摩擦锥
    2) 偏航力矩   |half_track * (f_R - f_L)| <= M_z,max(q)     <- 含 scrub 折扣
    3) 俯仰带     |pitch_k| <= pitch_limit                     <- 保线性化有效
    4) 速度带     |v_k| <= speed_limit

其中 1) 永远可行（``U = 0`` 满足），3)/4) 是状态约束，可能不可行；求解器报不可行
时会自动松掉 3)/4) 再解一次，仍失败才让上层走兜底控制。

矩阵的稀疏结构在 :meth:`CondensedMPCQP.set_model` 里一次性定下来（P 取满上三角、
A 取满阵），之后每拍只 ``update(q=..., l=..., u=...)``，不重新分解稀疏结构。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sparse

from .model import INPUT_DIM, STATE_DIM, DiscreteModel

_SOLVED = (1, 2)  # OSQP_SOLVED, OSQP_SOLVED_INACCURATE
_INFINITY = 1.0e12


@dataclass(frozen=True)
class QPWeights:
    """状态/输入权重。

    ``pitch`` / ``pitch_rate`` / ``yaw`` / ``yaw_rate`` 沿用
    ``APP.config.LQRConfig.q_diagonal`` 的标定；``position`` / ``velocity`` 由
    MuJoCo 零速度抗扰扫描重新确定，用于让扰动后的位置回到原点（见
    ``docs/修改日志.md`` 与 ``tests/full_state_eval.py``）。

    ``uncontrollable_regularization`` 是方案 §1.1 要求的"缺失方向正则下界"：接触力
    的力扳手映射秩为 5，绕接触连线 ``û`` 的力矩恒不可控，而它的主要来源是共模轮力
    （两轮同向）。该值大于 0 时，对这个共模输入方向额外加一份二次正则，使 Hessian
    在几何退化（``û`` 逼近俯仰轴）时仍保持良态。默认 0.0：本实现里逐轮力权重
    ``force`` 已经大于 0，Hessian 本来就是正定的（实测 ``cond(P) ≈ 9.3e4``），
    这项只在把力权重调到 0 或上层要把俯仰权重大幅拉高时才需要打开。
    """

    # 2026-09-25 多工况联合扫描结果（small 0.2 / mid 0.5 / persist 持续推，
    # 固定 WBC pg=320/vg=320）：最优 51200/2000/80000/960。
    # 2026-09-27 试过 6400/480 候选（实测大扰动工况），但 viewer 复测在小扰动下
    # 出现极限环（位置持续震荡不衰减）——已回滚。
    position: float = 80000.0
    velocity: float = 2000.0
    pitch: float = 51200.0
    pitch_rate: float = 960.0
    yaw: float = 60.0
    yaw_rate: float = 8.0
    force: float = 0.02
    force_rate: float = 0.08
    terminal_scale: float = 5.0
    # P2：终端代价模式。"scale" = 终端状态权重 × terminal_scale（旧行为）；
    # "lqr" = 终端权重取离散 Riccati 解 P（在缩放变量空间计算，见 _solve_riccati）。
    terminal_mode: str = "scale"
    uncontrollable_regularization: float = 0.0
    # B3（A4-4）：8 状态模型的髋权重。初值只用于跑通流程，正式值由 B4/T5 扫描决定。
    hip_angle: float = 50.0  # 2026-09-28 部署为默认（E1/E4 推荐值；旧值 200）
    hip_rate: float = 10.0
    hip_torque: float = 0.02

    def state_diagonal(self, state_dim: int = STATE_DIM) -> np.ndarray:
        base = [self.position, self.velocity, self.pitch, self.pitch_rate, self.yaw, self.yaw_rate]
        if state_dim > STATE_DIM:
            base = base + [self.hip_angle, self.hip_rate]
        return np.array(base[:state_dim], dtype=float)

    def input_diagonal(self, input_dim: int = INPUT_DIM) -> np.ndarray:
        base = [self.force, self.force]
        if input_dim > INPUT_DIM:
            base = base + [self.hip_torque]
        return np.array(base[:input_dim], dtype=float)

    def input_rate_diagonal(self, input_dim: int = INPUT_DIM) -> np.ndarray:
        """输入变化率权重；当前所有输入共用 force_rate（B3 泛化）。"""
        return np.full(int(input_dim), float(self.force_rate))



@dataclass
class StageBounds:
    """逐步（stage-wise）的约束上下界。数组第 0 维长度必须等于 horizon。"""

    force_lower: np.ndarray
    force_upper: np.ndarray
    yaw_moment: np.ndarray
    pitch_limit: float | None = None
    speed_limit: float | None = None
    # B2：非轮力输入（如髋力矩）的箱型上下界，形状 (horizon, input_dim-2)；None 表示不约束。
    extra_lower: np.ndarray | None = None
    extra_upper: np.ndarray | None = None
    # B3：8 状态模型的共模髋角状态约束 |q_c| <= hip_angle_limit（None 表示不约束）。
    hip_angle_limit: float | None = None


@dataclass
class QPResult:
    status: str
    status_value: int
    iterations: int
    solve_time: float
    relaxed: bool = False
    objective: float = 0.0

    @property
    def solved(self) -> bool:
        return self.status_value in _SOLVED


def _dense_csc(matrix: np.ndarray) -> sparse.csc_matrix:
    """把稠密矩阵转成"全非零"CSC，保证稀疏结构不会因为数值恰好为 0 而变化。"""
    matrix = np.asarray(matrix, dtype=float)
    rows, columns = matrix.shape
    indices = np.tile(np.arange(rows, dtype=np.int32), columns)
    indptr = (np.arange(columns + 1, dtype=np.int32) * rows)
    data = np.asfortranarray(matrix).ravel(order="F").copy()
    return sparse.csc_matrix((data, indices, indptr), shape=(rows, columns))


def _dense_triu_csc(matrix: np.ndarray) -> sparse.csc_matrix:
    """满上三角 CSC（OSQP 只读 P 的上三角）。"""
    matrix = np.asarray(matrix, dtype=float)
    size = matrix.shape[0]
    indices = np.concatenate([np.arange(column + 1, dtype=np.int32) for column in range(size)])
    indptr = np.concatenate(([0], np.cumsum(np.arange(1, size + 1)))).astype(np.int32)
    data = np.concatenate([matrix[: column + 1, column] for column in range(size)])
    return sparse.csc_matrix((data, indices, indptr), shape=(size, size))


def _triu_data(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0]
    return np.concatenate([matrix[: column + 1, column] for column in range(size)])


# max_iter 的取法有实测依据：解落在箱型边界上（轮力饱和）时 OSQP 需要约 1120 次
# 迭代（H=25、dt=0.02、带髋力矩漂移），400 会让这种工况 100% 报"达到迭代上限"并退到
# 兜底——也就是最容易饱和的工况恰好是 MPC 最不可用的工况。取 2000 留约 1.8 倍余量。
# 不饱和时同规模问题只要 10~80 次，所以这个上限只在极端工况才被用到。
DEFAULT_OSQP_SETTINGS: dict[str, object] = {
    "verbose": False,
    "eps_abs": 1.0e-5,
    "eps_rel": 1.0e-5,
    "max_iter": 2000,
    "polishing": False,
    "warm_starting": True,
    "check_termination": 10,
}


class CondensedMPCQP:
    """把降阶模型 + 权重编译成一个规模固定的 OSQP 问题，之后只更新向量。"""

    def __init__(
        self,
        horizon: int,
        weights: QPWeights | None = None,
        osqp_settings: dict[str, object] | None = None,
        input_dim: int = INPUT_DIM,
        input_scaling: bool = True,
    ) -> None:
        if int(horizon) < 1:
            raise ValueError("horizon 必须 >= 1")
        if int(input_dim) < 2:
            raise ValueError("input_dim 至少为 2（两个轮力）")
        self.horizon = int(horizon)
        self.input_dim = int(input_dim)
        # F3：只供测试的开关（关闭后跳过 E2 的输入缩放，用于缩放不变性检查）。
        self._input_scaling_enabled = bool(input_scaling)
        self.weights = weights or QPWeights()
        self.settings = dict(DEFAULT_OSQP_SETTINGS)
        self.settings.update(osqp_settings or {})
        self.variables = self.horizon * self.input_dim
        # 状态带行数：输入箱型 + 偏航力矩 + 俯仰带 + 速度带（+ 髋角带，仅 8 状态模型）
        self._extra_state_rows = 1 if self.input_dim > INPUT_DIM else 0
        self.rows = self.horizon * (self.input_dim + 3 + self._extra_state_rows)
        self._solver: osqp.OSQP | None = None
        self._state_dim = STATE_DIM
        self._pitch_rows = np.arange(self.horizon) * STATE_DIM + 2
        self._velocity_rows = np.arange(self.horizon) * STATE_DIM + 1
        self._hip_rows = np.arange(self.horizon) * STATE_DIM + 6
        self._yaw_slice = slice(self.variables, self.variables + self.horizon)
        self._pitch_slice = slice(
            self.variables + self.horizon, self.variables + 2 * self.horizon
        )
        self._speed_slice = slice(
            self.variables + 2 * self.horizon, self.variables + 3 * self.horizon
        )
        self._hip_slice = (
            slice(self.variables + 3 * self.horizon, self.rows)
            if self._extra_state_rows else None
        )
        self.Phi = np.zeros((self.horizon * STATE_DIM, STATE_DIM))
        self.Gamma = np.zeros((self.horizon * STATE_DIM, self.variables))
        self.Psi = np.zeros(self.horizon * STATE_DIM)
        self._drift_map = np.zeros((self.horizon * STATE_DIM, STATE_DIM))
        self._condition_number: float | None = None

    # ---- 编译 ------------------------------------------------------------
    def set_model(self, discrete: DiscreteModel, half_track: float) -> None:
        """由离散模型重建 Phi/Gamma/Psi、P、A。模型不变时不需要重复调用。"""
        horizon = self.horizon
        n = int(discrete.A.shape[0])
        m = int(discrete.B.shape[1])
        if m != self.input_dim:
            raise ValueError(f"模型输入维数 {m} 与 QP input_dim {self.input_dim} 不一致")
        self._state_dim = n
        self._pitch_rows = np.arange(horizon) * n + 2
        self._velocity_rows = np.arange(horizon) * n + 1
        self._hip_rows = np.arange(horizon) * n + 6
        A, B, w = discrete.A, discrete.B, discrete.w
        # E2（#17）：输入缩放。非轮力输入（τ_c）的 B 列范数比轮力列大一个量级，
        # 直接进 Hessian 会把条件数推到 1e8。内部换成 ũ = u / s（s = ‖B[:,j]‖/‖B[:,0]‖），
        # 对外接口不变；6 状态（m=2）时 scale=1，行为逐位不变。
        self._input_scale = np.ones(m)
        if m > INPUT_DIM and self._input_scaling_enabled:
            base_norm = max(float(np.linalg.norm(B[:, 0])), 1e-9)
            for j in range(INPUT_DIM, m):
                self._input_scale[j] = max(base_norm / float(np.linalg.norm(B[:, j])), 1e-9)
        scaled_B = B * self._input_scale[None, :]
        powers = [np.eye(n)]
        for _ in range(horizon):
            powers.append(powers[-1] @ A)

        self.Phi = np.vstack([powers[k + 1] for k in range(horizon)])
        self.Gamma = np.zeros((horizon * n, horizon * m))
        for k in range(horizon):
            for j in range(k + 1):
                self.Gamma[k * n : (k + 1) * n, j * m : (j + 1) * m] = powers[k - j] @ scaled_B
        self._drift_map = np.zeros((horizon * n, n))
        for column in range(n):
            unit = np.zeros(n)
            unit[column] = 1.0
            accumulated = np.zeros(n)
            for k in range(horizon):
                accumulated = A @ accumulated + unit
                self._drift_map[k * n : (k + 1) * n, column] = accumulated
        self.Psi = self._drift_map @ np.asarray(w, dtype=float)

        state_weight = np.tile(self.weights.state_diagonal(n), horizon)
        terminal_matrix = None
        if self.weights.terminal_mode == "lqr":
            # P2：终端代价 x_N' P x_N（P 是满矩阵；state_weight 只支持对角，终端块置零后
            # 在 Hessian / 梯度里单独加满矩阵项）。
            terminal_matrix = self._solve_riccati(discrete, m, scaled_B)
            state_weight[(horizon - 1) * n:] = 0.0
        elif self.weights.terminal_mode == "scale":
            state_weight[-n:] *= float(self.weights.terminal_scale)
        else:
            raise ValueError(f"未知 terminal_mode: {self.weights.terminal_mode}")
        input_weight = np.tile(self.weights.input_diagonal(m) * self._input_scale ** 2, horizon)
        rate_weight = np.tile(self.weights.input_rate_diagonal(m) * self._input_scale ** 2, horizon)

        difference = np.zeros((horizon * m, horizon * m))
        for k in range(horizon):
            difference[k * m : (k + 1) * m, k * m : (k + 1) * m] = np.eye(m)
            if k > 0:
                difference[k * m : (k + 1) * m, (k - 1) * m : k * m] = -np.eye(m)
        previous_selector = np.zeros((horizon * m, m))
        previous_selector[:m, :] = np.eye(m)

        weighted_gamma = self.Gamma.T * state_weight
        hessian = 2.0 * (
            weighted_gamma @ self.Gamma
            + np.diag(input_weight)
            + difference.T @ (difference * rate_weight[:, None])
        )
        q_gain = 2.0 * weighted_gamma
        if terminal_matrix is not None:
            terminal_gamma = self.Gamma[(horizon - 1) * n:, :]
            hessian = hessian + 2.0 * (terminal_gamma.T @ terminal_matrix @ terminal_gamma)
            q_gain = q_gain.copy()
            q_gain[:, (horizon - 1) * n:] += 2.0 * (terminal_gamma.T @ terminal_matrix)
        uncontrollable = float(self.weights.uncontrollable_regularization)
        if uncontrollable > 0.0:
            # 共模轮力（两轮同向）产生俯仰力矩，也就是不可控方向 û 的主要来源；
            # 给它加一份正则，几何退化时 Hessian 不至于没有下界。
            common_mode = np.full((m, m), 0.5)
            hessian = hessian + 2.0 * uncontrollable * np.kron(np.eye(horizon), common_mode)
        hessian = 0.5 * (hessian + hessian.T)
        self._q_gain = q_gain
        self._rate_gain = -2.0 * difference.T @ (previous_selector * rate_weight[:, None])

        constraint = np.zeros((self.rows, self.variables))
        constraint[: self.variables, :] = np.eye(self.variables)
        for k in range(horizon):
            constraint[self.variables + k, k * m + 0] = float(half_track)
            constraint[self.variables + k, k * m + 1] = -float(half_track)
        constraint[self._pitch_slice, :] = self.Gamma[self._pitch_rows, :]
        constraint[self._speed_slice, :] = self.Gamma[self._velocity_rows, :]
        if self._hip_slice is not None:
            constraint[self._hip_slice, :] = self.Gamma[self._hip_rows, :]

        self._hessian = hessian
        self._constraint = constraint
        self._condition_number = None
        if self._solver is None:
            self._solver = osqp.OSQP()
            self._solver.setup(
                _dense_triu_csc(hessian),
                np.zeros(self.variables),
                _dense_csc(constraint),
                np.full(self.rows, -_INFINITY),
                np.full(self.rows, _INFINITY),
                **self.settings,
            )
        else:
            self._solver.update(
                Px=_triu_data(hessian),
                Ax=np.asfortranarray(constraint).ravel(order="F").copy(),
            )

    # ---- 每拍求解 --------------------------------------------------------
    def _solve_riccati(self, discrete, input_dim: int, scaled_B) -> np.ndarray:
        """P2：终端权重 = 离散 Riccati 解 P（缩放变量空间，R 与 Gamma 的尺度一致）。"""
        from scipy.linalg import solve_discrete_are

        state_dim = int(discrete.A.shape[0])
        Q = np.diag(self.weights.state_diagonal(state_dim))
        R = np.diag(np.maximum(self.weights.input_diagonal(input_dim) * self._input_scale ** 2, 1e-9))
        return np.asarray(solve_discrete_are(discrete.A, scaled_B, Q, R), dtype=float)

    def _bounds(
        self,
        bounds: StageBounds,
        free_response: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = np.empty(self.rows, dtype=float)
        upper = np.empty(self.rows, dtype=float)
        if self.input_dim == INPUT_DIM:
            lower[: self.variables] = np.asarray(bounds.force_lower, dtype=float).ravel()
            upper[: self.variables] = np.asarray(bounds.force_upper, dtype=float).ravel()
        else:
            lower_input = np.full((self.horizon, self.input_dim), -_INFINITY)
            upper_input = np.full((self.horizon, self.input_dim), _INFINITY)
            lower_input[:, :2] = np.asarray(bounds.force_lower, dtype=float)
            upper_input[:, :2] = np.asarray(bounds.force_upper, dtype=float)
            if bounds.extra_lower is not None:
                lower_input[:, 2:] = np.asarray(bounds.extra_lower, dtype=float)
            if bounds.extra_upper is not None:
                upper_input[:, 2:] = np.asarray(bounds.extra_upper, dtype=float)
            lower_input = lower_input / self._input_scale[None, :]
            upper_input = upper_input / self._input_scale[None, :]
            lower[: self.variables] = lower_input.ravel()
            upper[: self.variables] = upper_input.ravel()
        moment = np.abs(np.asarray(bounds.yaw_moment, dtype=float)).ravel()
        lower[self._yaw_slice] = -moment
        upper[self._yaw_slice] = moment

        pitch_free = free_response[self._pitch_rows]
        if bounds.pitch_limit is None:
            lower[self._pitch_slice] = -_INFINITY
            upper[self._pitch_slice] = _INFINITY
        else:
            limit = abs(float(bounds.pitch_limit))
            lower[self._pitch_slice] = -limit - pitch_free
            upper[self._pitch_slice] = limit - pitch_free

        speed_free = free_response[self._velocity_rows]
        if bounds.speed_limit is None:
            lower[self._speed_slice] = -_INFINITY
            upper[self._speed_slice] = _INFINITY
        else:
            limit = abs(float(bounds.speed_limit))
            lower[self._speed_slice] = -limit - speed_free
            upper[self._speed_slice] = limit - speed_free

        if self._hip_slice is not None:
            hip_free = free_response[self._hip_rows]
            if bounds.hip_angle_limit is None:
                lower[self._hip_slice] = -_INFINITY
                upper[self._hip_slice] = _INFINITY
            else:
                limit = abs(float(bounds.hip_angle_limit))
                lower[self._hip_slice] = -limit - hip_free
                upper[self._hip_slice] = limit - hip_free
        return lower, upper

    def _input_bounds(self, bounds: StageBounds) -> tuple[np.ndarray, np.ndarray]:
        """完整的输入箱型上下界，形状 (horizon, input_dim)（含 extra 输入，B3）。"""
        lower = np.asarray(bounds.force_lower, dtype=float)
        upper = np.asarray(bounds.force_upper, dtype=float)
        if self.input_dim > INPUT_DIM:
            width = self.input_dim - INPUT_DIM
            extra_lower = (np.full((self.horizon, width), -_INFINITY)
                           if bounds.extra_lower is None
                           else np.asarray(bounds.extra_lower, dtype=float))
            extra_upper = (np.full((self.horizon, width), _INFINITY)
                           if bounds.extra_upper is None
                           else np.asarray(bounds.extra_upper, dtype=float))
            lower = np.hstack((lower, extra_lower))
            upper = np.hstack((upper, extra_upper))
        # 返回缩放变量空间的界（E2）
        return lower / self._input_scale[None, :], upper / self._input_scale[None, :]

    def drift_response(self, drift: np.ndarray | None = None) -> np.ndarray:
        """常值漂移项在时域上的响应 ``Psi``。

        ``drift`` 是**每拍可变**的已知输入（髋力矩折算来的俯仰角加速度，方案 §1.5），
        单位是"每步增量"，与 :meth:`ReducedModel.discretize` 产出的 ``w`` 同量纲；
        连续时间量要先乘 :meth:`ReducedModel.drift_step_map`。传入时用事先算好的
        线性映射重算，不动任何稀疏结构；不传时退回模型自带的漂移项。
        """
        if drift is None:
            return self.Psi
        return self._drift_map @ np.asarray(drift, dtype=float).reshape(self._state_dim)

    def condition_number(self) -> float:
        """Hessian 的条件数（方案 §1.1 要求监控的量）。

        只与模型和权重有关，与每拍的状态/参考无关，所以第一次算完就缓存，之后调用
        不产生开销。
        """
        if self._condition_number is None:
            self._condition_number = float(np.linalg.cond(self._hessian))
        return self._condition_number

    def solve(
        self,
        state: np.ndarray,
        reference_states: np.ndarray,
        bounds: StageBounds,
        previous_input: np.ndarray,
        warm_start: np.ndarray | None = None,
        drift: np.ndarray | None = None,
    ) -> tuple[np.ndarray, QPResult]:
        """解一拍 QP，返回 ``(forces (N, 2), QPResult)``。

        状态约束不可行时自动松掉俯仰/速度带重解一次，并在结果里标记 ``relaxed``。
        ``drift`` 是本拍的可变已知输入，见 :meth:`drift_response`。
        """
        if self._solver is None:
            raise RuntimeError("请先调用 set_model()")
        state = np.asarray(state, dtype=float).reshape(self._state_dim)
        reference = np.asarray(reference_states, dtype=float)
        if reference.shape != (self.horizon + 1, self._state_dim):
            raise ValueError(
                f"reference_states 应为 ({self.horizon + 1}, {self._state_dim})，实际 {reference.shape}"
            )
        free_response = self.Phi @ state + self.drift_response(drift)
        target = reference[1:].reshape(-1)
        linear = self._q_gain @ (free_response - target)
        previous_scaled = (np.asarray(previous_input, dtype=float).reshape(self.input_dim)
                           / self._input_scale)
        linear += self._rate_gain @ previous_scaled
        lower, upper = self._bounds(bounds, free_response)

        self._solver.update(q=linear, l=lower, u=upper)
        if warm_start is not None:
            warm = (np.asarray(warm_start, dtype=float).reshape(self.horizon, self.input_dim)
                    / self._input_scale[None, :])
            self._solver.warm_start(x=warm.ravel())
        result = self._solver.solve()
        relaxed = False
        if int(result.info.status_val) not in _SOLVED:
            relaxed = True
            lower[self._pitch_slice] = -_INFINITY
            upper[self._pitch_slice] = _INFINITY
            lower[self._speed_slice] = -_INFINITY
            upper[self._speed_slice] = _INFINITY
            if self._hip_slice is not None:
                # E3 修复：髋角带也必须参与降级重解，否则不可行时永远失败
                lower[self._hip_slice] = -_INFINITY
                upper[self._hip_slice] = _INFINITY
            self._solver.update(l=lower, u=upper)
            result = self._solver.solve()

        status_value = int(result.info.status_val)
        solution = np.asarray(result.x, dtype=float) if result.x is not None else None
        if status_value in _SOLVED and solution is not None and np.all(np.isfinite(solution)):
            forces = solution.reshape(self.horizon, self.input_dim)
        else:
            forces = np.zeros((self.horizon, self.input_dim), dtype=float)
        # 逐轮箱型（力矩限 + 摩擦锥）是安全性约束，无论 QP 结果如何都必须成立
        input_lower, input_upper = self._input_bounds(bounds)  # 缩放空间
        forces = np.clip(forces, input_lower, input_upper)
        forces = forces * self._input_scale[None, :]           # 还原为实际输入 u
        return forces, QPResult(
            status=str(result.info.status),
            status_value=status_value,
            iterations=int(result.info.iter),
            solve_time=float(result.info.run_time),
            relaxed=relaxed,
            objective=float(result.info.obj_val),
        )
