"""接触消力、动力学可实现性和严格优先级零空间求解。

M qacc + h = S.T tau + Jc.T f；接触约束为 Jc qacc + bias = 0。
消力后必须满足执行器左零空间的动力学等式，不能仅回代最小二乘力矩。
适配器负责沿移动几何接触点计算 bias，不能直接代入固定材料点向心加速度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


def dynamically_consistent_inverse(
    jacobian: np.ndarray,
    mass_matrix: np.ndarray,
    damping: float = 0.0,
) -> np.ndarray:
    """M 加权（动力学一致）阻尼伪逆 ``J̄ = M⁻¹Jᵀ(J M⁻¹ Jᵀ + λ²I)⁻¹``。

    ``damping`` 是奇异鲁棒项（§5.3 的排程量）。取 0 时是精确的动力学一致逆，
    此时 ``J̄ J`` 是 M 自伴投影算子、``N = I − J̄J`` 是精确的零空间投影算子；
    阻尼一旦大于 0 就不再是严格投影算子（这是阻尼的固有代价）。
    """
    J = np.atleast_2d(np.asarray(jacobian, dtype=float))
    M = np.asarray(mass_matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"质量矩阵必须是方阵，实际 {M.shape}")
    if J.shape[1] != M.shape[0]:
        raise ValueError(f"雅可比的列数必须等于 nv={M.shape[0]}，实际 {J.shape}")
    lam = float(damping)
    if lam < 0.0:
        raise ValueError("阻尼不能为负")

    inverse_mass_jacobian_t = np.linalg.solve(M, J.T)
    gram = J @ inverse_mass_jacobian_t
    if lam > 0.0:
        gram = gram + lam * lam * np.eye(gram.shape[0])
    # 用带截断的伪逆而不是 solve：任务行相关（或与已有约束冲突）时 gram 会奇异，
    # 硬解会给出 1e16 量级的解，把整个 q̈ 冲垮；截断掉那些奇异方向才是"最小二乘"。
    gram_inverse = np.linalg.pinv(gram, rcond=1e-12, hermitian=True)
    return inverse_mass_jacobian_t @ gram_inverse


@dataclass(frozen=True)
class TaskLevel:
    """一个优先级层级：J qacc = desired - bias。

    activation 缩放本级纠正增量；为 0 时释放本级权限。
    部分激活仍保留本级优先权，跨零的秩变化不保证力矩连续。
    """

    jacobian: np.ndarray
    desired_acceleration: np.ndarray
    bias_acceleration: np.ndarray | None = None  # J̇ q̇
    damping: float = 0.0
    name: str = ""
    activation: float = 1.0

    def __post_init__(self) -> None:
        J = np.atleast_2d(np.asarray(self.jacobian, dtype=float))
        desired = np.atleast_1d(np.asarray(self.desired_acceleration, dtype=float))
        if desired.size == 0:
            raise ValueError("任务的目标加速度不能为空")
        if J.shape[0] != desired.size:
            raise ValueError(
                f"任务 {self.name or '<unnamed>'} 的行数 {J.shape[0]} 与目标加速度"
                f"长度 {desired.size} 不一致"
            )
        bias = np.zeros(J.shape[0]) if self.bias_acceleration is None else np.asarray(
            self.bias_acceleration, dtype=float
        )
        if bias.size != J.shape[0]:
            raise ValueError(f"任务 {self.name or '<unnamed>'} 的偏置项长度不匹配")
        if not np.isfinite(self.activation) or not 0.0 <= self.activation <= 1.0:
            raise ValueError("激活因子必须在 [0,1] 内")
        if not np.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("阻尼必须有限且非负")
        if not all(np.all(np.isfinite(v)) for v in (J, desired, bias)):
            raise ValueError("任务输入必须有限")
        object.__setattr__(self, "jacobian", J)
        object.__setattr__(self, "desired_acceleration", desired)
        object.__setattr__(self, "bias_acceleration", bias)

    @property
    def dimension(self) -> int:
        return int(self.jacobian.shape[0])


@dataclass(frozen=True)
class TaskOutcome:
    """某一级任务解完之后的结果，用于残差监控与 §7.4 的零空间性质验证。"""

    name: str
    desired: np.ndarray
    achieved: np.ndarray
    residual: np.ndarray
    residual_norm: float
    conditioning: float
    activated: float = 1.0
    exhausted: bool = False

    def satisfied(self, tolerance: float = 1e-9) -> bool:
        """残差是否在容差内。

        不要写成 ``residual_norm == 0``：无阻尼、恰好欠约束时残差也只是 1e-16 量级，
        精确判零会把正常解判成失败。
        """
        return self.residual_norm <= float(tolerance)


@dataclass(frozen=True)
class HierarchyResult:
    """递归零空间解完一整条优先级链的结果。

    ``increments`` 是每一级对 ``q̈`` 的增量 ``Δq̈_i``（第 0 项是硬接触层）。
    积分起来就是 ``q̈ = Σ Δq̈_i``（零空间法是线性叠加的）。SNS 靠它把总力矩
    按层级分解，从而实现"分层限值"（§5.1）。
    """

    qacc: np.ndarray
    outcomes: tuple[TaskOutcome, ...]
    increments: tuple[np.ndarray, ...]
    null_space: np.ndarray
    contact_residual: np.ndarray
    contact_residual_norm: float
    contact_conditioning: float

    def outcome(self, name: str) -> TaskOutcome:
        for item in self.outcomes:
            if item.name == name:
                return item
        raise KeyError(name)


class IncrementLimiter(Protocol):
    """限制本级加速度增量，同时保持已有高优先级输出。"""

    def scale(self, level: TaskLevel, qacc: np.ndarray, increment: np.ndarray) -> float: ...


class NullSpaceHierarchy:
    """质量加权零空间递推。阻尼只用于本级增益，不用于剩余空间投影。

    每个实例只分解一次质量矩阵；每级只对剩余自由度做一次小规模 SVD。
    """

    AUTHORITY_TOLERANCE = 1.0e-9
    RANK_TOLERANCE = 1.0e-12

    def __init__(self, mass_matrix: np.ndarray, damping: float = 0.0,
                 authority_tolerance: float = AUTHORITY_TOLERANCE) -> None:
        M = np.asarray(mass_matrix, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"质量矩阵必须是方阵，实际 {M.shape}")
        if not np.all(np.isfinite(M)) or not np.allclose(M, M.T):
            raise ValueError("质量矩阵必须有限且对称正定")
        if not np.isfinite(damping) or float(damping) < 0.0:
            raise ValueError("阻尼必须有限且非负")
        if not np.isfinite(authority_tolerance) or float(authority_tolerance) < 0.0:
            raise ValueError("权限容差必须有限且非负")
        self.mass_matrix = M.copy()
        self.damping = float(damping)
        self.authority_tolerance = float(authority_tolerance)
        # Z.T M Z = I；后续只在 Z 的列空间内分解。
        self._whitener = np.linalg.solve(np.linalg.cholesky(M).T, np.eye(len(M)))

    @property
    def dof(self) -> int:
        return int(self.mass_matrix.shape[0])

    def _factor(self, jacobian: np.ndarray, basis: np.ndarray, damping: float):
        u, singular, vt = np.linalg.svd(jacobian @ basis, full_matrices=True)
        rank = int(np.sum(singular > self.RANK_TOLERANCE * singular[0])) if singular.size else 0
        if rank:
            values = singular[:rank]
            gain = (basis @ vt[:rank].T * (values / (values**2 + damping**2))) @ u[:, :rank].T
        else:
            gain = np.zeros((self.dof, jacobian.shape[0]))
        remaining = basis @ vt[rank:].T
        if not singular.size:
            condition = float("inf")
        else:
            smallest = singular[-1] if len(singular) == jacobian.shape[0] else 0.0
            denominator = smallest**2 + damping**2
            condition = float((singular[0]**2 + damping**2) / denominator) if denominator else float("inf")
        return gain, remaining, condition

    def solve(self, contact_jacobian: np.ndarray, contact_bias: np.ndarray | None,
              tasks: tuple[TaskLevel, ...] | list[TaskLevel],
              injected: tuple[TaskLevel, ...] | list[TaskLevel] = (), *,
              hard_constraint: TaskLevel | None = None,
              limiter: IncrementLimiter | None = None,
              initial_acceleration: np.ndarray | None = None) -> HierarchyResult:
        """硬接触与欠驱动动力学联立，然后按优先级求解各任务。

        hard_constraint 由 ContactElimination.dynamics_constraint 提供。
        未提供时仅求运动学任务。initial_acceleration 用于可行的降级初始解。
        """
        Jc = np.atleast_2d(np.asarray(contact_jacobian, dtype=float))
        if Jc.shape[1] != self.dof:
            raise ValueError(f"接触雅可比列数必须等于 nv={self.dof}，实际 {Jc.shape}")
        bias = np.zeros(Jc.shape[0]) if contact_bias is None else np.asarray(contact_bias, dtype=float).reshape(-1)
        if bias.size != Jc.shape[0]:
            raise ValueError("接触偏置项长度必须等于接触雅可比的行数")
        hard_jacobian, target = Jc, -bias
        if hard_constraint is not None:
            hard_jacobian = np.vstack((Jc, hard_constraint.jacobian))
            target = np.concatenate((-bias, hard_constraint.desired_acceleration - hard_constraint.bias_acceleration))
        inverse, basis, _ = self._factor(hard_jacobian, self._whitener, 0.0)
        qacc = inverse @ target if initial_acceleration is None else np.asarray(initial_acceleration, dtype=float).copy()
        if not np.allclose(hard_jacobian @ qacc, target, atol=1e-8, rtol=1e-8):
            raise ValueError("硬接触与动力学约束不相容，不能生成可执行加速度")
        increments = [qacc.copy()]
        levels = list(injected) + list(tasks)
        diagnostics = []
        for level in levels:
            projected = level.jacobian @ basis
            reference = np.linalg.norm(level.jacobian @ self._whitener)
            exhausted = np.linalg.norm(projected) <= self.authority_tolerance * reference or reference == 0.0
            if exhausted or level.activation == 0.0:
                increment = np.zeros(self.dof)
                condition = float("inf")
            else:
                gain, remaining, condition = self._factor(level.jacobian, basis, level.damping)
                error = level.desired_acceleration - level.bias_acceleration - level.jacobian @ qacc
                increment = level.activation * (gain @ error)
                if limiter is not None:
                    increment *= limiter.scale(level, qacc, increment)
                qacc += increment
                basis = remaining
            increments.append(increment)
            diagnostics.append((condition, exhausted))
        outcomes = []
        for level, (condition, exhausted) in zip(levels, diagnostics):
            achieved = level.jacobian @ qacc + level.bias_acceleration
            residual = achieved - level.desired_acceleration
            outcomes.append(TaskOutcome(level.name, level.desired_acceleration.copy(), achieved,
                                        residual, float(np.linalg.norm(residual)), condition,
                                        activated=level.activation, exhausted=exhausted))
        contact_residual = Jc @ qacc + bias
        return HierarchyResult(qacc, tuple(outcomes), tuple(increments),
                               basis @ basis.T @ self.mass_matrix, contact_residual,
                               float(np.linalg.norm(contact_residual)), self._conditioning(Jc, 0.0))

    def _conditioning(self, jacobian: np.ndarray, damping: float) -> float:
        if not jacobian.shape[0]:
            return 1.0
        weighted = jacobian @ self._whitener
        gram = weighted @ weighted.T + damping**2 * np.eye(jacobian.shape[0])
        return float(np.linalg.cond(gram))


@dataclass(frozen=True)
class ContactElimination:
    """``Jcᵀ`` 的正交分解结果（方案 §4.1）。

    ``recover_torque`` / ``recover_contact_force`` 互为逆过程：给定同一个
    ``q̈``，先算 τ 再回代出 f，得到的 f 与直接求解一致。
    """

    q_c: np.ndarray          # nv × rank(Jc)，接触力的像空间
    q_u: np.ndarray          # nv × (nv−nc)，接触零空间（力消掉之后剩下的方向）
    r_factor: np.ndarray     # rank(Jc) × nc，同一 SVD 基下的接触坐标
    selection: np.ndarray    # na × nv
    mass_matrix: np.ndarray
    damping: float = 0.0
    _actuation: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    _projector: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    @classmethod
    def from_matrices(
        cls,
        contact_jacobian: np.ndarray,
        selection: np.ndarray,
        mass_matrix: np.ndarray,
        damping: float = 0.0,
    ) -> "ContactElimination":
        Jc = np.atleast_2d(np.asarray(contact_jacobian, dtype=float))
        S = np.atleast_2d(np.asarray(selection, dtype=float))
        M = np.asarray(mass_matrix, dtype=float)
        nv = M.shape[0]
        if Jc.shape[1] != nv:
            raise ValueError(f"接触雅可比列数必须等于 nv={nv}，实际 {Jc.shape}")
        if S.shape[1] != nv:
            raise ValueError(f"选择矩阵列数必须等于 nv={nv}，实际 {S.shape}")
        if Jc.shape[0] >= nv:
            raise ValueError("接触约束行数必须小于 nv，否则没有剩余自由度")

        # Jcᵀ = [Q_c Q_u] [R; 0]，Q_u 的列与 Jcᵀ 的列空间正交 → Q_uᵀ Jcᵀ = 0
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("阻尼必须有限且非负")
        # 基与坐标来自同一 SVD；按实际秩释放空间。
        u_matrix, singular, vt = np.linalg.svd(Jc.T, full_matrices=True)
        rank = int(np.sum(singular > 1e-12 * singular[0])) if singular.size else 0
        q_c = u_matrix[:, :rank]
        q_u = u_matrix[:, rank:]
        r_factor = singular[:rank, None] * vt[:rank]
        actuation = q_u.T @ S.T
        projector = np.linalg.pinv(actuation, rcond=1e-12) @ q_u.T
        return cls(
            q_c=q_c,
            q_u=q_u,
            r_factor=r_factor,
            selection=S,
            mass_matrix=M,
            damping=float(damping),
            _actuation=actuation,
            _projector=projector,
        )

    @property
    def actuation_matrix(self) -> np.ndarray:
        """``Q_uᵀ Sᵀ``，形状 ``(nv−nc) × na``（本机 5×4，超定）。"""
        return self._actuation

    @property
    def actuation_conditioning(self) -> float:
        """``Q_uᵀ Sᵀ`` 的条件数（§4.1 要求逐拍监控的健康指标）。"""
        return float(np.linalg.cond(self._actuation))

    @property
    def unactuated_directions(self) -> np.ndarray:
        """接触消力后的执行器左零空间，每列一个欠驱动方向。"""
        u, singular, _ = np.linalg.svd(self._actuation, full_matrices=True)
        rank = int(np.sum(singular > 1e-12 * singular[0])) if singular.size else 0
        return u[:, rank:]

    @property
    def unactuated_direction(self) -> np.ndarray:
        directions = self.unactuated_directions
        if directions.shape[1] != 1:
            raise ValueError("欠驱动方向不是唯一一维，请使用 unactuated_directions")
        return directions[:, 0]

    def dynamics_constraint(self, bias: np.ndarray) -> TaskLevel | None:
        """W(M qacc+h)=0，W 为执行器左零空间映回广义空间。"""
        W = self.unactuated_directions.T @ self.q_u.T
        if not W.shape[0]:
            return None
        return TaskLevel(W @ self.mass_matrix, -W @ np.asarray(bias, dtype=float), name="dynamics")

    @property
    def torque_gain(self) -> np.ndarray:
        """``A = (Q_uᵀSᵀ)⁺ Q_uᵀ M``，形状 ``na × nv``：``τ`` 对 ``q̈`` 的线性部分。

        SNS 靠它把"执行器 i 固定在 c_i"写成加速度层的等式约束，也靠它把总力矩
        分解到各个优先级层级上（§5.1 的分层限值）。
        """
        return self._projector @ self.mass_matrix

    def torque_from_bias(self, bias: np.ndarray) -> np.ndarray:
        """``τ`` 里与 ``q̈`` 无关的那部分 ``(Q_uᵀSᵀ)⁺ Q_uᵀ h``。"""
        return self._projector @ np.asarray(bias, dtype=float)

    def recover_torque(self, qacc: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """``τ = (Q_uᵀ Sᵀ)⁺ Q_uᵀ (M q̈ + h)``。"""
        return self.torque_gain @ np.asarray(qacc, dtype=float) + self.torque_from_bias(bias)

    def actuation_residual(self, qacc: np.ndarray, bias: np.ndarray, torque: np.ndarray) -> np.ndarray:
        """``Q_uᵀ Sᵀ τ − Q_uᵀ(M q̈ + h)``：执行器补不上的那部分（欠驱动残差）。

        本机的欠驱动维数是 1（俯仰），所以残差应当只落在
        :attr:`unactuated_direction` 张成的那一维上。
        """
        projected = self.q_u.T @ (self.mass_matrix @ np.asarray(qacc, dtype=float)
                                  + np.asarray(bias, dtype=float))
        return self._actuation @ np.asarray(torque, dtype=float) - projected

    def recover_contact_force(
        self,
        qacc: np.ndarray,
        bias: np.ndarray,
        torque: np.ndarray,
    ) -> np.ndarray:
        """``f = R⁻¹ Q_cᵀ (M q̈ + h − Sᵀ τ)``，用于事后摩擦锥校验（§5.2）。"""
        net = (
            self.mass_matrix @ np.asarray(qacc, dtype=float)
            + np.asarray(bias, dtype=float)
            - self.selection.T @ np.asarray(torque, dtype=float)
        )
        return np.linalg.lstsq(self.r_factor, self.q_c.T @ net, rcond=1e-12)[0]
