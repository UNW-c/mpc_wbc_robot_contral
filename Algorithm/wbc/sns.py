"""按严格优先级限制零空间任务的力矩增量。

先保证完整动力学等式，然后逐级使用硬件限值与任务预算的剩余区间。
不再对所有任务共同缩放，也不反复求解整条层级。法向单边性与摩擦锥
仍需接触适配层提供额外约束，feasible 仅表示等式动力学和力矩箱可行。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .projection import (ContactElimination, HierarchyResult, NullSpaceHierarchy, TaskLevel,
                         dynamically_consistent_inverse)

_ZERO_TOLERANCE = 1e-9
_SLOPE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ActuatorLimits:
    """执行器硬件力矩限（本机：髋 ±10、轮 ±5，方案 §6）。"""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        low = np.atleast_1d(np.asarray(self.lower, dtype=float)).copy()
        high = np.atleast_1d(np.asarray(self.upper, dtype=float)).copy()
        if low.shape != high.shape:
            raise ValueError("上下限形状必须一致")
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("力矩上下限必须有限")
        if low.ndim != 1 or not low.size:
            raise ValueError("力矩上下限必须是非空向量")
        if np.any(low > high):
            raise ValueError("存在下限大于上限的执行器")
        object.__setattr__(self, "lower", low)
        object.__setattr__(self, "upper", high)

    @classmethod
    def symmetric(cls, limit: np.ndarray | float) -> "ActuatorLimits":
        magnitude = np.atleast_1d(np.asarray(limit, dtype=float))
        return cls(lower=-magnitude, upper=magnitude)

    @property
    def size(self) -> int:
        return int(self.upper.size)

    def violation(self, torque: np.ndarray) -> np.ndarray:
        """逐执行器越限量，未越限处为 0（非负）。"""
        value = np.asarray(torque, dtype=float)
        return np.maximum(np.maximum(self.lower - value, value - self.upper), 0.0)

    def clamp(self, torque: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(torque, dtype=float), self.lower, self.upper)

    def contains(self, torque: np.ndarray, tolerance: float = _ZERO_TOLERANCE) -> bool:
        return bool(np.all(self.violation(torque) <= tolerance))


@dataclass(frozen=True)
class SNSResult:
    """一次 SNS 解算的结果与健康指标。"""

    torque: np.ndarray
    qacc: np.ndarray
    hierarchy: HierarchyResult
    scaling: dict[str, float]
    clamped: dict[int, float]
    attribution: np.ndarray
    iterations: int
    feasible: bool
    clip_applied: bool = False
    fallback_used: bool = False
    dynamics_residual_norm: float = 0.0

    def scale_of(self, name: str) -> float:
        return float(self.scaling.get(name, 1.0))


def maximum_scale(
    rest: np.ndarray,
    contribution: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """求满足 ``lower ≤ rest + s·contribution ≤ upper`` 的最大 ``s ∈ [0, 1]``。

    逐执行器解出 ``s`` 的可行区间再取交集。``contribution`` 为零的执行器不约束
    ``s``，但 ``rest`` 自己越界时整条无解（返回 0，由上层报不可行）。
    """
    rest = np.asarray(rest, dtype=float)
    contribution = np.asarray(contribution, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    best = 1.0
    required = 0.0
    for index in range(contribution.size):
        slope = contribution[index]
        if abs(slope) <= _SLOPE_TOLERANCE:
            # 这一级推不动该执行器：rest 自己越界就整条无解
            if rest[index] < lower[index] - _ZERO_TOLERANCE or rest[index] > upper[index] + _ZERO_TOLERANCE:
                return 0.0
            continue
        first = (lower[index] - rest[index]) / slope
        second = (upper[index] - rest[index]) / slope
        low, high = min(first, second), max(first, second)
        # 与 [0, 1] 求交：交集为空说明这一级怎么缩都压不回界内
        low = max(low, 0.0)
        high = min(high, 1.0)
        if high < low - _ZERO_TOLERANCE:
            return 0.0
        required = max(required, low)
        best = min(best, high)
        if best < required - _ZERO_TOLERANCE:
            return 0.0
    return float(min(max(best, 0.0), 1.0))


class _TorqueBudgetLimiter:
    """每级只对一个仿射方向求区间交集，不重复分解任务链。"""

    def __init__(self, gain, offset, limits, budgets, tasks):
        self.gain = gain
        self.offset = offset
        self.limits = limits
        self.budgets = budgets
        self.scaling = {task.name: 0.0 for task in tasks}

    def scale(self, level: TaskLevel, qacc: np.ndarray, increment: np.ndarray) -> float:
        torque = self.gain @ qacc + self.offset
        contribution = self.gain @ increment
        lower, upper = self.limits.lower, self.limits.upper
        rest, slope = torque, contribution
        if level.name in self.budgets:
            budget_low, budget_high = self.budgets[level.name]
            rest = np.concatenate((rest, np.zeros_like(contribution)))
            slope = np.concatenate((slope, contribution))
            lower = np.concatenate((lower, budget_low))
            upper = np.concatenate((upper, budget_high))
        value = maximum_scale(rest, slope, lower, upper)
        self.scaling[level.name] = value
        return value


class SaturationResolver:
    """严格优先级的零空间增量限幅。

    每级沿当前纠正方向缩放，冻结已达到的高优先级输出。该保守分配不搜索
    执行器冗余空间中的全局最大任务缩放，不宣称等价于完整最优 SNS。
    scaling 表示本级纠正增量的比例，不是原始目标加速度的比例。
    """

    def __init__(self, hierarchy: NullSpaceHierarchy, elimination: ContactElimination,
                 max_rounds: int | None = None) -> None:
        self.hierarchy = hierarchy
        self.elimination = elimination
        # 保留旧接口；当前固定两次层级调用，无钳位迭代。
        self.max_rounds = int(max_rounds) if max_rounds is not None else elimination.selection.shape[0] + 2
        if self.max_rounds < 2:
            raise ValueError("求解预算至少允许两次层级调用")

    def resolve(self, contact_jacobian: np.ndarray, contact_bias: np.ndarray | None,
                tasks: tuple[TaskLevel, ...] | list[TaskLevel], limits: ActuatorLimits,
                budgets: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                dynamics_bias: np.ndarray | None = None,
                external_force: np.ndarray | None = None) -> SNSResult:
        tasks = list(tasks)
        if not tasks:
            raise ValueError("至少要有一个任务层级")
        if len({task.name for task in tasks}) != len(tasks):
            raise ValueError("任务名不能重复")
        Jc = np.atleast_2d(np.asarray(contact_jacobian, dtype=float))
        bias = np.zeros(Jc.shape[0]) if contact_bias is None else np.asarray(contact_bias, dtype=float).reshape(-1)
        if len(bias) != Jc.shape[0]:
            raise ValueError("接触偏置长度必须等于接触雅可比的行数")
        h = np.zeros(self.hierarchy.dof) if dynamics_bias is None else np.asarray(dynamics_bias, dtype=float).reshape(-1)
        if len(h) != self.hierarchy.dof:
            raise ValueError("动力学偏置长度必须等于 nv")
        if external_force is not None:
            disturbance = np.asarray(external_force, dtype=float)
            if disturbance.shape != (self.hierarchy.dof,) or not np.all(np.isfinite(disturbance)):
                raise ValueError("external_force must be a finite nv-dimensional vector")
            # M*qacc + h = S.T*tau + J.T*f + external. This remains inside
            # equality/torque limiting; no post-solve torque patch is permitted.
            h = h - disturbance
        if limits.size != self.elimination.selection.shape[0]:
            raise ValueError("限值个数必须等于执行器个数")
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(Jc)) or not np.all(np.isfinite(bias)):
            raise ValueError("动力学输入必须有限")
        checked_budgets = {}
        for name, bounds in (budgets or {}).items():
            budget = ActuatorLimits(*bounds)
            if budget.size != limits.size:
                raise ValueError(f"任务 {name} 的预算长度必须等于执行器个数")
            if not budget.contains(np.zeros(limits.size)):
                raise ValueError("增量力矩预算必须包含零")
            checked_budgets[name] = (budget.lower, budget.upper)
        e = self.elimination
        hard = e.dynamics_constraint(h)
        base = self.hierarchy.solve(Jc, bias, [], hard_constraint=hard)
        base_torque = e.recover_torque(base.qacc, h)
        fallback = not limits.contains(base_torque)
        if fallback:
            # 最小加速度基线本身超限时，本拍退出任务跟踪，并从实际有界力矩
            # 正向重算加速度。不能裁剪力矩后仍返回裁剪前的 qacc。
            torque = limits.clamp(base_torque)
            free = np.linalg.solve(e.mass_matrix, e.selection.T @ torque - h)
            inverse = dynamically_consistent_inverse(Jc, e.mass_matrix)
            qacc = free + inverse @ (-bias - Jc @ free)
            result = self.hierarchy.solve(Jc, bias, [replace(task, activation=0.0) for task in tasks],
                                          hard_constraint=hard, initial_acceleration=qacc)
            scaling = {task.name: 0.0 for task in tasks}
            attribution = np.zeros((1 + len(tasks), limits.size))
            attribution[0] = torque
        else:
            limiter = _TorqueBudgetLimiter(e.torque_gain, e.torque_from_bias(h), limits, checked_budgets, tasks)
            result = self.hierarchy.solve(Jc, bias, tasks, hard_constraint=hard,
                                          limiter=limiter, initial_acceleration=base.qacc)
            torque = e.recover_torque(result.qacc, h)
            scaling = limiter.scaling
            # 无权限但目标已满足，不应误报任务需要缩放。
            for outcome in result.outcomes:
                if outcome.exhausted and outcome.satisfied():
                    scaling[outcome.name] = 1.0
            attribution = self.attribution(result, h)
        force = e.recover_contact_force(result.qacc, h, torque)
        residual = e.mass_matrix @ result.qacc + h - e.selection.T @ torque - Jc.T @ force
        dynamics_norm = float(np.linalg.norm(residual))
        feasible = limits.contains(torque) and result.contact_residual_norm < 1e-7 and dynamics_norm < 1e-7
        clamped = {i: float(torque[i]) for i in range(limits.size)
                   if abs(torque[i] - limits.lower[i]) <= _ZERO_TOLERANCE
                   or abs(torque[i] - limits.upper[i]) <= _ZERO_TOLERANCE} if fallback else {}
        return SNSResult(torque, result.qacc, result, scaling, clamped, attribution,
                         2, feasible, clip_applied=fallback, fallback_used=fallback,
                         dynamics_residual_norm=dynamics_norm)

    def attribution(self, result: HierarchyResult, bias: np.ndarray) -> np.ndarray:
        gain = self.elimination.torque_gain
        rows = [gain @ increment for increment in result.increments]
        rows[0] = rows[0] + self.elimination.torque_from_bias(bias)
        return np.vstack(rows)
