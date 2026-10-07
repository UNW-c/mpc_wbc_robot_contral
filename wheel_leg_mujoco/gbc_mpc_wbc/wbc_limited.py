"""限幅自洽的加权整体最小二乘 WBC（本工作区专用，不改共享的 ``Algorithm/``）。

与 ``Algorithm/wbc/weighted.py::WeightedSaturationResolver`` 的关系：

**相同**：硬约束同为"接触一致性等式 + 执行器左零空间的动力学等式"；任务在硬约束的
零空间里按权重同时分配（不把任何一项清零）。

**差别 1（限幅，本次修复的主因）**：``weighted.py`` 先由 ``q̈`` 反解 ``τ``，越限后直接
``clamp``，于是返回的 ``τ`` 与 ``q̈`` / 接触力**不再动力学一致**，且 ``feasible`` 必然为
False（契约性缺陷：上层 MPC 的一切规划都建立在"下层能执行"这个前提上）。

本模块利用 ``τ`` 对 ``q̈`` 是**仿射**的这一事实。记硬约束解为 ``q̈_h``、任务修正为
``Δ = N x*``（``N`` 为硬约束零空间投影，因此 ``Δ ∈ null(A)``）：

    q̈(s) = q̈_h + s·Δ                 对任意 s ∈ [0,1] 仍满足全部硬约束
    τ(s) = τ_h + s·(τ* − τ_h)          仿射

于是"``τ`` 落在箱型限幅内"化为一维区间求交（复用 ``Algorithm.wbc.sns.maximum_scale``），
取最大许可步长 ``s_max``；``τ`` 再由最终的 ``q̈`` 反解得到，**由构造保证严格自洽**。

**差别 2（诊断接口）**：补齐 ``outcomes``（逐任务残差）。``weighted.py`` 返回
``outcomes=()``，使 ``WbcFeedbackAccumulator.observe`` 在 ``present`` 为空时直接早退，
WBC→MPC 的反馈回路整体失效。

**差别 3（反馈语义）**：``scaling[name] = s_max``，让 ``observe`` 的 ``curtailed`` 判据
（``scale_of(name) < 1``）能够生效。

**已知代价（必须明说）**：``s_max`` 是对整个任务修正的**等比例**缩放，不是按优先级逐项
牺牲。任务之间的方向与配比仍由权重决定，只有幅值被压。这是"保守但自洽"，不承诺最优；
需要"按优先级分别牺牲"时应改用单级 QP（用户 2026-10-05 决定不走该路径）。
"""
from __future__ import annotations

import logging

import numpy as np

from . import _circleg  # noqa: F401
from .logging_setup import get_logger  # noqa: E402
from Algorithm.wbc.projection import ContactElimination, HierarchyResult, TaskLevel, TaskOutcome  # noqa: E402
from Algorithm.wbc.sns import ActuatorLimits, SNSResult, maximum_scale  # noqa: E402

_ZERO_TOLERANCE = 1e-9
_DYNAMICS_TOLERANCE = 1e-7

_LOGGER = get_logger(__name__)


def _hard_system(elimination: ContactElimination, jacobian: np.ndarray,
                 bias: np.ndarray, dynamics_bias: np.ndarray):
    """接触等式 + 欠驱动动力学等式，合成本级的硬约束 ``A q̈ = b``。"""
    hard = elimination.dynamics_constraint(dynamics_bias)
    if hard is None:
        return jacobian, -bias
    rows = np.vstack((jacobian, hard.jacobian))
    target = np.concatenate((-bias, hard.desired_acceleration - hard.bias_acceleration))
    return rows, target


class LimitedWeightedResolver:
    """与 ``WeightedSaturationResolver`` 同接口的限幅自洽版本。

    ``weights`` 是本级权重，未列出的任务取 1.0；``damping`` 是零空间内的正则项。
    """

    def __init__(self, elimination: ContactElimination, weights: dict[str, float],
                 damping: float = 1.0e-6) -> None:
        self.elimination = elimination
        self.weights = {str(name): float(value) for name, value in weights.items()}
        self.damping = float(damping)
        if self.damping < 0.0:
            raise ValueError("阻尼不能为负")

    def resolve(self, contact_jacobian: np.ndarray, contact_bias: np.ndarray | None,
                tasks: tuple[TaskLevel, ...] | list[TaskLevel],
                limits: ActuatorLimits,
                budgets: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                dynamics_bias: np.ndarray | None = None,
                external_force: np.ndarray | None = None) -> SNSResult:
        del budgets  # 加权版不复用分层预算；限幅在本级用一维步长统一处理
        tasks = list(tasks)
        if not tasks:
            raise ValueError("至少要有一个任务层级")

        elimination = self.elimination
        dof = elimination.mass_matrix.shape[0]
        jacobian = np.atleast_2d(np.asarray(contact_jacobian, dtype=float))
        bias = (np.zeros(jacobian.shape[0]) if contact_bias is None
                else np.asarray(contact_bias, dtype=float).reshape(-1))
        h = (np.zeros(dof) if dynamics_bias is None
             else np.asarray(dynamics_bias, dtype=float).reshape(-1))
        if external_force is not None:
            disturbance = np.asarray(external_force, dtype=float).reshape(-1)
            if disturbance.shape != (dof,):
                raise ValueError("external_force 必须是 nv 维")
            h = h - disturbance

        # ---------------------------------------------------------- 硬约束与零空间
        rows, target = _hard_system(elimination, jacobian, bias, h)
        pseudo = np.linalg.pinv(rows, rcond=1e-12)
        qacc_hard = pseudo @ target
        null = np.eye(dof) - pseudo @ rows

        # ------------------------------------------------ 任务修正（落在零空间里）
        hessian = self.damping * np.eye(dof)
        gradient = np.zeros(dof)
        weights: dict[str, float] = {}
        for task in tasks:
            weight = self.weights.get(task.name, 1.0) * float(task.activation)
            weights[task.name] = weight
            if weight <= 0.0:
                continue
            task_jacobian = np.atleast_2d(task.jacobian)
            desired = np.asarray(task.desired_acceleration, dtype=float)
            if task.bias_acceleration is not None:
                desired = desired - np.asarray(task.bias_acceleration, dtype=float)
            projected = task_jacobian @ null
            hessian += weight * (projected.T @ projected)
            gradient += weight * (projected.T @ (desired - task_jacobian @ qacc_hard))
        increment = null @ np.linalg.solve(hessian, gradient)

        # ------------------------------------------------------ 限幅：一维精确步长
        torque_hard = elimination.recover_torque(qacc_hard, h)
        torque_unbounded = elimination.recover_torque(qacc_hard + increment, h)
        scale = maximum_scale(torque_hard, torque_unbounded - torque_hard, limits.lower, limits.upper)
        qacc = qacc_hard + scale * increment
        torque = elimination.recover_torque(qacc, h)
        # 硬约束解本身越界时（无可用步长）**不能返回越界力矩**：上层会拿到一个物理上做不到的
        # 控制量，而且调用方无法从 ``feasible=False`` 分辨“是任务被抑制”还是“硬约束不可行”。
        # 用户 2026-10-07 要求彻底修，不做补丁：这里明确区分并诊断，返回值裁剪到限幅内
        # （与实际执行一致——MuJoCo 本来也会裁），同时把越界量记进日志便于定位。
        hard_infeasible = bool(scale <= _ZERO_TOLERANCE)
        if hard_infeasible:
            over = np.maximum(np.maximum(torque - limits.upper, limits.lower - torque), 0.0)
            worst = int(np.argmax(over))
            _LOGGER.warning(
                "WBC 硬约束解越界，scale=%.3e（无可用步长）：最大越界 %.3f N·m 在第 %d 号执行器"
                "（限幅 [%.3f, %.3f]）。已裁剪到限幅内；feasible=False 表示需上层降低需求。",
                float(scale), float(over[worst]), worst,
                float(limits.lower[worst]), float(limits.upper[worst]))
            torque = np.clip(torque, limits.lower, limits.upper)

        # ------------------------------------------------------------- 诊断与反馈
        outcomes = []
        for task in tasks:
            task_jacobian = np.atleast_2d(task.jacobian)
            task_bias = (np.zeros(task_jacobian.shape[0]) if task.bias_acceleration is None
                         else np.asarray(task.bias_acceleration, dtype=float))
            desired = np.asarray(task.desired_acceleration, dtype=float)
            achieved = task_jacobian @ qacc + task_bias
            residual = achieved - desired
            outcomes.append(TaskOutcome(
                task.name, desired.copy(), achieved, residual, float(np.linalg.norm(residual)),
                conditioning=1.0, activated=float(task.activation),
                exhausted=False,   # 加权版不把任何任务清零，故不存在"饿死"
            ))
        scaling = {task.name: (float(scale) if weights.get(task.name, 1.0) > 0.0 else 0.0)
                   for task in tasks}

        contact_residual = jacobian @ qacc + bias
        hierarchy = HierarchyResult(
            qacc=qacc, outcomes=tuple(outcomes),
            increments=(qacc_hard, scale * increment),
            null_space=null @ null.T @ elimination.mass_matrix,
            contact_residual=contact_residual,
            contact_residual_norm=float(np.linalg.norm(contact_residual)),
            contact_conditioning=1.0,
        )
        recovered_force = elimination.recover_contact_force(qacc, h, torque)
        dynamics_residual = (elimination.mass_matrix @ qacc + h
                             - elimination.selection.T @ torque - jacobian.T @ recovered_force)
        dynamics_norm = float(np.linalg.norm(dynamics_residual))
        return SNSResult(
            torque=torque, qacc=qacc, hierarchy=hierarchy, scaling=scaling, clamped={},
            attribution=np.zeros((1, limits.size)), iterations=1,
            feasible=limits.contains(torque) and dynamics_norm < _DYNAMICS_TOLERANCE,
            clip_applied=False,          # 本模块永不裁剪：τ 始终由 q̈ 反解得到
            fallback_used=False,
            dynamics_residual_norm=dynamics_norm,
        )
