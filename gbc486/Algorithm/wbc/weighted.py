"""加权整体最小二乘 WBC：所有任务在硬约束零空间里按权重同时分配。

对照用途：现有 ``SaturationResolver`` 用严格优先级，执行器数少于任务维数时
低优先级任务会被清零（零空间耗尽）。本模块把全部任务合并成一个加权最小二乘，
低优先级任务永远分到份额，代价是优先级变软。

权重是新的可调量，当前无实测标定依据。
"""
from __future__ import annotations

import numpy as np

from .projection import ContactElimination, HierarchyResult, TaskLevel
from .sns import ActuatorLimits, SNSResult

_ZERO_TOLERANCE = 1e-9


def _hard_system(elimination: ContactElimination, jacobian: np.ndarray,
                 bias: np.ndarray, dynamics_bias: np.ndarray):
    """接触等式 + 欠驱动动力学等式，合成本级的硬约束 ``A qacc = b``。"""
    hard = elimination.dynamics_constraint(dynamics_bias)
    if hard is None:
        return jacobian, -bias
    rows = np.vstack((jacobian, hard.jacobian))
    target = np.concatenate((-bias, hard.desired_acceleration - hard.bias_acceleration))
    return rows, target


class WeightedSaturationResolver:
    """与 :class:`Algorithm.wbc.sns.SaturationResolver` 同接口的加权最小二乘版本。

    ``weights`` 是本级权重；未列出的任务取 1.0。权重越大越优先，但不会把别人
    清零。``damping`` 是零空间内的正则项，避免任务行相关时病态。
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
        del budgets  # 加权版不复用分层预算
        tasks = list(tasks)
        if not tasks:
            raise ValueError("至少要有一个任务层级")
        e = self.elimination
        dof = e.mass_matrix.shape[0]
        jacobian = np.atleast_2d(np.asarray(contact_jacobian, dtype=float))
        bias = (np.zeros(jacobian.shape[0]) if contact_bias is None
                else np.asarray(contact_bias, dtype=float).reshape(-1))
        h = np.zeros(dof) if dynamics_bias is None else np.asarray(dynamics_bias, dtype=float).reshape(-1)
        if external_force is not None:
            disturbance = np.asarray(external_force, dtype=float).reshape(-1)
            if disturbance.shape != (dof,):
                raise ValueError("external_force 必须是 nv 维")
            h = h - disturbance

        rows, target = _hard_system(e, jacobian, bias, h)
        pseudo = np.linalg.pinv(rows, rcond=1e-12)
        qacc = pseudo @ target
        null = np.eye(dof) - pseudo @ rows

        hessian = self.damping * np.eye(dof)
        gradient = np.zeros(dof)
        scaling: dict[str, float] = {}
        for task in tasks:
            weight = self.weights.get(task.name, 1.0) * float(task.activation)
            scaling[task.name] = 1.0 if weight > 0.0 else 0.0
            if weight <= 0.0:
                continue
            task_jacobian = np.atleast_2d(task.jacobian)
            desired = np.asarray(task.desired_acceleration, dtype=float)
            if task.bias_acceleration is not None:
                desired = desired - np.asarray(task.bias_acceleration, dtype=float)
            projected = task_jacobian @ null
            hessian += weight * (projected.T @ projected)
            gradient += weight * (projected.T @ (desired - task_jacobian @ qacc))
        qacc = qacc + null @ np.linalg.solve(hessian, gradient)

        torque = e.recover_torque(qacc, h)
        clip_applied = not limits.contains(torque)
        if clip_applied:
            torque = limits.clamp(torque)
        contact_residual = jacobian @ qacc + bias
        hierarchy = HierarchyResult(
            qacc=qacc, outcomes=(), increments=(), null_space=null @ null.T @ e.mass_matrix,
            contact_residual=contact_residual,
            contact_residual_norm=float(np.linalg.norm(contact_residual)),
            contact_conditioning=1.0,
        )
        residual = (e.mass_matrix @ qacc + h - e.selection.T @ torque
                    - jacobian.T @ e.recover_contact_force(qacc, h, torque))
        dynamics_norm = float(np.linalg.norm(residual))
        return SNSResult(
            torque=torque, qacc=qacc, hierarchy=hierarchy, scaling=scaling, clamped={},
            attribution=np.zeros((1, limits.size)), iterations=1,
            feasible=limits.contains(torque) and dynamics_norm < 1e-7,
            clip_applied=clip_applied, fallback_used=False,
            dynamics_residual_norm=dynamics_norm,
        )
