"""单级 QP 全身控制：决策变量 ``[qacc, f, tau]``。

与零空间法（``NullSpaceHierarchy``）的区别：

- 零空间法把任务当硬等式逐级投影，执行器数少于任务维数时低优先级任务被
  耗尽（``scaling`` 恒为 0），且力矩是反解出来的副产品，链条里没有任何一项
  在"要求出力矩"。
- 本模块把 ``qacc`` / 接触力 ``f`` / 关节力矩 ``tau`` 全部作为决策变量，
  任务只进入**代价项**（软目标，按权重竞争，不会被清零）。"期望状态"是
  "尽量达到"而不是"必须满足"，不会因为某一条硬约束把其他状态挤掉。

硬约束（四类）：

1. 浮动基座动力学等式 ``M qacc + h = S' tau + Jc' f``；
2. 接触一致性等式 ``Jc qacc + contact_bias = 0``——接触力不是自由变量，
   没有这条约束时 QP 可以随意组合 ``f`` 与 ``tau`` 来"满足"任务，解在
   真实物理中不可实现（任务与决策变量的对映错误）；
3. 执行器力矩上下限；
4. 接触力法向非负（不能拉地面）。

摩擦锥约束暂不加入（2026-09-22 用户决定：先删除降低复杂度，效果不好再加回）。

不可行兜底（两级，错误必须记日志，不允许静默失败）：

1. 完整约束 QP 不可行 → 去掉接触一致性等式再解（``f`` 回归自由变量）；
2. 仍不可行 → 输出零力矩与无控制加速度，``fallback_used=True``。
"""
from __future__ import annotations

import logging

import numpy as np
import osqp
import scipy.sparse as sparse

from .projection import ContactElimination, HierarchyResult, TaskLevel, TaskOutcome
from .sns import ActuatorLimits, SNSResult

_LOGGER = logging.getLogger(__name__)

_DEFAULT_SETTINGS = {
    "verbose": False,
    "eps_abs": 1e-7,
    "eps_rel": 1e-7,
    "max_iter": 20000,
    "polish": True,
    "warm_starting": True,
}

_SOLVED = 1  # osqp status_val 的求解成功码


class TaskQpResolver:
    """与 :class:`Algorithm.wbc.sns.SaturationResolver` 同接口的 QP 版本。

    ``task_weights`` 是任务名到权重的映射；未列出的任务取 1.0。
    权重越大越优先，但任何任务都不会被清零。
    """

    def __init__(self, elimination: ContactElimination, task_weights: dict[str, float],
                 torque_weight: float = 1.0e-3, force_weight: float = 1.0e-3,
                 normal_indices: tuple[int, ...] = (1, 3),
                 settings: dict | None = None) -> None:
        self.elimination = elimination
        self.weights = {str(name): float(value) for name, value in task_weights.items()}
        self.torque_weight = float(torque_weight)
        self.force_weight = float(force_weight)
        # 接触力排列为 [切向, 法向, 切向, 法向, 共模]（见 APP/mujoco_mpc_wbc.py 的法向判定）。
        self.normal_indices = tuple(int(i) for i in normal_indices)
        self.settings = dict(_DEFAULT_SETTINGS)
        if settings:
            self.settings.update(settings)
        # 日志限流：求解器在 250 Hz 循环里每拍调用，只在 fallback 状态跳变时输出。
        self._fallback_state = "ok"

    # ---------------------------------------------------------------- helpers
    def _cost(self, tasks: list[TaskLevel], dof: int, na: int, nc: int
              ) -> tuple[sparse.csc_matrix, np.ndarray]:
        """组装 ``0.5 z' P z + q' z``，其中 ``z = [qacc, f, tau]``。

        每个任务的残差是 ``J qacc + bias - desired``（``J qacc = desired - bias``）。
        """
        size = dof + nc + na
        hessian = sparse.csc_matrix((size, size))
        gradient = np.zeros(size)

        def accumulate(weight: float, block: np.ndarray, target: np.ndarray) -> None:
            nonlocal hessian, gradient
            matrix = sparse.csc_matrix(block)
            hessian = hessian + (matrix.T @ matrix) * weight
            gradient = gradient - (matrix.T @ target) * weight

        for task in tasks:
            weight = self.weights.get(task.name, 1.0) * float(task.activation)
            if weight <= 0.0:
                continue
            jacobian = np.atleast_2d(np.asarray(task.jacobian, dtype=float))
            target = (np.asarray(task.desired_acceleration, dtype=float)
                      - np.asarray(task.bias_acceleration, dtype=float))
            block = np.zeros((jacobian.shape[0], size))
            block[:, :dof] = jacobian
            accumulate(weight, block, target)

        if self.force_weight > 0.0:
            block = np.zeros((nc, size))
            block[:, dof:dof + nc] = np.eye(nc)
            accumulate(self.force_weight, block, np.zeros(nc))

        if self.torque_weight > 0.0:
            block = np.zeros((na, size))
            block[:, dof + nc:] = np.eye(na)
            accumulate(self.torque_weight, block, np.zeros(na))

        return hessian + 1.0e-9 * sparse.eye(size), gradient

    def _normal_block(self, dof: int, nc: int, na: int) -> np.ndarray | None:
        """法向非负 ``-f_n <= 0``，写成 ``A_norm z <= 0``。

        每个法向下标一行；切向与共模分量不受此约束。
        """
        rows: list[np.ndarray] = []
        for normal in self.normal_indices:
            if normal >= nc:
                continue
            row = np.zeros(nc)
            row[normal] = -1.0
            rows.append(row)
        if not rows:
            return None
        block = np.asarray(rows)
        return np.hstack([np.zeros((block.shape[0], dof)), block,
                          np.zeros((block.shape[0], na))])

    def _constraints(self, dynamics, contact, torque_upper, torque_lower,
                     normal_block, h: np.ndarray, bias: np.ndarray,
                     limits: ActuatorLimits, na: int) -> tuple:
        """把参与本次求解的约束块装配成 ``l <= A z <= u``。"""
        blocks = [dynamics]
        lower_parts = [-h]
        upper_parts = [-h]
        if contact is not None:
            blocks.append(contact)
            lower_parts.append(-bias)
            upper_parts.append(-bias)
        blocks.append(torque_upper)
        lower_parts.append(np.full(na, -np.inf))
        upper_parts.append(limits.upper)
        blocks.append(torque_lower)
        lower_parts.append(-limits.upper)
        upper_parts.append(-limits.lower)
        if normal_block is not None:
            blocks.append(normal_block)
            lower_parts.append(np.full(normal_block.shape[0], -np.inf))
            upper_parts.append(np.zeros(normal_block.shape[0]))
        return (sparse.vstack(blocks, format="csc"),
                np.concatenate(lower_parts), np.concatenate(upper_parts))

    @staticmethod
    def _attempt(hessian, gradient, condition, lower, upper,
                 settings: dict) -> tuple[int, np.ndarray, int]:
        """求解一次 QP，返回 ``(status_val, z, iterations)``。"""
        solver = osqp.OSQP()
        solver.setup(P=hessian, q=gradient, A=condition, l=lower, u=upper, **settings)
        solution = solver.solve()
        status = int(solution.info.status_val)
        return status, np.asarray(solution.x, dtype=float), int(solution.info.iter)

    def _log_state(self, state: str, level: int, message: str, *args) -> None:
        """只在 fallback 状态跳变时输出，避免高频循环刷屏。"""
        if self._fallback_state == state:
            return
        self._fallback_state = state
        _LOGGER.log(level, message, *args)

    def _outcome(self, task: TaskLevel, qacc: np.ndarray) -> TaskOutcome:
        achieved = task.jacobian @ qacc + task.bias_acceleration
        residual = achieved - task.desired_acceleration
        return TaskOutcome(task.name, task.desired_acceleration.copy(), achieved, residual,
                           float(np.linalg.norm(residual)), 1.0,
                           activated=task.activation, exhausted=False)

    def _result(self, tasks, mass, selection, jacobian, bias, h,
                qacc, force, torque, limits, iterations, feasible, fallback_used) -> SNSResult:
        residual = mass @ qacc + h - selection.T @ torque - jacobian.T @ force
        dynamics_norm = float(np.linalg.norm(residual))
        # feasible 判据 1e-5：OSQP（eps=1e-7）在条件数恶化工况（侧向冲击、单轮卸载）
        # 下解的残差可达 1e-6 量级，1e-7 判据会误报失败并停机；1e-5 对应力矩量级的
        # 相对误差约 1e-6，物理上可忽略。
        contact_residual = jacobian @ qacc + bias
        null = np.eye(mass.shape[0]) - np.linalg.pinv(jacobian) @ jacobian
        hierarchy = HierarchyResult(
            qacc=qacc,
            outcomes=tuple(self._outcome(task, qacc) for task in tasks),
            increments=(),  # QP 没有逐级增量语义
            null_space=null @ null.T @ mass,
            contact_residual=contact_residual,
            contact_residual_norm=float(np.linalg.norm(contact_residual)),
            contact_conditioning=1.0,
        )
        scaling = {
            str(task.name): (1.0 if self.weights.get(task.name, 1.0) * float(task.activation) > 0.0
                             else 0.0)
            for task in tasks
        }
        return SNSResult(
            torque=torque, qacc=qacc, hierarchy=hierarchy, scaling=scaling, clamped={},
            attribution=np.zeros((1, limits.size)), iterations=iterations,
            feasible=bool(feasible and dynamics_norm < 1e-5), clip_applied=not limits.contains(torque),
            fallback_used=fallback_used, dynamics_residual_norm=dynamics_norm,
        )

    # ------------------------------------------------------------------ solve
    def resolve(self, contact_jacobian: np.ndarray, contact_bias: np.ndarray | None,
                tasks: tuple[TaskLevel, ...] | list[TaskLevel],
                limits: ActuatorLimits,
                budgets: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                dynamics_bias: np.ndarray | None = None,
                external_force: np.ndarray | None = None) -> SNSResult:
        del budgets  # QP 没有逐级预算语义，保持接口一致
        tasks = list(tasks)
        if not tasks:
            raise ValueError("至少要有一个任务层级")
        if len({task.name for task in tasks}) != len(tasks):
            raise ValueError("任务名不能重复")
        e = self.elimination
        mass = e.mass_matrix
        dof = mass.shape[0]
        selection = e.selection
        na = selection.shape[0]
        jacobian = np.atleast_2d(np.asarray(contact_jacobian, dtype=float))
        nc = jacobian.shape[0]
        bias = (np.zeros(nc) if contact_bias is None
                else np.asarray(contact_bias, dtype=float).reshape(-1))
        if bias.size != nc:
            raise ValueError("接触偏置长度必须等于接触雅可比的行数")
        h = (np.zeros(dof) if dynamics_bias is None
             else np.asarray(dynamics_bias, dtype=float).reshape(-1))
        if h.size != dof:
            raise ValueError("动力学偏置长度必须等于 nv")
        if external_force is not None:
            h = h - np.asarray(external_force, dtype=float).reshape(-1)

        hessian, gradient = self._cost(tasks, dof, na, nc)

        dynamics = sparse.hstack([sparse.csc_matrix(mass),
                                  sparse.csc_matrix(-jacobian.T),
                                  sparse.csc_matrix(-selection.T)], format="csc")
        contact = sparse.hstack([sparse.csc_matrix(jacobian),
                                 sparse.csc_matrix((nc, nc + na))], format="csc")
        torque_upper = sparse.hstack([sparse.csc_matrix((na, dof + nc)),
                                      sparse.eye(na)], format="csc")
        torque_lower = sparse.hstack([sparse.csc_matrix((na, dof + nc)),
                                      -sparse.eye(na)], format="csc")
        normal_block = self._normal_block(dof, nc, na)

        # 第一级：完整约束（动力学 + 接触一致性 + 力矩限 + 法向非负）。
        condition, lower, upper = self._constraints(
            dynamics, contact, torque_upper, torque_lower, normal_block, h, bias, limits, na)
        status, z, iterations = self._attempt(hessian, gradient, condition, lower, upper,
                                              self.settings)
        fallback_used = False
        if status != _SOLVED:
            self._log_state("fallback", logging.WARNING,
                            "一级 QP 不可行（status=%d），去掉接触等式降级重解", status)
            # 第二级：去掉接触一致性等式，f 回归自由变量。
            condition, lower, upper = self._constraints(
                dynamics, None, torque_upper, torque_lower, normal_block, h, bias, limits, na)
            status, z, iterations = self._attempt(hessian, gradient, condition, lower, upper,
                                                  self.settings)
            fallback_used = True
            if status != _SOLVED:
                self._log_state("dead", logging.ERROR,
                                "二级 QP 仍不可行（status=%d），输出零力矩与无控制加速度", status)
                qacc = np.linalg.solve(mass, -h)
                return self._result(tasks, mass, selection, jacobian, bias, h, qacc,
                                    np.zeros(nc), np.zeros(na), limits, iterations,
                                    feasible=False, fallback_used=True)
        else:
            self._log_state("ok", logging.INFO, "QP 恢复完整约束求解（status=%d）", status)

        qacc = z[:dof]
        force = z[dof:dof + nc]
        torque = z[dof + nc:]
        return self._result(tasks, mass, selection, jacobian, bias, h,
                            qacc, force, torque, limits, iterations,
                            feasible=True, fallback_used=fallback_used)


def task_qp_resolver_factory(task_weights: dict[str, float],
                             torque_weight: float = 1.0e-3,
                             force_weight: float = 1.0e-3,
                             normal_indices: tuple[int, ...] = (1, 3),
                             settings: dict | None = None):
    """返回 ``(mass_matrix, elimination) -> TaskQpResolver`` 工厂，供运行时注入。

    ``mass_matrix`` 参数保留以对齐注入接口；消去器里已携带质量矩阵。
    """
    def factory(mass_matrix: np.ndarray, elimination: ContactElimination) -> TaskQpResolver:
        del mass_matrix
        return TaskQpResolver(elimination, task_weights=task_weights,
                              torque_weight=torque_weight, force_weight=force_weight,
                              normal_indices=normal_indices, settings=settings)
    return factory
