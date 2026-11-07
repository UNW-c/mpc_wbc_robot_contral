"""WBC 任务语义与优先级集合。

activation 缩放本级相对已有解的纠正增量；为零时释放该级权限。
部分激活仍保留严格优先权；该机制不保证接触切换或跨零时力矩连续。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .projection import TaskLevel

# §4.3 的优先级表。数字越小优先级越高；列在这里是为了让各层名称有单一真值来源，
# 避免各处硬编码字符串。
PRIORITY_NAMES = (
    "contact_force_feedforward",  # 0：MPC 的 f* 前馈
    "pitch",  # 1：俯仰 + 角速度阻尼（两轮同向）
    "roll",  # 2：横滚（两髋反向）
    "yaw",  # 3：偏航（两轮反向）
    "height",  # 4：机身高度（两髋同向）
    "forward_velocity",  # 5：前向速度
    "regularization",  # 6：关节正则 + 阻尼 + 远离死点偏置
)


@dataclass(frozen=True)
class Task:
    """一个优先级层级。

    ``activation`` 是该级的纠正增量缩放因子；``damping``
    是该级伪逆的奇异鲁棒阻尼（§5.3 的排程量）；``budget`` 是该级的力矩预算
    ``(下限, 上限)``，不给就用硬件限值。
    """

    name: str
    jacobian: np.ndarray
    desired_acceleration: np.ndarray
    bias_acceleration: np.ndarray | None = None
    activation: float = 1.0
    damping: float = 0.0
    budget: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        activation = float(self.activation)
        if not 0.0 <= activation <= 1.0:
            raise ValueError(f"任务 {self.name} 的激活因子必须在 [0, 1] 内")
        if float(self.damping) < 0.0:
            raise ValueError(f"任务 {self.name} 的阻尼不能为负")
        object.__setattr__(self, "activation", activation)

    @property
    def active(self) -> bool:
        return self.activation > 0.0

    def activated(self) -> TaskLevel:
        """保留原始目标和雅可比，由求解器缩放纠正增量。"""
        return TaskLevel(
            jacobian=self.jacobian,
            desired_acceleration=np.asarray(self.desired_acceleration, dtype=float),
            activation=self.activation,
            bias_acceleration=self.bias_acceleration,
            damping=self.damping,
            name=self.name,
        )

    def with_activation(self, activation: float) -> "Task":
        return replace(self, activation=float(activation))


@dataclass(frozen=True)
class TaskSet:
    """按优先级排好序的任务集合，高优先级在前。

    ``inactive_below`` 用来一次性把某级以下的全部降权（§4.4 的冲击处理：
    "检测到 |f_z| 突变 → 20 ms 内把 L3~L6 权重降到 0.2，只保 L1/L2"）。
    """

    tasks: tuple[Task, ...] = ()

    def __post_init__(self) -> None:
        names = [task.name for task in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError("任务名不能重复，否则缩放因子无法按名字归位")

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def names(self) -> tuple[str, ...]:
        return tuple(task.name for task in self.tasks)

    def level(self, name: str) -> Task:
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(name)

    def levels(self) -> list[TaskLevel]:
        return [task.activated() for task in self.tasks]

    def budgets(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """收集各级的力矩预算，喂给 SNS 的分层限值。"""
        return {task.name: task.budget for task in self.tasks if task.budget is not None}

    def scaled(self, activations: dict[str, float]) -> "TaskSet":
        """按 ``{任务名: 激活因子}`` 生成新的一份（未列出的保持原值）。"""
        return TaskSet(
            tuple(
                task.with_activation(activations[task.name])
                if task.name in activations
                else task
                for task in self.tasks
            )
        )

    def damp_below(self, index: int, factor: float) -> "TaskSet":
        """把第 ``index`` 级及其之后的任务降权 ``factor`` 倍（冲击保护，§4.4）。"""
        if not 0.0 <= float(factor) <= 1.0:
            raise ValueError("降权因子必须在 [0, 1] 内")
        return TaskSet(
            tuple(
                task.with_activation(task.activation * float(factor))
                if order >= index
                else task
                for order, task in enumerate(self.tasks)
            )
        )

    def regularized(self) -> "TaskSet":
        """把"关节正则"那一级的激活因子交给排程表（占位，实际由调用方填）。"""
        return self


@dataclass
class TaskDiagnostics:
    """逐级的解算结果，供上层判断健康度（§5.4 的 s 回馈通道）。"""

    scaling: dict[str, float] = field(default_factory=dict)
    residual: dict[str, float] = field(default_factory=dict)

    @property
    def fully_satisfied(self) -> bool:
        return all(value >= 1.0 - 1e-9 for value in self.scaling.values())
