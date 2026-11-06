"""下层零空间 WBC（方案 §4 / §5）。

本包是**算法核心**，只吃矩阵不吃仿真：输入是浮动基约束动力学的各个块
（``M, h, S, Jc, J̇c q̇, q̇``）与任务表，输出是关节力矩、接触力与健康指标。
MuJoCo 适配（怎么取这些矩阵、``J̇c q̇`` 用 ``mj_objectAcceleration`` 还是差分）
留在接入时再写，所以核心算法可以用解析构造的模型做单元测试。

模块分工：

* :mod:`Algorithm.wbc.projection` —— 正交分解消力、动力学一致伪逆、递归零空间（§4.1 / §4.2）
* :mod:`Algorithm.wbc.sns`        —— 饱和钳位、任务缩放、分层限值（§5.1）
* :mod:`Algorithm.wbc.schedule`   —— §5.3 的统一增益排程表（单一真值来源）
* :mod:`Algorithm.wbc.tasks`      —— 任务语义与优先级集合（§4.3 / §4.4）

尚未实现（等接入时再补）：把 §4.3 的 L0~L6 具体拼出来的装配层、慢环/快环的频率
分离（§2.2）、以及 SNS 缩放因子的 ``s`` 回馈通道（§5.4）。
"""

from __future__ import annotations

from .projection import (
    ContactElimination,
    HierarchyResult,
    NullSpaceHierarchy,
    TaskLevel,
    TaskOutcome,
    dynamically_consistent_inverse,
)
from .schedule import (
    ScheduleParameters,
    dead_point_factor,
    decoupling_angle,
    decoupling_damping,
    height_task_activation,
    height_task_activation_from_factor,
    height_task_damping,
    roll_gain_scale,
    roll_gain_scale_from_factor,
    scrub_moment,
    smoothstep,
    yaw_moment_budget,
    yaw_rate_feedforward,
)
from .sns import ActuatorLimits, SNSResult, SaturationResolver, maximum_scale
from .tasks import PRIORITY_NAMES, Task, TaskDiagnostics, TaskSet

__all__ = [
    "PRIORITY_NAMES",
    "ActuatorLimits",
    "ContactElimination",
    "HierarchyResult",
    "NullSpaceHierarchy",
    "SNSResult",
    "SaturationResolver",
    "ScheduleParameters",
    "Task",
    "TaskDiagnostics",
    "TaskLevel",
    "TaskOutcome",
    "TaskSet",
    "dead_point_factor",
    "decoupling_angle",
    "decoupling_damping",
    "dynamically_consistent_inverse",
    "height_task_activation",
    "height_task_damping",
    "maximum_scale",
    "roll_gain_scale",
    "scrub_moment",
    "smoothstep",
    "yaw_moment_budget",
    "yaw_rate_feedforward",
]
