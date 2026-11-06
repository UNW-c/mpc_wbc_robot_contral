"""速度输入、积分参考、六状态 MPC 与优化轨迹接口。

默认驾驶只接收 MotionCommand(velocity=..., yaw_rate=...)。
高频 WBC 采样 solution.trajectory；MPC 等效轮力保留诊断用途。
MuJoCo 仅在 ReducedModel.from_mujoco 被显式调用时导入。
"""

from __future__ import annotations

from .contact import (
    BODY_AXIS_PITCH,
    BODY_AXIS_ROLL,
    BODY_AXIS_YAW,
    CONTACT_LATERAL_OFFSET,
    CONTACT_LONGITUDINAL_OFFSET,
    ContactAxis,
    ContactGeometry,
    pitch_controllability,
)
from .controller import ChassisState, MPCSettings, MPCSolution, WheelMPC
from .model import (
    DEFAULT_PITCH_INERTIA,
    INPUT_DIM,
    INPUT_NAMES,
    STATE_DIM,
    STATE_NAMES,
    DiscreteModel,
    ReducedModel,
    wheel_half_track,
    yaw_inertia_from_model,
)
from .planner import (
    MotionCommand,
    MotionLimits,
    PlannerMode,
    ReferenceTrajectory,
    VelocityPlanner,
    ReferenceMode,
    ReferenceFeedback,
)
from .qp import CondensedMPCQP, QPResult, QPWeights, StageBounds
from .trajectory import MotionSample, OptimizedTrajectory

__all__ = [
    "BODY_AXIS_PITCH",
    "BODY_AXIS_ROLL",
    "BODY_AXIS_YAW",
    "CONTACT_LATERAL_OFFSET",
    "CONTACT_LONGITUDINAL_OFFSET",
    "DEFAULT_PITCH_INERTIA",
    "INPUT_DIM",
    "INPUT_NAMES",
    "STATE_DIM",
    "STATE_NAMES",
    "ChassisState",
    "CondensedMPCQP",
    "ContactAxis",
    "ContactGeometry",
    "DiscreteModel",
    "MPCSettings",
    "MPCSolution",
    "MotionCommand",
    "MotionLimits",
    "MotionSample",
    "OptimizedTrajectory",
    "PlannerMode",
    "QPResult",
    "QPWeights",
    "ReducedModel",
    "ReferenceTrajectory",
    "StageBounds",
    "VelocityPlanner",
    "ReferenceMode",
    "ReferenceFeedback",
    "WheelMPC",
    "pitch_controllability",
    "wheel_half_track",
    "yaw_inertia_from_model",
]
