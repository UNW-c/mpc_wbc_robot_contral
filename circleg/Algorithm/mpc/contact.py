"""两接触点连线的几何：``u`` 方向、与横向轴的夹角、各机身轴的投影（方案 §1.1）。

地面接触力只作用在两个接触点上，接触力到机身力扳手的映射秩为 5：沿两接触点连线
方向的等值反向力对既无合力也无合力矩，所以"绕 ``u`` 的力矩"这一维恒不可控。记
``û`` 为接触连线方向、``β`` 为它与横向轴 ``ŷ`` 的夹角，则

* 可控力矩子空间是 ``{m : m ⊥ û}``，维数 2；
* 俯仰轴 ``ŷ`` 落在其中的投影长度 = ``sin β``（``pitch_controllability``）；
* 平地上 ``û = normalize(0.12·sin q, 0.24, 0)``，故 ``β = atan(0.5·sin q)``。

| 髋角 q | β | sin β |
|---|---|---|
| 0（死点） | 0 | 0（俯仰完全不可控） |
| 25°（§1.4 建议工作点） | 11.9° | 0.207 |
| 90° | 26.57° | 0.447 |

**本模块只计算几何量，不参与权重排程。** 实测线性化是在髋关节自由的前提下测的，
俯仰权限主要来自髋反作用而不是地面力，不受上面这条秩 5 限制；若按 ``sin β`` 去压
俯仰跟踪权重，会把这个机器人最重要的那个信号（``θ*(t)``）削掉。当前用途是监控与
下层对接：把每个髋角下的可控比例交给上层，用于判断俯仰通道是否可信、以及在完整
SRBD（含横滚与接触点 ``r_i``）落地时把权重真正写到这个基底里。

本模块只依赖 numpy，不依赖 MuJoCo。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 两接触点的纵向错开量（全跨度），即 2 × 0.06 m（方案 §1.2）
CONTACT_LONGITUDINAL_OFFSET = 0.12
# 两接触点的横向间距，即 2 × half_track = 2 × 0.12 m
CONTACT_LATERAL_OFFSET = 0.24

BODY_AXIS_PITCH = 1
BODY_AXIS_ROLL = 0
BODY_AXIS_YAW = 2


@dataclass(frozen=True)
class ContactAxis:
    """某一髋角下的接触连线几何。``direction`` 是单位向量 ``û``。"""

    direction: np.ndarray
    hip_angle: float = 0.0
    height_difference: float = 0.0

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction, dtype=float).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("接触连线方向必须有非零范数")
        object.__setattr__(self, "direction", direction / norm)

    @property
    def tilt(self) -> float:
        """``β``：接触连线与横向轴 ``ŷ`` 的夹角 (rad)。"""
        return math.acos(float(np.clip(abs(self.direction[BODY_AXIS_PITCH]), -1.0, 1.0)))

    def controllability(self, axis: int) -> float:
        """机身某轴在可控子空间 ``⊥û`` 里的投影长度。

        ``axis`` 取 ``BODY_AXIS_PITCH`` / ``ROLL`` / ``YAW``。返回值 1 表示该轴完全
        可控，0 表示该轴的力矩方向恰好就是不可控的 ``û``。
        """
        component = float(self.direction[int(axis)])
        return math.sqrt(max(1.0 - component * component, 0.0))

    @property
    def pitch_controllability(self) -> float:
        """``sin β``：俯仰轴落在可控子空间里的比例（方案 §1.1）。"""
        return self.controllability(BODY_AXIS_PITCH)

    @property
    def pitch_uncontrollable(self) -> float:
        """``cos β``：俯仰轴在不可控方向 ``û`` 上的比例。"""
        return abs(float(self.direction[BODY_AXIS_PITCH]))

    @property
    def roll_controllability(self) -> float:
        return self.controllability(BODY_AXIS_ROLL)

    @property
    def yaw_controllability(self) -> float:
        """平地上 ``û`` 没有 z 分量，偏航恒为 1。"""
        return self.controllability(BODY_AXIS_YAW)


@dataclass(frozen=True)
class ContactGeometry:
    """接触几何常数。默认值取自已核对的 URDF/XML（方案 §1.1）。"""

    longitudinal_offset: float = CONTACT_LONGITUDINAL_OFFSET
    lateral_offset: float = CONTACT_LATERAL_OFFSET

    def axis(self, hip_angle: float, height_difference: float = 0.0) -> ContactAxis:
        """由差模髋角给出接触连线几何。

        ``hip_angle`` 是**差模髋角** ``q_diff = 0.5*(q1 - sign*q3)``：两轮前后错开
        ``0.12*sin(q_diff)`` 才使接触连线相对横向轴倾斜；两腿同相位（共模）摆动不产生错开（#13）。
        ``height_difference``
        是两接触点的高度差 ``Δz``，平地为 0，粗糙地形上不为 0 时 ``û`` 还会向 z 倾，
        不可控方向进一步偏离横向。
        """
        hip_angle = float(hip_angle)
        x_component = self.longitudinal_offset * math.sin(hip_angle)
        direction = np.array(
            [x_component, self.lateral_offset, float(height_difference)],
            dtype=float,
        )
        if not np.all(np.isfinite(direction)):
            raise ValueError("接触几何输入必须是有限值")
        return ContactAxis(
            direction=direction,
            hip_angle=hip_angle,
            height_difference=float(height_difference),
        )


def pitch_controllability(hip_angle: float, height_difference: float = 0.0) -> float:
    """平地上俯仰可控比例的闭式：``0.5|sin q| / sqrt(1 + 0.25 sin²q)``。"""
    return ContactGeometry().axis(hip_angle, height_difference).pitch_controllability
