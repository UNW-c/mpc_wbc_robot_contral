"""接触连线几何（方案 §1.1）：û 方向、倾角 β、各机身轴的可控比例。

这些量是本机器人构型的根，全部有闭式解，所以测试写成"对照文档表格 + 对照闭式"，
不依赖任何实现细节。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Algorithm.mpc import (
    BODY_AXIS_PITCH,
    BODY_AXIS_ROLL,
    BODY_AXIS_YAW,
    CONTACT_LATERAL_OFFSET,
    CONTACT_LONGITUDINAL_OFFSET,
    ContactAxis,
    ContactGeometry,
    pitch_controllability,
)

DEAD_POINT = 0.0
WORKING_POINT = math.radians(25.0)  # 方案 §1.4 建议的标称工作点
MID_STROKE = math.pi / 2.0


def test_geometry_constants_match_the_documented_assembly():
    """0.12 = 两接触点纵向错开，0.24 = 2 × half_track（方案 §1.1 / §1.2）。"""
    assert CONTACT_LONGITUDINAL_OFFSET == pytest.approx(0.12)
    assert CONTACT_LATERAL_OFFSET == pytest.approx(0.24)


def test_dead_point_pitch_is_fully_uncontrollable():
    """q=0 时 û 恰是横向轴：不可控力矩方向就是俯仰，俯仰可控比例为 0。"""
    axis = ContactGeometry().axis(DEAD_POINT)
    np.testing.assert_allclose(axis.direction, [0.0, 1.0, 0.0], atol=1e-15)
    assert axis.tilt == pytest.approx(0.0, abs=1e-15)
    assert axis.pitch_controllability == pytest.approx(0.0, abs=1e-15)
    assert axis.pitch_uncontrollable == pytest.approx(1.0)
    # 死点处横滚与偏航都不受影响
    assert axis.roll_controllability == pytest.approx(1.0)
    assert axis.yaw_controllability == pytest.approx(1.0)


def test_mid_stroke_tilt_is_26_57_degrees():
    """q=π/2 时 û 偏 atan(0.5)=26.57°，俯仰可控比例 sin β=0.4472（方案 §1.1 表）。"""
    axis = ContactGeometry().axis(MID_STROKE)
    assert math.degrees(axis.tilt) == pytest.approx(26.5651, abs=1e-4)
    assert axis.pitch_controllability == pytest.approx(0.4472136, rel=1e-6)
    assert axis.roll_controllability == pytest.approx(np.sqrt(0.8), rel=1e-6)
    np.testing.assert_allclose(axis.direction, [0.12, 0.24, 0.0] / np.hypot(0.12, 0.24))


def test_working_point_matches_the_documented_row():
    """标称工作点 q0=25° 是横滚控制最差点与俯仰权限之间的折中（方案 §1.4）。"""
    axis = ContactGeometry().axis(WORKING_POINT)
    assert math.degrees(axis.tilt) == pytest.approx(11.93, abs=0.01)
    assert axis.pitch_controllability == pytest.approx(0.2067, abs=1e-4)
    assert axis.roll_controllability > 0.97


def test_pitch_controllability_matches_the_closed_form():
    """闭式 ``0.5|sin q| / sqrt(1 + 0.25 sin²q)``。"""
    for hip_angle in np.linspace(0.0, math.pi, 41):
        closed_form = 0.5 * abs(math.sin(hip_angle)) / math.sqrt(
            1.0 + 0.25 * math.sin(hip_angle) ** 2
        )
        assert pitch_controllability(float(hip_angle)) == pytest.approx(closed_form, rel=1e-12)


def test_pitch_controllability_is_maximal_at_mid_stroke():
    """俯仰可控比例在整个行程上不超过 0.4472——地面力永远给不出纯俯仰力矩。"""
    angles = np.linspace(0.0, 2.0 * math.pi, 721)
    values = np.array([pitch_controllability(float(a)) for a in angles])
    assert values.max() == pytest.approx(0.4472136, rel=1e-6)
    assert values.min() == pytest.approx(0.0, abs=1e-15)
    # 半行程对称：sin(q) 的周期是 π
    assert pitch_controllability(MID_STROKE) == pytest.approx(pitch_controllability(MID_STROKE + math.pi))


def test_direction_is_a_unit_vector():
    for hip_angle in np.linspace(-math.pi, math.pi, 33):
        axis = ContactGeometry().axis(float(hip_angle))
        assert np.linalg.norm(axis.direction) == pytest.approx(1.0)


def test_height_difference_tilts_the_axis_towards_vertical():
    """粗糙地形上 Δz ≠ 0 让 û 向 z 倾，不可控方向"进一步偏离横向轴"（方案 §1.1）。

    注意方向：偏离横向轴意味着 β 变大，俯仰的可控比例反而上升；代价是 û 出现 z
    分量，偏航不再完全可控（平地上是 1）。
    """
    flat = ContactGeometry().axis(WORKING_POINT)
    rough = ContactGeometry().axis(WORKING_POINT, height_difference=0.05)
    assert rough.direction[2] > flat.direction[2]
    assert rough.tilt > flat.tilt
    assert rough.pitch_controllability > flat.pitch_controllability
    assert rough.yaw_controllability < 1.0
    assert flat.yaw_controllability == pytest.approx(1.0)


def test_roll_and_pitch_controllability_sum_of_squares():
    """û 是单位向量：三个机身轴的可控比例平方和恒为 3 - 1 = 2。"""
    for hip_angle in np.linspace(0.0, math.pi, 25):
        axis = ContactGeometry().axis(float(hip_angle))
        total = (
            axis.roll_controllability**2
            + axis.pitch_controllability**2
            + axis.yaw_controllability**2
        )
        assert total == pytest.approx(2.0)


def test_axis_rejects_a_degenerate_direction():
    with pytest.raises(ValueError, match="非零范数"):
        ContactAxis(direction=np.zeros(3))
    with pytest.raises(ValueError, match="有限值"):
        ContactGeometry().axis(float("nan"))


def test_controllability_rejects_a_bad_axis_index():
    axis = ContactGeometry().axis(WORKING_POINT)
    with pytest.raises(IndexError):
        axis.controllability(5)
    # 三个具名轴都能正常取到
    for name in (BODY_AXIS_ROLL, BODY_AXIS_PITCH, BODY_AXIS_YAW):
        assert 0.0 <= axis.controllability(name) <= 1.0
