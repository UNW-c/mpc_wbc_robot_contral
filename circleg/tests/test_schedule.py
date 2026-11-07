"""方案 §5.3 的统一增益排程表。

排程表的每一条都能对着文档的公式逐项核对，所以这里写成"对照文档数值 + 单调性
+ 端点行为"，而不是只测有没有返回值。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Algorithm.wbc.schedule import (
    DEFAULT_SCRUB_MOMENT_COEFFICIENT,
    DEFAULT_YAW_MOMENT_FLOOR,
    DEFAULT_YAW_MOMENT_LIMIT,
    ScheduleParameters,
    dead_point_factor,
    decoupling_angle,
    decoupling_damping,
    height_task_activation,
    height_task_damping,
    roll_gain_scale,
    scrub_moment,
    smoothstep,
    yaw_moment_budget,
    yaw_rate_feedforward,
)

MID_STROKE = math.pi / 2.0
WORKING_POINT = math.radians(25.0)


# ---- smoothstep ----------------------------------------------------------


def test_smoothstep_is_clamped_and_c1_at_the_edges():
    assert smoothstep(-1.0, 0.0, 1.0) == pytest.approx(0.0)
    assert smoothstep(0.0, 0.0, 1.0) == pytest.approx(0.0)
    assert smoothstep(1.0, 0.0, 1.0) == pytest.approx(1.0)
    assert smoothstep(2.0, 0.0, 1.0) == pytest.approx(1.0)
    assert smoothstep(0.5, 0.0, 1.0) == pytest.approx(0.5)
    # 导数在端点为零：正是"淡入淡出"要的性质
    step = 1e-6
    assert abs(smoothstep(step, 0.0, 1.0) - smoothstep(0.0, 0.0, 1.0)) < 1e-9
    assert abs(smoothstep(1.0 - step, 0.0, 1.0) - smoothstep(1.0, 0.0, 1.0)) < 1e-9


def test_smoothstep_is_monotone():
    values = [smoothstep(x, 0.2, 0.8) for x in np.linspace(0.0, 1.0, 41)]
    assert all(later >= earlier for earlier, later in zip(values, values[1:]))


def test_smoothstep_rejects_a_degenerate_window():
    with pytest.raises(ValueError, match="上边界"):
        smoothstep(0.5, 1.0, 1.0)


# ---- 死点因子 ------------------------------------------------------------


def test_dead_point_factor_vanishes_at_both_dead_points():
    assert dead_point_factor(0.0) == pytest.approx(0.0, abs=1e-15)
    assert dead_point_factor(math.pi) == pytest.approx(0.0, abs=1e-15)
    assert dead_point_factor(MID_STROKE) == pytest.approx(1.0)
    # 半行程对称
    assert dead_point_factor(WORKING_POINT) == pytest.approx(dead_point_factor(math.pi - WORKING_POINT))


# ---- 高度任务：阻尼 ∝ 1/(p²+ε)，激活 ∝ smoothstep·p²  -------------------


def test_height_damping_matches_the_documented_schedule():
    """``1/(p² + ε)``，ε = 0.01（§1.4 / §5.3）。"""
    epsilon = ScheduleParameters().height_damping_epsilon
    assert height_task_damping(MID_STROKE) == pytest.approx(1.0 / (1.0 + epsilon))
    # 死点处权限为零 → 阻尼拉满（约 100 倍）
    assert height_task_damping(0.0) == pytest.approx(1.0 / epsilon)
    # 死点与中行程的比值是 100·(1+ε)，约 101 倍
    assert height_task_damping(0.0) / height_task_damping(MID_STROKE) == pytest.approx(
        100.0 * (1.0 + epsilon), rel=1e-9
    )
    # 单调：越靠近死点阻尼越大
    angles = np.linspace(0.0, MID_STROKE, 21)
    values = [height_task_damping(float(a)) for a in angles]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))


def test_height_damping_is_periodic_over_the_full_stroke():
    for angle in (0.3, 0.7, 1.2):
        assert height_task_damping(angle) == pytest.approx(height_task_damping(angle + math.pi))


def test_height_activation_fades_in_away_from_the_dead_point():
    """``[3°, 8°]`` 上淡入，之后再乘 p²（§4.4）。"""
    params = ScheduleParameters()
    assert height_task_activation(0.0) == pytest.approx(0.0)
    # 淡入窗口内严格介于 0 和 1
    inside = math.radians(5.5)
    value = height_task_activation(inside)
    assert 0.0 < value < math.sin(inside) ** 2
    # 窗口之外就只剩 p² 这一项
    outside = math.radians(20.0)
    assert height_task_activation(outside) == pytest.approx(math.sin(outside) ** 2)
    # 死区以下被 smoothstep 压到 0（不是直接跳变）
    assert height_task_activation(math.radians(1.0)) == pytest.approx(0.0)
    del params


def test_height_activation_is_monotone_on_the_rising_half():
    angles = np.linspace(0.0, MID_STROKE, 41)
    values = [height_task_activation(float(a)) for a in angles]
    assert all(later >= earlier for earlier, later in zip(values, values[1:]))


# ---- 横滚回路增益 ∝ p ----------------------------------------------------


def test_roll_gain_follows_sine_of_the_hip_angle():
    """髋→法向力增益 ∝ 1/p，所以横滚回路增益要反着按 p 排程（§1.4）。"""
    assert roll_gain_scale(0.0) == pytest.approx(0.0)
    assert roll_gain_scale(MID_STROKE) == pytest.approx(1.0)
    assert roll_gain_scale(WORKING_POINT) == pytest.approx(math.sin(WORKING_POINT))
    # 相比死点，工作点的增益是 0.42 倍（而不是无穷倍）
    assert roll_gain_scale(WORKING_POINT) == pytest.approx(0.4226, abs=1e-4)


# ---- 偏航：scrub 折扣与前馈（§1.2 / §1.3） -------------------------------


def test_scrub_moment_matches_the_documented_number():
    """``5.59·sin q``：中行程 5.59 N·m，死点为 0（§1.2）。"""
    assert scrub_moment(MID_STROKE) == pytest.approx(DEFAULT_SCRUB_MOMENT_COEFFICIENT)
    assert scrub_moment(0.0) == pytest.approx(0.0)
    assert scrub_moment(math.pi) == pytest.approx(0.0, abs=1e-15)


def test_yaw_budget_matches_the_documented_numbers():
    """满值 7.5 N·m；中行程 scrub 吃掉 74.5%，只剩 1.91 N·m（§1.2）。"""
    flat = yaw_moment_budget(0.0)
    mid = yaw_moment_budget(MID_STROKE)
    assert flat == pytest.approx(DEFAULT_YAW_MOMENT_LIMIT)
    assert mid == pytest.approx(DEFAULT_YAW_MOMENT_LIMIT - DEFAULT_SCRUB_MOMENT_COEFFICIENT)
    assert mid == pytest.approx(1.91, abs=0.01)
    # 吃掉的比例正好是文档里的 74.5%
    assert (flat - mid) / flat == pytest.approx(0.745, abs=0.002)


def test_yaw_budget_has_a_floor():
    parameters = ScheduleParameters(scrub_moment_coefficient=1.0e3)
    assert yaw_moment_budget(MID_STROKE, parameters) == pytest.approx(DEFAULT_YAW_MOMENT_FLOOR)


def test_yaw_rate_feedforward_is_the_parasitic_yaw():
    """``0.5·cos q·q̇``：死点附近 q̇=1 rad/s 给 0.5 rad/s（§1.3）。"""
    assert yaw_rate_feedforward(0.0, 1.0) == pytest.approx(0.5)
    assert yaw_rate_feedforward(MID_STROKE, 1.0) == pytest.approx(0.0, abs=1e-15)
    assert yaw_rate_feedforward(0.0, -2.0) == pytest.approx(-1.0)
    # 半行程升降的净偏航 = ∫0.5cos q dq = 0.5 rad = 28.6°
    angles = np.linspace(0.0, MID_STROKE, 20001)
    integral = np.trapezoid(
        [yaw_rate_feedforward(float(a), 1.0) for a in angles], angles
    )
    assert integral == pytest.approx(0.5, rel=1e-6)
    assert math.degrees(integral) == pytest.approx(28.6, abs=0.1)


# ---- 俯仰/横滚解耦 -------------------------------------------------------


def test_decoupling_angle_matches_the_documented_geometry():
    """``β = atan(0.5·sin q)``，与 Algorithm.mpc.contact 的 tilt 是同一个量。"""
    assert decoupling_angle(0.0) == pytest.approx(0.0)
    assert decoupling_angle(MID_STROKE) == pytest.approx(math.atan(0.5))
    assert math.degrees(decoupling_angle(MID_STROKE)) == pytest.approx(26.565, abs=1e-3)
    assert decoupling_angle(WORKING_POINT) == pytest.approx(math.atan(0.5 * math.sin(WORKING_POINT)))


def test_decoupling_angle_agrees_with_the_mpc_contact_geometry():
    """两处各算一套就会漂移，这里直接把两个模块对上（§1.1 的同一个 β）。"""
    from Algorithm.mpc.contact import ContactGeometry

    geometry = ContactGeometry()
    for angle in np.linspace(0.0, math.pi, 25):
        # MPC 侧的 tilt 是 û 与横向轴的夹角；平地上两者必须一致
        assert decoupling_angle(float(angle)) == pytest.approx(
            geometry.axis(float(angle)).tilt, abs=1e-12
        )


def test_decoupling_damping_scales_with_the_coupling_strength():
    nominal = 2.0
    assert decoupling_damping(0.0, nominal) == pytest.approx(0.0)
    assert decoupling_damping(MID_STROKE, nominal) == pytest.approx(
        nominal * math.sin(math.atan(0.5))
    )
    # 中行程耦合最强，阻尼最大
    angles = np.linspace(0.0, MID_STROKE, 21)
    values = [decoupling_damping(float(a), nominal) for a in angles]
    assert all(later >= earlier for earlier, later in zip(values, values[1:]))


# ---- 与 MPC 侧常数的一致性（防两处各硬编码一套） -------------------------


def test_yaw_constants_match_the_mpc_settings():
    """排程表的偏航三件套必须与 MPC 的 ``MPCSettings`` 一致，否则两层会各说各话。"""
    from Algorithm.mpc import MPCSettings

    settings = MPCSettings()
    assert DEFAULT_YAW_MOMENT_LIMIT == pytest.approx(settings.yaw_moment_limit)
    assert DEFAULT_SCRUB_MOMENT_COEFFICIENT == pytest.approx(settings.scrub_moment_coefficient)
    assert DEFAULT_YAW_MOMENT_FLOOR == pytest.approx(settings.yaw_moment_floor)
