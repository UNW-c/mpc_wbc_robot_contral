"""VelocityPlanner：位移指令 -> 速度/俯仰参考。全部用模拟数据闭环推进。"""

from __future__ import annotations

import numpy as np
import pytest

from Algorithm.geometry import wrap_angle
from Algorithm.mpc import MotionCommand, MotionLimits, PlannerMode
from Algorithm.mpc.legacy_planner import LegacyPositionPlanner as VelocityPlanner

DT = 0.02
HORIZON = 25


def run_closed_loop(
    planner: VelocityPlanner,
    command: MotionCommand,
    ticks: int,
    *,
    position: float = 0.0,
    yaw: float = 0.0,
):
    """假设下层完美跟踪：每拍把实测位移换成规划器自己承诺的第一步。"""
    history = []
    for _ in range(ticks):
        reference = planner.plan(position, yaw, command, DT, HORIZON)
        history.append(reference)
        position = float(reference.position[1])
        yaw = float(reference.yaw[1])
    return position, yaw, history


def test_mode_inference():
    assert MotionCommand(position=1.0).mode is PlannerMode.POSITION
    assert MotionCommand(velocity=1.0).mode is PlannerMode.VELOCITY
    assert MotionCommand().mode is PlannerMode.VELOCITY
    assert MotionCommand(heading=0.3).yaw_mode is PlannerMode.POSITION
    assert MotionCommand(yaw_rate=0.3).yaw_mode is PlannerMode.VELOCITY


def test_states_shape_and_column_order():
    planner = VelocityPlanner()
    reference = planner.plan(0.0, 0.0, MotionCommand(position=1.0), DT, HORIZON)
    states = reference.states()
    assert states.shape == (HORIZON + 1, 6)
    assert reference.horizon == HORIZON
    np.testing.assert_allclose(states[:, 0], reference.position)
    np.testing.assert_allclose(states[:, 1], reference.velocity)
    np.testing.assert_allclose(states[:, 2], reference.pitch)
    np.testing.assert_allclose(states[:, 3], reference.pitch_rate)
    np.testing.assert_allclose(states[:, 4], reference.yaw)
    np.testing.assert_allclose(states[:, 5], reference.yaw_rate)


def test_position_mode_converges_to_target():
    planner = VelocityPlanner()
    target = 1.0
    position, _, _ = run_closed_loop(planner, MotionCommand(position=target), 600)
    assert abs(position - target) < 0.01
    assert abs(planner.speed) < 0.02


def test_position_mode_does_not_overshoot_much():
    planner = VelocityPlanner()
    target = 0.6
    _, _, history = run_closed_loop(planner, MotionCommand(position=target), 500)
    reached = np.array([reference.position[0] for reference in history])
    # 开方律 + 加速度层再开方，允许一点点超调但不应该反复冲过头
    assert reached.max() <= target + 0.02


def test_kinematic_limits_respected():
    limits = MotionLimits()
    planner = VelocityPlanner(limits)
    _, _, history = run_closed_loop(planner, MotionCommand(position=2.0, heading=1.5), 400)
    for reference in history:
        assert np.all(np.abs(reference.velocity) <= limits.max_speed + 1e-9)
        assert np.all(np.abs(reference.acceleration) <= limits.max_acceleration + 1e-9)
        assert np.all(np.abs(reference.yaw_rate) <= limits.max_yaw_rate + 1e-9)
        jerk = np.abs(np.diff(reference.acceleration)) / DT
        assert np.all(jerk <= limits.max_jerk + 1e-6)


def test_velocity_mode_ramps_then_holds():
    planner = VelocityPlanner()
    command = MotionCommand(velocity=0.8)
    _, _, history = run_closed_loop(planner, command, 200)
    assert history[0].velocity[1] < 0.8  # 第一拍不能瞬间到位
    assert planner.speed == pytest.approx(0.8, abs=1e-3)
    assert planner.acceleration == pytest.approx(0.0, abs=1e-3)


def test_velocity_mode_is_speed_limited():
    limits = MotionLimits()
    planner = VelocityPlanner(limits)
    _, _, _ = run_closed_loop(planner, MotionCommand(velocity=10.0), 200)
    assert planner.speed == pytest.approx(limits.max_speed, abs=1e-3)


@pytest.mark.parametrize("command", [MotionCommand(velocity=0.5), MotionCommand(position=1.0)])
def test_position_reference_is_the_speed_integral(command):
    """位置参考必须等于规划速度的积分，否则位置项和速度项会在代价里互相打架。

    Ruckig 的位移是分段三次多项式（速度分段二次）的精确积分，所以这里用梯形法
    核对：单区间截断误差约 ``dt^3 * jerk / 12``，dt=0.02、jerk=8 时只有 5e-6。
    """
    planner = VelocityPlanner()
    reference = planner.plan(0.3, 0.0, command, DT, HORIZON)
    trapezoid = 0.5 * (reference.velocity[:-1] + reference.velocity[1:]) * DT
    np.testing.assert_allclose(np.diff(reference.position), trapezoid, atol=1e-5)
    assert reference.position[0] == pytest.approx(0.3)


def test_position_reference_anchors_to_measurement():
    """每拍重新锚定到实测位移，不做积分累积。"""
    planner = VelocityPlanner()
    planner.plan(0.0, 0.0, MotionCommand(position=1.0), DT, HORIZON)
    reference = planner.plan(5.0, 0.0, MotionCommand(position=1.0), DT, HORIZON)
    assert reference.position[0] == pytest.approx(5.0)


def test_deadband_holds_still():
    limits = MotionLimits()
    planner = VelocityPlanner(limits)
    inside = limits.position_deadband * 0.5
    reference = planner.plan(0.0, 0.0, MotionCommand(position=inside), DT, HORIZON)
    np.testing.assert_allclose(reference.velocity, 0.0)
    np.testing.assert_allclose(reference.acceleration, 0.0)
    np.testing.assert_allclose(reference.pitch, 0.0)


def test_heading_deadband_holds_still():
    limits = MotionLimits()
    planner = VelocityPlanner(limits)
    inside = limits.heading_deadband * 0.5
    reference = planner.plan(0.0, 0.0, MotionCommand(heading=inside), DT, HORIZON)
    np.testing.assert_allclose(reference.yaw_rate, 0.0)


def test_pitch_reference_is_quasi_static_tilt():
    limits = MotionLimits()
    planner = VelocityPlanner(limits)
    reference = planner.plan(0.0, 0.0, MotionCommand(position=2.0), DT, HORIZON)
    expected = np.clip(
        np.arctan2(reference.acceleration, limits.gravity),
        -limits.pitch_reference_limit,
        limits.pitch_reference_limit,
    )
    np.testing.assert_allclose(reference.pitch, expected, atol=1e-15)
    # 前进加速 -> 正俯仰参考（机身前倾）
    assert reference.pitch[1] > 0.0


def test_pitch_reference_is_clipped():
    limits = MotionLimits(pitch_reference_limit=0.01)
    planner = VelocityPlanner(limits)
    _, _, history = run_closed_loop(planner, MotionCommand(position=3.0), 40)
    for reference in history:
        assert np.all(np.abs(reference.pitch) <= 0.01 + 1e-12)
    assert np.max(np.abs(history[-1].pitch)) == pytest.approx(0.01, abs=1e-9)


def test_pitch_rate_is_the_pitch_difference():
    planner = VelocityPlanner()
    reference = planner.plan(0.0, 0.0, MotionCommand(position=2.0), DT, HORIZON)
    np.testing.assert_allclose(
        reference.pitch_rate[:-1], np.diff(reference.pitch) / DT, atol=1e-12
    )
    assert reference.pitch_rate[-1] == pytest.approx(reference.pitch_rate[-2])


def test_heading_error_takes_the_short_way_round():
    """yaw=3.0 目标 -3.0：真实误差是 +0.283 rad，不是 -6.0 rad。"""
    planner = VelocityPlanner()
    reference = planner.plan(0.0, 3.0, MotionCommand(heading=-3.0), DT, HORIZON)
    assert wrap_angle(-3.0 - 3.0) > 0.0
    assert reference.yaw_rate[1] > 0.0


def test_heading_mode_converges():
    planner = VelocityPlanner()
    _, yaw, _ = run_closed_loop(planner, MotionCommand(heading=1.0), 400)
    assert abs(wrap_angle(yaw - 1.0)) < 0.02


def test_reset_clears_state():
    planner = VelocityPlanner()
    run_closed_loop(planner, MotionCommand(position=2.0, heading=1.0), 60)
    assert planner.speed != 0.0
    planner.reset()
    assert planner.speed == 0.0
    assert planner.acceleration == 0.0
    assert planner.yaw_rate == 0.0
    assert planner.yaw_acceleration == 0.0
    planner.reset(speed=0.4, yaw_rate=-0.2)
    assert planner.speed == pytest.approx(0.4)
    assert planner.yaw_rate == pytest.approx(-0.2)


def test_backward_command_is_signed():
    planner = VelocityPlanner()
    position, _, _ = run_closed_loop(planner, MotionCommand(position=-0.8), 500)
    assert position == pytest.approx(-0.8, abs=0.01)


def test_only_first_step_is_committed():
    """plan() 展开整条时域，但内部状态只前进一拍。"""
    planner = VelocityPlanner()
    reference = planner.plan(0.0, 0.0, MotionCommand(position=1.0), DT, HORIZON)
    assert planner.speed == pytest.approx(reference.velocity[1])
    assert planner.acceleration == pytest.approx(reference.acceleration[1])
