"""A2（第 15 次排查）：S 曲线斜坡与髋角参考前馈的单元测试。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from Algorithm.mpc.planner import MotionCommand, MotionLimits, VelocityPlanner, _RampState

_DT = 0.02
_UP, _DOWN, _JERK = 1.5, 1.7, 6.0


def _run(state, target, steps, jerk=_JERK):
    velocities, accels = [state.velocity], [state.acceleration]
    for _ in range(steps):
        state = state.advance_jerk(target, _DT, _UP, _DOWN, jerk)
        velocities.append(state.velocity)
        accels.append(state.acceleration)
    return state, np.array(velocities), np.array(accels)


def test_jerk_bound_and_final_speed():
    state, v, a = _run(_RampState(), 2.5, 200)
    assert state.velocity == pytest.approx(2.5, abs=1e-9)
    assert state.acceleration == pytest.approx(0.0, abs=1e-9)
    # 加速度变化率不超过 jerk（相邻 MPC 步之间）
    assert np.max(np.abs(np.diff(a))) <= _JERK * _DT + 1e-9
    assert np.max(a) <= _UP + 1e-9
    # 不过冲
    assert np.max(v) <= 2.5 + 1e-9


def test_triangular_profile_small_step():
    """速度差很小：加速度达不到限值就要回落（三角形剖面），且不过冲。"""
    state, v, a = _run(_RampState(), 0.1, 100)
    assert state.velocity == pytest.approx(0.1, abs=1e-9)
    assert np.max(a) < _UP                     # 没有到达限值
    assert np.max(v) <= 0.1 + 1e-9


def test_reversal_passes_through_zero_with_braking_limit():
    state, v, a = _run(_RampState(velocity=2.0), -2.0, 300)
    assert state.velocity == pytest.approx(-2.0, abs=1e-9)
    assert np.min(a) >= -_DOWN - 1e-9
    assert np.max(np.abs(np.diff(a))) <= _JERK * _DT + 1e-9


def test_position_is_integral_of_velocity():
    state = _RampState()
    positions, velocities = [0.0], [0.0]
    for _ in range(150):
        state = state.advance_jerk(2.0, _DT, _UP, _DOWN, _JERK)
        positions.append(state.position)
        velocities.append(state.velocity)
    trapezoid = np.concatenate(([0.0], np.cumsum(0.5 * (np.array(velocities[1:]) + velocities[:-1]) * _DT)))
    assert np.max(np.abs(np.array(positions) - trapezoid)) < 1e-3


def test_planner_unchanged_when_jerk_limit_off():
    """use_jerk_limit=False（默认）时规划器输出与旧 advance 逐位一致。"""
    limits = MotionLimits()
    assert limits.use_jerk_limit is False
    a, b = VelocityPlanner(limits), VelocityPlanner(replace(limits, use_jerk_limit=False))
    for k in range(40):
        command = MotionCommand(velocity=1.0 if k < 20 else 0.0, yaw_rate=0.0)
        ra = a.plan(0.0, 0.0, command, _DT, 25, measured_speed=0.0, measured_yaw_rate=0.0)
        rb = b.plan(0.0, 0.0, command, _DT, 25, measured_speed=0.0, measured_yaw_rate=0.0)
        assert np.array_equal(ra.states(), rb.states())


def test_planner_s_curve_acceleration_continuous():
    limits = replace(MotionLimits(), use_jerk_limit=True, max_jerk=_JERK,
                     commanded_acceleration=_UP, max_braking=_DOWN, max_acceleration=_DOWN)
    planner = VelocityPlanner(limits)
    accels = []
    for k in range(120):
        command = MotionCommand(velocity=2.0 if k < 80 else 0.0, yaw_rate=0.0)
        reference = planner.plan(0.0, 0.0, command, _DT, 25, measured_speed=planner.speed,
                                 measured_yaw_rate=0.0)
        accels.append(reference.acceleration[0])
    assert np.max(np.abs(np.diff(accels))) <= _JERK * _DT + 1e-6


def _fake_mpc(feedforward: bool):
    """只用 _apply_hip_feedforward：不构建 QP，给一个带能力表的 8 状态模型桩。"""
    from types import SimpleNamespace
    from Algorithm.mpc.controller import MPCSettings, WheelMPC
    mpc = WheelMPC.__new__(WheelMPC)
    mpc.settings = MPCSettings(hip_reference_feedforward=feedforward, hip_angle_limit=0.8)
    mpc.model = SimpleNamespace(state_dim=8, metadata={"accel_capacity": {
        "a": [0.0, 0.5, 1.0, 1.5, 1.8, 1.9], "q_c": [0.0, 0.186, 0.386, 0.621, 0.804, 0.881]}})
    return mpc


def test_hip_feedforward_matches_capacity_table():
    mpc = _fake_mpc(True)
    reference = np.zeros((4, 8))
    mpc._apply_hip_feedforward(reference, np.array([1.0, -1.0, 0.5, 0.0]))
    assert reference[:, 6] == pytest.approx([0.386, -0.386, 0.186, 0.0])


def test_hip_feedforward_clipped_to_90_percent_of_limit():
    mpc = _fake_mpc(True)
    reference = np.zeros((2, 8))
    mpc._apply_hip_feedforward(reference, np.array([1.9, -1.9]))
    assert reference[:, 6] == pytest.approx([0.72, -0.72])


def test_hip_feedforward_off_leaves_reference_untouched():
    mpc = _fake_mpc(False)
    reference = np.zeros((3, 8))
    mpc._apply_hip_feedforward(reference, np.array([1.0, 1.0, 1.0]))
    assert np.all(reference == 0.0)


def _fake_tilt_mpc(enabled: bool):
    from types import SimpleNamespace
    from Algorithm.mpc.controller import MPCSettings, WheelMPC
    mpc = WheelMPC.__new__(WheelMPC)
    mpc.settings = MPCSettings(pitch_feedforward=enabled, hip_angle_limit=0.8)
    mpc.model = SimpleNamespace(state_dim=8, metadata={
        "accel_capacity": {"a": [0.0, 1.0, 1.8, 1.9], "q_c": [0.0, 0.386, 0.804, 0.881]},
        "accel_capacity_tilted": {"theta": float(np.radians(15.0)), "q_c": [0.0, 0.8, 0.9],
                                  "a": [0.21, 2.24, 2.43]}})
    mpc._previous_theta_ff = 0.0
    return mpc


def test_pitch_feedforward_zero_within_hip_capacity():
    mpc = _fake_tilt_mpc(True)
    reference = np.zeros((5, 8))
    mpc._apply_pitch_feedforward(reference, np.full(5, 1.0))   # 1.0 < 0.85 * a_hip
    assert np.all(reference[:, 2] == 0.0)


def test_pitch_feedforward_sign_limit_and_rate():
    mpc = _fake_tilt_mpc(True)
    reference = np.zeros((40, 8))
    mpc._apply_pitch_feedforward(reference, np.full(40, -2.5))  # 刹车：后仰
    theta = reference[:, 2]
    assert np.all(theta <= 0.0)
    limit = mpc.settings.pitch_feedforward_max
    assert np.min(theta) >= -limit - 1e-12
    assert np.min(theta) == pytest.approx(-limit)
    assert np.max(np.abs(np.diff(theta))) <= 1.0 * 0.02 + 1e-12


def test_pitch_feedforward_off_unchanged():
    mpc = _fake_tilt_mpc(False)
    reference = np.zeros((5, 8))
    mpc._apply_pitch_feedforward(reference, np.full(5, 2.5))
    assert np.all(reference == 0.0)
