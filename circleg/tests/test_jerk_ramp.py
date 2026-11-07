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


# ---- 第 24 / 27 次排查：刹车 jerk（B2）与分模式髋限（方案 B） ----

def test_braking_jerk_used_only_when_speed_magnitude_decreases():
    from Algorithm.mpc.planner import _RampState
    slow, fast = 5.0, 40.0
    # 起步：速度幅值增大，用 drive jerk
    up = _RampState(0.0, 0.0, 0.0).advance_jerk(2.0, 0.02, 2.0, 2.0, slow, braking_jerk=fast)
    assert up.acceleration == pytest.approx(slow * 0.02, rel=1e-6)
    # 刹车：速度幅值减小，用 braking jerk
    down = _RampState(0.0, 2.0, 0.0).advance_jerk(0.0, 0.02, 2.0, 2.0, slow, braking_jerk=fast)
    assert down.acceleration == pytest.approx(-fast * 0.02, rel=1e-6)
    # braking_jerk=None 与旧行为逐位一致
    a = _RampState(0.0, 2.0, 0.0).advance_jerk(0.0, 0.02, 2.0, 2.0, slow)
    b = _RampState(0.0, 2.0, 0.0).advance_jerk(0.0, 0.02, 2.0, 2.0, slow, braking_jerk=None)
    assert (a.position, a.velocity, a.acceleration) == (b.position, b.velocity, b.acceleration)


def _drive_limit_mpc(drive=1.2):
    from types import SimpleNamespace
    from Algorithm.mpc.controller import MPCSettings, WheelMPC
    mpc = WheelMPC.__new__(WheelMPC)
    mpc.settings = MPCSettings(hip_angle_limit=0.8, drive_hip_angle_limit=drive)
    mpc.active_hip_limit = 0.8
    mpc.planner = SimpleNamespace(commanded_motion=False)
    return mpc


def test_drive_hip_limit_opens_only_under_command():
    from Algorithm.mpc.controller import ChassisState
    mpc = _drive_limit_mpc()
    mpc._update_hip_limit(ChassisState(hip_angle=0.0))
    assert mpc.active_hip_limit == pytest.approx(0.8)
    mpc.planner.commanded_motion = True
    mpc._update_hip_limit(ChassisState(hip_angle=0.0))
    assert mpc.active_hip_limit == pytest.approx(1.2)


def test_hold_limit_ramps_down_and_never_ratchets_up():
    from Algorithm.mpc.controller import ChassisState
    mpc = _drive_limit_mpc()
    mpc.active_hip_limit = 1.2
    step = mpc.settings.hip_limit_relax_rate * mpc.settings.dt
    # 髋角仍大时回落受 |q| + margin 托底（这里托底 1.15 低于斜坡值 1.2 - step，斜坡起作用）
    mpc._update_hip_limit(ChassisState(hip_angle=1.1))
    assert mpc.active_hip_limit == pytest.approx(1.2 - step)
    # 托底高于斜坡值时，停在 |q| + margin
    mpc._update_hip_limit(ChassisState(hip_angle=1.18))
    assert mpc.active_hip_limit == pytest.approx(1.2 - step)
    mpc.active_hip_limit = 1.2
    mpc._update_hip_limit(ChassisState(hip_angle=1.18))
    assert mpc.active_hip_limit == pytest.approx(1.2)
    # 髋角回到 0 后按斜坡回落，最终停在 hip_angle_limit
    mpc._update_hip_limit(ChassisState(hip_angle=0.0))
    assert mpc.active_hip_limit == pytest.approx(1.2 - step)
    for _ in range(200):
        mpc._update_hip_limit(ChassisState(hip_angle=0.0))
    assert mpc.active_hip_limit == pytest.approx(0.8)
    # 零指令下受扰把髋推到 0.8 以上，约束不跟着抬升
    mpc._update_hip_limit(ChassisState(hip_angle=1.0))
    assert mpc.active_hip_limit == pytest.approx(0.8)


def test_drive_hip_limit_none_keeps_old_behaviour():
    from Algorithm.mpc.controller import ChassisState
    mpc = _drive_limit_mpc(drive=None)
    mpc.planner.commanded_motion = True
    mpc._update_hip_limit(ChassisState(hip_angle=1.0))
    assert mpc.active_hip_limit == pytest.approx(0.8)


# ---- 第 28 次排查：R2'（中途反转先卸加速度） / R1'（软复位保留斜坡） ----

def test_unwind_uses_drive_jerk_while_acceleration_opposes_target():
    from Algorithm.mpc.planner import _RampState
    drive_jerk, brake_jerk = 5.0, 40.0
    common = dict(target=0.0, dt=0.02, acceleration=2.0, braking=2.0, jerk=drive_jerk,
                  substeps=20, braking_jerk=brake_jerk)
    unwound = _RampState(0.0, 1.0, 2.0).advance_jerk(**common, unwind=True)
    direct = _RampState(0.0, 1.0, 2.0).advance_jerk(**common, unwind=False)
    # 反转瞬间：unwind 先用起步 jerk 卸加速度，减速幅度明显小于直接用刹车 jerk。
    # 两者都还受收尾二次型限制（速度 1.0 m/s、目标 0，一拍最多减 jerk*dt），所以都不为负。
    assert unwound.acceleration > direct.acceleration
    assert unwound.acceleration >= 2.0 - drive_jerk * 0.02 - 1e-9
    assert direct.acceleration >= 2.0 - brake_jerk * 0.02 - 1e-9
    assert direct.acceleration < 2.0 - drive_jerk * 0.02
    # 两者最终都能停住
    for state in (unwound, direct):
        while state.velocity > 0.0:
            state = state.advance_jerk(**common, unwind=True)
        assert state.velocity == pytest.approx(0.0, abs=1e-9)


def test_unwind_flag_off_is_unchanged():
    from Algorithm.mpc.planner import _RampState
    a = _RampState(0.0, 1.0, 2.0).advance_jerk(0.0, 0.02, 2.0, 2.0, 5.0, braking_jerk=40.0)
    b = _RampState(0.0, 1.0, 2.0).advance_jerk(0.0, 0.02, 2.0, 2.0, 5.0, braking_jerk=40.0, unwind=False)
    assert (a.position, a.velocity, a.acceleration) == (b.position, b.velocity, b.acceleration)


def test_soft_reanchor_keeps_velocity_acceleration_and_command_flag():
    from Algorithm.mpc.planner import VelocityPlanner, _RampState
    planner = VelocityPlanner()
    planner._forward = _RampState(1.0, 2.0, 1.5)
    planner._yaw = _RampState(0.3, 0.1)
    planner._commanded = True
    planner._initialized = True
    planner.soft_reanchor(5.0, 0.5)
    assert planner._forward.position == pytest.approx(5.0)
    assert planner._forward.velocity == pytest.approx(2.0)
    assert planner._forward.acceleration == pytest.approx(1.5)
    assert planner._yaw.position == pytest.approx(0.5)
    assert planner._yaw.velocity == pytest.approx(0.1)
    assert planner.commanded_motion is True
    assert planner.valid is True
    # 硬复位仍然是旧行为：斜坡清零、commanded 清零
    planner.reset(speed=1.0, yaw_rate=0.0, position=0.0, yaw=0.0)
    assert planner._forward.velocity == pytest.approx(1.0)
    assert planner._forward.acceleration == pytest.approx(0.0)
    assert planner.commanded_motion is False


# ---- 第 31 次排查 P1：指令刹车期间位置参考跟随实测（减速位移不计入误差） ----

def _brake_planner(reanchor: bool):
    from Algorithm.mpc.planner import MotionLimits, VelocityPlanner, _RampState, ReferenceMode
    planner = VelocityPlanner(MotionLimits(reanchor_position_while_braking=reanchor))
    planner._forward = _RampState(1.0, 2.0, 0.0)
    planner._yaw = _RampState(0.0, 0.0)
    planner._initialized = True
    planner._commanded = True
    planner.mode = ReferenceMode.DRIVE
    return planner


def test_p1_reanchors_position_while_braking():
    from Algorithm.mpc.planner import ReferenceFeedback
    from Algorithm.mpc.planner import MotionCommand
    planner = _brake_planner(True)
    # 指令松开 → 进入 BRAKE；实测位置已经比参考落后 0.2 m，开启 P1 后参考应立即跟到实测
    planner.plan(1.2, 0.0, MotionCommand(velocity=0.0, yaw_rate=0.0), 0.02, 5,
                 measured_speed=2.0, feedback=ReferenceFeedback(contact=(True, True)))
    # 重锚到实测 1.2 后，本拍再按斜坡速度前推一小步（2.0 m/s × 0.02 s）
    assert 1.2 <= planner._forward.position <= 1.26


def test_p1_off_keeps_planned_stop_point():
    from Algorithm.mpc.planner import MotionCommand, ReferenceFeedback
    planner = _brake_planner(False)
    planner.plan(1.2, 0.0, MotionCommand(velocity=0.0, yaw_rate=0.0), 0.02, 5,
                 measured_speed=2.0, feedback=ReferenceFeedback(contact=(True, True)))
    assert planner._forward.position < 1.06   # 仍从规划停点 1.0 起积分，不跟随实测


def test_p1_does_not_apply_without_command():
    """没有速度指令驱动时（零指令站立 / 受扰恢复）不得重锚，零输入行为保持不变。"""
    from Algorithm.mpc.planner import MotionCommand, ReferenceFeedback
    planner = _brake_planner(True)
    planner._commanded = False
    planner.plan(1.2, 0.0, MotionCommand(velocity=0.0, yaw_rate=0.0), 0.02, 5,
                 measured_speed=0.0, feedback=ReferenceFeedback(contact=(True, True)))
    assert planner._forward.position < 1.06
