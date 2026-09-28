"""Approved velocity-reference contract, including event and horizon semantics."""
import numpy as np
import pytest

from Algorithm.mpc import MotionCommand, MotionLimits, VelocityPlanner, ReferenceFeedback, ReferenceMode


def step(planner, velocity=0.0, **kwargs):
    return planner.plan(kwargs.pop("position", 0.0), kwargs.pop("yaw", 0.0),
                        MotionCommand(velocity=velocity, yaw_rate=kwargs.pop("turn", 0.0)),
                        kwargs.pop("dt", 0.02), kwargs.pop("horizon", 25), **kwargs)


@pytest.mark.parametrize("sign", [-1, 1])
def test_acceleration_braking_and_reverse_integrate_events(sign):
    p = VelocityPlanner(MotionLimits(commanded_acceleration=1.0, max_braking=2.0))
    p.reset(speed=sign * 0.1, position=0.0, yaw=0.0)
    r = step(p, -sign * 0.3, dt=0.2)
    # 0.05 s to zero: 0.0025 m, followed by 0.15 s in reverse: -0.01125 m.
    assert r.position[1] == pytest.approx(sign * -0.00875)
    assert r.velocity[1] == pytest.approx(sign * -0.15)
    assert r.velocity[2] == pytest.approx(sign * -0.3)
    assert p.speed == r.velocity[1]
    assert np.max(np.abs(r.acceleration)) <= 2.0


def test_reaching_target_mid_step_uses_exact_integral():
    p = VelocityPlanner(MotionLimits(commanded_acceleration=1.0))
    r = step(p, 0.05, dt=0.2)
    assert r.position[1] == pytest.approx(0.5 * 0.05**2 + 0.05 * 0.15)


def test_drive_reference_follows_the_measurement_while_velocity_persists():
    p = VelocityPlanner()
    r = step(p, 0.5)
    following = step(p, 0.5, position=-0.1)
    # 有速度指令时，位移锚点跟随实测，不再按上一拍积分前推；速度参考仍然持续。
    assert following.position[0] == -0.1
    assert following.velocity[0] == r.velocity[1]
    np.testing.assert_allclose(following.pitch, 0.0)


def test_preview_lead_does_not_change_committed_clock():
    plain, shifted = VelocityPlanner(), VelocityPlanner()
    r = step(plain, 0.5)
    s = step(shifted, 0.5, lead=0.005)
    assert plain.speed == shifted.speed
    # C10 后 commanded_acceleration = 1.2：位移斜坡 1.2*lead = 0.006 m/s、0.5*1.2*lead^2 = 1.5e-5 m
    assert s.velocity[0] == pytest.approx(0.006)
    assert s.position[0] == pytest.approx(0.000015)
    assert r.position[0] == 0.0


def test_release_brakes_then_captures_hold_once_and_restores_push():
    p = VelocityPlanner()
    p.reset(speed=0.4, position=0.0, yaw=0.0)
    r = step(p, measured_speed=0.4)
    assert r.position[1] > 0 and p.mode == ReferenceMode.BRAKE
    for _ in range(30):
        step(p, position=0.04, measured_speed=0.0)
    assert p.mode == ReferenceMode.HOLD
    resets = p.reset_count
    r = step(p, position=0.07, measured_speed=0.02)
    # HOLD 捕获时锚定的是参考位置 = 0.4^2 / (2*max_braking)。
    # C10 前 max_braking = 2.0 → 0.04；C10 后 1.4 → 0.057143。
    assert r.position[0] == pytest.approx(0.4 ** 2 / (2.0 * 1.4), rel=1e-6)
    assert p.reset_count == resets
    assert p.mode == ReferenceMode.HOLD


def test_unstable_pose_prevents_hold_capture():
    p = VelocityPlanner()
    for _ in range(30):
        step(p, feedback=ReferenceFeedback(pitch=0.2))
    assert p.mode == ReferenceMode.BRAKE


def test_airborne_pause_and_contact_chatter_do_not_restore_old_reference():
    p = VelocityPlanner()
    step(p, 0.4)
    r = step(p, 0.4, feedback=ReferenceFeedback(contact=(False, False)))
    assert not p.valid
    np.testing.assert_allclose(np.diff(r.position), 0.0)
    for _ in range(5):
        step(p, 0.4, feedback=ReferenceFeedback(contact=(True, True)))
        step(p, 0.4, feedback=ReferenceFeedback(contact=(False, False)))
    assert not p.valid
    for _ in range(5):
        r = step(p, 0.4, position=2.0, measured_speed=0.15)
    assert p.valid and p.last_reset_reason == "stable_resume"
    assert r.position[0] == 2.0 and r.velocity[0] == 0.15


def test_saturation_requires_final_motion_residual_and_persistence():
    p = VelocityPlanner()
    for _ in range(6):
        step(p, 0.4, feedback=ReferenceFeedback(saturated=True))
    assert p.capability == 1.0
    for _ in range(6):
        r = step(p, 0.4, feedback=ReferenceFeedback(saturated=True, motion_residual=2.0))
    assert p.capability == 0.25
    assert r.velocity[-1] == pytest.approx(0.1)
    step(p, 0.4)
    assert 0.25 < p.capability < 1.0


def test_slip_and_single_contact_keep_the_drive_target_on_the_measurement():
    p = VelocityPlanner()
    step(p, 0.4, feedback=ReferenceFeedback(contact=(True, False)))
    assert p.valid and p.capability == 0.25
    committed = p.reset_count
    r = step(p, 0.4, position=2.0, measured_speed=0.2, feedback=ReferenceFeedback(slipping=True))
    # 有速度指令时位移锚点跟随实测（2.0），不需要额外重锚；打滑不改变这一点。
    assert p.reset_count == committed
    assert p.last_reset_reason == "enable"
    assert r.position[0] == 2.0
    assert r.velocity[-1] == 0.0


def test_yaw_integral_is_unwrapped_across_pi():
    p = VelocityPlanner()
    r = step(p, yaw=3.14, turn=0.4)
    following = step(p, yaw=-3.14, turn=0.4)
    assert following.yaw[0] == r.yaw[1]
    assert following.yaw[-1] > np.pi


@pytest.mark.parametrize("feedback", [ReferenceFeedback(enabled=False), ReferenceFeedback(command_fresh=False)])
def test_disabled_or_stale_input_suspends_reference(feedback):
    p = VelocityPlanner()
    step(p, 0.4, feedback=feedback)
    assert not p.valid


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_command_rejected(value):
    with pytest.raises(ValueError):
        step(VelocityPlanner(), value)


def test_absolute_position_is_not_silently_used_for_driving():
    with pytest.raises(ValueError, match="velocity/yaw_rate"):
        VelocityPlanner().plan(0, 0, MotionCommand(position=1.0), 0.02, 25)
