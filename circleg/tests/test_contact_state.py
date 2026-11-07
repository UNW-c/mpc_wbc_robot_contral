"""T-C1 / T-C2（第 17 次排查）：离地检测器单元测试。"""
from __future__ import annotations

import pytest

from Algorithm.contact_state import (ContactEvent, GroundContactConfig, GroundContactEstimator,
                                     JumpPhaseHint)

_DT = 0.004


def _feed(estimator, forces, **kwargs):
    outputs = []
    for f in forces:
        outputs.append(estimator.update(f, _DT, **kwargs))
    return outputs


def test_initial_state_is_contact():
    assert GroundContactEstimator().contact == (True, True)


def test_hysteresis_band_does_not_switch():
    estimator = GroundContactEstimator()
    out = _feed(estimator, [(6.0, 14.0), (14.0, 6.0)] * 20)
    assert all(o.contact == (True, True) for o in out)


def test_single_frame_drop_ignored_two_frames_switch():
    estimator = GroundContactEstimator()
    out = _feed(estimator, [(4.0, 40.0), (40.0, 40.0)])
    assert out[0].contact == (True, True) and out[1].contact == (True, True)
    out = _feed(estimator, [(4.0, 40.0), (4.0, 40.0)])
    assert out[1].contact == (False, True)
    assert ContactEvent.SINGLE_LIFT in out[1].events


def test_liftoff_and_touchdown_events():
    estimator = GroundContactEstimator()
    out = _feed(estimator, [(0.0, 0.0)] * 2)
    assert out[-1].contact == (False, False) and ContactEvent.LIFTOFF in out[-1].events
    out = _feed(estimator, [(30.0, 30.0)] * 3)
    assert out[-1].contact == (True, True)
    assert ContactEvent.TOUCHDOWN_BOTH in out[-1].events


def test_thrust_phase_blocks_early_liftoff():
    """推力段髋角未过门限时，力的弹跳不能判离地。"""
    estimator = GroundContactEstimator()
    out = _feed(estimator, [(0.0, 0.0)] * 5, phase=JumpPhaseHint.THRUST, hip_angle=0.5)
    assert all(o.contact == (True, True) for o in out)
    out = _feed(estimator, [(0.0, 0.0)] * 2, phase=JumpPhaseHint.THRUST, hip_angle=2.5)
    assert out[-1].contact == (False, False)


def test_flight_phase_blocks_early_touchdown():
    estimator = GroundContactEstimator()
    _feed(estimator, [(0.0, 0.0)] * 2)
    estimator.set_predicted_flight(0.4)
    # 0.1 s 内出现接触抖动（< 0.6 × 0.4 = 0.24 s）：不判着地
    out = _feed(estimator, [(40.0, 40.0)] * 5, phase=JumpPhaseHint.FLIGHT)
    assert all(o.contact == (False, False) for o in out)


def test_landing_phase_uses_lower_on_threshold():
    estimator = GroundContactEstimator()
    _feed(estimator, [(0.0, 0.0)] * 2)
    out = _feed(estimator, [(10.0, 10.0)] * 3, phase=JumpPhaseHint.LANDING)   # 10 N < 15 但 > 7.5
    assert out[-1].contact == (True, True)


def test_imu_freefall_triggers_liftoff_in_jump_phases():
    estimator = GroundContactEstimator()
    out = _feed(estimator, [(40.0, 40.0)] * 2, phase=JumpPhaseHint.THRUST, hip_angle=2.5,
                specific_force_z=0.5)
    assert out[-1].contact == (False, False)


def test_imu_conflict_recorded_outside_jump():
    estimator = GroundContactEstimator()
    out = _feed(estimator, [(40.0, 40.0)] * 3, specific_force_z=0.5)
    assert out[-1].contact == (True, True)
    assert out[-1].imu_conflict is True


# ---- 第 28 次排查 R4：非跳跃相位的离地需要运动学旁证 ----

def test_lift_gate_blocks_false_liftoff_without_evidence():
    """法向力卸空但机身没有上升、也没失重：不判离地（急反转的瞬时卸载）。"""
    config = GroundContactConfig(lift_height=0.01)
    est = GroundContactEstimator(config)
    out = None
    for _ in range(5):
        out = est.update((2.0, 2.0), 0.004, base_rise=0.0, specific_force_z=9.0)
    assert out.contact == (True, True)


def test_lift_gate_allows_liftoff_when_body_rises():
    config = GroundContactConfig(lift_height=0.01)
    est = GroundContactEstimator(config)
    out = None
    for _ in range(5):
        out = est.update((2.0, 2.0), 0.004, base_rise=0.05, specific_force_z=9.0)
    assert out.contact == (False, False)


def test_lift_gate_allows_liftoff_on_free_fall():
    config = GroundContactConfig(lift_height=0.01)
    est = GroundContactEstimator(config)
    out = None
    for _ in range(5):
        out = est.update((2.0, 2.0), 0.004, base_rise=0.0, specific_force_z=1.0)
    assert out.contact == (False, False)


def test_lift_gate_off_by_default():
    est = GroundContactEstimator()
    out = None
    for _ in range(5):
        out = est.update((2.0, 2.0), 0.004, base_rise=0.0, specific_force_z=9.0)
    assert out.contact == (False, False)


# ---- 第 29 次排查 V3：非跳跃相位的离地确认时间（去抖） ----

def test_idle_debounce_suppresses_transient_unload():
    """非跳跃相位：卸空 5 帧（20 ms）不判离地，第 6 帧（24 ms）才判。"""
    config = GroundContactConfig(off_duration_idle=0.024)
    est = GroundContactEstimator(config)
    out = None
    for _ in range(5):
        out = est.update((2.0, 2.0), _DT)
    assert out.contact == (True, True)
    out = est.update((2.0, 2.0), _DT)
    assert out.contact == (False, False)
    assert ContactEvent.LIFTOFF in out.events


def test_idle_debounce_does_not_delay_jump_phases():
    """推力 / 飞行相位仍用 off_duration（2 帧），起跳判定不受去抖影响。"""
    config = GroundContactConfig(off_duration_idle=0.024)
    est = GroundContactEstimator(config)
    out = None
    for _ in range(2):
        out = est.update((0.0, 0.0), _DT, phase=JumpPhaseHint.THRUST, hip_angle=2.5)
    assert out.contact == (False, False)


def test_idle_debounce_default_is_old_behaviour():
    est = GroundContactEstimator()
    assert est.idle_off_duration == pytest.approx(est.config.off_duration)
    out = None
    for _ in range(2):
        out = est.update((0.0, 0.0), _DT)
    assert out.contact == (False, False)   # 默认仍是 2 帧
