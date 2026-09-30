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
