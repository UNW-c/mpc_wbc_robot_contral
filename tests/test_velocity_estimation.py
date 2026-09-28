"""Synthetic encoder/IMU trajectories; no MuJoCo or controller gain changes."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from Algorithm.velocity_estimation import (CrankGeometry, CrankWheelOdometry, InertialAcceleration,
    VelocityFilterConfig, VelocityBiasKalman, ProprioceptiveVelocityEstimator)


def geometry():
    return CrankGeometry(np.array([[-0.01994, -0.11, -0.093022], [-0.01994, 0.13, -0.093022]]),
                         0.06, 0.16, wheel_axial_offsets=(-0.01, -0.01))


@pytest.mark.parametrize("speed,pitch_rate,hip_rates", [(0.7, 0., [0., 0.]),
    (0., 0., [0.5, -0.8]), (-0.4, 0.3, [-0.7, 0.2]), (0.2, -0.4, [0.3, 0.3])])
def test_encoder_reconstruction_removes_parent_motion(speed, pitch_rate, hip_rates):
    g = geometry()
    odometry = CrankWheelOdometry(g)
    pitch = 0.17
    hip_angles = np.array([0.4, -0.6])
    hip_rates = np.array(hip_rates)
    def centers(t):
        rotation = Rotation.from_rotvec([0., pitch + pitch_rate * t, 0.]).as_matrix()
        points = []
        for i, sign in enumerate((-1, 1)):
            angle = hip_angles[i] + hip_rates[i] * t
            crank_rotation = Rotation.from_rotvec([0., sign * angle, 0.]).as_matrix()
            points.append(np.array([speed * t, 0., 0.]) + rotation @ (
                g.hip_origins[i] + crank_rotation @ [0., g.wheel_axial_offsets[i], g.crank_length]))
        return np.array(points)
    epsilon = 1e-6
    center_speed = (centers(epsilon)[:, 0] - centers(-epsilon)[:, 0]) / (2 * epsilon)
    absolute_wheel_spin = center_speed / g.wheel_radius
    wheel_rates = (absolute_wheel_spin - pitch_rate - np.array([-1, 1]) * hip_rates) / [1, -1]
    observed = odometry.observe(Rotation.from_rotvec([0., pitch, 0.]).as_matrix(), [0., pitch_rate, 0.],
                               hip_angles, hip_rates, wheel_rates, contact=(True, True), slipping=(False, False))
    np.testing.assert_allclose(observed, [speed, speed], atol=1e-9)


def test_yaw_rotation_and_single_wheel_contact_recover_body_origin_speed():
    g = geometry()
    odometry = CrankWheelOdometry(g)
    speed, yaw_rate = 0.4, 0.8
    # Wheel-center heading speeds differ by omega cross r; raw mean is not used.
    forward = speed - yaw_rate * (g.hip_origins[:, 1] + g.wheel_axial_offsets)
    rates = forward / g.wheel_radius / [1, -1]
    observed = odometry.observe(Rotation.from_rotvec([0., 0., 1.2]).as_matrix(), [0., 0., yaw_rate],
                               [0., 0.], [0., 0.], rates, contact=(True, False), slipping=(False, False))
    assert observed[0] == pytest.approx(speed)
    assert observed[1] is None


def test_slip_and_airborne_wheels_do_not_create_pseudo_measurements():
    observed = CrankWheelOdometry(geometry()).observe(np.eye(3), [0., 0., 0.], [0., 0.],
              [0., 0.], [1000., -1000.], contact=(True, False), slipping=(True, False))
    assert observed == (None, None)


def test_imu_fixed_lever_arm_removes_rotational_acceleration():
    # IMU is 0.1 m ahead of the origin, spinning about it at 2 rad/s.
    transfer = InertialAcceleration([-0.1, 0., 0.], [0., 0., -9.81])
    value = transfer.forward([-0.4, 0., 9.81], [0., 0., 2.], [0., 0., 0.], np.eye(3))
    assert value == pytest.approx(0., abs=1e-12)


def filter_():
    return VelocityBiasKalman(VelocityFilterConfig(0.1, 1e-5, 0.01, (0.01, 0.01)), np.eye(3))


def test_bias_filter_separates_constant_accelerometer_offset():
    f = filter_()
    for _ in range(1200):
        f.update(0.2, (0., 0.), 0.01)
    np.testing.assert_allclose(f.state, [0., 0., 0.2], atol=1e-3)
    assert np.linalg.eigvalsh(f.covariance).min() > 0.


def test_missing_all_measurements_propagates_covariance_without_false_confidence():
    f = filter_()
    old = f.covariance
    dt = 0.1
    f.update(None, (None, None), dt)
    transition = np.array([[1, dt, 0], [0, 1, 0], [0, 0, 1]])
    assert np.linalg.eigvalsh(f.covariance - transition @ old @ transition.T).min() > 0.
    assert f.covariance[0, 0] > old[0, 0]


def test_composed_estimator_integrates_measured_motion_and_ignores_airborne_spin():
    f = filter_()
    f.reset(velocity=0.5, position=1.)
    estimator = ProprioceptiveVelocityEstimator(CrankWheelOdometry(geometry()), f)
    for _ in range(100):
        estimator.update(0., 0.01, body_to_world=np.eye(3), gyro_body=[0., 0., 0.],
                         hip_angles=[0., 0.], hip_rates=[0., 0.], wheel_rates=[300., -300.],
                         contact=(False, False), slipping=(False, False))
    assert f.position == pytest.approx(1.5, abs=1e-12)
    assert f.state[0] == pytest.approx(0.5)


def test_invalid_rotation_and_measurement_are_rejected():
    with pytest.raises(ValueError):
        CrankWheelOdometry(geometry()).observe(-np.eye(3), [0]*3, [0]*2, [0]*2, [0]*2,
                                               contact=(True, True), slipping=(False, False))
    f = filter_()
    with pytest.raises(ValueError):
        f.update(float("nan"), (0., 0.), 0.01)
    np.testing.assert_array_equal(f.state, 0.)
