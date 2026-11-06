"""Proprioceptive body-state estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import PlanarLegGeometry, quaternion_pitch


@dataclass(frozen=True)
class EstimatorTuning:
    wheel_radius: float
    gravity: float
    pitch_measurement_std: float
    pitch_rate_measurement_std: float
    wheel_speed_measurement_std: float
    wheel_speed_yaw_std: float
    wheel_speed_leg_std_gain: float


class BodyStateKalman:
    """Kalman estimator driven only by IMU and joint-encoder measurements."""

    def __init__(
        self,
        initial_wheel_angles: np.ndarray,
        geometry: PlanarLegGeometry,
        tuning: EstimatorTuning,
    ) -> None:
        self.geometry = geometry
        self.tuning = tuning
        self.state = np.zeros(3, dtype=float)  # pitch, pitch rate, forward speed
        self.covariance = np.diag([0.04**2, 0.08**2, 0.20**2])
        self.previous_wheel_angles = np.asarray(initial_wheel_angles, dtype=float).copy()
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.x = 0.0
        self.y = 0.0
        self.forward_path = 0.0

    def _wheel_speed(self, dt: float, wheel_angles: np.ndarray) -> float:
        angles = np.asarray(wheel_angles, dtype=float)
        angular_velocity = (angles - self.previous_wheel_angles) / max(float(dt), 1.0e-6)
        self.previous_wheel_angles = angles.copy()
        right_forward_speed = self.tuning.wheel_radius * angular_velocity[0]
        left_forward_speed = -self.tuning.wheel_radius * angular_velocity[1]
        return float(0.5 * (right_forward_speed + left_forward_speed))

    def update(
        self,
        dt: float,
        gyro: np.ndarray,
        accelerometer: np.ndarray,
        orientation: np.ndarray,
        wheel_angles: np.ndarray,
        hip_angles: np.ndarray,
    ) -> None:
        dt = max(float(dt), 1.0e-6)
        gyro = np.asarray(gyro, dtype=float)
        accelerometer = np.asarray(accelerometer, dtype=float)
        wheel_speed = self._wheel_speed(dt, wheel_angles)

        gyro_pitch_rate = float(gyro[1])
        orientation_pitch = quaternion_pitch(orientation)
        forward_acceleration = float(
            accelerometer[0] + self.tuning.gravity * np.sin(float(self.state[0]))
        )

        transition = np.array(
            [[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        predicted_state = transition @ self.state
        predicted_state[1] = gyro_pitch_rate
        predicted_state[2] += forward_acceleration * dt
        process_noise = np.diag([2.0e-5 + 2.0e-3 * dt, 2.0e-3, 0.12 * dt])
        predicted_covariance = transition @ self.covariance @ transition.T + process_noise

        measurement = np.array([orientation_pitch, gyro_pitch_rate, wheel_speed], dtype=float)
        innovation = measurement - predicted_state
        measured_leg_length = float(
            np.mean([self.geometry.length(value) for value in np.asarray(hip_angles, dtype=float)])
        )
        leg_progress = np.clip(
            (measured_leg_length - self.geometry.minimum_length)
            / (self.geometry.maximum_length - self.geometry.minimum_length),
            0.0,
            1.0,
        )
        straight_wheel_std = (
            self.tuning.wheel_speed_measurement_std
            + self.tuning.wheel_speed_leg_std_gain * leg_progress
        )
        wheel_speed_std = float(
            straight_wheel_std
            + (self.tuning.wheel_speed_yaw_std - straight_wheel_std)
            * np.clip(abs(float(gyro[2])), 0.0, 1.0)
        )
        measurement_noise = np.diag(
            [
                self.tuning.pitch_measurement_std**2,
                self.tuning.pitch_rate_measurement_std**2,
                wheel_speed_std**2,
            ]
        )
        innovation_covariance = predicted_covariance + measurement_noise
        kalman_gain = predicted_covariance @ np.linalg.inv(innovation_covariance)
        self.state = predicted_state + kalman_gain @ innovation
        self.covariance = (np.eye(3) - kalman_gain) @ predicted_covariance

        self.yaw_rate = float(gyro[2])
        self.yaw += self.yaw_rate * dt
        self.x += self.state[2] * np.cos(self.yaw) * dt
        self.y += self.state[2] * np.sin(self.yaw) * dt
        self.forward_path += self.state[2] * dt

    @property
    def pitch(self) -> float:
        return float(self.state[0])

    @property
    def pitch_rate(self) -> float:
        return float(self.state[1])

    @property
    def forward_velocity(self) -> float:
        return float(self.state[2])
