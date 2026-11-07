"""Crank-wheel kinematics and contact-gated velocity/bias estimation.

Reference point B is the BODY FRAME ORIGIN, not the wheel midpoint or moving COM.
Only the signed horizontal heading velocity is observed. The rolling model assumes
a horizontal plane and no tangential slip; invalid wheels supply no observation.
No simulator or application dependencies are used.
"""
from dataclasses import dataclass
import numpy as np
from .observers.validation import positive, vector, spd


def _rotation(value):
    result = np.asarray(value, dtype=float)
    if (result.shape != (3, 3) or not np.all(np.isfinite(result))
            or not np.allclose(result.T @ result, np.eye(3), atol=1e-10)
            or not np.isclose(np.linalg.det(result), 1.0, atol=1e-10)):
        raise ValueError("body_to_world must be a proper rotation")
    return result


@dataclass(frozen=True)
class CrankGeometry:
    hip_origins: np.ndarray  # Two body-frame mounting points, right then left.
    crank_length: float
    wheel_radius: float
    hip_axis_signs: tuple[int, int] = (-1, 1)
    wheel_axis_signs: tuple[int, int] = (1, -1)
    wheel_axial_offsets: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        points = np.asarray(self.hip_origins, dtype=float)
        if points.shape != (2, 3) or not np.all(np.isfinite(points)):
            raise ValueError("hip_origins must contain two finite 3D points")
        positive(self.crank_length, "crank_length")
        positive(self.wheel_radius, "wheel_radius")
        vector(self.wheel_axial_offsets, 2, "wheel_axial_offsets")
        if any(len(s) != 2 or any(x not in (-1, 1) for x in s)
               for s in (self.hip_axis_signs, self.wheel_axis_signs)):
            raise ValueError("Joint axes must have two +/- y signs")
        points = points.copy()
        points.setflags(write=False)
        object.__setattr__(self, "hip_origins", points)


class CrankWheelOdometry:
    def __init__(self, geometry: CrankGeometry):
        self.geometry = geometry

    def centers_and_derivatives(self, hip_angles):
        angles = vector(hip_angles, 2, "hip_angles")
        signs = np.asarray(self.geometry.hip_axis_signs)
        length = self.geometry.crank_length
        offsets = length * np.column_stack((signs * np.sin(angles), np.zeros(2), np.cos(angles)))
        # Collision/wheel centers can be displaced along the axle from the joint origin.
        offsets[:, 1] += self.geometry.wheel_axial_offsets
        jacobians = length * np.column_stack((signs * np.cos(angles), np.zeros(2), -np.sin(angles)))
        return self.geometry.hip_origins + offsets, jacobians

    def observe(self, body_to_world, gyro_body, hip_angles, hip_rates, wheel_rates, *,
                contact: tuple[bool, bool], slipping: tuple[bool, bool]) -> tuple[float | None, float | None]:
        rotation = _rotation(body_to_world)
        gyro = vector(gyro_body, 3, "gyro_body")
        hips = vector(hip_rates, 2, "hip_rates")
        wheels = vector(wheel_rates, 2, "wheel_rates")
        if len(contact) != 2 or len(slipping) != 2:
            raise ValueError("Two per-wheel contact/slip flags required")
        centers, derivatives = self.centers_and_derivatives(hip_angles)
        heading = rotation[:, 0].copy()
        heading[2] = 0.0
        if np.linalg.norm(heading) < 1e-8:
            return (None, None)
        heading /= np.linalg.norm(heading)
        axis = rotation[:, 1]
        downward = np.array([0.0, 0.0, -1.0])
        radius_direction = downward - axis * (axis @ downward)
        if np.linalg.norm(radius_direction) < 1e-8:
            return (None, None)
        radius = self.geometry.wheel_radius * radius_direction / np.linalg.norm(radius_direction)
        omega_body = rotation @ gyro
        observations = []
        for i in range(2):
            if not contact[i] or slipping[i]:
                observations.append(None)
                continue
            wheel_omega = omega_body + axis * (self.geometry.hip_axis_signs[i] * hips[i]
                                                + self.geometry.wheel_axis_signs[i] * wheels[i])
            center_velocity = -np.cross(wheel_omega, radius)
            offset_velocity = np.cross(omega_body, rotation @ centers[i]) + rotation @ derivatives[i] * hips[i]
            observations.append(float(heading @ (center_velocity - offset_velocity)))
        return tuple(observations)


class InertialAcceleration:
    """Transfer specific force from a fixed IMU site to a fixed body point.

    imu_to_reference is the BODY-frame displacement from IMU to reference.
    Rotating/moving COM offsets require additional terms and are not accepted here.
    """
    def __init__(self, imu_to_reference, gravity_world):
        self.offset = vector(imu_to_reference, 3, "imu_to_reference")
        self.gravity = vector(gravity_world, 3, "gravity_world")

    def forward(self, specific_force_body, gyro_body, angular_acceleration_body, body_to_world):
        rotation = _rotation(body_to_world)
        force = vector(specific_force_body, 3, "specific_force_body")
        omega = vector(gyro_body, 3, "gyro_body")
        alpha = vector(angular_acceleration_body, 3, "angular_acceleration_body")
        corrected = force + np.cross(alpha, self.offset) + np.cross(omega, np.cross(omega, self.offset))
        heading = rotation[:, 0].copy()
        heading[2] = 0.0
        if np.linalg.norm(heading) < 1e-8:
            raise ValueError("Horizontal heading undefined")
        heading /= np.linalg.norm(heading)
        return float(heading @ (rotation @ corrected + self.gravity))


@dataclass(frozen=True)
class VelocityFilterConfig:
    jerk_spectral_density: float
    bias_spectral_density: float
    acceleration_variance: float
    wheel_variances: tuple[float, float]

    def __post_init__(self):
        for name in ("jerk_spectral_density", "bias_spectral_density", "acceleration_variance"):
            positive(getattr(self, name), name)
        if len(self.wheel_variances) != 2:
            raise ValueError("Two wheel observation variances required")
        for variance in self.wheel_variances:
            positive(variance, "wheel variance")


class VelocityBiasKalman:
    """[v,a,b] filter; covariance is updated only for actual observations.

    Per-wheel errors are assumed independent conditional on compensation. Shared
    IMU/geometry errors require conservative variances or a correlated sensor model.
    """
    def __init__(self, config: VelocityFilterConfig, initial_covariance):
        self.config = config
        self._initial_covariance = spd(initial_covariance, 3, "initial_covariance")
        self.reset()

    def reset(self, velocity=0.0, position=0.0):
        if not np.all(np.isfinite([velocity, position])):
            raise ValueError("Reset values must be finite")
        self._state = np.array([velocity, 0.0, 0.0])
        self._covariance = self._initial_covariance.copy()
        self.position = float(position)

    @property
    def state(self):
        return self._state.copy()

    @property
    def covariance(self):
        return self._covariance.copy()

    def _correct(self, row, measured, variance):
        if not np.isfinite(measured):
            raise ValueError("Valid observations must be finite")
        gain = self._covariance @ row / (row @ self._covariance @ row + variance)
        self._state += gain * (measured - row @ self._state)
        residual = np.eye(3) - np.outer(gain, row)
        self._covariance = residual @ self._covariance @ residual.T + variance * np.outer(gain, gain)
        self._covariance = 0.5 * (self._covariance + self._covariance.T)

    def update(self, acceleration: float | None, wheel_velocities, dt: float):
        dt = positive(dt, "dt")
        if len(wheel_velocities) != 2:
            raise ValueError("Two optional wheel velocities required")
        if any(x is not None and not np.isfinite(x) for x in (*wheel_velocities, acceleration)):
            raise ValueError("Observations must be finite or explicitly missing")
        old_velocity = self._state[0]
        transition = np.array([[1., dt, 0.], [0., 1., 0.], [0., 0., 1.]])
        q = self.config.jerk_spectral_density
        process = np.array([[q * dt**3 / 3., q * dt**2 / 2., 0.],
                            [q * dt**2 / 2., q * dt, 0.], [0., 0., self.config.bias_spectral_density * dt]])
        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + process
        if acceleration is not None:
            self._correct(np.array([0., 1., 1.]), acceleration, self.config.acceleration_variance)
        for value, variance in zip(wheel_velocities, self.config.wheel_variances):
            if value is not None:
                self._correct(np.array([1., 0., 0.]), value, variance)
        self.position += 0.5 * dt * (old_velocity + self._state[0])
        return self.state


class ProprioceptiveVelocityEstimator:
    """Composition keeps kinematics replaceable and feeds only valid wheel data."""
    def __init__(self, odometry: CrankWheelOdometry, filter_: VelocityBiasKalman):
        self.odometry, self.filter = odometry, filter_

    def update(self, acceleration, dt, **kinematics):
        observations = self.odometry.observe(**kinematics)
        return self.filter.update(acceleration, observations, dt)
