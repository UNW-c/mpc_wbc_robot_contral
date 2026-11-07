"""High-level chassis control orchestration.

This module owns references, command semantics and simulator I/O.  The actual
LQR, Kalman, PD and torque-mixing mathematics live in :mod:`Algorithm`.
"""

from __future__ import annotations

from typing import Any, Protocol

import mujoco
import numpy as np

from Algorithm.controllers import AdaptiveHipPD, WheelTorqueMixer, YawPD
from Algorithm.estimator import BodyStateKalman, EstimatorTuning
from Algorithm.geometry import PlanarLegGeometry
from Algorithm.lqr import LQRDesign, PendulumParameters, WheelLQR

from .config import DEFAULT_CONFIG, AppConfig
from .mujoco_robot import MuJoCoRobot


class CommandSource(Protocol):
    def read(self) -> float: ...
    def read_yaw_rate(self) -> float: ...
    def read_leg_length_rate(self) -> float: ...
    def read_reset_posture(self) -> bool: ...


class ChassisController:
    """Compose all algorithms into one stateful robot controller."""

    def __init__(
        self,
        model: Any,
        data: Any,
        config: AppConfig = DEFAULT_CONFIG,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        self.robot = MuJoCoRobot(model, data)
        self.geometry = PlanarLegGeometry(config.leg.base_length, config.leg.link_length)
        self.estimator_tuning = EstimatorTuning(
            wheel_radius=config.robot.wheel_radius,
            gravity=config.estimator.gravity,
            pitch_measurement_std=config.estimator.pitch_measurement_std,
            pitch_rate_measurement_std=config.estimator.pitch_rate_measurement_std,
            wheel_speed_measurement_std=config.estimator.wheel_speed_measurement_std,
            wheel_speed_yaw_std=config.estimator.wheel_speed_yaw_std,
            wheel_speed_leg_std_gain=config.estimator.wheel_speed_leg_std_gain,
        )
        self.estimator = self._new_estimator()
        self.parameters = PendulumParameters.from_model(model, data)
        self.design = LQRDesign.from_model(
            model,
            data,
            self.parameters,
            config.lqr.Q,
            config.lqr.R,
        )
        self.right_lqr = WheelLQR(
            self.design,
            config.lqr.right_local_sign,
            config.robot.wheel_torque_limit,
        )
        self.left_lqr = WheelLQR(
            self.design,
            config.lqr.left_local_sign,
            config.robot.wheel_torque_limit,
        )
        hip = config.hip
        self.hip_controller = AdaptiveHipPD(
            hip.kp,
            hip.kd,
            hip.kp_max,
            hip.kd_max,
            hip.torque_limit,
            hip.torque_limit_max,
            hip.kp_mid_bump,
            hip.kd_mid_bump,
            hip.load_feedforward,
            hip.load_feedforward_scale,
        )
        self.yaw_controller = YawPD(config.yaw.kp, config.yaw.kd, config.yaw.torque_limit)
        self.torque_mixer = WheelTorqueMixer(config.robot.wheel_torque_limit)

        self.x_reference = self.estimator.x
        self.path_reference = self.estimator.forward_path
        self.velocity_reference = 0.0
        self.hip_angle_reference = config.leg.angle_zero
        self.leg_length_reference = self.geometry.length(config.leg.angle_zero)
        self.physical_velocity_reference = 0.0
        self._velocity_command = 0.0
        self._yaw_unwrapped = self.estimator.yaw
        self.yaw_reference = self._yaw_unwrapped
        self.yaw_rate_reference = 0.0
        self._last_yaw_command = 0.0
        self._yaw_control_active = False
        self._yaw_position_hold_active = False
        self._yaw_hold_origin = np.array([self.estimator.x, self.estimator.y], dtype=float)
        self._path_hold_origin = self.estimator.forward_path
        self._lqr_velocity = self.estimator.forward_velocity
        self.yaw_torque = 0.0
        self._last_data_time = float(self.data.time)

    def _new_estimator(self) -> BodyStateKalman:
        return BodyStateKalman(self.robot.wheel_angles(), self.geometry, self.estimator_tuning)

    def _update_estimator(self, dt: float) -> None:
        self.estimator.update(
            dt,
            self.robot.sensor("imu_gyro"),
            self.robot.sensor("imu_acc"),
            self.robot.sensor("imu_quat"),
            self.robot.wheel_angles(),
            self.robot.hip_angles(),
        )

    def reset_runtime_state(self) -> None:
        """Re-anchor all Python state after MuJoCo resets simulation data."""
        mujoco.mj_forward(self.model, self.data)
        self.estimator = self._new_estimator()
        self.x_reference = self.estimator.x
        self.path_reference = self.estimator.forward_path
        self.velocity_reference = 0.0
        self.physical_velocity_reference = 0.0
        self._velocity_command = 0.0
        self.set_hip_angle_reference(float(np.mean(self.robot.hip_angles())))
        self._yaw_unwrapped = self.estimator.yaw
        self.yaw_reference = self.estimator.yaw
        self.yaw_rate_reference = 0.0
        self._last_yaw_command = 0.0
        self._yaw_control_active = False
        self._yaw_position_hold_active = False
        self._yaw_hold_origin = np.array([self.estimator.x, self.estimator.y], dtype=float)
        self._path_hold_origin = self.estimator.forward_path
        self._lqr_velocity = self.estimator.forward_velocity
        self.yaw_torque = 0.0
        self._last_data_time = float(self.data.time)

    def update_velocity_reference(self, commanded_speed: float, dt: float) -> None:
        commanded_speed = float(commanded_speed)
        self._velocity_command = commanded_speed
        maximum_change = self.config.lqr.velocity_reference_rate * max(float(dt), 0.0)
        self.velocity_reference += float(
            np.clip(
                commanded_speed - self.velocity_reference,
                -maximum_change,
                maximum_change,
            )
        )
        if abs(self.velocity_reference) > 1.0e-9:
            self._yaw_position_hold_active = False
            self.x_reference += self.velocity_reference * float(dt)
            self.path_reference += self.velocity_reference * float(dt)
        elif self._yaw_position_hold_active:
            self.x_reference = float(self._yaw_hold_origin[0])
            self.path_reference = self._path_hold_origin
        else:
            self.x_reference = self.estimator.x
            self.path_reference = self.estimator.forward_path
        self.physical_velocity_reference = self._velocity_command

    def update_yaw_reference(self, commanded_yaw_rate: float, dt: float) -> None:
        commanded_yaw_rate = float(commanded_yaw_rate)
        if abs(commanded_yaw_rate) > 1.0e-9:
            self._yaw_control_active = True
            if abs(self.velocity_reference) <= 1.0e-9:
                if not self._yaw_position_hold_active:
                    self._yaw_hold_origin = np.array([self.estimator.x, self.estimator.y], dtype=float)
                    self._path_hold_origin = self.estimator.forward_path
                self._yaw_position_hold_active = True
                self.x_reference = float(self._yaw_hold_origin[0])
        if abs(commanded_yaw_rate) <= 1.0e-9 and abs(self._last_yaw_command) > 1.0e-9:
            self.yaw_reference = self.estimator.yaw
        self.yaw_rate_reference = commanded_yaw_rate
        if abs(commanded_yaw_rate) > 1.0e-9:
            self.yaw_reference += commanded_yaw_rate * float(dt)
        self._last_yaw_command = commanded_yaw_rate

    def operating_point(self) -> tuple[float, float]:
        pitch_reference = float(
            np.clip(
                self.config.lqr.velocity_pitch_gain * self.velocity_reference,
                -self.config.lqr.pitch_reference_limit,
                self.config.lqr.pitch_reference_limit,
            )
        )
        return pitch_reference, 0.0

    def set_hip_angle_reference(self, angle: float) -> None:
        bounded = float(np.clip(angle, 0.0, np.pi))
        self.hip_angle_reference = bounded
        self.leg_length_reference = self.geometry.length(bounded)

    def reset_posture_reference(self) -> None:
        """Restore the configured default leg posture without resetting motion."""
        self.set_hip_angle_reference(self.config.leg.angle_zero)

    def set_leg_length_reference(self, length: float) -> None:
        bounded = float(
            np.clip(length, self.geometry.minimum_length, self.geometry.maximum_length)
        )
        self.leg_length_reference = bounded
        self.hip_angle_reference = self.geometry.angle(bounded)

    def update_leg_length_reference(self, command_rate: float, dt: float) -> None:
        self.set_leg_length_reference(
            self.leg_length_reference + float(command_rate) * float(dt)
        )

    def hip_pd_torque(self, joint_name: str) -> float:
        progress = np.clip(
            (self.leg_length_reference - self.geometry.minimum_length)
            / (self.geometry.maximum_length - self.geometry.minimum_length),
            0.0,
            1.0,
        )
        return self.hip_controller.torque(
            self.robot.joint_position(joint_name),
            self.robot.joint_velocity(joint_name),
            self.hip_angle_reference,
            float(progress),
            self.robot.joint_load(joint_name),
        )

    def yaw_pd_torque(self) -> float:
        return self.yaw_controller.torque(
            self.estimator.yaw,
            self.yaw_reference,
            self.estimator.yaw_rate,
            self.yaw_rate_reference,
        )

    def mix_wheel_torques(
        self,
        right_base_torque: float,
        left_base_torque: float,
        yaw_torque: float,
    ) -> tuple[float, float]:
        return self.torque_mixer.mix(right_base_torque, left_base_torque, yaw_torque)

    def apply(
        self,
        dt: float = 0.0,
        keyboard: CommandSource | None = None,
    ) -> tuple[float, float, float, float]:
        current_time = float(self.data.time)
        if current_time + max(float(dt), 1.0e-6) < self._last_data_time:
            self.reset_runtime_state()
        self._last_data_time = current_time
        self._update_estimator(dt)
        self._yaw_unwrapped = self.estimator.yaw
        if keyboard is not None:
            self.update_velocity_reference(keyboard.read(), dt)
            self.update_yaw_reference(keyboard.read_yaw_rate(), dt)
            if keyboard.read_reset_posture():
                self.reset_posture_reference()
            self.update_leg_length_reference(keyboard.read_leg_length_rate(), dt)

        if self._yaw_control_active:
            x = self.estimator.forward_path
            x_reference = self.path_reference
        else:
            x = self.estimator.x
            x_reference = self.x_reference
        velocity = self.estimator.forward_velocity
        self._lqr_velocity = velocity
        pitch_reference, equilibrium_torque = self.operating_point()

        right_wheel = self.right_lqr.torque(
            x,
            x_reference,
            self.estimator.pitch,
            pitch_reference,
            velocity,
            self.velocity_reference,
            self.estimator.pitch_rate,
            equilibrium_torque,
        )
        left_wheel = self.left_lqr.torque(
            x,
            x_reference,
            self.estimator.pitch,
            pitch_reference,
            velocity,
            self.velocity_reference,
            self.estimator.pitch_rate,
            equilibrium_torque,
        )
        right_hip = self.hip_pd_torque("joint1")
        left_hip = self.hip_pd_torque("joint3")
        self.yaw_torque = self.yaw_pd_torque()
        right_wheel, left_wheel = self.mix_wheel_torques(
            right_wheel,
            left_wheel,
            self.yaw_torque,
        )
        self.robot.apply_torques(right_wheel, left_wheel, right_hip, left_hip)
        return right_wheel, left_wheel, right_hip, left_hip


# Original public name retained for experiment scripts.
MinimalController = ChassisController
