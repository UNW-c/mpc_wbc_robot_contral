"""Small, testable control-law objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveHipPD:
    """Hip PD whose gains follow the commanded leg extension."""

    kp_min: float
    kd_min: float
    kp_max: float
    kd_max: float
    torque_min: float
    torque_max: float
    kp_mid_bump: float
    kd_mid_bump: float
    load_feedforward_enabled: bool
    load_feedforward_scale: float

    def torque(
        self,
        position: float,
        velocity: float,
        target_position: float,
        extension_progress: float,
        joint_load: float,
    ) -> float:
        progress = float(np.clip(extension_progress, 0.0, 1.0))
        midpoint_profile = 4.0 * progress * (1.0 - progress)
        kp = self.kp_min + (self.kp_max - self.kp_min) * progress + self.kp_mid_bump * midpoint_profile
        kd = self.kd_min + (self.kd_max - self.kd_min) * progress + self.kd_mid_bump * midpoint_profile
        torque_limit = self.torque_min + (self.torque_max - self.torque_min) * progress
        feedforward = 0.0
        if self.load_feedforward_enabled and progress > 0.05:
            feedforward = -self.load_feedforward_scale * float(joint_load)
        command = feedforward - kp * (float(position) - float(target_position)) - kd * float(velocity)
        return float(np.clip(command, -torque_limit, torque_limit))


@dataclass(frozen=True)
class YawPD:
    kp: float
    kd: float
    torque_limit: float

    def torque(
        self,
        yaw: float,
        yaw_reference: float,
        yaw_rate: float,
        yaw_rate_reference: float,
    ) -> float:
        command = self.kp * (float(yaw_reference) - float(yaw))
        command += self.kd * (float(yaw_rate_reference) - float(yaw_rate))
        return float(np.clip(command, -self.torque_limit, self.torque_limit))


@dataclass(frozen=True)
class WheelTorqueMixer:
    """Combine longitudinal and yaw commands without corrupting longitudinal torque."""

    torque_limit: float

    def mix(
        self,
        right_base_torque: float,
        left_base_torque: float,
        yaw_torque: float,
    ) -> tuple[float, float]:
        longitudinal = 0.5 * (float(right_base_torque) - float(left_base_torque))
        yaw_min = -self.torque_limit + abs(longitudinal)
        yaw_max = self.torque_limit - abs(longitudinal)
        yaw = float(np.clip(float(yaw_torque), yaw_min, yaw_max))
        right = float(np.clip(longitudinal + yaw, -self.torque_limit, self.torque_limit))
        left = float(np.clip(-longitudinal + yaw, -self.torque_limit, self.torque_limit))
        return right, left
