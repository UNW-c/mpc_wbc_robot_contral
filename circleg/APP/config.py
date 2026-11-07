"""Central, immutable application and controller configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from Algorithm.mpc import MPCSettings


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "plane": ROOT / "URDF_Model" / "mjcf" / "URDF_Model_plane.xml",
    "rough": ROOT / "URDF_Model" / "mjcf" / "URDF_Model_rough.xml",
    # RMUC 2026 场地地面（替代无限平面）。与 plane/rough 并存，走 main.py --terrain field
    "field": ROOT / "URDF_Model" / "mjcf" / "URDF_Model_field.xml",
}


@dataclass(frozen=True)
class LegConfig:
    base_length: float = 0.080
    link_length: float = 0.060
    angle_zero: float = 0.0
    command_rate: float = 0.120


@dataclass(frozen=True)
class HipConfig:
    torque_limit: float = 4.0
    kp: float = 2.0
    kd: float = 0.5
    kp_max: float = 10.0
    kd_max: float = 2.0
    torque_limit_max: float = 10.0
    kp_mid_bump: float = 40.0
    kd_mid_bump: float = 8.0
    load_feedforward: bool = True
    load_feedforward_scale: float = 0.6


@dataclass(frozen=True)
class LQRConfig:
    q_diagonal: tuple[float, float, float, float] = (16.0, 800.0, 500.0, 180.0)
    r: float = 1.0
    right_local_sign: float = 1.0
    left_local_sign: float = -1.0
    velocity_reference_rate: float = 1.2
    velocity_pitch_gain: float = 0.12
    pitch_reference_limit: float = 0.12

    @property
    def Q(self) -> np.ndarray:
        return np.diag(self.q_diagonal)

    @property
    def R(self) -> np.ndarray:
        return np.array([[self.r]], dtype=float)


@dataclass(frozen=True)
class YawConfig:
    kp: float = 1.0
    kd: float = 0.65
    torque_limit: float = 0.8
    command_rate: float = 9.6


@dataclass(frozen=True)
class EstimatorConfig:
    gravity: float = 9.81
    pitch_measurement_std: float = 4.0
    pitch_rate_measurement_std: float = 0.015
    wheel_speed_measurement_std: float = 2.0
    wheel_speed_yaw_std: float = 3.0
    wheel_speed_leg_std_gain: float = 8.0


@dataclass(frozen=True)
class KeyboardConfig:
    speed_step: float = 2.5


@dataclass(frozen=True)
class RobotConfig:
    wheel_radius: float = 0.160
    wheel_torque_limit: float = 5.0


def default_mpc_settings() -> "MPCSettings":
    """上层 MPC 的默认设置。

    延迟导入 ``Algorithm.mpc``：它依赖 osqp，而 osqp 只在启用 MPC 时才需要，
    不应该让原有 LQR 通路因为缺少可选依赖而起不来。
    """
    from Algorithm.mpc import MPCSettings

    return MPCSettings(
        wheel_torque_limit=RobotConfig().wheel_torque_limit,
    )


@dataclass(frozen=True)
class AppConfig:
    leg: LegConfig = field(default_factory=LegConfig)
    hip: HipConfig = field(default_factory=HipConfig)
    lqr: LQRConfig = field(default_factory=LQRConfig)
    yaw: YawConfig = field(default_factory=YawConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    model_paths: dict[str, Path] = field(default_factory=lambda: dict(MODEL_PATHS))


DEFAULT_CONFIG = AppConfig()
