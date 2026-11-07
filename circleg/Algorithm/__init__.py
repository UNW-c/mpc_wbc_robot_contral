"""Reusable control and state-estimation algorithms for CirclegRobot."""

from .controllers import AdaptiveHipPD, WheelTorqueMixer, YawPD
from .estimator import BodyStateKalman, EstimatorTuning
from .geometry import PlanarLegGeometry, quaternion_pitch, quaternion_yaw, wrap_angle
from .lqr import LQRDesign, PendulumParameters, WheelLQR, lqr_gain_from_linearization

__all__ = [
    "AdaptiveHipPD",
    "BodyStateKalman",
    "EstimatorTuning",
    "LQRDesign",
    "PendulumParameters",
    "PlanarLegGeometry",
    "WheelLQR",
    "WheelTorqueMixer",
    "YawPD",
    "lqr_gain_from_linearization",
    "quaternion_pitch",
    "quaternion_yaw",
    "wrap_angle",
]
