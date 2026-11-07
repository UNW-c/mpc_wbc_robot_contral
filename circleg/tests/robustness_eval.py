"""Deterministic base-push experiments and a normalized robustness cost.

Each cost term is normalized by the square of an allowed deviation (Bryson
rule), so the terms are comparable without hand-tuned scaling.
"""
from __future__ import annotations

import logging

import numpy as np

from Algorithm.mpc import MotionCommand
from APP.mujoco_mpc_wbc import Disturbance, MuJoCoMpcWbcSimulation

_LOGGER = logging.getLogger(__name__)

# Bryson normalizers: 1 / (allowed deviation)^2.
_ALLOWED_POSITION = 0.05  # [m]
_ALLOWED_SPEED = 0.20  # [m/s]
_ALLOWED_PITCH = 0.10  # [rad]
_WHEEL_TORQUE_LIMIT = 5.0  # [N*m] wheel ctrlrange in URDF_Model_plane.xml
# No measured basis yet; chosen well above the O(10) cost of a non-failing case.
_FAIL_PENALTY = 1000.0

_PUSH_TIME = 5.0  # [s]
_EVALUATION_SECONDS = 20.0  # [s]

_BASE_CASES: dict[str, tuple[float, float, float]] = {
    "forward": (0.20, 0.0, 0.0),
    "backward": (-0.20, 0.0, 0.0),
    "lateral": (0.0, 0.20, 0.0),
}


def _integral(values: np.ndarray, times: np.ndarray) -> float:
    """Trapezoidal integral, independent of the numpy trapz/trapezoid rename."""
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(times)))


def case_cost(summary: dict, records: list[dict]) -> tuple[float, dict[str, float]]:
    """Normalized cost of one experiment; smaller is better."""
    times = np.array([record["time"] for record in records])
    states = np.array([record["state"] for record in records])
    torques = np.array([record["torque"] for record in records])
    terms = {
        "position": _integral(states[:, 0] ** 2, times) / _ALLOWED_POSITION**2,
        "speed": _integral(states[:, 1] ** 2, times) / _ALLOWED_SPEED**2,
        "pitch": _integral(states[:, 2] ** 2, times) / _ALLOWED_PITCH**2,
        "effort": _integral(np.sum(torques**2, axis=1), times) / _WHEEL_TORQUE_LIMIT**2,
        "terminal_position": float(states[-1, 0] ** 2) / _ALLOWED_POSITION**2,
        "terminal_speed": float(states[-1, 1] ** 2) / _ALLOWED_SPEED**2,
        "failure": 0.0 if summary["stop_reason"] is None else _FAIL_PENALTY,
    }
    return float(sum(terms.values())), terms


def run_case(
    delta_velocity: tuple[float, float, float],
    seconds: float = _EVALUATION_SECONDS,
) -> tuple[dict, list[dict]]:
    simulation = MuJoCoMpcWbcSimulation()
    push = Disturbance(time=_PUSH_TIME, delta_velocity=tuple(float(v) for v in delta_velocity))
    summary = simulation.run(MotionCommand(velocity=0.0, yaw_rate=0.0), seconds, None, [push])
    return summary, simulation.records


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    total = 0.0
    for name, delta_velocity in _BASE_CASES.items():
        summary, records = run_case(delta_velocity)
        cost, terms = case_cost(summary, records)
        total += cost
        _LOGGER.info(
            "%s dV=%s cost=%.3f stop=%s seconds=%.3f terms=%s",
            name,
            delta_velocity,
            cost,
            summary["stop_reason"],
            summary["simulated_seconds"],
            {key: round(value, 3) for key, value in terms.items()},
        )
    _LOGGER.info("baseline total cost = %.3f", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
