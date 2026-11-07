"""Geometry and orientation calculations with no application dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def quaternion_pitch(quaternion: np.ndarray) -> float:
    """Return pitch from a MuJoCo-style ``[w, x, y, z]`` quaternion."""
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))


def quaternion_yaw(quaternion: np.ndarray) -> float:
    """Return yaw from a MuJoCo-style ``[w, x, y, z]`` quaternion."""
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


@dataclass(frozen=True)
class PlanarLegGeometry:
    """Two-link leg relation ``L(q) = base - link*cos(q)``."""

    base_length: float
    link_length: float

    @property
    def minimum_length(self) -> float:
        return float(self.base_length - self.link_length)

    @property
    def maximum_length(self) -> float:
        return float(self.base_length + self.link_length)

    def length(self, angle: float) -> float:
        return float(self.base_length - self.link_length * np.cos(float(angle)))

    def angle(self, length: float) -> float:
        normalized = np.clip(
            (self.base_length - float(length)) / self.link_length,
            -1.0,
            1.0,
        )
        return float(np.arccos(normalized))
