"""Bounded scalar-output RLS with explicit learning permission and excitation.

This estimates a chosen regression y = phi.T theta. It cannot distinguish an
unknown external force from a parameter error with the same regression signature.
Callers must freeze learning during contact changes, slip and unknown disturbances.
"""
from dataclasses import dataclass
import numpy as np
from .validation import positive, vector, spd


@dataclass(frozen=True)
class RLSConfig:
    forgetting: float
    measurement_variance: float
    excitation_floor: float
    excitation_decay: float

    def __post_init__(self):
        positive(self.forgetting, "forgetting")
        positive(self.measurement_variance, "measurement_variance")
        positive(self.excitation_floor, "excitation_floor")
        positive(self.excitation_decay, "excitation_decay")
        if self.forgetting > 1 or self.excitation_decay > 1:
            raise ValueError("forgetting and excitation_decay must be <= 1")


class RecursiveLeastSquares:
    def __init__(self, initial, covariance, lower, upper, config: RLSConfig):
        self.size = len(initial)
        if self.size < 1:
            raise ValueError("RLS requires at least one parameter")
        self.initial = vector(initial, self.size, "initial")
        self.initial_covariance = spd(covariance, self.size, "covariance")
        self.lower = vector(lower, self.size, "lower")
        self.upper = vector(upper, self.size, "upper")
        if np.any(self.lower > self.upper) or np.any(self.initial < self.lower) or np.any(self.initial > self.upper):
            raise ValueError("Invalid parameter bounds or initial parameters")
        self.config = config
        self.reset()

    def reset(self):
        self._theta = self.initial.copy()
        self._covariance = self.initial_covariance.copy()
        self._information = np.zeros((self.size, self.size))
        self.updated = False
        self.projected = False

    @property
    def parameters(self):
        return self._theta.copy()

    @property
    def covariance(self):
        return self._covariance.copy()

    @property
    def excited(self):
        return bool(np.linalg.eigvalsh(self._information)[0] >= self.config.excitation_floor)

    @property
    def usable(self):
        return self.excited and not self.projected

    def predict(self, features):
        return float(vector(features, self.size, "features") @ self._theta)

    def update(self, features, measurement: float, *, learning_allowed: bool) -> float:
        phi = vector(features, self.size, "features")
        if not np.isfinite(measurement):
            raise ValueError("measurement must be finite")
        self.updated = False
        self._information *= self.config.excitation_decay
        if not learning_allowed or np.linalg.norm(phi) <= np.finfo(float).eps:
            return self.predict(phi)
        self._information += np.outer(phi, phi) / self.config.measurement_variance
        # Collect excitation before adapting; do not inflate covariance while frozen.
        if not self.excited:
            return self.predict(phi)
        prior = self._covariance / self.config.forgetting
        variance = self.config.measurement_variance
        gain = prior @ phi / (variance + phi @ prior @ phi)
        candidate = self._theta + gain * (measurement - phi @ self._theta)
        bounded = np.clip(candidate, self.lower, self.upper)
        self.projected = bool(np.any(candidate != bounded))
        self._theta = bounded
        # Joseph form preserves positive definiteness under finite precision.
        residual_map = np.eye(self.size) - np.outer(gain, phi)
        self._covariance = residual_map @ prior @ residual_map.T + variance * np.outer(gain, gain)
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        self.updated = True
        return self.predict(phi)
