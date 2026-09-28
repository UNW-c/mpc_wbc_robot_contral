"""Exclusive, bounded compensation routing; no blind summation of observers."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .validation import positive, vector


class DisturbanceSpace(str, Enum):
    REDUCED_ACCELERATION = "forward_pitch_yaw_acceleration"
    GENERALIZED_FORCE = "generalized_force"


@dataclass(frozen=True)
class DisturbanceEstimate:
    value: np.ndarray
    space: DisturbanceSpace
    source: str
    timestamp: float
    context: int
    valid: bool

    def __post_init__(self):
        values = np.asarray(self.value, dtype=float)
        if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError("Estimate must be a finite nonempty vector")
        if not np.isfinite(self.timestamp) or not self.source:
            raise ValueError("Estimate requires a finite timestamp and source")
        if not isinstance(self.space, DisturbanceSpace):
            raise ValueError("Explicit disturbance space required")
        if self.space == DisturbanceSpace.REDUCED_ACCELERATION and len(values) != 3:
            raise ValueError("Reduced acceleration order is forward, pitch, yaw")
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "value", values)


class CompensationRouter:
    """Choose one source and one layer for a context; stale data withdraws immediately.

    RLS is not automatically added to ESO/momentum. To use RLS + residual ESO,
    callers must build one combined estimate after including the SAME frozen RLS
    model in the ESO known input. Changing that model requires a context reset.
    """

    def __init__(self, source: str, space: DisturbanceSpace, bounds, slew_rates, max_age: float):
        if not source or not isinstance(space, DisturbanceSpace):
            raise ValueError("A named source and explicit space are required")
        self.source, self.space = source, space
        self.size = len(bounds)
        if self.size < 1 or (space == DisturbanceSpace.REDUCED_ACCELERATION and self.size != 3):
            raise ValueError("Invalid compensation dimension")
        self.bounds = vector(bounds, self.size, "bounds")
        self.slew_rates = vector(slew_rates, self.size, "slew_rates")
        if np.any(self.bounds < 0) or np.any(self.slew_rates <= 0):
            raise ValueError("bounds must be nonnegative and slew rates positive")
        self.max_age = positive(max_age, "max_age")
        self.reset()

    def reset(self):
        self._value = np.zeros(self.size)
        self._context = None
        self._last_time = None

    def route(self, estimate: DisturbanceEstimate | None, *, timestamp: float, dt: float,
              context: int, enabled: bool) -> np.ndarray:
        dt = positive(dt, "dt")
        if not np.isfinite(timestamp) or (self._last_time is not None and timestamp <= self._last_time):
            raise ValueError("Compensation clock must be finite and strictly increasing")
        if context != self._context:
            self._value[:] = 0.0
        self._context, self._last_time = context, timestamp
        if estimate is not None and (estimate.source != self.source or estimate.space != self.space):
            self._value[:] = 0.0
            raise ValueError("Compensation source/space mismatch; do not stack observers")
        if (not enabled or estimate is None or not estimate.valid or estimate.context != context
                or not 0 <= timestamp - estimate.timestamp <= self.max_age):
            self._value[:] = 0.0
            return self._value.copy()
        target = np.clip(vector(estimate.value, self.size, "estimate"), -self.bounds, self.bounds)
        self._value += np.clip(target - self._value, -self.slew_rates * dt, self.slew_rates * dt)
        return self._value.copy()
