"""Shadow RLS plus residual ESO with explicit model commit boundaries."""
import numpy as np
from .compensation import DisturbanceEstimate, DisturbanceSpace
from .eso import VelocityESO
from .rls import RecursiveLeastSquares
from .validation import vector


class AdaptiveAccelerationObserver:
    """Three acceleration channels [forward,pitch,yaw], with arbitrary regressors.

    RLS learns only when explicitly permitted and never changes the active model
    mid-interval. Committing a trained model resets ESO: its residual was relative
    to the OLD model. The published value adds frozen RLS and ESO residual once.
    Identifiability of the chosen features remains the caller's responsibility.
    """
    SOURCE = "rls_residual_eso"

    def __init__(self, regressions: tuple[RecursiveLeastSquares, ...], eso: VelocityESO):
        if len(regressions) != 3 or eso.size != 3:
            raise ValueError("Exactly three reduced acceleration channels required")
        if len({id(r) for r in regressions}) != 3:
            raise ValueError("Each channel requires an independent RLS instance")
        self.regressions, self.eso = regressions, eso
        self._active = tuple(r.parameters for r in regressions)
        self.revision = 0

    def _features(self, features):
        if len(features) != 3:
            raise ValueError("Three channel regressors required")
        return tuple(vector(phi, r.size, "features") for phi, r in zip(features, self.regressions))

    def learn(self, features, measured_residual_acceleration, *, learning_allowed: bool):
        rows = self._features(features)
        measurement = vector(measured_residual_acceleration, 3, "measured_residual_acceleration")
        for regression, phi, value in zip(self.regressions, rows, measurement):
            regression.update(phi, value, learning_allowed=learning_allowed)

    def commit(self, *, steady_supported: bool):
        if not steady_supported or not all(r.usable for r in self.regressions):
            return False
        self._active = tuple(r.parameters for r in self.regressions)
        self.revision += 1
        self.eso.reset()
        return True

    def update(self, measured_velocity, nominal_interval_acceleration, interval_features,
               current_features, dt, *, timestamp: float, context: int, trusted: bool):
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        interval_rows = self._features(interval_features)
        current_rows = self._features(current_features)
        nominal = vector(nominal_interval_acceleration, 3, "nominal_interval_acceleration")
        slow_interval = np.array([phi @ theta for phi, theta in zip(interval_rows, self._active)])
        slow_current = np.array([phi @ theta for phi, theta in zip(current_rows, self._active)])
        residual = self.eso.update(measured_velocity, nominal + slow_interval, dt,
                                   trusted=trusted, context=(context, self.revision))
        return DisturbanceEstimate(slow_current + residual, DisturbanceSpace.REDUCED_ACCELERATION,
                                   self.SOURCE, timestamp, context, self.eso.valid and trusted)
