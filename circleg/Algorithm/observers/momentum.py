"""Generalized momentum residual, without measured acceleration.

For M vdot + h = tau_applied + tau_contact + tau_external,
p=Mv and pdot = tau_applied + tau_contact - h + Mdot*v + tau_external.
Mdot*v and all forces must use the SAME generalized-velocity basis as M and v.
Known contact forces must come from an independent trusted measurement/estimate;
reusing forces inferred by imposing zero external force makes this circular.
applied_force/contact_force are MEAN forces over the preceding sample interval;
the bias and Mdot*v are endpoint values, integrated by the trapezoid rule.
"""
import numpy as np
from .validation import positive, vector, spd


class GeneralizedMomentumObserver:
    def __init__(self, gains, warmup: float):
        self.size = len(gains)
        if self.size < 1:
            raise ValueError("Momentum observer requires positive dimension")
        self.gains = vector(gains, self.size, "gains")
        if np.any(self.gains <= 0):
            raise ValueError("gains must be positive")
        self.warmup = positive(warmup, "warmup")
        self.reset()

    def reset(self):
        self._previous_momentum = None
        self._previous_bias_rate = None
        self._context = None
        self._residual = np.zeros(self.size)
        self._elapsed = 0.0
        self.valid = False

    @property
    def residual(self):
        return self._residual.copy()

    def update(self, mass, velocity, applied_force, dynamics_bias, mass_rate_velocity,
               contact_force, dt: float, *, trusted: bool, contact_known: bool, context=0):
        matrix = spd(mass, self.size, "mass")
        v = vector(velocity, self.size, "velocity")
        applied = vector(applied_force, self.size, "applied_force")
        h = vector(dynamics_bias, self.size, "dynamics_bias")
        mass_rate = vector(mass_rate_velocity, self.size, "mass_rate_velocity")
        contact = vector(contact_force, self.size, "contact_force")
        dt = positive(dt, "dt")
        if not trusted or not contact_known:
            self.reset()
            return self.residual
        momentum = matrix @ v
        bias_rate = mass_rate - h
        if self._previous_momentum is None or context != self._context:
            self.reset()
            self._previous_momentum = momentum
            self._previous_bias_rate = bias_rate
            self._context = context
            return self.residual
        # Applied torque is usually held, not linearly interpolated across updates.
        # Trapezoid-averaging old/new commands invents impulses at command steps.
        known_impulse = dt * (applied + contact) + 0.5 * dt * (bias_rate + self._previous_bias_rate)
        innovation = momentum - self._previous_momentum - known_impulse
        # Implicit integration of residual feedback is stable for every positive dt.
        self._residual = (self._residual + self.gains * innovation) / (1.0 + self.gains * dt)
        self._previous_momentum = momentum
        self._previous_bias_rate = bias_rate
        self._elapsed += dt
        self.valid = self._elapsed + 1e-12 >= self.warmup
        return self.residual
