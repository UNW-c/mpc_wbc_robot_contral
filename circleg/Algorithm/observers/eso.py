"""Discrete linear ESO for velocity channels: v_dot = a_known + d, d_dot = 0.

Inputs are measured velocities and the known mean acceleration over the elapsed
interval, computed from applied actuation, not the requested unsaturated command.
The correction places both observer poles at exp(-bandwidth * dt); no Euler
stability restriction or numerical acceleration measurement is required.
"""
import numpy as np
from .validation import vector, positive


class VelocityESO:
    def __init__(self, size: int, bandwidth: float, warmup: float):
        if size < 1 or int(size) != size:
            raise ValueError("size must be a positive integer")
        self.size = int(size)
        self.bandwidth = positive(bandwidth, "bandwidth")
        self.warmup = positive(warmup, "warmup")
        self.reset()

    def reset(self):
        self._velocity = np.zeros(self.size)
        self._disturbance = np.zeros(self.size)
        self._initialized = False
        self._elapsed = 0.0
        self._context = None
        self.valid = False

    @property
    def disturbance(self):
        return self._disturbance.copy()

    @property
    def velocity(self):
        return self._velocity.copy()

    def update(self, measured_velocity, known_acceleration, dt: float, *,
               trusted: bool, context=0):
        measured = vector(measured_velocity, self.size, "measured_velocity")
        acceleration = vector(known_acceleration, self.size, "known_acceleration")
        dt = positive(dt, "dt")
        if not trusted:
            self.reset()
            return self.disturbance
        if not self._initialized or context != self._context:
            self.reset()
            self._velocity = measured
            self._context = context
            self._initialized = True
            return self.disturbance
        pole = np.exp(-self.bandwidth * dt)
        prediction = self._velocity + dt * (acceleration + self._disturbance)
        innovation = measured - prediction
        self._velocity = prediction + (1.0 - pole**2) * innovation
        self._disturbance += (1.0 - pole)**2 / dt * innovation
        self._elapsed += dt
        self.valid = self._elapsed + 1e-12 >= self.warmup
        return self.disturbance
