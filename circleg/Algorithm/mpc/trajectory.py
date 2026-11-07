"""MPC 优化轨迹的时间采样接口，不依赖控制器或仿真运行时。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import INPUT_DIM, STATE_DIM

_POSITION_COLUMNS = (0, 2, 4)
_VELOCITY_COLUMNS = (1, 3, 5)
_TIME_TOLERANCE = 1e-10


@dataclass(frozen=True)
class MotionSample:
    """同一时刻的状态、加速度和等效轮力；状态顺序沿用 ReducedModel。"""

    state: np.ndarray
    acceleration: np.ndarray  # 位置通道加速度（6 状态：forward/pitch/yaw；8 状态再加共模髋）
    force: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class OptimizedTrajectory:
    """优化后的节点采用三次 Hermite 插值，位置与速度及加速度相互一致。

    力在每个 MPC 区间内零阶保持。插值匹配优化节点的位置和速度；区间内部
    是连续轨迹近似，不宣称再次满足所有 QP 约束。禁止在有效期外静默外推。
    """

    states: np.ndarray
    forces: np.ndarray
    dt: float
    start_time: float = 0.0
    # B3：由模型提供的维度与插值列（6 状态默认 (0,2,4)/(1,3,5)；8 状态为 (0,2,4,6)/(1,3,5,7)）。
    state_dim: int = STATE_DIM
    input_dim: int = INPUT_DIM
    position_columns: tuple = _POSITION_COLUMNS
    velocity_columns: tuple = _VELOCITY_COLUMNS

    def __post_init__(self) -> None:
        states = np.array(self.states, dtype=float, copy=True)
        forces = np.array(self.forces, dtype=float, copy=True)
        if forces.ndim != 2 or forces.shape[1] != self.input_dim or len(forces) < 1:
            raise ValueError("轨迹必须包含至少一个双轮力采样")
        if states.shape != (len(forces) + 1, self.state_dim):
            raise ValueError("状态节点数必须为力采样数加一，且包含六个状态")
        if not np.isfinite(self.dt) or self.dt <= 0.0 or not np.isfinite(self.start_time):
            raise ValueError("轨迹时间参数必须有限，周期必须为正")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(forces)):
            raise ValueError("轨迹必须包含有限数值")
        states.setflags(write=False)
        forces.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "forces", forces)

    @property
    def duration(self) -> float:
        return len(self.forces) * self.dt

    @property
    def valid_until(self) -> float:
        return self.start_time + self.duration

    def sample(self, elapsed: float) -> MotionSample:
        """按相对轨迹起点的时间采样；角度节点须使用连续、未折返的表示。"""
        elapsed = float(elapsed)
        if not np.isfinite(elapsed) or elapsed < -_TIME_TOLERANCE or elapsed > self.duration + _TIME_TOLERANCE:
            raise ValueError("采样时间超出轨迹有效期")
        elapsed = float(np.clip(elapsed, 0.0, self.duration))
        index = min(int(np.floor(elapsed / self.dt + _TIME_TOLERANCE)), len(self.forces) - 1)
        u = float(np.clip((elapsed - index * self.dt) / self.dt, 0.0, 1.0))
        p0 = self.states[index, list(self.position_columns)]
        p1 = self.states[index + 1, list(self.position_columns)]
        v0 = self.states[index, list(self.velocity_columns)]
        v1 = self.states[index + 1, list(self.velocity_columns)]
        # 在归一化区间构造多项式；同一组系数的导数用于所有前馈量。
        a = 2.0 * (p0 - p1) + self.dt * (v0 + v1)
        b = 3.0 * (p1 - p0) - self.dt * (2.0 * v0 + v1)
        c = self.dt * v0
        position = ((a * u + b) * u + c) * u + p0
        velocity = ((3.0 * a * u + 2.0 * b) * u + c) / self.dt
        acceleration = (6.0 * a * u + 2.0 * b) / self.dt**2
        state = np.empty(self.state_dim)
        state[list(self.position_columns)] = position
        state[list(self.velocity_columns)] = velocity
        return MotionSample(state, acceleration, self.forces[index].copy(), self.start_time + elapsed)

    def sample_at(self, timestamp: float) -> MotionSample:
        return self.sample(float(timestamp) - self.start_time)
