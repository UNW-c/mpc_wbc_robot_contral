"""Historical Ruckig position planner; explicit comparison only, not the driving default."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig

from Algorithm.geometry import wrap_angle

_OK = (Result.Working, Result.Finished)


from .planner import MotionCommand, MotionLimits, PlannerMode, ReferenceTrajectory


class _Axis:
    """单自由度在线轨迹生成器。位置以相对误差输入，输出是相对位移。"""

    def __init__(self, max_velocity: float, max_acceleration: float, max_jerk: float) -> None:
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.max_jerk = float(max_jerk)
        self.velocity = 0.0
        self.acceleration = 0.0
        self._input = InputParameter(1)
        self._input.max_velocity = [self.max_velocity]
        self._input.max_acceleration = [self.max_acceleration]
        self._input.max_jerk = [self.max_jerk]
        self._output = OutputParameter(1)
        self._otg: Ruckig | None = None
        self._dt = 0.0

    def reset(self, velocity: float = 0.0) -> None:
        self.velocity = float(np.clip(velocity, -self.max_velocity, self.max_velocity))
        self.acceleration = 0.0

    def sample(
        self,
        error: float | None,
        target_velocity: float,
        dt: float,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``error is None`` 时走速度接口，否则走位置接口（目标速度恒为 0）。

        返回 ``(相对位移, 速度, 加速度)``，长度都是 ``count``，索引 0 即当前时刻。
        """
        dt = float(dt)
        if self._otg is None or dt != self._dt:
            self._otg = Ruckig(1, dt)
            self._dt = dt

        parameters = self._input
        parameters.current_position = [0.0]
        # 实测速度可能超出规划限幅（例如被推了一把），先夹回可行域再喂给 Ruckig
        parameters.current_velocity = [
            float(np.clip(self.velocity, -self.max_velocity, self.max_velocity))
        ]
        parameters.current_acceleration = [
            float(np.clip(self.acceleration, -self.max_acceleration, self.max_acceleration))
        ]
        parameters.target_acceleration = [0.0]
        if error is None:
            parameters.control_interface = ControlInterface.Velocity
            parameters.target_velocity = [
                float(np.clip(target_velocity, -self.max_velocity, self.max_velocity))
            ]
        else:
            parameters.control_interface = ControlInterface.Position
            parameters.target_position = [float(error)]
            parameters.target_velocity = [0.0]

        result = self._otg.update(parameters, self._output)
        if result not in _OK:
            raise RuntimeError(f"Ruckig 轨迹生成失败: {result}")

        trajectory = self._output.trajectory
        offsets = np.empty(count, dtype=float)
        velocities = np.empty(count, dtype=float)
        accelerations = np.empty(count, dtype=float)
        for index in range(count):
            position, velocity, acceleration = trajectory.at_time(index * dt)
            offsets[index] = position[0]
            velocities[index] = velocity[0]
            accelerations[index] = acceleration[0]

        # 只承诺第一步：下一拍会拿新的测量重新规划
        if count > 1:
            self.velocity = float(velocities[1])
            self.acceleration = float(accelerations[1])
        return offsets, velocities, accelerations


class LegacyPositionPlanner:
    """有状态的速度规划器：每个控制周期滚动重规划一次，并给出整条时域参考。

    内部只保存两个轴各自的 ``velocity`` / ``acceleration``。位置参考每拍都重新
    锚定到实测值上，所以不存在积分饱和；位移误差完全通过 ``v*`` 传递。
    """

    def __init__(self, limits: MotionLimits | None = None) -> None:
        self.limits = limits or MotionLimits()
        self._forward = _Axis(
            self.limits.max_speed,
            self.limits.max_acceleration,
            self.limits.max_jerk,
        )
        self._yaw = _Axis(
            self.limits.max_yaw_rate,
            self.limits.max_yaw_acceleration,
            self.limits.max_yaw_jerk,
        )

    @property
    def speed(self) -> float:
        return self._forward.velocity

    @property
    def acceleration(self) -> float:
        return self._forward.acceleration

    @property
    def yaw_rate(self) -> float:
        return self._yaw.velocity

    @property
    def yaw_acceleration(self) -> float:
        return self._yaw.acceleration

    def reset(self, speed: float = 0.0, yaw_rate: float = 0.0) -> None:
        self._forward.reset(speed)
        self._yaw.reset(yaw_rate)

    def _forward_request(
        self,
        command: MotionCommand,
        position: float,
    ) -> tuple[float | None, float]:
        if command.mode is PlannerMode.VELOCITY:
            return None, float(command.velocity or 0.0)
        error = float(command.position) - float(position)
        if abs(error) <= self.limits.position_deadband:
            return None, 0.0  # 死区内改为"刹到静止并保持"
        return error, 0.0

    def _yaw_request(self, command: MotionCommand, yaw: float) -> tuple[float | None, float]:
        if command.yaw_mode is PlannerMode.VELOCITY:
            return None, float(command.yaw_rate or 0.0)
        error = wrap_angle(float(command.heading) - float(yaw))
        if abs(error) <= self.limits.heading_deadband:
            return None, 0.0
        return error, 0.0

    def plan(
        self,
        position: float,
        yaw: float,
        command: MotionCommand,
        dt: float,
        horizon: int,
    ) -> ReferenceTrajectory:
        """展开 ``horizon + 1`` 个采样点的参考轨迹，并把内部状态推进一步。"""
        limits = self.limits
        dt = float(dt)
        count = int(horizon) + 1

        forward_error, forward_velocity = self._forward_request(command, position)
        offsets, speeds, accelerations = self._forward.sample(
            forward_error, forward_velocity, dt, count
        )
        yaw_error, yaw_velocity = self._yaw_request(command, yaw)
        yaw_offsets, yaw_rates, _ = self._yaw.sample(yaw_error, yaw_velocity, dt, count)

        pitch = np.clip(
            np.arctan2(accelerations, limits.gravity),
            -limits.pitch_reference_limit,
            limits.pitch_reference_limit,
        )
        pitch_rate = np.empty_like(pitch)
        if count > 1:
            pitch_rate[:-1] = np.diff(pitch) / max(dt, 1.0e-9)
            pitch_rate[-1] = pitch_rate[-2] if count > 2 else pitch_rate[0]
        else:
            pitch_rate[0] = 0.0

        return ReferenceTrajectory(
            position=float(position) + offsets,
            velocity=speeds,
            acceleration=accelerations,
            pitch=pitch,
            pitch_rate=pitch_rate,
            yaw=float(yaw) + yaw_offsets,
            yaw_rate=yaw_rates,
        )
