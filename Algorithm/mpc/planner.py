"""Velocity commands, physical acceleration/braking ramps and persistent travel references.

Position commands belong to legacy_planner.LegacyPositionPlanner only. This ramp
bounds acceleration, not jerk. Reference generation is independent of MuJoCo.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np
from Algorithm.geometry import wrap_angle

_LOGGER = logging.getLogger(__name__)


class PlannerMode(str, Enum):
    """指令语义。"""

    POSITION = "position"
    VELOCITY = "velocity"


@dataclass(frozen=True)
class MotionLimits:
    """物理加速/制动限值与参考状态机阈值，尚未针对真实机器人调参。

    max_jerk、max_yaw_jerk、position_deadband、heading_deadband、
    pitch_reference_limit、gravity 仅供显式旧规划器使用。
    当前斜坡不承诺 jerk 连续；姿态软目标采用 equilibrium_pitch。
    """

    max_speed: float = 4.00  # 2026-09-28 用户定稿：键鼠可控最大速度 4.0 m/s
    # 2026-09-28 用户确认（C10）：按 hip_angle_limit = 0.55 时约 1.4 m/s^2 的持续能力对齐。
    # 依据：K1 实测刹车段 ramp_accel 比值 0.69（规划 2.00 只做到 1.39）、髋角触碰限值 34.8%。
    max_acceleration: float = 1.40
    # 起步斜坡：原 0.80 偏保守（实测能到 1.17 倍），与持续能力对齐后提到 1.2（留约 15% 余量）。
    commanded_acceleration: float = 1.20
    max_braking: float = 1.40
    max_jerk: float = 8.00
    max_yaw_rate: float = 2.00
    max_yaw_acceleration: float = 6.00
    max_yaw_braking: float = 6.00
    max_yaw_jerk: float = 30.0
    position_deadband: float = 0.005
    heading_deadband: float = 0.017
    pitch_reference_limit: float = 0.25
    gravity: float = 9.81
    velocity_deadband: float = 0.005
    yaw_rate_deadband: float = 0.01
    hold_speed: float = 0.02
    hold_yaw_rate: float = 0.03
    hold_pitch: float = 0.10
    hold_pitch_rate: float = 0.15
    hold_dwell: float = 0.20
    touchdown_dwell: float = 0.10
    position_error_limit: float = 0.50
    yaw_error_limit: float = 0.70
    residual_limit: float = 0.50
    saturation_dwell: float = 0.10
    capability_recovery: float = 0.50
    equilibrium_pitch: float = 0.0

    def __post_init__(self) -> None:
        for name in ("max_speed", "max_acceleration", "commanded_acceleration", "max_braking",
                     "max_yaw_rate",
                     "max_yaw_acceleration", "max_yaw_braking", "hold_dwell",
                     "touchdown_dwell", "position_error_limit", "yaw_error_limit",
                     "residual_limit", "saturation_dwell", "capability_recovery"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("velocity_deadband", "yaw_rate_deadband", "hold_speed",
                     "hold_yaw_rate", "hold_pitch", "hold_pitch_rate"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.equilibrium_pitch):
            raise ValueError("equilibrium_pitch must be finite")


@dataclass(frozen=True)
class MotionCommand:
    """驾驶入口只接受 velocity 和 yaw_rate。

    position/heading 仅供 LegacyPositionPlanner 的历史对照测试；
    VelocityPlanner 和默认 WheelMPC 会明确拒绝这些绝对位置输入。
    """

    position: float | None = None
    velocity: float | None = None
    heading: float | None = None
    yaw_rate: float | None = None

    @property
    def mode(self) -> PlannerMode:
        return PlannerMode.VELOCITY if self.position is None else PlannerMode.POSITION

    @property
    def yaw_mode(self) -> PlannerMode:
        return PlannerMode.VELOCITY if self.heading is None else PlannerMode.POSITION


@dataclass(frozen=True)
class ReferenceTrajectory:
    """时域上的参考量，长度都是 ``horizon + 1``。"""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    pitch: np.ndarray
    pitch_rate: np.ndarray
    yaw: np.ndarray
    yaw_rate: np.ndarray

    @property
    def horizon(self) -> int:
        return int(self.position.size) - 1

    def states(self, state_dim: int = 6) -> np.ndarray:
        """按 ``[s, v, pitch, pitch_rate, yaw, yaw_rate]`` 拼成 (N+1, 6) 状态参考。

        ``state_dim > 6`` 时补零列（共模髋角的参考为 0，让 MPC 自己权衡用髋还是用倾角，B3）。
        planner 保持与具体模型无关，默认 6 列。
        """
        columns = [
            self.position,
            self.velocity,
            self.pitch,
            self.pitch_rate,
            self.yaw,
            self.yaw_rate,
        ]
        if state_dim > 6:
            zeros = np.zeros_like(self.position)
            columns = columns + [zeros, zeros]
        return np.column_stack(columns[:state_dim])


class ReferenceMode(str, Enum):
    RESET = "reset"
    DRIVE = "drive"
    BRAKE = "brake"
    HOLD = "hold"


@dataclass(frozen=True)
class ReferenceFeedback:
    """Final motion residuals and contact trust, never a task increment scale."""

    contact: tuple[bool, bool] = (True, True)
    enabled: bool = True
    command_fresh: bool = True
    slipping: bool = False
    saturated: bool = False
    motion_residual: float = 0.0
    pitch: float = 0.0
    pitch_rate: float = 0.0


@dataclass(frozen=True)
class _RampState:
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0

    def advance(self, target: float, dt: float, acceleration: float, braking: float) -> _RampState:
        """Integrate exactly across zero and target-speed events within one step."""
        position, velocity, remaining = self.position, self.velocity, dt
        # At most two events: stop before reversal, then reach the new target.
        for _ in range(3):
            if remaining <= 1e-14:
                break
            endpoint = 0.0 if velocity * target < 0.0 else target
            delta = endpoint - velocity
            if abs(delta) <= 1e-14:
                position += velocity * remaining
                remaining = 0.0
                break
            limit = braking if abs(endpoint) < abs(velocity) else acceleration
            rate = np.copysign(limit, delta)
            elapsed = min(remaining, abs(delta) / limit)
            position += velocity * elapsed + 0.5 * rate * elapsed**2
            velocity += rate * elapsed
            remaining -= elapsed
            if abs(velocity - endpoint) < 1e-12:
                velocity = endpoint
        endpoint = 0.0 if velocity * target < 0.0 else target
        delta = endpoint - velocity
        rate = 0.0 if abs(delta) < 1e-12 else np.copysign(
            braking if abs(endpoint) < abs(velocity) else acceleration, delta)
        return _RampState(float(position), float(velocity), float(rate))


class VelocityPlanner:
    """One plan call commits dt, while the horizon only advances copied state.

    Call once per MPC tick. Measurements initialize/reset the persistent reference;
    HOLD captures its anchor once, preserving a restoring error after a push.
    """

    def __init__(self, limits: MotionLimits | None = None) -> None:
        self.limits = limits or MotionLimits()
        self.reset()

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

    def reset(self, speed: float = 0.0, yaw_rate: float = 0.0, *,
              position: float | None = None, yaw: float | None = None) -> None:
        if not np.all(np.isfinite([speed, yaw_rate, position or 0.0, yaw or 0.0])):
            raise ValueError("Reference reset must be finite")
        self._forward = _RampState(position or 0.0, speed)
        self._yaw = _RampState(yaw or 0.0, yaw_rate)
        self._initialized = position is not None and yaw is not None
        self.mode = ReferenceMode.RESET
        self.valid = False
        self.last_reset_reason = "enable"
        self.reset_count = 0
        self.capability = 1.0
        self._hold_time = self._touchdown_time = self._saturation_time = 0.0
        self._suspended = False

    def _align(self, position: float, yaw: float, speed: float, yaw_rate: float, reason: str,
               reset_position: bool = True) -> None:
        # reset_position=False keeps the travel target where it was: a pure
        # velocity hold must not move the position goal away from the origin.
        self._forward = _RampState(position if reset_position else self._forward.position, speed)
        self._yaw = _RampState(yaw, yaw_rate)
        self._initialized = True
        self._hold_time = 0.0
        self.mode = ReferenceMode.RESET
        self.last_reset_reason = reason
        self.reset_count += 1
        _LOGGER.info("Reference reset: %s, s=%.4f v=%.4f yaw=%.4f reset_position=%s",
                     reason, self._forward.position, speed, yaw, reset_position)

    def plan(self, position: float, yaw: float, command: MotionCommand, dt: float,
             horizon: int, *, measured_speed: float | None = None,
             measured_yaw_rate: float | None = None,
             feedback: ReferenceFeedback | None = None, lead: float = 0.0) -> ReferenceTrajectory:
        if command.position is not None or command.heading is not None:
            raise ValueError("Driving accepts velocity/yaw_rate only; use LegacyPositionPlanner explicitly for position research")
        speed = self.speed if measured_speed is None else measured_speed
        yaw_rate = self.yaw_rate if measured_yaw_rate is None else measured_yaw_rate
        v_cmd = 0.0 if command.velocity is None else command.velocity
        w_cmd = 0.0 if command.yaw_rate is None else command.yaw_rate
        fb = feedback or ReferenceFeedback()
        if not np.all(np.isfinite([position, yaw, speed, yaw_rate, dt, lead, v_cmd, w_cmd,
                                  fb.pitch, fb.pitch_rate, fb.motion_residual])):
            raise ValueError("Reference inputs must be finite")
        if dt <= 0 or lead < 0 or horizon < 1 or int(horizon) != horizon:
            raise ValueError("dt/horizon must be positive and lead nonnegative")
        limits = self.limits
        if not self._initialized:
            self._align(position, yaw, speed, yaw_rate, "enable")
        # Compare yaw near the unwrapped internal reference without wrapping the reference itself.
        measured_yaw = self._yaw.position + wrap_angle(yaw - self._yaw.position)
        if not fb.enabled or not fb.command_fresh or not any(fb.contact):
            if not self._suspended:
                self.last_reset_reason = "disabled" if not fb.enabled else "stale_command" if not fb.command_fresh else "airborne"
                _LOGGER.info("Ground reference suspended: %s", self.last_reset_reason)
            self._suspended = True
            self._touchdown_time = 0.0
        elif self._suspended:
            self._touchdown_time = self._touchdown_time + dt if all(fb.contact) and not fb.slipping else 0.0
            if self._touchdown_time + 1e-12 >= limits.touchdown_dwell:
                self._align(position, measured_yaw, speed, yaw_rate, "stable_resume")
                self._suspended = False
        self.valid = not self._suspended
        if not self.valid:
            self.mode = ReferenceMode.RESET
            n = int(horizon) + 1
            zero = np.zeros(n)
            return ReferenceTrajectory(np.full(n, self._forward.position), zero.copy(), zero.copy(),
                                       np.full(n, limits.equilibrium_pitch), zero.copy(),
                                       np.full(n, self._yaw.position), zero.copy())
        blocked = fb.saturated and abs(fb.motion_residual) > limits.residual_limit
        self._saturation_time = self._saturation_time + dt if blocked else 0.0
        restricted = fb.slipping or not all(fb.contact) or self._saturation_time >= limits.saturation_dwell
        if restricted:
            self.capability = 0.0 if fb.slipping else min(self.capability, 0.25)
        else:
            self.capability = min(1.0, self.capability + limits.capability_recovery * dt)
        if abs(wrap_angle(self._yaw.position - measured_yaw)) > limits.yaw_error_limit:
            self._align(position, measured_yaw, speed, yaw_rate, "tracking_error_limit",
                        reset_position=False)
            self.capability = 0.0
        target_v = np.clip(v_cmd, -limits.max_speed, limits.max_speed) if abs(v_cmd) > limits.velocity_deadband else 0.0
        target_w = np.clip(w_cmd, -limits.max_yaw_rate, limits.max_yaw_rate) if abs(w_cmd) > limits.yaw_rate_deadband else 0.0
        driving = bool(target_v or target_w)
        if driving:
            self.mode = ReferenceMode.DRIVE
            self._hold_time = 0.0
            # The travel target follows the measurement while driving, so the
            # unavoidable acceleration/braking lag is not read as an error.
            self._forward = _RampState(position, self._forward.velocity)
        elif self.mode != ReferenceMode.HOLD:
            self.mode = ReferenceMode.BRAKE
            stopped = (abs(self.speed) < 1e-12 and abs(self.yaw_rate) < 1e-12
                       and abs(speed) <= limits.hold_speed and abs(yaw_rate) <= limits.hold_yaw_rate
                       and abs(fb.pitch - limits.equilibrium_pitch) <= limits.hold_pitch
                       and abs(fb.pitch_rate) <= limits.hold_pitch_rate
                       and all(fb.contact) and not fb.slipping)
            self._hold_time = self._hold_time + dt if stopped else 0.0
            if self._hold_time + 1e-12 >= limits.hold_dwell:
                self._align(position, measured_yaw, 0.0, 0.0, "hold_capture",
                            reset_position=False)
                self.mode = ReferenceMode.HOLD
        target_v *= self.capability
        target_w *= self.capability
        # A commanded drive uses the gentler ramp so the starting lean stays small;
        # the zero-command hold and recovery keep the original acceleration limit.
        acceleration = limits.commanded_acceleration if driving else limits.max_acceleration
        forward, turning = self._forward, self._yaw
        self._forward = forward.advance(target_v, dt, acceleration, limits.max_braking)
        self._yaw = turning.advance(target_w, dt, limits.max_yaw_acceleration, limits.max_yaw_braking)
        forward = forward.advance(target_v, lead, acceleration, limits.max_braking)
        turning = turning.advance(target_w, lead, limits.max_yaw_acceleration, limits.max_yaw_braking)
        rows = []
        for _ in range(int(horizon) + 1):
            rows.append((forward.position, forward.velocity, forward.acceleration, turning.position, turning.velocity))
            forward = forward.advance(target_v, dt, acceleration, limits.max_braking)
            turning = turning.advance(target_w, dt, limits.max_yaw_acceleration, limits.max_yaw_braking)
        values = np.asarray(rows)
        return ReferenceTrajectory(values[:, 0], values[:, 1], values[:, 2],
                                   np.full(len(rows), limits.equilibrium_pitch), np.zeros(len(rows)),
                                   values[:, 3], values[:, 4])


