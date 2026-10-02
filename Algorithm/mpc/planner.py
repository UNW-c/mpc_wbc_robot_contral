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

    max_speed: float = 2.50  # 2026-09-29 用户定稿：键鼠可控最大速度 2.5 m/s
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
    # A2（第 15 次排查）：S 曲线开关。False = 旧行为（只限加速度，加速度阶跃）。
    # 开启时加速度变化率不超过 max_jerk；max_jerk 由 WheelMPC 按起步限值 / jerk_ramp_time 写回。
    use_jerk_limit: bool = False
    # B2（第 24 次排查）：刹车段（速度幅值减小）单独的 jerk 上限；None = 与 max_jerk 相同（旧行为）。
    max_braking_jerk: float | None = None
    # 方案 B（第 27 次排查）：零指令（未经速度指令驱动）时的刹车 / jerk 限值；None = 与上面相同。
    # 由 WheelMPC 按 hip_angle_limit 写回，使零指令站立与受扰恢复的规划与分模式之前一致。
    hold_braking: float | None = None
    hold_jerk: float | None = None
    hold_braking_jerk: float | None = None
    # R2'（第 28 次排查）：加速度与目标方向相反（例如加速中途松键）时，先用 max_jerk 把加速度
    # 卸到 0，再进入刹车 jerk，避免参考加速度在一拍内大幅翻转。False = 旧行为。
    unwind_before_brake: bool = False
    # P1（第 31 次排查）：指令刹车期间位置参考跟随实测位置，使"减速段多滑出去的位移"不计入
    # 位置误差，避免停住后被位置环拉回（回弹）。False = 旧行为（位置参考冻在规划停点）。
    reanchor_position_while_braking: bool = False

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
        if self.max_braking_jerk is not None and (not np.isfinite(self.max_braking_jerk)
                                                   or self.max_braking_jerk <= 0):
            raise ValueError("max_braking_jerk must be finite and positive")
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

    def advance_jerk(self, target: float, dt: float, acceleration: float, braking: float,
                     jerk: float, substeps: int = 20, braking_jerk: float | None = None,
                     unwind: bool = False) -> _RampState:
        """A2（第 15 次排查）：加速度以 ``jerk`` 为速率变化的 S 曲线斜坡。

        每个子步内 jerk 为常数，取值由“下一子步结束后还能否按 ``jerk`` 把加速度减到 0
        而不越过目标”精确决定（解一元二次方程，不靠启发式判据）：

        - 期望加速度 ``a_des``：朝最终目标 ``target`` 的方向取限值（速度幅值减小时用
          ``braking``，否则用 ``acceleration``）；反向时加速度不在 0 速处回落，连续穿越。
        - 收尾：设子步末加速度为 a1，其后按 jerk 减到 0 还会多出 ``a1²/(2·jerk)`` 的速度增量。
          取满足 ``子步内速度增量 + a1²/(2·jerk) ≤ |剩余速度差|`` 的最大 a1，再与 a_des、
          jerk 限幅取交集，保证不过冲。
        - 吸附：速度差 < 1e-5 m/s 且加速度在一个 jerk 子步以内时直接落到目标。
        - B2：``braking_jerk`` 给出时，速度幅值减小的子步改用它（None = 与 ``jerk`` 相同）。
        - R2'：``unwind=True`` 时，若当前加速度与朝目标的方向相反，该子步用 ``jerk``
          （而不是 ``braking_jerk``）把加速度往 0 卸，过零后再按速度幅值选 jerk。
        """
        position, velocity, accel = self.position, self.velocity, self.acceleration
        h = float(dt) / int(substeps)
        drive_jerk = float(jerk)
        brake_jerk = drive_jerk if braking_jerk is None else float(braking_jerk)
        for _ in range(int(substeps)):
            remaining = target - velocity
            jerk = brake_jerk if (velocity * target < 0.0 or abs(target) < abs(velocity)) else drive_jerk
            if unwind and accel * remaining < 0.0:
                jerk = drive_jerk
            # 收尾段按二次方程逼近时残差呈几何衰减，1e-5 m/s 以内直接吸附（物理上可忽略）
            if abs(remaining) < 1e-5 and abs(accel) < jerk * h + 1e-12:
                # 到达：加速度最后一步降到 0，速度吸附到目标
                position += velocity * h + 0.5 * accel * h * h
                velocity, accel = float(target), 0.0
                continue
            sign = 1.0 if remaining > 0.0 else -1.0
            speed_down = velocity * target < 0.0 or abs(target) < abs(velocity)
            a_des = sign * (braking if speed_down else acceleration)
            # 在 sign 方向上工作：u = sign*a，r = |remaining|
            u0, r = sign * accel, abs(remaining)
            # 子步内常 jerk：u1 = u0 + j*h，速度增量 (u0+u1)/2*h；收尾需要 u1²/(2J)
            # 约束：(u0+u1)/2*h + max(u1,0)²/(2J) ≤ r  →  u1²/(2J) + u1*h/2 + (u0*h/2 - r) ≤ 0
            c = u0 * h / 2.0 - r
            disc = (h / 2.0) ** 2 - 4.0 * (1.0 / (2.0 * jerk)) * c
            u1_brake = (-(h / 2.0) + np.sqrt(max(disc, 0.0))) / (2.0 * (1.0 / (2.0 * jerk)))
            u1 = min(sign * a_des, u1_brake, u0 + jerk * h)
            u1 = max(u1, u0 - jerk * h)
            accel_next = sign * u1
            j = (accel_next - accel) / h
            position += velocity * h + 0.5 * accel * h * h + j * h ** 3 / 6.0
            velocity += accel * h + 0.5 * j * h * h
            accel = accel_next
        return _RampState(float(position), float(velocity), float(accel))


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
    def commanded_motion(self) -> bool:
        """方案 B：True = 正在执行速度指令，或执行指令后的刹车（尚未进入 HOLD / 复位）。"""
        return bool(getattr(self, "_commanded", False))

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
        self._commanded = False

    def soft_reanchor(self, position: float, yaw: float) -> None:
        """R1'：只把位置 / 偏航重锚到实测值，保留斜坡速度、加速度和 commanded 标记。"""
        if not np.all(np.isfinite([position, yaw])):
            raise ValueError("Reference reanchor must be finite")
        self._forward = _RampState(float(position), self._forward.velocity, self._forward.acceleration)
        self._yaw = _RampState(float(yaw), self._yaw.velocity, self._yaw.acceleration)
        self._initialized = True
        self._suspended = False
        self.valid = True
        self.last_reset_reason = "soft_reanchor"

    def _align(self, position: float, yaw: float, speed: float, yaw_rate: float, reason: str,
               reset_position: bool = True) -> None:
        # reset_position=False keeps the travel target where it was: a pure
        # velocity hold must not move the position goal away from the origin.
        self._forward = _RampState(position if reset_position else self._forward.position, speed)
        self._yaw = _RampState(yaw, yaw_rate)
        self._initialized = True
        self._hold_time = 0.0
        self._commanded = False
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
            self._commanded = True
            # The travel target follows the measurement while driving, so the
            # unavoidable acceleration/braking lag is not read as an error.
            # A2: keep the ramp acceleration so the jerk-limited S-curve survives re-anchoring
            # (acceleration is recomputed by advance() anyway when use_jerk_limit is False).
            self._forward = _RampState(position, self._forward.velocity, self._forward.acceleration)
        elif self.mode != ReferenceMode.HOLD:
            self.mode = ReferenceMode.BRAKE
            if limits.reanchor_position_while_braking and self.commanded_motion:
                # P1：只重锚位置，保留斜坡速度与加速度（减速仍按规划进行）
                self._forward = _RampState(position, self._forward.velocity,
                                           self._forward.acceleration)
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
        braking, jerk, braking_jerk = limits.max_braking, limits.max_jerk, limits.max_braking_jerk
        if not self.commanded_motion:
            # 方案 B：零指令站立 / 受扰恢复沿用 hip_angle_limit 口径的限值
            if limits.hold_braking is not None:
                braking = limits.hold_braking
            if limits.hold_jerk is not None:
                jerk, braking_jerk = limits.hold_jerk, limits.hold_braking_jerk
        if limits.use_jerk_limit:
            def step_forward(state: _RampState, duration: float) -> _RampState:
                if duration <= 0.0:
                    return state
                return state.advance_jerk(target_v, duration, acceleration, braking,
                                          jerk, braking_jerk=braking_jerk,
                                          unwind=limits.unwind_before_brake)
        else:
            def step_forward(state: _RampState, duration: float) -> _RampState:
                return state.advance(target_v, duration, acceleration, braking)
        forward, turning = self._forward, self._yaw
        self._forward = step_forward(forward, dt)
        self._yaw = turning.advance(target_w, dt, limits.max_yaw_acceleration, limits.max_yaw_braking)
        forward = step_forward(forward, lead)
        turning = turning.advance(target_w, lead, limits.max_yaw_acceleration, limits.max_yaw_braking)
        rows = []
        for _ in range(int(horizon) + 1):
            rows.append((forward.position, forward.velocity, forward.acceleration, turning.position, turning.velocity))
            forward = step_forward(forward, dt)
            turning = turning.advance(target_w, dt, limits.max_yaw_acceleration, limits.max_yaw_braking)
        values = np.asarray(rows)
        return ReferenceTrajectory(values[:, 0], values[:, 1], values[:, 2],
                                   np.full(len(rows), limits.equilibrium_pitch), np.zeros(len(rows)),
                                   values[:, 3], values[:, 4])


