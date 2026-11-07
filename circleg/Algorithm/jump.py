"""跳跃状态机（第 17 次排查）：PREPARE → THRUST → FLIGHT → LANDING → RECOVER。不依赖 MuJoCo。

J-M 离线结论（docs/测试结果与数据分析.md 第 17 次排查）：
- 推力段满力矩（每髋 10 N·m），不做角动量反馈（反馈会分流推力，高度掉到 190 mm 以下）；
- 起跳前机身后仰 θ0 = −20°：推力段前翻冲量把它"转正"，离地角动量从 0.86 降到 0.31；
- 空中用轮子反作用力矩把机身俯仰拉平（τ_w = kp·θ + kd·θ̇），髋 PD 到 q_land；
- 着地后髋阻抗吸收冲击，轮子做轮速匹配（把空中积累的自转卸掉）。

接口约定（与 J-M 一致；接入 APP 时按 ``_hip_axis_sign`` 与 ctrl 通道顺序换算）：
- 共模髋角 q：0 = 下死点（站姿，机身最低），增大方向为起跳伸展方向；
- 输出 ``hip_torque``：共模髋力矩（每髋），``wheel_torque``：两轮同向力矩（每轮）；
- ``wheel_torque = None`` 表示该阶段轮子交还上层平衡控制。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .contact_state import ContactEvent, JumpPhaseHint

_G = 9.81


class JumpPhase(str, Enum):
    IDLE = "idle"
    PREPARE = "prepare"
    THRUST = "thrust"
    FLIGHT = "flight"
    LANDING = "landing"
    RECOVER = "recover"
    ABORT = "abort"


@dataclass(frozen=True)
class JumpParams:
    hip_torque_max: float = 10.0            # [N·m] 每髋（XML ctrlrange）
    wheel_torque_max: float = 5.0           # [N·m] 每轮
    # PREPARE：机身后仰到 θ0（由上层平衡控制执行，本模块只给参考与判据）
    preload_pitch: float = -0.524           # [rad] −30°（MuJoCo 开环搜索，第 17 次排查方案 B）
    prepare_tolerance: float = 0.087        # [rad] 到位误差 5°（第 19 次排查：快速蓄力）
    prepare_dwell: float = 0.004            # [s] 一个 WBC 周期（不要求静止驻留）
    preload_rate: float = 3.0               # [rad/s] 预倾参考斜坡速率（交给 MPC；6 rad/s 时失稳）
    prepare_timeout: float = 1.5            # [s]
    # THRUST
    thrust_timeout: float = 0.15            # [s]
    thrust_stop_hip: float = 3.6            # [rad] 超过仍未离地则停止推力
    thrust_brake_hip: float = 2.0           # [rad] 超过后髋反向满力矩制动曲柄
    thrust_pitch_kp: float = 20.0           # 推力段轮子俯仰 PD（每轮力矩 = kp·θ + kd·θ̇，正值使机身后仰）
    thrust_pitch_kd: float = 6.0
    # FLIGHT：髋 PD + 轮子反作用姿态控制
    land_hip: float = 0.0                   # [rad] 空中收回下死点（站姿）再着地：q≈π/2 质心偏前 6 cm、q=π 髋握不住曲柄，都无法着地平衡
    flight_hip_kp: float = 60.0
    flight_hip_kd: float = 3.0
    flight_pitch_kp: float = 20.0           # 飞行段轮子姿态控制（每轮力矩 = kp·θ + kd·θ̇，正值使机身后仰）
    flight_pitch_kd: float = 6.0
    # 着地对准：预计飞行时间的最后 (1 − flight_align_fraction) 段，kd 乘以 flight_align_kd_scale，把 θ̇ 压到 0
    flight_align_fraction: float = 0.7
    flight_align_kd_scale: float = 3.0
    flight_timeout_scale: float = 1.5       # 预计飞行时间的倍数
    # LANDING：髋阻抗 + 轮速匹配
    landing_hip_kp: float = 60.0
    landing_hip_kd: float = 3.0
    wheel_match_gain: float = 0.05          # [N·m·s/rad]
    # 着地段俯仰平衡：轮力矩（每轮）= 轮速匹配 − (kp·θ + kd·θ̇)/2 ... 符号按 J-M：正轮力矩使机身后仰
    landing_pitch_kp: float = 30.0
    landing_pitch_kd: float = 3.0
    landing_timeout: float = 0.4
    landing_settle_speed: float = 0.05      # [m/s] 竖直速度
    landing_settle_hip_rate: float = 0.5    # [rad/s]
    # RECOVER：髋回到站姿。recover_to_stance=False 时停在着地姿态（高位站稳，C25）
    recover_to_stance: bool = True
    # 第 19 次排查（落地收敛）：原 0.8 s 期间髋被 PD 锁住、WBC 只剩轮子，交还时产生二次瞬态；
    # land_hip=0 已是站姿，RECOVER 只需确认 |q|<0.05 就交还 WBC（一个 WBC 周期）
    recover_duration: float = 0.004
    recover_hip_kp: float = 30.0
    recover_hip_kd: float = 2.0
    # ABORT
    abort_pitch: float = 1.2                # [rad] 任意阶段超过即中止


@dataclass(frozen=True)
class JumpObservation:
    time: float
    hip_angle: float                        # 共模髋角
    hip_rate: float
    wheel_speed: float                      # 两轮平均角速度（世界系，正向与前进一致）
    body_forward_speed: float
    body_vertical_speed: float
    pitch: float
    pitch_rate: float
    contact: tuple[bool, bool]              # 由 GroundContactEstimator 给出
    events: tuple = ()
    wheel_radius: float = 0.16


@dataclass(frozen=True)
class JumpCommand:
    phase: JumpPhase
    hip_torque: float | None                # 每髋共模力矩；None = 交还上层
    wheel_torque: float | None              # 每轮力矩；None = 交还上层
    pitch_reference: float | None           # PREPARE 时给上层平衡控制的俯仰参考
    override_balance: bool                  # True：上层 MPC / WBC 暂停
    contact_hint: JumpPhaseHint


class JumpController:
    def __init__(self, params: JumpParams | None = None):
        self.p = params or JumpParams()
        self.reset()

    def reset(self) -> None:
        self.phase = JumpPhase.IDLE
        self._phase_start = None
        self._dwell = 0.0
        self.predicted_flight = None
        self.takeoff_speed = None
        self.log: list[tuple[float, str]] = []

    def request(self) -> bool:
        """外部起跳指令。只有 IDLE 时接受。"""
        if self.phase != JumpPhase.IDLE:
            return False
        self._enter(JumpPhase.PREPARE, None)
        return True

    def _elapsed(self, now: float) -> float:
        return now - self._phase_start if self._phase_start is not None else 0.0

    def _enter(self, phase: JumpPhase, time: float | None) -> None:
        self.phase = phase
        self._phase_start = time
        self._dwell = 0.0
        self.log.append((time if time is not None else float("nan"), phase.value))

    def _clip_hip(self, value: float) -> float:
        return float(np.clip(value, -self.p.hip_torque_max, self.p.hip_torque_max))

    def _clip_wheel(self, value: float) -> float:
        return float(np.clip(value, -self.p.wheel_torque_max, self.p.wheel_torque_max))

    def step(self, obs: JumpObservation, dt: float) -> JumpCommand:
        p = self.p
        if self._phase_start is None:
            self._phase_start = obs.time
        if self.phase not in (JumpPhase.IDLE, JumpPhase.ABORT) and abs(obs.pitch) > p.abort_pitch:
            self._enter(JumpPhase.ABORT, obs.time)

        if self.phase == JumpPhase.IDLE:
            return JumpCommand(self.phase, None, None, None, False, JumpPhaseHint.NONE)

        if self.phase == JumpPhase.PREPARE:
            if not all(obs.contact) or self._elapsed(obs.time) > p.prepare_timeout:
                self._enter(JumpPhase.ABORT, obs.time)
                return self._abort_command(obs)
            settled = abs(obs.pitch - p.preload_pitch) < p.prepare_tolerance
            self._dwell = self._dwell + dt if settled else 0.0
            if self._dwell + 1e-12 >= p.prepare_dwell:
                self._enter(JumpPhase.THRUST, obs.time)
            else:
                return JumpCommand(self.phase, None, None, p.preload_pitch, False, JumpPhaseHint.NONE)

        if self.phase == JumpPhase.THRUST:
            if ContactEvent.LIFTOFF in obs.events or not any(obs.contact):
                self.takeoff_speed = max(float(obs.body_vertical_speed), 0.0)
                self.predicted_flight = 2.0 * self.takeoff_speed / _G
                self._enter(JumpPhase.FLIGHT, obs.time)
            elif self._elapsed(obs.time) > p.thrust_timeout or obs.hip_angle > p.thrust_stop_hip:
                self._enter(JumpPhase.ABORT, obs.time)
                return self._abort_command(obs)
            else:
                hip = p.hip_torque_max if obs.hip_angle < p.thrust_brake_hip else -p.hip_torque_max
                wheel = self._clip_wheel(p.thrust_pitch_kp * obs.pitch + p.thrust_pitch_kd * obs.pitch_rate)
                return JumpCommand(self.phase, hip, wheel, None, True, JumpPhaseHint.THRUST)

        if self.phase == JumpPhase.FLIGHT:
            timeout = p.flight_timeout_scale * max(self.predicted_flight or 0.0, 0.1)
            if ContactEvent.TOUCHDOWN_ANY in obs.events or any(obs.contact):
                self._enter(JumpPhase.LANDING, obs.time)
            elif self._elapsed(obs.time) > timeout:
                self._enter(JumpPhase.ABORT, obs.time)
                return self._abort_command(obs)
            else:
                hip = p.flight_hip_kp * (p.land_hip - obs.hip_angle) - p.flight_hip_kd * obs.hip_rate
                # J-M 口径：两轮合计力矩 = kp·θ + kd·θ̇；每轮取一半。临近着地时加大阻尼，把 θ̇ 压到 0
                kd = p.flight_pitch_kd
                if self.predicted_flight and self._elapsed(obs.time) > p.flight_align_fraction * self.predicted_flight:
                    kd *= p.flight_align_kd_scale
                wheel = p.flight_pitch_kp * obs.pitch + kd * obs.pitch_rate
                return JumpCommand(self.phase, self._clip_hip(hip), self._clip_wheel(wheel), None,
                                   True, JumpPhaseHint.FLIGHT)

        if self.phase == JumpPhase.LANDING:
            settled = (all(obs.contact) and abs(obs.body_vertical_speed) < p.landing_settle_speed
                       and abs(obs.hip_rate) < p.landing_settle_hip_rate)
            if settled or self._elapsed(obs.time) > p.landing_timeout:
                self._enter(JumpPhase.RECOVER, obs.time)
            else:
                hip = p.landing_hip_kp * (p.land_hip - obs.hip_angle) - p.landing_hip_kd * obs.hip_rate
                # 着地后立即恢复俯仰平衡（倒立摆）：与飞行段同号约定，正轮力矩使机身后仰
                balance = p.landing_pitch_kp * obs.pitch + p.landing_pitch_kd * obs.pitch_rate
                wheel = self._clip_wheel(self._wheel_match(obs) + balance)
                return JumpCommand(self.phase, self._clip_hip(hip), wheel, None, True, JumpPhaseHint.LANDING)

        if self.phase == JumpPhase.RECOVER:
            fraction = min(self._elapsed(obs.time) / p.recover_duration, 1.0)
            if not p.recover_to_stance:
                # 高位站稳：髋保持在着地姿态，平衡交还上层（本模块只给髋）
                hip = p.recover_hip_kp * (p.land_hip - obs.hip_angle) - p.recover_hip_kd * obs.hip_rate
                if fraction >= 1.0:
                    self._enter(JumpPhase.IDLE, obs.time)
                return JumpCommand(self.phase, self._clip_hip(hip), None, None, False, JumpPhaseHint.NONE)
            target = p.land_hip * (1.0 - fraction)
            if fraction >= 1.0 and abs(obs.hip_angle) < 0.05:
                self._enter(JumpPhase.IDLE, obs.time)
                return JumpCommand(self.phase, None, None, None, False, JumpPhaseHint.NONE)
            hip = p.recover_hip_kp * (target - obs.hip_angle) - p.recover_hip_kd * obs.hip_rate
            return JumpCommand(self.phase, self._clip_hip(hip), None, None, False, JumpPhaseHint.NONE)

        return self._abort_command(obs)

    def _wheel_match(self, obs: JumpObservation) -> float:
        target = obs.body_forward_speed / obs.wheel_radius
        return self._clip_wheel(self.p.wheel_match_gain * (target - obs.wheel_speed))

    def _abort_command(self, obs: JumpObservation) -> JumpCommand:
        # 中止：髋阻尼 + 轮速匹配，交由上层判断是否停机
        hip = -self.p.flight_hip_kd * obs.hip_rate
        return JumpCommand(JumpPhase.ABORT, self._clip_hip(hip), self._wheel_match(obs), None, True,
                           JumpPhaseHint.NONE)
