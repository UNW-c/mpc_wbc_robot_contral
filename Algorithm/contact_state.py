"""离地检测（第 14 / 17 次排查）：法向力双阈值滞回 + 跳跃相位门控 + IMU 比力旁证。

不依赖 MuJoCo。输入为每轮法向力（仿真真值或估计值），可选机身竖直比力与跳跃相位提示，
输出滤波后的逐轮接触状态与事件。参考：同济 §6.5（双阈值滞回；跳跃模式结合动作相位，
不能只凭一帧支撑力切换）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .supervision import ContactDetectionConfig, ContactDetector


class JumpPhaseHint(str, Enum):
    """由跳跃状态机提供；普通行驶时为 NONE。"""

    NONE = "none"
    THRUST = "thrust"
    FLIGHT = "flight"
    LANDING = "landing"


class ContactEvent(str, Enum):
    LIFTOFF = "liftoff"                     # 两轮都离地（整车）
    SINGLE_LIFT = "single_lift"             # 其中一轮离地
    TOUCHDOWN_ANY = "touchdown_any"         # 从整车离地到任一轮着地
    TOUCHDOWN_BOTH = "touchdown_both"       # 恢复双轮接地


@dataclass(frozen=True)
class GroundContactConfig:
    off_force: float = 5.0          # [N] 低于此值判为离地（名义单轮约 46.6 N 的 10%）
    on_force: float = 15.0          # [N] 高于此值判为接地
    off_duration: float = 0.008     # [s] 连续 2 个 WBC 周期
    on_duration: float = 0.012      # [s] 连续 3 个 WBC 周期
    # 跳跃相位门控
    takeoff_min_hip: float = 1.5            # [rad] 推力段只有共模髋角超过它才允许判离地
    flight_min_fraction: float = 0.6        # 飞行段在预计飞行时间的这一比例之前不判着地
    landing_on_scale: float = 0.5           # 着地段接地门限缩放（更快确认着地）
    # IMU 比力旁证：竖直比力低于它并持续 imu_duration 即单独触发整车离地
    imu_freefall: float = 2.0               # [m/s^2]
    imu_duration: float = 0.008             # [s]


@dataclass
class ContactOutput:
    contact: tuple[bool, bool]
    events: list[ContactEvent] = field(default_factory=list)
    imu_conflict: bool = False


class GroundContactEstimator:
    """逐轮滞回检测 + 相位门控。每个控制周期调用一次 :meth:`update`。"""

    def __init__(self, config: GroundContactConfig | None = None,
                 initial: tuple[bool, bool] = (True, True)):
        self.config = config or GroundContactConfig()
        self._initial = tuple(bool(v) for v in initial)
        self.reset()

    def _detector(self, on_force: float) -> ContactDetector:
        c = self.config
        return ContactDetector(ContactDetectionConfig(on_force=on_force, off_force=c.off_force,
                                                      on_duration=c.on_duration,
                                                      off_duration=c.off_duration))

    def reset(self, initial: tuple[bool, bool] | None = None) -> None:
        initial = self._initial if initial is None else tuple(bool(v) for v in initial)
        self._normal = self._detector(self.config.on_force)
        self._landing = self._detector(self.config.on_force * self.config.landing_on_scale)
        for detector in (self._normal, self._landing):
            detector._contact = list(initial)   # ContactDetector 默认初值为 (False, False)
        self.contact = initial
        self._imu_time = 0.0
        self._flight_time = 0.0
        self.predicted_flight = None

    def set_predicted_flight(self, duration: float | None) -> None:
        """跳跃状态机在离地时给出预计飞行时间（2·v_to/g），用于飞行段门控。"""
        self.predicted_flight = None if duration is None else max(float(duration), 0.0)

    def update(self, normals, dt: float, *, phase: JumpPhaseHint = JumpPhaseHint.NONE,
               hip_angle: float | None = None, specific_force_z: float | None = None,
               trusted=(True, True)) -> ContactOutput:
        c = self.config
        normals = np.asarray(normals, dtype=float)
        detector = self._landing if phase == JumpPhaseHint.LANDING else self._normal
        raw, _ = detector.update(tuple(normals), tuple(trusted), dt)
        # 两个检测器保持同步（切换相位时不丢状态）
        other = self._normal if detector is self._landing else self._landing
        other._contact = list(raw)
        other._elapsed = [0.0, 0.0]
        proposed = list(raw)
        previous = self.contact

        # 相位门控
        if phase == JumpPhaseHint.THRUST:
            # 推力段不判"接地"（本来就在地上）；离地只在髋角足够大时才允许
            allow_lift = hip_angle is not None and abs(hip_angle) >= c.takeoff_min_hip
            for i in range(2):
                if previous[i] and not proposed[i] and not allow_lift:
                    proposed[i] = True
        if phase == JumpPhaseHint.FLIGHT:
            self._flight_time += dt
            early = (self.predicted_flight is not None
                     and self._flight_time < c.flight_min_fraction * self.predicted_flight)
            for i in range(2):
                if not previous[i] and proposed[i] and early:
                    proposed[i] = False
        else:
            self._flight_time = 0.0

        # IMU 比力旁证：接近自由落体持续一段时间，单独触发整车离地
        imu_conflict = False
        if specific_force_z is not None and np.isfinite(specific_force_z):
            self._imu_time = self._imu_time + dt if specific_force_z < c.imu_freefall else 0.0
            imu_air = self._imu_time + 1e-12 >= c.imu_duration
            if imu_air and any(proposed) and phase in (JumpPhaseHint.THRUST, JumpPhaseHint.FLIGHT):
                proposed = [False, False]
            elif imu_air and all(proposed):
                imu_conflict = True          # 法向力说在地上、IMU 说在空中：以法向力为准并记录
        # 门控结果回写到检测器，避免下一拍立即翻回
        for d in (self._normal, self._landing):
            d._contact = list(proposed)
        current = tuple(proposed)
        self.contact = current
        return ContactOutput(current, self._events(previous, current), imu_conflict)

    @staticmethod
    def _events(previous, current) -> list[ContactEvent]:
        events = []
        if any(previous) and not any(current):
            events.append(ContactEvent.LIFTOFF)
        elif all(previous) and sum(current) == 1:
            events.append(ContactEvent.SINGLE_LIFT)
        if not any(previous) and any(current):
            events.append(ContactEvent.TOUCHDOWN_ANY)
        if not all(previous) and all(current):
            events.append(ContactEvent.TOUCHDOWN_BOTH)
        return events
