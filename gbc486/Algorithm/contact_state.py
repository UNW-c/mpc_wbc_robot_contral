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
    # V3（第 29 次排查）：非跳跃相位（NONE）的离地确认时间。None = 与 off_duration 相同（旧行为）。
    # 调大后，短暂的法向力卸空（髋力矩反作用等）不再判成离地，切断
    # "卸空 -> 判离地、控制置零 -> 重锚参考 -> 又一次过指令" 的循环；跳跃相位仍用 off_duration，
    # 起跳判定不受影响。
    off_duration_idle: float | None = None
    # 跳跃相位门控
    takeoff_min_hip: float = 1.5            # [rad] 推力段只有共模髋角超过它才允许判离地
    flight_min_fraction: float = 0.6        # 飞行段在预计飞行时间的这一比例之前不判着地
    landing_on_scale: float = 0.5           # 着地段接地门限缩放（更快确认着地）
    # IMU 比力旁证：竖直比力低于它并持续 imu_duration 即单独触发整车离地
    imu_freefall: float = 2.0               # [m/s^2]
    imu_duration: float = 0.008             # [s]
    # R4（第 28 次排查）：非跳跃相位的离地需要运动学旁证——法向力低，但机身上升量小于
    # lift_height 且竖直比力没有接近真失重（> imu_freefall）时，判为法向力瞬时卸载
    # （例如急反转把轮子瞬间卸空），不判离地。None = 关闭（旧行为，默认）。
    # 注：初版旁证用了 6.0 m/s^2 作替代条件，实测该值过松（误判瞬间比力约 2.9），
    # 旁证形同虚设；现在只认真失重（沿用电量阈值 imu_freefall）。
    lift_height: float | None = None         # [m]


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

    def _detector(self, on_force: float, off_duration: float | None = None) -> ContactDetector:
        c = self.config
        return ContactDetector(ContactDetectionConfig(
            on_force=on_force, off_force=c.off_force, on_duration=c.on_duration,
            off_duration=c.off_duration if off_duration is None else off_duration))

    @property
    def idle_off_duration(self) -> float:
        c = self.config
        return c.off_duration if c.off_duration_idle is None else float(c.off_duration_idle)

    def reset(self, initial: tuple[bool, bool] | None = None) -> None:
        initial = self._initial if initial is None else tuple(bool(v) for v in initial)
        # 三个检测器：idle = 非跳跃相位（可单独设确认时间）、air = 推力 / 飞行、landing = 着地
        self._idle = self._detector(self.config.on_force, self.idle_off_duration)
        self._air = self._detector(self.config.on_force)
        self._landing = self._detector(self.config.on_force * self.config.landing_on_scale)
        for detector in (self._idle, self._air, self._landing):
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
               trusted=(True, True), base_rise: float | None = None,
               geometry_contact=None) -> ContactOutput:
        c = self.config
        normals = np.asarray(normals, dtype=float)
        if phase == JumpPhaseHint.LANDING:
            detector = self._landing
        elif phase in (JumpPhaseHint.THRUST, JumpPhaseHint.FLIGHT):
            detector = self._air
        else:
            detector = self._idle
        raw, _ = detector.update(tuple(normals), tuple(trusted), dt)
        # 三个检测器保持同步（切换相位时不丢状态）
        for other in (self._idle, self._air, self._landing):
            if other is detector:
                continue
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

        # LAND5（2026-10-17，用户同意）：几何接触是“是否离地”的硬证据，与力矩无关，
        # 用来否决法向力判据。力判据在“主动清零力矩”模式下会自证循环（离地档清零力矩
        # -> 法向力 0 -> 判离地）。geometry_contact=None 时行为与改动前完全一致。
        if geometry_contact is not None:
            for i in range(2):
                if bool(geometry_contact[i]):
                    proposed[i] = True

        # R4：非跳跃相位，法向力判离地还要求运动学旁证（机身上升或接近失重），否则维持接地
        if (phase == JumpPhaseHint.NONE and c.lift_height is not None and base_rise is not None):
            free_fall = (specific_force_z is not None and np.isfinite(specific_force_z)
                         and specific_force_z < c.imu_freefall)
            lifted = base_rise > c.lift_height or free_fall
            if not lifted:
                for i in range(2):
                    if previous[i] and not proposed[i]:
                        proposed[i] = True
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
        for d in (self._idle, self._air, self._landing):
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
