"""Pure algorithm supervision: one decision owns reference and WBC permissions."""
from dataclasses import dataclass, replace
from enum import Enum
import logging
from typing import Callable
import numpy as np
from .mpc import ChassisState, MotionCommand, WheelMPC, ReferenceFeedback
from .observers import CompensationRouter, DisturbanceEstimate, DisturbanceSpace
from .observers.validation import positive
from .wbc.projection import ContactElimination, NullSpaceHierarchy, TaskLevel
from .wbc.sns import ActuatorLimits, SaturationResolver, SNSResult

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContactDetectionConfig:
    on_force: float
    off_force: float
    on_duration: float
    off_duration: float

    def __post_init__(self):
        positive(self.on_force, "on_force")
        if not np.isfinite(self.off_force) or not 0 <= self.off_force < self.on_force:
            raise ValueError("Require 0 <= off_force < on_force")
        positive(self.on_duration, "on_duration")
        positive(self.off_duration, "off_duration")


class ContactDetector:
    """Hysteresis in force and time. Invalid sensing immediately removes trust."""
    def __init__(self, config: ContactDetectionConfig):
        self.config = config
        self.reset()

    def reset(self):
        self._contact = [False, False]
        self._elapsed = [0.0, 0.0]

    def update(self, normals, trusted, dt):
        dt = positive(dt, "dt")
        if len(normals) != 2 or len(trusted) != 2:
            raise ValueError("Two per-wheel force/trust values required")
        budget = []
        for i, (normal, valid) in enumerate(zip(normals, trusted)):
            if not valid or not np.isfinite(normal):
                self._contact[i], self._elapsed[i] = False, 0.0
                budget.append(0.0)
                continue
            # Never carry an old high force budget through debounce.
            budget.append(max(float(normal), 0.0))
            change = normal <= self.config.off_force if self._contact[i] else normal >= self.config.on_force
            duration = self.config.off_duration if self._contact[i] else self.config.on_duration
            self._elapsed[i] = self._elapsed[i] + dt if change else 0.0
            if self._elapsed[i] + 1e-12 >= duration:
                self._contact[i] = not self._contact[i]
                self._elapsed[i] = 0.0
        return tuple(self._contact), tuple(budget)


class ControlMode(str, Enum):
    DISABLED = "disabled"
    AIRBORNE = "airborne"
    SINGLE_SUPPORT = "single_support"
    RECOVERY = "recovery"
    GROUND = "ground"


@dataclass(frozen=True)
class ControlDecision:
    mode: ControlMode
    contact: tuple[bool, bool]
    timestamp: float
    context: int
    reset_reference: bool

    def __post_init__(self):
        if not isinstance(self.mode, ControlMode) or len(self.contact) != 2 or not np.isfinite(self.timestamp):
            raise ValueError("Invalid control decision")
        if not isinstance(self.context, int) or self.context < 0:
            raise ValueError("Context must be a nonnegative integer")
        count = sum(bool(c) for c in self.contact)
        if ((self.mode in (ControlMode.GROUND, ControlMode.RECOVERY) and count != 2)
                or (self.mode == ControlMode.SINGLE_SUPPORT and count != 1)
                or (self.mode == ControlMode.AIRBORNE and count != 0)):
            raise ValueError("Control mode and contact set disagree")

    @property
    def motion_enabled(self):
        return self.mode == ControlMode.GROUND

    @property
    def balance_enabled(self):
        return self.mode in (ControlMode.GROUND, ControlMode.RECOVERY, ControlMode.SINGLE_SUPPORT)


# R1'（第 28 次排查）：短时脱离（急反转造成的瞬时卸空、落地抖动等）走软复位的时间窗口 [s]。
# 取 touchdown_dwell（0.10 s）的 2.5 倍：覆盖一次卸空-恢复往返，又明显短于真实飞行时间
# （跳跃约 0.5 s，必须走硬复位）。
_SOFT_RESET_WINDOW = 0.25


class ControlSupervisor:
    def __init__(self, recovery_dwell: float):
        self.recovery_dwell = positive(recovery_dwell, "recovery_dwell")
        self.reset()

    def reset(self):
        self._time = None
        self._stable_time = 0.0
        self._mode = ControlMode.DISABLED
        self._contact = (False, False)
        self._context = 0

    def update(self, *, timestamp: float, dt: float, contact: tuple[bool, bool],
               enabled: bool, command_fresh: bool, stable: bool, slipping: bool) -> ControlDecision:
        dt = positive(dt, "dt")
        if not np.isfinite(timestamp) or (self._time is not None and timestamp <= self._time):
            raise ValueError("Supervisor clock must be finite and strictly increasing")
        if self._time is not None and not np.isclose(timestamp - self._time, dt, atol=1e-9, rtol=1e-6):
            raise ValueError("dt must match elapsed supervisor time")
        if len(contact) != 2:
            raise ValueError("Two contacts required")
        contact = tuple(bool(c) for c in contact)
        self._time = timestamp
        if not enabled or not command_fresh:
            mode = ControlMode.DISABLED
        elif not any(contact):
            mode = ControlMode.AIRBORNE
        elif not all(contact):
            mode = ControlMode.SINGLE_SUPPORT
        else:
            self._stable_time = self._stable_time + dt if stable and not slipping else 0.0
            mode = ControlMode.GROUND if self._stable_time + 1e-12 >= self.recovery_dwell else ControlMode.RECOVERY
        if mode in (ControlMode.DISABLED, ControlMode.AIRBORNE, ControlMode.SINGLE_SUPPORT):
            self._stable_time = 0.0
        changed = mode != self._mode or contact != self._contact
        reset_reference = mode == ControlMode.GROUND and self._mode != ControlMode.GROUND
        if changed:
            self._context += 1
            _LOGGER.info("Control mode %s -> %s, contact=%s, context=%d", self._mode.value, mode.value, contact, self._context)
        self._mode, self._contact = mode, contact
        return ControlDecision(mode, contact, timestamp, self._context, reset_reference)


class WbcFeedbackAccumulator:
    """Aggregate the whole WBC interval; ignore exhausted regularization tasks."""
    def __init__(self, motion_tasks: tuple[str, ...]):
        if not motion_tasks:
            raise ValueError("At least one named motion task is required")
        self.motion_tasks = motion_tasks
        self.reset()

    def reset(self):
        self._residual = 0.0
        self._saturated = False

    def observe(self, result: SNSResult, limits: ActuatorLimits):
        outcomes = {item.name: item for item in result.hierarchy.outcomes}
        present = [outcomes[name] for name in self.motion_tasks if name in outcomes]
        if not present:
            return
        residual = max(item.residual_norm for item in present)
        self._residual = max(self._residual, residual)
        active_bound = np.any(np.isclose(result.torque, limits.lower, atol=1e-8, rtol=0)) or np.any(
            np.isclose(result.torque, limits.upper, atol=1e-8, rtol=0))
        curtailed = any(result.scale_of(item.name) < 1.0 - 1e-8 for item in present)
        self._saturated |= bool(result.fallback_used or active_bound or curtailed)

    def consume(self, state: ChassisState, *, enabled=True, command_fresh=True, slipping=False):
        feedback = ReferenceFeedback(contact=state.contact, enabled=enabled, command_fresh=command_fresh,
                                     slipping=slipping, saturated=self._saturated, motion_residual=self._residual,
                                     pitch=state.pitch, pitch_rate=state.pitch_rate)
        self.reset()
        return feedback


def _nullspace_resolver(mass: np.ndarray, elimination: ContactElimination) -> SaturationResolver:
    """默认 WBC 求解器工厂：零空间层级 + 饱和限幅。"""
    return SaturationResolver(NullSpaceHierarchy(mass), elimination)


@dataclass(frozen=True)
class ContactBlock:
    jacobian: np.ndarray
    bias: np.ndarray
    wheels: tuple[int, ...]

    def __post_init__(self):
        j = np.asarray(self.jacobian, dtype=float)
        b = np.asarray(self.bias, dtype=float)
        if j.ndim != 2 or b.shape != (j.shape[0],) or not np.all(np.isfinite(j)) or not np.all(np.isfinite(b)):
            raise ValueError("Invalid contact block")
        if not self.wheels or any(i not in (0, 1) for i in self.wheels):
            raise ValueError("Contact block must declare wheel ownership")
        object.__setattr__(self, "jacobian", j.copy())
        object.__setattr__(self, "bias", b.copy())


class SupervisedControl:
    """Algorithm-only composition; applications provide matrices and balance tasks.

    A single optional router chooses MPC acceleration OR WBC generalized force,
    preventing duplicate disturbance compensation across the two layers.
    Airborne/disabled output is explicit zero torque; aerial posture control is
    outside this stage. Single support preserves supplied balance tasks only.

    ``wbc_resolver_factory`` 是 ``(mass_matrix, elimination) -> resolver`` 的工厂，
    默认返回零空间法（``SaturationResolver``）；可用
    ``Algorithm.wbc.task_qp.task_qp_resolver_factory`` 运行时替换为单级 QP。
    """
    def __init__(self, mpc: WheelMPC, compensation: CompensationRouter | None = None,
                 wbc_resolver_factory: Callable[[np.ndarray, ContactElimination], object]
                 = _nullspace_resolver):
        self.mpc = mpc
        self.compensation = compensation
        self._wbc_resolver_factory = wbc_resolver_factory
        self._active_mpc_context = None
        # R1'（第 28 次排查）：上一次脱离地面（motion 禁止）开始的时间；短时往返走软复位
        self._detached_since = None

    def solve_mpc(self, state: ChassisState, command: MotionCommand, decision: ControlDecision,
                  feedback: ReferenceFeedback, estimate: DisturbanceEstimate | None = None):
        state = replace(state, contact=decision.contact)
        if not decision.motion_enabled or not feedback.enabled or not feedback.command_fresh:
            short = (self._detached_since is not None
                     and decision.timestamp - self._detached_since <= _SOFT_RESET_WINDOW)
            if self._detached_since is None:
                self._detached_since = decision.timestamp
                short = True
            # R1'：短时脱离（急反转时的瞬时卸载等）只软复位，保留斜坡；超时或禁用才硬复位
            soft = short and feedback.enabled and feedback.command_fresh
            self.mpc.reset(state, soft=soft)
            self._active_mpc_context = None
            if self.compensation and self.compensation.space == DisturbanceSpace.REDUCED_ACCELERATION:
                self.compensation.reset()
            return None
        if self._active_mpc_context != decision.context:
            soft = (self._detached_since is not None
                    and decision.timestamp - self._detached_since <= _SOFT_RESET_WINDOW)
            self.mpc.reset(state, soft=soft)
            self._active_mpc_context = decision.context
        self._detached_since = None
        feedback = replace(feedback, contact=state.contact, pitch=state.pitch, pitch_rate=state.pitch_rate)
        acceleration = None
        if self.compensation and self.compensation.space == DisturbanceSpace.REDUCED_ACCELERATION:
            acceleration = self.compensation.route(estimate, timestamp=decision.timestamp,
                                                   dt=self.mpc.settings.dt, context=decision.context, enabled=True)
        return self.mpc.solve(state, command, timestamp=decision.timestamp, feedback=feedback,
                              disturbance_acceleration=acceleration)

    def solve_wbc(self, *, mass, selection, dynamics_bias, contacts: tuple[ContactBlock, ...],
                  balance_tasks: tuple[TaskLevel, ...], motion_tasks: tuple[TaskLevel, ...],
                  limits: ActuatorLimits, decision: ControlDecision, dt: float,
                  estimate: DisturbanceEstimate | None = None,
                  external_force: np.ndarray | None = None,
                  weight_overrides: dict | None = None):
        if not decision.balance_enabled:
            if self.compensation and self.compensation.space == DisturbanceSpace.GENERALIZED_FORCE:
                self.compensation.reset()
            # None deliberately makes no claim about qacc under zero applied torque.
            return np.zeros(limits.size), None
        nv = np.asarray(mass).shape[0]
        active = [block for block in contacts if all(decision.contact[i] for i in block.wheels)]
        if not active:
            raise ValueError("Supported mode requires its contact Jacobian blocks")
        represented = {i for block in active for i in block.wheels}
        if any(c and i not in represented for i, c in enumerate(decision.contact)):
            raise ValueError("Missing contact model for a supported wheel")
        if any(block.jacobian.shape[1] != nv for block in active):
            raise ValueError("Contact dimension must match mass matrix")
        jacobian = np.vstack([block.jacobian for block in active])
        bias = np.concatenate([block.bias for block in active])
        tasks = list(balance_tasks) + (list(motion_tasks) if decision.motion_enabled else [])
        if not balance_tasks:
            raise ValueError("Supported control requires explicit balance tasks")
        # external_force：调用方直接注入的广义等效力（跳跃用，方案 v3 第 6 节）。
        # 注入期间**跳过扰动补偿**：两者写同一条通道，且补偿会把跳跃力当扰动去对抗（用户定案）。
        external = None
        if external_force is not None:
            external = np.asarray(external_force, dtype=float).reshape(-1)
            if self.compensation and self.compensation.space == DisturbanceSpace.GENERALIZED_FORCE:
                _LOGGER.warning("已注入外部等效力：本拍跳过扰动补偿（两者互斥）")
        elif self.compensation and self.compensation.space == DisturbanceSpace.GENERALIZED_FORCE:
            external = self.compensation.route(estimate, timestamp=decision.timestamp, dt=dt,
                                               context=decision.context, enabled=decision.motion_enabled)
        elimination = ContactElimination.from_matrices(jacobian, selection, mass)
        resolver = self._wbc_resolver_factory(mass, elimination)
        # WP8（2026-10-09）：按相位传入的显式权重重载；None 时行为完全不变。
        # 不要求所有后端都实现该接口：不支持的求解器直接跳过，不报错。
        if weight_overrides:
            setter = getattr(resolver, "set_weight_overrides", None)
            if setter is not None:
                setter(weight_overrides)
        result = resolver.resolve(
            jacobian, bias, tasks, limits, dynamics_bias=dynamics_bias, external_force=external)
        return result.torque.copy(), result
