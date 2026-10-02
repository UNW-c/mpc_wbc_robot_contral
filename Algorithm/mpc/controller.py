"""低频六状态 MPC：速度指令 -> 持久积分参考 -> 优化状态与等效轮力。

WBC 应采样 OptimizedTrajectory，而非直接跟踪优化前的参考。
降阶模型、髋力矩漂移和接触箱约束存在独立的适用边界；不能把本模块
视为已具备完整离地/失能管理的应用控制器。参考暂停时的控制退出须由
装配层消费 planner.valid，并配合接触模式处理，当前尚未接入 APP。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

from Algorithm.geometry import wrap_angle

from .contact import ContactAxis, ContactGeometry
from .model import INPUT_DIM, STATE_DIM, DiscreteModel, ReducedModel
from .planner import MotionCommand, MotionLimits, ReferenceFeedback, ReferenceTrajectory, VelocityPlanner
from .qp import CondensedMPCQP, QPResult, QPWeights, StageBounds
from .trajectory import OptimizedTrajectory

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPCSettings:
    """MPC 的全部可调量。默认值来自 docs/MPC-WBC方案.md §3 与 §6 的数值汇总。"""

    dt: float = 0.02
    horizon: int = 25
    weights: QPWeights = field(default_factory=QPWeights)
    limits: MotionLimits = field(default_factory=MotionLimits)
    wheel_torque_limit: float = 5.0
    friction_coefficient: float = 1.0
    friction_margin: float = 0.75
    minimum_normal_force: float = 2.0
    nominal_normal_force: float = 46.6
    yaw_moment_limit: float = 7.5
    scrub_moment_coefficient: float = 5.59
    yaw_moment_floor: float = 0.5
    # 俯仰硬约束上限 [rad]。0.785 rad = 45°，用户于 2026-09-22 指定的安全上限，
    # 用于放开 MPC 可用俯仰幅度（换制动能力）。注意这是硬约束，不是权重：
    # 只调 QPWeights.pitch/pitch_rate 无法突破它。原值 0.30 rad 是旧的工作点。
    pitch_limit: float = 0.785
    speed_limit: float = 3.00  # MPC 速度状态带；随 max_speed=2.5 调整（2026-09-29）
    enable_state_bounds: bool = True
    max_consecutive_failures: int = 3
    # 状态预推时长：估计延迟 + 求解耗时（方案 §3.4）。0 表示不补偿。
    prediction_lead: float = 0.0
    # Hessian 条件数报警阈值（方案 §1.1）。实测 25 步视界约 9.3e4。
    condition_number_limit: float = 1.0e6
    # 是否把髋力矩当已知输入喂进俯仰方程（方案 §1.5）。
    compensate_hip_torque: bool = True
    # B3：8 状态模型的共模髋角状态约束 [rad]（模型可信范围，B1 显示约对应 1.2 m/s^2）。
    hip_angle_limit: float = 0.55  # 2026-09-28 部署为默认（E3 选定；旧值 0.45）
    # B3：8 状态模型的髋力矩输入上下界 [N*m]（XML actuator ctrlrange）。
    hip_torque_limit: float = 10.0
    osqp_settings: dict[str, object] = field(default_factory=dict)
    # A1（第 15 次排查）：用模型 metadata 的稳态能力表把规划限值对齐 hip_angle_limit。
    # 关闭时保持 MotionLimits 里的手填值（1.2 / 1.4 / 1.4）。
    auto_accel_limits: bool = True
    accel_fraction_up: float = 0.85     # 起步限值系数（agent2 设计值）
    accel_fraction_down: float = 0.95   # 刹车与零指令恢复限值系数（agent2 设计值）
    # A2（第 15 次排查）：use_jerk_limit=True 时 max_jerk = 起步限值 / jerk_ramp_time
    # （加速度在这段时间内从 0 升满），由 WheelMPC 初始化时写回 limits。
    jerk_ramp_time: float = 0.25
    # B2（第 24 次排查）：刹车段 jerk 爬升时间；None = 与起步共用 max_jerk（旧行为）。
    # 给定时 max_braking_jerk = 刹车限值 / brake_jerk_ramp_time。
    brake_jerk_ramp_time: float | None = None
    # A2：髋角参考前馈。True 时参考第 6/7 列 = q_ss(a_ref) 及其差分（限幅到髋限 90%）；
    # False = 旧行为（髋角参考为 0）。只对带稳态能力表的 hip_input 模型生效。
    hip_reference_feedforward: bool = False
    # A3（第 16 次排查）：起步 / 刹车限值改按“髋限 + 前倾 15°”的持续能力计算
    # （metadata["accel_capacity_tilted"]），不分高低速；零指令恢复斜坡仍按只用髋的能力。
    tilt_assisted_limits: bool = False
    # A3b：倾角参考。超出“只用髋”能力的那部分参考加速度，按髋限处的倾角灵敏度
    # da/dθ = (a_tilt − a_hip)/θ_tilt 换算成前倾参考，限幅 pitch_feedforward_max、变化率 pitch_ff_rate_limit。
    # 只给倾角参考，不给髋角参考（A2 的教训）。
    pitch_feedforward: bool = False
    pitch_feedforward_max: float = 0.2618   # 15°（C14）
    pitch_ff_rate_limit: float = 1.0        # [rad/s]
    # Z1（第 31 次排查）：用 ZMP / 质心偏移表述生成参考，替换俯仰前馈的拟合式。
    # 倒立摆关系 Δ = z_com·a/g（Δ = 质心相对接地点的前后偏移），再按 zmp_split 分配到
    # 俯仰（Δ_tilt = z_com·sinθ）与曲柄（Δ_crank = r_crank·sin q_c）。zmp_lead 让偏移参考
    # 提前收回，使 v→0 时 Δ→0（消除停住后收姿态造成的回弹）。
    zmp_reference: bool = False
    zmp_split: float = 0.5           # 偏移中由曲柄承担的比例（0 = 全俯仰，1 = 全曲柄）
    zmp_lead: float = 0.0            # [s] 偏移参考的提前量
    zmp_com_height: float = 0.180    # [m] 质心相对接地点高度（zmp_probe 实测 0.177~0.182）
    zmp_crank_radius: float = 0.06   # [m] 曲柄偏心半径（构型给定）
    zmp_crank_rate_limit: float = 3.0  # [rad/s] 曲柄参考变化率上限
    # B1（第 24 次排查）：刹车（参考加速度与参考速度反号）时允许的后仰倾角 [rad]。
    # None = 与 pitch_feedforward_max 相同（旧行为）。给定时：刹车限值按该倾角的能力表
    # （metadata["accel_capacity_tilted_brake"]，由 APP 按此角度生成）计算，倾角参考的刹车段
    # 也用这张表的灵敏度和这个上限。
    brake_pitch_max: float | None = None
    # 方案 B（第 27 次排查）：分模式髋角限值。给定时，规划器 DRIVE / BRAKE（有速度指令及其刹车）
    # 用这个较大的限值（MPC 髋状态约束 + 起步 / 刹车规划限值）；HOLD / RESET（零指令站立、受扰
    # 恢复）仍用 hip_angle_limit，零指令恢复限值也按 hip_angle_limit。None = 旧行为。
    drive_hip_angle_limit: float | None = None
    # 约束从驾驶限值回落到 hip_angle_limit 的速率 [rad/s]，且不低于 |q_c| + hip_limit_margin，
    # 避免刹车结束转 HOLD 时约束突降导致 QP 不可行或冲击。
    hip_limit_relax_rate: float = 0.5
    hip_limit_margin: float = 0.05
    # R2'（第 28 次排查）：中途反转先卸加速度（写入 limits.unwind_before_brake）。
    unwind_before_brake: bool = False
    # P1（第 31 次排查）：指令刹车期间位置参考跟随实测（减速位移不计入误差，消除回弹）。
    reanchor_position_while_braking: bool = False
    # R1'（第 28 次排查）：监督层短时离地 / 恢复往返时保留规划器斜坡与 commanded 标记。
    preserve_ramp_on_recontact: bool = False


@dataclass(frozen=True)
class ChassisState:
    """MPC 需要的全部反馈量。字段与 ``Algorithm.estimator.BodyStateKalman`` 对应。"""

    position: float = 0.0
    velocity: float = 0.0
    pitch: float = 0.0
    pitch_rate: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0
    # 共模髋角 0.5*(q1 + sign*q3)（sign = dot(axis1, axis3)）：腿的前后摆动，纵向用（T7）。
    hip_angle: float = 0.0
    # 共模髋角速度（8 状态模型的 q_c_dot，B3）。
    hip_rate: float = 0.0
    # 差模髋角 0.5*(q1 - sign*q3)：决定两轮前后错开 0.12*sin(q_diff)，用于 scrub 与接触连线（#13）。
    hip_differential: float = 0.0
    # 单腿权限因子 mean(|sin q1|, |sin q3|)，用于横滚/高度任务排程；None 时退回 |sin(hip_angle)|（#13）。
    hip_authority: float | None = None
    # 两髋力矩之和（j1_ctrl + j3_ctrl）。髋轴就是俯仰轴，这个和是机身俯仰上的
    # 直接力矩 ±20 N·m，方案 §1.5 要求把它当已知输入喂进俯仰方程。
    hip_torque: float = 0.0
    normal_forces: tuple[float, float] | None = None
    contact: tuple[bool, bool] = (True, True)

    def vector(self, state_dim: int = STATE_DIM) -> np.ndarray:
        """按模型维数拼状态向量；state_dim > 6 时追加共模髋角与髋角速度（B3）。"""
        values = [
            self.position,
            self.velocity,
            self.pitch,
            self.pitch_rate,
            self.yaw,
            self.yaw_rate,
        ]
        if state_dim > STATE_DIM:
            values = values + [self.hip_angle, self.hip_rate]
        return np.array(values[:state_dim], dtype=float)


@dataclass(frozen=True)
class MPCSolution:
    """一拍的求解结果。``pitch_reference`` 是交给下层 WBC 的主要接口。"""

    force: np.ndarray
    force_trajectory: np.ndarray
    wheel_torque: tuple[float, float]
    drive_force: float
    yaw_moment: float
    reference: ReferenceTrajectory
    predicted_states: np.ndarray
    qp: QPResult
    contact: ContactAxis
    condition_number: float
    fallback: bool
    trajectory: OptimizedTrajectory

    @property
    def state(self) -> np.ndarray:
        """本拍实际喂给 QP 的状态（做了状态预推就是预推后的，否则等于实测）。"""
        return np.asarray(self.predicted_states[0], dtype=float)

    # 便捷属性指向下一优化节点；高频 WBC 应按实际时间调用 trajectory.sample。
    @property
    def pitch_reference(self) -> float:
        return float(self.predicted_states[1, 2])

    @property
    def pitch_rate_reference(self) -> float:
        return float(self.predicted_states[1, 3])

    @property
    def velocity_reference(self) -> float:
        return float(self.predicted_states[1, 1])

    @property
    def position_reference(self) -> float:
        return float(self.predicted_states[1, 0])


class WheelMPC:
    """50 Hz 上层 MPC。一个实例对应一个固定的降阶模型。"""

    def __init__(self, model: ReducedModel, settings: MPCSettings | None = None) -> None:
        self.model = model
        self.settings = self._apply_auto_accel_limits(settings or MPCSettings(), model)
        self.discrete = model.discretize(self.settings.dt)
        self.qp = CondensedMPCQP(
            self.settings.horizon,
            self.settings.weights,
            self.settings.osqp_settings,
            input_dim=model.input_dim,
        )
        self.qp.set_model(self.discrete, model.half_track)
        self._previous_theta_ff = 0.0
        self.last_theta_ff = 0.0
        self._previous_crank_ff = 0.0
        self.last_crank_ff = 0.0
        # 外部俯仰参考偏置（第 17 次排查：跳跃 PREPARE 阶段的预倾），叠加在参考第 2 列上，
        # 沿视界按 pitch_ff_rate_limit 斜坡逼近，避免阶跃
        self.pitch_bias_target = 0.0
        self._pitch_bias = 0.0
        self.pitch_bias_rate = 0.5          # [rad/s] 偏置斜坡速率（第 19 次排查：跳跃蓄力由 APP 设定）
        # 方案 B：当前生效的 MPC 髋角约束（分模式，带回落斜坡）
        self.active_hip_limit = float(self.settings.hip_angle_limit)
        self.planner = VelocityPlanner(self.settings.limits)
        self.fallback_gain = self._design_fallback_gain()
        self.geometry = ContactGeometry()
        self.consecutive_failures = 0
        self.degraded = False
        self.conditioning_warned = False
        self._previous_force = np.zeros(model.input_dim, dtype=float)
        self._previous_trajectory = np.zeros((self.settings.horizon, model.input_dim), dtype=float)
        self._hip_drift_map = model.drift_step_map(self.settings.dt)
        self._lead_model = self._build_lead_model()
        self._disturbance_map = model.drift_step_map(self.settings.dt)
        self._disturbance_lead_map = model.drift_step_map(self.settings.prediction_lead)

    @staticmethod
    def _apply_auto_accel_limits(settings: MPCSettings, model: ReducedModel) -> MPCSettings:
        """A1（第 15 次排查）：按 hip_angle_limit 从稳态能力表反查规划限值。

        - hip_input 模型：metadata["accel_capacity"] 给出 (a, q_c) 表，
          a_cap = interp(hip_angle_limit)，起步 = accel_fraction_up * a_cap，
          刹车与恢复 = accel_fraction_down * a_cap；
        - held_hip 模型没有这张表，保持手填值并打一行 INFO 说明；
        - auto_accel_limits=False 时原样返回（测试要求逐位一致）。
        """
        if not settings.auto_accel_limits:
            return settings
        capacity = model.metadata.get("accel_capacity")
        if not isinstance(capacity, dict):
            _LOGGER.info("A1：模型 %s 没有稳态能力表，规划限值保持手填值"
                         "（起步 %.2f / 刹车 %.2f / 恢复 %.2f m/s^2）",
                         model.source, settings.limits.commanded_acceleration,
                         settings.limits.max_braking, settings.limits.max_acceleration)
            return settings
        q_table = np.asarray(capacity["q_c"], dtype=float)
        a_table = np.asarray(capacity["a"], dtype=float)
        tilted = model.metadata.get("accel_capacity_tilted")
        brake_table = model.metadata.get("accel_capacity_tilted_brake")

        def planning_limits(hip_limit: float) -> tuple[float, float, float]:
            """按给定髋限计算 (起步, 刹车, 零指令恢复)。"""
            a_cap = float(np.interp(hip_limit, q_table, a_table))
            up = float(settings.accel_fraction_up) * a_cap
            down = float(settings.accel_fraction_down) * a_cap
            recover = down
            if settings.tilt_assisted_limits and isinstance(tilted, dict):
                a_tilt = float(np.interp(hip_limit, np.asarray(tilted["q_c"], dtype=float),
                                         np.asarray(tilted["a"], dtype=float)))
                up = float(settings.accel_fraction_up) * a_tilt
                down = float(settings.accel_fraction_down) * a_tilt
                if settings.brake_pitch_max is not None and isinstance(brake_table, dict):
                    a_brake = float(np.interp(hip_limit, np.asarray(brake_table["q_c"], dtype=float),
                                              np.asarray(brake_table["a"], dtype=float)))
                    down = float(settings.accel_fraction_down) * a_brake
            return up, down, recover

        def jerks(up: float, down: float) -> tuple[float, float | None]:
            jerk = up / max(float(settings.jerk_ramp_time), 1e-3)
            bjerk = (None if settings.brake_jerk_ramp_time is None
                     else down / max(float(settings.brake_jerk_ramp_time), 1e-3))
            return jerk, bjerk

        hold_limit = float(settings.hip_angle_limit)
        hold_up, hold_down, recover = planning_limits(hold_limit)
        drive_limit = (float(settings.drive_hip_angle_limit) if settings.drive_hip_angle_limit is not None
                       else hold_limit)
        if drive_limit != hold_limit:
            up, down, _ = planning_limits(drive_limit)
        else:
            up, down = hold_up, hold_down
        limits = replace(settings.limits, commanded_acceleration=up,
                         max_acceleration=recover, max_braking=down,
                         unwind_before_brake=bool(settings.unwind_before_brake),
                         reanchor_position_while_braking=bool(
                             settings.reanchor_position_while_braking))
        if limits.use_jerk_limit:
            jerk, bjerk = jerks(up, down)
            limits = replace(limits, max_jerk=jerk, max_braking_jerk=bjerk)
        if drive_limit != hold_limit:
            # 方案 B：零指令（未经速度指令驱动）的刹车 / jerk 按 hip_angle_limit
            hold_jerk, hold_bjerk = jerks(hold_up, hold_down)
            limits = replace(limits, hold_braking=hold_down,
                             hold_jerk=hold_jerk if limits.use_jerk_limit else None,
                             hold_braking_jerk=hold_bjerk if limits.use_jerk_limit else None)
            _LOGGER.info("方案 B：驾驶髋限 %.2f rad（起步 %.2f / 刹车 %.2f），零指令髋限 %.2f rad"
                         "（刹车 %.2f / 恢复 %.2f）", drive_limit, up, down, hold_limit, hold_down, recover)
        _LOGGER.info("A1：髋限 %.2f rad；规划限值 起步 %.2f / 刹车 %.2f / 恢复 %.2f m/s^2；"
                     "S 曲线 %s（jerk %.2f / 刹车 jerk %s m/s^3）", hold_limit, up, down, recover,
                     "开" if limits.use_jerk_limit else "关", limits.max_jerk,
                     "-" if limits.max_braking_jerk is None else "%.2f" % limits.max_braking_jerk)
        return replace(settings, limits=limits)

    def _apply_pitch_feedforward(self, reference_states: np.ndarray, acceleration: np.ndarray,
                                 velocity: np.ndarray | None = None) -> None:
        """A3b（第 16 次排查）：参考加速度超出髋能力的部分换算成前倾参考。

        θ_ff = sign(a)·clip((|a| − a_hip)/k_θ, 0, θ_max)，其中 a_hip、a_tilt 分别是髋限处只用髋 /
        前倾 θ_tilt 的持续能力，k_θ = (a_tilt − a_hip)/θ_tilt。沿视界按 pitch_ff_rate_limit 限变化率
        （起点是上一拍的首节点值），第 3 列取差分。开关关闭或缺表时不动。
        """
        if not self.settings.pitch_feedforward:
            return
        hip_table = self.model.metadata.get("accel_capacity")
        tilt_table = self.model.metadata.get("accel_capacity_tilted")
        if not isinstance(hip_table, dict) or not isinstance(tilt_table, dict):
            return
        q_lim = float(self.settings.drive_hip_angle_limit
                      if self.settings.drive_hip_angle_limit is not None and self.planner.commanded_motion
                      else self.settings.hip_angle_limit)
        a_hip = float(np.interp(q_lim, hip_table["q_c"], hip_table["a"]))
        a_tilt = float(np.interp(q_lim, tilt_table["q_c"], tilt_table["a"]))
        theta_tilt = float(tilt_table["theta"])
        k_theta = max((a_tilt - a_hip) / max(theta_tilt, 1e-6), 1e-6)
        a = np.asarray(acceleration, dtype=float)
        excess = np.maximum(np.abs(a) - float(self.settings.accel_fraction_up) * a_hip, 0.0)
        theta_ff = np.sign(a) * np.minimum(excess / k_theta, float(self.settings.pitch_feedforward_max))
        brake_table = self.model.metadata.get("accel_capacity_tilted_brake")
        if (self.settings.brake_pitch_max is not None and isinstance(brake_table, dict)
                and velocity is not None):
            # B1：刹车段（a 与 v 反号）按刹车倾角表的灵敏度与上限换算
            a_brake = float(np.interp(q_lim, brake_table["q_c"], brake_table["a"]))
            theta_brake = float(brake_table["theta"])
            k_brake = max((a_brake - a_hip) / max(theta_brake, 1e-6), 1e-6)
            braking = np.asarray(velocity, dtype=float) * a < 0.0
            theta_b = np.sign(a) * np.minimum(excess / k_brake, float(self.settings.brake_pitch_max))
            theta_ff = np.where(braking, theta_b, theta_ff)
        step = float(self.settings.pitch_ff_rate_limit) * float(self.settings.dt)
        previous = self._previous_theta_ff
        for index in range(len(theta_ff)):
            theta_ff[index] = float(np.clip(theta_ff[index], previous - step, previous + step))
            previous = theta_ff[index]
        self._previous_theta_ff = float(theta_ff[1]) if len(theta_ff) > 1 else float(theta_ff[0])
        reference_states[:, 2] = reference_states[:, 2] + theta_ff
        reference_states[:, 3] = np.gradient(theta_ff, float(self.settings.dt)) if len(theta_ff) > 1 else 0.0
        self.last_theta_ff = float(theta_ff[0])

    def _update_hip_limit(self, state: ChassisState) -> None:
        """方案 B（第 27 次排查）：按规划器模式切换 MPC 髋角约束。

        速度指令驱动中（DRIVE 及其后的 BRAKE）-> drive_hip_angle_limit（立即放开）；
        其余（零指令站立、HOLD、复位、受扰恢复）-> hip_angle_limit，
        但以 hip_limit_relax_rate 回落，且不低于 |q_c| + hip_limit_margin（避免约束突降）。
        """
        hold = float(self.settings.hip_angle_limit)
        drive = self.settings.drive_hip_angle_limit
        if drive is None:
            self.active_hip_limit = hold
            return
        if self.planner.commanded_motion:
            self.active_hip_limit = float(drive)
            return
        step = float(self.settings.hip_limit_relax_rate) * float(self.settings.dt)
        floor = min(abs(float(state.hip_angle)) + float(self.settings.hip_limit_margin), float(drive))
        # 只降不升：floor 只用来减缓回落，不能让零指令下的约束跟着髋角往上抬（H3 实测会棘轮到驾驶限值）
        self.active_hip_limit = min(self.active_hip_limit, max(hold, floor, self.active_hip_limit - step))

    def _apply_zmp_reference(self, reference_states: np.ndarray, acceleration: np.ndarray) -> None:
        """Z1（第 31 次排查）：按 ZMP / 质心偏移表述写俯仰与曲柄参考。

        Δ*(t) = z_com·a(t+lead)/g，其中 a 为参考加速度；Δ_tilt = (1−split)·Δ*、Δ_crank = split·Δ*，
        再换算 θ = asin(Δ_tilt/z_com)（限幅 pitch_feedforward_max）、q = asin(Δ_crank/r)（限幅髋限 90%）。
        两者都按各自速率上限沿视界斜坡，起点接上一拍的首节点值，避免阶跃。
        """
        settings = self.settings
        step = float(settings.dt)
        a = np.asarray(acceleration, dtype=float)
        lead = max(int(round(float(settings.zmp_lead) / max(step, 1e-9))), 0)
        if lead:
            a = np.concatenate([a[lead:], np.full(min(lead, len(a)), a[-1])])
        delta = float(settings.zmp_com_height) * a / 9.81
        split = float(np.clip(settings.zmp_split, 0.0, 1.0))
        z = max(float(settings.zmp_com_height), 1e-6)
        r = max(float(settings.zmp_crank_radius), 1e-6)
        theta_max = float(settings.pitch_feedforward_max)
        crank_max = 0.9 * float(settings.hip_angle_limit)
        tilt = np.clip(np.arcsin(np.clip((1.0 - split) * np.abs(delta) / z, 0.0, np.sin(theta_max))),
                       0.0, theta_max) * np.sign(a)
        crank = np.clip(np.arcsin(np.clip(split * np.abs(delta) / r, 0.0, np.sin(crank_max))),
                        0.0, crank_max) * np.sign(a)
        theta_step = float(settings.pitch_ff_rate_limit) * step
        crank_step = float(settings.zmp_crank_rate_limit) * step
        previous_theta, previous_crank = self._previous_theta_ff, self._previous_crank_ff
        for index in range(len(tilt)):
            previous_theta = float(np.clip(tilt[index], previous_theta - theta_step, previous_theta + theta_step))
            previous_crank = float(np.clip(crank[index], previous_crank - crank_step, previous_crank + crank_step))
            tilt[index], crank[index] = previous_theta, previous_crank
        if len(tilt) > 1:
            self._previous_theta_ff = float(tilt[1])
            self._previous_crank_ff = float(crank[1])
        else:
            self._previous_theta_ff = float(tilt[0])
            self._previous_crank_ff = float(crank[0])
        self.last_theta_ff = float(tilt[0])
        self.last_crank_ff = float(crank[0])
        reference_states[:, 2] = reference_states[:, 2] + tilt
        reference_states[:, 3] = np.gradient(tilt, step) if len(tilt) > 1 else 0.0
        reference_states[:, 6] = crank
        reference_states[:, 7] = np.gradient(crank, step) if len(crank) > 1 else 0.0

    def _apply_pitch_bias(self, reference_states: np.ndarray) -> None:
        """外部俯仰偏置：从当前值按 pitch_bias_rate 斜坡逼近 pitch_bias_target，沿视界写入参考。"""
        target = float(self.pitch_bias_target)
        if target == 0.0 and self._pitch_bias == 0.0:
            return
        step = float(self.pitch_bias_rate) * float(self.settings.dt)
        values = np.empty(len(reference_states))
        current = self._pitch_bias
        for index in range(len(values)):
            current = float(np.clip(target, current - step, current + step))
            values[index] = current
        self._pitch_bias = float(values[1]) if len(values) > 1 else float(values[0])
        reference_states[:, 2] = reference_states[:, 2] + values
        reference_states[:, 3] = reference_states[:, 3] + np.gradient(values, float(self.settings.dt))

    def _apply_hip_feedforward(self, reference_states: np.ndarray, acceleration: np.ndarray) -> None:
        """A2（第 15 次排查）：参考髋角 = 维持参考加速度所需的稳态共模髋角。

        用 metadata["accel_capacity"]（a -> q_c，θ = 0 的稳态族）按 |a| 插值、按 a 的符号取号，
        限幅到 hip_angle_limit 的 90%；第 7 列取相邻节点差分。开关关闭、模型没有髋状态
        或没有能力表时原样返回（髋角参考保持 0）。
        """
        if not self.settings.hip_reference_feedforward or self.model.state_dim < 8:
            return
        capacity = self.model.metadata.get("accel_capacity")
        if not isinstance(capacity, dict):
            return
        a_table = np.asarray(capacity["a"], dtype=float)
        q_table = np.asarray(capacity["q_c"], dtype=float)
        a = np.asarray(acceleration, dtype=float)
        q_ff = np.sign(a) * np.interp(np.abs(a), a_table, q_table)
        cap = 0.9 * float(self.settings.hip_angle_limit)
        q_ff = np.clip(q_ff, -cap, cap)
        reference_states[:, 6] = q_ff
        reference_states[:, 7] = np.gradient(q_ff, float(self.settings.dt)) if len(q_ff) > 1 else 0.0

    def _build_lead_model(self) -> DiscreteModel | None:
        """状态预推用的离散模型（方案 §3.4）。

        预推时长是个常数，所以矩阵指数只算一次；每拍只是拿上一拍的力做零阶保持前推。
        """
        lead = float(self.settings.prediction_lead)
        if lead < 0.0:
            raise ValueError("prediction_lead 不能为负")
        if lead > float(self.settings.dt):
            raise ValueError("prediction_lead 不应超过一拍，超过的话零阶保持假设不成立")
        if lead == 0.0:
            return None
        return self.model.discretize(lead)

    @property
    def horizon(self) -> int:
        return self.settings.horizon

    def _design_fallback_gain(self) -> np.ndarray:
        """在同一降阶模型上设计连续 LQR 增益，复用 ``Algorithm.lqr`` 的里卡提求解。"""
        from Algorithm.lqr import lqr_gain_from_linearization

        weights = self.settings.weights
        Q = np.diag(weights.state_diagonal(self.model.state_dim))
        R = np.diag(np.maximum(weights.input_diagonal(self.model.input_dim), 1.0e-6))
        return np.asarray(lqr_gain_from_linearization(self.model.A, self.model.B, Q, R), dtype=float)

    def reset(self, state: ChassisState | None = None, *, soft: bool = False) -> None:
        """重新锚定内部状态（仿真复位、状态机切换后必须调用）。

        R1'（第 28 次排查）：``soft=True`` 且 ``preserve_ramp_on_recontact`` 开启时，只重锚位置 /
        偏航到实测值，保留规划器的速度、加速度斜坡与 commanded 标记（短时离地 / 恢复往返用）。
        """
        if soft and self.settings.preserve_ramp_on_recontact and state is not None:
            self.planner.soft_reanchor(state.position, state.yaw)
            self.consecutive_failures = 0
            self.degraded = False
            return
        self.planner.reset(
            speed=0.0 if state is None else state.velocity,
            yaw_rate=0.0 if state is None else state.yaw_rate,
            position=None if state is None else state.position,
            yaw=None if state is None else state.yaw,
        )
        self._previous_force[:] = 0.0
        self._previous_trajectory[:] = 0.0
        self._previous_theta_ff = 0.0
        self._previous_crank_ff = 0.0
        self._pitch_bias = 0.0
        self.pitch_bias_target = 0.0
        self.consecutive_failures = 0
        self.degraded = False
        self.conditioning_warned = False

    # ---- 约束排程 --------------------------------------------------------
    def wheel_force_limit(self, state: ChassisState) -> np.ndarray:
        """逐轮纵向力上界 = min(力矩限, 摩擦锥)，腾空轮为 0。

        只处理前两个（轮力）输入（B2）；其余输入由 ``StageBounds.extra_*`` 约束。
        """
        settings = self.settings
        torque_bound = settings.wheel_torque_limit / max(self.model.wheel_radius, 1.0e-9)
        nominal = settings.nominal_normal_force
        normals = state.normal_forces if state.normal_forces is not None else (nominal, nominal)
        cone_gain = settings.friction_coefficient * settings.friction_margin
        # 轮力固定为前两个输入（B2）：其余输入（如髋力矩）由 StageBounds.extra_* 约束。
        bounds = np.empty(2, dtype=float)
        for index in range(2):
            if not state.contact[index]:
                bounds[index] = 0.0
                continue
            normal = max(float(normals[index]), 0.0)
            bounds[index] = min(torque_bound, cone_gain * normal)
        return np.maximum(bounds, 0.0)

    def yaw_moment_limit(self, state: ChassisState, force_limit: np.ndarray) -> float:
        """scrub 折扣后的偏航力矩预算。

        ``scrub_moment_coefficient * |sin q|`` 是两轮前后错开 ``0.12 sin q`` 造成的
        库仑 scrub 阻力矩（docs/MPC-WBC方案.md §1.2）：它先把偏航权限吃掉一部分，
        剩下的才是 MPC 能用的。同时不能超过箱型约束本身能产生的力矩。

        ``q`` 是**差模髋角**（``state.hip_differential``）：两腿同相位（共模）摆动
        不产生前后错开（#13）。
        """
        settings = self.settings
        scrub = settings.scrub_moment_coefficient * abs(float(np.sin(state.hip_differential)))
        budget = max(settings.yaw_moment_limit - scrub, settings.yaw_moment_floor)
        achievable = self.model.half_track * float(np.sum(force_limit))
        return float(min(budget, achievable))

    def stage_bounds(self, state: ChassisState) -> StageBounds:
        force_limit = self.wheel_force_limit(state)
        upper = np.tile(force_limit, (self.horizon, 1))
        moment = self.yaw_moment_limit(state, force_limit)
        enable = self.settings.enable_state_bounds
        extra_lower = extra_upper = None
        if self.model.input_dim > INPUT_DIM:
            width = self.model.input_dim - INPUT_DIM
            extra_upper = np.full((self.horizon, width), float(self.settings.hip_torque_limit))
            extra_lower = -extra_upper
        return StageBounds(
            force_lower=-upper,
            force_upper=upper,
            yaw_moment=np.full(self.horizon, moment, dtype=float),
            pitch_limit=self.settings.pitch_limit if enable else None,
            speed_limit=self.settings.speed_limit if enable else None,
            extra_lower=extra_lower,
            extra_upper=extra_upper,
            hip_angle_limit=(self.active_hip_limit
                             if enable and self.model.input_dim > INPUT_DIM else None),
        )

    # ---- 已知输入与状态预推 ----------------------------------------------
    def hip_drift(self, state: ChassisState) -> np.ndarray:
        """髋力矩折算成的**离散每步**漂移增量（方案 §1.5）。关闭补偿时返回零。

        ``hip_torque_drift`` 给的是连续角加速度，这里乘以 ``M(dt)`` 换成一步内的
        实际增量，量纲才与 QP 里的漂移项一致。
        """
        if not self.settings.compensate_hip_torque:
            return np.zeros(self.model.state_dim, dtype=float)
        return self._hip_drift_map @ self.model.hip_torque_drift(state.hip_torque)

    def contact_axis(self, state: ChassisState) -> ContactAxis:
        """当前髋角下的接触连线几何（方案 §1.1）。

        输入是**差模髋角**（``state.hip_differential``）：接触连线的前后倾斜
        由两轮前后错开产生，共模摆动不产生错开（#13）。
        """
        return self.geometry.axis(state.hip_differential)

    def predict_state(self, state: ChassisState) -> np.ndarray:
        """把反馈状态前推 ``prediction_lead``（方案 §3.4）。

        估计器交出来的状态是上一拍的，求解本身还要 1~2 ms；不补偿的话这部分相位
        滞后会直接吃掉稳定裕度。前推用上一拍解出的力做零阶保持，与 QP 自己的预测
        模型完全一致，所以不会引入新的模型误差。
        """
        vector = state.vector(self.model.state_dim)
        if self._lead_model is None:
            return vector
        return self._lead_model.A @ vector + self._lead_model.B @ self._previous_force + self._lead_model.w

    def _check_conditioning(self) -> float:
        """监控 Hessian 条件数，越限时只在跨越阈值那一次报警（方案 §1.1）。"""
        condition = self.qp.condition_number()
        limit = float(self.settings.condition_number_limit)
        if limit > 0.0 and condition > limit and not self.conditioning_warned:
            self.conditioning_warned = True
            _LOGGER.warning(
                "QP Hessian 条件数 %.3e 超过阈值 %.3e，缺失方向的正则可能不足"
                "（方案 §1.1）",
                condition,
                limit,
            )
        elif limit > 0.0 and condition <= limit:
            self.conditioning_warned = False
        return condition

    # ---- 主循环 ----------------------------------------------------------
    def _input_limits(self, bounds: StageBounds) -> tuple[np.ndarray, np.ndarray]:
        """完整的输入箱型上下界，形状 (horizon, input_dim)（E2 泛化）。"""
        lower = np.asarray(bounds.force_lower, dtype=float)
        upper = np.asarray(bounds.force_upper, dtype=float)
        if self.model.input_dim > 2 and bounds.extra_lower is not None:
            lower = np.hstack((lower, np.asarray(bounds.extra_lower, dtype=float)))
            upper = np.hstack((upper, np.asarray(bounds.extra_upper, dtype=float)))
        return lower, upper

    def _fallback_force(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        bounds: StageBounds,
    ) -> np.ndarray:
        force = -self.fallback_gain @ (state - reference)
        lower = np.full(self.model.input_dim, -np.inf)
        upper = np.full(self.model.input_dim, np.inf)
        lower[:2] = bounds.force_lower[0]
        upper[:2] = bounds.force_upper[0]
        if self.model.input_dim > 2 and bounds.extra_lower is not None:
            lower[2:] = bounds.extra_lower[0]
            upper[2:] = bounds.extra_upper[0]
        return np.clip(force, lower, upper)

    def predict(
        self,
        state: np.ndarray,
        forces: np.ndarray,
        drift: np.ndarray | None = None,
    ) -> np.ndarray:
        """用凝聚矩阵还原预测轨迹，返回 ``(N + 1, 6)``（含当前状态）。

        ``drift`` 必须与 ``solve`` 里传给 QP 的保持一致，否则这里画出来的俯仰轨迹
        会漏掉髋力矩那一段，看起来像"MPC 没看见扰动"。
        """
        # E2：Gamma 在缩放变量空间（ũ = u / s）构造，这里的力来自 u 空间，先换算。
        scaled = (np.asarray(forces, dtype=float).reshape(-1, self.model.input_dim)
                  / self.qp._input_scale[None, :])
        flat = (
            self.qp.Phi @ state
            + self.qp.Gamma @ scaled.ravel()
            + self.qp.drift_response(drift)
        )
        return np.vstack((state.reshape(1, self.model.state_dim),
                          flat.reshape(self.horizon, self.model.state_dim)))

    def solve(self, state: ChassisState, command: MotionCommand, *, timestamp: float = 0.0,
              feedback: ReferenceFeedback | None = None,
              disturbance_acceleration: np.ndarray | None = None) -> MPCSolution:
        """每个 MPC 周期调用一次；command 只接收速度和偏航角速度。"""
        # Known additive dynamics, ordered forward/pitch/yaw acceleration.
        # Compensation belongs inside prediction, not an unconstrained torque sum.
        continuous_disturbance = np.zeros(self.model.state_dim)
        if disturbance_acceleration is not None:
            acceleration = np.asarray(disturbance_acceleration, dtype=float)
            if acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
                raise ValueError("disturbance_acceleration must contain three finite components")
            continuous_disturbance[[1, 3, 5]] = acceleration
        vector = self.predict_state(state) + self._disturbance_lead_map @ continuous_disturbance
        reference = self.planner.plan(
            state.position, state.yaw, command, self.settings.dt, self.horizon,
            measured_speed=state.velocity, measured_yaw_rate=state.yaw_rate,
            feedback=ReferenceFeedback(contact=state.contact, pitch=state.pitch,
                                       pitch_rate=state.pitch_rate) if feedback is None else feedback,
            lead=self.settings.prediction_lead,
        )
        vector[4] = reference.yaw[0] + wrap_angle(vector[4] - reference.yaw[0])
        reference_states = reference.states(self.model.state_dim)
        if self.settings.zmp_reference:
            self._apply_zmp_reference(reference_states, reference.acceleration)
        else:
            self._apply_hip_feedforward(reference_states, reference.acceleration)
            self._apply_pitch_feedforward(reference_states, reference.acceleration, reference.velocity)
        self._apply_pitch_bias(reference_states)
        self._update_hip_limit(state)
        bounds = self.stage_bounds(state)
        warm_start = np.vstack(
            (self._previous_trajectory[1:], self._previous_trajectory[-1:])
        ).ravel()

        drift = self.hip_drift(state) + self._disturbance_map @ continuous_disturbance
        forces, info = self.qp.solve(
            vector,
            reference_states,
            bounds,
            self._previous_force,
            warm_start,
            drift=drift,
        )
        condition_number = self._check_conditioning()

        fallback = not info.solved
        if fallback:
            self.consecutive_failures += 1
            threshold = self.settings.max_consecutive_failures
            if self.consecutive_failures < threshold:
                # 一级兜底：沿用上一拍解的下一格（方案 §3.4）。比直接跳到 LQR 平滑得多，
                # 也保住了 MPC 之前那一步的规划意图；代价是响应慢一拍。
                # 必须按**本拍**的接触约束裁剪：上一拍还在用力的轮子这一拍可能已经离地，
                # 沿用下来的力会违反单边性。逐轮箱型是无条件安全约束。
                lower_limits, upper_limits = self._input_limits(bounds)
                forces = np.clip(
                    np.vstack((self._previous_trajectory[1:], self._previous_trajectory[-1:])),
                    lower_limits,
                    upper_limits,
                )
            else:
                # 二级兜底：连续失败达阈值，退到同一降阶模型上的连续 LQR 增益。
                forces = np.tile(
                    self._fallback_force(vector, reference_states[0], bounds),
                    (self.horizon, 1),
                )
            force = forces[0]
            _LOGGER.debug(
                "MPC 求解未成功(%s)，连续失败 %d 次，本拍走%s",
                info.status,
                self.consecutive_failures,
                "上一拍解的下一格" if self.consecutive_failures < threshold else "LQR 兜底",
            )
            if self.consecutive_failures == 1:
                _LOGGER.warning(
                    "MPC 求解未成功(%s)，本拍起走兜底通路（连续 %d 拍才退 LQR）",
                    info.status,
                    threshold,
                )
            if self.consecutive_failures >= threshold and not self.degraded:
                self.degraded = True
                _LOGGER.error(
                    "MPC 连续 %d 拍求解失败，置位 degraded，上层应切回纯 LQR（需 reset() 复位）",
                    self.consecutive_failures,
                )
        else:
            if self.consecutive_failures:
                _LOGGER.info("MPC 恢复求解成功，连续失败计数清零")
            self.consecutive_failures = 0
            force = forces[0]

        self._previous_trajectory = np.asarray(forces, dtype=float).copy()
        self._previous_force = np.asarray(force, dtype=float).copy()
        half_track = self.model.half_track
        predicted = self.predict(vector, self._previous_trajectory, drift)
        return MPCSolution(
            force=self._previous_force.copy(),
            force_trajectory=self._previous_trajectory.copy(),
            wheel_torque=self.model.torque_from_force(force),
            drive_force=float(force[0] + force[1]),
            yaw_moment=float(half_track * (force[0] - force[1])),
            reference=reference,
            predicted_states=predicted,
            qp=info,
            contact=self.contact_axis(state),
            condition_number=condition_number,
            fallback=fallback,
            trajectory=OptimizedTrajectory(
                predicted, self._previous_trajectory, self.settings.dt,
                float(timestamp) + self.settings.prediction_lead,
                state_dim=self.model.state_dim,
                input_dim=self.model.input_dim,
                position_columns=self.model.position_columns,
                velocity_columns=self.model.velocity_columns,
            ),
        )
