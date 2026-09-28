"""低频六状态 MPC：速度指令 -> 持久积分参考 -> 优化状态与等效轮力。

WBC 应采样 OptimizedTrajectory，而非直接跟踪优化前的参考。
降阶模型、髋力矩漂移和接触箱约束存在独立的适用边界；不能把本模块
视为已具备完整离地/失能管理的应用控制器。参考暂停时的控制退出须由
装配层消费 planner.valid，并配合接触模式处理，当前尚未接入 APP。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from Algorithm.geometry import wrap_angle

from .contact import ContactAxis, ContactGeometry
from .model import INPUT_DIM, STATE_DIM, DiscreteModel, ReducedModel
from .planner import MotionCommand, MotionLimits, ReferenceTrajectory, VelocityPlanner, ReferenceFeedback
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
    speed_limit: float = 1.60
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
        self.settings = settings or MPCSettings()
        self.discrete = model.discretize(self.settings.dt)
        self.qp = CondensedMPCQP(
            self.settings.horizon,
            self.settings.weights,
            self.settings.osqp_settings,
            input_dim=model.input_dim,
        )
        self.qp.set_model(self.discrete, model.half_track)
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

    def reset(self, state: ChassisState | None = None) -> None:
        """重新锚定内部状态（仿真复位、状态机切换后必须调用）。"""
        self.planner.reset(
            speed=0.0 if state is None else state.velocity,
            yaw_rate=0.0 if state is None else state.yaw_rate,
            position=None if state is None else state.position,
            yaw=None if state is None else state.yaw,
        )
        self._previous_force[:] = 0.0
        self._previous_trajectory[:] = 0.0
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
            hip_angle_limit=(self.settings.hip_angle_limit
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
