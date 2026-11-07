"""解析被控对象的 MPC/WBC 多频率联调。

MPC 输出优化后的六状态轨迹；WBC 跟踪同一时间采样的状态与加速度。
默认关闭实验力前馈。测试对象为过驱动平面模型，不能替代真实 MuJoCo 验证。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from Algorithm.mpc import (
    ChassisState,
    MPCSettings,
    MotionCommand,
    QPWeights,
    ReducedModel,
    WheelMPC,
)
from Algorithm.wbc.projection import ContactElimination, NullSpaceHierarchy, TaskLevel
from Algorithm.wbc.sns import ActuatorLimits, SaturationResolver
from Algorithm.timing import ControlTiming
from Algorithm.supervision import WbcFeedbackAccumulator

from tests.planar_robot import PlanarTwoWheelRobot

# 法向力的标称分配：两个轮子对半担体重
NOMINAL_NORMAL_SHARE = 0.5
DEFAULT_HALF_TRACK = 0.12
DEFAULT_YAW_INERTIA = 0.09


def total_weight(robot: PlanarTwoWheelRobot) -> float:
    return float(sum(robot._masses) * robot.gravity)


def normal_rows(robot: PlanarTwoWheelRobot) -> np.ndarray:
    """四维接触力里法向分量所在的下标（切向 0/2、法向 1/3）。"""
    del robot
    return np.array([1, 3])


def find_static_equilibrium(robot: PlanarTwoWheelRobot, q: np.ndarray):
    """求 ``q̈ = 0`` 的静态解 ``(τ0, f0, 残差)``。

    ``Sᵀτ + Jcᵀf = h`` 是 7 个方程、8 个未知量，有 1 维解族。取**最小范数解**：
    它必须用 ``lstsq`` 精确满足方程（不能用加权最小二乘——方程系数跨度从 0.03 到 96，
    加权会被先验项带偏，残差压不下去，得到的根本不是平衡态）。

    对称站姿下最小范数解就是自然解：髋力矩撑住外张的腿、两轮法向各担一半体重、
    切向约为零。解族的方向只改变**切向力**（法向沿解族恒定，实测如此），所以沿解族
    怎么走都调不出更好的法向——法向为负时只能说明位形本身不对称（夹具的
    ``standing_pose`` 曾经用同号髋角，两条腿朝同一侧倾，就是这个毛病）。
    """
    selection = robot.selection()
    contact = robot.contact_jacobian(q)
    gravity = robot.gravity_bias(q)
    actuator_count = selection.shape[0]

    system = np.hstack([selection.T, contact.T])
    solution, *_ = np.linalg.lstsq(system, gravity, rcond=None)
    return (
        solution[:actuator_count],
        solution[actuator_count:],
        float(np.linalg.norm(system @ solution - gravity)),
    )


def rolling_velocity(robot: PlanarTwoWheelRobot, q: np.ndarray, x_rate: float, pitch_rate: float):
    """构造一个满足接触约束的整机速度。

    直接往 ``q̇`` 里塞 ẋ/θ̇ 会被接触投影掉（接触点的速度必须为零），所以先把它落到
    接触零空间里再返回。线性化时扰动速度要用这个。
    """
    contact = robot.contact_jacobian(q)
    # 同时固定被线性化的状态分量；单纯投影种子会改变 x_rate/pitch_rate。
    selected = np.eye(7)[[0, 2, 1]]
    return np.linalg.solve(np.vstack((contact, selected)),
                           np.array([0.0, 0.0, 0.0, 0.0, x_rate, pitch_rate, 0.0]))


def linearize_plant(robot: PlanarTwoWheelRobot, q0: np.ndarray, torque0: np.ndarray):
    """在平衡位形上线性化出 MPC 要的 ``A_long`` / ``B_long``。

    状态序沿用 ``Algorithm.lqr`` 的 ``[x, pitch, x_dot, pitch_dot]``；输入是
    **差动模式单轮力矩** ``u``（两轮各加 ``u``，净地面力 ``F = 2u/R``），与
    :meth:`ReducedModel.from_longitudinal` 的约定一致。
    """
    position_step = 1.0e-6
    velocity_step = 1.0e-6
    torque_step = 1.0e-3
    zero_velocity = np.zeros(7)

    def derivative(qpos: np.ndarray, qvel: np.ndarray, torque: np.ndarray) -> np.ndarray:
        qacc, _ = robot.constrained_acceleration(qpos, qvel, torque)
        return np.array([qvel[0], qvel[2], qacc[0], qacc[2]])

    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    plus_q, minus_q = q0.copy(), q0.copy()
    plus_q[2] += position_step
    minus_q[2] -= position_step
    A[:, 1] = (
        derivative(plus_q, zero_velocity, torque0) - derivative(minus_q, zero_velocity, torque0)
    ) / (2.0 * position_step)
    A[:, 2] = (
        derivative(q0, rolling_velocity(robot, q0, velocity_step, 0.0), torque0)
        - derivative(q0, rolling_velocity(robot, q0, -velocity_step, 0.0), torque0)
    ) / (2.0 * velocity_step)
    A[:, 3] = (
        derivative(q0, rolling_velocity(robot, q0, 0.0, velocity_step), torque0)
        - derivative(q0, rolling_velocity(robot, q0, 0.0, -velocity_step), torque0)
    ) / (2.0 * velocity_step)

    plus_torque, minus_torque = torque0.copy(), torque0.copy()
    plus_torque[1] += torque_step
    plus_torque[3] += torque_step
    minus_torque[1] -= torque_step
    minus_torque[3] -= torque_step
    B = np.zeros((4, 1))
    B[:, 0] = (
        derivative(q0, zero_velocity, plus_torque) - derivative(q0, zero_velocity, minus_torque)
    ) / (2.0 * torque_step)
    return A, B


def build_mpc(robot: PlanarTwoWheelRobot, q0: np.ndarray, torque0: np.ndarray, **overrides):
    """由虚拟被控对象的实测线性化装配一个 MPC。

    这里固定标称髋力矩做局部差分，不是髋角保持控制的闭环辨识。
    解析模型与真实 MuJoCo 的参数和驱动权限不同，两者不可互换。
    """
    A_long, B_long = linearize_plant(robot, q0, torque0)
    model = ReducedModel.from_longitudinal(
        A_long,
        B_long,
        wheel_radius=robot.radius,
        half_track=DEFAULT_HALF_TRACK,
        yaw_inertia=DEFAULT_YAW_INERTIA,
        source="planar-two-wheel",
        metadata={"unstable_pole": float(np.sqrt(max(A_long[3, 1], 0.0)))},
    )
    weights = QPWeights()
    settings = MPCSettings(weights=weights, **overrides)
    return WheelMPC(model, settings), model, (A_long, B_long)


class PlanarWbc:
    """方案 §4.3 的 L0/L1/L4/L5/L6 在平面双腿虚拟被控对象上的装配。"""

    def __init__(
        self,
        robot: PlanarTwoWheelRobot,
        *,
        pitch_gains: tuple[float, float] = (80.0, 10.0),
        height_gains: tuple[float, float] = (100.0, 12.0),
        velocity_gain: float = 4.0,
        regularize_gains: tuple[float, float] = (60.0, 6.0),
        torque_limits: float = 40.0,
        nominal_pose: tuple[float, ...] = (0.6, 0.0, -0.6, 0.0),
        nominal_height: float | None = None,
        force_feedforward: bool = False,
        feedforward_tasks: tuple[str, ...] = ("forward_velocity",),
    ) -> None:
        self.robot = robot
        self.pitch_gains = pitch_gains
        self.height_gains = height_gains
        self.velocity_gain = velocity_gain
        self.regularize_gains = regularize_gains
        self.torque_limits = torque_limits
        self.nominal_pose = np.array(nominal_pose, dtype=float)[[0, 2]]  # 只留髋
        self.nominal_height = float(robot.standing_pose()[1] if nominal_height is None else nominal_height)
        self.force_feedforward = force_feedforward
        self.feedforward_tasks = tuple(feedforward_tasks)
        self.joint_columns = (3, 4, 5, 6)
        self._resolver = None
        self._elimination = None

    # ---- 每拍重建：位形变了矩阵就变了 ---------------------------------
    def _rebuild(self, q: np.ndarray) -> None:
        mass = self.robot.mass_matrix(q)
        contact = self.robot.contact_jacobian(q)
        self._elimination = ContactElimination.from_matrices(
            contact, self.robot.selection(), mass
        )
        self._resolver = SaturationResolver(NullSpaceHierarchy(mass), self._elimination)

    def contact_force_reference(self, mpc_force: np.ndarray) -> np.ndarray:
        """把 MPC 的两路纵向力摆到四维接触力的切向位置上，**法向留 0**。

        法向那一格故意不填：MPC 只决定纵向力，支撑体重是接触约束（也就是 WBC 与
        地面）的事。若把法向也填成 ``mg/2``，通过 :meth:`force_acceleration` 换算
        出来就是一份向上的加速度，等于把"撑住体重"重复计了一次——实测基线力矩会从
        ``[-1.4, -1.5, 1.4, 1.5]`` 被顶到 ``[-5.2, -5.5, 5.2, 5.5]``。
        """
        return np.array(
            [
                float(mpc_force[0]),
                0.0,
                float(mpc_force[1]),
                0.0,
            ]
        )

    def force_acceleration(self, q: np.ndarray, force: np.ndarray) -> np.ndarray:
        """MPC 的接触力折算成的关节加速度增量 ``M⁻¹Jcᵀf*``（§4.3 的 L0）。

        **不能**把 ``f*`` 当成广义力加进 ``h``（也就是写成 ``h − Jcᵀf*``）：QR 消力的
        构造保证 ``Q_uᵀJcᵀ ≡ 0``，而执行器解正是经过 ``P = (Q_uᵀSᵀ)⁺Q_uᵀ`` 投影出来的，
        所以接触力的广义力对执行器解**恒等于零**（实测 ``P·Jcᵀ = 1.3e-17``）。力只有
        经过 ``M⁻¹``、以加速度形式出现才能影响解。

        所以 L0 的正确形态是：把它产生的那份加速度投影到各级任务空间上叠加，见
        :meth:`solve`。
        """
        robot = self.robot
        return np.linalg.solve(
            robot.mass_matrix(q), robot.contact_jacobian(q).T @ np.asarray(force, dtype=float)
        )

    # ---- 主解算 --------------------------------------------------------
    def solve(
        self,
        q: np.ndarray,
        qvel: np.ndarray,
        mpc_force: np.ndarray,
        pitch_reference: tuple[float, float, float],
        velocity_reference: tuple[float, float],
        *, forward_acceleration: float = 0.0,
    ) -> dict:
        """跑一拍 WBC，返回关节力矩与诊断量。"""
        self._rebuild(q)
        theta, theta_rate = float(q[2]), float(qvel[2])
        pitch_target, pitch_rate_target, pitch_acc_target = pitch_reference
        velocity, velocity_target = velocity_reference

        kp_pitch, kd_pitch = self.pitch_gains
        kp_height, kd_height = self.height_gains
        kp_reg, kd_reg = self.regularize_gains
        # 只正则髋关节：轮关节角会随行驶距离一直累计（滚动），把它拉回标称值
        # 会产生 −1500 rad/s² 量级的荒唐目标，实测就是这么把滚动顶住的。
        hip_columns = (3, 5)
        regularizer = np.zeros((len(hip_columns), 7))
        for row, column in enumerate(hip_columns):
            regularizer[row, column] = 1.0

        tasks = [
            # L1 俯仰：θ*, θ̇*, θ̈* 全前馈（§4.3）
            TaskLevel(
                jacobian=np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
                desired_acceleration=np.array(
                    [
                        pitch_acc_target
                        + kp_pitch * (pitch_target - theta)
                        + kd_pitch * (pitch_rate_target - theta_rate)
                    ]
                ),
                name="pitch",
            ),
            # L4 机身高度
            TaskLevel(
                jacobian=np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                desired_acceleration=np.array(
                    [
                        kp_height * (self.nominal_height - float(q[1]))
                        + kd_height * (-float(qvel[1]))
                    ]
                ),
                name="height",
            ),
            # L5：优化轨迹的线加速度前馈与速度误差反馈。实验力前馈开启时不重复叠加。
            TaskLevel(
                jacobian=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                desired_acceleration=np.array([
                    (0.0 if self.force_feedforward else forward_acceleration)
                    + self.velocity_gain * (velocity_target - velocity)
                ]),
                name="forward_velocity",
            ),
            # L6 关节正则：把关节拉回标称位形，填满剩下的零空间
            TaskLevel(
                jacobian=regularizer,
                desired_acceleration=kp_reg * (self.nominal_pose - q[list(hip_columns)])
                + kd_reg * (-qvel[list(hip_columns)]),
                name="regularization",
            ),
        ]

        force_reference = self.contact_force_reference(mpc_force)
        if self.force_feedforward:
            # L0：把 MPC 的接触力折算成加速度，投影叠加到指定的任务上。
            #
            # 只喂给前向速度那一级，**不能**也加到俯仰任务上：MPC 的力同时含有
            # "往前推"和"往前撅"两重效果，而俯仰已经有 ``θ*`` 这条驱动通路了，
            # 两边一起喂会互相抵消——实测 L1 的纠正项 −11.0 rad/s² 被 L0 的
            # +11.1 rad/s² 抹平，俯仰卡在 0.135 rad 稳态误差上不动。
            # 方案 §0 的结论也是这个：俯仰的驱动信号是 ``θ*``，力负责前向。
            acceleration = self.force_acceleration(q, force_reference)
            tasks = [
                replace(
                    level,
                    desired_acceleration=level.desired_acceleration
                    + level.jacobian @ acceleration,
                )
                if level.name in self.feedforward_tasks
                else level
                for level in tasks
            ]
        result = self._resolver.resolve(
            self.robot.contact_jacobian(q),
            self.robot.contact_bias(q, qvel),
            tasks,
            ActuatorLimits.symmetric(np.full(4, self.torque_limits)),
            dynamics_bias=self.robot.gravity_bias(q),
        )
        return {
            "torque": result.torque,
            "sns": result,
            "force_reference": force_reference,
            "tasks": {level.name: level for level in tasks},
        }


class JointLoop:
    """把 MPC、WBC 与虚拟被控对象接成一个环。

    时序：MPC 默认 50 Hz、WBC 默认 250 Hz、物理积分 1000 Hz。
    WBC 每次读取最新状态并采样优化轨迹，力矩在相邻 WBC 更新间保持。
    """

    def __init__(
        self,
        robot: PlanarTwoWheelRobot,
        mpc: WheelMPC,
        wbc: PlanarWbc,
        physics_dt: float = 0.001,
        torque_limits: float = 40.0,
        wbc_dt: float = 0.004,
    ) -> None:
        self.robot = robot
        self.mpc = mpc
        self.wbc = wbc
        self.timing = ControlTiming(physics_dt, wbc_dt, mpc.settings.dt)
        if mpc.settings.prediction_lead != 0.0:
            raise ValueError("无延迟解析联调要求 prediction_lead=0")
        self.physics_dt = self.timing.physics_dt
        self.torque_limits = float(torque_limits)

    def pitch_acceleration_reference(self, solution, elapsed: float = 0.0) -> float:
        """与同一时间采样的姿态和角速度一致的角加速度。"""
        return float(solution.trajectory.sample(elapsed).acceleration[1])

    def state_of(self, q: np.ndarray, qvel: np.ndarray) -> ChassisState:
        return ChassisState(
            position=float(q[0]),
            velocity=float(qvel[0]),
            pitch=float(q[2]),
            pitch_rate=float(qvel[2]),
            hip_angle=0.0,
            normal_forces=(NOMINAL_NORMAL_SHARE * total_weight(self.robot),) * 2,
        )

    def run(
        self,
        command: MotionCommand,
        ticks: int,
        q: np.ndarray,
        qvel: np.ndarray | None = None,
    ) -> dict:
        qvel = np.zeros(7) if qvel is None else np.asarray(qvel, dtype=float).copy()
        q = np.asarray(q, dtype=float).copy()
        steps = self.timing.mpc_steps
        control_history = {key: [] for key in ("time", "torque", "pitch_reference", "pitch", "scaling", "dynamics_residual", "fallback")}
        normal_minimum = float("inf")
        capability = WbcFeedbackAccumulator(("forward_velocity",))
        wbc_calls = 0
        history = {
            "reference_mode": [],
            "position_reference": [],
            "time": [],
            "position": [],
            "velocity": [],
            "pitch": [],
            "pitch_reference": [],
            "velocity_reference": [],
            "torque": [],
            "contact_force": [],
            "mpc_force": [],
            "force_reference": [],
            "scaling": [],
            "mpc_fallback": [],
            "mpc_iterations": [],
        }
        for tick in range(ticks):
            state = self.state_of(q, qvel)
            start_time = tick * self.mpc.settings.dt
            current_command = command(start_time) if callable(command) else command
            solution = self.mpc.solve(state, current_command, timestamp=start_time,
                                      feedback=capability.consume(state))
            history["reference_mode"].append(self.mpc.planner.mode.value)
            history["position_reference"].append(solution.reference.position[0])
            contact_force = np.zeros(4)
            for step in range(steps):
                if self.timing.wbc_due(step):
                    elapsed = step * self.physics_dt
                    sample = solution.trajectory.sample(elapsed)
                    wbc_out = self.wbc.solve(
                        q, qvel, sample.force,
                        (sample.state[2], sample.state[3], sample.acceleration[1]),
                        (float(qvel[0]), sample.state[1]),
                        forward_acceleration=float(sample.acceleration[0]),
                    )
                    torque = wbc_out["torque"]
                    capability.observe(wbc_out["sns"], ActuatorLimits.symmetric(
                        np.full(torque.size, self.wbc.torque_limits)))
                    if np.max(np.abs(torque)) > self.torque_limits + 1e-9:
                        raise RuntimeError("WBC 力矩超出联调执行器限制")
                    wbc_calls += 1
                    control_history["time"].append(start_time + elapsed)
                    control_history["torque"].append(torque.copy())
                    control_history["pitch_reference"].append(sample.state[2])
                    control_history["pitch"].append(q[2])
                    control_history["scaling"].append(dict(wbc_out["sns"].scaling))
                    control_history["dynamics_residual"].append(wbc_out["sns"].dynamics_residual_norm)
                    control_history["fallback"].append(wbc_out["sns"].fallback_used)
                q, qvel, _, contact_force = self.robot.step(q, qvel, torque, self.physics_dt)
                normal_minimum = min(normal_minimum, float(np.min(contact_force[[1, 3]])))

            history["time"].append(len(history["time"]) * self.mpc.settings.dt)
            history["position"].append(float(q[0]))
            history["velocity"].append(float(qvel[0]))
            history["pitch"].append(float(q[2]))
            history["pitch_reference"].append(solution.pitch_reference)
            history["velocity_reference"].append(solution.velocity_reference)
            history["torque"].append(torque.copy())
            history["contact_force"].append(contact_force.copy())
            history["mpc_force"].append(np.asarray(solution.force, dtype=float).copy())
            history["force_reference"].append(wbc_out["force_reference"].copy())
            history["scaling"].append(dict(wbc_out["sns"].scaling))
            history["mpc_fallback"].append(bool(solution.fallback))
            history["mpc_iterations"].append(int(solution.qp.iterations))

        return {
            "final": (q, qvel),
            "control_history": {key: np.asarray(value, dtype=object if key == "scaling" else None) for key, value in control_history.items()},
            "mpc_calls": ticks, "wbc_calls": wbc_calls,
            "minimum_normal_force": normal_minimum,
            "history": {key: np.asarray(value, dtype=object) if key == "scaling" else np.asarray(value)
                        for key, value in history.items()},
        }
