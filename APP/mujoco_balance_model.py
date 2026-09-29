"""Local rolling model with the same held-hip assumption as zero-speed WBC.

No gains are fitted: matrices come from constrained rigid-body equations using
MuJoCo's mass, gravity, geometry and actuator transmissions. This model is local
to flat-ground parking, and is not a model schedule for leg motion.
"""
from __future__ import annotations

import logging

import math

import numpy as np
import mujoco
from dataclasses import dataclass
from scipy.optimize import brentq, fsolve, least_squares

from Algorithm.mpc import ReducedModel
from Algorithm.mpc.model import wheel_half_track, yaw_inertia_from_model

_LOGGER = logging.getLogger(__name__)


class HeldHipModelBuilder:
    """Eliminate ideal hip-hold reactions before linearizing wheel inputs."""

    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.robot = dynamics.robot
        self.model = self.robot.model
        self.pose = self.robot.data.qpos.copy()
        self.hip_rows = np.eye(self.model.nv)[[
            int(self.model.joint(name).dofadr[0]) for name in ("joint1", "joint3")]]
        self.radius = float(self.model.geom_size[dynamics.wheels[0], 0])

    def derivative(self, state, force):
        data = self.robot.data
        data.qpos[:] = self.pose
        data.qpos[0] += state[0]
        mujoco.mju_quatIntegrate(data.qpos[3:7], np.array([0., state[2], state[4]]), 1.)
        data.qvel[:] = 0.
        self.robot.forward()
        rows, _, blocks = self.dynamics.matrices((True, True))
        contact = np.vstack([b.jacobian for b in blocks])
        constraints = np.vstack((contact, self.hip_rows))
        # Three independent moving coordinates close the ten velocity equations.
        mapping = np.vstack((constraints, rows[[0, 2, 3]]))
        data.qvel[:] = np.linalg.solve(mapping, np.r_[np.zeros(len(constraints)), state[[1, 3, 5]]])
        self.robot.forward()
        rows, task_bias, blocks = self.dynamics.matrices((True, True))
        contact = np.vstack([b.jacobian for b in blocks])
        bias = np.r_[np.concatenate([b.bias for b in blocks]), np.zeros(2)]
        constraints = np.vstack((contact, self.hip_rows))
        mass = self.dynamics.mass()
        torque = np.zeros(self.model.nu)
        torque[int(self.model.actuator("j2_ctrl").id)] = self.radius * force[0]
        torque[int(self.model.actuator("j4_ctrl").id)] = -self.radius * force[1]
        kkt = np.block([[mass, -constraints.T],
                        [constraints, np.zeros((len(constraints), len(constraints)))]])
        acceleration = np.linalg.solve(kkt, np.r_[
            self.dynamics.selection.T @ torque - data.qfrc_bias + data.qfrc_passive, -bias])[:self.model.nv]
        result = np.empty(6)
        result[[0, 2, 4]] = state[[1, 3, 5]]
        result[[1, 3, 5]] = rows[[0, 2, 3]] @ acceleration + task_bias[[0, 2, 3]]
        return result

    def build(self):
        data = self.robot.data
        qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        try:
            epsilon = 1e-5
            # A derivative alone drops gravity's affine term when the nominal
            # pose is not equilibrium. Determine the nearby zero-input trim.
            def pitch_acceleration(pitch):
                state = np.zeros(6)
                state[2] = pitch
                return self.derivative(state, np.zeros(2))[3]
            equilibrium_pitch = brentq(pitch_acceleration, -0.01, 0.01, xtol=1e-14)
            mujoco.mju_quatIntegrate(self.pose[3:7], np.array([0., equilibrium_pitch, 0.]), 1.)
            a = np.column_stack([(self.derivative(np.eye(6)[i]*epsilon, np.zeros(2))
                                  - self.derivative(-np.eye(6)[i]*epsilon, np.zeros(2)))/(2*epsilon)
                                 for i in range(6)])
            b = np.column_stack([(self.derivative(np.zeros(6), np.eye(2)[i]*epsilon)
                                  - self.derivative(np.zeros(6), -np.eye(2)[i]*epsilon))/(2*epsilon)
                                 for i in range(2)])
            self.affine = self.derivative(np.zeros(6), np.zeros(2))
            # Explicit residual is retained for diagnostics, not fitted away.
            track = wheel_half_track(self.model, data)
            inertia = yaw_inertia_from_model(self.model, data, self.radius, track)
            return ReducedModel(a, b, self.radius, track, inertia, source="mujoco-held-hips",
                                metadata={"nominal_pitch_acceleration": float(self.affine[3]),
                                          "nominal_forward_acceleration": float(self.affine[1]),
                                          "equilibrium_pitch": equilibrium_pitch})
        finally:
            data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, ctrl
            self.robot.forward()


@dataclass(frozen=True)
class HipInputLinearization:
    """T7a（A3）：共模髋为状态、共模髋力矩为输入的 8 状态线性化。

    只用于建模核验，未接入控制路径；A4 再决定维度泛化方式（不改
    ``Algorithm/mpc/model.py`` 的 ``STATE_DIM / INPUT_DIM`` 常量）。
    """

    A: np.ndarray
    B: np.ndarray
    affine: np.ndarray
    metadata: dict
    state_names: tuple
    input_names: tuple


class HipInputModelBuilder:
    """T7a（A3）：把共模髋角/角速度作为状态、共模髋力矩作为输入的线性化。

    与 :class:`HeldHipModelBuilder` 的区别只在约束集合：差模髋（决定两轮前后
    错开 0.12*sin(q_diff)）仍用速度约束握住，共模髋自由、由 ``tau_c`` 驱动。

    状态 ``[s, v, theta, theta_dot, psi, psi_dot, q_c, q_c_dot]``，
    输入 ``[f_right, f_left, tau_c]``。``q_c`` 与 ``tau_c`` 的符号与 APP 的
    共模定义一致（``0.5*(q1 + sign*q3)``，当前 ``sign = -1``）；
    ``tau_c`` 以 ``j1_ctrl = +tau_c``、``j3_ctrl = sign*tau_c`` 施加，
    对应共模广义力 ``2*tau_c``（差模分量为零）。
    """

    STATE_NAMES = ("s", "v", "theta", "theta_dot", "psi", "psi_dot", "q_c", "q_c_dot")
    INPUT_NAMES = ("f_right", "f_left", "tau_c")

    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.robot = dynamics.robot
        self.model = self.robot.model
        self.pose = self.robot.data.qpos.copy()
        self.hip_q = [int(self.model.joint(name).qposadr[0]) for name in ("joint1", "joint3")]
        self.hip_dof = [int(self.model.joint(name).dofadr[0]) for name in ("joint1", "joint3")]
        self.hip_rows = np.eye(self.model.nv)[self.hip_dof]
        self.radius = float(self.model.geom_size[dynamics.wheels[0], 0])
        axis1 = np.asarray(self.model.joint("joint1").axis, dtype=float)
        axis3 = np.asarray(self.model.joint("joint3").axis, dtype=float)
        self.hip_sign = float(np.dot(axis1, axis3))
        if not np.isclose(self.hip_sign, -1.0):
            raise ValueError(f"髋轴方向关系变化（dot={self.hip_sign}），共模/差模定义需要复核")
        # q_c = 0.5*(q1 + sign*q3) 的角速度行；q_diff = 0.5*(q1 - sign*q3) 的角速度行
        self.common_row = 0.5 * (self.hip_rows[0] + self.hip_sign * self.hip_rows[1])
        self.differential_row = 0.5 * (self.hip_rows[0] - self.hip_sign * self.hip_rows[1])

    def _set_hip(self, q_c: float) -> None:
        """按共模角设置两髋位形（差模保持在 0）：q1 = q_c、q3 = sign*q_c。"""
        data = self.robot.data
        data.qpos[self.hip_q[0]] = float(q_c)
        data.qpos[self.hip_q[1]] = self.hip_sign * float(q_c)

    def derivative(self, state: np.ndarray, force: np.ndarray) -> np.ndarray:
        """连续时间导数（8 状态）。state=[s,v,theta,theta_dot,psi,psi_dot,q_c,q_c_dot]，
        force=[f_right,f_left,tau_c]。"""
        state = np.asarray(state, dtype=float).reshape(8)
        force = np.asarray(force, dtype=float).reshape(3)
        data = self.robot.data
        data.qpos[:] = self.pose
        data.qpos[0] += state[0]
        mujoco.mju_quatIntegrate(data.qpos[3:7], np.array([0.0, state[2], state[4]]), 1.0)
        self._set_hip(state[6])
        data.qvel[:] = 0.0
        self.robot.forward()
        rows, _, blocks = self.dynamics.matrices((True, True))
        contact = np.vstack([b.jacobian for b in blocks])
        constraints = np.vstack((contact, self.differential_row[None, :]))
        # 十个速度方程：接触 + 差模髋 + 前进/俯仰/偏航 + 共模髋
        mapping = np.vstack((constraints, rows[[0, 2, 3]], self.common_row[None, :]))
        target = np.r_[np.zeros(len(constraints)), state[[1, 3, 5]], state[7]]
        data.qvel[:] = np.linalg.solve(mapping, target)
        self.robot.forward()
        rows, task_bias, blocks = self.dynamics.matrices((True, True))
        contact = np.vstack([b.jacobian for b in blocks])
        bias = np.r_[np.concatenate([b.bias for b in blocks]), 0.0]
        constraints = np.vstack((contact, self.differential_row[None, :]))
        mass = self.dynamics.mass()
        torque = np.zeros(self.model.nu)
        torque[int(self.model.actuator("j2_ctrl").id)] = self.radius * force[0]
        torque[int(self.model.actuator("j4_ctrl").id)] = -self.radius * force[1]
        torque[int(self.model.actuator("j1_ctrl").id)] = force[2]
        torque[int(self.model.actuator("j3_ctrl").id)] = self.hip_sign * force[2]
        kkt = np.block([[mass, -constraints.T],
                        [constraints, np.zeros((len(constraints), len(constraints)))]])
        acceleration = np.linalg.solve(kkt, np.r_[
            self.dynamics.selection.T @ torque - data.qfrc_bias + data.qfrc_passive,
            -bias])[:self.model.nv]
        result = np.empty(8)
        result[[0, 2, 4, 6]] = state[[1, 3, 5, 7]]
        result[[1, 3, 5]] = rows[[0, 2, 3]] @ acceleration + task_bias[[0, 2, 3]]
        result[7] = float(self.common_row @ acceleration)
        return result

    def trim(self, hip_angle: float):
        """工作点 (theta_trim, tau_trim)：零轮力下 theta_ddot = 0 且 q_c_ddot = 0。

        髋需要持续保持力矩时，该力矩记入 metadata 与 affine（不拟合掉）。
        """
        def residual(x):
            theta, tau = x
            state = np.zeros(8)
            state[2] = theta
            state[6] = hip_angle
            result = self.derivative(state, np.array([0.0, 0.0, tau]))
            return np.array([result[3], result[7]])

        solution, _, converged, message = fsolve(residual, np.array([0.0, 0.0]), full_output=True)
        if not converged:
            raise RuntimeError(f"工作点求解未收敛：{message}")
        return float(solution[0]), float(solution[1])

    def steady_state(self, acceleration: float, theta_weight: float = 1.0e6):
        """B1（#14）：沿稳态加速度族求工作点 (theta, q_c, F, tau_c)。

        非线性求解 theta_ddot = 0、q_c_ddot = 0、v_dot = a（两轮取相同轮力），
        欠定部分按 theta 加权最小二乘处理（theta_weight 越大越倾向 theta = 0）。
        初值取 a = 0.9 附近的经验比例（A3 稳态分解 + T8）。
        """
        weight = math.sqrt(max(float(theta_weight), 0.0))
        scale = float(acceleration) / 0.9
        guess = np.array([0.0, 0.37 * scale, 5.0 * scale, 0.32 * scale])

        def residual(y):
            theta, q_c, force_each, tau = (float(v) for v in y)
            state = np.zeros(8)
            state[2] = theta
            state[6] = q_c
            derivative = self.derivative(state, np.array([force_each, force_each, tau]))
            return np.array([derivative[3], derivative[7],
                             derivative[1] - float(acceleration), weight * theta])

        solution = least_squares(residual, guess, xtol=1e-13, ftol=1e-13, gtol=1e-13)
        theta, q_c, force_each, tau = (float(v) for v in solution.x)
        return dict(theta=theta, q_c=q_c, force=2.0 * force_each, tau_c=tau,
                    force_per_wheel=force_each, cost=float(solution.cost),
                    optimality=float(solution.optimality), success=bool(solution.success),
                    residual=float(np.linalg.norm(residual(solution.x))))

    def build_at(self, state: np.ndarray, force: np.ndarray) -> HipInputLinearization:
        """B1：在任意状态/输入处中心差分线性化（复用 build 的差分结构）。"""
        data = self.robot.data
        qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        try:
            epsilon = 1.0e-5
            state = np.asarray(state, dtype=float).reshape(8)
            force = np.asarray(force, dtype=float).reshape(3)
            affine = self.derivative(state, force)
            A = np.column_stack([
                (self.derivative(state + epsilon * np.eye(8)[i], force)
                 - self.derivative(state - epsilon * np.eye(8)[i], force)) / (2.0 * epsilon)
                for i in range(8)])
            B = np.column_stack([
                (self.derivative(state, force + epsilon * np.eye(3)[i])
                 - self.derivative(state, force - epsilon * np.eye(3)[i])) / (2.0 * epsilon)
                for i in range(3)])
            return HipInputLinearization(
                A=A, B=B, affine=affine,
                metadata={"theta": float(state[2]), "q_c": float(state[6])},
                state_names=self.STATE_NAMES, input_names=self.INPUT_NAMES)
        finally:
            data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, ctrl
            self.robot.forward()

    # A1（第 15 次排查）：稳态加速度能力表的取值范围与步长。
    # 依据：B1/M3 的稳态族（a 从 0 到 2.4），超过 2.4 时 q_c 已接近机构极限。
    ACCEL_CAPACITY_STEP = 0.1
    ACCEL_CAPACITY_MAX = 2.4

    def accel_capacity_table(self, step: float = ACCEL_CAPACITY_STEP,
                             maximum: float = ACCEL_CAPACITY_MAX) -> dict[str, list[float]]:
        """A1：q_c -> a 的稳态能力表（每 0.1 m/s^2 一个点）。

        由 :meth:`steady_state` 沿稳态族求解（theta 加权最小二乘，倾向 theta=0），
        q_c 随 a 单调递增（B1/M3 已验证），供 WheelMPC 用 hip_angle_limit 反查
        持续加速度能力。返回值是纯 list，能直接进 metadata 并随 summary 序列化。
        """
        # steady_state 经 derivative 会改写 robot.data，必须恢复（同 build 的做法），
        # 否则后续仿真从被污染的状态起跑。
        data = self.robot.data
        qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        try:
            accelerations = np.arange(0.0, float(maximum) + 0.5 * float(step), float(step))
            a_values: list[float] = []
            q_values: list[float] = []
            for acceleration in accelerations:
                solution = self.steady_state(float(acceleration))
                if not solution["success"]:
                    raise RuntimeError(
                        f"稳态能力表在 a={acceleration:.2f} 处求解失败："
                        f"cost={solution['cost']:.3e} optimality={solution['optimality']:.3e}")
                q_c = float(solution["q_c"])
                # 实测：a >= 2.2 时解退化，q_c 饱和（约 1.35 rad）不再增加。
                # 插值要求 q_c 严格递增，所以截断到最后一个有效点。
                if q_values and q_c <= q_values[-1]:
                    _LOGGER.info("A1：能力表在 a=%.2f 处饱和（q_c=%.4f 不再增加），"
                                 "截断到 a=%.2f", acceleration, q_c, a_values[-1])
                    break
                a_values.append(float(acceleration))
                q_values.append(q_c)
            return {"a": a_values, "q_c": q_values}
        finally:
            data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, ctrl
            self.robot.forward()

    def build_reduced(self, hip_angle: float = 0.0) -> ReducedModel:
        """B3：把 8 状态线性化包装成 ReducedModel（供 WheelMPC 使用）。

        metadata 里放 ``equilibrium_pitch = theta_trim``（照旧键名，APP 无需分叉），
        以及 ``theta_trim / tau_trim / hip_angle`` 供参考。
        """
        linear = self.build(hip_angle)
        track = wheel_half_track(self.model, self.robot.data)
        inertia = yaw_inertia_from_model(self.model, self.robot.data, self.radius, track)
        metadata = dict(linear.metadata)
        metadata["equilibrium_pitch"] = float(linear.metadata["theta_trim"])
        # A1：稳态能力表（q_c -> a），供 WheelMPC 自动换算规划限值。
        metadata["accel_capacity"] = self.accel_capacity_table()
        return ReducedModel(
            linear.A, linear.B, self.radius, track, inertia,
            source="mujoco-hip-input",
            metadata=metadata,
        )

    def build(self, hip_angle: float = 0.0) -> HipInputLinearization:
        """在给定共模髋角工作点上做中心差分线性化（A 8x8、B 8x3）。"""
        data = self.robot.data
        qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        try:
            epsilon = 1.0e-5
            theta_trim, tau_trim = self.trim(hip_angle)
            trim_state = np.zeros(8)
            trim_state[2] = theta_trim
            trim_state[6] = hip_angle
            trim_force = np.array([0.0, 0.0, tau_trim])
            affine = self.derivative(trim_state, trim_force)
            A = np.column_stack([
                (self.derivative(trim_state + epsilon * np.eye(8)[i], trim_force)
                 - self.derivative(trim_state - epsilon * np.eye(8)[i], trim_force)) / (2.0 * epsilon)
                for i in range(8)])
            B = np.column_stack([
                (self.derivative(trim_state, trim_force + epsilon * np.eye(3)[i])
                 - self.derivative(trim_state, trim_force - epsilon * np.eye(3)[i])) / (2.0 * epsilon)
                for i in range(3)])
            return HipInputLinearization(
                A=A,
                B=B,
                affine=affine,
                metadata={"hip_angle": float(hip_angle), "theta_trim": theta_trim,
                          "tau_trim": tau_trim,
                          "nominal_pitch_acceleration": float(affine[3]),
                          "nominal_hip_acceleration": float(affine[7])},
                state_names=self.STATE_NAMES,
                input_names=self.INPUT_NAMES,
            )
        finally:
            data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, ctrl
            self.robot.forward()
