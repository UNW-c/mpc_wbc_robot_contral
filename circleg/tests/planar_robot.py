"""MPC + WBC 联调用的虚拟被控对象：平面双腿轮式机器人。

``q = (x, z, θ, hip_R, wheel_R, hip_L, wheel_L)`` → ``nv = 7``；每条腿一个髋关节
加一个轮关节 → ``na = 4``；两个轮地接触各 2 行（切向 + 法向）→ ``nc = 4``。

**与真机的结构差异（必须说清楚）**：接触一致子空间还剩 ``7 − 4 = 3`` 维自由度，
执行器 4 个，所以是**过驱动**；真机是 ``10 − 5 = 5`` 维自由度、4 个执行器，
净欠驱动 1 维（俯仰）。本模型能验证接口handshake、任务层级与 SNS，但不复制真机的
欠驱动结构——用它下结论时只下"接口对得上、层级按预期工作"这类结论。

链：机身（质心在原点，左右髋各偏 ``hip_offset``）→ 髋角 ``hip`` → 大腿到轮心
（``thigh_length``）→ 轮角 ``wheel`` → 轮缘接触点（半径 ``radius``）。左右腿在 x 上
镜像。运动学、质量矩阵、重力偏置、接触偏置全部解析/半解析给出。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.conftest import WBC_GRAVITY, rotation, rotation_derivative


@dataclass
class PlanarTwoWheelRobot:
    body_mass: float = 8.0
    thigh_mass: float = 0.5
    wheel_mass: float = 0.4
    body_inertia: float = 0.05
    thigh_inertia: float = 0.004
    wheel_inertia: float = 0.006
    hip_offset: float = 0.05
    thigh_length: float = 0.15
    hip_drop: float = 0.20
    radius: float = 0.06
    gravity: float = WBC_GRAVITY

    def __post_init__(self) -> None:
        self._masses = (
            self.body_mass,
            self.thigh_mass,
            self.wheel_mass,
            self.thigh_mass,
            self.wheel_mass,
        )
        self._inertias = (
            self.body_inertia,
            self.thigh_inertia,
            self.wheel_inertia,
            self.thigh_inertia,
            self.wheel_inertia,
        )
        self._hip_offsets = {
            "R": np.array([self.hip_offset, -self.hip_drop]),
            "L": np.array([-self.hip_offset, -self.hip_drop]),
        }
        self._thigh_offsets = {
            "R": np.array([0.0, -self.thigh_length]),
            "L": np.array([0.0, -self.thigh_length]),
        }
        self._contact_offsets = {
            "R": np.array([0.0, -self.radius]),
            "L": np.array([0.0, -self.radius]),
        }

    # ---- 结构 ----------------------------------------------------------
    def _angles(self, q: np.ndarray) -> dict[str, tuple[float, float, float]]:
        theta = float(q[2])
        return {
            "R": (theta, theta + float(q[3]), theta + float(q[3]) + float(q[4])),
            "L": (theta, theta + float(q[5]), theta + float(q[5]) + float(q[6])),
        }

    @staticmethod
    def hip_index(leg: str) -> int:
        return 3 if leg == "R" else 5

    @staticmethod
    def wheel_index(leg: str) -> int:
        return 4 if leg == "R" else 6

    def _segment_offsets(self, leg: str, point: str) -> tuple[np.ndarray, ...]:
        if point == "body":
            return ()
        if point == "wheel":
            return (self._hip_offsets[leg], self._thigh_offsets[leg])
        if point == "contact":
            return (
                self._hip_offsets[leg],
                self._thigh_offsets[leg],
                self._contact_offsets[leg],
            )
        raise ValueError(f"未知的点 {point}")

    def point_position(self, q: np.ndarray, leg: str, point: str) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        angles = self._angles(q)[leg]
        position = np.array([q[0], q[1]], dtype=float)
        for segment, offset in enumerate(self._segment_offsets(leg, point)):
            position = position + rotation(angles[segment]) @ offset
        return position

    def point_jacobian(self, q: np.ndarray, leg: str, point: str) -> np.ndarray:
        """2×7 位置雅可比。

        绝对角依次是 ``θ``、``θ+hip``、``θ+hip+wheel``，所以 ``∂/∂θ`` 收集所有段、
        ``∂/∂hip`` 从第 2 段起、``∂/∂wheel`` 只作用在第 3 段。
        """
        q = np.asarray(q, dtype=float)
        angles = self._angles(q)[leg]
        hip_column = self.hip_index(leg)
        wheel_column = self.wheel_index(leg)
        jacobian = np.zeros((2, 7))
        jacobian[0, 0] = 1.0
        jacobian[1, 1] = 1.0
        for segment, offset in enumerate(self._segment_offsets(leg, point)):
            contribution = rotation_derivative(angles[segment]) @ offset
            jacobian[:, 2] += contribution
            if segment >= 1:
                jacobian[:, hip_column] += contribution
            if segment >= 2:
                jacobian[:, wheel_column] += contribution
        return jacobian

    def spin_row(self, leg: str) -> np.ndarray:
        """轮子**绝对**自转角速度在广义速度上的系数：``θ̇ + hiṗ + wheel̇``。"""
        row = np.zeros(7)
        row[2] = 1.0
        row[self.hip_index(leg)] = 1.0
        row[self.wheel_index(leg)] = 1.0
        return row

    def contact_jacobian(self, q: np.ndarray) -> np.ndarray:
        """两轮接触约束各 2 行，拼成 4×7：``[切向, 法向] × 2``。

        **平地接触点恒在轮心正下方**（世界系固定偏移 ``(0, −r)``），所以它的位置
        雅可比就是轮心雅可比——轮关节对接触点位置没有直接贡献。滚动无滑移靠
        *材料点*速度为 0 来表达::

            v_材料 = v_几何 + ω_轮 × (接触点 − 轮心) = v_几何 − r·ω_轮·x̂ = 0

        所以切向行是 ``轮心雅可比的 x 行 − r·(轮子绝对自转行)``，法向行就是轮心
        雅可比的 z 行。

        早先的写法把接触点定义成 ``轮心 + R(θ+hip+wheel)(0,−r)``（跟着轮子转），
        那会让轮关节与髋关节对接触点的作用方向**平行**（腿在站立位形上退化，
        俯仰被锁死、线性化后 ``B`` 的俯仰行恒为零），而且接触法向也跟着轮子转。
        """
        rows = []
        for leg in ("R", "L"):
            wheel_center = self.point_jacobian(q, leg, "wheel")
            rows.append(wheel_center[0] - self.radius * self.spin_row(leg))
            rows.append(wheel_center[1])
        return np.vstack(rows)

    def contact_points(self, q: np.ndarray) -> np.ndarray:
        """接触点位置：轮心正下方 ``r`` 处。"""
        q = np.asarray(q, dtype=float)
        return np.vstack(
            [
                self.point_position(q, leg, "wheel") + np.array([0.0, -self.radius])
                for leg in ("R", "L")
            ]
        )

    def selection(self) -> np.ndarray:
        """4×7：两个髋关节 + 两个轮关节。"""
        selection = np.zeros((4, 7))
        for row, column in enumerate((3, 4, 5, 6)):
            selection[row, column] = 1.0
        return selection

    # ---- 动力学 --------------------------------------------------------
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        matrix = np.zeros((7, 7))
        body = np.zeros((2, 7))
        body[0, 0] = 1.0
        body[1, 1] = 1.0
        matrix += self.body_mass * (body.T @ body)
        spin_body = np.zeros(7)
        spin_body[2] = 1.0
        matrix += self.body_inertia * np.outer(spin_body, spin_body)

        for leg in ("R", "L"):
            hip_column, wheel_column = self.hip_index(leg), self.wheel_index(leg)
            for point, mass, inertia in (
                ("wheel", self.thigh_mass, self.thigh_inertia),
                ("contact", self.wheel_mass, self.wheel_inertia),
            ):
                # 轮质量位于轮心；材料轮缘点只用于运动学，不能当作轮质心。
                jacobian = self.point_jacobian(q, leg, "wheel")
                matrix += mass * (jacobian.T @ jacobian)
                spin = np.zeros(7)
                spin[2] = 1.0
                spin[hip_column] = 1.0
                if point == "contact":
                    spin[wheel_column] = 1.0
                matrix += inertia * np.outer(spin, spin)
        return matrix

    def gravity_bias(self, q: np.ndarray) -> np.ndarray:
        bias = np.zeros(7)
        bias[1] = self.gravity * self.body_mass
        for leg in ("R", "L"):
            for point, mass in (("wheel", self.thigh_mass), ("contact", self.wheel_mass)):
                bias += self.gravity * mass * self.point_jacobian(q, leg, "wheel")[1]
        return bias

    def contact_bias(self, q: np.ndarray, qvel: np.ndarray, step: float = 1e-7) -> np.ndarray:
        """``J̇c q̇``：沿速度方向对接触雅可比做中心差分。"""
        q = np.asarray(q, dtype=float)
        qvel = np.asarray(qvel, dtype=float)
        plus = self.contact_jacobian(q + step * qvel)
        minus = self.contact_jacobian(q - step * qvel)
        return ((plus - minus) / (2.0 * step)) @ qvel

    # ---- 前向 ----------------------------------------------------------
    def constrained_acceleration(
        self, q: np.ndarray, qvel: np.ndarray, torque: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """等式接触约束下的加速度与接触力。

        ``M q̈ + h = Sᵀτ + Jcᵀf`` 与 ``Jc q̈ = −J̇c q̇`` 联立::

            q̈_free = M⁻¹(Sᵀτ − h)
            f = (JcM⁻¹Jcᵀ)⁻¹(−J̇c q̇ − Jc q̈_free)
            q̈ = q̈_free + M⁻¹Jcᵀf

        与 WBC 侧是同一套投影数学，所以两边对"接触力是多少"的理解必然一致——
        这正是联调能对上的前提。
        """
        mass = self.mass_matrix(q)
        bias_gravity = self.gravity_bias(q)
        contact = self.contact_jacobian(q)
        bias_contact = self.contact_bias(q, qvel)
        free = np.linalg.solve(
            mass, self.selection().T @ np.asarray(torque, dtype=float) - bias_gravity
        )
        gram = contact @ np.linalg.solve(mass, contact.T)
        force = np.linalg.solve(gram, -bias_contact - contact @ free)
        return free + np.linalg.solve(mass, contact.T @ force), force

    def step(self, q: np.ndarray, qvel: np.ndarray, torque: np.ndarray, dt: float):
        """半隐式欧拉推进一步，返回 ``(q, qvel, qacc, force)``。"""
        qacc, force = self.constrained_acceleration(q, qvel, torque)
        qvel_next = np.asarray(qvel, dtype=float) + dt * qacc
        q_next = np.asarray(q, dtype=float) + dt * qvel_next
        return q_next, qvel_next, qacc, force

    # ---- 便利量 --------------------------------------------------------
    def standing_pose(self, splay: float = 0.6, wheel_angle: float = 0.0) -> np.ndarray:
        """轮子刚好落地的**对称**站立位形（接触点 z 对齐地面）。

        ``splay`` 是左右腿向外张开的量：髋角取**相反符号**才是对称站姿。同号会让
        两条腿朝同一侧倾（接触点都跑到 x<0 去），那样连对称的静态平衡都不存在
        ——踩过这个坑，所以参数名从 ``hip_angle`` 改成了 ``splay``。
        """
        q = np.array([0.0, 0.0, 0.0, splay, wheel_angle, -splay, -wheel_angle])
        contact_z = float(np.min(self.contact_points(q)[:, 1]))
        q[1] = -contact_z
        return q
