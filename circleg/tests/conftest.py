"""MPC 单元测试的共享夹具：全部用模拟数据，不需要 MuJoCo 运行时。

``MEASURED_A_LONG`` / ``MEASURED_B_LONG`` 是从 MuJoCo 实测线性化里**记录下来**的
数值（``MuJoCoRobot.load("plane")``、q=0 双轮接触、髋关节自由），不是手推公式。
用它当主夹具而不是 ``ReducedModel.cart_pole``，因为 cart_pole 的人造参数条件数差
（实测 H cond ~9.3e4，cart_pole 会让 OSQP 撞 max_iter），测出来的失败不是代码问题。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from Algorithm.mpc import ReducedModel

# 记录自 MuJoCo：状态序 [x, pitch, x_dot, pitch_dot]，输入为差动模式单轮力矩
MEASURED_A_LONG = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 4.241716, 0.0, 0.000024],
        [0.0, 58.571747, 0.0, 0.000944],
    ],
    dtype=float,
)
MEASURED_B_LONG = np.array([[0.0], [0.0], [1.596313], [-22.811775]], dtype=float)
MEASURED_HALF_TRACK = 0.120
MEASURED_YAW_INERTIA = 0.09347262652009686
WHEEL_RADIUS = 0.160


def make_measured_model(yaw_damping: float = 0.0) -> ReducedModel:  # noqa: D103
    return ReducedModel.from_longitudinal(
        MEASURED_A_LONG,
        MEASURED_B_LONG,
        wheel_radius=WHEEL_RADIUS,
        half_track=MEASURED_HALF_TRACK,
        yaw_inertia=MEASURED_YAW_INERTIA,
        yaw_damping=yaw_damping,
        source="recorded-mujoco",
    )


@pytest.fixture
def measured_model() -> ReducedModel:
    return make_measured_model()


class FakeMjModel:
    """``yaw_inertia_from_model`` / ``wheel_half_track`` 需要的最小 MuJoCo 接口。"""

    def __init__(
        self,
        body_names: tuple[str, ...],
        body_mass: np.ndarray,
        body_inertia: np.ndarray,
        geom_names: tuple[str, ...],
    ) -> None:
        # 索引 0 是 worldbody，和真实 MuJoCo 一致
        self._body_index = {name: index + 1 for index, name in enumerate(body_names)}
        self.nbody = len(body_names) + 1
        self.body_mass = np.concatenate(([0.0], np.asarray(body_mass, dtype=float)))
        self.body_inertia = np.vstack((np.zeros(3), np.asarray(body_inertia, dtype=float)))
        self._geom_index = {name: index for index, name in enumerate(geom_names)}

    def body(self, name: str):
        return type("_Named", (), {"id": self._body_index[name]})()

    def geom(self, name: str):
        return type("_Named", (), {"id": self._geom_index[name]})()


class FakeMjData:
    def __init__(self, xipos: np.ndarray, geom_xpos: np.ndarray, nbody: int) -> None:
        self.xipos = np.vstack((np.zeros(3), np.asarray(xipos, dtype=float)))
        self.geom_xpos = np.asarray(geom_xpos, dtype=float)
        # 所有体都不旋转：主惯量的 zz 直接就是世界系 zz
        self.ximat = np.tile(np.eye(3).ravel(), (nbody, 1))


@pytest.fixture
def fake_mujoco() -> tuple[FakeMjModel, FakeMjData]:
    """两个点质量轮 + 一个机身，几何上对称，偏航惯量可以手算核对。"""
    body_names = ("base_link", "joint2_Link", "joint4_Link")
    body_mass = np.array([8.0, 1.0, 1.0])
    # 只有 I_yy（索引 1）被当作轮子自旋惯量，I_zz（索引 2）进复合惯量
    body_inertia = np.array(
        [
            [0.10, 0.10, 0.20],
            [0.02, 0.008, 0.02],
            [0.02, 0.008, 0.02],
        ]
    )
    geom_names = ("right_wheel_collision", "left_wheel_collision")
    model = FakeMjModel(body_names, body_mass, body_inertia, geom_names)
    xipos = np.array(
        [
            [0.0, 0.0, 0.30],
            [0.0, -0.12, 0.16],
            [0.0, +0.12, 0.16],
        ]
    )
    geom_xpos = np.array([[0.0, -0.12, 0.16], [0.0, +0.12, 0.16]])
    data = FakeMjData(xipos, geom_xpos, model.nbody)
    return model, data

# ---------------------------------------------------------------------------
# WBC 单元测试用的解析模型
# ---------------------------------------------------------------------------

WBC_GRAVITY = 9.81


def rotation(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def rotation_derivative(angle: float) -> np.ndarray:
    """平面旋转矩阵对角度的导数 ``dR/dα``。"""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[-sine, -cosine], [cosine, -sine]])


@dataclass
class PlanarContactRobot:
    """平面浮动基 + 两关节 + 单点接触的解析模型，给 WBC 当模拟数据。

    广义坐标 ``q = (x, z, θ, φ1, φ2)`` → ``nv = 5``；两个关节各一个执行器
    ``na = 2``；接触点是切向 + 法向两行 ``nc = 2``。结构上和本机同形：浮动基
    欠驱动 + 少量执行器 + 点接触，欠驱动维数同样是 1。

    链：机身（质心在机身原点）→ 髋（偏移 ``a1``，随 ``θ`` 转）→ 膝（``a2``，
    随 ``θ+φ1`` 转）→ 轮心（``a3``，随 ``θ+φ1+φ2`` 转）→ 接触点（再偏移 ``a4``，
    与轮心同帧）。

    质量矩阵按"点质量 + 转动惯量"的复合公式构造，必然对称正定；重力偏置取
    ``h = g·Σ m_i ∂z_i/∂q``。Coriolis 项取 0：测试用 ``q̇ = 0``，该项本就为零。
    运动学全部解析给出（链式求和 + 旋转矩阵导数），不用有限差分，这样"零空间
    性质必须完全不变"这类断言可以用很紧的容差。
    """

    base_mass: float = 8.0
    link_mass: float = 1.0
    wheel_mass: float = 1.0
    base_inertia: float = 0.05
    link_inertia: float = 0.005
    wheel_inertia: float = 0.005
    base_to_hip: tuple[float, float] = (0.0, -0.20)
    hip_to_knee: tuple[float, float] = (0.0, -0.15)
    knee_to_wheel: tuple[float, float] = (0.0, -0.05)
    wheel_to_contact: tuple[float, float] = (0.0, -0.06)
    gravity: float = WBC_GRAVITY

    def __post_init__(self) -> None:
        first = np.array(self.base_to_hip, dtype=float)
        second = np.array(self.hip_to_knee, dtype=float)
        third = np.array(self.knee_to_wheel, dtype=float)
        fourth = np.array(self.wheel_to_contact, dtype=float)
        # 每个点用到的段偏移：机身质心在原点，接触点与轮心同帧所以 a3、a4 可合并
        self._segment_offsets = (
            (),
            (first, second),
            (first, second, third),
            (first, second, third + fourth),
        )
        self._masses = (self.base_mass, self.link_mass, self.wheel_mass)
        self._inertias = (self.base_inertia, self.link_inertia, self.wheel_inertia)
        # 各点角速度在广义速度上的系数：机身 θ̇、腿 θ̇+φ̇1、轮 θ̇+φ̇1+φ̇2
        self._spin_rows = (
            np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
        )

    # ---- 运动学 --------------------------------------------------------
    @staticmethod
    def _angles(q: np.ndarray) -> tuple[float, float, float]:
        """三段连体的绝对角：θ、θ+φ1、θ+φ1+φ2。"""
        theta = float(q[2])
        return theta, theta + float(q[3]), theta + float(q[3]) + float(q[4])

    def point_position(self, q: np.ndarray, index: int) -> np.ndarray:
        """第 ``index`` 个点的位置（0=机身质心、1=膝、2=轮心、3=接触点）。"""
        q = np.asarray(q, dtype=float)
        angles = self._angles(q)
        position = np.array([q[0], q[1]], dtype=float)
        for segment, offset in enumerate(self._segment_offsets[index]):
            position = position + rotation(angles[segment]) @ offset
        return position

    def point_jacobian(self, q: np.ndarray, index: int) -> np.ndarray:
        """第 ``index`` 个点的位置雅可比，2×5。

        ``∂p/∂θ`` 收集所有段，``∂p/∂φ1`` 从第 2 段起，``∂p/∂φ2`` 从第 3 段起——
        因为绝对角是 ``θ, θ+φ1, θ+φ1+φ2``。
        """
        q = np.asarray(q, dtype=float)
        angles = self._angles(q)
        jacobian = np.zeros((2, 5))
        jacobian[0, 0] = 1.0
        jacobian[1, 1] = 1.0
        for segment, offset in enumerate(self._segment_offsets[index]):
            contribution = rotation_derivative(angles[segment]) @ offset
            jacobian[:, 2] += contribution
            if segment >= 1:
                jacobian[:, 3] += contribution
            if segment >= 2:
                jacobian[:, 4] += contribution
        return jacobian

    def contact_jacobian(self, q: np.ndarray) -> np.ndarray:
        return self.point_jacobian(q, 3)

    def contact_point(self, q: np.ndarray) -> np.ndarray:
        return self.point_position(q, 3)

    # ---- 动力学 --------------------------------------------------------
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """复合质量矩阵 ``M = Σ m_i J_viᵀ J_vi + Σ I_i J_ωiᵀ J_ωi``。"""
        matrix = np.zeros((5, 5))
        for index, (mass, inertia) in enumerate(zip(self._masses, self._inertias)):
            jacobian = self.point_jacobian(q, index)
            matrix += mass * (jacobian.T @ jacobian)
            spin = self._spin_rows[index]
            matrix += inertia * np.outer(spin, spin)
        return matrix

    def gravity_bias(self, q: np.ndarray) -> np.ndarray:
        """重力广义力 ``h = g·Σ m_i ∂z_i/∂q``（即 ``M q̈ + h = ...`` 里的 h）。"""
        bias = np.zeros(5)
        for index, mass in enumerate(self._masses):
            bias += self.gravity * mass * self.point_jacobian(q, index)[1]
        return bias

    def selection(self) -> np.ndarray:
        """执行器选择矩阵：两个关节各一个（2×5）。"""
        selection = np.zeros((2, 5))
        selection[0, 3] = 1.0
        selection[1, 4] = 1.0
        return selection


@pytest.fixture
def planar_robot() -> PlanarContactRobot:
    return PlanarContactRobot()
