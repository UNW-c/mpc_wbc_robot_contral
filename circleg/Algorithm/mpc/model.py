"""MPC 用的降阶刚体模型。

状态只保留物理上可达、且能被独立控制的 6 个量::

    x = [s, v, pitch, pitch_rate, yaw, yaw_rate]

被删掉的状态及原因（见 docs/MPC-WBC方案.md §0 权限账本、§1.2 横向过约束）：

* ``p_y, v_y``  横向没有执行器，是非完整约束下由 ``yaw`` 决定的从动量，不是自由状态。
* ``p_z, v_z``  机身高度由偏心曲柄当位置源直接给定，不是靠法向力积分出来的，
  交给腿部规划 / WBC，不进 MPC。
* ``roll, roll_rate``  横滚靠左右差动法向力，权限是瞬时的、没有非最小相位问题；
  单支撑时权限直接为零，用 MPC 去预测它没有意义。交给 WBC 第二优先级。

决策量压缩成每步 2 个：两轮的纵向地面力 ``u = [f_right, f_left]`` (N)。这样

* 净驱动力   = ``f_right + f_left``
* 偏航力矩   = ``half_track * (f_right - f_left)``
* 电机力矩限 = ``|f_i| <= torque_limit / wheel_radius``   （箱型约束）
* 摩擦锥     = ``|f_i| <= mu * f_z,i``                    （同一行箱型约束）

于是摩擦锥和力矩限退化成逐轮上下界，QP 规模只有 ``2N`` 个变量。

纵向/俯仰子块直接复用 ``Algorithm.lqr`` 在 MuJoCo 接触位形上实测的 4x4 线性化，
因为它已经把滚动约束、接触雅可比、自由髋关节全都包含进去了，比手推 SRBD 可靠。
力单位换算：实测的输入是"差动模式单轮力矩 u"（j2=+u, j4=-u），对应净地面力
``F = 2u/R``，所以 ``B_force = B_torque * R / 2``。

已知建模误差（诚实记录，不要当成精确模型）：实测线性化时 j1/j3 力矩为 0，即
**髋关节是自由的**。真正上 WBC 后髋会被握住，等效质量会变大、不稳定极点会变，
届时应当带髋约束重新实测一次。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import expm

STATE_NAMES = ("s", "v", "pitch", "pitch_rate", "yaw", "yaw_rate")
INPUT_NAMES = ("f_right", "f_left")
STATE_DIM = len(STATE_NAMES)
INPUT_DIM = len(INPUT_NAMES)

PITCH_RATE_INDEX = 3

# 机身俯仰惯量（方案 §1.5）。该值取自 XML 里 base_link 惯量的第三项
# ``[0.053692, 0.053501, 0.053191]``：文档标为 iyy，实际是那一组里的 I_zz 分量，
# ``20 N·m / 0.053191 = 376 rad/s²`` 与文档自洽。含腿的复合俯仰惯量约 0.0605
# （同样 20 N·m 只有 330 rad/s²），需要更精确时由调用方覆盖这个参数。
DEFAULT_PITCH_INERTIA = 0.053191

# lqr.py 的状态序是 [x, pitch, x_dot, pitch_dot]，本模块是 [s, v, pitch, pitch_dot]
_LQR_TO_REDUCED = (0, 2, 1, 3)


@dataclass(frozen=True)
class DiscreteModel:
    """``x[k+1] = A x[k] + B u[k] + w``，其中 ``w`` 是常值漂移项的离散化。"""

    A: np.ndarray
    B: np.ndarray
    w: np.ndarray
    dt: float


@dataclass(frozen=True)
class ReducedModel:
    """连续时间降阶模型 ``x_dot = A x + B u + drift``。"""

    A: np.ndarray
    B: np.ndarray
    wheel_radius: float
    half_track: float
    yaw_inertia: float
    pitch_inertia: float = DEFAULT_PITCH_INERTIA
    source: str = "unspecified"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A 应为方阵，实际 {self.A.shape}")
        if self.B.ndim != 2 or self.B.shape[0] != self.A.shape[0]:
            raise ValueError(f"B 的行数应等于状态维数，实际 {self.B.shape}")
        if float(self.pitch_inertia) <= 0.0:
            raise ValueError("pitch_inertia 必须为正")

    @property
    def state_dim(self) -> int:
        """状态维数（由 A 的形状决定；6 状态模型为 6）。"""
        return int(self.A.shape[0])

    @property
    def input_dim(self) -> int:
        """输入维数（由 B 的形状决定；6 状态模型为 2）。"""
        return int(self.B.shape[1])

    @property
    def position_columns(self) -> tuple:
        """位置状态列号（偶数位：s / theta / psi / ...），由状态维数推导。"""
        return tuple(range(0, self.state_dim, 2))

    @property
    def velocity_columns(self) -> tuple:
        """速度状态列号（奇数位），由状态维数推导。"""
        return tuple(range(1, self.state_dim, 2))

    @property
    def unstable_pole(self) -> float:
        """俯仰通道的开环不稳定极点 (rad/s)，0 表示没有右半平面极点。"""
        eigenvalues = np.linalg.eigvals(self.A)
        real = float(np.max(eigenvalues.real))
        return max(real, 0.0)

    def discretize(self, dt: float, drift: np.ndarray | None = None) -> DiscreteModel:
        """用矩阵指数做精确离散化（把 ``u`` 和 ``drift`` 一起当常值输入增广）。"""
        dt = float(dt)
        n, m = self.state_dim, self.input_dim
        drift_vector = np.zeros(n) if drift is None else np.asarray(drift, dtype=float)
        augmented = np.zeros((n + m + 1, n + m + 1))
        augmented[:n, :n] = self.A
        augmented[:n, n : n + m] = self.B
        augmented[:n, -1] = drift_vector
        exponential = expm(augmented * dt)
        return DiscreteModel(
            A=np.ascontiguousarray(exponential[:n, :n]),
            B=np.ascontiguousarray(exponential[:n, n : n + m]),
            w=np.ascontiguousarray(exponential[:n, -1]),
            dt=dt,
        )

    def drift_step_map(self, dt: float) -> np.ndarray:
        """连续漂移 -> 离散每步增量的映射 ``M(dt) = ∫₀^dt e^{Aτ} dτ``。

        :meth:`hip_torque_drift` 给的是连续时间量（rad/s²），而凝聚预测里的漂移项
        是**每步增量**。两者差一个 ``M(dt)``：小步长下 ``≈ dt``，直接拿连续值当
        离散值用会差 50 倍。这个映射与 :meth:`discretize` 的增广矩阵指数是同一套
        数学，两者必须一致（有单元测试守着）。
        """
        n = self.state_dim
        augmented = np.zeros((2 * n, 2 * n))
        augmented[:n, :n] = self.A
        augmented[:n, n:] = np.eye(n)
        return np.ascontiguousarray(expm(augmented * float(dt))[:n, n:])

    def hip_torque_drift(self, hip_torque: float) -> np.ndarray:
        """两髋力矩之和折算成俯仰角加速度漂移项（方案 §1.5）。

        髋轴沿 y，就是俯仰轴，所以 ``j1_ctrl + j3_ctrl`` 直接叠加成机身俯仰力矩
        （满量程 ±20 N·m，是轮子俯仰权限的 1.1~1.8 倍）。MPC 必须把它当**已知输入**
        加进俯仰方程，否则升降腿的扰动会把规划出来的 ``f*`` 直接打乱。

        符号约定：正髋力矩产生正俯仰角加速度。这一条需要 M0 数值核验（方案 §7.3
        "施加纯髋力矩阶跃，测机身俯仰角加速度是否 ≈ τ_hip / I_yy"）。
        """
        drift = np.zeros(self.state_dim, dtype=float)
        drift[PITCH_RATE_INDEX] = float(hip_torque) / float(self.pitch_inertia)
        return drift

    def torque_from_force(self, force: np.ndarray) -> tuple[float, float]:
        """把 ``[f_right, f_left]`` 换成 ``MuJoCoRobot.apply_torques`` 要的两个 ctrl 值。

        符号与 ``Algorithm.controllers.WheelTorqueMixer`` 一致：j2 取 ``+R*f_right``、
        j4 取 ``-R*f_left``，于是共模 = 偏航、差模 = 前进。
        """
        force = np.asarray(force, dtype=float).reshape(-1)
        return (
            float(self.wheel_radius * force[0]),
            float(-self.wheel_radius * force[1]),
        )

    # ---- 构造入口 --------------------------------------------------------
    @classmethod
    def from_longitudinal(
        cls,
        A_longitudinal: np.ndarray,
        B_longitudinal: np.ndarray,
        *,
        wheel_radius: float,
        half_track: float,
        yaw_inertia: float,
        yaw_damping: float = 0.0,
        source: str = "longitudinal",
        metadata: dict[str, float] | None = None,
    ) -> "ReducedModel":
        """由纵向 4x4 实测线性化 + 偏航惯量拼出 6 状态模型。

        ``A_longitudinal`` / ``B_longitudinal`` 用 ``Algorithm.lqr`` 的状态序
        ``[x, pitch, x_dot, pitch_dot]``，``B`` 是对"差动模式单轮力矩"的导数。
        """
        A_long = np.asarray(A_longitudinal, dtype=float)
        B_long = np.asarray(B_longitudinal, dtype=float).reshape(4, -1)[:, 0]
        if A_long.shape != (4, 4):
            raise ValueError(f"A_longitudinal 应为 4x4，实际 {A_long.shape}")
        wheel_radius = float(wheel_radius)
        yaw_inertia = float(yaw_inertia)
        if yaw_inertia <= 0.0:
            raise ValueError("yaw_inertia 必须为正")

        order = list(_LQR_TO_REDUCED)
        A = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
        A[:4, :4] = A_long[np.ix_(order, order)]
        A[4, 5] = 1.0
        A[5, 5] = -float(yaw_damping) / yaw_inertia

        # 单轮力矩 u -> 净地面力 F = 2u/R，所以 dx/dt 对 F 的导数是 B_u * R/2
        force_column = B_long[order] * (wheel_radius / 2.0)
        B = np.zeros((STATE_DIM, INPUT_DIM), dtype=float)
        B[:4, 0] = force_column
        B[:4, 1] = force_column
        B[5, 0] = float(half_track) / yaw_inertia
        B[5, 1] = -float(half_track) / yaw_inertia

        return cls(
            A=A,
            B=B,
            wheel_radius=wheel_radius,
            half_track=float(half_track),
            yaw_inertia=yaw_inertia,
            source=source,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_mujoco(
        cls,
        model: Any,
        data: Any,
        *,
        wheel_radius: float | None = None,
        yaw_damping: float = 0.0,
    ) -> "ReducedModel":
        """在当前 MuJoCo 接触位形上实测模型（复用 ``Algorithm.lqr`` 的线性化）。

        mujoco 只在这里被导入，所以整个 ``Algorithm.mpc`` 包在没有 MuJoCo 的
        环境里依然可以导入并用模拟数据测试。
        """
        from Algorithm.lqr import measure_contact_linearization

        A_long, B_long = measure_contact_linearization(model, data)
        radius = float(
            wheel_radius
            if wheel_radius is not None
            else model.geom_size[int(model.geom("right_wheel_collision").id), 0]
        )
        half_track = wheel_half_track(model, data)
        inertia = yaw_inertia_from_model(model, data, radius, half_track)
        return cls.from_longitudinal(
            A_long,
            B_long,
            wheel_radius=radius,
            half_track=half_track,
            yaw_inertia=inertia,
            yaw_damping=yaw_damping,
            source="mujoco",
            metadata={"unstable_pole": float(np.sqrt(max(A_long[3, 1], 0.0)))},
        )

    @classmethod
    def cart_pole(
        cls,
        *,
        pendulum_mass: float,
        cart_mass: float,
        com_height: float,
        wheel_radius: float = 0.160,
        half_track: float = 0.120,
        yaw_inertia: float = 0.0964,
        gravity: float = 9.81,
        yaw_damping: float = 0.0,
    ) -> "ReducedModel":
        """教科书小车-倒立摆兜底模型，只在拿不到 MuJoCo 时当离线替身用。

        ``com_height`` 是摆质心到轮轴的距离，``cart_mass`` 是含反射轮惯量的等效
        平动质量。注意：本机的实测线性化因为髋关节自由，结构上**不是**小车-倒立
        摆，两者数值能差一个量级。权威模型是 :meth:`from_mujoco`，不要用这个标增益。
        """
        m = float(pendulum_mass)
        M = float(cart_mass)
        length = float(com_height)
        A_long = np.zeros((4, 4), dtype=float)
        A_long[0, 2] = 1.0
        A_long[1, 3] = 1.0
        A_long[2, 1] = m * gravity / M
        A_long[3, 1] = (M + m) * gravity / (M * length)
        scale = 2.0 / float(wheel_radius)  # 差动模式单轮力矩 u -> 净地面力 F = 2u/R
        B_long = np.zeros((4, 1), dtype=float)
        B_long[2, 0] = scale / M
        B_long[3, 0] = -scale / (M * length)
        return cls.from_longitudinal(
            A_long,
            B_long,
            wheel_radius=wheel_radius,
            half_track=half_track,
            yaw_inertia=yaw_inertia,
            yaw_damping=yaw_damping,
            source="cart_pole",
            metadata={"com_height": length, "cart_mass": M, "pendulum_mass": m},
        )


def wheel_half_track(model: Any, data: Any) -> float:
    """两个轮碰撞体在 y 方向的半间距（偏航力臂）。"""
    right = int(model.geom("right_wheel_collision").id)
    left = int(model.geom("left_wheel_collision").id)
    return float(abs(data.geom_xpos[left, 1] - data.geom_xpos[right, 1]) / 2.0)


def yaw_inertia_from_model(
    model: Any,
    data: Any,
    wheel_radius: float,
    half_track: float,
    wheel_bodies: tuple[str, ...] = ("joint2_Link", "joint4_Link"),
) -> float:
    """整机绕过质心竖直轴的等效偏航惯量。

    两部分：复合刚体惯量（各体主惯量旋到世界系取 zz，再加平行轴项），以及原地
    转向时轮子必须自旋带来的**反射惯量** ``2 I_spin (half_track/R)^2``。后者按
    "绕接触点连线中点纯滚动"算，这正好是 docs/MPC-WBC方案.md §1.2 里把横向
    无滑移集总到接触中点的那个假设；实际的 scrub 阻力不进惯量，而是通过收紧
    偏航力矩上界来体现。
    """
    body_ids = [index for index in range(1, int(model.nbody))]
    masses = np.asarray(model.body_mass[body_ids], dtype=float)
    positions = np.asarray(data.xipos[body_ids], dtype=float)
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("模型总质量为零，无法计算偏航惯量")
    center = np.sum(masses[:, None] * positions, axis=0) / total_mass

    inertia = 0.0
    for mass, body_id, position in zip(masses, body_ids, positions):
        rotation = np.asarray(data.ximat[body_id], dtype=float).reshape(3, 3)
        principal = np.diag(np.asarray(model.body_inertia[body_id], dtype=float))
        inertia += float((rotation @ principal @ rotation.T)[2, 2])
        offset = position - center
        inertia += float(mass) * float(offset[0] ** 2 + offset[1] ** 2)

    spin_ratio = float(half_track) / max(float(wheel_radius), 1.0e-9)
    for name in wheel_bodies:
        spin = float(model.body_inertia[int(model.body(name).id)][1])
        inertia += spin * spin_ratio * spin_ratio
    return float(inertia)
