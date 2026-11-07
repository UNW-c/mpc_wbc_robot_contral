"""ReducedModel：状态序置换、力单位换算、偏航块、离散化。"""

from __future__ import annotations

import numpy as np
import pytest

from Algorithm.mpc import INPUT_DIM, STATE_DIM, ReducedModel, wheel_half_track, yaw_inertia_from_model
from Algorithm.mpc.model import _LQR_TO_REDUCED

from .conftest import (
    MEASURED_A_LONG,
    MEASURED_B_LONG,
    MEASURED_HALF_TRACK,
    MEASURED_YAW_INERTIA,
    WHEEL_RADIUS,
    make_measured_model,
)


def test_shape_validation():
    # B2（2026-09-27）泛化后：A 必须方阵、B 行数必须等于状态维数；
    # 不再硬编码 6x6 / 6x2（8 状态模型合法）。
    good_A = np.zeros((STATE_DIM, STATE_DIM))
    good_B = np.zeros((STATE_DIM, INPUT_DIM))
    with pytest.raises(ValueError, match="A 应为"):
        ReducedModel(np.zeros((4, 3)), good_B, 0.16, 0.12, 0.09)
    with pytest.raises(ValueError, match="B 的行数"):
        ReducedModel(good_A, np.zeros((4, INPUT_DIM)), 0.16, 0.12, 0.09)
    # 泛化示例：8 状态 / 3 输入形状可用
    ReducedModel(np.zeros((8, 8)), np.zeros((8, 3)), 0.16, 0.12, 0.09)


def test_from_longitudinal_rejects_bad_shapes():
    with pytest.raises(ValueError, match="A_longitudinal"):
        ReducedModel.from_longitudinal(
            np.zeros((3, 3)),
            np.zeros((4, 1)),
            wheel_radius=0.16,
            half_track=0.12,
            yaw_inertia=0.09,
        )
    with pytest.raises(ValueError, match="yaw_inertia"):
        ReducedModel.from_longitudinal(
            np.zeros((4, 4)),
            np.zeros((4, 1)),
            wheel_radius=0.16,
            half_track=0.12,
            yaw_inertia=0.0,
        )


def test_state_order_permutation(measured_model):
    """lqr 的 [x, pitch, x_dot, pitch_dot] 要重排成 [s, v, pitch, pitch_rate]。"""
    order = list(_LQR_TO_REDUCED)
    np.testing.assert_allclose(measured_model.A[:4, :4], MEASURED_A_LONG[np.ix_(order, order)])
    # 重排后的运动学行：s_dot = v、pitch_dot = pitch_rate
    assert measured_model.A[0, 1] == pytest.approx(1.0)
    assert measured_model.A[2, 3] == pytest.approx(1.0)
    # 重力耦合：俯仰角推动纵向加速度，也推动俯仰加速度
    assert measured_model.A[1, 2] == pytest.approx(MEASURED_A_LONG[2, 1])
    assert measured_model.A[3, 2] == pytest.approx(MEASURED_A_LONG[3, 1])


def test_force_unit_conversion(measured_model):
    """实测输入是差动单轮力矩 u，净地面力 F = 2u/R，所以 B_force = B_torque * R/2。"""
    order = list(_LQR_TO_REDUCED)
    expected = MEASURED_B_LONG.reshape(4)[order] * (WHEEL_RADIUS / 2.0)
    np.testing.assert_allclose(measured_model.B[:4, 0], expected)
    # 两轮对纵向/俯仰的贡献完全对称
    np.testing.assert_allclose(measured_model.B[:4, 0], measured_model.B[:4, 1])
    # 非最小相位：正的地面力让机身向后倒（俯仰加速度为负）
    assert measured_model.B[3, 0] < 0.0
    assert measured_model.B[1, 0] > 0.0


def test_yaw_block(measured_model):
    assert measured_model.A[4, 5] == pytest.approx(1.0)
    assert measured_model.A[5, 5] == pytest.approx(0.0)
    arm = MEASURED_HALF_TRACK / MEASURED_YAW_INERTIA
    assert measured_model.B[5, 0] == pytest.approx(+arm)
    assert measured_model.B[5, 1] == pytest.approx(-arm)
    # 纵向输入不该串到偏航，偏航状态也不该串回纵向
    np.testing.assert_allclose(measured_model.A[:4, 4:], 0.0)
    np.testing.assert_allclose(measured_model.B[:4, 0] - measured_model.B[:4, 1], 0.0)


def test_yaw_damping_enters_the_diagonal():
    model = make_measured_model(yaw_damping=0.4)
    assert model.A[5, 5] == pytest.approx(-0.4 / MEASURED_YAW_INERTIA)


def test_unstable_pole_matches_pitch_stiffness(measured_model):
    """开环不稳定极点应当就是俯仰刚度项的平方根。"""
    assert measured_model.unstable_pole == pytest.approx(
        float(np.sqrt(MEASURED_A_LONG[3, 1])), rel=1e-3
    )
    assert measured_model.unstable_pole > 0.0


def test_stable_model_reports_zero_pole():
    A = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    model = ReducedModel(A, np.zeros((STATE_DIM, INPUT_DIM)), 0.16, 0.12, 0.09)
    assert model.unstable_pole == 0.0


def test_discretize_first_order_for_small_dt(measured_model):
    dt = 1.0e-6
    discrete = measured_model.discretize(dt)
    np.testing.assert_allclose(discrete.A, np.eye(STATE_DIM) + measured_model.A * dt, atol=1e-9)
    np.testing.assert_allclose(discrete.B, measured_model.B * dt, atol=1e-9)
    assert discrete.dt == pytest.approx(dt)


def test_discretize_is_exact_under_step_composition(measured_model):
    """精确离散化的判据：一个 dt 步必须等于两个 dt/2 步的复合。"""
    dt = 0.02
    full = measured_model.discretize(dt)
    half = measured_model.discretize(dt / 2.0)
    np.testing.assert_allclose(full.A, half.A @ half.A, atol=1e-12)
    np.testing.assert_allclose(full.B, half.A @ half.B + half.B, atol=1e-12)


def test_discretize_drift(measured_model):
    dt = 0.02
    drift = np.array([0.0, 0.3, 0.0, -0.2, 0.0, 0.05])
    plain = measured_model.discretize(dt)
    drifted = measured_model.discretize(dt, drift)
    np.testing.assert_allclose(plain.w, 0.0, atol=1e-14)
    np.testing.assert_allclose(drifted.w, drift * dt, atol=1e-3)
    # 漂移项不改变 A/B
    np.testing.assert_allclose(drifted.A, plain.A, atol=1e-14)
    np.testing.assert_allclose(drifted.B, plain.B, atol=1e-14)
    half = measured_model.discretize(dt / 2.0, drift)
    np.testing.assert_allclose(drifted.w, half.A @ half.w + half.w, atol=1e-12)


def test_pitch_inertia_default_and_validation():
    """默认值取方案 §1.5 的 0.053191，20 N·m 对应 376 rad/s²。"""
    model = make_measured_model()
    assert model.pitch_inertia == pytest.approx(0.053191)
    assert 20.0 / model.pitch_inertia == pytest.approx(376.0, abs=0.1)
    with pytest.raises(ValueError, match="pitch_inertia"):
        ReducedModel(np.zeros((STATE_DIM, STATE_DIM)), np.zeros((STATE_DIM, INPUT_DIM)), 0.16, 0.12, 0.09, 0.0)


def test_hip_torque_drift_is_a_pure_pitch_acceleration(measured_model):
    """髋轴就是俯仰轴：两髋力矩之和只进俯仰角加速度那一行（方案 §1.5）。"""
    drift = measured_model.hip_torque_drift(20.0)
    expected = np.zeros(STATE_DIM)
    expected[3] = 20.0 / measured_model.pitch_inertia
    np.testing.assert_allclose(drift, expected)
    assert drift[3] == pytest.approx(376.0, abs=0.1)
    # 零力矩就是零漂移，且关于力矩线性
    np.testing.assert_allclose(measured_model.hip_torque_drift(0.0), 0.0)
    np.testing.assert_allclose(
        measured_model.hip_torque_drift(-6.0), -3.0 * measured_model.hip_torque_drift(2.0)
    )


def test_drift_step_map_matches_exact_discretization(measured_model):
    """连续漂移 -> 每步增量的映射必须与 discretize 的增广矩阵指数一致。"""
    dt = 0.02
    drift = np.array([0.0, 0.3, 0.0, -40.0, 0.0, 0.0])
    step_map = measured_model.drift_step_map(dt)
    assert step_map.shape == (STATE_DIM, STATE_DIM)
    np.testing.assert_allclose(step_map @ drift, measured_model.discretize(dt, drift).w, atol=1e-12)
    # 小步长下，被直接激励的两行一阶近似就是 drift * dt；其余行是 A 的耦合带出来的，
    # 量级小但非零（俯仰↔纵向在这台机器人上是强耦合，不能按 0 处理）
    step = step_map @ drift
    assert step[1] == pytest.approx(drift[1] * dt, rel=5e-2)
    assert step[3] == pytest.approx(drift[3] * dt, rel=5e-2)
    assert abs(step[2]) > 0.0  # 俯仰角由俯仰角速度积分而来，一步内不为零


def test_drift_step_map_is_linear_and_vanishes_without_drift(measured_model):
    step_map = measured_model.drift_step_map(0.02)
    np.testing.assert_allclose(step_map @ np.zeros(STATE_DIM), 0.0, atol=1e-15)
    drift = measured_model.hip_torque_drift(5.0)
    np.testing.assert_allclose(step_map @ (2.0 * drift), 2.0 * (step_map @ drift))


def test_torque_from_force_signs(measured_model):
    """j2 取 +R*f_right、j4 取 -R*f_left，与 WheelTorqueMixer 的共模=偏航一致。"""
    right, left = measured_model.torque_from_force(np.array([10.0, 10.0]))
    assert right == pytest.approx(+WHEEL_RADIUS * 10.0)
    assert left == pytest.approx(-WHEEL_RADIUS * 10.0)
    # 纯前进（两轮同号同量）在 ctrl 上表现为差模
    assert right == pytest.approx(-left)
    # 纯偏航（两轮反向）在 ctrl 上表现为共模
    right, left = measured_model.torque_from_force(np.array([10.0, -10.0]))
    assert right == pytest.approx(left)


def test_cart_pole_structure():
    m, M, length = 2.0, 10.0, 0.07
    model = ReducedModel.cart_pole(pendulum_mass=m, cart_mass=M, com_height=length)
    assert model.A[1, 2] == pytest.approx(m * 9.81 / M)
    assert model.A[3, 2] == pytest.approx((M + m) * 9.81 / (M * length))
    # B 的力单位已经换算过：scale = 2/R 再乘 R/2 正好抵消
    assert model.B[1, 0] == pytest.approx(1.0 / M)
    assert model.B[3, 0] == pytest.approx(-1.0 / (M * length))
    assert model.source == "cart_pole"


def test_metadata_and_source_are_recorded(measured_model):
    assert measured_model.source == "recorded-mujoco"
    assert isinstance(measured_model.metadata, dict)


def test_wheel_half_track_from_fake_model(fake_mujoco):
    model, data = fake_mujoco
    assert wheel_half_track(model, data) == pytest.approx(0.12)


def test_yaw_inertia_from_fake_model(fake_mujoco):
    """手算核对：复合 Izz = 0.2688，加反射自旋惯量 2*0.008*(0.12/0.16)^2 = 0.009。"""
    model, data = fake_mujoco
    inertia = yaw_inertia_from_model(model, data, wheel_radius=0.16, half_track=0.12)
    assert inertia == pytest.approx(0.2688 + 0.009)


def test_yaw_inertia_reflected_term_scales_with_track(fake_mujoco):
    model, data = fake_mujoco
    narrow = yaw_inertia_from_model(model, data, wheel_radius=0.16, half_track=0.06)
    wide = yaw_inertia_from_model(model, data, wheel_radius=0.16, half_track=0.24)
    assert wide > narrow
    assert wide - narrow == pytest.approx(2 * 0.008 * ((0.24 / 0.16) ** 2 - (0.06 / 0.16) ** 2))
