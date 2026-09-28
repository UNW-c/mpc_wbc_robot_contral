"""WheelMPC：约束排程、兜底路径、参考对齐、warm start 记账。

全部用模拟数据（``tests/conftest.py`` 里记录的实测线性化），不需要 MuJoCo 运行时，
也不做闭环仿真——下层 WBC 还不存在，闭环跑不起来，这里只验证上层算法自身是否
自洽：约束排程符合 docs/MPC-WBC方案.md 的公式，输出的力确实落在排程出的界里，
轨迹参考与解算结果对得上。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from Algorithm.mpc import (
    INPUT_DIM,
    STATE_DIM,
    ChassisState,
    MPCSettings,
    MotionCommand,
    QPResult,
    WheelMPC,
)

from .conftest import WHEEL_RADIUS, make_measured_model

NOMINAL_NORMAL_FORCE = 46.6
TORQUE_BOUND = MPCSettings().wheel_torque_limit / WHEEL_RADIUS
OSQP_FAILURE = QPResult(
    status="maximum iterations reached",
    status_value=7,
    iterations=400,
    solve_time=0.01,
)


def make_mpc(**overrides) -> WheelMPC:
    """按需覆盖若干设置项，其余用 ``MPCSettings`` 的默认值。"""
    return WheelMPC(make_measured_model(), dataclasses.replace(MPCSettings(), **overrides))


def fail_solver(mpc: WheelMPC, monkeypatch) -> None:
    """把 QP 打成"永远求解失败"，用来单独测兜底通路。"""
    zeros = np.zeros((mpc.horizon, INPUT_DIM), dtype=float)
    monkeypatch.setattr(mpc.qp, "solve", lambda *args, **kwargs: (zeros, OSQP_FAILURE))


# ---- 构造 ----------------------------------------------------------------


def test_construction_builds_discrete_model_and_fallback_gain():
    mpc = make_mpc()
    assert mpc.horizon == MPCSettings().horizon
    assert mpc.discrete.dt == pytest.approx(MPCSettings().dt)
    assert mpc.fallback_gain.shape == (INPUT_DIM, STATE_DIM)
    assert np.all(np.isfinite(mpc.fallback_gain))
    assert mpc.consecutive_failures == 0
    assert not mpc.degraded


def test_horizon_setting_is_honoured():
    mpc = make_mpc(horizon=10, dt=0.05)
    assert mpc.horizon == 10
    assert mpc.discrete.dt == pytest.approx(0.05)
    assert mpc.qp.horizon == 10


# ---- 接触约束排程 --------------------------------------------------------


def test_flat_ground_force_limit_is_the_motor_torque_bound():
    """平地上绑住性能的是电机力矩，不是摩擦锥（方案 §3.2）。"""
    mpc = make_mpc()
    limit = mpc.wheel_force_limit(
        ChassisState(normal_forces=(NOMINAL_NORMAL_FORCE, NOMINAL_NORMAL_FORCE))
    )
    np.testing.assert_allclose(limit, TORQUE_BOUND)
    assert TORQUE_BOUND == pytest.approx(5.0 / 0.16)


def test_friction_cone_takes_over_when_the_normal_force_drops():
    """单轮法向力掉下来时，逐轮上界切换到摩擦锥。"""
    mpc = make_mpc()
    limit = mpc.wheel_force_limit(ChassisState(normal_forces=(10.0, 40.0)))
    assert limit[0] == pytest.approx(0.75 * 10.0)  # 摩擦锥绑定
    assert limit[1] == pytest.approx(min(TORQUE_BOUND, 0.75 * 40.0))  # 力矩限仍绑定


def test_airborne_wheel_gets_zero_force_authority():
    mpc = make_mpc()
    limit = mpc.wheel_force_limit(ChassisState(contact=(True, False)))
    assert limit[0] == pytest.approx(TORQUE_BOUND)
    assert limit[1] == 0.0


def test_low_normal_force_does_not_inflate_friction_authority():
    mpc = make_mpc()
    limit = mpc.wheel_force_limit(ChassisState(normal_forces=(1.0, 1.0)))
    np.testing.assert_allclose(limit, 0.75 * np.ones(2))


def test_torque_limit_setting_moves_the_force_bound():
    mpc = make_mpc(wheel_torque_limit=2.56)
    np.testing.assert_allclose(mpc.wheel_force_limit(ChassisState()), 2.56 / WHEEL_RADIUS)


def test_force_limit_is_symmetric_and_non_negative():
    mpc = make_mpc()
    state = ChassisState(normal_forces=(1.0, 40.0), contact=(True, True))
    bounds = mpc.stage_bounds(state)
    assert bounds.force_lower.shape == (mpc.horizon, INPUT_DIM)
    np.testing.assert_allclose(bounds.force_lower, -bounds.force_upper)
    assert np.all(bounds.force_upper >= 0.0)


# ---- 偏航权限（scrub 折扣，方案 §1.2） -----------------------------------


def test_yaw_moment_limit_is_the_nominal_budget_on_flat_ground():
    mpc = make_mpc()
    limit = mpc.yaw_moment_limit(ChassisState(hip_angle=0.0), np.full(INPUT_DIM, TORQUE_BOUND))
    assert limit == pytest.approx(MPCSettings().yaw_moment_limit)
    # 7.5 N·m 正好等于 2 * 31.25 N * 0.12 m
    assert limit == pytest.approx(2.0 * TORQUE_BOUND * 0.120)


def test_scrub_discount_scales_with_the_hip_angle():
    """中行程 scrub 拿走 74.5% 的偏航权限（方案 §1.2）。"""
    mpc = make_mpc()
    forces = np.full(INPUT_DIM, TORQUE_BOUND)
    flat = mpc.yaw_moment_limit(ChassisState(hip_angle=0.0), forces)
    mid = mpc.yaw_moment_limit(ChassisState(hip_differential=np.pi / 2.0), forces)
    assert mid == pytest.approx(flat - MPCSettings().scrub_moment_coefficient)
    assert mid == pytest.approx(1.91, abs=0.01)
    # 死点（q=0 或 q=π）scrub 消失，权限回到满值
    assert mpc.yaw_moment_limit(ChassisState(hip_angle=np.pi), forces) == pytest.approx(flat)


def test_yaw_moment_limit_never_exceeds_what_the_force_box_can_deliver():
    mpc = make_mpc()
    state = ChassisState(contact=(True, False))
    limit = mpc.yaw_moment_limit(state, mpc.wheel_force_limit(state))
    assert limit == pytest.approx(0.120 * TORQUE_BOUND)  # 单轮只剩 3.75 N·m
    assert limit < MPCSettings().yaw_moment_limit


def test_yaw_moment_limit_has_a_floor():
    mpc = make_mpc(scrub_moment_coefficient=1.0e3)
    limit = mpc.yaw_moment_limit(ChassisState(hip_differential=np.pi / 2.0), np.full(INPUT_DIM, TORQUE_BOUND))
    assert limit == pytest.approx(MPCSettings().yaw_moment_floor)


# ---- 状态带 --------------------------------------------------------------


def test_stage_bounds_carry_the_state_bands():
    mpc = make_mpc()
    bounds = mpc.stage_bounds(ChassisState())
    np.testing.assert_allclose(bounds.yaw_moment, MPCSettings().yaw_moment_limit)
    assert bounds.pitch_limit == pytest.approx(MPCSettings().pitch_limit)
    assert bounds.speed_limit == pytest.approx(MPCSettings().speed_limit)


def test_state_bands_can_be_switched_off():
    bounds = make_mpc(enable_state_bounds=False).stage_bounds(ChassisState())
    assert bounds.pitch_limit is None
    assert bounds.speed_limit is None
    # 安全性约束（逐轮箱型 + 偏航力矩）不受开关影响
    assert np.all(bounds.force_upper > 0.0)


# ---- 求解结果 ------------------------------------------------------------


def test_solution_derived_quantities_are_consistent():
    mpc = make_mpc()
    solution = mpc.solve(ChassisState(), MotionCommand(velocity=0.8))
    assert solution.qp.solved
    assert not solution.fallback
    right, left = solution.wheel_torque
    assert right == pytest.approx(WHEEL_RADIUS * solution.force[0])
    assert left == pytest.approx(-WHEEL_RADIUS * solution.force[1])
    assert solution.drive_force == pytest.approx(float(np.sum(solution.force)))
    assert solution.yaw_moment == pytest.approx(0.120 * (solution.force[0] - solution.force[1]))
    assert solution.force_trajectory.shape == (mpc.horizon, INPUT_DIM)
    assert solution.predicted_states.shape == (mpc.horizon + 1, STATE_DIM)


def test_solution_reference_accessors_align_with_the_horizon():
    """下层参考来自优化后的节点，规划器目标只用作 QP 跟踪代价。"""
    mpc = make_mpc()
    solution = mpc.solve(ChassisState(), MotionCommand(velocity=0.8))
    reference = solution.predicted_states[1]
    assert solution.pitch_reference == pytest.approx(reference[2])
    assert solution.pitch_rate_reference == pytest.approx(reference[3])
    assert solution.velocity_reference == pytest.approx(reference[1])
    assert solution.position_reference == pytest.approx(reference[0])
    assert abs(solution.pitch_reference - solution.reference.pitch[1]) > 1e-5
    np.testing.assert_allclose(solution.trajectory.sample(mpc.settings.dt).state, reference)


def test_first_tick_force_is_within_the_scheduled_bounds():
    mpc = make_mpc()
    state = ChassisState(velocity=0.6, pitch=0.04)
    solution = mpc.solve(state, MotionCommand(velocity=1.0))
    limit = mpc.wheel_force_limit(state)
    assert np.all(np.abs(solution.force) <= limit + 1e-9)


def test_saturated_solution_respects_the_contact_limit_exactly():
    """法向力压低到摩擦锥绑定时，解必须停在锥面上而不是越过。"""
    mpc = make_mpc()
    state = ChassisState(normal_forces=(5.0, NOMINAL_NORMAL_FORCE))
    solution = mpc.solve(state, MotionCommand(velocity=1.2))
    limit = mpc.wheel_force_limit(state)
    assert np.all(np.abs(solution.force_trajectory) <= limit + 1e-9)
    assert np.max(np.abs(solution.force_trajectory[:, 0])) == pytest.approx(limit[0], abs=1e-3)


def test_lifted_wheel_receives_no_force():
    mpc = make_mpc()
    solution = mpc.solve(ChassisState(contact=(True, False)), MotionCommand(velocity=0.5))
    np.testing.assert_allclose(solution.force_trajectory[:, 1], 0.0, atol=1e-9)


def test_predict_matches_a_manual_rollout():
    """``predict`` 的凝聚公式必须与被控对象逐拍推进的结果一致。"""
    mpc = make_mpc()
    state = ChassisState(position=0.1, pitch=0.02)
    solution = mpc.solve(state, MotionCommand(velocity=0.5))
    expected = [state.vector()]
    current = state.vector()
    for index in range(mpc.horizon):
        current = (
            mpc.discrete.A @ current
            + mpc.discrete.B @ solution.force_trajectory[index]
            + mpc.discrete.w
        )
        expected.append(current)
    np.testing.assert_allclose(solution.predicted_states, np.array(expected), atol=1e-12)


def test_warm_start_is_the_previous_trajectory_shifted_by_one(monkeypatch):
    mpc = make_mpc()
    seen: list[np.ndarray] = []
    original = mpc.qp.solve

    def spy(*args, **kwargs):
        seen.append(np.array(args[4], dtype=float))
        return original(*args, **kwargs)

    monkeypatch.setattr(mpc.qp, "solve", spy)
    first = mpc.solve(ChassisState(), MotionCommand(velocity=0.8))
    np.testing.assert_allclose(seen[0], 0.0)  # 第一拍没有历史
    mpc.solve(ChassisState(velocity=0.2), MotionCommand(velocity=0.8))
    expected = np.vstack((first.force_trajectory[1:], first.force_trajectory[-1:])).ravel()
    np.testing.assert_allclose(seen[1], expected)


# ---- 状态预推（方案 §3.4） -----------------------------------------------


def test_state_prediction_is_off_by_default():
    mpc = make_mpc()
    solution = mpc.solve(ChassisState(position=0.2, velocity=0.4), MotionCommand(velocity=0.5))
    np.testing.assert_allclose(solution.state, [0.2, 0.4, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(solution.predicted_states[0], solution.state)


def test_state_prediction_advances_the_state_by_the_lead():
    """lead=dt 且上一拍力为零时，预推结果就是被控对象走一步。"""
    mpc = make_mpc(prediction_lead=MPCSettings().dt)
    state = ChassisState(position=0.2, velocity=0.4, pitch=0.01)
    predicted = mpc.predict_state(state)
    expected = mpc.discrete.A @ state.vector() + mpc.discrete.w
    np.testing.assert_allclose(predicted, expected, atol=1e-12)
    assert predicted[0] > state.position  # 确实往前推了


def test_state_prediction_uses_the_previous_force():
    """预推必须用上一拍解出的力做零阶保持，否则等于假设电机突然卸力。"""
    mpc = make_mpc(prediction_lead=MPCSettings().dt)
    state = ChassisState(position=0.2, velocity=0.4)
    first = mpc.solve(state, MotionCommand(velocity=0.8))
    assert np.any(np.abs(first.force) > 0.0)

    predicted = mpc.predict_state(state)
    expected = (
        mpc.discrete.A @ state.vector()
        + mpc.discrete.B @ first.force
        + mpc.discrete.w
    )
    np.testing.assert_allclose(predicted, expected, atol=1e-12)
    # 少了 B @ u_prev 这一项就只是 free response，两者必须不同
    assert not np.allclose(predicted, mpc.discrete.A @ state.vector() + mpc.discrete.w)


def test_reference_lead_integrates_planned_acceleration():
    """参考的位移锚点要跟着预推走，否则 QP 会去追一个 v·lead 的假位移误差。"""
    mpc = make_mpc(prediction_lead=MPCSettings().dt)
    state = ChassisState(position=0.0, velocity=0.6)
    # 必须在求解前取：求解会把 _previous_force 换成新的，之后 predict_state 用的
    # 就是新力了，不再是这一拍实际喂给 QP 的那个状态
    predicted_before = mpc.predict_state(state)
    solution = mpc.solve(state, MotionCommand(velocity=1.2))

    assert predicted_before[0] > 0.0
    np.testing.assert_allclose(solution.state, predicted_before, atol=1e-15)
    assert solution.reference.position[0] == pytest.approx(
        state.position + state.velocity * mpc.settings.prediction_lead
        + 0.5 * mpc.settings.limits.commanded_acceleration * mpc.settings.prediction_lead**2)
    assert solution.position_reference > predicted_before[0]


@pytest.mark.parametrize("lead", [-0.001, 0.05])
def test_prediction_lead_is_validated(lead):
    with pytest.raises(ValueError, match="prediction_lead"):
        make_mpc(prediction_lead=lead)


# ---- 髋力矩作为已知输入（方案 §1.5） -------------------------------------


def test_hip_torque_is_converted_from_continuous_to_step_increment():
    """连续角加速度要先乘 M(dt) 才能当 QP 的漂移项，否则虚增 50 倍。"""
    mpc = make_mpc()
    state = ChassisState(hip_torque=20.0)
    continuous = mpc.model.hip_torque_drift(20.0)
    assert continuous[3] == pytest.approx(376.0, abs=0.1)  # rad/s^2
    discrete = mpc._hip_drift_map @ continuous
    np.testing.assert_allclose(mpc.hip_drift(state), discrete)
    # 一步内的俯仰角速度增量远小于连续角加速度本身
    assert discrete[3] == pytest.approx(continuous[3] * MPCSettings().dt, rel=0.1)


def test_hip_torque_compensation_can_be_switched_off():
    mpc = make_mpc(compensate_hip_torque=False)
    np.testing.assert_allclose(mpc.hip_drift(ChassisState(hip_torque=20.0)), 0.0)
    off = mpc.solve(ChassisState(hip_torque=20.0), MotionCommand(velocity=0.0))
    plain = make_mpc().solve(ChassisState(), MotionCommand(velocity=0.0))
    np.testing.assert_allclose(off.force, plain.force)


def test_positive_hip_torque_drives_positive_common_mode_force():
    """髋力矩是俯仰上的直接扰动，两轮要用同向力去顶（共模，不带偏航力矩）。"""
    state = ChassisState(hip_torque=1.0)
    solution = make_mpc().solve(state, MotionCommand(velocity=0.0))
    assert solution.qp.solved
    assert np.all(solution.force > 0.0)
    assert solution.force[0] == pytest.approx(solution.force[1], rel=1e-6)
    assert solution.yaw_moment == pytest.approx(0.0, abs=1e-9)


def test_hip_torque_compensation_grows_with_the_disturbance():
    forces = []
    for torque in (0.5, 1.0, 2.0):
        solution = make_mpc().solve(ChassisState(hip_torque=torque), MotionCommand(velocity=0.0))
        assert solution.qp.solved
        forces.append(float(solution.force[0]))
    assert forces[0] < forces[1] < forces[2]


def test_negative_hip_torque_reverses_the_compensation():
    state = ChassisState(hip_torque=-1.0)
    solution = make_mpc().solve(state, MotionCommand(velocity=0.0))
    assert np.all(solution.force < 0.0)


def test_hip_torque_enters_the_predicted_trajectory():
    """``predict`` 必须带上同一份漂移，否则画出来的轨迹漏掉髋力矩那一段。"""
    mpc = make_mpc()
    state = ChassisState(hip_torque=1.0)
    solution = mpc.solve(state, MotionCommand(velocity=0.0))
    drift = mpc.hip_drift(state)
    assert np.any(drift != 0.0)

    with_drift = mpc.predict(solution.state, solution.force_trajectory, drift)
    without_drift = mpc.predict(solution.state, solution.force_trajectory)
    assert not np.allclose(with_drift, without_drift)
    np.testing.assert_allclose(solution.predicted_states, with_drift, atol=1e-12)
    # 髋力矩只进俯仰那两行，偏航完全不受影响
    np.testing.assert_allclose(with_drift[:, 4:], without_drift[:, 4:], atol=1e-12)


def test_hip_torque_shows_up_as_a_pitch_rate_drift_under_zero_force():
    """零轮力时预测轨迹里应当直接看到髋力矩推出来的俯仰角速度。

    这也是把连续量换成每步增量的端到端核对：4 N·m -> 75.2 rad/s²，
    一步 20 ms 内积累约 1.5 rad/s，若漏乘 M(dt) 会差 50 倍。
    """
    mpc = make_mpc()
    state = ChassisState(hip_torque=4.0)
    prediction = mpc.predict(state.vector(), np.zeros((mpc.horizon, INPUT_DIM)), mpc.hip_drift(state))
    assert prediction[1, 3] == pytest.approx(4.0 / 0.053191 * 0.02, rel=0.1)
    assert prediction[1, 2] > 0.0
    # 不带漂移时这条轨迹是平的
    flat = mpc.predict(state.vector(), np.zeros((mpc.horizon, INPUT_DIM)))
    np.testing.assert_allclose(flat[:, 2:4], 0.0, atol=1e-12)


# ---- 接触几何与条件数监控（方案 §1.1） -----------------------------------


def test_solution_exposes_the_contact_geometry():
    mpc = make_mpc()
    solution = mpc.solve(ChassisState(hip_differential=np.pi / 2.0), MotionCommand(velocity=0.0))
    assert np.degrees(solution.contact.tilt) == pytest.approx(26.57, abs=0.01)
    assert solution.contact.pitch_controllability == pytest.approx(0.4472, rel=1e-3)
    # 与控制器上的显式接口一致
    assert mpc.contact_axis(ChassisState(hip_differential=np.pi / 2.0)).tilt == pytest.approx(
        solution.contact.tilt
    )


def test_condition_number_is_reported_but_does_not_warn_by_default(caplog):
    mpc = make_mpc()
    with caplog.at_level("WARNING", logger="Algorithm.mpc.controller"):
        solution = mpc.solve(ChassisState(), MotionCommand(velocity=0.8))
    # 实测 25 步视界约 8.66e5（position/velocity 权重 80000/2000，2026-09-25 定稿），
    # 低于默认阈值 1.0e6；旧值 9.86e4 是 20000/4000 时代的实测（期望过时同步，2026-09-27）
    assert solution.condition_number == pytest.approx(8.66e5, rel=0.05)
    assert solution.condition_number < MPCSettings().condition_number_limit
    assert not mpc.conditioning_warned
    assert "条件数" not in caplog.text


def test_conditioning_warning_fires_once_per_crossing(caplog):
    mpc = make_mpc(condition_number_limit=1.0)
    with caplog.at_level("WARNING", logger="Algorithm.mpc.controller"):
        mpc.solve(ChassisState(), MotionCommand(velocity=0.8))
        assert mpc.conditioning_warned
        first = caplog.text.count("条件数")
        mpc.solve(ChassisState(), MotionCommand(velocity=0.8))
    assert first == 1
    assert caplog.text.count("条件数") == first  # 只在跨越阈值那一次报警


def test_conditioning_monitor_can_be_disabled():
    mpc = make_mpc(condition_number_limit=0.0)
    solution = mpc.solve(ChassisState(), MotionCommand(velocity=0.8))
    assert solution.condition_number > 0.0
    assert not mpc.conditioning_warned


# ---- 终端代价的稳态流形（方案 §3.3） -------------------------------------


def test_terminal_reference_is_the_steady_state_manifold():
    """终端目标就是 (v_x*, atan(a*/g), 0)，定速巡航时俯仰参考与加速度都归零。

    视界要够长：0.5 s（N=25）时 2 m/s² 的斜坡还没跑完，终端仍带着 0.12 rad 的前倾；
    1 s（N=50）才落到稳态流形上。
    """
    mpc = make_mpc(horizon=50)
    solution = mpc.solve(ChassisState(velocity=0.8), MotionCommand(velocity=0.8))
    reference = solution.reference
    terminal = mpc.horizon
    assert reference.pitch[terminal] == pytest.approx(
        float(np.arctan2(reference.acceleration[terminal], MPCSettings().limits.gravity)),
        abs=1e-12,
    )
    assert reference.velocity[terminal] == pytest.approx(0.8, abs=1e-6)
    assert reference.acceleration[terminal] == pytest.approx(0.0, abs=1e-6)
    assert reference.pitch[terminal] == pytest.approx(0.0, abs=1e-6)
    assert reference.pitch_rate[terminal] == pytest.approx(0.0, abs=1e-6)


def test_ramp_finishes_with_equilibrium_pose_reference():
    mpc = make_mpc()
    reference = mpc.solve(ChassisState(), MotionCommand(velocity=0.8)).reference
    # 0.5 s 视界内以 commanded_acceleration=1.2 加速，只到 0.6 m/s（斜坡未走完）
    # （2026-09-28 C10 把该限值从 0.8 提到 1.2，期望同步）
    assert reference.velocity[-1] == pytest.approx(0.6)
    np.testing.assert_allclose(reference.pitch, 0.0)
    np.testing.assert_allclose(reference.pitch_rate, 0.0)


# ---- 兜底与降级 ----------------------------------------------------------


def test_first_failure_reuses_the_previous_solution_shifted_by_one(monkeypatch):
    """一级兜底：沿用上一拍解的下一格（方案 §3.4），不是立刻跳到 LQR。"""
    mpc = make_mpc()
    first = mpc.solve(ChassisState(), MotionCommand(velocity=0.8))
    assert first.qp.solved

    fail_solver(mpc, monkeypatch)
    state = ChassisState(velocity=0.2)
    solution = mpc.solve(state, MotionCommand(velocity=0.8))

    assert solution.fallback
    assert not solution.qp.solved
    assert mpc.consecutive_failures == 1
    assert not mpc.degraded
    expected = np.vstack((first.force_trajectory[1:], first.force_trajectory[-1:]))
    np.testing.assert_allclose(solution.force_trajectory, expected)
    # 沿用的上一拍解本来就在同一个接触界内
    limit = mpc.wheel_force_limit(state)
    assert np.all(np.abs(solution.force) <= limit + 1e-9)


def test_reused_trajectory_is_clipped_to_the_current_contact_limits(monkeypatch):
    """上一拍还在用力的轮子这一拍离地时，沿用下来的力必须被裁掉。

    逐轮箱型是安全约束，兜底通路也不能绕过（否则离地轮会被继续施力）。
    """
    mpc = make_mpc()
    first = mpc.solve(ChassisState(), MotionCommand(velocity=0.8))
    assert np.all(np.abs(first.force_trajectory[:, 1]) > 0.0)

    fail_solver(mpc, monkeypatch)
    state = ChassisState(velocity=0.2, contact=(True, False))
    solution = mpc.solve(state, MotionCommand(velocity=0.8))

    assert solution.fallback
    assert mpc.consecutive_failures == 1
    np.testing.assert_allclose(solution.force_trajectory[:, 1], 0.0, atol=1e-12)
    limit = mpc.wheel_force_limit(state)
    assert np.all(np.abs(solution.force_trajectory) <= limit + 1e-12)


def test_lqr_takes_over_after_the_failure_threshold(monkeypatch):
    """二级兜底：连续失败达阈值后退到 LQR 增益，时域内整条恒定。"""
    mpc = make_mpc()
    threshold = MPCSettings().max_consecutive_failures
    state = ChassisState(velocity=0.4)
    command = MotionCommand(velocity=0.0)
    fail_solver(mpc, monkeypatch)
    for _ in range(threshold - 1):
        mpc.solve(state, command)
    assert not mpc.degraded

    solution = mpc.solve(state, command)
    assert mpc.degraded
    assert np.any(np.abs(solution.force) > 0.0)
    np.testing.assert_allclose(
        solution.force_trajectory, np.tile(solution.force, (mpc.horizon, 1))
    )
    # 兜底解同样受接触约束，不是无界输出
    limit = mpc.wheel_force_limit(state)
    assert np.all(np.abs(solution.force) <= limit + 1e-12)


def test_fallback_gain_stabilises_the_reduced_model():
    """兜底增益必须是镇定增益：闭环极点全部落在左半平面。

    注意不要写成"力与速度误差反号"：这台机器人是非最小相位系统，速度超了 0.5 m/s
    时最优动作是把机身往后撂（俯仰角加速度 -220 rad/s^2）来减速，瞬时地面力反而是
    正向的 +60 N（dv/dt = +15.4 m/s^2）。"力与误差反号"只对最小相位系统成立。
    """
    mpc = make_mpc()
    closed_loop = mpc.model.A - mpc.model.B @ mpc.fallback_gain
    assert np.max(np.linalg.eigvals(closed_loop).real) < 0.0


def test_degraded_is_latched_after_the_failure_threshold(monkeypatch):
    settings = MPCSettings()
    mpc = make_mpc()
    fail_solver(mpc, monkeypatch)
    for _ in range(settings.max_consecutive_failures):
        mpc.solve(ChassisState(), MotionCommand(velocity=0.3))
    assert mpc.consecutive_failures == settings.max_consecutive_failures
    assert mpc.degraded


def test_recovery_clears_the_streak_but_keeps_the_degraded_latch(monkeypatch):
    mpc = make_mpc()
    fail_solver(mpc, monkeypatch)
    for _ in range(MPCSettings().max_consecutive_failures):
        mpc.solve(ChassisState(), MotionCommand(velocity=0.3))
    assert mpc.degraded

    monkeypatch.undo()
    solution = mpc.solve(ChassisState(), MotionCommand(velocity=0.3))
    assert solution.qp.solved
    assert mpc.consecutive_failures == 0
    # degraded 是给上层的锁存信号，只能由 reset() 复位
    assert mpc.degraded

    mpc.reset()
    assert not mpc.degraded


# ---- 复位 ----------------------------------------------------------------


def test_reset_clears_the_planner_and_the_failure_state(monkeypatch):
    mpc = make_mpc()
    fail_solver(mpc, monkeypatch)
    mpc.solve(ChassisState(), MotionCommand(velocity=0.5))
    assert mpc.consecutive_failures > 0

    mpc.reset()
    assert mpc.consecutive_failures == 0
    assert not mpc.degraded
    assert mpc.planner.speed == 0.0
    assert mpc.planner.yaw_rate == 0.0
    np.testing.assert_allclose(mpc._previous_force, 0.0)
    np.testing.assert_allclose(mpc._previous_trajectory, 0.0)


def test_reset_anchors_the_planner_on_the_measured_state():
    mpc = make_mpc()
    mpc.reset(ChassisState(velocity=0.3, yaw_rate=-0.2))
    assert mpc.planner.speed == pytest.approx(0.3, abs=1e-9)
    assert mpc.planner.yaw_rate == pytest.approx(-0.2, abs=1e-9)
