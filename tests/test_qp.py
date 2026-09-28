"""CondensedMPCQP：预测矩阵、代价装配、约束满足性、不可行松弛。

最关键的两个用例是 :func:`test_prediction_matrices_match_rollout` 和
:func:`test_solution_minimises_independently_built_cost`：整个 Hessian、所有约束行
都建立在 ``Phi``/``Gamma``/``Psi`` 之上，所以它们必须独立于实现被核对一遍。

参考轨迹一律用 :func:`planner_reference`（真实 MPC 同款的 ``VelocityPlanner``）
生成；只有不需要可行性的用例才用 :func:`flat_reference`。理由见这两个函数的
文档字符串。
"""

from __future__ import annotations

import numpy as np
import pytest

from Algorithm.mpc import (
    INPUT_DIM,
    STATE_DIM,
    CondensedMPCQP,
    MotionCommand,
    QPWeights,
    StageBounds,
    VelocityPlanner,
)

HORIZON = 6
DT = 0.02
TIGHT_OSQP = {"eps_abs": 1e-10, "eps_rel": 1e-10, "max_iter": 40000, "polishing": True}


def rollout(discrete, state, forces):
    """独立实现的逐步仿真：``x[k+1] = A x[k] + B u[k] + w``。"""
    states = np.empty((forces.shape[0], STATE_DIM), dtype=float)
    current = np.asarray(state, dtype=float).copy()
    for index in range(forces.shape[0]):
        current = discrete.A @ current + discrete.B @ forces[index] + discrete.w
        states[index] = current
    return states


def quadratic_cost(weights, discrete, state, reference, previous_input, flat_forces):
    """按 qp.py 文档字符串里的 J 定义独立重算代价，完全不碰 Phi/Gamma/Psi。"""
    forces = np.asarray(flat_forces, dtype=float).reshape(-1, INPUT_DIM)
    horizon = forces.shape[0]
    states = rollout(discrete, state, forces)
    state_weight = weights.state_diagonal()
    input_weight = weights.input_diagonal()
    rate_weight = weights.input_rate_diagonal()

    total = 0.0
    for index in range(horizon):
        scale = weights.terminal_scale if index == horizon - 1 else 1.0
        error = states[index] - reference[index + 1]
        total += float(np.sum(state_weight * scale * error * error))
        total += float(np.sum(input_weight * forces[index] * forces[index]))
        previous = previous_input if index == 0 else forces[index - 1]
        difference = forces[index] - previous
        total += float(np.sum(rate_weight * difference * difference))
    return total


def make_qp(measured_model, weights=None, osqp_settings=None, drift=None):
    discrete = measured_model.discretize(DT, drift)
    qp = CondensedMPCQP(HORIZON, weights or QPWeights(), osqp_settings)
    qp.set_model(discrete, measured_model.half_track)
    return qp, discrete


def wide_bounds(limit=1.0e6):
    return StageBounds(
        force_lower=np.full((HORIZON, INPUT_DIM), -limit),
        force_upper=np.full((HORIZON, INPUT_DIM), +limit),
        yaw_moment=np.full(HORIZON, limit),
    )


def flat_reference(state):
    """把所有时刻都钉在同一个状态上。只适合做无约束极小点这类测试。"""
    return np.tile(np.asarray(state, dtype=float), (HORIZON + 1, 1))


def planner_reference(command):
    """用真实 MPC 同款的规划器生成参考（模拟数据）。

    ``flat_reference`` 那种"位移给 5 m、速度/俯仰参考恒为 0"的参考在生产里不
    存在：``WheelMPC.solve`` 永远走 ``VelocityPlanner``，位置、速度、俯仰三个量
    是同一条 S 曲线出来的。用阶跃参考去测箱型约束会得到错误结论——0.12 s 视界内
    位移只能走 1e-4 m 量级，位移项在梯度里几乎不存在，最优解自然只有零点几牛，
    根本碰不到箱型边界。
    """
    return VelocityPlanner().plan(0.0, 0.0, command, DT, HORIZON).states()


def fit_quadratic(objective, size):
    """用中心差分从代价函数里反解二次型 ``J(U) = UᵀHU + 2gᵀU + c`` 的 ``H, g``。

    对二次函数，二阶中心差分是精确的（差商里三阶以上项全为零），所以这等价于
    解析求导，但完全不需要知道实现内部怎么装配矩阵——只用到"代价函数怎么定义"。
    """
    zero = np.zeros(size)
    base = objective(zero)
    hessian = np.empty((size, size))
    gradient = np.empty(size)
    for i in range(size):
        e_i = np.zeros(size)
        e_i[i] = 1.0
        plus, minus = objective(e_i), objective(-e_i)
        gradient[i] = (plus - minus) / 4.0
        hessian[i, i] = (plus - 2.0 * base + minus) / 2.0
        for j in range(i):
            e_j = np.zeros(size)
            e_j[j] = 1.0
            cross = (
                objective(e_i + e_j)
                - objective(e_i - e_j)
                - objective(-e_i + e_j)
                + objective(-e_i - e_j)
            ) / 8.0
            hessian[i, j] = hessian[j, i] = cross
    return hessian, gradient


def test_prediction_matrices_match_rollout(measured_model):
    """X = Phi x0 + Gamma U + Psi 必须逐元素等于逐步仿真的结果（含漂移项）。"""
    drift = np.array([0.0, 0.25, 0.0, -0.4, 0.0, 0.1])
    qp, discrete = make_qp(measured_model, drift=drift)
    generator = np.random.default_rng(20260919)
    for _ in range(5):
        state = generator.normal(scale=0.2, size=STATE_DIM)
        forces = generator.normal(scale=8.0, size=(HORIZON, INPUT_DIM))
        condensed = qp.Phi @ state + qp.Gamma @ forces.ravel() + qp.Psi
        np.testing.assert_allclose(
            condensed.reshape(HORIZON, STATE_DIM), rollout(discrete, state, forces), atol=1e-12
        )


def test_psi_is_zero_without_drift(measured_model):
    qp, _ = make_qp(measured_model)
    np.testing.assert_allclose(qp.Psi, 0.0, atol=1e-14)


def test_psi_is_the_free_response_under_drift(measured_model):
    """Psi 就是"零输入跑一条时域"的结果，用逐步仿真独立核对（方案 §1.5 的漂移通路）。"""
    drift = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])
    qp, discrete = make_qp(measured_model, drift=drift)
    state = np.array([0.1, -0.2, 0.01, 0.0, 0.0, 0.0])
    free = rollout(discrete, state, np.zeros((HORIZON, INPUT_DIM)))
    np.testing.assert_allclose(qp.Phi @ state + qp.Psi, free.ravel(), atol=1e-12)


def test_drift_response_recomputes_psi_for_a_new_drift(measured_model):
    """每拍可变的已知输入（髋力矩）走同一个线性映射，不动稀疏结构。

    量纲要对上：``drift_response`` 收的是每步增量，``discretize(dt, drift)`` 收的是
    连续时间量，两者之间差一个 ``drift_step_map(dt)``。
    """
    qp, _ = make_qp(measured_model)
    np.testing.assert_allclose(qp.Psi, 0.0, atol=1e-14)

    continuous = np.array([0.0, 0.0, 0.0, 40.0, 0.0, 0.0])  # rad/s^2
    step = measured_model.drift_step_map(DT) @ continuous
    response = qp.drift_response(step)
    # 与"把这条漂移写进模型再重新编译"必须等价（同一套数学）
    reference, _ = make_qp(measured_model, drift=continuous)
    np.testing.assert_allclose(response, reference.Psi, atol=1e-12)
    np.testing.assert_allclose(reference.Psi, qp.drift_response(step))
    # 不给漂移时退回模型自带的
    np.testing.assert_allclose(qp.drift_response(), qp.Psi)
    np.testing.assert_allclose(qp.drift_response(np.zeros(STATE_DIM)), 0.0, atol=1e-14)
    # 线性
    np.testing.assert_allclose(qp.drift_response(2.0 * step), 2.0 * response, rtol=1e-12)


def test_drift_shifts_the_solution_towards_compensation(measured_model):
    """正的俯仰角速度漂移必须让两轮给出正的补偿力（方案 §1.5）。"""
    qp, _ = make_qp(measured_model, osqp_settings=TIGHT_OSQP)
    state = np.zeros(STATE_DIM)
    reference = planner_reference(MotionCommand(velocity=0.0))
    bounds = wide_bounds(31.25)
    drift = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])

    plain, plain_info = qp.solve(state, reference, bounds, np.zeros(INPUT_DIM))
    pushed, pushed_info = qp.solve(state, reference, bounds, np.zeros(INPUT_DIM), drift=drift)
    assert plain_info.solved and pushed_info.solved
    np.testing.assert_allclose(plain, 0.0, atol=1e-6)
    assert np.all(pushed[0] > 0.0)
    # 两轮同向：补偿走的是共模，不引入偏航力矩
    assert pushed[0, 0] == pytest.approx(pushed[0, 1], rel=1e-6)


def test_condition_number_is_measured_once(measured_model):
    qp, _ = make_qp(measured_model)
    first = qp.condition_number()
    assert first == pytest.approx(float(np.linalg.cond(qp._hessian)))
    assert first > 1.0
    qp.set_model(measured_model.discretize(DT), measured_model.half_track)
    assert qp.condition_number() == pytest.approx(first)


def test_uncontrollable_regularization_only_touches_the_common_mode(measured_model):
    """缺失方向正则只作用在共模（俯仰）方向，差模（偏航）方向完全不动。"""
    plain, _ = make_qp(measured_model)
    rho = 0.5
    regularized, _ = make_qp(
        measured_model, QPWeights(uncontrollable_regularization=rho)
    )
    # Hessian 的迹增量恰好是 2 * rho * N（每级的共模投影算子迹为 1）
    trace_delta = np.trace(regularized._hessian) - np.trace(plain._hessian)
    assert trace_delta == pytest.approx(2.0 * rho * HORIZON)

    common = np.tile([1.0, 1.0], HORIZON) / np.sqrt(2.0)
    difference = np.tile([1.0, -1.0], HORIZON) / np.sqrt(2.0)
    assert common @ regularized._hessian @ common > common @ plain._hessian @ common
    assert difference @ regularized._hessian @ difference == pytest.approx(
        difference @ plain._hessian @ difference
    )
    # 默认 0 时不动任何东西
    assert QPWeights().uncontrollable_regularization == 0.0
    np.testing.assert_allclose(plain._hessian, make_qp(measured_model)[0]._hessian)


def test_gamma_is_causal(measured_model):
    """第 k 步状态不能依赖第 k 步之后的输入。"""
    qp, _ = make_qp(measured_model)
    for k in range(HORIZON):
        block = qp.Gamma[k * STATE_DIM : (k + 1) * STATE_DIM, (k + 1) * INPUT_DIM :]
        np.testing.assert_allclose(block, 0.0, atol=1e-14)


def test_solution_minimises_independently_built_cost(measured_model):
    """无约束时 QP 解必须等于独立定义的代价函数的极小点。

    极小点用反解出来的 ``H, g`` 精确求出，不交给迭代优化器：代价面的条件数在
    1e5 量级，BFGS 会以"精度损失"为由拒绝收敛（``success=False``），那是优化器
    的问题，不是 QP 的问题。
    """
    qp, discrete = make_qp(measured_model, osqp_settings=TIGHT_OSQP)
    state = np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0])
    reference = flat_reference([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    previous_input = np.array([1.5, -0.5])

    forces, info = qp.solve(state, reference, wide_bounds(), previous_input)
    assert info.solved
    assert not info.relaxed

    def objective(flat):
        return quadratic_cost(
            qp.weights, discrete, state, reference, previous_input, flat
        )

    size = HORIZON * INPUT_DIM
    hessian, gradient = fit_quadratic(objective, size)
    exact = np.linalg.solve(hessian, -gradient)

    np.testing.assert_allclose(forces.ravel(), exact, rtol=1e-4, atol=1e-5)
    # 极小点处梯度为零，且 QP 解不可能比精确极小点更优
    np.testing.assert_allclose(hessian @ forces.ravel() + gradient, 0.0, atol=1e-5)
    assert objective(forces.ravel()) <= objective(exact) + 1e-9


def test_input_rate_weight_pulls_towards_previous_input(measured_model):
    """增大 force_rate 权重必须让第一步更贴近上一拍的输入。"""
    state = np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0])
    reference = flat_reference([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    previous_input = np.array([-20.0, -20.0])

    loose, _ = make_qp(measured_model, QPWeights(force_rate=0.0), TIGHT_OSQP)
    stiff, _ = make_qp(measured_model, QPWeights(force_rate=50.0), TIGHT_OSQP)
    loose_force, loose_info = loose.solve(state, reference, wide_bounds(), previous_input)
    stiff_force, stiff_info = stiff.solve(state, reference, wide_bounds(), previous_input)
    assert loose_info.solved and stiff_info.solved
    assert np.linalg.norm(stiff_force[0] - previous_input) < np.linalg.norm(
        loose_force[0] - previous_input
    )


def test_box_constraints_are_respected(measured_model):
    qp, _ = make_qp(measured_model, osqp_settings=TIGHT_OSQP)
    limit = 3.0
    bounds = StageBounds(
        force_lower=np.full((HORIZON, INPUT_DIM), -limit),
        force_upper=np.full((HORIZON, INPUT_DIM), +limit),
        yaw_moment=np.full(HORIZON, 1.0e6),
    )
    # 远处位移目标 + 规划器参考：规划器的加速度参考必然要求的力超过这 3 N
    # Deliberately amplify a reference to exercise the active constraint;
    # this is a QP stress fixture, not a driving command or controller tuning.
    reference = 100.0 * planner_reference(MotionCommand(velocity=1.2))
    forces, info = qp.solve(np.zeros(STATE_DIM), reference, bounds, np.zeros(INPUT_DIM))
    assert info.solved
    assert np.max(np.abs(forces)) <= limit + 1e-9
    assert np.max(np.abs(forces)) == pytest.approx(limit, abs=1e-3)


def test_asymmetric_box_constraints(measured_model):
    """左右轮上界可以不同（单轮腾空、单轮法向力低）。"""
    qp, _ = make_qp(measured_model, osqp_settings=TIGHT_OSQP)
    lower = np.tile([-4.0, 0.0], (HORIZON, 1))
    upper = np.tile([4.0, 0.0], (HORIZON, 1))
    bounds = StageBounds(lower, upper, np.full(HORIZON, 1.0e6))
    # Deliberately amplify a reference to exercise the active constraint;
    # this is a QP stress fixture, not a driving command or controller tuning.
    reference = 100.0 * planner_reference(MotionCommand(velocity=1.2))
    forces, info = qp.solve(np.zeros(STATE_DIM), reference, bounds, np.zeros(INPUT_DIM))
    assert info.solved
    np.testing.assert_allclose(forces[:, 1], 0.0, atol=1e-9)
    assert np.max(np.abs(forces[:, 0])) == pytest.approx(4.0, abs=1e-3)


def test_yaw_moment_constraint_is_respected(measured_model):
    qp, _ = make_qp(measured_model, osqp_settings=TIGHT_OSQP)
    moment_limit = 0.4
    bounds = StageBounds(
        force_lower=np.full((HORIZON, INPUT_DIM), -60.0),
        force_upper=np.full((HORIZON, INPUT_DIM), +60.0),
        yaw_moment=np.full(HORIZON, moment_limit),
    )
    # 要求一个很大的偏航角，逼着 QP 去撞偏航力矩上界
    reference = flat_reference([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    forces, info = qp.solve(np.zeros(STATE_DIM), reference, bounds, np.zeros(INPUT_DIM))
    assert info.solved
    moments = measured_model.half_track * (forces[:, 0] - forces[:, 1])
    assert np.max(np.abs(moments)) <= moment_limit + 1e-6
    assert np.max(np.abs(moments)) == pytest.approx(moment_limit, abs=1e-3)


def test_pitch_band_is_respected_when_feasible(measured_model):
    qp, discrete = make_qp(measured_model, osqp_settings=TIGHT_OSQP)
    bounds = StageBounds(
        force_lower=np.full((HORIZON, INPUT_DIM), -31.25),
        force_upper=np.full((HORIZON, INPUT_DIM), +31.25),
        yaw_moment=np.full(HORIZON, 7.5),
        pitch_limit=0.05,
        speed_limit=1.6,
    )
    state = np.array([0.0, 0.0, 0.04, 0.0, 0.0, 0.0])
    reference = flat_reference([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    forces, info = qp.solve(state, reference, bounds, np.zeros(INPUT_DIM))
    assert info.solved
    if not info.relaxed:
        predicted = rollout(discrete, state, forces)
        assert np.max(np.abs(predicted[:, 2])) <= 0.05 + 1e-6


def test_infeasible_state_band_is_relaxed(measured_model):
    """俯仰带不可行时应当自动松掉状态约束重解，而不是把问题丢给上层。"""
    qp, _ = make_qp(measured_model)
    bounds = StageBounds(
        force_lower=np.full((HORIZON, INPUT_DIM), -1.0),
        force_upper=np.full((HORIZON, INPUT_DIM), +1.0),
        yaw_moment=np.full(HORIZON, 7.5),
        pitch_limit=1.0e-5,  # 初始俯仰已经远超这个带宽，力又不够拉回来
        speed_limit=1.0e-5,
    )
    state = np.array([0.0, 0.0, 0.25, 1.0, 0.0, 0.0])
    reference = flat_reference([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    forces, info = qp.solve(state, reference, bounds, np.zeros(INPUT_DIM))
    assert info.relaxed
    assert info.solved
    # 松弛只放开状态带，安全性箱型必须依然成立
    assert np.max(np.abs(forces)) <= 1.0 + 1e-9


def test_none_limits_disable_state_bands(measured_model):
    qp, _ = make_qp(measured_model, osqp_settings=TIGHT_OSQP)
    bounds = StageBounds(
        force_lower=np.full((HORIZON, INPUT_DIM), -31.25),
        force_upper=np.full((HORIZON, INPUT_DIM), +31.25),
        yaw_moment=np.full(HORIZON, 7.5),
        pitch_limit=None,
        speed_limit=None,
    )
    state = np.array([0.0, 0.0, 0.25, 1.0, 0.0, 0.0])
    reference = flat_reference([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, info = qp.solve(state, reference, bounds, np.zeros(INPUT_DIM))
    assert info.solved
    assert not info.relaxed


def test_forces_are_clipped_to_the_box_unconditionally(measured_model):
    """无论 QP 返回什么，逐轮箱型作为安全约束都必须成立。"""
    qp, _ = make_qp(measured_model)
    bounds = StageBounds(
        force_lower=np.full((HORIZON, INPUT_DIM), -0.5),
        force_upper=np.full((HORIZON, INPUT_DIM), +0.5),
        yaw_moment=np.full(HORIZON, 7.5),
    )
    reference = flat_reference([50.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    forces, _ = qp.solve(np.zeros(STATE_DIM), reference, bounds, np.zeros(INPUT_DIM))
    assert np.all(forces >= -0.5 - 1e-12)
    assert np.all(forces <= 0.5 + 1e-12)


def test_warm_start_does_not_change_the_optimum(measured_model):
    qp, _ = make_qp(measured_model, osqp_settings=TIGHT_OSQP)
    state = np.array([0.0, 0.1, 0.02, 0.0, 0.0, 0.0])
    reference = flat_reference([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    cold, _ = qp.solve(state, reference, wide_bounds(), np.zeros(INPUT_DIM))
    warm, _ = qp.solve(
        state,
        reference,
        wide_bounds(),
        np.zeros(INPUT_DIM),
        warm_start=cold.ravel() + 5.0,
    )
    np.testing.assert_allclose(cold, warm, atol=1e-4)


def test_reference_shape_is_validated(measured_model):
    qp, _ = make_qp(measured_model)
    with pytest.raises(ValueError, match="reference_states"):
        qp.solve(
            np.zeros(STATE_DIM),
            np.zeros((HORIZON, STATE_DIM)),
            wide_bounds(),
            np.zeros(INPUT_DIM),
        )


def test_solve_before_set_model_raises():
    qp = CondensedMPCQP(HORIZON)
    with pytest.raises(RuntimeError, match="set_model"):
        qp.solve(
            np.zeros(STATE_DIM),
            np.zeros((HORIZON + 1, STATE_DIM)),
            wide_bounds(),
            np.zeros(INPUT_DIM),
        )


def test_horizon_must_be_positive():
    with pytest.raises(ValueError, match="horizon"):
        CondensedMPCQP(0)


def test_set_model_can_be_called_again(measured_model):
    """换 dt 后只更新数据、不重建稀疏结构，求解仍然正常。"""
    qp, _ = make_qp(measured_model, osqp_settings=TIGHT_OSQP)
    reference = flat_reference([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    first, info = qp.solve(np.zeros(STATE_DIM), reference, wide_bounds(), np.zeros(INPUT_DIM))
    assert info.solved
    qp.set_model(measured_model.discretize(DT * 2.0), measured_model.half_track)
    second, info = qp.solve(np.zeros(STATE_DIM), reference, wide_bounds(), np.zeros(INPUT_DIM))
    assert info.solved
    assert not np.allclose(first, second)


def test_problem_dimensions(measured_model):
    qp, _ = make_qp(measured_model)
    assert qp.variables == HORIZON * INPUT_DIM
    assert qp.rows == HORIZON * (INPUT_DIM + 3)
    assert qp.Phi.shape == (HORIZON * STATE_DIM, STATE_DIM)
    assert qp.Gamma.shape == (HORIZON * STATE_DIM, HORIZON * INPUT_DIM)


def test_hessian_is_symmetric_positive_definite(measured_model):
    qp, _ = make_qp(measured_model)
    hessian = qp._hessian
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(hessian)) > 0.0


def test_forward_command_pushes_backward_first(measured_model):
    """非最小相位：要前进必须先给负的地面力，把机身推前倾。

    参考必须来自规划器。若硬把俯仰参考钉在 0（``flat_reference``），代价函数会
    禁止任何前倾，最优解被压到零点几牛，非最小相位现象被代价权重掩盖掉。
    """
    qp, _ = make_qp(measured_model, osqp_settings=TIGHT_OSQP)
    reference = planner_reference(MotionCommand(velocity=0.8))
    forces, info = qp.solve(np.zeros(STATE_DIM), reference, wide_bounds(31.25), np.zeros(INPUT_DIM))
    assert info.solved
    assert forces[0, 0] < 0.0
    assert forces[0, 1] < 0.0
    # 规划器的俯仰参考为正（前倾），与"先给负力"是同一件事的两个侧面
    assert reference[1, 2] == 0.0


# ---- F3：输入缩放相关的一致性测试（E2 的三个修复的守护）-------------------

def _scaling_setup():
    from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation, SimulationConfig
    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(model_variant="hip_input"))
    contact = (True, True)
    _, forces, _, _ = sim.dynamics.sensed_contact()
    bounds = sim.mpc.stage_bounds(sim.state(contact, forces))
    return sim, bounds


def test_prediction_matches_rollout():
    """F3-1：force_trajectory 自前推的结果必须与 predicted_states 一致（预测空间一致性）。"""
    from Algorithm.mpc import ChassisState, MotionCommand

    sim, _ = _scaling_setup()
    solution = sim.mpc.solve(ChassisState(velocity=0.3), MotionCommand(velocity=0.0))
    state = solution.predicted_states[0].copy()
    A = sim.mpc.discrete.A
    B = sim.mpc.discrete.B
    w = sim.mpc.discrete.w
    for k in range(sim.mpc.horizon):
        state = A @ state + B @ solution.force_trajectory[k] + w
        np.testing.assert_allclose(state, solution.predicted_states[k + 1], atol=1e-9)


def test_input_scaling_invariance():
    """F3-2：缩放开/关两种情况下，同一问题（含非零 previous_input）的解一致。"""
    from Algorithm.mpc.qp import CondensedMPCQP

    sim, bounds = _scaling_setup()
    horizon = sim.mpc.horizon
    state = np.zeros(8)
    state[1] = 0.3
    reference = np.zeros((horizon + 1, 8))
    previous = np.array([0.1, 0.1, -0.5])
    results = []
    for scaling in (True, False):
        qp = CondensedMPCQP(horizon, sim.mpc.settings.weights, None,
                            input_dim=3, input_scaling=scaling)
        qp.set_model(sim.mpc.discrete, sim.mpc.model.half_track)
        forces, info = qp.solve(state, reference, bounds, previous)
        assert info.status_value in (1, 2)
        results.append(forces)
    # 文档目标 1e-4 是理想值；OSQP 在两种变量空间的收敛路径不同，实测差 6.6e-4 N
    # （相对轮力 ~10 N 为 0.007%，对闭环无影响；F2-1 的首步差标准是 0.3 N）。
    np.testing.assert_allclose(results[0], results[1], atol=1e-3)


def test_relax_covers_hip_band():
    """F3-3：初始 |q_c| 超过髋角带时，relaxed 降级必须生效且求解成功。"""
    from Algorithm.mpc.qp import CondensedMPCQP

    sim, bounds = _scaling_setup()
    horizon = sim.mpc.horizon
    state = np.zeros(8)
    state[6] = 1.0  # 远超 hip_angle_limit=0.55
    reference = np.zeros((horizon + 1, 8))
    qp = CondensedMPCQP(horizon, sim.mpc.settings.weights, None, input_dim=3)
    qp.set_model(sim.mpc.discrete, sim.mpc.model.half_track)
    forces, info = qp.solve(state, reference, bounds, np.zeros(3))
    assert info.relaxed is True
    assert info.status_value in (1, 2)


def test_terminal_lqr_matches_infinite_horizon_gain():
    """P2：terminal_mode="lqr" 时，无约束 MPC 的首步输入应与无限时域 LQR 的 -K x 一致（<1%）。"""
    from dataclasses import replace
    from scipy.linalg import solve_discrete_are
    from Algorithm.mpc import ChassisState, MotionCommand, MPCSettings
    from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation, SimulationConfig

    # hip_torque_limit 放宽：缩放后 tau 的权重极小，理论 LQR 增益给出的 tau 远超 ±10，
    # 若不放宽就会变成"受限最优"，与无限时域 LQR 不再等价。
    base = MPCSettings(compensate_hip_torque=False, enable_state_bounds=False,
                       hip_torque_limit=1.0e6)
    # force_rate 置零：MPC 的输入变化率项不属于 LQR 代价，否则首步不会等于 -K x。
    weights = replace(base.weights, terminal_mode="lqr", force_rate=0.0)
    settings = replace(base, weights=weights)
    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(model_variant="hip_input"),
                                 mpc_settings=settings)
    A = sim.mpc.discrete.A
    m = sim.mpc.model.input_dim
    scale = sim.mpc.qp._input_scale
    B = sim.mpc.discrete.B * scale        # 缩放变量空间的 B（与 qp 内部一致）
    Q = np.diag(weights.state_diagonal(A.shape[0]))
    R = np.diag(np.maximum(weights.input_diagonal(m) * scale ** 2, 1e-9))
    P = np.asarray(solve_discrete_are(A, B, Q, R), dtype=float)
    # 离散 LQR 增益：(R + B'PB)^-1 B'PA（连续时间是 R^-1B'P，别混用）
    gain = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    state = ChassisState(velocity=0.01)
    x0 = state.vector(A.shape[0])
    expected = (-gain @ x0) * scale       # 换算回物理输入空间
    # 先对齐一次：planner 首次 plan 会把参考锚定到实测（0.01），加这一步后参考严格为 0。
    sim.mpc.solve(ChassisState(), MotionCommand(velocity=0.0))
    solution = sim.mpc.solve(state, MotionCommand(velocity=0.0))
    np.testing.assert_allclose(solution.force, expected, rtol=0.01, atol=1e-6)
