"""正交分解消力与递归零空间（方案 §4.1 / §4.2）。

用 ``PlanarContactRobot`` 这个解析模型当模拟数据：nq=5、na=2、nc=2，浮动基
欠驱动维数 1，和本机同形但可以逐项手算核对。最关键的一条是
:func:`test_lower_priority_task_does_not_disturb_higher_priority`——它是零空间法
唯一的核心承诺（§7.4 验收清单里写着"完全不变"）。
"""

from __future__ import annotations

import numpy as np
import pytest

from Algorithm.wbc.projection import (
    ContactElimination,
    NullSpaceHierarchy,
    TaskLevel,
    dynamically_consistent_inverse,
)

Q = np.array([0.0, 0.5, 0.08, -0.30, 0.12])
JOINT_1 = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
JOINT_2 = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])


def build(planar_robot, q=Q, damping=0.0):
    mass = planar_robot.mass_matrix(q)
    contact = planar_robot.contact_jacobian(q)
    selection = planar_robot.selection()
    elimination = ContactElimination.from_matrices(contact, selection, mass, damping)
    hierarchy = NullSpaceHierarchy(mass, damping)
    return mass, contact, selection, elimination, hierarchy


# ---- 正交分解（§4.1） ----------------------------------------------------


def test_contact_frame_is_orthogonal_and_spans_the_contact_jacobian(planar_robot):
    mass, contact, _, elimination, _ = build(planar_robot)
    q_full = np.hstack([elimination.q_c, elimination.q_u])
    # Q 正交
    np.testing.assert_allclose(q_full.T @ q_full, np.eye(mass.shape[0]), atol=1e-12)
    # Q_u 的列与接触雅可比的列空间正交 → 左乘 Q_uᵀ 正好把接触力消掉
    np.testing.assert_allclose(elimination.q_u.T @ contact.T, 0.0, atol=1e-12)
    # Q_c 张成 Jcᵀ 的列空间
    np.testing.assert_allclose(
        elimination.q_c @ (elimination.q_c.T @ contact.T), contact.T, atol=1e-12
    )


def test_actuation_matrix_is_overdetermined_by_exactly_one(planar_robot):
    """``Q_uᵀSᵀ`` 是 3×2，秩 2——净欠驱动正好 1 维（§0 的权限账）。"""
    mass, _, _, elimination, _ = build(planar_robot)
    actuation = elimination.actuation_matrix
    assert actuation.shape == (mass.shape[0] - 2, 2)
    assert np.linalg.matrix_rank(actuation, tol=1e-9) == 2
    direction = elimination.unactuated_direction
    assert direction.shape == (actuation.shape[0],)
    np.testing.assert_allclose(actuation.T @ direction, 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)


def test_actuation_conditioning_is_reported(planar_robot):
    _, _, _, elimination, _ = build(planar_robot)
    expected = float(np.linalg.cond(elimination.actuation_matrix))
    assert elimination.actuation_conditioning == pytest.approx(expected)
    assert elimination.actuation_conditioning > 1.0


def test_torque_gain_and_offset_reproduce_recover_torque(planar_robot):
    """``τ = A q̈ + P h`` 必须与直接算 ``(Q_uᵀSᵀ)⁺Q_uᵀ(Mq̈+h)`` 逐元素一致。"""
    mass, _, _, elimination, _ = build(planar_robot)
    bias = planar_robot.gravity_bias(Q)
    rng = np.random.default_rng(7)
    for _ in range(5):
        qacc = rng.normal(size=mass.shape[0])
        expected = np.linalg.pinv(elimination.actuation_matrix, rcond=1e-12) @ (
            elimination.q_u.T @ (mass @ qacc + bias)
        )
        np.testing.assert_allclose(
            elimination.torque_gain @ qacc + elimination.torque_from_bias(bias),
            expected,
            atol=1e-12,
        )


def test_unactuated_direction_is_the_left_null_space(planar_robot):
    """执行器补不上的残差只能落在欠驱动那一维上。"""
    mass, _, selection, elimination, _ = build(planar_robot)
    bias = planar_robot.gravity_bias(Q)
    rng = np.random.default_rng(11)
    qacc = rng.normal(size=mass.shape[0])
    torque = elimination.recover_torque(qacc, bias)

    residual = elimination.actuation_residual(qacc, bias, torque)
    direction = elimination.unactuated_direction
    # 残差完全沿着欠驱动方向（其余分量必须为零）
    tangential = residual - direction * float(direction @ residual)
    np.testing.assert_allclose(tangential, 0.0, atol=1e-10)
    # 而且它确实非零：一般 q̈ 是执行器实现不了的
    assert np.linalg.norm(residual) > 1e-6
    del selection


def test_contact_force_recovery_closes_the_dynamics(planar_robot):
    """回代出的 ``f`` 必须把动力学残差压到只剩欠驱动那一维。

    这就是 §7.4 里"QR 消力正确 | 回代的 f vs data.cfrc_ext"在无 MuJoCo 下的等价
    说法：接触力吸收掉接触零空间之外的全部残差。
    """
    mass, contact, selection, elimination, _ = build(planar_robot)
    bias = planar_robot.gravity_bias(Q)
    rng = np.random.default_rng(3)

    for _ in range(5):
        torque = rng.normal(size=2) * 3.0
        force = rng.normal(size=2) * 30.0
        qacc = np.linalg.solve(mass, selection.T @ torque - bias + contact.T @ force)
        recovered = elimination.recover_contact_force(qacc, bias, torque)
        np.testing.assert_allclose(recovered, force, atol=1e-9)

        closure = mass @ qacc + bias - selection.T @ torque - contact.T @ recovered
        np.testing.assert_allclose(closure, 0.0, atol=1e-9)


def test_contact_force_recovery_leaves_only_the_unactuated_residual(planar_robot):
    """任意 ``q̈`` 下，回代 ``f`` 后的残差只能落在欠驱动方向（§0 的秩 1 缺口）。"""
    mass, contact, selection, elimination, _ = build(planar_robot)
    bias = planar_robot.gravity_bias(Q)
    rng = np.random.default_rng(23)

    qacc = rng.normal(size=mass.shape[0])
    torque = elimination.recover_torque(qacc, bias)
    force = elimination.recover_contact_force(qacc, bias, torque)
    residual = mass @ qacc + bias - selection.T @ torque - contact.T @ force

    direction = elimination.q_u @ elimination.unactuated_direction
    direction = direction / np.linalg.norm(direction)
    tangential = residual - direction * float(direction @ residual)
    np.testing.assert_allclose(tangential, 0.0, atol=1e-9)
    assert np.linalg.norm(residual) > 1e-6


# ---- 动力学一致伪逆（§4.2） ----------------------------------------------


def test_dynamically_consistent_inverse_is_exact_without_damping(planar_robot):
    mass = planar_robot.mass_matrix(Q)
    jacobian = np.vstack([JOINT_1, JOINT_2])
    inverse = dynamically_consistent_inverse(jacobian, mass)
    np.testing.assert_allclose(jacobian @ inverse, np.eye(2), atol=1e-12)

    null_space = np.eye(5) - inverse @ jacobian
    # 零空间投影算子：幂等、把 J̄ 的像消掉
    np.testing.assert_allclose(null_space @ null_space, null_space, atol=1e-12)
    np.testing.assert_allclose(null_space @ inverse, 0.0, atol=1e-12)
    # 而且是在质量矩阵度量下的自伴算子（这正是"动力学一致"的含义）
    np.testing.assert_allclose(null_space.T @ mass, mass @ null_space, atol=1e-12)


def test_dynamically_consistent_inverse_reduces_to_minimum_norm_for_identity_mass(planar_robot):
    """M = I 时应退化成普通最小范数伪逆。"""
    jacobian = np.vstack([JOINT_1, JOINT_2])
    inverse = dynamically_consistent_inverse(jacobian, np.eye(5))
    np.testing.assert_allclose(inverse, np.linalg.pinv(jacobian), atol=1e-12)


def test_damping_keeps_the_solution_finite_at_a_singular_task(planar_robot):
    """奇异任务（雅可比行相关）不加阻尼会炸，加了阻尼应当有界。"""
    mass = planar_robot.mass_matrix(Q)
    duplicated = np.vstack([JOINT_1, JOINT_1])  # 两行完全相同 → 秩 1
    damped = dynamically_consistent_inverse(duplicated, mass, damping=0.5)
    assert np.all(np.isfinite(damped))
    assert np.linalg.norm(damped) < 1e3


def test_damping_must_be_non_negative(planar_robot):
    mass = planar_robot.mass_matrix(Q)
    with pytest.raises(ValueError, match="阻尼"):
        dynamically_consistent_inverse(JOINT_1, mass, damping=-1.0)
    with pytest.raises(ValueError, match="阻尼"):
        NullSpaceHierarchy(mass, damping=-1.0)


# ---- 递归零空间（§4.2 / §7.4） -------------------------------------------


def test_hierarchy_respects_the_hard_contact_constraint(planar_robot):
    """接触一致性是硬优先级，任何任务都不能破坏它。"""
    mass, contact, _, _, hierarchy = build(planar_robot)
    bias = contact @ np.array([0.3, -0.1, 0.2, 0.05, -0.02])
    tasks = [
        TaskLevel(JOINT_1, np.array([1.0]), name="joint1"),
        TaskLevel(JOINT_2, np.array([-2.0]), name="joint2"),
    ]
    result = hierarchy.solve(contact, bias, tasks)
    np.testing.assert_allclose(result.contact_residual, 0.0, atol=1e-12)
    assert result.contact_residual_norm < 1e-12
    del mass


def test_lower_priority_task_does_not_disturb_higher_priority(planar_robot):
    """零空间法的唯一核心承诺：低优先级任务只能动高优先级剩下的自由度。

    §7.4 的验收标准是"完全不变"，所以这里用 1e-12 的容差。
    """
    mass, contact, _, _, hierarchy = build(planar_robot)
    # 注意区分：contact_bias 是 nc 维的 J̇c q̇，动力学偏置 h 是 nv 维的另一个量
    contact_bias = np.array([0.2, -0.1])
    top = TaskLevel(JOINT_1, np.array([1.5]), name="top")
    bottom = TaskLevel(JOINT_2, np.array([-3.0]), name="bottom")

    alone = hierarchy.solve(contact, contact_bias, [top])
    stacked = hierarchy.solve(contact, contact_bias, [top, bottom])

    np.testing.assert_allclose(
        stacked.outcome("top").achieved, alone.outcome("top").achieved, atol=1e-12
    )
    np.testing.assert_allclose(stacked.outcome("top").residual, alone.outcome("top").residual, atol=1e-12)
    # 低优先级确实动到了解，但只在剩下的零空间里
    assert not np.allclose(stacked.qacc, alone.qacc)
    # 底层的目标确实被满足了（还有自由度）
    np.testing.assert_allclose(stacked.outcome("bottom").achieved, [-3.0], atol=1e-9)
    del mass


def test_three_levels_keep_the_top_two_intact(planar_robot):
    """叠三层时，下面两层都不能扰动最上面那一级。"""
    mass, contact, _, _, hierarchy = build(planar_robot)
    contact_bias = np.array([0.2, -0.1])
    levels = [
        TaskLevel(JOINT_1, np.array([0.8]), name="L1"),
        TaskLevel(JOINT_2, np.array([-1.2]), name="L2"),
    ]
    third = TaskLevel(np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]), np.array([5.0]), name="L3")

    without = hierarchy.solve(contact, contact_bias, levels)
    with_third = hierarchy.solve(contact, contact_bias, levels + [third])
    for name in ("L1", "L2"):
        np.testing.assert_allclose(
            with_third.outcome(name).achieved, without.outcome(name).achieved, atol=1e-12
        )
    del mass


def test_injected_level_outranks_user_tasks(planar_robot):
    """SNS 注入的钳位层必须排在用户任务之前，否则钳位会被任务顶掉。"""
    mass, contact, _, elimination, hierarchy = build(planar_robot)
    contact_bias = np.array([0.2, -0.1])
    # 恢复力矩要的是动力学偏置 h（nv 维），和接触约束的 J̇c q̇ 不是一个东西
    dynamics_bias = planar_robot.gravity_bias(Q)
    task = TaskLevel(JOINT_1, np.array([100.0]), name="greedy")

    free = hierarchy.solve(contact, contact_bias, [task])
    free_torque = elimination.recover_torque(free.qacc, dynamics_bias)
    assert abs(free_torque[0]) > 1.0

    gain = elimination.torque_gain
    inject = TaskLevel(
        jacobian=gain[0].reshape(1, -1),
        desired_acceleration=np.array([0.0 - elimination.torque_from_bias(dynamics_bias)[0]]),
        name="clamp[0]",
    )
    clamped = hierarchy.solve(contact, contact_bias, [task], injected=[inject])
    torque = elimination.recover_torque(clamped.qacc, dynamics_bias)
    np.testing.assert_allclose(torque[0], 0.0, atol=1e-9)
    # 被钳之后扭矩确实变小了，而不是被任务顶回去
    assert abs(torque[0]) < abs(free_torque[0])
    del mass


def test_increments_sum_to_the_acceleration(planar_robot):
    """``increments`` 是零空间法的线性叠加项，加起来必须等于 ``q̈``（SNS 靠它做分层限值）。"""
    mass, contact, _, _, hierarchy = build(planar_robot)
    contact_bias = np.array([0.2, -0.1])
    tasks = [
        TaskLevel(JOINT_1, np.array([1.0]), name="a"),
        TaskLevel(JOINT_2, np.array([0.5]), name="b"),
    ]
    result = hierarchy.solve(contact, contact_bias, tasks)
    assert len(result.increments) == len(tasks) + 1  # 接触层 + 两个任务层
    np.testing.assert_allclose(np.sum(result.increments, axis=0), result.qacc, atol=1e-12)
    del mass


def test_task_outcome_reports_conditioning(planar_robot):
    """每级的条件数要上报（§5.3 里"监控条件数"那一项）。"""
    mass, contact, _, _, hierarchy = build(planar_robot)
    contact_bias = np.array([0.2, -0.1])

    scalar = hierarchy.solve(contact, contact_bias, [TaskLevel(JOINT_1, np.array([1.0]), name="a")])
    # 一维任务的 gram 是 1×1，条件数恒为 1
    assert scalar.outcome("a").conditioning == pytest.approx(1.0)
    assert scalar.outcome("a").residual_norm == pytest.approx(0.0, abs=1e-9)
    assert scalar.outcome("a").satisfied()

    vector = hierarchy.solve(
        contact, contact_bias, [TaskLevel(np.vstack([JOINT_1, JOINT_2]), np.array([1.0, 1.0]), name="b")]
    )
    assert vector.outcome("b").conditioning > 1.0
    assert result_contact_conditioning(contact, mass) == pytest.approx(vector.contact_conditioning)


def result_contact_conditioning(contact, mass) -> float:
    """接触层的条件数必须等于 ``cond(Jc M⁻¹ Jcᵀ)`` 的独立计算值。"""
    return float(np.linalg.cond(contact @ np.linalg.solve(mass, contact.T)))


def test_outcome_lookup_rejects_unknown_names(planar_robot):
    mass, contact, _, _, hierarchy = build(planar_robot)
    result = hierarchy.solve(contact, None, [TaskLevel(JOINT_1, np.array([0.0]), name="a")])
    with pytest.raises(KeyError):
        result.outcome("nope")
    del mass


# ---- 输入校验 ------------------------------------------------------------


def test_task_level_validates_shapes():
    with pytest.raises(ValueError, match="行数"):
        TaskLevel(JOINT_1, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="偏置项"):
        TaskLevel(JOINT_1, np.array([1.0]), bias_acceleration=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        TaskLevel(JOINT_1, np.array([]))


def test_elimination_validates_inputs(planar_robot):
    mass = planar_robot.mass_matrix(Q)
    contact = planar_robot.contact_jacobian(Q)
    selection = planar_robot.selection()
    with pytest.raises(ValueError, match="接触雅可比列数"):
        ContactElimination.from_matrices(np.zeros((2, 3)), selection, mass)
    with pytest.raises(ValueError, match="选择矩阵列数"):
        ContactElimination.from_matrices(contact, np.zeros((2, 3)), mass)
    with pytest.raises(ValueError, match="必须小于"):
        ContactElimination.from_matrices(np.eye(5), selection, mass)
    with pytest.raises(ValueError, match="方阵"):
        NullSpaceHierarchy(np.zeros((3, 4)))


def test_hierarchy_validates_contact_shapes(planar_robot):
    mass, contact, _, _, hierarchy = build(planar_robot)
    with pytest.raises(ValueError, match="接触雅可比列数"):
        hierarchy.solve(np.zeros((2, 3)), None, [TaskLevel(JOINT_1, np.array([0.0]))])
    with pytest.raises(ValueError, match="接触偏置项长度"):
        hierarchy.solve(contact, np.zeros(3), [TaskLevel(JOINT_1, np.array([0.0]))])
    del mass
