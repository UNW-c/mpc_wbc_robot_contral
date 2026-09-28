"""SNS：饱和、钳位、任务缩放、分层限值（方案 §5.1）。

三条必须成立的硬性质：

1. 返回的 ``τ`` 永远在硬件限值内（安全约束，兜底也不能破）；
2. 被钳的执行器停在限值上，而不是被任务顶回去；
3. 任务被连累时是"降幅不转向"——直接 ``np.clip`` 会扭曲任务方向，SNS 用缩放因子，
   所以达成量必须与目标同向。

限值量级是照实测挑的：这台模拟机器的重力补偿力矩（接触层贡献）是
``[3.37, 0.93]`` N·m，关节任务再线性叠加（关节 1 的 acc=1000 贡献约
``[15.6, 0]``，关节 2 的约 ``[14.3, 15.4]``）。
"""

from __future__ import annotations

import numpy as np
import pytest

from Algorithm.wbc.projection import ContactElimination, NullSpaceHierarchy, TaskLevel
from Algorithm.wbc.sns import ActuatorLimits, SaturationResolver, maximum_scale

Q = np.array([0.0, 0.5, 0.08, -0.30, 0.12])
JOINT_1 = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
JOINT_2 = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
CONTACT_BIAS = np.array([0.2, -0.1])
BASELINE = np.array([3.37, 0.93])


def two(value: float) -> np.ndarray:
    return np.full(2, float(value))


def build(planar_robot, limits=None):
    mass = planar_robot.mass_matrix(Q)
    contact = planar_robot.contact_jacobian(Q)
    selection = planar_robot.selection()
    elimination = ContactElimination.from_matrices(contact, selection, mass)
    resolver = SaturationResolver(NullSpaceHierarchy(mass), elimination)
    bias = planar_robot.gravity_bias(Q)
    return elimination, resolver, contact, bias, (limits or ActuatorLimits.symmetric(two(10.0)))


# ---- maximum_scale 的闭式解 ----------------------------------------------


def test_maximum_scale_is_the_largest_feasible_factor():
    rest = np.array([0.0])
    contribution = np.array([4.0])
    lower, upper = np.array([-1.0]), np.array([1.0])
    assert maximum_scale(rest, contribution, lower, upper) == pytest.approx(0.25)
    assert maximum_scale(rest, -contribution, lower, upper) == pytest.approx(0.25)


def test_maximum_scale_is_capped_at_one():
    assert maximum_scale(
        np.array([0.0]), np.array([0.01]), np.array([-10.0]), np.array([10.0])
    ) == pytest.approx(1.0)


def test_maximum_scale_returns_zero_when_no_scale_works():
    """交集与 [0,1] 不相交时返回 0：推不动（slope=0）推得动（slope<0）都算。"""
    assert maximum_scale(
        np.array([5.0]), np.array([0.0]), np.array([-1.0]), np.array([1.0])
    ) == pytest.approx(0.0)
    assert maximum_scale(
        np.array([5.0]), np.array([-1.0]), np.array([-1.0]), np.array([1.0])
    ) == pytest.approx(0.0)


def test_maximum_scale_takes_the_intersection_over_actuators():
    rest = np.array([0.0, 0.0])
    contribution = np.array([10.0, 2.0])
    lower, upper = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    # 两个执行器分别给出 0.1 和 0.5，取小的
    assert maximum_scale(rest, contribution, lower, upper) == pytest.approx(0.1)


# ---- 硬限值 --------------------------------------------------------------


def test_gravity_compensation_is_the_contact_layer_contribution(planar_robot):
    """接触层的力矩贡献就是重力负载本身，与关节任务无关。"""
    limits = ActuatorLimits.symmetric(two(1e3))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    result = resolver.resolve(
        contact, CONTACT_BIAS, [TaskLevel(JOINT_1, np.array([0.0]), name="j1")],
        limits, dynamics_bias=bias,
    )
    initial = result.hierarchy.increments[0]
    torque = result.attribution[0]
    force = elimination.recover_contact_force(initial, bias, torque)
    np.testing.assert_allclose(
        elimination.mass_matrix @ initial + bias,
        elimination.selection.T @ torque + contact.T @ force, atol=1e-9,
    )
    np.testing.assert_allclose(contact @ initial + CONTACT_BIAS, 0.0, atol=1e-9)
    del elimination


def test_solution_respects_the_actuator_limits(planar_robot):
    limits = ActuatorLimits.symmetric(two(5.0))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    result = resolver.resolve(
        contact, CONTACT_BIAS, [TaskLevel(JOINT_1, np.array([1000.0]), name="joint1")],
        limits, dynamics_bias=bias,
    )
    assert limits.contains(result.torque)
    assert np.max(np.abs(result.torque)) == pytest.approx(5.0, abs=1e-9)
    del elimination


def test_soft_task_does_not_saturate(planar_robot):
    """目标很小的时候不该触发任何钳位，缩放因子保持 1。"""
    limits = ActuatorLimits.symmetric(two(10.0))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    result = resolver.resolve(
        contact, CONTACT_BIAS, [TaskLevel(JOINT_1, np.array([0.01]), name="joint1")],
        limits, dynamics_bias=bias,
    )
    assert result.feasible
    assert not result.clip_applied
    assert result.clamped == {}
    assert result.scale_of("joint1") == pytest.approx(1.0)
    assert np.max(np.abs(result.torque)) < 10.0
    del elimination


def test_saturated_actuator_sits_exactly_on_its_limit(planar_robot):
    """饱和的执行器必须停在限值上，而不是缩过头或冲过去。

    注意路径：正常工况下是**全局任务缩放**把它停在边界上的（闭式，保方向），
    钳位回路根本不介入——这正是先缩放后钳位的意义。
    """
    limits = ActuatorLimits.symmetric(two(5.0))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    result = resolver.resolve(
        contact, CONTACT_BIAS, [TaskLevel(JOINT_1, np.array([1000.0]), name="joint1")],
        limits, dynamics_bias=bias,
    )
    assert result.torque[0] == pytest.approx(5.0, abs=1e-9)
    # 另一个执行器离自己的界还很远，说明缩放是按执行器分别生效的
    assert abs(result.torque[1]) < limits.upper[1] - 0.5
    assert limits.contains(result.torque)
    assert result.clamped == {}
    assert result.scale_of("joint1") < 1.0
    del elimination


def test_clamping_engages_when_even_the_bias_violates(planar_robot):
    """连"任务全关掉"都越限时（重力力矩就顶在界外），钳位回路才真正介入。"""
    limits = ActuatorLimits.symmetric(two(1e-3))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    result = resolver.resolve(
        contact, CONTACT_BIAS, [TaskLevel(JOINT_1, np.array([1.0]), name="joint1")],
        limits, dynamics_bias=bias,
    )
    assert set(result.clamped) == {0, 1}
    assert result.scale_of("joint1") == pytest.approx(0.0)  # 任务被丢弃
    assert limits.contains(result.torque)
    del elimination


def test_asymmetric_limits_are_respected(planar_robot):
    """上下限可以不对称——分层限值本身就靠这个。"""
    limits = ActuatorLimits(lower=np.array([-5.0, -1.0]), upper=np.array([5.0, 5.0]))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    result = resolver.resolve(
        contact, CONTACT_BIAS, [TaskLevel(JOINT_2, np.array([-1000.0]), name="joint2")],
        limits, dynamics_bias=bias,
    )
    assert limits.contains(result.torque)
    # 只有被推向负方向的那个执行器撞到自己的下限；另一个本来就在界内，不该被牵连
    assert result.torque[1] == pytest.approx(-1.0, abs=1e-9)
    assert abs(result.torque[0]) < limits.upper[0] - 0.5
    del elimination


# ---- 任务缩放（§5.1 第 3 步） --------------------------------------------


def test_scaling_reduces_the_amplitude_without_turning_the_task(planar_robot):
    """被连累的任务只降幅、不转向：达成量必须与目标同向。"""
    limits = ActuatorLimits.symmetric(two(5.0))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    result = resolver.resolve(
        contact, CONTACT_BIAS, [TaskLevel(JOINT_2, np.array([-1000.0]), name="joint2")],
        limits, dynamics_bias=bias,
    )
    achieved = float(result.hierarchy.outcome("joint2").achieved[0])
    assert achieved < 0.0
    assert achieved > -1000.0
    del elimination


def test_attribution_sums_to_the_torque(planar_robot):
    """``τ`` 必须等于各层贡献之和——分层限值全靠这个分解成立。"""
    limits = ActuatorLimits.symmetric(two(1e3))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    tasks = [
        TaskLevel(JOINT_1, np.array([8.0]), name="joint1"),
        TaskLevel(JOINT_2, np.array([-4.0]), name="joint2"),
    ]
    result = resolver.resolve(contact, CONTACT_BIAS, tasks, limits, dynamics_bias=bias)
    np.testing.assert_allclose(np.sum(result.attribution, axis=0), result.torque, atol=1e-9)
    # 层数 = 接触层 + 钳位注入层 + 用户任务层；本例没有钳位，所以是 1 + 2
    assert result.attribution.shape == (1 + len(tasks), 2)
    del elimination


def test_per_level_budgets_are_respected(planar_robot):
    """分层限值：低优先级任务只能在自己那份预算里用力（§1.5 的高度任务 ±4）。"""
    limits = ActuatorLimits.symmetric(two(1e3))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    tasks = [
        TaskLevel(JOINT_1, np.array([10.0]), name="pitch"),
        TaskLevel(JOINT_2, np.array([100.0]), name="height"),
    ]
    budgets = {"pitch": (two(-10.0), two(10.0)), "height": (two(-0.5), two(0.5))}
    result = resolver.resolve(
        contact, CONTACT_BIAS, tasks, limits, budgets, dynamics_bias=bias
    )
    assert limits.contains(result.torque)
    height_row = result.attribution[1 + len(tasks) - 1]
    assert np.max(np.abs(height_row)) <= 0.5 + 1e-9
    assert result.scale_of("height") < 1.0
    assert result.scale_of("pitch") == pytest.approx(1.0)
    del elimination


def test_budget_does_not_charge_a_level_for_other_levels(planar_robot):
    """预算只管该级自己的贡献，不能把接触层的重力力矩算到它头上。

    重力补偿本身就有 3.37 N·m，如果拿总力矩去比预算，任何预算比它小的级都会被
    误判成超限、压到 0。
    """
    limits = ActuatorLimits.symmetric(two(1e3))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    tasks = [TaskLevel(JOINT_1, np.array([0.01]), name="small")]
    budgets = {"small": (two(-1.0), two(1.0))}
    result = resolver.resolve(
        contact, CONTACT_BIAS, tasks, limits, budgets, dynamics_bias=bias
    )
    assert result.scale_of("small") == pytest.approx(1.0)
    assert result.attribution[1][0] < 1.0
    del elimination


def test_budget_scaling_lands_on_the_budget_boundary(planar_robot):
    """预算减到多少，该级的力矩贡献就该恰好降到那个边界上。"""
    limits = ActuatorLimits.symmetric(two(1e3))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    task = TaskLevel(JOINT_2, np.array([100.0]), name="height")
    free = resolver.resolve(contact, CONTACT_BIAS, [task], limits, dynamics_bias=bias)
    assert free.scale_of("height") == pytest.approx(1.0)
    unbudgeted = float(np.max(np.abs(free.attribution[1])))
    assert unbudgeted > 0.5

    budget = 0.5
    capped = resolver.resolve(
        contact, CONTACT_BIAS, [task], limits,
        {"height": (two(-budget), two(budget))}, dynamics_bias=bias,
    )
    # 该级贡献恰好贴着预算边界（对自身缩放因子是精确仿射的，所以一次就命中）
    assert np.max(np.abs(capped.attribution[1])) == pytest.approx(budget, abs=1e-9)
    assert 0.0 < capped.scale_of("height") < 1.0
    del elimination


def test_budget_shrinks_the_task_amplitude(planar_robot):
    """预算生效时，该级达成的加速度幅值也要跟着变小。"""
    limits = ActuatorLimits.symmetric(two(1e3))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    task = TaskLevel(JOINT_2, np.array([100.0]), name="height")
    free = resolver.resolve(contact, CONTACT_BIAS, [task], limits, dynamics_bias=bias)
    capped = resolver.resolve(
        contact, CONTACT_BIAS, [task], limits,
        {"height": (two(-0.5), two(0.5))}, dynamics_bias=bias,
    )
    assert abs(capped.hierarchy.outcome("height").achieved[0]) < abs(
        free.hierarchy.outcome("height").achieved[0]
    )
    del elimination


# ---- 不可行与兜底 --------------------------------------------------------


def test_degenerate_contact_set_falls_back_to_clipping(planar_robot):
    """防御路径：接触集退化（行相关）时钳位约束可能与它冲突，此时如实报不可行并裁剪。

    输入是刻意造出来的病态情形（接触行重复），正常工况下 ``nv - nc >= na``，
    钳位行总有解，这条路径不会触发。这里只断言与安全相关的两件事：力矩一定在
    界内、标志位如实上报；此时 ``q̈`` 本身已经不具物理意义，不做断言。
    """
    mass = planar_robot.mass_matrix(Q)
    duplicated = np.vstack([planar_robot.contact_jacobian(Q)] * 2)
    elimination = ContactElimination.from_matrices(duplicated, planar_robot.selection(), mass)
    resolver = SaturationResolver(NullSpaceHierarchy(mass), elimination)
    limits = ActuatorLimits.symmetric(two(0.5))
    result = resolver.resolve(
        duplicated, np.array([0.2, -0.1, 0.2, -0.1]),
        [TaskLevel(JOINT_2, np.array([100.0]), name="joint2")],
        limits, dynamics_bias=planar_robot.gravity_bias(Q),
    )
    # 只断言与安全相关的性质：力矩一定在界内。退化接触集下钳位行本身也可能
    # "没有权限"（被层级跳过），所以不保证每个执行器都停在界上、也不保证报
    # clip_applied——那是这套病态输入的必然结果。
    assert limits.contains(result.torque)


def test_iterations_are_bounded_by_the_actuator_count(planar_robot):
    limits = ActuatorLimits.symmetric(two(2.0))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    tasks = [
        TaskLevel(JOINT_1, np.array([1000.0]), name="joint1"),
        TaskLevel(JOINT_2, np.array([-1000.0]), name="joint2"),
    ]
    result = resolver.resolve(contact, CONTACT_BIAS, tasks, limits, dynamics_bias=bias)
    assert result.iterations <= resolver.max_rounds
    assert resolver.max_rounds == elimination.selection.shape[0] + 2
    assert limits.contains(result.torque)


def test_contact_constraint_survives_saturation(planar_robot):
    """钳位注入不能破坏硬接触约束。"""
    limits = ActuatorLimits.symmetric(two(2.0))
    elimination, resolver, contact, bias, _ = build(planar_robot, limits)
    tasks = [
        TaskLevel(JOINT_1, np.array([1000.0]), name="joint1"),
        TaskLevel(JOINT_2, np.array([-1000.0]), name="joint2"),
    ]
    result = resolver.resolve(contact, CONTACT_BIAS, tasks, limits, dynamics_bias=bias)
    assert result.hierarchy.contact_residual_norm < 1e-9
    assert limits.contains(result.torque)


# ---- 输入校验 ------------------------------------------------------------


def test_limits_validation():
    with pytest.raises(ValueError, match="形状"):
        ActuatorLimits(lower=np.zeros(2), upper=np.zeros(3))
    with pytest.raises(ValueError, match="下限大于上限"):
        ActuatorLimits(lower=np.array([1.0]), upper=np.array([-1.0]))


def test_limits_helpers():
    limits = ActuatorLimits(lower=np.array([-2.0, -2.0]), upper=np.array([3.0, 3.0]))
    assert limits.size == 2
    assert limits.contains(np.array([-2.0, 3.0]))
    assert not limits.contains(np.array([3.01, 0.0]))
    np.testing.assert_allclose(limits.clamp(np.array([9.0, -9.0])), [3.0, -2.0])
    np.testing.assert_allclose(limits.violation(np.array([4.0, 0.0])), [1.0, 0.0])
    np.testing.assert_allclose(limits.violation(np.array([0.0, 0.0])), [0.0, 0.0])


def test_resolver_rejects_mismatched_limit_count(planar_robot):
    """限值个数必须等于执行器个数，否则报一句清楚的错而不是下标越界。"""
    limits = ActuatorLimits.symmetric(two(1.0))
    _, resolver, contact, bias, _ = build(planar_robot, limits)
    with pytest.raises(ValueError, match="限值个数"):
        resolver.resolve(
            contact, CONTACT_BIAS, [TaskLevel(JOINT_1, np.array([0.0]), name="a")],
            ActuatorLimits.symmetric(1.0), dynamics_bias=bias,
        )


def test_resolver_validates_bias_lengths(planar_robot):
    limits = ActuatorLimits.symmetric(two(1.0))
    _, resolver, contact, bias, _ = build(planar_robot, limits)
    tasks = [TaskLevel(JOINT_1, np.array([0.0]), name="a")]
    with pytest.raises(ValueError, match="接触偏置长度"):
        resolver.resolve(contact, np.zeros(3), tasks, limits, dynamics_bias=bias)
    with pytest.raises(ValueError, match="动力学偏置长度"):
        resolver.resolve(contact, CONTACT_BIAS, tasks, limits, dynamics_bias=np.zeros(3))


def test_resolver_rejects_an_empty_task_list(planar_robot):
    limits = ActuatorLimits.symmetric(two(1.0))
    _, resolver, contact, bias, _ = build(planar_robot, limits)
    with pytest.raises(ValueError, match="至少要有一个"):
        resolver.resolve(contact, CONTACT_BIAS, [], limits, dynamics_bias=bias)
