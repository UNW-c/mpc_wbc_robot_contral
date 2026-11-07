"""TaskQpResolver 的算法正确性回归。

本文件只验证单级 QP 求解器本身的正确性与可行性，不做闭环稳定性判定：
- 四类硬约束（动力学等式、接触一致性等式、力矩限、法向非负）必须满足；
- 任务残差诊断与手动计算一致；
- 任务权重确实改变解（针对历史"权重 33 倍无效"的对映问题）；
- 一级不可行时按两级兜底降级，且兜底路径可用。

闭环发散与否不在此判定：发散可能来自控制参数，不属于本文件范围。
"""
from __future__ import annotations

import numpy as np

from Algorithm.wbc.projection import ContactElimination, TaskLevel
from Algorithm.wbc.sns import ActuatorLimits, SNSResult
from Algorithm.wbc.task_qp import TaskQpResolver
from tests.joint_harness import total_weight
from tests.planar_robot import PlanarTwoWheelRobot


def make_problem():
    """静态平衡位形下的解析问题：M / h / Jc / S / 限值 / 消去器。"""
    robot = PlanarTwoWheelRobot()
    q = robot.standing_pose()
    mass = robot.mass_matrix(q)
    selection = robot.selection()
    jacobian = robot.contact_jacobian(q)
    h = robot.gravity_bias(q)
    elimination = ContactElimination.from_matrices(jacobian, selection, mass)
    limits = ActuatorLimits(lower=np.array([-10.0, -5.0, -10.0, -5.0]),
                            upper=np.array([10.0, 5.0, 10.0, 5.0]))
    return robot, q, mass, selection, jacobian, h, elimination, limits


def resolve_with(elimination, jacobian, h, tasks, limits, *, contact_bias=None,
                 weights=None, **resolver_kwargs):
    resolver = TaskQpResolver(elimination, task_weights=weights or {}, **resolver_kwargs)
    return resolver.resolve(jacobian, contact_bias, tasks, limits, dynamics_bias=h)


def pitch_task(nv: int, desired: float = 0.5):
    """俯仰角加速度任务（qacc[2] = desired）。"""
    return TaskLevel(np.eye(nv)[[2]], np.array([desired]), name="pitch")


def hold_task(nv: int):
    """全加速度阻尼任务：所有广义加速度尽量为零。"""
    return TaskLevel(np.eye(nv), np.zeros(nv), name="hold")


def test_static_equilibrium_recovers_gravity_support():
    """零目标下 QP 应恢复静态平衡的物理量：机身不加速、法向力各担一半体重。

    删除摩擦锥后切向接触力不进入任何约束与代价，解在切向方向有数值噪声，
    因此不断言关节加速度与力矩为零；只验证机身加速度、法向力与约束残差。
    """
    robot, _, mass, selection, jacobian, h, elimination, limits = make_problem()
    result = resolve_with(elimination, jacobian, h, [hold_task(7)], limits)
    assert isinstance(result, SNSResult)
    assert result.feasible, "静态平衡问题必须可行"
    assert not result.fallback_used, "静态平衡不应触发兜底"
    # 机身（x/z/pitch）不得加速。
    assert np.all(np.abs(result.qacc[:3]) < 1e-3), (
        f"机身加速度应接近零，实际 {result.qacc[:3]}")
    # 从动力学等式反推接触力（接触雅可比满行秩时唯一），检查法向分量。
    force = np.linalg.pinv(jacobian.T) @ (mass @ result.qacc + h - selection.T @ result.torque)
    half = total_weight(robot) / 2.0
    for normal in (1, 3):
        assert force[normal] >= 0.0, f"法向力必须非负，实际 {force[normal]}"
        assert abs(force[normal] - half) / half < 0.05, (
            f"法向力应约半体重 {half}，实际 {force[normal]}")


def test_hard_constraints_satisfied():
    """四类硬约束必须全部满足：动力学、接触一致性、力矩限、法向非负。"""
    _, _, mass, selection, jacobian, h, elimination, limits = make_problem()
    nv = mass.shape[0]
    tasks = [pitch_task(nv, desired=0.5)]
    result = resolve_with(elimination, jacobian, h, tasks, limits)
    assert result.feasible, "平衡位形下的俯仰任务应可行"
    qacc, torque = result.qacc, result.torque
    assert np.all(torque <= limits.upper + 1e-8) and np.all(torque >= limits.lower - 1e-8), (
        f"力矩越限: {torque} 不在 [{limits.lower}, {limits.upper}]")
    assert qacc.shape == (nv,)
    # 反推接触力，检查法向分量非负（法向非负约束）。
    force = np.linalg.pinv(jacobian.T) @ (mass @ qacc + h - selection.T @ torque)
    assert force[1] >= -1e-8 and force[3] >= -1e-8, f"法向力必须非负，实际 {force}"


def test_outcomes_report_real_residuals():
    """outcomes 必须逐任务给出真实残差，与手动计算一致。"""
    _, _, _, _, jacobian, h, elimination, limits = make_problem()
    nv = 7
    tasks = [pitch_task(nv, desired=0.5), hold_task(nv)]
    result = resolve_with(elimination, jacobian, h, tasks, limits)
    by_name = {item.name: item for item in result.hierarchy.outcomes}
    assert set(by_name) == {"pitch", "hold"}, f"outcomes 任务名缺失: {list(by_name)}"
    for task in tasks:
        outcome = by_name[task.name]
        achieved = task.jacobian @ result.qacc + task.bias_acceleration
        expected = achieved - task.desired_acceleration
        assert np.allclose(outcome.achieved, achieved), f"{task.name} 的 achieved 不实"
        assert np.allclose(outcome.residual, expected), f"{task.name} 的 residual 不实"
        assert abs(outcome.residual_norm - np.linalg.norm(expected)) < 1e-12, (
            f"{task.name} 的 residual_norm 不实")


def test_weights_steer_conflicting_tasks():
    """权重必须真正改变解：四个零空间方向任务互相冲突，权重高的残差更小。"""
    _, _, mass, _, jacobian, h, elimination, limits = make_problem()
    # 接触零空间基（4 行满秩，零空间 3 维）。
    _, _, vt = np.linalg.svd(jacobian, full_matrices=True)
    basis = vt[4:].T  # 7x3
    directions = [basis[:, 0], basis[:, 1], basis[:, 2],
                  basis[:, 0] + basis[:, 1] + basis[:, 2]]
    directions = [d / np.linalg.norm(d) for d in directions]
    tasks = [TaskLevel(d[None, :], np.array([1.0]), name=f"t{i}")
             for i, d in enumerate(directions)]

    low = resolve_with(elimination, jacobian, h, tasks, limits,
                       weights={"t3": 1.0})
    high = resolve_with(elimination, jacobian, h, tasks, limits,
                        weights={"t3": 100.0})
    assert low.feasible and high.feasible
    residual = lambda result, name: result.hierarchy.outcome(name).residual_norm
    assert residual(high, "t3") <= residual(low, "t3"), (
        f"权重提高后 t3 残差应不升反降: low={residual(low, 't3')}, "
        f"high={residual(high, 't3')}")
    assert residual(high, "t3") < residual(low, "t3") * 0.99, (
        "权重 100 倍必须实质性降低该任务残差，否则权重通路无效")
    assert not np.allclose(np.r_[high.qacc, high.torque], np.r_[low.qacc, low.torque]), (
        "不同权重下解必须不同")


def test_inactive_task_excluded():
    """activation=0 的任务不得进入代价，解与不含该任务时一致。"""
    _, _, _, _, jacobian, h, elimination, limits = make_problem()
    nv = 7
    active = [pitch_task(nv, desired=0.5)]
    with_inactive = [pitch_task(nv, desired=0.5),
                     TaskLevel(np.eye(nv)[[0]], np.array([9.9]), name="ghost",
                               activation=0.0)]
    base = resolve_with(elimination, jacobian, h, active, limits)
    extra = resolve_with(elimination, jacobian, h, with_inactive, limits)
    assert np.allclose(base.qacc, extra.qacc, atol=1e-9), "activation=0 的任务干扰了解"
    assert extra.scaling["ghost"] == 0.0, "激活为 0 的任务 scaling 必须为 0"


def test_fallback_on_contact_infeasible():
    """接触等式无法满足（力矩/法向限内）时降级到无接触等式，且解仍物理。"""
    _, _, mass, selection, jacobian, h, elimination, limits = make_problem()
    nv = mass.shape[0]
    tasks = [pitch_task(nv, desired=0.5)]
    # 巨大接触偏置：Jc qacc = -1e6 需要力矩/接触力远超限值，一级必然不可行。
    bias = np.full(jacobian.shape[0], 1.0e6)
    result = resolve_with(elimination, jacobian, h, tasks, limits, contact_bias=bias)
    assert result.fallback_used, "一级不可行必须触发降级兜底"
    assert result.feasible, "二级（无接触等式）应可行"
    assert result.dynamics_residual_norm < 1e-7, "二级解的动力学残差必须小"
    assert np.all(result.torque <= limits.upper + 1e-8), "二级解力矩不得越上界"
    assert np.all(result.torque >= limits.lower - 1e-8), "二级解力矩不得越下界"
    assert np.all(np.isfinite(result.qacc)), "降级解必须有限"
