"""任务语义与优先级集合（方案 §4.3 / §4.4）。

最关键的一条是 :func:`test_activation_never_touches_the_jacobian`：激活因子乘错
位置（乘到 ``J`` 而不是 ``ẍ*``）会破坏零空间的投影性质，让低优先级泄漏到高优先级，
而症状非常隐蔽。所以这里直接从结构上把它钉死。
"""

from __future__ import annotations

import numpy as np
import pytest

from Algorithm.wbc.tasks import PRIORITY_NAMES, Task, TaskDiagnostics, TaskSet

JACOBIAN = np.array([[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])


def make_task(name: str, activation: float = 1.0, desired=(1.0, -2.0)) -> Task:
    return Task(
        name=name,
        jacobian=JACOBIAN,
        desired_acceleration=np.array(desired, dtype=float),
        activation=activation,
    )


def test_activation_never_touches_the_jacobian():
    """任务传递原始目标、雅可比和激活因子，由求解器处理增量。"""
    task = make_task("height", activation=0.4)
    level = task.activated()
    np.testing.assert_allclose(level.jacobian, task.jacobian, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(level.desired_acceleration, task.desired_acceleration)
    assert level.activation == pytest.approx(0.4)
    assert level.damping == task.damping


def test_full_activation_is_identity():
    task = make_task("pitch")
    level = task.activated()
    np.testing.assert_allclose(level.desired_acceleration, task.desired_acceleration)
    assert task.active


def test_zero_activation_still_carries_the_jacobian():
    """保留原始任务供诊断；求解器依据 activation=0 释放权限。"""
    task = make_task("height", activation=0.0)
    level = task.activated()
    np.testing.assert_allclose(level.desired_acceleration, task.desired_acceleration)
    assert level.activation == 0.0
    np.testing.assert_allclose(level.jacobian, task.jacobian)
    assert not task.active


def test_activation_is_validated():
    with pytest.raises(ValueError, match="激活因子"):
        make_task("bad", activation=1.5)
    with pytest.raises(ValueError, match="激活因子"):
        make_task("bad", activation=-0.1)
    with pytest.raises(ValueError, match="阻尼"):
        Task(name="bad", jacobian=JACOBIAN, desired_acceleration=np.zeros(2), damping=-1.0)


def test_with_activation_returns_a_copy():
    task = make_task("pitch")
    faded = task.with_activation(0.2)
    assert faded.activation == pytest.approx(0.2)
    assert task.activation == pytest.approx(1.0)  # 原对象不动


# ---- 优先级集合 ----------------------------------------------------------


def test_task_set_preserves_priority_order():
    tasks = TaskSet((make_task("pitch"), make_task("roll"), make_task("height")))
    assert tasks.names() == ("pitch", "roll", "height")
    assert len(tasks) == 3
    assert [level.name for level in tasks.levels()] == ["pitch", "roll", "height"]


def test_task_set_rejects_duplicate_names():
    with pytest.raises(ValueError, match="不能重复"):
        TaskSet((make_task("pitch"), make_task("pitch")))


def test_level_lookup_and_missing_name():
    tasks = TaskSet((make_task("pitch"),))
    assert tasks.level("pitch").name == "pitch"
    with pytest.raises(KeyError):
        tasks.level("nope")


def test_budgets_are_collected_only_from_levels_that_have_one():
    """分层限值挂在任务上，只有给了预算的级才会被 SNS 约束（§5.1）。"""
    budget = (np.full(2, -4.0), np.full(2, 4.0))
    tasks = TaskSet(
        (
            make_task("pitch"),
            Task(
                name="height",
                jacobian=JACOBIAN,
                desired_acceleration=np.zeros(2),
                budget=budget,
            ),
        )
    )
    collected = tasks.budgets()
    assert set(collected) == {"height"}
    np.testing.assert_allclose(collected["height"][0], budget[0])
    np.testing.assert_allclose(collected["height"][1], budget[1])


def test_scaled_only_changes_the_listed_levels():
    tasks = TaskSet((make_task("pitch"), make_task("roll", activation=0.5)))
    scaled = tasks.scaled({"roll": 0.2})
    assert scaled.level("pitch").activation == pytest.approx(1.0)
    assert scaled.level("roll").activation == pytest.approx(0.2)
    assert tasks.level("roll").activation == pytest.approx(0.5)  # 原集合不动


def test_damp_below_implements_the_impact_protection():
    """§4.4：落地冲击时把低优先级整体降权，只保最上面两级。"""
    tasks = TaskSet(
        (
            make_task("pitch"),
            make_task("roll"),
            make_task("yaw"),
            make_task("height"),
            make_task("forward_velocity"),
        )
    )
    damped = tasks.damp_below(2, factor=0.2)
    assert damped.level("pitch").activation == pytest.approx(1.0)
    assert damped.level("roll").activation == pytest.approx(1.0)
    for name in ("yaw", "height", "forward_velocity"):
        assert damped.level(name).activation == pytest.approx(0.2)


def test_damp_below_validates_the_factor():
    tasks = TaskSet((make_task("pitch"),))
    with pytest.raises(ValueError, match="降权因子"):
        tasks.damp_below(0, 1.5)


def test_priority_names_match_the_documented_table():
    """§4.3 的优先级表：L0 力前馈、L1 俯仰、L2 横滚、L3 偏航、L4 高度、L5 速度、L6 正则。"""
    assert PRIORITY_NAMES[0] == "contact_force_feedforward"
    assert PRIORITY_NAMES[1] == "pitch"
    assert PRIORITY_NAMES[2] == "roll"
    assert PRIORITY_NAMES[3] == "yaw"
    assert PRIORITY_NAMES[4] == "height"
    assert PRIORITY_NAMES[5] == "forward_velocity"
    assert PRIORITY_NAMES[6] == "regularization"
    assert len(PRIORITY_NAMES) == 7


# ---- 诊断 ----------------------------------------------------------------


def test_task_diagnostics_reports_full_satisfaction():
    diagnostics = TaskDiagnostics(scaling={"pitch": 1.0, "height": 1.0})
    assert diagnostics.fully_satisfied
    diagnostics = TaskDiagnostics(scaling={"pitch": 1.0, "height": 0.4})
    assert not diagnostics.fully_satisfied
    assert TaskDiagnostics().fully_satisfied  # 空集合视为满足
