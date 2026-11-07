"""A1（第 15 次排查）：规划限值由 hip_angle_limit 自动计算。

对应 docs/算法漏洞排查记录.md「第 15 次排查：交给 agent1 的执行清单 -> A1」。
"""

from __future__ import annotations

import dataclasses

import pytest

from Algorithm.mpc import MPCSettings, ReducedModel, WheelMPC


@pytest.fixture(scope="module")
def hip_input_model() -> ReducedModel:
    """带稳态能力表的 hip_input 降阶模型（构建需要求解稳态族，模块内复用）。"""
    from APP.mujoco_balance_model import HipInputModelBuilder
    from APP.mujoco_mpc_wbc import MuJoCoDynamics
    from APP.mujoco_robot import MuJoCoRobot

    robot = MuJoCoRobot.load("plane")
    builder = HipInputModelBuilder(MuJoCoDynamics(robot))
    return builder.build_reduced(0.0)


def test_capacity_table_is_monotonic(hip_input_model: ReducedModel) -> None:
    """能力表按 a 递增给出 q_c，且覆盖设计范围 [0, 2.4]。"""
    capacity = hip_input_model.metadata["accel_capacity"]
    accelerations = capacity["a"]
    q_values = capacity["q_c"]
    assert accelerations[0] == pytest.approx(0.0)
    # 实测 a >= 2.2 的稳态解退化（q_c 饱和），表在该处截断；至少覆盖到 2.0。
    assert accelerations[-1] >= 2.0
    assert len(accelerations) >= 20
    assert all(b > a for a, b in zip(q_values, q_values[1:]))


def test_limits_match_motor_table(hip_input_model: ReducedModel) -> None:
    """插值结果与 B1 / M3 的稳态数据一致：q=0.55 -> 1.36，q=0.8 -> 1.79。"""
    for hip_limit, expected_capacity in ((0.55, 1.36), (0.8, 1.79)):
        settings = MPCSettings(hip_angle_limit=hip_limit, compensate_hip_torque=False)
        controller = WheelMPC(hip_input_model, settings)
        limits = controller.settings.limits
        capacity = limits.commanded_acceleration / settings.accel_fraction_up
        assert capacity == pytest.approx(expected_capacity, abs=0.03)
        assert limits.commanded_acceleration == pytest.approx(
            settings.accel_fraction_up * capacity)
        assert limits.max_braking == pytest.approx(settings.accel_fraction_down * capacity)
        assert limits.max_acceleration == pytest.approx(settings.accel_fraction_down * capacity)


def test_limits_follow_fractions(hip_input_model: ReducedModel) -> None:
    """比例系数可改：0.85 / 0.95 换成 0.8 / 0.9 后限值按新系数缩放。"""
    base = WheelMPC(hip_input_model, MPCSettings(
        hip_angle_limit=0.8, compensate_hip_torque=False))
    capacity = base.settings.limits.commanded_acceleration / 0.85
    custom = WheelMPC(hip_input_model, MPCSettings(
        hip_angle_limit=0.8, compensate_hip_torque=False,
        accel_fraction_up=0.8, accel_fraction_down=0.9))
    limits = custom.settings.limits
    assert limits.commanded_acceleration == pytest.approx(0.8 * capacity)
    assert limits.max_braking == pytest.approx(0.9 * capacity)
    assert limits.max_acceleration == pytest.approx(0.9 * capacity)


def test_disabled_keeps_manual_limits(hip_input_model: ReducedModel) -> None:
    """auto_accel_limits=False 时规划器参数与手填值逐位一致。"""
    manual = MPCSettings(hip_angle_limit=0.8, compensate_hip_torque=False)
    settings = MPCSettings(hip_angle_limit=0.8, auto_accel_limits=False,
                           compensate_hip_torque=False)
    controller = WheelMPC(hip_input_model, settings)
    assert controller.settings.limits == manual.limits
    assert controller.settings.limits.commanded_acceleration == 1.2
    assert controller.settings.limits.max_braking == 1.4
    assert controller.settings.limits.max_acceleration == 1.4


class _FakeModel:
    """没有能力表的模型（模拟 held_hip）：只实现 _apply_auto_accel_limits 用到的字段。"""

    source = "mujoco-held-hip"
    metadata: dict[str, object] = {}


def test_model_without_capacity_keeps_manual_limits() -> None:
    """held_hip 等没有能力表的模型保持手填值，且返回同一个对象（不做 replace）。"""
    settings = MPCSettings(compensate_hip_torque=False)
    result = WheelMPC._apply_auto_accel_limits(settings, _FakeModel())
    assert result is settings
    assert result.limits.commanded_acceleration == 1.2
    assert result.limits.max_braking == 1.4


def test_hold_hip_settings_are_not_mutated(hip_input_model: ReducedModel) -> None:
    """输入 settings 不被原地修改（WheelMPC 用 dataclasses.replace 返回新对象）。"""
    settings = MPCSettings(hip_angle_limit=0.8, compensate_hip_torque=False)
    snapshot = dataclasses.replace(settings)
    WheelMPC(hip_input_model, settings)
    assert settings == snapshot
