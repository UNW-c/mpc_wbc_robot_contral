"""MPC 优化轨迹与 WBC 的多频率解析闭环验证。

平面对象过驱动，未模拟侧向摩擦、单边接触切换，也未使用 MuJoCo。
不能据此宣称三维轮圈腿已稳定；欠驱动动力学另由独立反例测试覆盖。
"""

from __future__ import annotations

import numpy as np
import pytest

from Algorithm.mpc import ChassisState, MotionCommand
from Algorithm.wbc.projection import ContactElimination
from tests.joint_harness import (
    JointLoop,
    PlanarWbc,
    build_mpc,
    find_static_equilibrium,
    total_weight,
)
from tests.planar_robot import PlanarTwoWheelRobot

PHYSICS_DT = 0.001
TORQUE_LIMIT = 40.0


@pytest.fixture
def rig():
    """装配好的 (被控对象, MPC, WBC, 闭环, 平衡位形, 静态解)。"""
    robot = PlanarTwoWheelRobot()
    q0 = robot.standing_pose()
    torque0, force0, residual = find_static_equilibrium(robot, q0)
    mpc, model, linearization = build_mpc(robot, q0, torque0)
    wbc = PlanarWbc(robot)
    loop = JointLoop(robot, mpc, wbc, physics_dt=PHYSICS_DT)
    return {
        "robot": robot,
        "q0": q0,
        "torque0": torque0,
        "force0": force0,
        "residual": residual,
        "mpc": mpc,
        "model": model,
        "linearization": linearization,
        "wbc": wbc,
        "loop": loop,
    }


# ---- 被控对象与线性化 ----------------------------------------------------


def test_static_equilibrium_is_exact_and_physically_valid(rig):
    """静态解必须精确满足 ``Sᵀτ + Jcᵀf = h``，且两轮都受压（不是被地面拉住）。"""
    assert rig["residual"] < 1e-9
    force = rig["force0"]
    assert force[1] > 0.0 and force[3] > 0.0
    np.testing.assert_allclose(force[1] + force[3], total_weight(rig["robot"]), rtol=1e-6)
    qacc, dynamic_force = rig["robot"].constrained_acceleration(
        rig["q0"], np.zeros(7), rig["torque0"]
    )
    assert np.linalg.norm(qacc) < 1e-9
    np.testing.assert_allclose(dynamic_force, force, atol=1e-9)


def test_linearized_model_is_an_unstable_inverted_pendulum(rig):
    """俯仰必须是不稳定的（倒立摆），否则闭环验证没有意义。"""
    assert 5.0 < rig["model"].unstable_pole < 20.0
    A_long, B_long = rig["linearization"]
    assert B_long[2, 0] > 0.0  # 正的切向力把机身前推
    assert B_long[3, 0] > 0.0  # 同时把它往前撅（非最小相位）


# ---- 接口 handshake ------------------------------------------------------


def test_wbc_holds_the_equilibrium(rig):
    """零任务指令下，WBC 应当把静止平衡原样保持住。"""
    robot = rig["robot"]
    torque = rig["wbc"].solve(rig["q0"], np.zeros(7), np.zeros(2), (0.0, 0.0, 0.0), (0.0, 0.0))[
        "torque"
    ]
    assert np.max(np.abs(torque)) < TORQUE_LIMIT
    qacc, force = robot.constrained_acceleration(rig["q0"], np.zeros(7), torque)
    # 机身那三维接近不动；残留的 1e-4 量级来自"最小范数力矩"与"最小范数(τ,f)解"
    # 在内部力分配上的差别——两个都是合法的静态解族成员，不是误差。
    assert np.linalg.norm(qacc[:3]) < 1e-3
    assert np.linalg.norm(qacc) < 1e-2
    assert force[1] > 0.0 and force[3] > 0.0


def test_contact_forces_cannot_reach_the_actuators(rig):
    """QR 消力的核心事实：``P·Jcᵀ ≡ 0``，所以接触力**命令不动**执行器。

    这一条决定了方案 §4.3 的 L0（``τ_ff = −Jcᵀf*`` 的驱动空间投影）在零空间框架里
    恒等于零：执行器解经过 ``P = (Q_uᵀSᵀ)⁺Q_uᵀ`` 投影，而消力构造保证
    ``Q_uᵀJcᵀ ≡ 0``。要命令接触力，必须换一种把 ``f`` 留在决策变量里的 WBC 结构
    （也就是文档特意避开的 QP-WBC）。
    """
    robot = rig["robot"]
    elimination = ContactElimination.from_matrices(
        robot.contact_jacobian(rig["q0"]), robot.selection(), robot.mass_matrix(rig["q0"])
    )
    projector = np.linalg.pinv(elimination.actuation_matrix, rcond=1e-12) @ elimination.q_u.T
    assert np.max(np.abs(projector @ robot.contact_jacobian(rig["q0"]).T)) < 1e-12
    # 但经过 M⁻¹ 之后就不为零了——力只能以加速度形式进入解
    assert np.max(np.abs(elimination.torque_gain @ robot.contact_jacobian(rig["q0"]).T)) > 0.1


def test_force_feedforward_path_is_wired_when_enabled(rig):
    """把 ``f*`` 折算成加速度前馈后，它确实能改变关节力矩（前提：喂给有权限的层级）。"""
    robot, q0 = rig["robot"], rig["q0"]
    wbc = PlanarWbc(robot, force_feedforward=True)
    reference = (0.0, 0.0, 0.0)
    base = wbc.solve(q0, np.zeros(7), np.zeros(2), reference, (0.0, 0.0))["torque"]
    pushed = wbc.solve(q0, np.zeros(7), np.array([10.0, 10.0]), reference, (0.0, 0.0))["torque"]
    assert not np.allclose(base, pushed)
    # 前向力应当让两轮都给出同向力矩
    delta = pushed - base
    assert np.sign(delta[1]) == np.sign(delta[3])
    # 关掉这条通路就完全不生效
    quiet = PlanarWbc(robot, force_feedforward=False)
    np.testing.assert_allclose(
        quiet.solve(q0, np.zeros(7), np.array([10.0, 10.0]), reference, (0.0, 0.0))["torque"],
        quiet.solve(q0, np.zeros(7), np.zeros(2), reference, (0.0, 0.0))["torque"],
    )


def test_pitch_reference_forward_uses_angular_units(rig):
    """``θ̈*`` 前馈必须来自俯仰参考的差分，不能拿纵向线加速度顶替（量纲错）。"""
    loop = rig["loop"]
    solution = rig["mpc"].solve(ChassisState(), MotionCommand(velocity=0.3))
    sample = solution.trajectory.sample(0.0)
    assert loop.pitch_acceleration_reference(solution) == pytest.approx(sample.acceleration[1])
    epsilon = 1e-6
    derivative = (solution.trajectory.sample(epsilon).state[3] - sample.state[3]) / epsilon
    assert loop.pitch_acceleration_reference(solution) == pytest.approx(derivative, rel=1e-3)



# ---- 闭环 ----------------------------------------------------------------


@pytest.mark.parametrize("command", [0.2, 0.3, 0.5, 0.8])
def test_velocity_command_is_tracked(rig, command):
    result = rig["loop"].run(MotionCommand(velocity=command), ticks=200, q=rig["q0"])
    _, qvel = result["final"]
    history = result["history"]
    assert qvel[0] == pytest.approx(command, abs=0.02)
    # 俯仰被稳住：加速段的前倾是必要的（θ* 本身就有量级），峰值不超过 0.25 rad
    # （0.8 m/s 指令实测 0.212），末态回到零附近
    assert np.max(np.abs(history["pitch"])) < 0.25
    assert abs(history["pitch"][-1]) < 0.02
    # 走过的距离要跟得上（起段有 jerk 限幅的加速过程，略小于匀速理想值）
    travelled = history["position"][-1] - history["position"][0]
    ideal = command * len(history["velocity"]) * 0.02
    assert 0.7 * ideal < travelled < 1.05 * ideal


def test_torques_and_contact_forces_stay_physical(rig):
    """全程 τ 在限值内、两轮法向力为正（负法向等于"地面拉住轮子"，仿真已失效）。"""
    result = rig["loop"].run(MotionCommand(velocity=0.5), ticks=200, q=rig["q0"])
    history = result["history"]
    assert np.max(np.abs(history["torque"])) < TORQUE_LIMIT
    normals = history["contact_force"][:, [1, 3]]
    assert np.min(normals) > 0.0
    assert result["minimum_normal_force"] > 0.0
    assert not np.any(result["control_history"]["fallback"])
    assert np.max(result["control_history"]["dynamics_residual"]) < 1e-8
    assert np.mean(np.sum(normals, axis=1)) == pytest.approx(
        total_weight(rig["robot"]), rel=0.2
    )


def test_pitch_follows_the_mpc_reference(rig):
    """WBC 的 L1 必须真的在追 MPC 的 ``θ*``（这是 §0 说的全系统最重要的信号）。"""
    history = rig["loop"].run(MotionCommand(velocity=0.6), ticks=200, q=rig["q0"])["history"]
    error = np.abs(history["pitch"] - history["pitch_reference"])
    assert np.max(error) < 0.1
    assert np.max(np.abs(history["pitch_reference"])) > 0.01  # 参考确实在动


def test_no_mpc_fallback_and_no_wbc_saturation(rig):
    """名义工况下不该触发生效兜底，也不该让 WBC 触发任务缩放。"""
    history = rig["loop"].run(MotionCommand(velocity=0.4), ticks=200, q=rig["q0"])["history"]
    assert int(np.sum(history["mpc_fallback"])) == 0
    # 正则级在这个过驱动平面模型中无剩余权限；不可将其误报为已完整执行。
    for name in ("pitch", "height", "forward_velocity"):
        assert min(scaling[name] for scaling in history["scaling"]) == pytest.approx(1.0)
    assert all(scaling["regularization"] == 0.0 for scaling in history["scaling"])


def test_l0_does_not_cancel_the_pitch_task(rig):
    """L0 若同时喂给俯仰任务，会把 L1 的纠正项抹平——所以默认只喂前向那一级。

    这条是实测踩出来的：力同时含"往前推"和"往前撅"，而俯仰已经有 ``θ*`` 这条驱动
    通路，两边一起喂会互相抵消（实测 L1 的 −11.0 rad/s² 被 L0 的 +11.1 抹平，
    俯仰卡在 0.135 rad 的稳态误差上不动）。
    """
    assert "pitch" not in rig["wbc"].feedforward_tasks
    assert "forward_velocity" in rig["wbc"].feedforward_tasks
    # 打开 L0 且只喂前向时，俯仰仍然被稳住
    loop = JointLoop(rig["robot"], rig["mpc"], PlanarWbc(rig["robot"], force_feedforward=True))
    history = loop.run(MotionCommand(velocity=0.3), ticks=150, q=rig["q0"])["history"]
    assert abs(history["pitch"][-1]) < 0.05
    assert np.min(history["contact_force"][:, [1, 3]]) > 0.0
