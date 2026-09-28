"""A3（T7a）：HipInputModelBuilder 的核验（8 状态 + 髋输入）。

五步（按排查记录 A3）：
1. 线性化自检（derivative 与 A x + B u + affine）；
2. 开环 B 列（0.1 N*m 共模轮力矩 / 0.1 N*m 共模髋力矩）；
3. 开环 A 列（初始 q_c = 0.05）；
4. 稳态分解（a = 0.9 m/s^2，两种权重）；
5. 可控性（纵向子块）。

运行（打印诊断数值）：
    .venv/Scripts/python.exe -m pytest tests/test_hip_input_model.py -s
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import mujoco

from APP.mujoco_balance_model import HipInputModelBuilder
from APP.mujoco_mpc_wbc import MuJoCoDynamics
from APP.mujoco_robot import MuJoCoRobot

_OPEN_LOOP_TORQUE = 0.1      # [N*m]
_OPEN_LOOP_DURATION = 0.02   # [s]
# 测量窗口 1-10 ms：启动瞬态之后、非线性漂移之前。
# 依据（2026-09-27 窗口扫描）：A3-2b 髋力矩在 1-10 ms 误差 <1.5%，
# 用 5-50 ms 会因 q_c 不稳定发散（q_c_ddot ~13 rad/s^2）涨到 20%。
_WINDOW_START = 0.001        # [s]
_WINDOW_END = 0.010          # [s]
_HIP_KP = 200.0              # [N*m/rad] 差模髋握持 PD


@pytest.fixture(scope="module")
def builder():
    robot = MuJoCoRobot.load("plane")
    dynamics = MuJoCoDynamics(robot)
    return HipInputModelBuilder(dynamics)


@pytest.fixture(scope="module")
def model(builder):
    return builder.build(0.0)


def test_step1_linearization_selfcheck(builder, model):
    rng = np.random.default_rng(20260927)
    worst = 0.0
    for _ in range(5):
        state = rng.uniform(-1e-3, 1e-3, size=8)
        force = rng.uniform(-0.1, 0.1, size=3)
        measured = builder.derivative(state, force)
        predicted = model.affine + model.A @ state + model.B @ force
        worst = max(worst, float(np.linalg.norm(measured - predicted) / max(np.linalg.norm(measured), 1e-9)))
    print(f"\n[A3-1] 线性化最大相对误差 = {worst:.3e}")
    assert worst < 0.01


def _open_loop(builder, *, q_c0=0.0, wheel_torque=0.0, hip_torque=0.0,
               trim_at=None, duration=_OPEN_LOOP_DURATION):
    """差模髋 PD 握持（目标 0）+ 共模髋自由（由 hip_torque 驱动）。

    共模广义力 Q_c = F1 + sign*F3；要 Q_c = 2*tau_c、Q_d = tau_d 时：
    F1 = tau_c + tau_d/2，F3 = sign*(tau_c - tau_d/2)。
    返回 (times, v, theta_rate, hip_common, hip_common_rate)。
    """
    robot = builder.robot
    dynamics = builder.dynamics
    data = robot.data
    mj = robot.model
    hip_q = builder.hip_q
    hip_dof = builder.hip_dof
    sign = builder.hip_sign
    acts = {name: int(mj.actuator(name).id)
            for name in ("j1_ctrl", "j3_ctrl", "j2_ctrl", "j4_ctrl")}
    qpos0, qvel0, ctrl0 = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    try:
        # trim 位置与初始状态分离：A 列测试在 q_c=0 的平衡点上给 q_c 加偏移
        theta_trim, _ = builder.trim(q_c0 if trim_at is None else trim_at)
        data.qpos[:] = builder.pose
        mujoco.mju_quatIntegrate(data.qpos[3:7], np.array([0.0, theta_trim, 0.0]), 1.0)
        data.qpos[hip_q[0]] = float(q_c0)
        data.qpos[hip_q[1]] = sign * float(q_c0)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        robot.forward()
        inertia = float(dynamics.mass()[hip_dof[0], hip_dof[0]])
        kp_d = _HIP_KP
        kd_d = min(2.0 * np.sqrt(kp_d * inertia), 0.8 * 2.0 * inertia / robot.timestep)

        steps = int(round(duration / robot.timestep))
        times = np.zeros(steps + 1)
        velocity = np.zeros(steps + 1)
        theta_rate = np.zeros(steps + 1)
        hip_common = np.zeros(steps + 1)
        hip_common_rate = np.zeros(steps + 1)
        for step in range(steps + 1):
            rows = dynamics.matrices((True, True))[0]
            velocity[step] = float(rows[0] @ data.qvel)
            theta_rate[step] = float(rows[2] @ data.qvel)
            hip_common_rate[step] = float(builder.common_row @ data.qvel)
            hip_common[step] = 0.5 * (data.qpos[hip_q[0]] + sign * data.qpos[hip_q[1]])
            if step == steps:
                break
            tau_d = -kp_d * hip_common[step] - kd_d * hip_common_rate[step]
            data.ctrl[acts["j1_ctrl"]] = float(hip_torque + 0.5 * tau_d)
            data.ctrl[acts["j3_ctrl"]] = float(sign * (hip_torque - 0.5 * tau_d))
            data.ctrl[acts["j2_ctrl"]] = +wheel_torque
            data.ctrl[acts["j4_ctrl"]] = -wheel_torque
            mujoco.mj_step(mj, data)
            times[step + 1] = (step + 1) * robot.timestep
    finally:
        data.qpos[:], data.qvel[:], data.ctrl[:] = qpos0, qvel0, ctrl0
        robot.forward()
    return times, velocity, theta_rate, hip_common, hip_common_rate


def _slopes(times, values):
    mask = (times >= _WINDOW_START) & (times <= _WINDOW_END)
    t = times[mask]
    return float((values[mask][-1] - values[mask][0]) / (t[-1] - t[0]))


def test_step2_open_loop_b_columns(builder, model):
    radius = builder.radius
    force_each = _OPEN_LOOP_TORQUE / radius
    times, velocity, theta_rate, _, hip_rate = _open_loop(builder, wheel_torque=_OPEN_LOOP_TORQUE)
    v_dot = _slopes(times, velocity)
    theta_ddot = _slopes(times, theta_rate)
    hip_ddot = _slopes(times, hip_rate)
    v_pred = float((model.B[1, 0] + model.B[1, 1]) * force_each)
    theta_pred = float((model.B[3, 0] + model.B[3, 1]) * force_each)
    hip_pred = float((model.B[7, 0] + model.B[7, 1]) * force_each)
    print(f"\n[A3-2a] 轮力矩 0.1 N*m: v_dot {v_dot:+.4f} (预测 {v_pred:+.4f}) "
          f"theta_ddot {theta_ddot:+.4f} (预测 {theta_pred:+.4f}) "
          f"q_c_ddot {hip_ddot:+.4f} (预测 {hip_pred:+.4f})")
    assert abs(v_dot - v_pred) / abs(v_pred) < 0.15
    assert abs(theta_ddot - theta_pred) / abs(theta_pred) < 0.15
    assert abs(hip_ddot - hip_pred) / abs(hip_pred) < 0.20

    times, velocity, theta_rate, _, hip_rate = _open_loop(builder, hip_torque=_OPEN_LOOP_TORQUE)
    v_dot = _slopes(times, velocity)
    theta_ddot = _slopes(times, theta_rate)
    hip_ddot = _slopes(times, hip_rate)
    v_pred = float(model.B[1, 2] * _OPEN_LOOP_TORQUE)
    theta_pred = float(model.B[3, 2] * _OPEN_LOOP_TORQUE)
    hip_pred = float(model.B[7, 2] * _OPEN_LOOP_TORQUE)
    print(f"[A3-2b] 髋力矩 0.1 N*m: v_dot {v_dot:+.4f} (预测 {v_pred:+.4f}) "
          f"theta_ddot {theta_ddot:+.4f} (预测 {theta_pred:+.4f}) "
          f"q_c_ddot {hip_ddot:+.4f} (预测 {hip_pred:+.4f})")
    assert abs(v_dot - v_pred) / abs(v_pred) < 0.15
    assert abs(theta_ddot - theta_pred) / abs(theta_pred) < 0.15
    assert abs(hip_ddot - hip_pred) / abs(hip_pred) < 0.20


def test_step3_open_loop_a_column(builder, model):
    q_c0 = 0.05
    times, velocity, theta_rate, _, hip_rate = _open_loop(builder, q_c0=q_c0, trim_at=0.0)
    v_dot = _slopes(times, velocity)
    theta_ddot = _slopes(times, theta_rate)
    hip_ddot = _slopes(times, hip_rate)
    v_pred = float(model.A[1, 6] * q_c0)
    theta_pred = float(model.A[3, 6] * q_c0)
    hip_pred = float(model.A[7, 6] * q_c0)
    print(f"\n[A3-3] q_c0=0.05: v_dot {v_dot:+.4f} (预测 {v_pred:+.4f}) "
          f"theta_ddot {theta_ddot:+.4f} (预测 {theta_pred:+.4f}) "
          f"q_c_ddot {hip_ddot:+.4f} (预测 {hip_pred:+.4f})")
    assert abs(theta_ddot - theta_pred) / abs(theta_pred) < 0.25
    assert abs(v_dot - v_pred) / abs(v_pred) < 0.30
    assert abs(hip_ddot - hip_pred) / abs(hip_pred) < 0.30


def test_step4_steady_state_decomposition(model):
    """a = 0.9 m/s^2 的稳态 (theta, q_c, F, tau_c)，两种权重。"""
    a_target = 0.9
    C = np.zeros((3, 4))
    for row, state_row in enumerate((3, 7, 1)):
        C[row, 0] = model.A[state_row, 2]
        C[row, 1] = model.A[state_row, 6]
        C[row, 2] = 0.5 * (model.B[state_row, 0] + model.B[state_row, 1])
        C[row, 3] = model.B[state_row, 2]
    b = np.array([0.0, 0.0, a_target])

    def solve(weights, label):
        y0 = np.linalg.lstsq(C, b, rcond=None)[0]
        _, _, vt = np.linalg.svd(C)
        null = vt[-1]
        W = np.diag(weights)
        t = -float((null @ W @ y0) / (null @ W @ null))
        y = y0 + null * t
        theta, q_c, total_force, tau_c = y
        print(f"[A3-4] {label}: theta={theta:+.4f} rad, q_c={q_c:+.4f} rad, "
              f"F={total_force:+.2f} N, tau_c={tau_c:+.4f} N*m")
        return y

    y_a = solve([1e6, 1.0, 1e-6, 1e-6], "theta权重 1e6（近 T8 行为）")
    solve([1.0, 1.0, 1e-6, 1e-6], "权重相等")
    np.testing.assert_allclose(C @ y_a, b, atol=1e-9)
    print("[A3-4] T8 实测参考: theta ~ 0, q_c 0.30~0.37, F ~ 9 N, tau_c ~ 0.3 N*m")
    assert abs(y_a[1]) < 1.0 and abs(y_a[3]) < 10.0


def test_step5_controllability(model):
    sub = [0, 1, 2, 3, 6, 7]
    A = model.A[np.ix_(sub, sub)]
    B = np.column_stack([0.5 * (model.B[sub, 0] + model.B[sub, 1]), model.B[sub, 2]])
    blocks = [B]
    for _ in range(5):
        blocks.append(A @ blocks[-1])
    ctrb = np.hstack(blocks)
    rank = int(np.linalg.matrix_rank(ctrb))
    smallest = float(np.min(np.linalg.svd(ctrb, compute_uv=False)))
    print(f"\n[A3-5] 纵向子块可控性矩阵: 秩={rank}/6, 最小奇异值={smallest:.3e}")
    assert rank == 6
    assert smallest > 1e-9


def test_step6_steady_state_family(builder):
    """B1（#14）：沿稳态加速度族的工作点与 A/B 变化、不稳定极点。"""
    order = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    family = {}
    print()
    for a in order:
        sol = builder.steady_state(a)
        state = np.zeros(8)
        state[2] = sol["theta"]
        state[6] = sol["q_c"]
        force = np.array([sol["force_per_wheel"], sol["force_per_wheel"], sol["tau_c"]])
        family[a] = (sol, builder.build_at(state, force))
        print(f"[B1] a={a:+.1f}: theta={sol['theta']:+.4f} q_c={sol['q_c']:+.4f} "
              f"F={sol['force']:+.2f} tau_c={sol['tau_c']:+.4f} success={sol['success']} "
              f"residual={sol['residual']:.2e}")

    base = family[0.0][1]
    sub = [0, 1, 2, 3, 6, 7]
    print("[B1] 相对 a=0 的 A/B 变化（分母 max(|元素|, 行最大值 5%)）：")
    for a in order:
        if a == 0.0:
            continue
        model = family[a][1]
        rels = []
        for row in (1, 3, 7):
            denom = np.maximum(np.abs(base.A[row]), 0.05 * np.max(np.abs(base.A[row])))
            rels.append(np.abs(model.A[row] - base.A[row]) / denom)
        for col in range(3):
            denom = np.maximum(np.abs(base.B[:, col]), 0.05 * np.max(np.abs(base.B[:, col])))
            rels.append(np.abs(model.B[:, col] - base.B[:, col]) / denom)
        allrel = np.concatenate(rels)
        poles = np.linalg.eigvals(model.A[np.ix_(sub, sub)])
        print(f"  a={a:+.1f}: 平均变化={np.mean(allrel):.1%} 峰值={np.max(allrel):.1%} "
              f"最大极点={float(np.max(poles.real)):+.2f} rad/s")

    print("[B1] 边界探针 a=±2.0：")
    for a in (-2.0, 2.0):
        sol = builder.steady_state(a)
        print(f"  a={a:+.1f}: theta={sol['theta']:+.4f} q_c={sol['q_c']:+.4f} F={sol['force']:+.2f} "
              f"tau_c={sol['tau_c']:+.4f} success={sol['success']} residual={sol['residual']:.2e}")
