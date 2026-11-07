"""T0：HeldHipModelBuilder 降阶模型的稳态关系核验（排查记录 #9）。

只做诊断，不修改控制代码。五步：
1. 解析稳态增益 k_theta / k_force（由 A/B 求）；
2. 刚体估算 k_theta（整机质心相对接触中点的几何关系，取量级判据）；
3. 线性化自检（derivative 与 A x + B u + affine 对比，含 rows 顺序核对）；
4. 开环核验 B 列：PD 握髋 + 恒定前进轮力矩，与 B 列对比；
5. 开环核验 A 列：初始倾角（绕接触点）+ 零轮力矩，与 A 列对比。

测量方法说明（2026-09-27 实测定标）：
- 髋 PD 的阻尼项必须满足离散稳定 kd < 2*I_hip/dt，否则髋力矩在正负限幅间
  振荡并污染速度测量（第一次实验用 kd=20 就是这个问题，I_hip=0.0164）；
- 姿态变化会平移轮子，必须先补偿机身位置使接触点回到原位，否则质心相对
  接触点的偏移会把结果符号弄反；
- 初值用"跳过前 5 ms 的差商"而不是全窗口二次拟合：启动瞬态（PD 建立、
  接触调整）会污染拟合。

运行（打印诊断数值）：
    .venv/Scripts/python.exe -m pytest tests/test_held_hip_model.py -s
"""
from __future__ import annotations

import numpy as np
import pytest
import mujoco

from Algorithm.geometry import quaternion_pitch
from APP.mujoco_balance_model import HeldHipModelBuilder
from APP.mujoco_mpc_wbc import MuJoCoDynamics
from APP.mujoco_robot import MuJoCoRobot

_GRAVITY = 9.81
_OPEN_LOOP_TORQUE = 0.1      # [N*m] 每轮，前进方向（ctrl 差模：j2=+tau, j4=-tau）
_OPEN_LOOP_DURATION = 0.05   # [s] 仿真时长
_SKIP_WINDOW = 0.005         # [s] 跳过启动瞬态
_HIP_KP = 200.0              # [N*m/rad] 握髋 PD 刚度（T0 要求 >= 200）


@pytest.fixture(scope="module")
def rig():
    robot = MuJoCoRobot.load("plane")
    dynamics = MuJoCoDynamics(robot)
    builder = HeldHipModelBuilder(dynamics)
    model = builder.build()
    return robot, dynamics, builder, model


def _steady_gains(model):
    A, B = model.A, model.B
    k_theta = float(B[3, 0] / (A[1, 2] * B[3, 0] - A[3, 2] * B[1, 0]))
    k_force = float(-A[3, 2] * k_theta / B[3, 0])
    return k_theta, k_force


def _contact_midpoint(robot, dynamics, radius):
    data = robot.data
    points = []
    for geom in dynamics.wheels:
        axle = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        down = np.array([0.0, 0.0, -1.0])
        radial = down - axle * float(axle @ down)
        radial = radial / np.linalg.norm(radial)
        points.append(data.geom_xpos[geom].copy() + radius * radial)
    return 0.5 * (points[0] + points[1])


def _hip_gains(dynamics, hip_v):
    """稳定握髋 PD：kd 取临界阻尼与离散稳定边界（kd < 2I/dt）的较小值。"""
    inertia = float(dynamics.mass()[hip_v[0], hip_v[0]])
    dt = float(dynamics.robot.timestep)
    kd = min(2.0 * np.sqrt(_HIP_KP * inertia), 0.8 * 2.0 * inertia / dt)
    return _HIP_KP, float(kd)


def _run_open_loop(rig, *, initial_pitch=0.0, wheel_torque=0.0, duration=_OPEN_LOOP_DURATION):
    """PD 握髋开环实验。

    initial_pitch 绕接触点施加（先转姿态再补偿平移）；wheel_torque 给 j2=+tau、
    j4=-tau（前进）。返回 (times, pitch, pitch_rate, velocity) 序列。
    """
    robot, dynamics, builder, model = rig
    data = robot.data
    mj = robot.model
    hip_q = [int(mj.joint(name).qposadr[0]) for name in ("joint1", "joint3")]
    hip_v = [int(mj.joint(name).dofadr[0]) for name in ("joint1", "joint3")]
    actuators = {name: int(mj.actuator(name).id)
                 for name in ("j1_ctrl", "j3_ctrl", "j2_ctrl", "j4_ctrl")}
    ranges = mj.actuator_ctrlrange
    kp, kd = _hip_gains(dynamics, hip_v)

    base = builder.pose.copy()
    qpos0, qvel0, ctrl0 = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    try:
        data.qpos[:] = base
        data.qvel[:] = 0.0
        robot.forward()
        mid0 = _contact_midpoint(robot, dynamics, builder.radius)
        equilibrium = float(model.metadata["equilibrium_pitch"])
        data.qpos[:] = base
        mujoco.mju_quatIntegrate(data.qpos[3:7], np.array([0.0, equilibrium + initial_pitch, 0.0]), 1.0)
        robot.forward()
        mid = _contact_midpoint(robot, dynamics, builder.radius)
        data.qpos[0] += mid0[0] - mid[0]
        data.qpos[2] += mid0[2] - mid[2]
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        robot.forward()
        hip_target = data.qpos[hip_q].copy()

        steps = int(round(duration / robot.timestep))
        times = np.zeros(steps + 1)
        pitch = np.zeros(steps + 1)
        rate = np.zeros(steps + 1)
        velocity = np.zeros(steps + 1)
        pitch[0] = quaternion_pitch(data.qpos[3:7]) - equilibrium
        for step in range(steps):
            for index, name in enumerate(("j1_ctrl", "j3_ctrl")):
                error = hip_target[index] - data.qpos[hip_q[index]]
                command = kp * error - kd * data.qvel[hip_v[index]]
                lo, hi = ranges[actuators[name]]
                data.ctrl[actuators[name]] = float(np.clip(command, lo, hi))
            data.ctrl[actuators["j2_ctrl"]] = +wheel_torque
            data.ctrl[actuators["j4_ctrl"]] = -wheel_torque
            mujoco.mj_step(mj, data)
            times[step + 1] = (step + 1) * robot.timestep
            rows = dynamics.matrices((True, True))[0]
            pitch[step + 1] = quaternion_pitch(data.qpos[3:7]) - equilibrium
            rate[step + 1] = float(rows[2] @ data.qvel)
            velocity[step + 1] = float(rows[0] @ data.qvel)
    finally:
        data.qpos[:], data.qvel[:], data.ctrl[:] = qpos0, qvel0, ctrl0
        robot.forward()
    return times, pitch, rate, velocity


def _slopes(times, pitch, rate, velocity):
    """跳过瞬态后的平均斜率：dtheta_dot/dt 与 dv/dt。"""
    mask = times >= _SKIP_WINDOW
    t = times[mask]
    pitch_dot_slope = (rate[mask][-1] - rate[mask][0]) / (t[-1] - t[0])
    velocity_slope = (velocity[mask][-1] - velocity[mask][0]) / (t[-1] - t[0])
    return float(pitch_dot_slope), float(velocity_slope)


def test_step1_steady_state_gains(rig):
    _, _, _, model = rig
    A, B = model.A, model.B
    k_theta, k_force = _steady_gains(model)
    print(f"\n[T0-1] A[1,2]={A[1,2]:.6f} A[3,2]={A[3,2]:.6f}")
    print(f"[T0-1] B[1,:]={np.round(B[1], 6)} B[3,:]={np.round(B[3], 6)}")
    print(f"[T0-1] k_theta={k_theta:.4f} rad per (m/s^2)，k_force={k_force:.4f} N per (m/s^2)")
    print(f"[T0-1] 0.8 m/s^2 需要 theta={0.8 * k_theta:.4f} rad, F={0.8 * k_force:.2f} N")
    print(f"[T0-1] 2.0 m/s^2 需要 theta={2.0 * k_theta:.4f} rad, F={2.0 * k_force:.2f} N")
    assert k_theta > 0.0 and k_force > 0.0


def test_step2_rigid_body_estimate(rig):
    """整机质心几何估算 k_theta（与解析值比量级：比值落在 [0.5, 2] 内为通过）。"""
    robot, dynamics, builder, model = rig
    data = robot.data
    base = builder.pose.copy()
    radius = builder.radius
    heading = np.array([1.0, 0.0, 0.0])

    def com_and_mid(pitch):
        data.qpos[:] = base
        mujoco.mju_quatIntegrate(data.qpos[3:7], np.array([0.0, pitch, 0.0]), 1.0)
        robot.forward()
        com = np.asarray(data.subtree_com[0], dtype=float).copy()
        return com, _contact_midpoint(robot, dynamics, radius)

    com0, mid0 = com_and_mid(0.0)
    height = float(com0[2] - mid0[2])
    delta = 1.0e-3
    com_p, _ = com_and_mid(delta)
    com_m, _ = com_and_mid(-delta)
    ddx_dtheta = (float((com_p - com_m) @ heading)) / (2.0 * delta)
    k_theta_estimate = height / (_GRAVITY * abs(ddx_dtheta))
    k_theta_model, _ = _steady_gains(model)
    ratio = k_theta_estimate / k_theta_model
    print(f"\n[T0-2] 质心高度 h={height:.6f} m，d(dx)/d(theta)={ddx_dtheta:+.4f} m/rad")
    print(f"[T0-2] 几何估算 |k_theta|={k_theta_estimate:.4f}，解析 k_theta={k_theta_model:.4f}，比值={ratio:.3f}")
    if 0.5 <= ratio <= 2.0:
        print("[T0-2] 结论：与解析值同量级，未判定为模型错误（符号与 pitch 约定相关，只比量级）")
    else:
        print("[T0-2] 结论：与解析值相差超过 2 倍，疑似模型问题")
    assert 0.5 <= ratio <= 2.0


def test_step3_linearization_selfcheck(rig):
    """derivative() 与 A x + B u + affine 的线性化误差；核对 rows 行序。"""
    robot, dynamics, builder, model = rig
    rng = np.random.default_rng(20260927)
    affine = builder.affine
    A, B = model.A, model.B
    max_error = 0.0
    for _ in range(5):
        state = rng.uniform(-1.0e-3, 1.0e-3, size=6)
        force = rng.uniform(-0.1, 0.1, size=2)
        measured = builder.derivative(state, force)
        predicted = affine + A @ state + B @ force
        error = float(np.linalg.norm(measured - predicted) / max(np.linalg.norm(measured), 1e-9))
        max_error = max(max_error, error)
    print(f"\n[T0-3] 线性化最大相对误差 = {max_error:.3e}")
    assert max_error < 0.01

    data = robot.data
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    try:
        data.qpos[:] = builder.pose
        data.qvel[:] = 0.0
        data.qvel[4] = 1.0  # body-frame pitch rate
        robot.forward()
        rows = dynamics.kinematics(data)[0]
        rates = rows @ data.qvel
        print(f"[T0-3] 纯 pitch_rate 输入下 rows@qvel = {np.round(rates, 6)}（期望 [0, 0, 1, 0]）")
        assert abs(rates[2] - 1.0) < 1e-6
        assert abs(rates[0]) < 1e-6 and abs(rates[1]) < 1e-6 and abs(rates[3]) < 1e-6
    finally:
        data.qpos[:], data.qvel[:] = qpos, qvel
        robot.forward()


def test_step4_open_loop_b_column(rig):
    """B 列核验：0.1 N*m 前进轮力矩下的 v_dot / theta_ddot（跳过瞬态的平均斜率）。"""
    _, _, builder, model = rig
    force_each = _OPEN_LOOP_TORQUE / builder.radius
    times, _, rate, velocity = _run_open_loop(rig, wheel_torque=_OPEN_LOOP_TORQUE)
    theta_ddot, v_dot = _slopes(times, None, rate, velocity)
    v_pred = float(model.B[1, 0] * force_each + model.B[1, 1] * force_each)
    theta_pred = float(model.B[3, 0] * force_each + model.B[3, 1] * force_each)
    v_error = abs(v_dot - v_pred) / max(abs(v_pred), 1e-12)
    theta_error = abs(theta_ddot - theta_pred) / max(abs(theta_pred), 1e-12)
    print(f"\n[T0-4] 每轮 {_OPEN_LOOP_TORQUE} N*m -> 地面力 {force_each:.4f} N")
    print(f"[T0-4] v_dot 实测={v_dot:+.6f} 预测={v_pred:+.6f} 相对误差={v_error:.2%}")
    print(f"[T0-4] theta_ddot 实测={theta_ddot:+.6f} 预测={theta_pred:+.6f} 相对误差={theta_error:.2%}")
    assert v_error < 0.10
    assert theta_error < 0.10


def test_step5_open_loop_a_column(rig):
    """A 列核验：初始倾角（绕接触点）下的 v_dot / theta_ddot。

    theta_ddot 判据 25%；v_dot 量级很弱且被接触噪声覆盖，只检查同号。
    """
    _, _, _, model = rig
    theta0 = 0.05
    times, _, rate, velocity = _run_open_loop(rig, initial_pitch=theta0, wheel_torque=0.0)
    theta_ddot, v_dot = _slopes(times, None, rate, velocity)
    theta_pred = float(model.A[3, 2] * theta0)
    v_pred = float(model.A[1, 2] * theta0)
    theta_error = abs(theta_ddot - theta_pred) / max(abs(theta_pred), 1e-12)
    print(f"\n[T0-5] theta0={theta0} rad")
    print(f"[T0-5] theta_ddot 实测={theta_ddot:+.6f} 预测={theta_pred:+.6f} 相对误差={theta_error:.2%}")
    print(f"[T0-5] v_dot 实测={v_dot:+.6f} 预测={v_pred:+.6f}（弱信号，仅检查同号）")
    assert theta_error < 0.25
    assert np.sign(v_dot) == np.sign(v_pred) or abs(v_dot) < 0.02
