"""C28 方案 B：MuJoCo 开环推力搜索（第 17 次排查）。

从站姿（q = 0）出发，机身预倾 θ0、静止触地；推力段髋力矩剖面为：
  q < q_brake：每髋 +τ_max（共模伸展）；q ≥ q_brake：每髋 −τ_max（制动曲柄，把角动量还给机身）。
轮子力矩：wheel_mode = "zero" 或 "pd"（俯仰 PD：τ_w = kp·θ + kd·θ̇，每轮，限幅 ±5；正值使机身后仰）。
离地判据：两轮法向力都为 0 持续 2 ms。离地后停止推力（髋 PD 保持当前角，轮子 0），继续仿真 40 ms 读取离地状态。
输出：离地 v_z、θ、θ̇、q；腾空高度 h = v_z² / 2g（质心）；按离地时的轮底离地高度加上质心抛物线估计的轮底峰值。
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from APP.mujoco_robot import MuJoCoRobot  # noqa: E402
from APP.mujoco_mpc_wbc import MuJoCoDynamics  # noqa: E402

G = 9.81


def trial(theta0_deg: float, q_brake: float, wheel_mode: str = "zero", tau: float = 10.0,
          kp: float = 20.0, kd: float = 2.0) -> dict:
    robot = MuJoCoRobot.load("plane")
    model, data = robot.model, robot.data
    dyn = MuJoCoDynamics(robot)
    hq = [int(model.joint(n).qposadr[0]) for n in ("joint1", "joint3")]
    hv = [int(model.joint(n).dofadr[0]) for n in ("joint1", "joint3")]
    wheels = [model.geom("right_wheel_collision").id, model.geom("left_wheel_collision").id]
    robot.set_initial_pitch(np.radians(theta0_deg))
    robot._place_wheels_on_ground()
    mujoco.mj_forward(model, data)
    mass = float(np.sum(model.body_mass))
    lift_steps, lift = 0, None
    t0 = data.time
    for k in range(400):
        q = 0.5 * (data.qpos[hq[0]] - data.qpos[hq[1]])
        qd = 0.5 * (data.qvel[hv[0]] - data.qvel[hv[1]])
        att = dyn.attitude(data)
        rows, _ = dyn.kinematics(data)
        theta, theta_d = att[1], float((rows @ data.qvel)[2])
        if lift is None:
            h = tau if q < q_brake else -tau
            w = 0.0 if wheel_mode == "zero" else float(np.clip(kp * theta + kd * theta_d, -5, 5))
        else:
            h = float(np.clip(-3.0 * qd, -tau, tau))
            w = 0.0
        data.ctrl[:] = [h, -h, w, -w]
        mujoco.mj_step(model, data)
        contact, forces, _, _ = dyn.sensed_contact()
        if lift is None:
            lift_steps = lift_steps + 1 if not any(contact) else 0
            if lift_steps >= 2:
                mujoco.mj_comPos(model, data)
                clearance = min(data.geom_xpos[w_][2] for w_ in wheels) - 0.16
                vz_com = float(data.qvel[2])
                lift = dict(t=(data.time - t0) * 1000, vz=vz_com, theta=np.degrees(theta),
                            theta_d=np.degrees(theta_d), q=q, qd=qd, clearance=clearance)
        if lift is None and data.time - t0 > 0.25:
            break
    if lift is None:
        return dict(theta0=theta0_deg, q_brake=q_brake, wheel=wheel_mode, lifted=False)
    rise = max(lift["vz"], 0.0) ** 2 / (2 * G)
    return dict(theta0=theta0_deg, q_brake=q_brake, wheel=wheel_mode, lifted=True,
                rise=rise, wheel_peak=lift["clearance"] + rise, **lift)


if __name__ == "__main__":
    print("| θ0 ° | q_brake | 轮子 | 离地 ms | v_z | 离地 θ ° | 离地 dθ °/s | 离地 q | 离地 dq | 质心腾空 mm | 轮底峰值估计 mm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for theta0, qb, wm in itertools.product((-20, -10, 0, 10, 20), (1.2, 1.6, 2.0, 9.9), ("zero", "pd")):
        r = trial(theta0, qb, wm)
        if not r["lifted"]:
            print(f"| {theta0} | {qb} | {wm} | 未离地 | | | | | | | |", flush=True)
            continue
        print(f"| {theta0} | {qb if qb < 9 else '无'} | {wm} | {r['t']:.0f} | {r['vz']:.2f} | {r['theta']:+.1f} | {r['theta_d']:+.0f} "
              f"| {r['q']:.2f} | {r['qd']:+.0f} | {r['rise']*1000:.0f} | {r['wheel_peak']*1000:.0f} |", flush=True)
