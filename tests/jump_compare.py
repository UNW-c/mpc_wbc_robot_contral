"""C27 选项 A：J-M 与 MuJoCo 开环对拍（第 17 次排查，只做诊断）。

同一初始状态：站姿 q = 0，机身后仰 θ0 = −20°，静止，轮子刚好触地；
同一输入：t = 0 起每髋 +10 N·m 共模满力矩，轮子力矩 0，持续到离地后 20 ms。
MuJoCo 另做两组，用来区分差异来源：
- M1：原模型；
- M2：轮地接触改硬（solref = 0.002 1，只在本诊断中修改，不改 XML）；
- M3：原接触，但把两髋的 y 位置对称化（joint1 / joint3 都取 ±0.12），用于判断不对称的影响。
输出每 10 ms 的 v_z、θ、θ̇、q、单轮法向力，以及横滚 / 偏航。
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from APP.mujoco_robot import MuJoCoRobot  # noqa: E402
from APP.mujoco_mpc_wbc import MuJoCoDynamics  # noqa: E402
from tests.jump_model import PlanarJumpModel  # noqa: E402

THETA0 = np.radians(-20.0)
TAU = 10.0
DURATION = 0.12


def run_jm():
    m = PlanarJumpModel()
    s = m.standing_state(0.0, THETA0)
    rows, t, lift = [], 0.0, None
    while t < DURATION:
        s, n, _ = m.step(s, [2 * TAU, 0.0])
        t = s.t
        if lift is None and n <= 0.0 and m.wheel_bottom(s) > 0.0005:
            lift = t
        if abs(t / 0.01 - round(t / 0.01)) < 1e-6:
            rows.append((t, s.gd[1], np.degrees(s.g[2]), np.degrees(s.gd[2]), s.g[3], 0.5 * n, 0.0, 0.0))
        if lift is not None and t > lift + 0.02:
            break
    return rows, lift


def run_mujoco(variant: str):
    robot = MuJoCoRobot.load("plane")
    model, data = robot.model, robot.data
    if variant == "hard":
        for name in ("right_wheel_collision", "left_wheel_collision", "floor"):
            gid = model.geom(name).id
            model.geom_solref[gid] = [0.002, 1.0]
    if variant == "symmetric":
        model.body_pos[model.body("joint1_Link").id][1] = -0.12
        model.body_pos[model.body("joint3_Link").id][1] = +0.12
    dyn = MuJoCoDynamics(robot)
    hip_q = [int(model.joint(n).qposadr[0]) for n in ("joint1", "joint3")]
    hip_v = [int(model.joint(n).dofadr[0]) for n in ("joint1", "joint3")]
    # 初始：q = 0，机身后仰 θ0，放到地面
    data.qpos[hip_q] = 0.0
    data.qvel[:] = 0.0
    robot.set_initial_pitch(THETA0)
    robot._place_wheels_on_ground()
    mujoco.mj_forward(model, data)
    rows, lift = [], None
    t0 = data.time
    steps = int(DURATION / model.opt.timestep)
    for k in range(steps):
        data.ctrl[:] = [TAU, -TAU, 0.0, 0.0]          # j1 = +τ, j3 = sign·τ（sign = −1）
        mujoco.mj_step(model, data)
        t = data.time - t0
        contact, forces, _, _ = dyn.sensed_contact()
        normals = forces[:, 2]
        if lift is None and not any(contact):
            lift = t
        if (k + 1) % 10 == 0:
            rows_, _ = dyn.kinematics(data)
            rates = rows_ @ data.qvel
            att = dyn.attitude(data)
            q = 0.5 * (data.qpos[hip_q[0]] - data.qpos[hip_q[1]])
            rows.append((t, data.qvel[2], np.degrees(att[1]), np.degrees(rates[2]), q,
                         float(np.mean(normals)), np.degrees(att[0]), np.degrees(att[2])))
        if lift is not None and t > lift + 0.02:
            break
    return rows, lift


def show(name, rows, lift):
    print(f"\n[{name}] 离地 t = {lift * 1000:.0f} ms" if lift else f"\n[{name}] 未离地")
    print("| t ms | v_z | θ ° | dθ °/s | q | 单轮 N | 横滚 ° | 偏航 ° |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]*1000:.0f} | {r[1]:+.2f} | {r[2]:+.1f} | {r[3]:+.0f} | {r[4]:.2f} | {r[5]:.0f} | {r[6]:+.1f} | {r[7]:+.1f} |")


if __name__ == "__main__":
    show("J-M", *run_jm())
    show("MuJoCo 原模型", *run_mujoco("base"))
    show("MuJoCo 硬接触 solref 0.002", *run_mujoco("hard"))
    show("MuJoCo 两髋 y 对称", *run_mujoco("symmetric"))
