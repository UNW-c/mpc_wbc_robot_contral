"""ZMP / 质心偏移探针（第 31 次排查 Z0）：量出"刹车需要多少质心偏移"与"当前控制实际用了多少"。

物理关系：轮腿只有一条接地线，前后方向地面反力作用在接地点，故
    Δ ≜ x_com − x_contact（质心相对接地点的前后偏移）
    a = g · Δ / z_com          （倒立摆；a>0 前加速，需要质心在接地点之前）
本脚本逐拍（用 MuJoCo 重放 qpos）计算 Δ_act 与参考加速度对应的 Δ_cmd = z_com·a_ref/g，
并给出俯仰、共模髋角，用来看姿态摆幅是否超过了偏移需求。

用法：.venv/Scripts/python.exe tests/tuning/zmp_probe.py [速度] [--preset NAME]
输出只用 ASCII。
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import (MuJoCoMpcWbcSimulation, SimulationConfig,  # noqa: E402
                                _build_from_args, default_app_mpc_settings)

_G = 9.81


def build(preset: str | None):
    if preset is None:
        return SimulationConfig(), default_app_mpc_settings()
    config, settings, _ = _build_from_args(["--preset", preset])
    return config, settings


def probe(speed: float = 2.5, preset: str | None = "tuned-agile", t_brake: float = 4.0,
          duration: float = 8.0) -> dict:
    config, settings = build(preset)
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
    sim.run(lambda t: MotionCommand(velocity=speed if t < t_brake else 0.0, yaw_rate=0.0),
            duration, None, None, allow_driving=True)
    model, data = sim.robot.model, mujoco.MjData(sim.robot.model)
    wheels = [model.geom("right_wheel_collision").id, model.geom("left_wheel_collision").id]
    rows = []
    for r in sim.records:
        data.qpos[:] = r["qpos"]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com = data.subtree_com[0]
        contact_x = float(np.mean([data.geom_xpos[w][0] for w in wheels]))
        rows.append((r["time"], float(com[0]), float(com[2]), contact_x,
                     float(r["state"][2]), 0.5 * (r["qpos"][7] - r["qpos"][9]),
                     np.nan if r.get("reference_velocity") is None else r["reference_velocity"],
                     r.get("mode")))
    return dict(rows=rows, limits=sim.mpc.settings.limits, preset=preset, speed=speed, t_brake=t_brake)


def phase_stats(rows, lo, hi, a_ref):
    sel = [r for r in rows if lo <= r[0] <= hi]
    t = np.array([r[0] for r in sel])
    dx = np.array([r[1] - r[3] for r in sel])          # Delta_act
    z = np.array([r[2] for r in sel])
    th = np.degrees(np.array([r[4] for r in sel]))
    q = np.array([r[5] for r in sel])
    vref = np.array([r[6] for r in sel])
    a_meas = np.gradient(np.array([r[6] for r in sel]), t) if len(sel) > 2 else np.zeros(len(sel))
    delta_cmd = z * a_ref / _G
    return dict(n=len(sel), dx=float(np.mean(dx)), dx_max=float(np.max(np.abs(dx))),
                dcmd=float(np.mean(delta_cmd)), dcmd_max=float(np.max(np.abs(delta_cmd))),
                ratio=float(np.mean(np.abs(dx)) / max(np.mean(np.abs(delta_cmd)), 1e-9)),
                z=float(np.mean(z)), th_max=float(np.max(np.abs(th))), q_max=float(np.max(np.abs(q))),
                a_ref=a_ref, t_span=(float(t[0]), float(t[-1])))


def main() -> int:
    speed = 2.5
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        speed = float(args[0])
    preset = "tuned-agile"
    for i, a in enumerate(sys.argv):
        if a == "--preset" and i + 1 < len(sys.argv):
            preset = sys.argv[i + 1]
    p = probe(speed, preset)
    lim = p["limits"]
    print("ZMP preset=%s speed=%.2f 起步限值=%.2f 刹车限值=%.2f" % (
        preset or "default", speed, lim.commanded_acceleration, lim.max_braking))
    rows = p["rows"]
    up = phase_stats(rows, 0.3, p["t_brake"] - 0.1, lim.commanded_acceleration)
    dn = phase_stats(rows, p["t_brake"] + 0.05, p["t_brake"] + 1.6, -lim.max_braking)
    for tag, s in (("加速段", up), ("刹车段", dn)):
        print("ZMP %s 窗口=%s: Delta_cmd 均值 %+.4f max %.4f | Delta_act 均值 %+.4f max %.4f | "
              "实测/需求=%.2f | 俯仰峰 %.1f deg | 曲柄峰 %.2f | z_com=%.3f m" % (
                  tag, "%.2f-%.2f s" % s["t_span"], s["dcmd"], s["dcmd_max"], s["dx"], s["dx_max"],
                  s["ratio"], s["th_max"], s["q_max"], s["z"]))
    # 刹停时刻的姿态与偏移（回弹来源）
    i_stop = next((i for i, r in enumerate(rows) if r[0] >= p["t_brake"] and abs(r[6]) < 0.05), None)
    if i_stop:
        r = rows[i_stop]
        print("ZMP 停住瞬间 t=%.2f vref=%.2f: Delta_act=%+.4f 俯仰=%+.1f deg 曲柄=%+.2f" % (
            r[0], r[6], r[1] - r[3], np.degrees(r[4]), r[5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
