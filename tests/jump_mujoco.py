"""跳跃 MuJoCo 接入测试（第 17 次排查）：零速度与有速度两种输入。

用法（项目根）：.venv/Scripts/python.exe tests/jump_mujoco.py [--speed 0.5]
配置：tuned-fast-accel 参数 + jump_enabled。t = 2 s 起跳；有速度时先以指令速度行驶，2 s 时起跳，继续保持指令速度。
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation, _build_from_args  # noqa: E402


def run(speed: float, jump_time: float = 2.0, duration: float = 6.0, preset: str = "tuned-fast-accel"):
    config, settings, _ = _build_from_args(["--preset", preset])
    config = replace(config, jump_enabled=True)
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
    import APP.mujoco_mpc_wbc as app
    from APP.mujoco_mpc_wbc import _PRESETS
    if "hip_tracking" in _PRESETS[preset]:
        app._TASK_QP_WEIGHTS["hip_tracking"] = float(_PRESETS[preset]["hip_tracking"])
    command = MotionCommand(velocity=speed, yaw_rate=0.0)
    summary = sim.run(command, duration, None, None, allow_driving=True, jump_times=[jump_time])
    t = np.array([r["time"] for r in sim.records])
    z = np.array([r["qpos"][2] for r in sim.records])
    st = np.array([r["state"] for r in sim.records])
    # 轮底离地高度：两轮碰撞体中心 z − R 的最小值（记录里没有，按机身高度与髋角近似：直接用 qpos 重算）
    import mujoco
    m = sim.robot.model
    d = mujoco.MjData(m)
    wheels = [m.geom("right_wheel_collision").id, m.geom("left_wheel_collision").id]
    clearance = []
    for r in sim.records:
        d.qpos[:] = r["qpos"]
        mujoco.mj_kinematics(m, d)
        clearance.append(min(d.geom_xpos[w][2] for w in wheels) - 0.16)
    clearance = np.array(clearance)
    after = t >= jump_time
    events = {e["phase"]: e for e in sim.jump_log}
    tail = t >= t[-1] - 1.0
    return dict(summary=summary, log=sim.jump_log, clearance=float(clearance[after].max()),
                z_rise=float(z[after].max() - z[~after][-1]) if (~after).any() else float("nan"),
                pitch_max=float(np.degrees(np.abs(st[after, 2]).max())),
                tail_pitch=float(np.degrees(np.abs(st[tail, 2]).max())),
                tail_speed=float(np.mean(st[tail, 1])), final_phase=sim.jump.phase.value,
                stop=summary["stop_reason"])


if __name__ == "__main__":
    speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 0.0
    r = run(speed)
    print("阶段：", " -> ".join(f"{e['phase']}@{e['time']:.3f}" for e in r["log"]))
    for e in r["log"]:
        print(f"  {e['phase']:8s} t={e['time']:.3f} θ={np.degrees(e['pitch']):+6.1f}° q={e['hip']:+.2f} vz={e['vz']:+.2f} vx={e['vx']:+.2f} z={e['z']:.3f} 接触={e['contact']}")
    print(f"轮底离地峰值={r['clearance']*1000:.0f} mm  机身上升={r['z_rise']*1000:.0f} mm  跳后|θ|max={r['pitch_max']:.1f}°  "
          f"末 1 s |θ|max={r['tail_pitch']:.1f}° 末速度={r['tail_speed']:+.2f}  跳跃状态={r['final_phase']}  停止={r['stop']}")
