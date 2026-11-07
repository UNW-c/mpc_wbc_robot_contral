"""第 21 次排查 D1：飞行段轮子姿态 PD 是否饱和 + 扫参。只输出 ASCII。"""
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from landing_settle import measure  # noqa: E402


def flight_profile(speed=0.0, **jp):
    sim = MuJoCoMpcWbcSimulation()
    if jp:
        sim.jump.p = replace(sim.jump.p, **jp)
    sim.run(MotionCommand(velocity=speed, yaw_rate=0.0), 3.5, None, None, allow_driving=True, jump_times=[2.0])
    ev = {e["phase"]: e["time"] for e in sim.jump_log}
    m = sim.robot.model
    d = mujoco.MjData(m)
    wheels = [m.geom("right_wheel_collision").id, m.geom("left_wheel_collision").id]
    rows = []
    for r in sim.records:
        if ev.get("thrust", 9) - 0.01 <= r["time"] <= ev.get("landing", 0) + 0.01:
            d.qpos[:] = r["qpos"]
            mujoco.mj_kinematics(m, d)
            clr = min(d.geom_xpos[w][2] for w in wheels) - 0.16
            rows.append((r["time"], np.degrees(r["state"][2]), np.degrees(r["state"][3]), r["torque"][2], clr))
    return ev, np.array(rows)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "profile"
    if mode == "profile":
        ev, rows = flight_profile()
        print("thrust=%.3f flight=%.3f landing=%.3f" % (ev["thrust"], ev["flight"], ev["landing"]))
        fl = rows[(rows[:, 0] >= ev["flight"]) & (rows[:, 0] <= ev["landing"])]
        print("flight samples=%d wheel_sat_frac=%.2f clearance_max=%.0f mm" % (
            len(fl), np.mean(np.abs(fl[:, 3]) >= 4.99), fl[:, 4].max() * 1000))
        for t, th, thd, tw, clr in rows[::6]:
            print("t=%.3f th=%+6.1f thd=%+7.0f wheel=%+5.2f clr=%4.0f" % (t, th, thd, tw, clr * 1000))
    else:
        for kp, kd, sc in ((20, 6, 3), (10, 3, 3), (40, 12, 3), (80, 20, 3), (20, 6, 1), (20, 12, 5)):
            r = measure(0.0, flight_pitch_kp=kp, flight_pitch_kd=kd, flight_align_kd_scale=sc)
            print("kp=%3d kd=%3d align=%d settle=%s td_pitch=%s pitch_max=%s stop=%s" % (
                kp, kd, sc, "%.2f" % r.get("settle", -1), "%+.1f" % r.get("touchdown_pitch_deg", 0),
                "%.1f" % r.get("pitch_max_deg", 0), r["stop"]), flush=True)
