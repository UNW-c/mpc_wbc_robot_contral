"""第 21 次排查 D2：推力段出口姿态（预倾角）对着地残余速度 / 高度 / 收敛的影响。只输出 ASCII。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from landing_settle import measure  # noqa: E402

import numpy as np  # noqa: E402
from dataclasses import replace  # noqa: E402
import mujoco  # noqa: E402
from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation  # noqa: E402


def run(speed, **jp):
    sim = MuJoCoMpcWbcSimulation()
    if jp:
        sim.jump.p = replace(sim.jump.p, **jp)
    summ = sim.run(MotionCommand(velocity=speed, yaw_rate=0.0), 6.0, None, None, allow_driving=True, jump_times=[2.0])
    ev = {e["phase"]: e for e in sim.jump_log}
    m = sim.robot.model
    d = mujoco.MjData(m)
    wheels = [m.geom("right_wheel_collision").id, m.geom("left_wheel_collision").id]
    clr = 0.0
    for r in sim.records:
        if r["time"] >= 2.0:
            d.qpos[:] = r["qpos"]
            mujoco.mj_kinematics(m, d)
            clr = max(clr, min(d.geom_xpos[w][2] for w in wheels) - 0.16)
    return sim, summ, ev, clr


if __name__ == "__main__":
    speeds = [float(v) for v in sys.argv[2:]] or [0.0]
    pitches = [float(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [-30.0]
    for deg in pitches:
        for v in speeds:
            jp = dict(preload_pitch=np.radians(deg))
            sim, summ, ev, clr = run(v, **jp)
            r = measure(v, **jp)
            print("pre=%+.0f v=%.1f liftoff_vx=%s td_vx=%s clr=%.0f mm settle=%s [v=%s th=%s hip=%s] td_th=%s stop=%s phases=%s" % (
                deg, v, "%+.3f" % ev["flight"]["vx"] if "flight" in ev else "-",
                "%+.3f" % ev["landing"]["vx"] if "landing" in ev else "-", clr * 1000,
                *("%.2f" % r[k] if k in r else "-" for k in ("settle", "settle_v", "settle_theta", "settle_hip")),
                "%+.1f" % r["touchdown_pitch_deg"] if "touchdown_pitch_deg" in r else "-",
                summ["stop_reason"], r["phases"]), flush=True)
