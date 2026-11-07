"""第 28 次排查：加速未完成就松键（中途转刹车）复现。只输出 ASCII。

用法：.venv/Scripts/python.exe tests/tuning/reverse_mid_accel.py 配置名[,配置名] [松键时刻,...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation, SimulationConfig  # noqa: E402
from b_accel_sweep import settings_for  # noqa: E402


def run(name, t_release, v_cmd=2.5, start=1.0, duration=5.0, trace=False):
    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(), mpc_settings=settings_for(name))
    log = []
    orig = sim.mpc._update_hip_limit

    def spy(state):
        orig(state)
        log.append((sim.robot.time, sim.mpc.planner.mode.value, sim.mpc.planner.commanded_motion,
                    sim.mpc.active_hip_limit, sim.mpc.planner.acceleration, sim.mpc.last_theta_ff))
    sim.mpc._update_hip_limit = spy
    cmd = lambda t: MotionCommand(velocity=v_cmd if start <= t < start + t_release else 0.0, yaw_rate=0.0)
    summ = sim.run(cmd, duration, None, None, allow_driving=True)
    t = np.array([r["time"] for r in sim.records]); st = np.array([r["state"] for r in sim.records])
    c = np.array([r["contact"] for r in sim.records])
    hip = np.array([sim._hip_common(np.array(r["qpos"])[sim.hip_q]) for r in sim.records])
    m = t >= start
    i_rel = int(np.argmin(np.abs(t - (start + t_release))))
    res = dict(v_rel=st[i_rel, 1], th_max=np.degrees(abs(st[m, 2]).max()), hip_max=abs(hip[m]).max(),
               air=int((~c[m].all(axis=1)).sum()), stop=summ["stop_reason"],
               fb="%d/%d" % (summ["mpc_fallbacks"], summ["wbc_fallbacks"]))
    if trace:
        for tt, mode, cmdf, lim, a, thff in log:
            if start + t_release - 0.1 <= tt <= start + t_release + 1.2:
                i = int(np.argmin(np.abs(t - tt)))
                print("  t=%.2f mode=%-5s cmd=%d lim=%.2f a_ref=%+.2f thff=%+5.1f | v=%+.2f th=%+5.1f thd=%+6.0f q=%+.2f contact=%s tq=%s" % (
                    tt, mode, cmdf, lim, a, np.degrees(thff), st[i, 1], np.degrees(st[i, 2]), np.degrees(st[i, 3]),
                    hip[i], "".join("1" if x else "0" for x in c[i]), np.round(sim.records[i]["torque"], 1)))
    return res


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["base", "agile", "S1ab", "B-d1.0", "B-d1.2"]
    rels = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.3, 0.5, 0.7, 0.9]
    trace = "--trace" in sys.argv
    for n in names:
        for tr in rels:
            r = run(n, tr, trace=trace)
            print("rev %-7s release=%.1fs v_rel=%.2f |th|max=%5.1f |q|max=%.2f airborne=%3d fb=%s stop=%s" % (
                n, tr, r["v_rel"], r["th_max"], r["hip_max"], r["air"], r["fb"], r["stop"]), flush=True)
