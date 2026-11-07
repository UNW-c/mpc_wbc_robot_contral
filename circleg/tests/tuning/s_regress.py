"""第 26 次排查：S1/S2 候选的回归（零输入 H1-H4/Tf、刹车中被推 ±0.5、跳跃 0/1.5 m/s）。只输出 ASCII。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation, SimulationConfig, Disturbance  # noqa: E402
from tests.tuning import a1_acceptance as a1  # noqa: E402
from b_accel_sweep import settings_for  # noqa: E402

for name in sys.argv[1:]:
    s = settings_for(name)
    zero = []
    for case in ("H1", "H2", "H3", "H4", "Tf"):
        z = a1._run_zero_input(s, SimulationConfig(), case)
        zero.append("%s=%.2f/%d/%.1f/%s" % (case, z["settle"], z["crossings"], z["pitch_peak"], z.get("reason")))
    print("reg %s zero %s" % (name, " ".join(zero)), flush=True)
    for dv in (0.5, -0.5):
        sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(), mpc_settings=s)
        summ = sim.run(lambda t: MotionCommand(velocity=2.5 if t < 4.0 else 0.0, yaw_rate=0.0), 8.0, None,
                       [Disturbance(time=4.5, delta_velocity=(dv, 0, 0))], allow_driving=True)
        th = np.array([r["state"][2] for r in sim.records])
        c = np.array([r["contact"] for r in sim.records])
        print("reg %s push=%+.1f stop=%s |th|max=%.1f airborne=%d" % (
            name, dv, summ["stop_reason"], np.degrees(abs(th).max()), int((~c.all(axis=1)).sum())), flush=True)
    for v in (0.0, 1.5):
        sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(), mpc_settings=s)
        summ = sim.run(MotionCommand(velocity=v, yaw_rate=0.0), 5.0, None, None, allow_driving=True, jump_times=[2.0])
        print("reg %s jump v=%.1f stop=%s phases=%s" % (
            name, v, summ["stop_reason"], "->".join(e["phase"][:3] for e in sim.jump_log)), flush=True)
