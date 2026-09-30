"""落地收敛指标（第 20 次排查）：默认配置跳一次，统计着地到收敛的时间。

判据：着地后 |v - v_cmd| <= 0.03 m/s、|theta| <= 0.5 deg、|q| <= 0.05 rad 持续满足。
用法：.venv/Scripts/python.exe tests/tuning/landing_settle.py [速度 ...]
输出只用 ASCII，避免 Windows GBK 终端编码错误。
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation  # noqa: E402


def measure(speed: float = 0.0, jump_time: float = 2.0, duration: float = 6.0, **jump_params) -> dict:
    sim = MuJoCoMpcWbcSimulation()
    if jump_params:
        sim.jump.p = replace(sim.jump.p, **jump_params)
    summary = sim.run(MotionCommand(velocity=speed, yaw_rate=0.0), duration, None, None,
                      allow_driving=True, jump_times=[jump_time])
    events = {e["phase"]: e for e in sim.jump_log}
    result = dict(speed=speed, stop=summary["stop_reason"],
                  phases="->".join(e["phase"][:3] for e in sim.jump_log))
    if "landing" not in events:
        return result
    touchdown = events["landing"]["time"]
    t = np.array([r["time"] for r in sim.records])
    state = np.array([r["state"] for r in sim.records])
    qpos = np.array([r["qpos"] for r in sim.records])
    hip = 0.5 * (qpos[:, 7] - qpos[:, 9])
    after = t >= touchdown
    bad = (np.abs(state[:, 1] - speed) > 0.03) | (np.abs(state[:, 2]) > np.radians(0.5)) | (np.abs(hip) > 0.05)
    idx = np.where(bad & after)[0]

    def last(mask):
        i = np.where(mask & after)[0]
        return float(t[i[-1]] - touchdown) if len(i) else 0.0

    td = int(np.argmin(np.abs(t - touchdown)))
    result.update(
        # 分项：最后一次超限的时刻（相对着地），用于判断哪一项拖慢收敛
        settle_v=last(np.abs(state[:, 1] - speed) > 0.03),
        settle_theta=last(np.abs(state[:, 2]) > np.radians(0.5)),
        settle_hip=last(np.abs(hip) > 0.05),
        touchdown_pitch_rate_deg=float(np.degrees(state[td, 3])),
        landing_phase=float(events["recover"]["time"] - touchdown) if "recover" in events else float("nan"),
        settle=float(t[idx[-1]] - touchdown) if len(idx) else 0.0,
        touchdown_pitch_deg=float(np.degrees(state[np.argmin(np.abs(t - touchdown)), 2])),
        pitch_max_deg=float(np.degrees(np.abs(state[after, 2]).max())),
        speed_err_max=float(np.abs(state[after, 1] - speed).max()),
        hip_max=float(np.abs(hip[after]).max()),
    )
    return result


if __name__ == "__main__":
    speeds = [float(v) for v in sys.argv[1:]] or [0.0, 1.5]
    for v in speeds:
        r = measure(v)
        print("v=%.1f settle=%s s touchdown_pitch=%s deg pitch_max=%s deg speed_err=%s hip_max=%s stop=%s phases=%s" % (
            v, *("%.2f" % r[k] if k in r else "-" for k in
                 ("settle", "touchdown_pitch_deg", "pitch_max_deg", "speed_err_max", "hip_max")),
            r["stop"], r["phases"]), flush=True)
