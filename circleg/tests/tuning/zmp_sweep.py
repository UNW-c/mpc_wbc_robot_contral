"""第 31 次排查 Z1：ZMP / 质心偏移参考的联合扫描（分配比例 × 提前量）。

对每个配置跑 V4（0 → 2.5 → 0 → −2.5 → 0，每段 4 s），统计：
起步 / 刹车时间、刹停位移、停住后回弹量、俯仰峰、曲柄峰、MPC / WBC 降级、是否停机。
用法：.venv/Scripts/python.exe tests/tuning/zmp_sweep.py [--quick]
输出只用 ASCII。
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import (MuJoCoMpcWbcSimulation, SimulationConfig,  # noqa: E402
                                _build_from_args, default_app_mpc_settings)

# (标签, zmp_split, zmp_lead)
GRID = [("base(旧前馈)", None, None)]
GRID += [(f"tilt{int((1 - s) * 100)}/crank{int(s * 100)} lead{l}", s, l)
         for s in (0.0, 0.5, 1.0) for l in (0.0, 0.1, 0.2)]


def settings_for(split, lead):
    base = default_app_mpc_settings()
    if split is None:
        return base
    return replace(base, zmp_reference=True, zmp_split=float(split), zmp_lead=float(lead))


def run(split, lead, max_speed=2.5, duration=16.0) -> dict:
    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(), mpc_settings=settings_for(split, lead))
    command = lambda t: MotionCommand(  # noqa: E731
        velocity=max_speed if t < 4.0 else 0.0 if t < 8.0 else -max_speed if t < 12.0 else 0.0,
        yaw_rate=0.0)
    summary = sim.run(command, duration, None, None, allow_driving=True)
    t = np.array([r["time"] for r in sim.records])
    st = np.array([r["state"] for r in sim.records])
    qpos = np.array([r["qpos"] for r in sim.records])
    crank = 0.5 * (qpos[:, 7] - qpos[:, 9])
    v, x, th = st[:, 1], st[:, 0], st[:, 2]
    rise = np.where((t >= 0.0) & (v >= 0.9 * max_speed))[0]
    t_rise = float(t[rise[0]]) if rise.size else float("nan")
    stop = np.where((t >= 4.0) & (t <= 8.0) & (np.abs(v) < 0.05))[0]
    t_stop = float(t[stop[0]] - 4.0) if stop.size else float("nan")
    if stop.size:
        i0 = int(np.argmin(np.abs(t - 4.0)))
        i_s = int(stop[0])
        distance = float(abs(x[i_s] - x[i0]))
        after = slice(i_s, min(i_s + 750, len(t)))          # 停住后 3 s
        rebound = float(x[i_s] - x[after].min())            # 向后回弹
        forward = float(x[after].max() - x[i_s])
    else:
        distance = rebound = forward = float("nan")
    brake = (t >= 4.0) & (t <= 4.0 + (t_stop if np.isfinite(t_stop) else 4.0))
    return dict(t_rise=t_rise, t_stop=t_stop, distance=distance, rebound=rebound, forward=forward,
                pitch_up=float(np.degrees(th[(t >= 0.3) & (t <= 3.9)]).max()),
                pitch_dn=float(-np.degrees(th[brake]).min()) if brake.any() else float("nan"),
                pitch_max=float(np.degrees(np.abs(th).max())),
                crank_max=float(np.abs(crank).max()),
                crank_brake=float(np.abs(crank[brake]).max()) if brake.any() else float("nan"),
                mpc_fb=summary["mpc_fallbacks"], wbc_fb=summary["wbc_fallbacks"],
                stop=summary["stop_reason"])


def main() -> int:
    grid = GRID[:4] if "--quick" in sys.argv else GRID
    print("| 配置 | t_rise | t_stop | 刹停位移 | 回弹 | 前冲 | 起步俯仰 | 刹车俯仰 | |俯仰|max | 曲柄峰 | 刹车曲柄 | fb | 停机 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, split, lead in grid:
        r = run(split, lead)
        print("| %s | %.2f | %.2f | %.3f | %.3f | %.3f | %.1f° | %.1f° | %.1f° | %.2f | %.2f | %d/%d | %s |" % (
            label, r["t_rise"], r["t_stop"], r["distance"], r["rebound"], r["forward"],
            r["pitch_up"], r["pitch_dn"], r["pitch_max"], r["crank_max"], r["crank_brake"],
            r["mpc_fb"], r["wbc_fb"], r["stop"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
