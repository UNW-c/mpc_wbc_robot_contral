"""M4：闭环有效范围扫描（hip_angle_limit 的有效上限，q_ctrl）。

前提前置（M1~M3）：q_int = 无穷（无干涉）、q_clear ≈ 1.75 rad、q_auth = 0.881 rad
→ q_geo = 0.881。本脚本把 hip_guard 临时设为 q_geo + 0.1 = 0.98，
扫描 hip_angle_limit ∈ {0.55, 0.65, 0.75, 0.85}，工况 H2、H4、连推、V3。

判据（文档）：相邻两档之间，H4 回位时间改善 < 5%，或过零 > 3，或出现 fallback
→ 该档为"性能不再提升 / 开始失稳"的边界 q_ctrl。

运行：
    .venv/Scripts/python.exe -m tests.tuning.hip_limit_scan
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MotionCommand, MPCSettings
from APP.mujoco_mpc_wbc import Disturbance, MuJoCoMpcWbcSimulation, SimulationConfig
from tests.acceptance_compare import _SCENARIOS, _metrics, _v3_command

_Q_GEO = 0.881
_LIMITS = (0.55, 0.65, 0.75, 0.85)
_PUSH = 5.0
_DOUBLE_PUSH = [Disturbance(time=_PUSH, delta_velocity=(0.5, 0.0, 0.0)),
                Disturbance(time=_PUSH + 0.8, delta_velocity=(0.5, 0.0, 0.0))]


def _hip_settings(limit: float) -> MPCSettings:
    base = MPCSettings(compensate_hip_torque=False)
    weights = replace(base.weights, hip_angle=50.0)
    return replace(base, weights=weights, hip_angle_limit=float(limit))


def main() -> int:
    print(f"M4 闭环扫描（hip_guard 临时 = {_Q_GEO + 0.1:.2f}；q_geo = {_Q_GEO}）")
    print(f"{'limit':>7}{'工况':>7}{'位移峰':>9}{'回位':>7}{'俯仰峰°':>9}{'过零':>5}"
          f"{'髋角峰':>8}{'髋速峰':>8}{'髋占用':>8}{'fb':>6}{'停止':>14}")
    results = {}
    for limit in _LIMITS:
        settings = _hip_settings(limit)
        config = SimulationConfig(model_variant="hip_input", forward_mode="mpc",
                                  preview_lead=0.0, hip_guard=_Q_GEO + 0.1)
        for scenario in ("H2", "H4", "DOUBLE", "V3"):
            if scenario == "DOUBLE":
                spec = dict(duration=15.0, disturbances=_DOUBLE_PUSH)
            else:
                spec = _SCENARIOS[scenario]
            sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
            command = _v3_command if scenario == "V3" else MotionCommand(velocity=0.0, yaw_rate=0.0)
            summary = sim.run(command, spec["duration"], None, spec["disturbances"],
                              allow_driving=(scenario == "V3"))
            metric = _metrics(sim, summary, 5.0,
                              command_fn=_v3_command if scenario == "V3" else None)
            results[(limit, scenario)] = metric
            print(f"{limit:>7.2f}{scenario:>7}{metric['peak_position']:>9.4f}"
                  f"{metric['settle']:>7.2f}{metric['peak_pitch']:>9.2f}"
                  f"{metric['pitch_cross']:>5d}{metric['hip_angle_peak']:>8.3f}"
                  f"{metric['hip_speed_peak']:>8.2f}{metric['hip_occ']:>8.2f}"
                  f"{metric['mpc_fb']:>3d}/{metric['wbc_fb']:<2d}{str(metric['stop']):>14}")

    print("\n判据检查（相邻档位）：")
    for index in range(1, len(_LIMITS)):
        previous, current = _LIMITS[index - 1], _LIMITS[index]
        h4_prev = results[(previous, "H4")]
        h4_cur = results[(current, "H4")]
        settle_prev = h4_prev["settle"]
        settle_cur = h4_cur["settle"]
        if np.isfinite(settle_prev) and np.isfinite(settle_cur) and settle_prev > 0:
            improvement = (settle_prev - settle_cur) / settle_prev * 100.0
        else:
            improvement = float("nan")
        crossings = max(results[(current, s)]["pitch_cross"] for s in ("H2", "H4", "DOUBLE", "V3"))
        fallbacks = max(results[(current, s)]["mpc_fb"] + results[(current, s)]["wbc_fb"]
                        for s in ("H2", "H4", "DOUBLE", "V3"))
        flag = " <-- 边界" if (improvement != improvement or improvement < 5.0
                               or crossings > 3 or fallbacks > 0) else ""
        print(f"  {previous:.2f} -> {current:.2f}: H4 回位改善 {improvement:+.1f}%，"
              f"最大过零 {crossings}，fallback {fallbacks}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
