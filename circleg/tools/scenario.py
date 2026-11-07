"""跑一个标准工况并输出指标，供测试记录与文档引用。

用法：
    python tools/scenario.py forward
    python tools/scenario.py jump --json results/jump.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from APP.mujoco_mpc_wbc import (MuJoCoMpcWbcSimulation, SimulationConfig,  # noqa: E402
                                default_app_mpc_settings)
from Algorithm.mpc import MotionCommand  # noqa: E402

SCENARIOS = {
    "forward": dict(duration=6.0, velocity=1.5, hold=3.0, yaw_rate=0.0, yaw_hold=0.0),
    "spin": dict(duration=6.0, velocity=0.0, hold=3.0, yaw_rate=1.5, yaw_hold=4.2),
    "jump": dict(duration=5.0, velocity=0.0, hold=3.0, yaw_rate=0.0, yaw_hold=0.0,
                 jump_time=1.0),
    "forward_jump": dict(duration=6.0, velocity=1.0, hold=4.5, yaw_rate=0.0, yaw_hold=0.0,
                         jump_time=1.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circleg 标准工况指标")
    parser.add_argument("scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--json", type=Path, default=None, help="把指标写到 JSON 文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case = SCENARIOS[args.scenario]
    config = SimulationConfig(terrain="plane", jump_enabled="jump_time" in case)
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=default_app_mpc_settings())

    def command(time_s: float) -> MotionCommand:
        return MotionCommand(
            velocity=(case["velocity"] if time_s < case["hold"] else 0.0),
            yaw_rate=(case["yaw_rate"] if time_s < case["yaw_hold"] else 0.0))

    jump_times = None if "jump_time" not in case else [float(case["jump_time"])]
    summary = sim.run(command, case["duration"], allow_driving=True, jump_times=jump_times)

    records = sim.records
    time = np.array([r["time"] for r in records])
    state = np.array([r["state"] for r in records])
    qpos = np.array([r["qpos"] for r in records])
    metrics = {
        "scenario": args.scenario,
        "completed": summary["completed"],
        "stop_reason": summary["stop_reason"],
        "duration": float(case["duration"]),
        "max_abs_pitch_rad": float(np.abs(state[:, 2]).max()),
        "max_abs_speed_m_s": float(np.abs(state[:, 1]).max()),
        "final_speed_m_s": float(state[-1, 1]),
        "distance_m": float(sim.distance),
        "final_yaw_rad": float(state[-1, 4]),
        "max_abs_yaw_rate_rad_s": float(np.abs(state[:, 5]).max()),
        "base_height_min_m": float(qpos[:, 2].min()),
        "base_height_max_m": float(qpos[:, 2].max()),
        "jump_log": sim.jump_log,
        "time_s": time.tolist(),
        "pitch_rad": state[:, 2].tolist(),
        "speed_m_s": state[:, 1].tolist(),
        "yaw_rad": state[:, 4].tolist(),
        "yaw_rate_rad_s": state[:, 5].tolist(),
        "base_height_m": qpos[:, 2].tolist(),
    }
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items()
                      if key not in ("time_s", "pitch_rad", "speed_m_s", "yaw_rad",
                                     "yaw_rate_rad_s", "base_height_m", "jump_log")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
