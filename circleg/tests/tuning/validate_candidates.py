"""第 13 次排查：候选参数的验证集与敏感性检查。

验证集（寻优未使用）：+0.3 / +0.65 m/s 冲量、25 N x 1.0 s 持续推力、H6、V1、V2、V3。
敏感性（取代原阶段 C）：tuned-best 的阶段 1 四参数逐个 x0.8 / x1.2（limit +/-0.03），共 8 组。

运行：
    .venv/Scripts/python.exe -m tests.tuning.validate_candidates
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MotionCommand, MPCSettings
from APP.mujoco_mpc_wbc import Disturbance, MuJoCoMpcWbcSimulation, SimulationConfig
from tests.acceptance_compare import _metrics, _v1_command, _v2_command, _v3_command
from tests.tuning.grid_sweep import (_PARAM_KEYS, _evaluate_one, _read_rows, _worker_init)
from tests.tuning.objective import E5_PARAMS

_OUT = Path("artifacts/tuning/grid/20260929")
_PUSH = 5.0
_VERIFY = {
    "push+0.3": dict(duration=15.0, disturbances=[Disturbance(time=_PUSH, delta_velocity=(0.3, 0, 0))]),
    "push+0.65": dict(duration=15.0, disturbances=[Disturbance(time=_PUSH, delta_velocity=(0.65, 0, 0))]),
    "push25x1.0": dict(duration=17.0, disturbances=[Disturbance(time=_PUSH, force=(25.0, 0, 0),
                                                               torque=(0.0, 2.0, 0.0), duration=1.0)]),
    "H6": dict(duration=20.0, disturbances=[], initial_pitch=0.02),
    "V1": dict(duration=14.0, disturbances=[]),
    "V2": dict(duration=8.0, disturbances=[]),
    "V3": dict(duration=14.0, disturbances=[]),
}
_COMMANDS = {"V1": _v1_command, "V2": _v2_command, "V3": _v3_command}


def _settings(params: dict) -> MPCSettings:
    base = MPCSettings(compensate_hip_torque=False)
    weights = replace(base.weights,
                      position=float(params["position"]), velocity=float(params["velocity"]),
                      pitch=float(params["pitch"]), pitch_rate=float(params["pitch_rate"]),
                      hip_angle=float(params["hip_angle"]), hip_rate=float(params["hip_rate"]),
                      terminal_mode=str(params.get("terminal_mode", "scale")),
                      terminal_scale=float(params["terminal_scale"]))
    return replace(base, weights=weights, hip_angle_limit=float(params["hip_angle_limit"]))


def _config(params: dict) -> SimulationConfig:
    omega = float(params["tracking_bandwidth"])
    return replace(SimulationConfig(model_variant="hip_input", forward_mode="mpc", preview_lead=0.0),
                   tracking_velocity_gain=2.0 * omega, tracking_position_gain=omega * omega,
                   hip_guard=1.70)


def _run_verify(task):
    label, params, name = task
    spec = _VERIFY[name]
    command_fn = _COMMANDS.get(name)
    command = command_fn if command_fn is not None else MotionCommand(velocity=0.0, yaw_rate=0.0)
    sim = MuJoCoMpcWbcSimulation(config=_config(params), mpc_settings=_settings(params),
                                 initial_pitch=float(spec.get("initial_pitch", 0.0)))
    summary = sim.run(command, spec["duration"], None, spec["disturbances"],
                      allow_driving=command_fn is not None)
    event = spec.get("event_time", _PUSH)
    metric = _metrics(sim, summary, event, command_fn=command_fn)
    return label, name, metric


def main() -> int:
    candidates = json.loads((_OUT / "candidates.json").read_text(encoding="utf-8"))
    groups = dict(candidates)
    groups["E5"] = dict(E5_PARAMS, terminal_mode="scale")

    tasks = [(label, params, name) for label, params in groups.items() for name in _VERIFY]
    print(f"验证集：{len(groups)} 组 x {len(_VERIFY)} 工况 = {len(tasks)} 次评价")
    with Pool(12, initializer=_worker_init) as pool:
        results = list(pool.imap_unordered(_run_verify, tasks, chunksize=1))

    table = {}
    for label, name, metric in results:
        table.setdefault(label, {})[name] = metric
    fields = ["group", "scenario", "peak_position", "settle", "peak_pitch", "pitch_cross",
              "hip_angle_peak", "hip_speed_peak", "hip_occ", "accel_ratio", "overshoot", "stop"]
    with (_OUT / "verify.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in groups:
            for name in _VERIFY:
                metric = table.get(label, {}).get(name)
                if metric is None:
                    continue
                writer.writerow({"group": label, "scenario": name,
                                 "peak_position": round(metric["peak_position"], 4),
                                 "settle": round(metric["settle"], 3),
                                 "peak_pitch": round(metric["peak_pitch"], 3),
                                 "pitch_cross": metric["pitch_cross"],
                                 "hip_angle_peak": round(metric["hip_angle_peak"], 4),
                                 "hip_speed_peak": round(metric["hip_speed_peak"], 2),
                                 "hip_occ": round(metric["hip_occ"], 3),
                                 "accel_ratio": round(metric["accel_ratio"], 3)
                                 if metric["accel_ratio"] == metric["accel_ratio"] else "",
                                 "overshoot": round(metric["overshoot"], 4)
                                 if metric["overshoot"] == metric["overshoot"] else "",
                                 "stop": metric["stop"] or ""})
    print("\n验证集汇总（各组最差工况）：")
    print(f"{'组':>14}{'位移峰max':>10}{'回位max':>9}{'俯仰max':>9}{'过零max':>8}"
          f"{'髋角max':>9}{'髋速max':>9}{'停止':>12}")
    for label in groups:
        metrics = list(table.get(label, {}).values())
        if not metrics:
            continue
        def worst(key):
            values = [m[key] for m in metrics if m[key] == m[key]]
            return max(values) if values else float("nan")
        stops = {m["stop"] for m in metrics if m["stop"]}
        print(f"{label:>14}{worst('peak_position'):>10.4f}{worst('settle'):>9.2f}"
              f"{worst('peak_pitch'):>9.2f}{worst('pitch_cross'):>8.0f}"
              f"{worst('hip_angle_peak'):>9.3f}{worst('hip_speed_peak'):>9.1f}"
              f"{str(stops or '-'):>12}")

    # 敏感性：tuned-best 的阶段 1 四参数 x0.8 / x1.2（limit +/-0.03）
    best = dict(groups["tuned-best"])
    sensitivity = []
    for name in ("position", "velocity", "hip_rate"):
        for factor in (0.8, 1.2):
            params = dict(best)
            params[name] = best[name] * factor
            sensitivity.append((f"{name}x{factor}", params))
    for delta in (-0.03, 0.03):
        params = dict(best)
        params["hip_angle_limit"] = min(max(best["hip_angle_limit"] + delta, 0.55), 0.80)
        sensitivity.append((f"limit{delta:+.2f}", params))
    print(f"\n敏感性：{len(sensitivity)} 组（含 tuned-best 对照）")
    with Pool(12, initializer=_worker_init) as pool:
        sensitivity_results = list(pool.imap_unordered(_evaluate_one,
                                                       [params for _, params in sensitivity],
                                                       chunksize=1))
    print(f"{'变体':>18}{'J':>9}{'可行':>7}{'O1':>8}{'O2':>8}{'O3':>8}")
    for (label, _), (_, result) in zip(sensitivity, sensitivity_results):
        from tests.tuning.grid_sweep import _cost
        print(f"{label:>18}{_cost(result):>9.3f}{str(result.feasible):>7}"
              f"{result.objectives[0]:>8.2f}{result.objectives[1]:>8.2f}{result.objectives[2]:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
