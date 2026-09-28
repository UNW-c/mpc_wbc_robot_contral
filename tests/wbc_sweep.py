"""WBC/MPC 混合参数多工况扫描。

支持任意维度网格（--grid JSON）与三工况联合评价：

- ``small``：0.2 m/s 瞬时冲量（小扰动，保持现有表现）
- ``mid``：0.5 m/s 瞬时冲量
- ``persist``：持续推力 + 力矩（近 viewer 实测的大扰动）

综合分 ``J_score`` = Σ J_case / J_ref_case（各工况相对基准的比值之和，
基准取当前最优配置的实测 J），避免大扰动工况的绝对量级淹没小扰动。

用法::

    .venv/Scripts/python.exe -m tests.wbc_sweep --grid '{"mpc_velocity":[500,1000]}' --workers 12
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from Algorithm.mpc import MPCSettings, MotionCommand, QPWeights
from Algorithm.wbc.task_qp import task_qp_resolver_factory
from APP.config import ROOT
from APP.mujoco_mpc_wbc import (Disturbance, MuJoCoMpcWbcSimulation, SimulationConfig,
                                _TASK_QP_WEIGHTS)
from tests.full_state_eval import _overshoot, _settle_time, full_state_cost

_LOGGER = logging.getLogger(__name__)

SECONDS = 15.0
PUSH_TIME = 5.0
PARETO_KEYS = ("small_max_pitch", "small_settle_position", "persist_max_pitch",
               "persist_settle_position")
# 各工况相对基准（当前最优配置实测 J），用于综合分归一化。
CASE_REFS = {"small": 0.78, "mid": 12.89, "persist": 418.3}
# 用户 2026-09-27 viewer 实测扰动特征（前三次中位水平）；目标收敛时间基准
# 取定稿配置实测值，score 越小越好。
T_REFS = {"small": (0.0, 0.70), "mid": (0.0, 2.78), "persist": (0.71, 4.80)}
CASES = {
    "small": dict(delta_velocity=(0.2, 0.0, 0.0)),
    "mid": dict(delta_velocity=(0.5, 0.0, 0.0)),
    "persist": dict(force=(40.0, 0.0, 0.0), torque=(0.0, 3.0, 0.0), duration=0.20),
}

DEFAULTS = {
    # MPC 权重（2026-09-26 定稿）
    "mpc_pitch": 51200.0, "mpc_pitch_rate": 960.0,
    "mpc_position": 80000.0, "mpc_velocity": 2000.0,
    # WBC 配置
    "pg": 320.0, "vg": 320.0, "preview_lead": 0.02,
    "att_kp": 80.0, "att_kd": 10.0, "reg_kp": 60.0, "reg_kd": 6.0,
    "h_kp": 100.0, "h_kd": 12.0, "height_floor": 1.0,
    # WBC 任务权重
    "task_pitch": 100.0, "task_fv": 50.0, "task_reg": 1.0,
    "task_height": 50.0, "task_roll": 100.0, "task_yaw": 50.0,
}


def _build_simulation(params: dict) -> MuJoCoMpcWbcSimulation:
    weights = dict(_TASK_QP_WEIGHTS)
    weights["pitch"] = params["task_pitch"]
    weights["forward_velocity"] = params["task_fv"]
    weights["regularization"] = params["task_reg"]
    weights["height"] = params["task_height"]
    weights["roll"] = params["task_roll"]
    weights["yaw"] = params["task_yaw"]
    config = SimulationConfig(
        wbc_backend="qp", record_stride=4,
        position_gain=params["pg"], velocity_gain=params["vg"],
        preview_lead=params["preview_lead"],
        attitude_gains=(params["att_kp"], params["att_kd"]),
        regularize_gains=(params["reg_kp"], params["reg_kd"]),
        height_gains=(params["h_kp"], params["h_kd"]),
        height_activation_floor=params["height_floor"],
    )
    simulation = MuJoCoMpcWbcSimulation(
        config=config,
        mpc_settings=MPCSettings(
            weights=QPWeights(pitch=params["mpc_pitch"], pitch_rate=params["mpc_pitch_rate"],
                              position=params["mpc_position"], velocity=params["mpc_velocity"]),
            compensate_hip_torque=False))
    simulation.control._wbc_resolver_factory = task_qp_resolver_factory(weights)
    return simulation


def evaluate(item: tuple) -> dict:
    """跑一组参数（``item = (keys, values)``），三工况联合评价。"""
    keys, values = item
    params = dict(DEFAULTS)
    params.update(dict(zip(keys, values)))
    record = dict(params)
    score = 0.0
    for case_name, push_kwargs in CASES.items():
        simulation = _build_simulation(params)
        push = Disturbance(time=PUSH_TIME, **push_kwargs)
        summary = simulation.run(MotionCommand(velocity=0.0, yaw_rate=0.0), SECONDS, None, [push])
        times = np.array([r["time"] for r in simulation.records])
        states = np.array([r["state"] for r in simulation.records])
        after = times >= PUSH_TIME
        cost, terms = full_state_cost(summary, simulation.records, simulation.hip_q)
        record[f"{case_name}_J"] = cost
        record[f"{case_name}_max_pitch"] = float(np.max(np.abs(states[after, 2])))
        record[f"{case_name}_max_position"] = float(np.max(np.abs(states[after, 0])))
        record[f"{case_name}_settle_pitch"] = _settle_time(times, states[:, 2], 0.02, PUSH_TIME)
        record[f"{case_name}_settle_position"] = _settle_time(times, states[:, 0], 0.02, PUSH_TIME)
        record[f"{case_name}_overshoot"] = _overshoot(times, states[:, 2], PUSH_TIME)
        record[f"{case_name}_stop"] = summary["stop_reason"]
        record[f"{case_name}_term_pitch_peak"] = terms["pitch_peak"]
        # 收敛时间导向评分：姿态与位移收敛时间各自归一化（用户要求都要快）。
        fail = summary["stop_reason"] is not None
        score += ((record[f"{case_name}_settle_pitch"] + 0.05)
                  / (T_REFS[case_name][0] + 0.05)
                  + (record[f"{case_name}_settle_position"] + 0.05)
                  / (T_REFS[case_name][1] + 0.05)
                  + (5.0 if fail else 0.0))
    record["J_score"] = score
    return record


def _load_checkpoint(path: Path, keys: tuple[str, ...]) -> dict[tuple, dict]:
    done: dict[tuple, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done[tuple(row[k] for k in keys)] = row
    return done


def write_report(directory: Path, rows: list[dict], keys: tuple[str, ...]) -> None:
    import csv

    ranked = sorted(rows, key=lambda row: row["J_score"])
    fields = list(rows[0].keys())
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    header = " | ".join(keys)
    lines = ["# 多工况参数扫描报告", "",
             f"- 组数：{len(rows)}；扫描维度：{header}",
             f"- 工况：small(0.2 m/s)、mid(0.5 m/s)、persist(实测特征 40N/3Nm/0.20s)，{SECONDS:.0f} s",
             f"- 评分：收敛时间导向 = Σ (T_pitch/T_ref_p + T_pos/T_ref_x)，越小越好；"
             f"失败 +5；姿态限 30° 在表中标记", "",
             "## 按 score 排序（前 25）", "",
             f"| # | {header} | score | small Tp/Tx/pitch | mid Tp/Tx/pitch | pers Tp/Tx/pitch | stop |",
             "|---|---|---|---|---|---|---|"]
    for index, row in enumerate(ranked[:25], start=1):
        dims = " | ".join(f"{row[k]:.4g}" for k in keys)
        cells = []
        for case in ("small", "mid", "persist"):
            pitch = row[f"{case}_max_pitch"]
            mark = "!" if pitch > 0.5236 else ""
            cells.append(f"{row[f'{case}_settle_pitch']:.2f}/{row[f'{case}_settle_position']:.2f}/"
                         f"{np.degrees(pitch):.1f}°{mark}")
        stops = [str(row[f"{case}_stop"]) for case in ("small", "mid", "persist")]
        lines.append(f"| {index} | {dims} | {row['J_score']:.2f} | "
                     + " | ".join(cells) + f" | {','.join('ok' if s == 'None' else s[:12] for s in stops)} |")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=str, required=True,
                        help='JSON 网格，如 \'{"mpc_velocity":[500,1000],"task_height":[0,50]}\'')
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    spec = {name: tuple(float(v) for v in values)
            for name, values in json.loads(args.grid).items()}
    unknown = set(spec) - set(DEFAULTS)
    if unknown:
        parser.error(f"未知参数维度: {unknown}")
    keys = tuple(spec)
    grid = [(keys, values) for values in itertools.product(*(spec[k] for k in keys))]
    if args.limit:
        grid = grid[:args.limit]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    directory = args.output or (ROOT / "artifacts" / "sweep" / f"multi-{stamp}")
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "results.jsonl"
    done = _load_checkpoint(checkpoint_path, keys)
    remaining = [item for item in grid if tuple(item[1]) not in done]
    _LOGGER.info("扫描 %d 组（已完成 %d，剩余 %d），%d 进程，输出 %s",
                 len(grid), len(done), len(remaining), args.workers, directory)

    started = time.perf_counter()
    rows: list[dict] = list(done.values())
    checkpoint = checkpoint_path.open("a", encoding="utf-8")
    try:
        with mp.Pool(processes=args.workers) as pool:
            for record in pool.imap_unordered(evaluate, remaining, chunksize=1):
                rows.append(record)
                checkpoint.write(json.dumps(record, allow_nan=False) + "\n")
                checkpoint.flush()
                if len(rows) % 5 == 0 or len(rows) == len(grid):
                    elapsed = time.perf_counter() - started
                    speed = elapsed / max(1, len(rows) - len(done))
                    _LOGGER.info("完成 %d/%d，已用 %.0f s，预计剩余 %.0f s",
                                 len(rows), len(grid), elapsed, speed * (len(grid) - len(rows)))
    finally:
        checkpoint.close()
    write_report(directory, rows, keys)
    _LOGGER.info("完成，总耗时 %.0f s，报告：%s", time.perf_counter() - started,
                 directory / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
