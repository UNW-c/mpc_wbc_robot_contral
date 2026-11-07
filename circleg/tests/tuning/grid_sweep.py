"""第 13 次排查：代价函数 + 分阶段网格遍历（Pool + CSV，取代 Optuna 方案）。

阶段：
    1. position x velocity x hip_rate x hip_angle_limit（81 组）
    2. tracking_bandwidth x hip_tracking x hip_angle（27 组）
    3. pitch x pitch_rate x terminal_mode（18 组）
    4. 以阶段 1~3 的最优点为中心，阶段 1 的四个参数取 {÷1.5, x1, x1.5}（81 组）

用法：
    .venv/Scripts/python.exe -m tests.tuning.grid_sweep --stage 1 --workers 12 \
        --out artifacts/tuning/grid/20260929
中断后重新运行同一条命令：会读取已有 CSV 并跳过算过的参数组。

代价（标量）：
    可行   J = O1 / 1.0 + O2 / 1.2
    不可行 J = 100 + Σ max(0, 违反量)
O3（髋角峰）不进入 J，只记录；CSV 保存全部原始指标。
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.tuning.objective import E5_PARAMS, evaluate

_PARAM_KEYS = ("position", "velocity", "pitch", "pitch_rate", "hip_angle", "hip_rate",
               "terminal_mode", "terminal_scale", "tracking_bandwidth", "hip_tracking",
               "hip_angle_limit")
_SCENARIOS = ("T-a", "T-b", "T-c", "T-d", "T-e", "T-f")
_METRIC_FIELDS = ("settle", "peak_position", "peak_pitch", "pitch_cross", "hip_angle_peak",
                  "hip_speed_peak", "hip_occ", "final_position_abs")
_VIOLATION_KEYS = ("guard", "fallback", "pitch_peak_deg", "pitch_cross", "hip_speed_peak",
                   "hip_torque_occupation", "solve_p95_ms", "zeta")
# 固定字段集（不能从第一行推导：ζ 预检失败的行只有 violation_zeta）
_FIELDS = (["stage", "J", "feasible", "failure", "O1", "O2", "O3"]
           + [f"param_{key}" for key in _PARAM_KEYS]
           + [f"{name}_{field}" for name in _SCENARIOS for field in _METRIC_FIELDS]
           + [f"violation_{key}" for key in _VIOLATION_KEYS])

_GRIDS = {
    1: {"position": [2.0e4, 8.0e4, 3.2e5],
        "velocity": [500.0, 2000.0, 8000.0],
        "hip_rate": [0.3, 3.0, 10.0],
        "hip_angle_limit": [0.55, 0.65, 0.75]},
    2: {"tracking_bandwidth": [7.0, 10.0, 15.0],
        "hip_tracking": [50.0, 100.0, 200.0],
        "hip_angle": [15.0, 50.0, 150.0]},
    3: {"pitch": [2.0e4, 5.12e4, 1.5e5],
        "pitch_rate": [300.0, 960.0, 3000.0],
        "terminal_mode": ["scale", "lqr"]},
}


def _worker_init() -> None:
    """防止 numpy 的多线程与进程数叠加抢占 CPU（Optuna 线程方案低效的原因之一）。"""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def _evaluate_one(params: dict):
    return params, evaluate(params, horizon_after_push=4.0)


def _cost(result) -> float:
    if not result.feasible:
        return 100.0 + float(sum(max(0.0, v) for v in result.violations.values()))
    o1, o2, _ = result.objectives
    return float(o1) / 1.0 + float(o2) / 1.2


def _record(stage: int, params: dict, result) -> dict:
    row = {"stage": stage, "J": round(_cost(result), 6),
           "feasible": bool(result.feasible), "failure": result.failure or ""}
    row["O1"], row["O2"], row["O3"] = (round(float(v), 4) for v in result.objectives)
    for key in _PARAM_KEYS:
        row[f"param_{key}"] = params.get(key, "")
    for name in _SCENARIOS:
        metric = result.per_scenario.get(name)
        if metric is None:
            for field in ("settle", "peak_position", "peak_pitch", "pitch_cross",
                          "hip_angle_peak", "hip_speed_peak", "hip_occ", "final_position_abs"):
                row[f"{name}_{field}"] = ""
            continue
        row[f"{name}_settle"] = round(float(metric["settle"]), 4)
        row[f"{name}_peak_position"] = round(float(metric["peak_position"]), 4)
        row[f"{name}_peak_pitch"] = round(float(metric["peak_pitch"]), 3)
        row[f"{name}_pitch_cross"] = int(metric["pitch_cross"])
        row[f"{name}_hip_angle_peak"] = round(float(metric["hip_angle_peak"]), 4)
        row[f"{name}_hip_speed_peak"] = round(float(metric["hip_speed_peak"]), 3)
        row[f"{name}_hip_occ"] = round(float(metric["hip_occ"]), 3)
        row[f"{name}_final_position_abs"] = round(float(metric["final_position_abs"]), 5)
    for key in _VIOLATION_KEYS:
        value = result.violations.get(key)
        row[f"violation_{key}"] = "" if value is None else round(float(value), 5)
    return row


def _row_key(row: dict) -> tuple:
    """参数组的去重键（同时兼容 CSV 行 param_xxx 与 params 字典的裸名）。"""
    values = []
    for key in _PARAM_KEYS:
        raw = row.get(f"param_{key}", row.get(key))
        try:
            values.append(round(float(raw), 10))
        except (TypeError, ValueError):
            values.append(str(raw))
    return tuple(values)


def _read_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _best_of(paths) -> dict:
    rows = [row for path in paths for row in _read_rows(path)]
    if not rows:
        return dict(E5_PARAMS, terminal_mode="scale")
    best = min(rows, key=lambda row: float(row["J"]))
    params = {key: best.get(f"param_{key}") for key in _PARAM_KEYS}
    for key in ("position", "velocity", "pitch", "pitch_rate", "hip_angle", "hip_rate",
                "terminal_scale", "tracking_bandwidth", "hip_tracking", "hip_angle_limit"):
        params[key] = float(params[key])
    return params


def _base_params(stage: int, out_dir: Path) -> dict:
    if stage == 1:
        return dict(E5_PARAMS, terminal_mode="scale")
    previous = [out_dir / f"stage{k}.csv" for k in range(1, stage)]
    return _best_of(previous)


def _expand(stage: int, base: dict, out_dir: Path) -> list:
    grouped = []
    if stage in (1, 2, 3):
        names = list(_GRIDS[stage])
        for values in itertools.product(*(_GRIDS[stage][name] for name in names)):
            params = dict(base)
            for name, value in zip(names, values):
                if name == "terminal_mode":
                    params["terminal_mode"] = value
                    params["terminal_scale"] = 5.0 if value == "scale" else 5.0
                else:
                    params[name] = float(value)
            grouped.append(params)
        return grouped
    # 阶段 4：以阶段 1~3 的最优点为中心做细化
    center = _best_of([out_dir / f"stage{k}.csv" for k in (1, 2, 3)])
    for values in itertools.product(
            (1.0 / 1.5, 1.0, 1.5), (1.0 / 1.5, 1.0, 1.5),
            (1.0 / 1.5, 1.0, 1.5), (-0.05, 0.0, 0.05)):
        params = dict(center)
        params["position"] = center["position"] * values[0]
        params["velocity"] = center["velocity"] * values[1]
        params["hip_rate"] = center["hip_rate"] * values[2]
        params["hip_angle_limit"] = float(np.clip(center["hip_angle_limit"] + values[3],
                                                  0.55, 0.80))
        grouped.append(params)
    return grouped


def _write_top10(csv_path: Path, markdown_path: Path) -> None:
    rows = _read_rows(csv_path)
    rows.sort(key=lambda row: float(row["J"]))
    lines = [f"# {csv_path.stem} top10", "",
             f"共 {len(rows)} 组，可行 {sum(1 for row in rows if row['feasible'] == 'True')} 组。", "",
             "| 排名 | J | 可行 | O1 | O2 | O3 | position | velocity | hip_rate | limit |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for index, row in enumerate(rows[:10], start=1):
        lines.append(
            f"| {index} | {row['J']} | {row['feasible']} | {row['O1']} | {row['O2']} | {row['O3']} "
            f"| {row.get('param_position')} | {row.get('param_velocity')} "
            f"| {row.get('param_hip_rate')} | {row.get('param_hip_angle_limit')} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"stage{args.stage}.csv"
    base = _base_params(args.stage, out_dir)
    grid = _expand(args.stage, base, out_dir)
    done_keys = {_row_key(row) for row in _read_rows(csv_path)}
    todo = [params for params in grid if _row_key(params) not in done_keys]
    print(f"阶段 {args.stage}：共 {len(grid)} 组，已完成 {len(done_keys)}，本次 {len(todo)} 组"
          f"（{args.workers} 进程，预计 {len(todo) * 218 / 60 / args.workers:.0f} 分钟）")
    if not todo:
        _write_top10(csv_path, out_dir / f"stage{args.stage}_top10.md")
        print("无新组，已刷新 top10")
        return 0

    start = time.perf_counter()
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS, extrasaction="ignore")
        if not csv_path.exists() or not done_keys:
            writer.writeheader()
            stream.flush()
        with Pool(args.workers, initializer=_worker_init) as pool:
            for count, (params, result) in enumerate(pool.imap_unordered(_evaluate_one, todo,
                                                                         chunksize=1), start=1):
                row = _record(args.stage, params, result)
                writer.writerow(row)
                stream.flush()
                print(f"  [{count}/{len(todo)}] J={row['J']:.3f} 可行={row['feasible']} "
                      f"O1={row['O1']:.2f} O2={row['O2']:.2f} O3={row['O3']:.2f} "
                      f"({time.perf_counter() - start:.0f}s)", flush=True)
    _write_top10(csv_path, out_dir / f"stage{args.stage}_top10.md")
    elapsed = time.perf_counter() - start
    print(f"阶段 {args.stage} 完成：{len(todo)} 组，用时 {elapsed / 60:.1f} 分钟")
    print(f"CSV：{csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
