"""轮腿参数寻优：代价函数 + 分阶段联合网格遍历（口径照 circleg ``tests/tuning/grid_sweep.py``）。

设计依据：circleg ``docs/算法漏洞排查记录.md`` 第 13 次排查。要点：
- 每阶段只扫 4 个参数（成对参数放在同一阶段，二维以上联合扫描）；
- 未参与本阶段的参数取"上一阶段的最优值"（阶段 1 取当前部署默认值）；
- ``multiprocessing.Pool`` 并行，worker 初始化时把数学库线程数设为 1，避免与进程数叠加；
- 主进程逐组追加写 CSV（可中断续跑），阶段末打印 J 最小前 10 并写 ``stage<k>_top10.md``；
- CSV 保存全部原始指标，之后想改 J 的权重直接重排，不用重跑。

用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/tune_grid.py --stage 1 --workers 12
"""
from __future__ import annotations

# 必须在导入 numpy 之前设好，否则 worker 里的 BLAS 会多线程抢占（circleg 第 13 次排查记录）
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gbc_mpc_wbc.objective import (DEFAULTS, MODEL_VARIANT, SCENARIO_NAMES, TUNED, evaluate,  # noqa: E402
                                   scalar, validate)

# 每个阶段的网格：同阶段的参数一起联合扫描，其余参数取上一阶段最优
STAGES: dict[int, dict[str, list[float]]] = {
    # 阶段 1：腿角限幅（是硬约束、直接决定摆幅） + pitch 优先 + 纵向（成对）
    1: {"hip_angle_limit": [0.10, 0.15, 0.20, 0.25], "pitch": [1.5e5, 3.0e5, 6.0e5],
        "position": [2.0e4, 4.0e4], "velocity": [1333.3, 2666.7]},
    # 阶段 2：偏航 + 终端代价（成对）
    2: {"yaw": [30.0, 60.0, 120.0], "yaw_rate": [4.0, 8.0, 16.0],
        "terminal_scale": [3.0, 5.0, 10.0]},
    # 阶段 4（用户 2026-10-05 指定）：腿角阻尼三维联合 —— MPC 腿角/腿角速度权重
    # + WBC 腿角任务权重（验证 WBC/MPC 协调是否有影响）。hip_angle_limit 固定 0.15。
    4: {"hip_angle": [50.0, 100.0, 200.0],
        "hip_rate": [3.0, 8.0, 16.0, 30.0],
        "w_angle": [60.0, 100.0, 200.0]},
    # 阶段 5（用户 2026-10-05）：腿角预算 × 位置权重。两者耦合：前者是摆幅硬上限，
    # 后者决定 MPC 对腿的需求强度，共同决定"摆幅 vs 位移"的取舍面。
    5: {"hip_angle_limit": [0.10, 0.125, 0.15, 0.175, 0.20],
        "position": [2.0e4, 4.0e4, 8.0e4]},
}
_REFINE_STAGE = 3
_DAMPING_STAGE = 4
_BUDGET_STAGE = 5
_TUNED_BASED_STAGES = (_DAMPING_STAGE, _BUDGET_STAGE)
_REFINE_STAGE1_KEYS = ("hip_angle_limit", "pitch", "position", "velocity")
_REFINE_FACTORS = (1.0 / 1.5, 1.0, 1.5)

_FIELDS = tuple(DEFAULTS) + ("model_variant", "J", "feasible", "failure", "error_term", "effort",
                                 "priority_penalty", "priority_events", "channels", "per_scenario", "wall_s")


def _init_worker() -> None:
    import numpy as _np  # noqa: F401


def _worker(payload):
    params, scenarios, baseline = payload
    result = evaluate(params, scenarios=scenarios, baseline=baseline)
    return params, result


def _row(params: dict, result) -> dict:
    channels = {name: [None if not np.isfinite(v) else round(v, 5) for v in values]
                for name, values in result.channels.items()}
    row = {key: params.get(key) for key in DEFAULTS}
    row.update(model_variant=MODEL_VARIANT,
               J=round(scalar(result), 4), feasible=result.feasible, failure=result.failure or "",
               error_term=round(float(result.objectives[0]), 4),
               effort=round(float(result.objectives[1]), 4),
               priority_penalty=round(float(result.objectives[2]), 4),
               priority_events=result.violation_text,
               channels=json.dumps(channels),
               per_scenario=json.dumps({k: {kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv)
                                            for kk, vv in v.items()} for k, v in result.metrics.items()}),
               wall_s=round(result.wall_time, 1))
    return row


def _read_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add(tuple(round(float(row[key]), 6) for key in DEFAULTS))
    return done


def _best(path: Path) -> dict:
    best, best_j = dict(DEFAULTS), float("inf")
    if not path.exists():
        return best
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["feasible"] != "True":
                continue
            value = float(row["J"])
            if value < best_j:
                best_j = value
                best = {key: float(row[key]) for key in DEFAULTS}
    return best


def _stage_tasks(stage: int, base: dict) -> list:
    if stage == _REFINE_STAGE:
        grids = [[base[key] * factor for factor in _REFINE_FACTORS] for key in _REFINE_STAGE1_KEYS]
        keys = _REFINE_STAGE1_KEYS
    else:
        grid = STAGES[stage]
        keys = tuple(grid)
        grids = [grid[key] for key in keys]
    tasks = []
    for values in itertools.product(*grids):
        params = dict(base)
        params.update(zip(keys, values))
        tasks.append((params, None))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=sorted(STAGES) + [_REFINE_STAGE])
    parser.add_argument("--workers", type=int, default=14, help="进程数；每个 worker 内 BLAS 固定 1 线程（保证可复现）")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "tuning" / "grid")
    parser.add_argument("--prev", type=Path, default=None, help="上一阶段 CSV（默认按目录自动找）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 组（冒烟测试用，0 = 全部）")
    parser.add_argument("--validate", action="store_true", help="阶段末对 top1 跑验证集并写入报告")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"stage{args.stage}.csv"

    if args.stage == 1:
        base = dict(DEFAULTS)
    elif args.stage in _TUNED_BASED_STAGES:
        base = dict(TUNED)          # 以当前部署为基底，只扫本阶段的轴
    else:
        source = args.prev or out / ("stage1.csv" if args.stage == _REFINE_STAGE else f"stage{args.stage - 1}.csv")
        base = _best(source)
    tasks = _stage_tasks(args.stage, base)
    if args.limit:
        tasks = tasks[:args.limit]
    done = _read_done(csv_path)
    todo = [task for task in tasks
            if tuple(round(float(task[0][key]), 6) for key in DEFAULTS) not in done]
    print(f"阶段 {args.stage}：共 {len(tasks)} 组，已完成 {len(tasks) - len(todo)}，待跑 {len(todo)}；"
          f"base={ {key: round(base[key], 3) for key in DEFAULTS if base[key] != DEFAULTS[key]} }")
    print(f"CSV = {csv_path}")

    if todo:
        # 弱优先级校验需要基线通道分。缓存在 artifacts/tuning/ 下（全局共享，只算一次），
        # 否则 worker 里的 priority_penalty 恒为 0，且每个阶段都要重算一次白等。
        baseline_path = out.parent / "baseline_channels_v2.json"
        if baseline_path.exists():
            baseline = {name: tuple(values) for name, values in
                        json.loads(baseline_path.read_text(encoding="utf-8")).items()}
        else:
            print("计算基线通道分（弱优先级校验用，约 1 分钟）...", flush=True)
            baseline = evaluate(dict(DEFAULTS)).channels
            baseline_path.write_text(json.dumps(baseline, ensure_ascii=False), encoding="utf-8")
        todo = [(params, scenarios, baseline) for params, scenarios in todo]

    if todo:
        from multiprocessing import Pool

        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDS)
            if write_header:
                writer.writeheader()
            start = time.perf_counter()
            with Pool(args.workers, initializer=_init_worker) as pool:
                for index, (params, result) in enumerate(pool.imap_unordered(_worker, todo), 1):
                    writer.writerow(_row(params, result))
                    handle.flush()
                    if index % 5 == 0 or index == len(todo):
                        elapsed = time.perf_counter() - start
                        eta = elapsed / index * (len(todo) - index)
                        print(f"  {index}/{len(todo)}  {elapsed:.0f}s  预计剩余 {eta:.0f}s", flush=True)

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["feasible"] == "True":
                rows.append(row)
    rows.sort(key=lambda row: float(row["J"]))
    lines = [f"# 阶段 {args.stage} top10（J = Σ通道(误差+收敛) + 弱优先级罚；"
             f"W_EFFORT=0，出力只记录不排序）", "",
             "| 排名 | J | 误差+收敛 | 出力 | 弱优先级罚 | 事件 | " + " | ".join(DEFAULTS) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(DEFAULTS)]
    for rank, row in enumerate(rows[:10], 1):
        lines.append(f"| {rank} | {row['J']} | {row['error_term']} | {row['effort']} | "
                     f"{row['priority_penalty']} | {row['priority_events'][:60]} | "
                     + " | ".join(row[key] for key in DEFAULTS) + " |")
    if args.validate and rows:
        top = {key: float(rows[0][key]) for key in DEFAULTS}
        report = validate(top)
        lines += ["", "## 验证集（top1，不进 J，仅报告）", "",
                  "| 工况 | 通过 | 停机 | 俯仰峰 deg | 腿角速率峰 rad/s | 腿力矩占用 |",
                  "|---|---|---|---|---|---|"]
        for name, item in report.items():
            lines.append(f"| {name} | {'是' if item['passed'] else '否'} | {item['stop'] or '—'} | "
                         f"{item['peak_pitch_deg']} | {item['leg_speed_peak']} | {item['leg_occupation']} |")
    text = "\n".join(lines) + "\n"
    (out / f"stage{args.stage}_top10.md").write_text(text, encoding="utf-8")
    print(text)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        total = sum(1 for _ in csv.DictReader(handle))
    print(f"CSV 共 {total} 行，可行 {len(rows)} 行；工况集 {SCENARIO_NAMES}")


if __name__ == "__main__":
    main()
