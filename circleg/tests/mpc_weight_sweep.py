"""MPC 状态权重四维联合扫描：pitch × pitch_rate × position × velocity。

依据（工作区 AGENTS.md 第 3 节）：成对参数必须联合搜索、评价函数覆盖状态偏差与
控制代价、收敛时间显式参与、输出完整结果表与排序。数据表明 pitch 与
position/velocity 反向耦合（压低姿态峰值会减慢位移收敛），四维必须一次联合扫描。

工况：前向 0.2 m/s 冲量（t=5 s），12 s 评价。先用单工况筛全网格，
Pareto 头部再补后向与 1 kHz 精细记录复核。

用法::

    .venv/Scripts/python.exe -m tests.mpc_weight_sweep --workers 16

输出到 ``artifacts/sweep/<timestamp>/``：results.csv、results.json、report.md。
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from Algorithm.mpc import MPCSettings, MotionCommand, QPWeights
from APP.config import ROOT
from APP.mujoco_mpc_wbc import Disturbance, MuJoCoMpcWbcSimulation, SimulationConfig
from tests.full_state_eval import _overshoot, _settle_time, full_state_cost

_LOGGER = logging.getLogger(__name__)

GRID = {
    "pitch": (200.0, 400.0, 800.0, 1600.0, 3200.0),
    "pitch_rate": (15.0, 30.0, 60.0, 120.0, 240.0),
    "position": (5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
    "velocity": (1000.0, 2000.0, 4000.0, 8000.0, 16000.0),
}
PUSH_TIME = 5.0  # [s]
PUSH_VELOCITY = (0.2, 0.0, 0.0)  # [m/s] 前向
SECONDS = 15.0  # [s]
SETTLE_PITCH = 0.02  # [rad]
SETTLE_POSITION = 0.02  # [m]
PITCH_LIMIT_DEG10 = 0.1745  # [rad] 姿态峰值验收上限 10 deg
PARETO_KEYS = ("max_pitch", "max_position", "settle_pitch", "settle_position")


def evaluate(parameters: tuple[float, ...]) -> dict:
    """跑一组参数，返回全部指标（含 J 分解与任务出力摘要）。"""
    pitch, pitch_rate, position, velocity, position_gain, velocity_gain = parameters
    settings = MPCSettings(weights=QPWeights(pitch=pitch, pitch_rate=pitch_rate,
                                             position=position, velocity=velocity),
                           compensate_hip_torque=False)
    simulation = MuJoCoMpcWbcSimulation(
        config=SimulationConfig(wbc_backend="qp", record_stride=4,
                                position_gain=position_gain, velocity_gain=velocity_gain),
        mpc_settings=settings)
    push = Disturbance(time=PUSH_TIME, delta_velocity=PUSH_VELOCITY)
    summary = simulation.run(MotionCommand(velocity=0.0, yaw_rate=0.0), SECONDS, None, [push])
    cost, terms = full_state_cost(summary, simulation.records, simulation.hip_q)

    times = np.array([r["time"] for r in simulation.records])
    states = np.array([r["state"] for r in simulation.records])
    torque = np.array([r["torque"] for r in simulation.records])
    usage = np.array([r["task_goal"] for r in simulation.records], dtype=object)
    del usage
    after = times >= PUSH_TIME
    record = {
        "pitch": pitch,
        "pitch_rate": pitch_rate,
        "position": position,
        "velocity": velocity,
        "position_gain": position_gain,
        "velocity_gain": velocity_gain,
        "J": cost,
        "max_pitch": float(np.max(np.abs(states[after, 2]))),
        "max_position": float(np.max(np.abs(states[after, 0]))),
        "settle_pitch": _settle_time(times, states[:, 2], SETTLE_PITCH, PUSH_TIME),
        "settle_position": _settle_time(times, states[:, 0], SETTLE_POSITION, PUSH_TIME),
        "overshoot": _overshoot(times, states[:, 2], PUSH_TIME),
        "max_torque": float(np.max(np.abs(torque))),
        "stop": summary["stop_reason"],
        "mpc_fallbacks": summary["mpc_fallbacks"],
        "wbc_fallbacks": summary["wbc_fallbacks"],
    }
    for key, value in terms.items():
        record[f"term_{key}"] = value
    return record


def pareto_mask(rows: list[dict], keys: tuple[str, ...]) -> np.ndarray:
    """右下方不被支配的点（所有维度越小越好）。"""
    points = np.array([[float(row[key]) for key in keys] for row in rows])
    count = len(points)
    keep = np.ones(count, dtype=bool)
    for i in range(count):
        if not keep[i]:
            continue
        for j in range(count):
            if i == j or not keep[j]:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                keep[i] = False
                break
    return keep


def write_report(directory: Path, rows: list[dict]) -> None:
    import csv

    fields = list(rows[0].keys())
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (directory / "results.json").write_text(
        json.dumps(rows, indent=1, allow_nan=False), encoding="utf-8")

    ranked = sorted(rows, key=lambda row: row["J"])
    mask = pareto_mask(rows, PARETO_KEYS)
    pareto = [row for row, keep in zip(rows, mask) if keep]
    pareto.sort(key=lambda row: row["J"])
    lines = ["# MPC 权重四维联合扫描报告", "",
             f"- 组数：{len(rows)}；工况：前向 {PUSH_VELOCITY[0]} m/s 冲量，{SECONDS:.0f} s",
             f"- WBC: position_gain={rows[0].get('position_gain')}, "
             f"velocity_gain={rows[0].get('velocity_gain')}",
             f"- Pareto 键：{', '.join(PARETO_KEYS)}（越小越好）",
             f"- Pareto 点数：{len(pareto)}", "",
             "## 按 J 排序（前 30）", "",
             "| # | pitch | p_rate | pos | vel | J | max_pitch | max_pos | T_pitch | T_pos | over | stop |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for index, row in enumerate(ranked[:30], start=1):
        lines.append(
            f"| {index} | {row['pitch']:.0f} | {row['pitch_rate']:.0f} | {row['position']:.0f} | "
            f"{row['velocity']:.0f} | {row['J']:.2f} | {row['max_pitch']:.4f} | "
            f"{row['max_position']:.4f} | {row['settle_pitch']:.2f} | "
            f"{row['settle_position']:.2f} | {row['overshoot']:.4f} | {row['stop']} |")
    lines += ["", "## Pareto 前沿（姿态/位移峰值与收敛时间四维）", "",
              "| pitch | p_rate | pos | vel | J | max_pitch | max_pos | T_pitch | T_pos | over |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in pareto:
        lines.append(
            f"| {row['pitch']:.0f} | {row['pitch_rate']:.0f} | {row['position']:.0f} | "
            f"{row['velocity']:.0f} | {row['J']:.2f} | {row['max_pitch']:.4f} | "
            f"{row['max_position']:.4f} | {row['settle_pitch']:.2f} | "
            f"{row['settle_position']:.2f} | {row['overshoot']:.4f} |")
    over = [row for row in ranked if row["max_pitch"] > PITCH_LIMIT_DEG10]
    lines += ["", f"## 姿态峰值 > 10 deg 的组：{len(over)} / {len(rows)}", ""]
    for row in over[:20]:
        lines.append(f"- pitch={row['pitch']:.0f} p_rate={row['pitch_rate']:.0f} "
                     f"pos={row['position']:.0f} vel={row['velocity']:.0f} "
                     f"max_pitch={row['max_pitch']:.4f}")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_checkpoint(directory: Path) -> dict[tuple, dict]:
    """读回已完成的结果，支持中止后续跑（按参数组跳过）。"""
    path = directory / "results.jsonl"
    done: dict[tuple, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done[(row["pitch"], row["pitch_rate"], row["position"], row["velocity"],
              row["position_gain"], row["velocity_gain"])] = row
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 组（调试用）")
    parser.add_argument("--position-gain", type=float, default=0.0,
                        help="WBC 位置环增益（阶段扫描用）")
    parser.add_argument("--velocity-gain", type=float, default=4.0,
                        help="WBC 速度环增益（阶段扫描用）")
    parser.add_argument("--grid-pitch", type=str, default=None, help="逗号分隔 pitch 档位")
    parser.add_argument("--grid-pitch-rate", type=str, default=None)
    parser.add_argument("--grid-position", type=str, default=None)
    parser.add_argument("--grid-velocity", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name, value in (("pitch", args.grid_pitch), ("pitch_rate", args.grid_pitch_rate),
                        ("position", args.grid_position), ("velocity", args.grid_velocity)):
        if value:
            GRID[name] = tuple(float(x) for x in value.split(","))

    grid = [group + (args.position_gain, args.velocity_gain)
            for group in itertools.product(GRID["pitch"], GRID["pitch_rate"],
                                           GRID["position"], GRID["velocity"])]
    if args.limit:
        grid = grid[:args.limit]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    directory = args.output or (ROOT / "artifacts" / "sweep" / stamp)
    directory.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoint(directory)
    remaining = [group for group in grid if group not in done]
    _LOGGER.info("扫描 %d 组（已完成 %d，剩余 %d），%d 进程，输出 %s",
                 len(grid), len(done), len(remaining), args.workers, directory)

    started = time.perf_counter()
    rows: list[dict] = list(done.values())
    checkpoint = (directory / "results.jsonl").open("a", encoding="utf-8")
    try:
        with mp.Pool(processes=args.workers) as pool:
            for record in pool.imap_unordered(evaluate, remaining, chunksize=1):
                rows.append(record)
                checkpoint.write(json.dumps(record, allow_nan=False) + "\n")
                checkpoint.flush()
                if len(rows) % 25 == 0 or len(rows) == len(grid):
                    elapsed = time.perf_counter() - started
                    remaining_count = len(grid) - len(rows)
                    speed = elapsed / max(1, len(rows) - len(done))
                    _LOGGER.info("完成 %d/%d，已用 %.0f s，预计剩余 %.0f s",
                                 len(rows), len(grid), elapsed, speed * remaining_count)
    finally:
        checkpoint.close()
    write_report(directory, rows)
    _LOGGER.info("完成，总耗时 %.0f s，报告：%s", time.perf_counter() - started,
                 directory / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
