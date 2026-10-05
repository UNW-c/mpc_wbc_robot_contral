"""M3：6 状态 held-leg MPC + WBC 闭环（零速站立 + 小推扰），只验证可行性，不调参。

验收（部署方案第 5 节 M3 行）：
1. 开环不稳定极点与 √(g/h) 量级一致；
2. 零速站立 + 小扰动后回位，不触发 persistent_mpc_fallback。

扰动工况取 circleg ``tests/full_state_eval.py`` 的同名工况：t = 5 s 时给机身世界系速度增量
forward (+0.20, 0, 0) / backward (−0.20, 0, 0) / lateral (0, +0.20, 0)，评估到 12 s。
另跑 30 s 无扰动站立，检查 M2 的缓慢漂移是否被 MPC 消除。

用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/m3_mpc_wbc.py [--backends nullspace wls]
输出：results/m3_mpc_wbc.md / .json，时序 results/m3_<backend>_<case>.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gbc_mpc_wbc.model_builder import pendulum_pole  # noqa: E402
from gbc_mpc_wbc.simulation import (Disturbance, Gbc486MpcWbcSimulation, SimulationConfig,  # noqa: E402
                                    default_command)

RESULTS = ROOT / "results"
PUSH_TIME = 5.0
CASES = {
    "stand30": (None, 30.0),
    "forward": ((0.20, 0.0, 0.0), 12.0),
    "backward": ((-0.20, 0.0, 0.0), 12.0),
    "lateral": ((0.0, 0.20, 0.0), 12.0),
}
# 回位判据（报告用的测量口径，不是控制参数）：取 circleg full_state_eval 的同名阈值口径
SETTLE_POSITION = 0.01   # [m]
SETTLE_PITCH = 0.01      # [rad]


def settle_time(times, values, threshold, start):
    """start 之后最后一次 |value| > threshold 的时刻减 start；从未超过则 0；到结束仍超过则 inf。"""
    after = times >= start
    t, v = times[after], np.abs(values[after])
    outside = np.nonzero(v > threshold)[0]
    if not outside.size:
        return 0.0
    if outside[-1] == len(v) - 1:
        return float("inf")
    return float(t[outside[-1] + 1] - start)


def case_metrics(records, push):
    t = np.array([r["t"] for r in records])
    s = np.array([r["s"] for r in records])
    v = np.array([r["v"] for r in records])
    pitch = np.array([r["pitch"] for r in records])
    yaw = np.array([r["yaw"] for r in records])
    start = PUSH_TIME if push else 0.0
    after = t >= start
    tail = t >= t[-1] - 2.0
    return dict(
        peak_s=float(np.max(np.abs(s[after]))), peak_pitch=float(np.max(np.abs(pitch[after]))),
        peak_v=float(np.max(np.abs(v[after]))), peak_yaw=float(np.max(np.abs(yaw[after]))),
        settle_s=settle_time(t, s, SETTLE_POSITION, start), settle_pitch=settle_time(t, pitch, SETTLE_PITCH, start),
        final_s=float(s[-1]), final_v=float(v[-1]), final_pitch=float(pitch[-1]),
        tail_s_span=float(np.ptp(s[tail])), tail_pitch_span=float(np.ptp(pitch[tail])),
    )


def circleg_revision():
    """上层 circleg 的 git 版本 + 本进程实际导入的 Algorithm 模块文件 SHA-256（上层代码可能在变）。"""
    import hashlib
    root = ROOT.parent
    modules = sorted(name for name in sys.modules if name == "Algorithm" or name.startswith("Algorithm."))
    hashes = {}
    for name in modules:
        path = getattr(sys.modules[name], "__file__", None)
        if path:
            hashes[str(Path(path).relative_to(root)).replace("\\", "/")] = \
                hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "Algorithm"],
                               capture_output=True, text=True, check=True).stdout.splitlines()
        return dict(head=head, algorithm_uncommitted=[line[3:] for line in dirty if line.strip()],
                    imported_sha256=hashes)
    except (OSError, subprocess.CalledProcessError) as error:
        return dict(error=str(error), imported_sha256=hashes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["nullspace", "wls"], choices=["nullspace", "wls", "qp"])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    RESULTS.mkdir(exist_ok=True)
    revision = circleg_revision()  # 启动时立即记录：导入的就是这一版
    rows, model_info = [], None
    for backend in args.backends:
        for case in args.cases:
            push, duration = CASES[case]
            sim = Gbc486MpcWbcSimulation(SimulationConfig(backend=backend))
            disturbances = [] if push is None else [Disturbance(PUSH_TIME, delta_velocity=push)]
            summary = sim.run(default_command(), duration, disturbances)
            model_info = summary.pop("model")
            summary.update(case=case, push=push, metrics=case_metrics(sim.records, push),
                           mpc_settings=dict(wheel_torque_limit=sim.mpc.settings.wheel_torque_limit,
                                             nominal_normal_force=sim.mpc.settings.nominal_normal_force,
                                             yaw_moment_limit=sim.mpc.settings.yaw_moment_limit,
                                             scrub_moment_coefficient=sim.mpc.settings.scrub_moment_coefficient,
                                             limits=dict(commanded_acceleration=sim.mpc.settings.limits.commanded_acceleration,
                                                         max_acceleration=sim.mpc.settings.limits.max_acceleration,
                                                         max_braking=sim.mpc.settings.limits.max_braking)))
            rows.append(summary)
            with gzip.open(RESULTS / f"m3_{backend}_{case}.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(sim.records, handle)
            print(f"{backend:9s} {case:9s} completed={summary['completed']} stop={summary['stop_reason']} "
                  f"t={summary['simulated']:.2f}", flush=True)
    payload = dict(mujoco=mujoco.__version__, circleg=revision, model=model_info, runs=rows)
    (RESULTS / "m3_mpc_wbc.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    text = to_markdown(payload)
    (RESULTS / "m3_mpc_wbc.md").write_text(text, encoding="utf-8")
    print(text)


def to_markdown(p) -> str:
    model = p["model"]
    A, B = np.array(model["A"]), np.array(model["B"])
    h = model["metadata"]["com_height"]
    lines = [
        "# M3：6 状态 held-leg MPC + WBC 闭环", "",
        f"脚本：`tools/m3_mpc_wbc.py`（mujoco {p['mujoco']}；circleg HEAD {p['circleg'].get('head')}，"
        f"`Algorithm/` 未提交改动：{p['circleg'].get('algorithm_uncommitted') or '无'}；"
        f"导入文件 SHA-256 前 12 位见 json 的 `circleg.imported_sha256`；"
        f"`Algorithm/mpc/controller.py` = {p['circleg'].get('imported_sha256', {}).get('Algorithm/mpc/controller.py')}）。",
        "L0 = 0.42 m。MPC 50 Hz（circleg `WheelMPC` 原样复用），WBC 250 Hz，物理 1 kHz。"
        "增益 / 权重为 circleg 默认值直接搬用，**未调参**。", "",
        "## 1. 降阶模型（MuJoCo 实测线性化）", "",
        "状态 [s, v, θ, θ̇, ψ, ψ̇]，输入 [f_R, f_L] (N)。", "",
        "```", "A =", np.array2string(A, precision=4, suppress_small=True), "B =",
        np.array2string(B, precision=4, suppress_small=True), "```", "",
        f"- 特征值：{np.round(np.linalg.eigvals(A).real, 4).tolist()}；不稳定极点 **{model['unstable_pole']:.3f} rad/s**",
        f"- 质心高（接触点以上）h = {h:.4f} m，√(g/h) = {pendulum_pole(h):.3f} rad/s"
        f"（比值 {model['unstable_pole'] / pendulum_pole(h):.2f}）",
        f"- 轮半径 {model['wheel_radius']}，半轮距 {model['half_track']:.4f} m，偏航惯量 {model['yaw_inertia']:.4f} kg·m²，"
        f"配平俯仰 {model['metadata']['equilibrium_pitch']:.2e} rad", "",
        "## 2. 闭环结果", "",
        f"推扰：t = {PUSH_TIME} s 机身世界系速度增量；回位判据 |s| < {SETTLE_POSITION*1000:.0f} mm、"
        f"|θ| < {SETTLE_PITCH*1000:.0f} mrad（最后一次超出后的时刻；inf = 到结束仍超出）。", "",
        "| 后端 | 工况 | 完成 | 停机 | 时长 s | MPC 次数 / fallback / relaxed | WBC fallback / clip / infeasible |"
        " 峰值 s mm | 峰值 θ mrad | 峰值 v mm/s | s 回位 s | θ 回位 s | 末 s mm | 末 2 s s 峰峰 mm | max\\|τ\\| 轮 N·m |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in p["runs"]:
        mt = r["metrics"]
        lines.append(
            f"| {r['backend']} | {r['case']} | {'是' if r['completed'] else '否'} | {r['stop_reason'] or '—'} | "
            f"{r['simulated']:.2f} | {r['mpc_calls']} / {r['mpc_fallbacks']} / {r['mpc_relaxed']} | "
            f"{r['wbc_fallback']} / {r['wbc_clip']} / {r['wbc_infeasible']} | {mt['peak_s']*1e3:.1f} | "
            f"{mt['peak_pitch']*1e3:.1f} | {mt['peak_v']*1e3:.0f} | {mt['settle_s']:.2f} | {mt['settle_pitch']:.2f} | "
            f"{mt['final_s']*1e3:.2f} | {mt['tail_s_span']*1e3:.2f} | "
            f"{max(r['max_torque'][4:]):.2f} |")
    lines += ["", "| 后端 | 工况 | MPC 耗时 p50/p95 ms | WBC 耗时 p50/p95 ms | 模式拍数 | forward exhausted 拍数 |",
              "|---|---|---|---|---|---|"]
    for r in p["runs"]:
        tm = r["timing_ms"]
        lines.append(f"| {r['backend']} | {r['case']} | {tm['mpc']['p50']:.2f}/{tm['mpc']['p95']:.2f} | "
                     f"{tm['wbc']['p50']:.2f}/{tm['wbc']['p95']:.2f} | {r['modes']} | "
                     f"{r['exhausted']['forward_velocity']} |")
    settings = p["runs"][0]["mpc_settings"]
    lines += ["", f"MPC 机器人相关量（显式覆盖）：{settings}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
