"""M4：速度 / 偏航指令阶跃跟踪（6 状态 MPC + WBC），只验证可行性，不调参。

验收（部署方案第 5 节 M4 行）：阶跃速度能跟踪，不触发 attitude_guard / speed_guard。

指令（测试输入，不是控制参数；用户未指定数值，按 circleg MotionLimits 的
max_speed = 2.5 m/s、max_yaw_rate = 2.0 rad/s 取其中若干档）：
  t ∈ [HOLD_START, HOLD_END) 给指令，其余时间为 0，总时长 DURATION。

用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/m4_command_tracking.py [--backends nullspace] [--cases v0.5 ...]
输出：results/m4_command_tracking.md / .json，时序 results/m4_<backend>_<case>.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gbc_mpc_wbc.simulation import Gbc486MpcWbcSimulation, SimulationConfig  # noqa: E402
from Algorithm.mpc import MotionCommand  # noqa: E402
from m3_mpc_wbc import circleg_revision  # noqa: E402

RESULTS = ROOT / "results"
HOLD_START, HOLD_END, DURATION = 2.0, 8.0, 14.0
CASES = {                       # (速度 m/s, 偏航角速度 rad/s)
    "v0.5": (0.5, 0.0),
    "v1.0": (1.0, 0.0),
    "v2.0": (2.0, 0.0),
    "v-1.0": (-1.0, 0.0),
    "w1.0": (0.0, 1.0),
    "v1.0w0.5": (1.0, 0.5),
}
# 报告口径（测量用，不是控制参数）
CRUISE_WINDOW = (HOLD_END - 2.0, HOLD_END)   # 稳态跟踪评估窗
STOP_SPEED = 0.01                             # 停车判据 [m/s]


def make_command(v, w):
    def command(t):
        active = HOLD_START <= t < HOLD_END
        return MotionCommand(velocity=v if active else 0.0, yaw_rate=w if active else 0.0)
    return command


def first_time(t, mask):
    idx = np.nonzero(mask)[0]
    return float(t[idx[0]]) if idx.size else float("nan")


def metrics(records, v_cmd, w_cmd):
    t = np.array([r["t"] for r in records])
    v = np.array([r["v"] for r in records])
    w = np.array([r["yaw_rate"] for r in records])
    s = np.array([r["s"] for r in records])
    pitch = np.array([r["pitch"] for r in records])
    plan_v = np.array([np.nan if r["plan_v"] is None else r["plan_v"] for r in records])
    plan_w = np.array([np.nan if r["plan_w"] is None else r["plan_w"] for r in records])
    torque = np.abs(np.array([r["torque"] for r in records]))
    legs = np.array([r["leg"] for r in records])
    cruise = (t >= CRUISE_WINDOW[0]) & (t < CRUISE_WINDOW[1])
    drive = (t >= HOLD_START) & (t < HOLD_END)
    after = t >= HOLD_END
    out = dict(
        max_abs_pitch=float(np.max(np.abs(pitch))), max_wheel_torque=float(torque[:, 4:].max()),
        max_leg_torque=float(torque[:, :4].max()),
        wheel_torque_saturated_steps=int(np.sum(torque[:, 4:].max(axis=1) >= 4.0 - 1e-6)),
        max_leg_length_dev=float(np.abs(legs[:, [0, 2]] - 0.42).max()),
        max_leg_angle_dev=float(np.abs(legs[:, [1, 3]] - legs[0, 1]).max()),
        final_v=float(v[-1]), final_w=float(w[-1]),
    )
    if v_cmd:
        sign = np.sign(v_cmd)
        out.update(
            cruise_v_mean=float(v[cruise].mean()), cruise_v_error=float(v[cruise].mean() - v_cmd),
            cruise_v_ripple=float(np.ptp(v[cruise])),
            plan_tracking_rms=float(np.sqrt(np.nanmean((v[drive] - plan_v[drive]) ** 2))),
            plan_tracking_max=float(np.nanmax(np.abs(v[drive] - plan_v[drive]))),
            rise_90=first_time(t, (t >= HOLD_START) & (sign * v >= 0.9 * abs(v_cmd))) - HOLD_START,
            overshoot=float(max(sign * v[drive].max() if sign > 0 else -v[drive].min(), 0.0) / abs(v_cmd) - 1.0),
            start_reverse=float(max(-sign * v[(t >= HOLD_START) & (t < HOLD_START + 0.5)].min() * 1.0, 0.0)
                                if sign > 0 else max(v[(t >= HOLD_START) & (t < HOLD_START + 0.5)].max(), 0.0)),
            stop_time=first_time(t, after & (np.abs(v) < STOP_SPEED) & (np.abs(pitch) < 0.01)) - HOLD_END,
            stop_distance=float(abs(s[after][-1] - s[np.searchsorted(t, HOLD_END)])),
            brake_reverse=float(max(-sign * v[after].min(), 0.0) if sign > 0 else max(v[after].max(), 0.0)),
        )
    if w_cmd:
        out.update(
            cruise_w_mean=float(w[cruise].mean()), cruise_w_error=float(w[cruise].mean() - w_cmd),
            plan_w_tracking_max=float(np.nanmax(np.abs(w[drive] - plan_w[drive]))),
            rise_w_90=first_time(t, (t >= HOLD_START) & (w >= 0.9 * w_cmd)) - HOLD_START,
            final_heading=float(records[-1]["yaw"]),
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["nullspace", "wls"], choices=["nullspace", "wls", "qp"])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    RESULTS.mkdir(exist_ok=True)
    revision = circleg_revision()
    runs = []
    for backend in args.backends:
        for case in args.cases:
            v_cmd, w_cmd = CASES[case]
            sim = Gbc486MpcWbcSimulation(SimulationConfig(backend=backend))
            summary = sim.run(make_command(v_cmd, w_cmd), DURATION)
            summary.pop("model")
            summary.update(case=case, v_cmd=v_cmd, w_cmd=w_cmd, metrics=metrics(sim.records, v_cmd, w_cmd),
                           planner_limits=dict(max_speed=sim.mpc.settings.limits.max_speed,
                                               commanded_acceleration=sim.mpc.settings.limits.commanded_acceleration,
                                               max_braking=sim.mpc.settings.limits.max_braking,
                                               max_yaw_rate=sim.mpc.settings.limits.max_yaw_rate,
                                               max_yaw_acceleration=sim.mpc.settings.limits.max_yaw_acceleration))
            runs.append(summary)
            with gzip.open(RESULTS / f"m4_{backend}_{case}.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(sim.records, handle)
            print(f"{backend:9s} {case:9s} completed={summary['completed']} stop={summary['stop_reason']} "
                  f"t={summary['simulated']:.2f}", flush=True)
    payload = dict(mujoco=mujoco.__version__, circleg=revision, runs=runs)
    (RESULTS / "m4_command_tracking.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    text = to_markdown(payload)
    (RESULTS / "m4_command_tracking.md").write_text(text, encoding="utf-8")
    print(text)


def to_markdown(p) -> str:
    def f(v, n=2, scale=1.0):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * scale:.{n}f}"
    lim = p["runs"][0]["planner_limits"]
    lines = [
        "# M4：速度 / 偏航指令阶跃跟踪", "",
        f"脚本：`tools/m4_command_tracking.py`（mujoco {p['mujoco']}；circleg HEAD {p['circleg'].get('head')}，"
        f"`controller.py` = {p['circleg'].get('imported_sha256', {}).get('Algorithm/mpc/controller.py')}，"
        f"`planner.py` = {p['circleg'].get('imported_sha256', {}).get('Algorithm/mpc/planner.py')}）。",
        f"指令在 t ∈ [{HOLD_START}, {HOLD_END}) s 内保持，其余为 0，总时长 {DURATION} s。"
        f"规划限值（circleg 默认）：{lim}。**未调参**。", "",
        "## 1. 速度阶跃", "",
        f"稳态窗 t ∈ [{CRUISE_WINDOW[0]}, {CRUISE_WINDOW[1]}) s；规划跟踪 = 实测 v 与 planner 参考 v 之差（指令期间）；"
        f"停车 = 指令归零后 |v| < {STOP_SPEED} m/s 且 |θ| < 10 mrad 的首个时刻。", "",
        "| 后端 | 工况 | 完成 | 停机 | 稳态 v / 误差 m/s | 稳态纹波 mm/s | 规划跟踪 RMS / max m/s | 90% 上升 s | 超调 |"
        " 起步反向 mm/s | 停车 s / 距离 m | 刹车反向 mm/s | max\\|θ\\| ° | max 轮力矩 N·m（饱和拍数） | MPC relaxed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in p["runs"]:
        if not r["v_cmd"]:
            continue
        m = r["metrics"]
        lines.append(
            f"| {r['backend']} | {r['case']} | {'是' if r['completed'] else '否'} | {r['stop_reason'] or '—'} | "
            f"{f(m.get('cruise_v_mean'), 3)} / {f(m.get('cruise_v_error'), 3)} | {f(m.get('cruise_v_ripple'), 1, 1e3)} | "
            f"{f(m.get('plan_tracking_rms'), 3)} / {f(m.get('plan_tracking_max'), 3)} | {f(m.get('rise_90'))} | "
            f"{f(m.get('overshoot'), 1, 100)}% | {f(m.get('start_reverse'), 0, 1e3)} | "
            f"{f(m.get('stop_time'))} / {f(m.get('stop_distance'), 3)} | {f(m.get('brake_reverse'), 0, 1e3)} | "
            f"{f(m['max_abs_pitch'], 1, 57.29578)} | {f(m['max_wheel_torque'])}（{m['wheel_torque_saturated_steps']}） | "
            f"{r['mpc_relaxed']} |")
    lines += ["", "## 2. 偏航阶跃", "",
              "| 后端 | 工况 | 完成 | 停机 | 稳态 ψ̇ / 误差 rad/s | 规划跟踪 max rad/s | 90% 上升 s | 末航向 rad | "
              "max\\|θ\\| ° | max 轮力矩 N·m | MPC relaxed |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in p["runs"]:
        if not r["w_cmd"]:
            continue
        m = r["metrics"]
        lines.append(
            f"| {r['backend']} | {r['case']} | {'是' if r['completed'] else '否'} | {r['stop_reason'] or '—'} | "
            f"{f(m.get('cruise_w_mean'), 3)} / {f(m.get('cruise_w_error'), 3)} | {f(m.get('plan_w_tracking_max'), 3)} | "
            f"{f(m.get('rise_w_90'))} | {f(m.get('final_heading'), 3)} | {f(m['max_abs_pitch'], 1, 57.29578)} | "
            f"{f(m['max_wheel_torque'])} | {r['mpc_relaxed']} |")
    lines += ["", "## 3. 通用统计", "",
              "| 后端 | 工况 | MPC fallback | WBC fallback / clip / infeasible | max 腿力矩 N·m | 腿长偏差 max mm | "
              "腿角偏差 max mrad | max\\|roll\\| mrad | MPC / WBC p50 ms |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in p["runs"]:
        m, tm = r["metrics"], r["timing_ms"]
        lines.append(
            f"| {r['backend']} | {r['case']} | {r['mpc_fallbacks']} | {r['wbc_fallback']} / {r['wbc_clip']} / "
            f"{r['wbc_infeasible']} | {f(m['max_leg_torque'])} | {f(m['max_leg_length_dev'], 2, 1e3)} | "
            f"{f(m['max_leg_angle_dev'], 1, 1e3)} | {f(r['max_abs_roll'], 2, 1e3)} | "
            f"{tm['mpc']['p50']:.2f} / {tm['wbc']['p50']:.2f} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
