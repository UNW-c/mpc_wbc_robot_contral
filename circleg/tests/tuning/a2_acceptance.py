"""A2（第 15 次排查）验收：V4 下 jerk 开/关 × 髋角前馈开/关 四组对比。

复用 a1_acceptance 的工况与指标（tuned-fast、A1 自动限值开启），另加：
- 髋角过冲：起步段髋角峰值相对"斜坡中段髋角均值"的超出比例；
- 刹车段同样计算；
- 规划加速度的最大变化率（验证 S 曲线生效）。

用法（项目根）：
    .venv/Scripts/python.exe tests/tuning/a2_acceptance.py [--zero]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.tuning import a1_acceptance as a1  # noqa: E402
from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation  # noqa: E402


def settings(jerk: bool, feedforward: bool):
    base = a1._settings(True)
    return replace(base, limits=replace(base.limits, use_jerk_limit=jerk),
                   hip_reference_feedforward=feedforward)


def _overshoot(time, hip, t0, t1):
    """斜坡段 [t0, t1] 内：峰值 / 中段（40%~80% 时刻）均值 − 1；中段均值过小则返回 nan。"""
    mask = (time >= t0) & (time <= t1)
    if mask.sum() < 10:
        return float("nan")
    seg_t, seg = time[mask], np.abs(hip[mask])
    mid = (seg_t >= t0 + 0.4 * (t1 - t0)) & (seg_t <= t0 + 0.8 * (t1 - t0))
    plateau = float(np.mean(seg[mid])) if mid.any() else float("nan")
    if not np.isfinite(plateau) or plateau < 0.05:
        return float("nan")
    return float(np.max(seg) / plateau - 1.0)


def run_v4(jerk: bool, feedforward: bool) -> dict:
    return run_v4_with(settings(jerk, feedforward))


def run_v4_with(s) -> dict:
    max_speed = float(s.limits.max_speed)
    sim_holder = {}
    original = a1.MuJoCoMpcWbcSimulation

    class _Capture(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sim_holder["sim"] = self

    a1.MuJoCoMpcWbcSimulation = _Capture
    try:
        row = a1._run_drive(s, a1._config(), "V4", max_speed)
    finally:
        a1.MuJoCoMpcWbcSimulation = original
    sim = sim_holder["sim"]
    time = np.array([r["time"] for r in sim.records])
    hip = np.array([sim._hip_common(np.array(r["qpos"])[sim.hip_q]) for r in sim.records])
    t_rise, t_stop = row["t_rise"], row["t_stop"]
    row["hip_over_up"] = _overshoot(time, hip, 0.0, t_rise if np.isfinite(t_rise) else 4.0)
    row["hip_over_down"] = _overshoot(time, hip, 4.0, 4.0 + (t_stop if np.isfinite(t_stop) else 4.0))
    # 回位后的髋角残余摆动（斜坡结束到段末 1 s 内的髋角峰值，衡量"甩回 0"时的过冲）
    after = (time >= 4.0 + (t_stop if np.isfinite(t_stop) else 4.0)) & (time <= 8.0)
    row["hip_after_stop"] = float(np.max(np.abs(hip[after]))) if after.any() else float("nan")
    return row


def main() -> int:
    print("| jerk | 前馈 | 规划(起/刹) | t_rise | t_stop | 距离/理论 | 起步兑现 | 刹车兑现 | 髋峰 | 起步过冲 | 刹车过冲 | 停后髋残余 | 俯仰峰 | 停止 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for jerk in (False, True):
        for ff in (False, True):
            r = run_v4(jerk, ff)
            print(f"| {'开' if jerk else '关'} | {'开' if ff else '关'} | {r['planned_up']:.2f}/{r['planned_down']:.2f} "
                  f"| {r['t_rise']:.2f} | {r['t_stop']:.2f} | {r['distance']:.3f}/{r['theory']:.3f} "
                  f"| {r['up_ratio']:.2f} | {r['down_ratio']:.2f} | {r['hip_peak']:.3f} "
                  f"| {r['hip_over_up']:+.3f} | {r['hip_over_down']:+.3f} | {r['hip_after_stop']:.3f} "
                  f"| {r['pitch_peak']:.2f}° | {r['stop_reason']} |", flush=True)
    if "--zero" in sys.argv:
        print()
        print("| 工况 | jerk+前馈 关 | jerk+前馈 开 |")
        print("|---|---|---|")
        for name in ("H1", "H2", "H3", "H4", "Tf"):
            cells = []
            for on in (False, True):
                z = a1._run_zero_input(settings(on, on), a1._config(), name)
                cells.append(f"{z['settle']:.2f} / {z['crossings']} / {z['pitch_peak']:.1f}° / {z['stop_reason']}")
            print(f"| {name} | {cells[0]} | {cells[1]} |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
