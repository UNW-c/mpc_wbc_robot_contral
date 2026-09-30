"""A3（第 16 次排查）验收：V4 下 A2（tuned-fast-accel）与 A3a（+ 倾角辅助限值）对比，另跑零输入回归。

用法（项目根）：.venv/Scripts/python.exe tests/tuning/a3_acceptance.py [--zero]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.tuning import a1_acceptance as a1  # noqa: E402
from tests.tuning.a2_acceptance import run_v4_with  # noqa: E402


def settings(tilt: bool, pitch_ff: bool = False):
    base = a1._settings(True)
    extra = {"pitch_feedforward": True} if pitch_ff else {}
    return replace(base, limits=replace(base.limits, use_jerk_limit=True),
                   tilt_assisted_limits=tilt, **extra)


def main() -> int:
    rows = [("A2（S 曲线）", settings(False)), ("A3a（+倾角辅助限值）", settings(True))]
    if "--ff" in sys.argv:
        rows.append(("A3b（+倾角参考）", settings(True, True)))
    print("| 配置 | 规划(起/刹) | t_rise | t_stop | 距离/理论 | 起步兑现 | 刹车兑现 | 髋峰 | 起步髋过冲 | 停后髋残余 | 俯仰峰 | 停止 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in rows:
        r = run_v4_with(s)
        print(f"| {name} | {r['planned_up']:.2f}/{r['planned_down']:.2f} | {r['t_rise']:.2f} | {r['t_stop']:.2f} "
              f"| {r['distance']:.3f}/{r['theory']:.3f} | {r['up_ratio']:.2f} | {r['down_ratio']:.2f} "
              f"| {r['hip_peak']:.3f} | {r['hip_over_up']:+.3f} | {r['hip_after_stop']:.3f} "
              f"| {r['pitch_peak']:.2f}° | {r['stop_reason']} |", flush=True)
    if "--zero" in sys.argv:
        print()
        for name in ("H1", "H2", "H3", "H4", "Tf"):
            cells = []
            for s in (settings(False), settings(True, "--ff" in sys.argv)):
                z = a1._run_zero_input(s, a1._config(), name)
                cells.append(f"{z['settle']:.2f} / {z['crossings']} / {z['pitch_peak']:.2f}° / {z['reason']}")
            print(f"| {name} | {cells[0]} | {cells[1]} |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
