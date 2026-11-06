"""驱动工况 A/B：验证"速度输入 + ZMP"，并校核 planner 手填的加减速限值。

用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/drive_ab.py --workers 14 --cases D-a D-b D-c D-d D-e

口径（与零指令受扰的 T-a..T-f 分开）：
- 该脚本**不使用** TIER1_CONSTRAINTS 的 pitch 硬约束（1.0° 是受扰口径）；
  驱动按设计就要前倾，`split=0.3`、a=1.2 m/s² 对应俯仰参考约 4.9°。
  这里只报指标 + 是否触发保护，不做"通过/不通过"裁剪。
- 所有配置都以 `objective.TUNED` 为基底，只覆盖列出的字段，不改 TUNED。
"""
from __future__ import annotations

# 必须在导入 numpy 之前设好（与 tune_grid.py 同规则：worker 内 BLAS 固定单线程）
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gbc_mpc_wbc.objective import (DRIVE_CASES, DRIVE_HIP_ANGLE_LIMIT, DRIVE_ORDER,   # noqa: E402
                                   TUNED, evaluate_drive)

# 配置：名称 -> 覆盖到 MPCSettings 的字段（空 dict = 当前部署）
CONFIGS = {
    "部署(ZMP开 split0.3 驱动预算%.2f)" % DRIVE_HIP_ANGLE_LIMIT: dict(),
    "ZMP关": dict(zmp_reference=False),
    "split=0.0": dict(zmp_split=0.0),
    "split=0.6": dict(zmp_split=0.6),
    "split=1.0": dict(zmp_split=1.0),
    "驱动预算=0.125(不放开)": dict(drive_hip_angle_limit=0.125),
}

_FIELDS = ("case", "config", "stop", "guard", "speed_peak", "v_err_steady", "rise_time_s",
           "a_peak_up", "a_peak_down", "pitch_steady_deg", "pitch_peak_deg", "leg_peak_mrad",
           "wheel_occ", "leg_occ", "drift_mm", "fallback", "infeasible")
_HEADER = ("工况", "配置", "保护", "速度峰", "稳态v误差", "上升时间s", "加速峰", "减速峰",
           "稳态俯仰°", "俯仰峰°", "腿角峰mrad", "轮占用", "腿占用", "末端位移mm", "回退", "不可行")


def _worker(payload):
    config_name, overrides, case = payload
    report = evaluate_drive(TUNED, cases=[case], **overrides)
    return config_name, case, report[case]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--cases", nargs="*", default=list(DRIVE_ORDER))
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "drive")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(name, dict(overrides), case)
             for name, overrides in CONFIGS.items() for case in args.cases]
    print(f"驱动 A/B：{len(CONFIGS)} 配置 × {len(args.cases)} 工况 = {len(tasks)} 次")
    for name, overrides in CONFIGS.items():
        print(f"   {name}  覆盖={overrides}")

    from multiprocessing import Pool

    results = {}
    start = time.perf_counter()
    with Pool(args.workers) as pool:
        for index, (config_name, case, metric) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            results[(config_name, case)] = metric
            elapsed = time.perf_counter() - start
            print(f"  {index}/{len(tasks)}  {elapsed:.0f}s  预计剩余 "
                  f"{elapsed / index * (len(tasks) - index):.0f}s", flush=True)

    lines = ["# 驱动工况 A/B（速度输入 + ZMP）", "",
             "基底 = `objective.TUNED`（`hip_angle_limit=%.3f`）；驱动段预算 = `DRIVE_HIP_ANGLE_LIMIT=%.2f`。"
             % (TUNED["hip_angle_limit"], DRIVE_HIP_ANGLE_LIMIT),
             "不施加 TIER1 的 pitch 硬约束（那是零指令受扰口径）。", "",
             "| " + " | ".join(_HEADER) + " |",
             "|" + "---|" * len(_HEADER)]
    for config_name in CONFIGS:
        for case in args.cases:
            metric = results[(config_name, case)]
            values = [
                case, config_name, metric["stop"] or "—", metric["guard"] or "—",
                f"{metric['speed_peak']:.3f}", f"{metric['v_err_steady']:.4f}",
                f"{metric['rise_time_s']:.3f}", f"{metric['a_peak_up']:+.3f}",
                f"{metric['a_peak_down']:+.3f}", f"{metric['pitch_steady_deg']:.3f}",
                f"{metric['pitch_peak_deg']:.3f}", f"{metric['leg_peak_mrad']:.1f}",
                f"{metric['wheel_occ']:.3f}", f"{metric['leg_occ']:.3f}",
                f"{metric['drift_mm']:+.1f}", metric["fallback"], metric["infeasible"],
            ]
            lines.append("| " + " | ".join(str(v) for v in values) + " |")
    lines += ["", "## 刹车段（释放指令后）", "",
              "| 配置 | 工况 | 释放时刻 s | 越冲 mm | 刹车俯仰峰 deg | 停止用时 s |",
              "|---|---|---|---|---|---|"]
    for config_name in CONFIGS:
        for case in args.cases:
            for brake in results[(config_name, case)]["brakes"]:
                lines.append("| %s | %s | %.2f | %.1f | %.3f | %s |" % (
                    config_name, case, brake["release"], brake["overshoot_mm"],
                    brake["brake_pitch_peak_deg"],
                    "—" if brake["settle_s"] != brake["settle_s"] else f"{brake['settle_s']:.2f}"))
    report = out / "drive_ab.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {report}")


if __name__ == "__main__":
    main()
