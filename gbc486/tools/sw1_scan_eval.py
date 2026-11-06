"""SW1 二维扫描评价工具：FLIGHT_RETRACT_FLOOR x FLIGHT_RETRACT_TORQUE_LIMIT。

只读 results/*.json.gz，不改变任何控制行为。评价口径见部署文档 SW1 第四节。

用法：
    python tools/sw1_scan_eval.py [--results results]
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path

_LOGGER = logging.getLogger("sw1.eval")

LEG_LENGTH_NOMINAL = 0.28            # [m] 标称腿长
LEG_LENGTH_MIN = 0.114               # [m] 机械下限
LEG_LENGTH_MAX = 0.430               # [m] 机械上限
WHEEL_TORQUE_LIMIT = 6.0             # [N*m] 轮执行器 ctrlrange ±6
HOLD_SECONDS = 0.2                   # [s] T_settle 连续保持时长
PITCH_SETTLE = 0.10                  # [rad] T_settle 俯仰判据
LEG_SETTLE = 0.02                    # [m] T_settle 腿长判据

STAGES = [
    ("SW1_f016_t36", 0.16, 36.0), ("SW1_f016_t38", 0.16, 38.0), ("SW1_f016_t40", 0.16, 40.0),
    ("SW1_f018_t36", 0.18, 36.0), ("SW1_f018_t38", 0.18, 38.0), ("SW1_f018_t40", 0.18, 40.0),
    ("SW1_f019_t36", 0.19, 36.0), ("SW1_f019_t38", 0.19, 38.0), ("SW1_f019_t40", 0.19, 40.0),
    ("SW1_f020_t36", 0.20, 36.0), ("SW1_f020_t38", 0.20, 38.0), ("SW1_f020_t40", 0.20, 40.0),
]


def phase(frame):
    jump = frame.get("jump")
    return None if jump is None else jump.get("phase")


def leg_ref(frame):
    jump = frame.get("jump")
    return None if jump is None else jump.get("flight_leg_ref")


def wheel_gap(frame):
    value = frame["wheel_gap"]
    return max(value) if isinstance(value, list) else float(value)


def landing_first(frames):
    previous = None
    for frame in frames:
        if previous == "FLIGHT" and phase(frame) == "LANDING":
            return frame
        previous = phase(frame)
    return None


def settle_time(after, t_land):
    """从触地首帧起，到|pitch|<=0.10 且两腿长在0.28±0.02 连续保持0.2s 为止的时间。"""
    dt = after[1]["t"] - after[0]["t"] if len(after) > 1 else 0.001
    need = max(1, int(round(HOLD_SECONDS / dt)))
    ok = [abs(f["pitch"]) <= PITCH_SETTLE
          and abs(f["leg_length_R"] - LEG_LENGTH_NOMINAL) <= LEG_SETTLE
          and abs(f["leg_length_L"] - LEG_LENGTH_NOMINAL) <= LEG_SETTLE
          for f in after]
    for i in range(0, len(ok) - need + 1):
        if all(ok[i:i + need]):
            return round(after[i + need - 1]["t"] - t_land, 4)
    return 999.0


def evaluate(run):
    frames = run["frames"]
    summary = run["summary"]
    flight = [f for f in frames if phase(f) == "FLIGHT"]
    rising = [f for f in flight if leg_ref(f) is not None and leg_ref(f)[0] == "rising"]

    gap_flight = max(wheel_gap(f) for f in flight)
    gap_all = float(summary["max_positive_wheel_gap_m"])
    min_l_r = min(f["leg_length_R"] for f in flight)
    min_l_l = min(f["leg_length_L"] for f in flight)

    land = landing_first(frames)
    after = [f for f in frames if land is not None and f["t"] >= land["t"]]
    max_pitch_after = max(abs(f["pitch"]) for f in after) if after else float("nan")
    pitch_final = after[-1]["pitch"] if after else float("nan")
    leg_r = [f["leg_length_R"] for f in after]
    leg_l = [f["leg_length_L"] for f in after]
    pp_r = (max(leg_r) - min(leg_r)) if after else float("nan")
    pp_l = (max(leg_l) - min(leg_l)) if after else float("nan")
    violation = bool(after) and (min(min(leg_r), min(leg_l)) < LEG_LENGTH_MIN
                                 or max(max(leg_r), max(leg_l)) > LEG_LENGTH_MAX)
    t_settle = settle_time(after, land["t"]) if after else 999.0

    n39 = sum(1 for f in rising for ch in range(4) if abs(f["ctrl"][ch]) >= 39.0)
    wheel_sat = sum(1 for f in frames
                    if abs(f["ctrl"][4]) >= WHEEL_TORQUE_LIMIT - 1e-9
                    or abs(f["ctrl"][5]) >= WHEEL_TORQUE_LIMIT - 1e-9)

    min_l_ref = min(leg_ref(f)[1] for f in rising) if rising else float("nan")
    max_ctrl = max((max(abs(f["ctrl"][ch]) for ch in range(4)) for f in rising), default=float("nan"))

    abort = int(summary["phase_frame_counts"].get("ABORT", 0))
    h1 = gap_flight >= 0.200
    h2 = abort == 0
    h3 = not violation
    h4 = max_pitch_after < 1.0
    return {
        "gap_flight": round(gap_flight, 5), "gap_all": round(gap_all, 5),
        "min_l_r": round(min_l_r, 5), "min_l_l": round(min_l_l, 5),
        "max_pitch_after": round(max_pitch_after, 5), "pitch_final": round(pitch_final, 5),
        "pp_r": round(pp_r, 5), "pp_l": round(pp_l, 5), "pp_max": round(max(pp_r, pp_l), 5),
        "t_settle": t_settle, "abort": abort, "violation": violation,
        "n39": n39, "wheel_sat": wheel_sat,
        "min_l_ref": round(min_l_ref, 5), "max_ctrl": round(max_ctrl, 4),
        "H1": h1, "H2": h2, "H3": h3, "H4": h4, "pass_all": h1 and h2 and h3 and h4,
    }


def load(stage, results):
    path = Path(results) / ("%s_baseline_run01.json.gz" % stage)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--smoke", default=None,
                        help="只评价单份结果（如 RAMP1），用于工具冒烟")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.smoke:
        metrics = evaluate(load(args.smoke, args.results))
        _LOGGER.info("smoke %s: %s", args.smoke, json.dumps(metrics, ensure_ascii=False))
        return

    rows = []
    for stage, floor, tlim in STAGES:
        metrics = evaluate(load(stage, args.results))
        metrics.update({"stage": stage, "floor": floor, "tlim": tlim})
        rows.append(metrics)
        _LOGGER.info("%s floor=%.2f tlim=%.0f %s", stage, floor, tlim,
                     json.dumps(metrics, ensure_ascii=False))

    passed = [r for r in rows if r["pass_all"]]
    rank = {r["stage"]: index + 1 for index, r in enumerate(sorted(
        passed, key=lambda r: (r["max_pitch_after"], r["pp_max"], r["t_settle"],
                               r["n39"], -r["gap_flight"])))}
    _LOGGER.info("ranking (passing only): %s", json.dumps(rank, ensure_ascii=False))
    print(json.dumps({"rows": rows, "rank": rank}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
