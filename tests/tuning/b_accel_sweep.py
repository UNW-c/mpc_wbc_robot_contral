"""第 24 次排查：加减速机动性 M1 / B1 / B2 扫描（V4：0 -> 2.5 -> 0 -> -2.5 -> 0，每段 4 s）。

在当前默认（tuned-fast-accel）基础上只改：
- M1：pitch_ff_rate_limit（倾角参考变化率）、jerk_ramp_time（起步 jerk 爬升时间）
- B1：brake_pitch_max（刹车后仰上限，刹车限值按该倾角能力表重算）
- B2：brake_jerk_ramp_time（刹车 jerk 爬升时间）

新增指标 t_tilt80：起步指令后，机身前倾第一次达到"本次前馈峰值 80%"的时间。
用法：.venv/Scripts/python.exe tests/tuning/b_accel_sweep.py [配置名 ...] [--zero]
输出只用 ASCII。
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation, SimulationConfig, default_app_mpc_settings  # noqa: E402
from tests.tuning import a1_acceptance as a1  # noqa: E402

CONFIGS = {
    "base": {},
    "M1": dict(pitch_ff_rate_limit=3.0, jerk_ramp_time=0.15),
    "B1-20": dict(brake_pitch_max=np.radians(20.0)),
    "B1-25": dict(brake_pitch_max=np.radians(25.0)),
    "B2": dict(brake_jerk_ramp_time=0.12),
    "M1+B1-20+B2": dict(pitch_ff_rate_limit=3.0, jerk_ramp_time=0.15,
                        brake_pitch_max=np.radians(20.0), brake_jerk_ramp_time=0.12),
    "M1+B1-25+B2": dict(pitch_ff_rate_limit=3.0, jerk_ramp_time=0.15,
                        brake_pitch_max=np.radians(25.0), brake_jerk_ramp_time=0.12),
}


def settings_for(name: str):
    return replace(default_app_mpc_settings(), **CONFIGS[name])


def run_v4(name: str, max_speed: float = 2.5) -> dict:
    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(), mpc_settings=settings_for(name))

    def command(t):
        v = max_speed if t < 4.0 else 0.0 if t < 8.0 else -max_speed if t < 12.0 else 0.0
        return MotionCommand(velocity=v, yaw_rate=0.0)

    summary = sim.run(command, 16.0, None, None, allow_driving=True)
    t = np.array([r["time"] for r in sim.records])
    st = np.array([r["state"] for r in sim.records])
    v, x, th = st[:, 1], st[:, 0], st[:, 2]
    hip = np.array([sim._hip_common(np.array(r["qpos"])[sim.hip_q]) for r in sim.records])

    def first(mask):
        return float(t[np.argmax(mask)]) if mask.any() else float("nan")

    rise = first((t >= 0.0) & (v >= 0.9 * max_speed))
    stop = first((t >= 4.0) & (t <= 8.0) & (np.abs(v) < 0.05)) - 4.0
    i_b = int(np.argmin(np.abs(t - 4.0)))
    i_s = int(np.argmin(np.abs(t - (4.0 + stop)))) if np.isfinite(stop) else i_b
    dist = float(abs(x[i_s] - x[i_b]))
    # 前倾到位：起步段（0~2 s）机身俯仰为正（前倾），取峰值的 80%
    up = (t >= 0.0) & (t <= 2.0)
    peak_up = float(np.max(th[up]))
    t_tilt80 = first(up & (th >= 0.8 * peak_up)) if peak_up > np.radians(1.0) else float("nan")
    # 刹车后仰到位（4~6 s 内俯仰为负）
    down = (t >= 4.0) & (t <= 6.0)
    peak_dn = float(np.min(th[down]))
    t_btilt80 = (first(down & (th <= 0.8 * peak_dn)) - 4.0) if peak_dn < -np.radians(1.0) else float("nan")
    # 起步时髋先动：髋到 50% 峰值 与 机身到 50% 峰值的时间差
    hip_peak = float(np.max(np.abs(hip[up])))
    t_hip50 = first(up & (np.abs(hip) >= 0.5 * hip_peak))
    t_th50 = first(up & (th >= 0.5 * peak_up))
    # 刹车结束后的位置 / 速度残余（停住后 1 s 内）
    after = (t >= 4.0 + (stop if np.isfinite(stop) else 4.0)) & (t <= 8.0)
    limits = sim.mpc.settings.limits
    return dict(
        name=name, up=limits.commanded_acceleration, down=limits.max_braking,
        jerk=limits.max_jerk, bjerk=limits.max_braking_jerk or limits.max_jerk,
        t_rise=rise, t_stop=stop, dist=dist,
        t_tilt80=t_tilt80, t_btilt80=t_btilt80, lag_th_hip=t_th50 - t_hip50,
        pitch_up=np.degrees(peak_up), pitch_dn=np.degrees(peak_dn),
        pitch_peak=float(np.degrees(np.max(np.abs(th)))), hip_peak=float(np.max(np.abs(hip))),
        v_after=float(np.max(np.abs(v[after]))) if after.any() else float("nan"),
        stop_reason=summary["stop_reason"],
        fb="%d/%d" % (summary["mpc_fallbacks"], summary["wbc_fallbacks"]),
    )


def main() -> int:
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(CONFIGS)
    print("| cfg | up/down | jerk/bjerk | t_rise | t_stop | dist | t_tilt80 | t_btilt80 | th-hip lag | "
          "pitch +/- | |pitch|max | hip max | v_after | stop | fb |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        r = run_v4(n)
        print("| %s | %.2f/%.2f | %.1f/%.1f | %.2f | %.2f | %.3f | %.3f | %.3f | %+.3f | %+.1f/%+.1f | %.1f | %.3f | %.3f | %s | %s |" % (
            n, r["up"], r["down"], r["jerk"], r["bjerk"], r["t_rise"], r["t_stop"], r["dist"],
            r["t_tilt80"], r["t_btilt80"], r["lag_th_hip"], r["pitch_up"], r["pitch_dn"], r["pitch_peak"],
            r["hip_peak"], r["v_after"], r["stop_reason"], r["fb"]), flush=True)
    if "--zero" in sys.argv:
        print()
        print("| zero | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for case in ("H1", "H2", "H3", "H4", "Tf"):
            cells = []
            for n in names:
                z = a1._run_zero_input(settings_for(n), SimulationConfig(), case)
                cells.append("%.2f/%d/%.1f/%s" % (z["settle"], z["crossings"], z["pitch_peak"], z.get("reason")))
            print("| %s | %s |" % (case, " | ".join(cells)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
