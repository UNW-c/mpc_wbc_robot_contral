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
# 第 26 次排查 S1：在 tuned-agile 基础上继续压
from APP.mujoco_mpc_wbc import _PRESETS  # noqa: E402
_AGILE = dict(_PRESETS["tuned-agile"]["mpc_overrides"])
CONFIGS.update({
    "agile": _AGILE,
    "S1a-bj08": {**_AGILE, "brake_jerk_ramp_time": 0.08},
    "S1a-bj06": {**_AGILE, "brake_jerk_ramp_time": 0.06},
    "S1b-up20": {**_AGILE, "pitch_feedforward_max": np.radians(20.0)},
    "S1ab": {**_AGILE, "brake_jerk_ramp_time": 0.08, "pitch_feedforward_max": np.radians(20.0)},
    "S1-b22": {**_AGILE, "brake_pitch_max": np.radians(22.0)},
})
_S1 = {**_AGILE, "brake_jerk_ramp_time": 0.08, "pitch_feedforward_max": np.radians(20.0)}
CONFIGS.update({
    # S2 探底：单点线性化（q=0）下直接放宽髋角限值，看模型能否撑住
    "S2-q0.9": {**_S1, "hip_angle_limit": 0.9},
    "S2-q1.0": {**_S1, "hip_angle_limit": 1.0},
    "S2-q1.1": {**_S1, "hip_angle_limit": 1.1},
    "S2-q1.2": {**_S1, "hip_angle_limit": 1.2},
    # 第 27 次排查 方案 B：分模式髋限（驾驶 1.0 / 1.2，零指令 0.8）
    "B-d1.0": {**_S1, "drive_hip_angle_limit": 1.0},
    "B-d1.2": {**_S1, "drive_hip_angle_limit": 1.2},
})
CONFIGS["agile-v2"] = dict(_PRESETS["tuned-agile"]["mpc_overrides"])  # 更新后的 preset 本身
# 第 28 次排查：显式定义（不依赖 preset 当前内容），避免对比被 preset 更新污染
_V1 = dict(pitch_ff_rate_limit=3.0, jerk_ramp_time=0.15, brake_pitch_max=np.radians(20.0), brake_jerk_ramp_time=0.12)
CONFIGS.update({
    "v1-agile": _V1,                                                            # 第 24 次 tuned-agile
    "v1-S1ab": {**_V1, "brake_jerk_ramp_time": 0.08, "pitch_feedforward_max": np.radians(20.0)},
    "v1-M1": dict(pitch_ff_rate_limit=3.0, jerk_ramp_time=0.15),
    "v1-B1": dict(brake_pitch_max=np.radians(20.0)),
    "v1-B2": dict(brake_jerk_ramp_time=0.12),
})
# 第 28 次排查诊断：在新 tuned-agile 上只改刹车 jerk / 倾角速率
_AG2 = dict(_PRESETS["tuned-agile"]["mpc_overrides"])
CONFIGS.update({
    "d-bj12": {**_AG2, "brake_jerk_ramp_time": 0.12},
    "d-bj25": {**_AG2, "brake_jerk_ramp_time": 0.25},
    "d-bjNone": {**_AG2, "brake_jerk_ramp_time": None},
    "d-ff1": {**_AG2, "pitch_ff_rate_limit": 1.0},
})
# 第 28 次排查修复：R2'（unwind / 髋力矩变化率）、R1'（软复位）；R4 在 APP 默认开启（检测器配置）
CONFIGS.update({
    "r-unwind": {**_AG2, "unwind_before_brake": True},
    "r-soft": {**_AG2, "preserve_ramp_on_recontact": True},
    "r-all": {**_AG2, "unwind_before_brake": True, "preserve_ramp_on_recontact": True},
})


CONFIGS.update({
    # 单跑 R4 的对照（第 28 次排查：判据修正后仍无收益，默认关闭）
    "r-r4": _AG2,
})


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
        tau_hip=float(np.max(np.abs([r["torque"][0] for r in sim.records]))),
        stop_reason=summary["stop_reason"],
        fb="%d/%d" % (summary["mpc_fallbacks"], summary["wbc_fallbacks"]),
    )


def main() -> int:
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(CONFIGS)
    print("| cfg | up/down | jerk/bjerk | t_rise | t_stop | dist | t_tilt80 | t_btilt80 | th-hip lag | "
          "pitch +/- | |pitch|max | hip max | tau_hip | v_after | stop | fb |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        r = run_v4(n)
        print("| %s | %.2f/%.2f | %.1f/%.1f | %.2f | %.2f | %.3f | %.3f | %.3f | %+.3f | %+.1f/%+.1f | %.1f | %.3f | %.2f | %.3f | %s | %s |" % (
            n, r["up"], r["down"], r["jerk"], r["bjerk"], r["t_rise"], r["t_stop"], r["dist"],
            r["t_tilt80"], r["t_btilt80"], r["lag_th_hip"], r["pitch_up"], r["pitch_dn"], r["pitch_peak"],
            r["hip_peak"], r["tau_hip"], r["v_after"], r["stop_reason"], r["fb"]), flush=True)
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
