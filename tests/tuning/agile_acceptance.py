"""第 28 次排查收尾验收：以真实 preset 为准（tuned-agile vs 默认），覆盖
中途松键（震荡工况）、V4 加减速、零输入抗扰、跳跃。只输出 ASCII。

用法：.venv/Scripts/python.exe tests/tuning/agile_acceptance.py [--preset NAME]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import (MuJoCoMpcWbcSimulation, SimulationConfig,  # noqa: E402
                                _build_from_args, default_app_mpc_settings)
from tests.tuning import a1_acceptance as a1  # noqa: E402


def build(preset: str | None):
    if preset is None:
        return SimulationConfig(), default_app_mpc_settings()
    config, settings, _ = _build_from_args(["--preset", preset])
    return config, settings


def mid_release(config, settings, t_release, start=1.0, v_cmd=2.5, duration=5.0):
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
    command = lambda t: MotionCommand(  # noqa: E731
        velocity=v_cmd if start <= t < start + t_release else 0.0, yaw_rate=0.0)
    summary = sim.run(command, duration, None, None, allow_driving=True)
    time = np.array([r["time"] for r in sim.records])
    state = np.array([r["state"] for r in sim.records])
    contact = np.array([r["contact"] for r in sim.records])
    hip = np.array([sim._hip_common(np.array(r["qpos"])[sim.hip_q]) for r in sim.records])
    mask = time >= start
    return dict(air=int((~contact[mask].all(axis=1)).sum()),
                pitch=float(np.degrees(np.abs(state[mask, 2]).max())),
                hip=float(np.abs(hip[mask]).max()),
                wbc_fb=summary["wbc_fallbacks"], stop=summary["stop_reason"])


def v4(config, settings, max_speed=2.5):
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)

    def command(t):
        v = max_speed if t < 4.0 else 0.0 if t < 8.0 else -max_speed if t < 12.0 else 0.0
        return MotionCommand(velocity=v, yaw_rate=0.0)

    summary = sim.run(command, 16.0, None, None, allow_driving=True)
    time = np.array([r["time"] for r in sim.records])
    state = np.array([r["state"] for r in sim.records])
    hip = np.array([sim._hip_common(np.array(r["qpos"])[sim.hip_q]) for r in sim.records])
    velocity, position = state[:, 1], state[:, 0]
    rise_mask = (time >= 0.0) & (velocity >= 0.9 * max_speed)
    t_rise = float(time[np.argmax(rise_mask)]) if rise_mask.any() else float("nan")
    stop_mask = (time >= 4.0) & (time <= 8.0) & (np.abs(velocity) < 0.05)
    t_stop = float(time[np.argmax(stop_mask)] - 4.0) if stop_mask.any() else float("nan")
    index = int(np.argmin(np.abs(time - (4.0 + t_stop)))) if np.isfinite(t_stop) else 0
    brake_index = int(np.argmin(np.abs(time - 4.0)))
    return dict(t_rise=t_rise, t_stop=t_stop,
                distance=float(abs(position[index] - position[brake_index])),
                pitch=float(np.degrees(np.abs(state[:, 2]).max())),
                hip=float(np.abs(hip).max()),
                tau_hip=float(np.max(np.abs([r["torque"][0] for r in sim.records]))),
                mpc_fb=summary["mpc_fallbacks"], wbc_fb=summary["wbc_fallbacks"],
                stop=summary["stop_reason"])


def zero_input(config, settings):
    out = {}
    for case in ("H1", "H2", "H3", "H4", "Tf"):
        z = a1._run_zero_input(settings, config, case)
        out[case] = (z["settle"], z["crossings"], z["pitch_peak"], z.get("reason"))
    return out


def jump(config, settings, speed):
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
    summary = sim.run(MotionCommand(velocity=speed, yaw_rate=0.0), 5.0, None, None,
                      allow_driving=True, jump_times=[2.0])
    return "->".join(e["phase"][:3] for e in sim.jump_log), summary["stop_reason"]


def main() -> int:
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or [None, "tuned-agile"]
    rows = [(n, *build(n)) for n in names]
    print("== mid-release (airborne samples / pitch deg / wbc fallback) ==")
    print("| release | " + " | ".join(str(n or "default") for n, _, _ in rows) + " |")
    print("|---|" + "---|" * len(rows))
    for t_release in (0.3, 0.5, 0.7, 0.9):
        cells = []
        for _, config, settings in rows:
            r = mid_release(config, settings, t_release)
            cells.append("%d / %.1f / %d" % (r["air"], r["pitch"], r["wbc_fb"]))
        print("| %.1f s | %s |" % (t_release, " | ".join(cells)), flush=True)
    print()
    print("== V4 ==")
    for name, config, settings in rows:
        r = v4(config, settings)
        print("v4 %-12s t_rise=%.2f t_stop=%.2f dist=%.3f |pitch|max=%.1f hip_max=%.2f tau_hip=%.2f fb=%d/%d stop=%s" % (
            name or "default", r["t_rise"], r["t_stop"], r["distance"], r["pitch"], r["hip"],
            r["tau_hip"], r["mpc_fb"], r["wbc_fb"], r["stop"]), flush=True)
    print()
    print("== zero input (settle / crossings / pitch / reason) ==")
    for name, config, settings in rows:
        z = zero_input(config, settings)
        print("zero %-12s %s" % (name or "default",
                                 " ".join("%s=%.2f/%d/%.1f/%s" % (k, *v) for k, v in z.items())), flush=True)
    print()
    print("== jump ==")
    for name, config, settings in rows:
        for speed in (0.0, 1.5):
            phases, stop = jump(config, settings, speed)
            print("jump %-12s v=%.1f %s stop=%s" % (name or "default", speed, phases, stop), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
