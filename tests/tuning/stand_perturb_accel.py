"""第 29 次排查复现：静止受扰后立即起步加速导致轮子卸载 / 离地 flap。

场景：t = 2.5 s 施加水平推扰 dv，t = 3.0 s 起给 2.5 m/s 速度指令，跑 9 s。
统计起指令后的离地采样数、WBC 降级次数、俯仰峰、髋角峰。只输出 ASCII。

用法：.venv/Scripts/python.exe tests/tuning/stand_perturb_accel.py [推扰 m/s ...]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import (MuJoCoMpcWbcSimulation, SimulationConfig,  # noqa: E402
                                Disturbance, _build_from_args, default_app_mpc_settings)

# (标签, preset, MPCSettings 覆盖)
CONFIGS = [
    ("default", None, {}),
    ("agile hip1.2", "tuned-agile", {}),
    ("agile hip0.8", "tuned-agile", {"drive_hip_angle_limit": None}),
    ("agile hip1.0", "tuned-agile", {"drive_hip_angle_limit": 1.0}),
    ("agile ff15", "tuned-agile", {"pitch_feedforward_max": np.radians(15.0)}),
    ("agile noR2R1", "tuned-agile", {"unwind_before_brake": False,
                                     "preserve_ramp_on_recontact": False}),
]


def build(preset: str | None, overrides: dict):
    if preset is None:
        config, settings = SimulationConfig(), default_app_mpc_settings()
    else:
        config, settings, _ = _build_from_args(["--preset", preset])
    return config, (replace(settings, **overrides) if overrides else settings)


def run(label: str, preset: str | None, overrides: dict, dv: float = 0.3,
        t_push: float = 2.5, t_cmd: float = 3.0, duration: float = 9.0) -> dict:
    config, settings = build(preset, overrides)
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
    summary = sim.run(lambda t: MotionCommand(velocity=2.5 if t >= t_cmd else 0.0, yaw_rate=0.0),
                      duration, None, [Disturbance(time=t_push, delta_velocity=(dv, 0.0, 0.0))],
                      allow_driving=True)
    time = np.array([r["time"] for r in sim.records])
    state = np.array([r["state"] for r in sim.records])
    contact = np.array([r["contact"] for r in sim.records])
    hip = np.array([sim._hip_common(np.array(r["qpos"])[sim.hip_q]) for r in sim.records])
    mask = time >= t_cmd
    limits = sim.mpc.settings.limits
    return dict(label=label, acc=limits.commanded_acceleration, brake=limits.max_braking,
                ff=float(np.degrees(sim.mpc.settings.pitch_feedforward_max)),
                air=int((~contact[mask].all(axis=1)).sum()), wbc_fb=summary["wbc_fallbacks"],
                pitch=float(np.degrees(np.abs(state[mask, 2]).max())),
                hip=float(np.abs(hip[mask]).max()), stop=summary["stop_reason"])


def main() -> int:
    pushes = [float(v) for v in sys.argv[1:]] or [0.3]
    for dv in pushes:
        for label, preset, overrides in CONFIGS:
            r = run(label, preset, overrides, dv)
            print("RES push=%.2f %-14s acc=%.2f/%.2f ff=%.0f | air=%3d fb=%2d |th|max=%5.1f |hip|max=%.2f stop=%s" % (
                dv, r["label"], r["acc"], r["brake"], r["ff"], r["air"], r["wbc_fb"],
                r["pitch"], r["hip"], r["stop"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
