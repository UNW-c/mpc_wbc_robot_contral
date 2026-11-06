"""本项目的 MPC 参数 preset。

这些数值沿用项目初期从 circleg 工作区继承的调参结果，作为 GBC486 的
默认档位保留；改动 preset 会直接改变控制行为，属于调参范围。
``preset = None`` 时使用 ``tuned-fast-accel``。
"""
from __future__ import annotations

from dataclasses import replace

from Algorithm.mpc import MPCSettings

_PRESETS: dict[str, dict] = {
    "tuned-best": {"model_variant": "hip_input", "forward_mode": "mpc", "preview_lead": 0.0,
                   "hip_guard": 1.7, "tracking_bandwidth": 7.0, "hip_tracking": 200.0,
                   "hip_angle_limit": 0.75,
                   "weights": {"position": 20000.0, "velocity": 1333.3333333333333,
                               "pitch": 150000.0, "pitch_rate": 100.0, "hip_angle": 50.0,
                               "hip_rate": 2.4, "terminal_mode": "scale", "terminal_scale": 5.0}},
    "tuned-fast": {"model_variant": "hip_input", "forward_mode": "mpc", "preview_lead": 0.0,
                   "hip_guard": 1.7, "tracking_bandwidth": 7.0, "hip_tracking": 200.0,
                   "hip_angle_limit": 0.8,
                   "weights": {"position": 20000.0, "velocity": 1333.3333333333333,
                               "pitch": 150000.0, "pitch_rate": 300.0, "hip_angle": 50.0,
                               "hip_rate": 3.0, "terminal_mode": "scale", "terminal_scale": 5.0}},
    "tuned-gentle": {"model_variant": "hip_input", "forward_mode": "mpc", "preview_lead": 0.0,
                     "hip_guard": 1.7, "tracking_bandwidth": 7.0, "hip_tracking": 200.0,
                     "hip_angle_limit": 0.7,
                     "weights": {"position": 20000.0, "velocity": 1333.3333333333333,
                                 "pitch": 150000.0, "pitch_rate": 300.0, "hip_angle": 50.0,
                                 "hip_rate": 4.5, "terminal_mode": "scale", "terminal_scale": 5.0}},
}
_PRESETS["tuned-fast-accel"] = {
    **_PRESETS["tuned-fast"], "use_jerk_limit": True,
    "tilt_assisted_limits": True, "pitch_feedforward": True,
}
_PRESETS["tuned-agile"] = {
    **_PRESETS["tuned-fast-accel"],
    "mpc_overrides": {
        "pitch_ff_rate_limit": 3.0, "jerk_ramp_time": 0.15,
        "brake_pitch_max": 0.3491, "brake_jerk_ramp_time": 0.08,
        "pitch_feedforward_max": 0.3491, "drive_hip_angle_limit": 1.2,
        "unwind_before_brake": True, "preserve_ramp_on_recontact": True,
        "zmp_reference": True, "zmp_split": 0.3, "reanchor_position_while_braking": True,
    },
}
_PRESETS.update({
    "new": {"model_variant": "hip_input", "forward_mode": "mpc", "preview_lead": 0.0,
            "hip_guard": 1.70, "hip_angle_limit": 0.55},
    "old": {"model_variant": "held_hip", "forward_mode": "legacy", "preview_lead": 0.0},
})


def default_app_mpc_settings() -> MPCSettings:
    """默认 MPC 设置 = ``tuned-fast-accel``。"""
    preset = _PRESETS["tuned-fast-accel"]
    base = MPCSettings(compensate_hip_torque=False)
    return replace(base, weights=replace(base.weights, **preset["weights"]),
                   hip_angle_limit=float(preset["hip_angle_limit"]),
                   limits=replace(base.limits, use_jerk_limit=True),
                   tilt_assisted_limits=True, pitch_feedforward=True)
