"""A 系列（第 15 次排查）验收：V3 / V4 起步刹车指标 + 零输入回归。

对比 tuned-fast preset 下 auto_accel_limits 开 / 关两组，检查：
- t_rise：0 -> 90% max_speed 用时；
- t_stop：max_speed -> |v| < 0.05 用时；
- stop_distance：实测刹车距离与理论值（v^2 / 2a）；
- ramp_accel：起步段 / 刹车段的实测峰值加速度与规划值之比；
- 髋角峰、俯仰峰与俯仰过零、stop_reason；
- 零输入回归（H1 / H2 / H3 / H4 / T-f）的回位与俯仰。

用法（项目根）：
    .venv/Scripts/python.exe tests/tuning/a1_acceptance.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MotionCommand, MPCSettings
from APP.mujoco_mpc_wbc import (Disturbance, MuJoCoMpcWbcSimulation, SimulationConfig,
                                _PRESETS)

_SETTLE_THRESHOLD = 0.01     # [m] 位移回位判据
_SETTLE_DWELL = 1.0          # [s] 位移需保持的时间
_SPEED_STOP = 0.05           # [m/s] 刹车完成判据
_RISE_FRACTION = 0.9         # t_rise 的目标比例

_PRESET = _PRESETS["tuned-fast"]


def _settings(auto: bool) -> MPCSettings:
    base = MPCSettings(compensate_hip_torque=False)
    overrides = dict(_PRESET["weights"])
    return replace(base, weights=replace(base.weights, **overrides),
                   hip_angle_limit=float(_PRESET["hip_angle_limit"]),
                   auto_accel_limits=auto)


def _config(preview_lead: float = None) -> SimulationConfig:
    return SimulationConfig(
        model_variant=_PRESET["model_variant"],
        forward_mode=_PRESET["forward_mode"],
        preview_lead=float(_PRESET["preview_lead"] if preview_lead is None else preview_lead),
        hip_guard=float(_PRESET["hip_guard"]),
        tracking_velocity_gain=2.0 * float(_PRESET["tracking_bandwidth"]),
        tracking_position_gain=float(_PRESET["tracking_bandwidth"]) ** 2,
    )


def _series(sim, key_path):
    time = np.array([record["time"] for record in sim.records])
    state = np.array([record["state"] for record in sim.records])
    return time, state


def _smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _v4_command(time: float, max_speed: float) -> MotionCommand:
    if time < 4.0:
        velocity = max_speed
    elif time < 8.0:
        velocity = 0.0
    elif time < 12.0:
        velocity = -max_speed
    else:
        velocity = 0.0
    return MotionCommand(velocity=velocity, yaw_rate=0.0)


def _v3_command(time: float) -> MotionCommand:
    if time < 2.0:
        velocity = 0.0
    elif time < 5.0:
        velocity = 2.5
    elif time < 8.0:
        velocity = 0.0
    elif time < 11.0:
        velocity = -2.5
    else:
        velocity = 0.0
    return MotionCommand(velocity=velocity, yaw_rate=0.0)


def _ramp_metrics(time, velocity, t_start, t_span, limit) -> dict:
    """段内平均加速度（最小二乘斜率）与峰值加速度（中心差分，避开首尾 0.2 s）。

    直接用滑窗平滑会在段边界产生虚假梯度（命令跳变处），所以峰值只在段中间算，
    比值用平均加速度（与规划限值同口径）。
    """
    mask = (time >= t_start) & (time <= t_start + t_span)
    if mask.sum() < 3:
        return {"mean": float("nan"), "peak": float("nan"), "ratio": float("nan")}
    times = time[mask]
    segment = velocity[mask]
    mean_accel = float(np.polyfit(times - times[0], segment, 1)[0])
    center = (times >= times[0] + 0.2) & (times <= times[-1] - 0.2)
    if center.sum() >= 3:
        accel = np.gradient(segment[center], times[center])
        peak = float(np.max(np.abs(accel)))
    else:
        peak = abs(mean_accel)
    ratio = abs(mean_accel) / float(limit) if limit else float("nan")
    return {"mean": mean_accel, "peak": peak, "ratio": ratio}


def _run_drive(settings, config, kind: str, max_speed: float) -> dict:
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
    command = (_v4_command if kind == "V4" else _v3_command)
    duration = 16.0 if kind == "V4" else 14.0
    summary = sim.run(lambda t: command(t, max_speed) if kind == "V4" else command(t),
                      duration, None, None, allow_driving=True)
    time, state = _series(sim, None)
    velocity, position = state[:, 1], state[:, 0]
    pitch = np.degrees(state[:, 2])
    hip_q = sim.hip_q
    hip_sign = float(sim._hip_axis_sign)
    hip = np.array([0.5 * (record["qpos"][hip_q[0]] + hip_sign * record["qpos"][hip_q[1]])
                    for record in sim.records])

    # 段划分（起步开始, 起步结束=刹车开始, 刹车结束）
    rise_start, t_brake, brake_end = {"V4": (0.0, 4.0, 8.0), "V3": (2.0, 5.0, 8.0)}[kind]
    rise_mask = (time >= rise_start) & (velocity >= _RISE_FRACTION * max_speed)
    t_rise = float(time[np.argmax(rise_mask)] - rise_start) if rise_mask.any() else float("nan")
    stop_window = (time >= t_brake) & (time <= brake_end)
    stop_mask = stop_window & (np.abs(velocity) < _SPEED_STOP)
    if stop_mask.any():
        index = int(np.argmax(stop_mask))
        t_stop = float(time[index] - t_brake)
        distance = float(abs(position[index] - position[np.argmin(np.abs(time - t_brake))]))
    else:
        t_stop = float("nan")
        distance = float("nan")
    limits = sim.mpc.settings.limits
    # 拟合段取"斜坡段"：到达到 90% 目标速度为止，避免把匀速平台算进平均斜率
    rise_span = float(t_rise) if np.isfinite(t_rise) else (t_brake - rise_start)
    up = _ramp_metrics(time, velocity, rise_start, rise_span, limits.commanded_acceleration)
    # 刹车段同理只取"到停住为止"的区间，避免静止平台稀释平均减速度
    down_span = float(t_stop) if np.isfinite(t_stop) else (brake_end - t_brake)
    down = _ramp_metrics(time, velocity, t_brake, down_span, limits.max_braking)
    theoretical = max_speed ** 2 / (2.0 * limits.max_braking)
    hip_limit = float(sim.mpc.settings.hip_angle_limit)
    hip_window = np.abs(hip[(time >= rise_start) & (time <= brake_end)])
    hip_contact_ratio = float(np.mean(hip_window > 0.95 * hip_limit)) if hip_window.size else 0.0
    return {
        "t_rise": t_rise, "t_stop": t_stop, "distance": distance,
        "theory": theoretical,
        "up_mean": up["mean"], "up_peak": up["peak"], "up_ratio": up["ratio"],
        "down_mean": down["mean"], "down_peak": down["peak"], "down_ratio": down["ratio"],
        "pitch_peak": float(np.max(np.abs(pitch))),
        "hip_peak": float(np.nanmax(np.abs(hip))) if np.isfinite(hip).any() else float("nan"),
        "stop_reason": summary.get("stop_reason"),
        "completed": summary.get("completed"),
        "hip_contact_ratio": hip_contact_ratio,
        "planned_up": limits.commanded_acceleration,
        "planned_down": limits.max_braking,
    }


def _run_zero_input(settings, config, name: str) -> dict:
    scenarios = {
        "H1": (Disturbance(time=5.0, delta_velocity=(0.2, 0.0, 0.0)), 15.0, 0.0),
        "H2": (Disturbance(time=5.0, delta_velocity=(0.5, 0.0, 0.0)), 15.0, 0.0),
        "H3": (Disturbance(time=5.0, force=(40.0, 0.0, 0.0), torque=(0.0, 3.0, 0.0),
                           duration=0.20), 15.0, 0.0),
        "H4": (Disturbance(time=5.0, delta_velocity=(0.8, 0.0, 0.0)), 15.0, 0.0),
        "Tf": (Disturbance(time=5.0, delta_velocity=(0.0, 0.0, 0.0)), 20.0, 0.0),
    }
    disturbance, duration, pitch0 = scenarios[name]
    if name == "Tf":
        disturbances = [Disturbance(time=5.0 + i * 0.4, delta_velocity=(0.35, 0.0, 0.0))
                        for i in range(5)]
    else:
        disturbances = [disturbance]
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings, initial_pitch=pitch0)
    summary = sim.run(MotionCommand(velocity=0.0, yaw_rate=0.0), duration, None, disturbances)
    time, state = _series(sim, None)
    position, pitch = state[:, 0], np.degrees(state[:, 2])
    event = 5.0
    mask = time >= event
    settle = float("nan")
    tail = mask & (np.abs(position) < _SETTLE_THRESHOLD)
    if tail.any():
        index = int(np.argmax(tail))
        end = time[index] + _SETTLE_DWELL
        future = (time >= end)
        if future.any() and np.all(np.abs(position[future]) < _SETTLE_THRESHOLD):
            settle = float(time[index] - event)
    crossings = int(np.sum(np.diff(np.sign(pitch[mask])) != 0))
    return {"settle": settle, "crossings": crossings,
            "pitch_peak": float(np.max(np.abs(pitch[mask]))),
            "reason": summary.get("stop_reason")}


def _main() -> None:
    max_speed = 2.5
    print(f"max_speed={max_speed} m/s  preset=tuned-fast  "
          f"hip_angle_limit={_PRESET['hip_angle_limit']}")
    print()
    print("=== V4（0 -> 2.5 -> 0 -> -2.5 -> 0，每段 4 s）===")
    header = ("auto  规划(起/刹)  起步 a均值/比  刹车 a均值/比  t_rise  t_stop  "
              "distance/theory  髋峰(占比)  俯仰峰  停止")
    print(header)
    for auto in (False, True):
        settings = _settings(auto)
        row = _run_drive(settings, _config(), "V4", max_speed)
        print(f"{str(auto):>5}  {row['planned_up']:.2f}/{row['planned_down']:.2f}    "
              f"{row['up_mean']:.2f}/{row['up_ratio']:.2f}      "
              f"{row['down_mean']:.2f}/{row['down_ratio']:.2f}    "
              f"{row['t_rise']:.2f}   {row['t_stop']:.2f}   "
              f"{row['distance']:.3f}/{row['theory']:.3f}   "
              f"{row['hip_peak']:.3f}({row['hip_contact_ratio']*100:.0f}%)  "
              f"{row['pitch_peak']:.2f}   {row['stop_reason']}")
    print()
    print("=== V3（键盘序列 0 -> 2.5 -> 0 -> -2.5 -> 0）===")
    print(header)
    for auto in (False, True):
        settings = _settings(auto)
        row = _run_drive(settings, _config(), "V3", max_speed)
        print(f"{str(auto):>5}  {row['planned_up']:.2f}/{row['planned_down']:.2f}    "
              f"{row['up_mean']:.2f}/{row['up_ratio']:.2f}      "
              f"{row['down_mean']:.2f}/{row['down_ratio']:.2f}    "
              f"{row['t_rise']:.2f}   {row['t_stop']:.2f}   "
              f"{row['distance']:.3f}/{row['theory']:.3f}   "
              f"{row['hip_peak']:.3f}({row['hip_contact_ratio']*100:.0f}%)  "
              f"{row['pitch_peak']:.2f}   {row['stop_reason']}")
    print()
    print("=== 零输入回归（回位 / 俯仰过零 / 俯仰峰 / 停止）===")
    for name in ("H1", "H2", "H3", "H4", "Tf"):
        cells = []
        for auto in (False, True):
            metrics = _run_zero_input(_settings(auto), _config(), name)
            settle = f"{metrics['settle']:.2f}" if np.isfinite(metrics["settle"]) else "  --"
            cells.append(f"auto={str(auto):<5} 回位={settle} 过零={metrics['crossings']} "
                         f"俯仰={metrics['pitch_peak']:.1f} 停止={metrics['reason']}")
        print(f"  {name}: " + " | ".join(cells))


if __name__ == "__main__":
    _main()
