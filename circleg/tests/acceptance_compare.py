"""验收工况对比（H1/H2/H3/V1）+ 2x2 因子实验 + MPC 规划诊断。

用法：
    .venv/Scripts/python.exe tests/acceptance_compare.py              # baseline vs new
    .venv/Scripts/python.exe tests/acceptance_compare.py --matrix     # R1：4 组因子实验
    .venv/Scripts/python.exe tests/acceptance_compare.py --mpc-log    # R2：H1 的 MPC 规划日志

基线 = 改动前完整行为（legacy + preview_lead=0.02）；new = T1+T4（mpc + lead 0.0）。
指标口径（R4）：
- settle：位移持续 1 s 小于 1 cm 的第一时刻（相对事件时刻），不要求之后一直小于；
- hip_occ / wheel_occ：分别按髋（ctrl 0/1）与轮（ctrl 2/3）的 ctrlrange 归一化的最大占用；
- hip_angle_peak / hip_speed_peak：共模髋角 0.5*(q1-q3) 与其角速度的峰值（R3 定义）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Algorithm.mpc import MotionCommand
from APP.mujoco_mpc_wbc import Disturbance, MuJoCoMpcWbcSimulation, SimulationConfig

_SCENARIOS = {
    "H1": dict(duration=15.0, disturbances=[Disturbance(time=5.0, delta_velocity=(0.2, 0.0, 0.0))]),
    "H2": dict(duration=15.0, disturbances=[Disturbance(time=5.0, delta_velocity=(0.5, 0.0, 0.0))]),
    "H3": dict(duration=15.0, disturbances=[Disturbance(time=5.0, force=(40.0, 0.0, 0.0),
                                                        torque=(0.0, 3.0, 0.0), duration=0.20)]),
    "V1": dict(duration=14.0, disturbances=[]),
    "V2": dict(duration=8.0, disturbances=[]),
    # E4 鲁棒性补充（只报告，不作为 G 判据）
    "H4": dict(duration=15.0, disturbances=[Disturbance(time=5.0, delta_velocity=(0.8, 0.0, 0.0))]),
    "H5": dict(duration=15.0, disturbances=[Disturbance(time=5.0, delta_velocity=(-0.5, 0.0, 0.0))]),
    "H6": dict(duration=20.0, disturbances=[], initial_pitch=0.02),
    # K1：V3 模拟键盘实际输入（2.5 被规划器限到 1.2）
    "V3": dict(duration=14.0, disturbances=[]),
}
_V1_SEQUENCE = ((0.0, 0.0), (2.0, 1.0), (4.0, 0.0), (8.0, 1.0), (12.0, 0.0))
_V2_SEQUENCE = ((0.0, 0.0), (2.0, 1.0), (5.0, 0.0))
_V3_SEQUENCE = ((0.0, 0.0), (2.0, 2.5), (5.0, 0.0), (8.0, -2.5), (11.0, 0.0))

# R1 的 2x2 因子实验是 6 状态握髋模型的对照（2026-09-28 默认切换为 hip_input 后,
# 必须显式指定 model_variant，否则这几组会悄悄变成 hip_input）。
_MATRIX = [
    ("legacy+lead0.02", SimulationConfig(model_variant="held_hip", forward_mode="legacy", preview_lead=0.02)),
    ("legacy+lead0.0", SimulationConfig(model_variant="held_hip", forward_mode="legacy", preview_lead=0.0)),
    ("mpc+lead0.02", SimulationConfig(model_variant="held_hip", forward_mode="mpc", preview_lead=0.02)),
    ("mpc+lead0.0", SimulationConfig(model_variant="held_hip", forward_mode="mpc", preview_lead=0.0)),
]
def _hip_settings(hip_angle=200.0, hip_angle_limit=0.45, force_weight=None):
    """构造 hip_input 路径的 MPCSettings（E1/E3 用）。"""
    from dataclasses import replace
    from Algorithm.mpc import MPCSettings

    base = MPCSettings(compensate_hip_torque=False)
    weight_overrides = {"hip_angle": float(hip_angle)}
    if force_weight is not None:
        weight_overrides["force"] = float(force_weight)
    weights = replace(base.weights, **weight_overrides)
    return replace(base, weights=weights, hip_angle_limit=float(hip_angle_limit))


_HIP_CONFIG = SimulationConfig(model_variant="hip_input", forward_mode="mpc", preview_lead=0.0)
_NAMED = {
    "baseline": (_MATRIX[0][0], _MATRIX[0][1], None),
    "legacy0": (_MATRIX[1][0], _MATRIX[1][1], None),
    "new": (_MATRIX[3][0], _MATRIX[3][1], None),
    "hipin": ("hip_input+mpc", _HIP_CONFIG, None),
    # E1：B4 推荐的 hip_angle=50
    "hip50": ("hip_input+hip50", _HIP_CONFIG, _hip_settings(50.0, 0.45)),
    # E2-3b：force 权重 0.2（只对 hip_input 路径）+ E3：放宽髋角约束
    "hip50f2": ("hip50+f0.2+l0.45", _HIP_CONFIG, _hip_settings(50.0, 0.45, force_weight=0.2)),
    "hip50f2_l55": ("hip50+f0.2+l0.55", _HIP_CONFIG, _hip_settings(50.0, 0.55, force_weight=0.2)),
    "hip50f2_l65": ("hip50+f0.2+l0.65", _HIP_CONFIG, _hip_settings(50.0, 0.65, force_weight=0.2)),
    "hip50_l55": ("hip50+limit0.55", _HIP_CONFIG, _hip_settings(50.0, 0.55)),
    "hip50_l65": ("hip50+limit0.65", _HIP_CONFIG, _hip_settings(50.0, 0.65)),
}


def _v1_command(time: float) -> MotionCommand:
    velocity = 0.0
    for start, value in _V1_SEQUENCE:
        if time >= start:
            velocity = value
    return MotionCommand(velocity=velocity, yaw_rate=0.0)


def _v2_yaw_rate(time: float) -> float:
    yaw_rate = 0.0
    for start, value in _V2_SEQUENCE:
        if time >= start:
            yaw_rate = value
    return yaw_rate


def _v2_command(time: float) -> MotionCommand:
    return MotionCommand(velocity=0.0, yaw_rate=_v2_yaw_rate(time))


def _v3_velocity(time: float) -> float:
    velocity = 0.0
    for start, value in _V3_SEQUENCE:
        if time >= start:
            velocity = value
    return velocity


def _v3_command(time: float) -> MotionCommand:
    return MotionCommand(velocity=_v3_velocity(time), yaw_rate=0.0)


def _metrics(sim, summary, event_time: float, command_fn=None):
    records = sim.records
    time = np.array([r["time"] for r in records])
    state = np.array([r["state"] for r in records])
    torque = np.array([r["torque"] for r in records])
    qpos = np.array([r["qpos"] for r in records])
    qvel = np.array([r["qvel"] for r in records])
    qacc = np.array([r["predicted_qacc"] for r in records])
    residual = np.array([r["task_residuals"].get("forward_velocity", np.nan) for r in records])
    goal = np.array([float(np.linalg.norm(r["task_goal"].get("forward_velocity", [np.nan, 0.0])))
                     for r in records])
    limits = np.maximum(np.abs(sim.dynamics.limits.lower), np.abs(sim.dynamics.limits.upper))
    window = time >= event_time
    if not np.any(window):
        # 仿真在事件时刻前就停止（guard/失败）：返回空指标而不是崩溃
        return dict(peak_position=float("nan"), settle=float("nan"), peak_pitch=float("nan"),
                    pitch_cross=0, fv_ratio=float("nan"), hip_occ=float("nan"),
                    wheel_occ=float("nan"), hip_angle_peak=float("nan"),
                    hip_speed_peak=float("nan"), hip_qacc=float("nan"),
                    accel_ratio=float("nan"), overshoot=float("nan"),
                    mpc_fb=summary["mpc_fallbacks"], wbc_fb=summary["wbc_fallbacks"],
                    stop=f"early:{summary['stop_reason']}")
    hip_angle = 0.5 * (qpos[:, sim.hip_q[0]] - qpos[:, sim.hip_q[1]])
    hip_speed = 0.5 * (qvel[:, sim.hip_v[0]] - qvel[:, sim.hip_v[1]])
    peak_position = float(np.max(np.abs(state[window, 0])))
    # settle：位移持续 1 s 小于 1 cm
    step = float(np.mean(np.diff(time)))
    hold = max(1, int(round(1.0 / step)))
    settle = float("nan")
    for index in range(len(time)):
        if time[index] >= event_time and np.all(np.abs(state[index:index + hold, 0]) < 0.01):
            settle = float(time[index] - event_time)
            break
    peak_pitch = float(np.max(np.abs(np.degrees(state[window, 2]))))
    # 死区 0.1 度：无死区时数值噪声会产生上万次假穿越（B4 首轮扫描发现）
    pitch_deg = np.degrees(state[window, 2])
    significant = np.where(np.abs(pitch_deg) > 0.1, np.sign(pitch_deg), 0)
    # F1（#18）：只统计符号真正反转的次数（忽略 0——+1→0→+1 不算穿越）。
    nonzero = significant[significant != 0]
    pitch_cross = int(np.sum(nonzero[1:] != nonzero[:-1]))
    ratio = float(np.nanmean(residual[window] / np.maximum(goal[window], 1.0)))
    # G4（B4）：参考/实测加速度之比与速度超调（只在有指令的工况有意义）
    planner = np.array([r["planner_velocity"] if r["planner_velocity"] is not None else np.nan
                        for r in records])
    accel_ratio = float("nan")
    overshoot = float("nan")
    if command_fn is not None and np.any(np.isfinite(planner)):
        command_velocity = np.array([command_fn(x).velocity for x in time])
        acceleration_measured = np.gradient(state[:, 1], time)
        acceleration_reference = np.gradient(planner, time)
        # 分段端点差商（B4 修正：逐点梯度被 250 Hz 速度噪声污染，会报出 0.30 的假值；
        # 分段端点差商与 V1 分段诊断口径一致）。段划分对应 V1 序列：
        # 起步1(2-4s) / 减速1(4-7s) / 起步2(8-11.5s) / 减速2(12-14s)。
        del acceleration_measured, acceleration_reference
        ratios = []
        for t0, t1 in ((2.0, 4.0), (4.0, 7.0), (8.0, 11.5), (12.0, 14.0)):
            mask = (time >= t0) & (time <= t1)
            if not np.any(mask):
                continue
            span = float(time[mask][-1] - time[mask][0])
            a_meas = float(state[mask, 1][-1] - state[mask, 1][0]) / span
            a_ref = float(planner[mask][-1] - planner[mask][0]) / span
            if abs(a_ref) > 1e-6:
                ratios.append(a_meas / a_ref)
        if ratios:
            accel_ratio = float(np.mean(ratios))
        moving = command_velocity > 0.1
        if np.any(moving):
            overshoot = float(np.max(state[moving, 1] - command_velocity[moving]))
    hip_occ = float(np.max(np.abs(torque[window][:, :2]) / limits[:2]))
    wheel_occ = float(np.max(np.abs(torque[window][:, 2:]) / limits[2:]))
    hip_qacc = float(np.max(np.abs(qacc[window][:, sim.hip_v[0]:sim.hip_v[0] + 1])))
    hip_qacc = max(hip_qacc, float(np.max(np.abs(qacc[window][:, sim.hip_v[1]:sim.hip_v[1] + 1]))))
    return dict(peak_position=peak_position, settle=settle, peak_pitch=peak_pitch,
                final_position_abs=float(abs(state[window][-1, 0])) if np.any(window) else float("nan"),
                pitch_cross=pitch_cross, fv_ratio=ratio, hip_occ=hip_occ, wheel_occ=wheel_occ,
                hip_angle_peak=float(np.max(np.abs(hip_angle[window]))),
                hip_speed_peak=float(np.max(np.abs(hip_speed[window]))),
                hip_qacc=hip_qacc, accel_ratio=accel_ratio, overshoot=overshoot,
                mpc_fb=summary["mpc_fallbacks"], wbc_fb=summary["wbc_fallbacks"],
                stop=summary["stop_reason"])


def _print_header():
    print(f"{'工况':>4}{'配置':>16}{'位移峰':>9}{'回位时间':>9}{'俯仰峰°':>9}{'过零*':>6}"
          f"{'fv残差比':>9}{'髋占用':>8}{'轮占用':>8}{'髋角峰':>8}{'髋速峰':>8}{'髋acc峰':>9}"
          f"{'a比':>7}{'超调':>7}{'fb':>7}{'停止':>14}")


def _print_row(scenario, label, m):
    print(f"{scenario:>4}{label:>16}{m['peak_position']:>9.4f}{m['settle']:>9.2f}"
          f"{m['peak_pitch']:>9.3f}{m['pitch_cross']:>5d}{m['fv_ratio']:>9.2f}"
          f"{m['hip_occ']:>8.2f}{m['wheel_occ']:>8.2f}{m['hip_angle_peak']:>8.3f}"
          f"{m['hip_speed_peak']:>8.2f}{m['hip_qacc']:>9.1f}"
          f"{m['accel_ratio']:>7.2f}{m['overshoot']:>7.3f}"
          f"{m['mpc_fb']:>3d}/{m['wbc_fb']:<3d}{str(m['stop']):>14}")


def _run_scenarios(configs, scenarios):
    _print_header()
    for scenario in scenarios:
        spec = _SCENARIOS[scenario]
        for label, config, settings in configs:
            sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings,
                                         initial_pitch=float(spec.get("initial_pitch", 0.0)))
            command = {"V1": _v1_command, "V2": _v2_command, "V3": _v3_command}.get(
                scenario, MotionCommand(velocity=0.0, yaw_rate=0.0))
            summary = sim.run(command, spec["duration"], None, spec["disturbances"],
                              allow_driving=(scenario in ("V1", "V2", "V3")))
            command_fn = _v1_command if scenario == "V1" else None
            _print_row(scenario, label,
                       _metrics(sim, summary, 5.0 if scenario != "V1" else 2.0,
                                command_fn=command_fn))


def _run_mpc_log(config_name="new"):
    """R2：H1 + 指定配置下，扰动后第 1、5、25 次 MPC 求解的规划内容。"""
    spec = _SCENARIOS["H1"]
    label, config, settings = _NAMED.get(config_name, _NAMED["new"])
    print(f"配置：{label}")
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
    sim.solution_log = []
    sim.run(MotionCommand(velocity=0.0, yaw_rate=0.0), spec["duration"], None, spec["disturbances"])
    after = [entry for entry in sim.solution_log if entry["time"] >= 5.0]
    print(f"扰动后 MPC 求解 {len(after)} 次（总 {len(sim.solution_log)}）")
    for index in (1, 5, 25):
        if index > len(after):
            continue
        entry = after[index - 1]
        states = np.array(entry["predicted_states"])
        time = np.arange(len(states)) * 0.02
        velocity = states[:, 1]
        stopped = np.where(np.abs(velocity) < 0.02)[0]
        stop_time = float(time[stopped[0]]) if stopped.size else float("nan")
        extra = ""
        if states.shape[1] >= 8:
            extra = f" q_c峰={np.max(np.abs(states[:, 6])):.3f} q_c末={states[-1, 6]:+.3f}"
        print(f"[{index:>2}] t={entry['time']:.3f}s v0={velocity[0]:+.3f} v_end={velocity[-1]:+.3f} "
              f"v<0.02@={stop_time:.2f}s 终端位置={states[-1, 0]:+.3f} "
              f"俯仰峰={np.degrees(np.max(np.abs(states[:, 2]))):.2f}°{extra} "
              f"planner={entry['planner_mode']}/{entry['planner_reset_reason']} "
              f"force_peak={entry['force_peak']:.2f} relaxed={entry['qp_relaxed']}")


def _ramp_peak(time, values, t0, window=1.5, smooth=0.2):
    """K1：t0 后 window 秒内，0.2 s 滑动平均斜率的峰值（绝对值）。"""
    mask = (time >= t0) & (time <= t0 + window)
    t = time[mask]
    v = values[mask]
    if t.size < 3:
        return float("nan")
    step = float(np.mean(np.diff(t)))
    n = max(2, int(round(smooth / step)))
    if v.size <= n:
        return float("nan")
    slopes = (v[n:] - v[:-n]) / (t[n:] - t[:-n])
    return float(np.max(np.abs(slopes))) if slopes.size else float("nan")


def _stop_distance(time, position, velocity, t0, speed_threshold=0.02):
    """K1：指令归零时刻 t0 起，到速度首次 |v| < 阈值 的位移。"""
    index0 = int(np.argmin(np.abs(time - t0)))
    mask = (time >= t0) & (np.abs(velocity) < speed_threshold)
    if not np.any(mask):
        return float("nan")
    index1 = int(np.argmax(mask))
    return abs(float(position[index1] - position[index0]))


def _run_v3(configs):
    """K1：V3 键盘序列（2.5 -> 0 -> -2.5 -> 0）的起步/刹车分段指标。"""
    from Algorithm.mpc import MotionLimits

    limits = MotionLimits()
    print(f"起步/刹车分段（V3，规划限值：commanded={limits.commanded_acceleration:.2f} "
          f"max_braking={limits.max_braking:.2f} max_accel={limits.max_acceleration:.2f}）")
    print(f"{'配置':>16}{'段':>8}{'实测a':>8}{'规划a':>8}{'比值':>7}"
          f"{'超调':>8}{'停车距离':>9}{'理论':>8}{'髋角峰':>8}{'触碰%':>7}{'停止':>8}")
    for label, config, settings in configs:
        sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
        summary = sim.run(_v3_command, _SCENARIOS["V3"]["duration"], None, [], allow_driving=True)
        time = np.array([r["time"] for r in sim.records])
        state = np.array([r["state"] for r in sim.records])
        qpos = np.array([r["qpos"] for r in sim.records])
        planner = np.array([r["planner_velocity"] if r["planner_velocity"] is not None else np.nan
                            for r in sim.records])
        hip = 0.5 * (qpos[:, sim.hip_q[0]] - qpos[:, sim.hip_q[1]])
        limit = float(settings.hip_angle_limit) if settings is not None else 0.55
        v0 = float(np.max(np.abs(state[(time >= 2.0) & (time <= 5.0), 1])))
        for name, t0, kind in (("起步+", 2.0, "ramp"), ("刹车+", 5.0, "brake"),
                               ("起步-", 8.0, "ramp"), ("刹车-", 11.0, "brake")):
            measured = _ramp_peak(time, state[:, 1], t0)
            reference = _ramp_peak(time, planner, t0)
            ratio = measured / reference if reference and np.isfinite(reference) and reference > 1e-6 else float("nan")
            window = (time >= t0) & (time <= t0 + 3.0)
            if not np.any(window):
                print(f"{label:>16}{name:>8}{'--':>8}{'--':>8}{'--':>7}{'--':>8}{'--':>9}"
                      f"{'--':>8}{'--':>8}{'--':>7}{str(summary['stop_reason']):>8}")
                continue
            # V3 的指令会被规划器限到 max_speed；用"跟踪误差峰值"衡量超调。
            command_value = float(np.clip(_v3_velocity(t0), -1.2, 1.2))
            overshoot = float(np.max(np.abs(state[window, 1] - command_value)))
            touch = float(np.mean(np.abs(hip[window]) >= 0.99 * limit) * 100.0)
            if kind == "brake":
                distance = _stop_distance(time, state[:, 0], state[:, 1], t0)
                theoretical = v0 * v0 / (2.0 * limits.max_braking)
            else:
                distance = float("nan")
                theoretical = float("nan")
            print(f"{label:>16}{name:>8}{measured:>8.2f}{reference:>8.2f}{ratio:>7.2f}"
                  f"{overshoot:>8.3f}{distance:>9.4f}{theoretical:>8.4f}"
                  f"{float(np.max(np.abs(hip[window]))):>8.3f}{touch:>7.1f}{str(summary['stop_reason']):>8}")
            summary_stop = summary["stop_reason"]


def _run_v2(configs):
    """V2（A2 新增）：纯转向工况——零速度 + yaw_rate=1.0 持续 3 s。"""
    print(f"{'配置':>16}{'yaw误差峰':>10}{'yaw误差均值':>12}{'yaw_moment[min,max]':>22}"
          f"{'轮偏航峰':>10}{'停止':>14}")
    for label, config, settings in configs:
        sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
        sim.solution_log = []
        summary = sim.run(_v2_command, 8.0, None, [], allow_driving=True)
        time = np.array([r["time"] for r in sim.records])
        state = np.array([r["state"] for r in sim.records])
        torque = np.array([r["torque"] for r in sim.records])
        command = np.array([_v2_yaw_rate(x) for x in time])
        error = state[:, 5] - command
        window = time >= 2.0
        limits = np.array([float(entry["yaw_moment_limit"]) for entry in sim.solution_log])
        # 偏航由 j2 与 j4 同号驱动（j2=+R*f_R, j4=-R*f_L；f_R=-f_L 时同号）
        wheel_yaw = float(np.max(np.abs(torque[window][:, 2] + torque[window][:, 3])))
        span = f"[{limits.min():.2f}, {limits.max():.2f}]" if limits.size else "n/a"
        print(f"{label:>16}{float(np.max(np.abs(error[window]))):>10.3f}"
              f"{float(np.mean(np.abs(error[window]))):>12.3f}{span:>22}"
              f"{wheel_yaw:>10.3f}{str(summary['stop_reason']):>14}")


def _run_weight_sweep():
    """B4 附带扫描：hip_angle x position 共 9 组（H1/H2/V1），报告 G1/G2/G4 达标项计数。"""
    from dataclasses import replace
    from Algorithm.mpc import MPCSettings

    print(f"{'hip_w':>7}{'pos_w':>8}{'H1峰':>8}{'H1回位':>8}{'H1过零':>7}"
          f"{'H2峰':>8}{'H2回位':>8}{'H2过零':>7}{'V1a比':>7}{'V1超调':>8}"
          f"{'达标':>6}{'条件数':>10}{'P95ms':>7}{'停止':>8}")
    base = MPCSettings(compensate_hip_torque=False)
    for hip_w in (50.0, 200.0, 800.0):
        for pos_w in (20000.0, 80000.0, 320000.0):
            settings = replace(base, weights=replace(base.weights, hip_angle=hip_w, position=pos_w))
            config = SimulationConfig(model_variant="hip_input", forward_mode="mpc", preview_lead=0.0)
            row = {}
            stop = "None"
            solve_times = []
            for scenario in ("H1", "H2", "V1"):
                spec = _SCENARIOS[scenario]
                sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
                sim.solution_log = []
                command = _v1_command if scenario == "V1" else MotionCommand(velocity=0.0, yaw_rate=0.0)
                summary = sim.run(command, spec["duration"], None, spec["disturbances"],
                                  allow_driving=(scenario == "V1"))
                solve_times += [float(e["qp_solve_time"]) for e in sim.solution_log]
                m = _metrics(sim, summary, 5.0 if scenario != "V1" else 2.0,
                             command_fn=_v1_command if scenario == "V1" else None)
                row[scenario] = m
                if m["stop"]:
                    stop = f"{scenario}:{m['stop']}"
            score = 0
            score += int(row["H1"]["peak_position"] <= 0.05)
            score += int(row["H1"]["settle"] == row["H1"]["settle"] and row["H1"]["settle"] <= 1.0)
            score += int(row["H1"]["peak_pitch"] <= 10.0)
            score += int(row["H1"]["pitch_cross"] <= 3)
            score += int(row["H2"]["peak_position"] <= 0.15)
            score += int(row["H2"]["settle"] == row["H2"]["settle"] and row["H2"]["settle"] <= 2.0)
            score += int(row["H2"]["peak_pitch"] <= 10.0)
            score += int(row["H2"]["pitch_cross"] <= 3)
            score += int(row["V1"]["accel_ratio"] >= 0.9)
            score += int(row["V1"]["overshoot"] == row["V1"]["overshoot"] and row["V1"]["overshoot"] <= 0.1)
            condition = float(sim.mpc.qp.condition_number())
            p95 = float(np.percentile(solve_times, 95)) * 1000.0 if solve_times else float("nan")
            print(f"{hip_w:>7.0f}{pos_w:>8.0f}{row['H1']['peak_position']:>8.4f}"
                  f"{row['H1']['settle']:>8.2f}{row['H1']['pitch_cross']:>7d}"
                  f"{row['H2']['peak_position']:>8.4f}{row['H2']['settle']:>8.2f}"
                  f"{row['H2']['pitch_cross']:>7d}{row['V1']['accel_ratio']:>7.2f}"
                  f"{row['V1']['overshoot']:>8.3f}{score:>4d}/10{condition:>10.2e}"
                  f"{p95:>7.2f}{stop:>8}")


def _run_v1_detail(config_name="hipin"):
    """B4 诊断：V1 起步/减速各段的实测加速度 vs 规划器参考。"""
    label, config, settings = _NAMED.get(config_name, _NAMED["hipin"])
    spec = _SCENARIOS["V1"]
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
    summary = sim.run(_v1_command, spec["duration"], None, [], allow_driving=True)
    time = np.array([r["time"] for r in sim.records])
    state = np.array([r["state"] for r in sim.records])
    planner = np.array([r["planner_velocity"] if r["planner_velocity"] is not None else np.nan
                        for r in sim.records])
    mpc_ref = np.array([r["mpc_velocity"] if r["mpc_velocity"] is not None else np.nan
                        for r in sim.records])
    print(f"配置：{label}  停止：{summary['stop_reason']}")
    print(f"{'段':>10}{'a实测':>9}{'a规划':>9}{'a比值':>8}{'aMPC参考':>10}{'v_end':>8}"
          f"{'峰值a实测':>10}{'峰值a规划':>10}{'峰值比':>8}")
    for name, t0, t1 in (("起步1", 2.0, 4.0), ("减速1", 4.0, 7.0),
                         ("起步2", 8.0, 11.5), ("减速2", 12.0, 14.0)):
        mask = (time >= t0) & (time <= t1)
        if not np.any(mask):
            continue
        dt = float(np.mean(np.diff(time)))
        a_meas = float((state[mask, 1][-1] - state[mask, 1][0]) / (time[mask][-1] - time[mask][0]))
        a_ref = float((planner[mask][-1] - planner[mask][0]) / (time[mask][-1] - time[mask][0]))
        a_mpc = float((mpc_ref[mask][-1] - mpc_ref[mask][0]) / (time[mask][-1] - time[mask][0]))
        ratio = a_meas / a_ref if abs(a_ref) > 1e-6 else float("nan")
        peak_measured = _ramp_peak(time, state[:, 1], t0)
        peak_reference = _ramp_peak(time, planner, t0)
        peak_ratio = (peak_measured / peak_reference
                      if np.isfinite(peak_reference) and peak_reference > 1e-6 else float("nan"))
        print(f"{name:>10}{a_meas:>9.3f}{a_ref:>9.3f}{ratio:>8.2f}{a_mpc:>10.3f}"
              f"{state[mask, 1][-1]:>8.3f}{peak_measured:>10.2f}{peak_reference:>10.2f}{peak_ratio:>8.2f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="baseline,new")
    parser.add_argument("--scenarios", default="H1,H2,H3,V1")
    parser.add_argument("--matrix", action="store_true", help="R1：4 组因子实验")
    parser.add_argument("--mpc-log", action="store_true", help="R2：H1 的 MPC 规划日志")
    parser.add_argument("--log-config", default="new", help="--mpc-log 使用的配置名")
    parser.add_argument("--weight-sweep", action="store_true", help="B4：9 组权重扫描")
    parser.add_argument("--v1-detail", action="store_true", help="B4：V1 分段诊断")
    args = parser.parse_args()
    if args.mpc_log:
        _run_mpc_log(args.log_config)
        return 0
    if args.v1_detail:
        _run_v1_detail(args.log_config if args.log_config != "new" else "hipin")
        return 0
    if args.weight_sweep:
        _run_weight_sweep()
        return 0
    if args.matrix:
        configs = [(label, config, None) for label, config in _MATRIX]
    else:
        configs = [_NAMED[name.strip()] for name in args.configs.split(",") if name.strip()]
    scenarios = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    if scenarios == ["V2"]:
        _run_v2(configs)
        return 0
    if scenarios == ["V3"]:
        _run_v3(configs)
        return 0
    _run_scenarios(configs, scenarios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
