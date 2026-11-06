"""Diagnostic recorder for the current GBC486 MPC-WBC jump path.

This tool only enables the optional trace sink. It does not change controller inputs,
parameters, model files, thresholds, or control decisions.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from gbc_mpc_wbc.logging_setup import setup_logging
from gbc_mpc_wbc.simulation import Disturbance, Gbc486MpcWbcSimulation, SimulationConfig, default_command
from gbc_mpc_wbc import control as _control
from Algorithm.mpc import MotionCommand

_LOGGER = logging.getLogger(__name__)
RESULTS = ROOT / "results"


def _plain(value):
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _git_status():
    try:
        return subprocess.run(["git", "status", "--short"], cwd=PROJECT,
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return f"git status unavailable: {error}"


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"git commit unavailable: {error}"


def _sample_value(frame, name):
    sample = frame.get(name)
    if sample is None:
        return None
    return {"state": _plain(sample["state"]), "acceleration": _plain(sample["acceleration"])}


def _events(frames, release_end):
    events = []
    previous_phase = None
    for frame in frames:
        jump = frame.get("jump") or {}
        phase = jump.get("phase")
        if phase != previous_phase:
            events.append({"event": f"phase_{phase}", "t": frame["t"], "phase": phase,
                           "wheel_gap": _plain(frame["wheel_gap"]),
                           "normal_forces": _plain(frame["normal_forces"]),
                           "contact": _plain(frame["contact_estimator"]),
                           "pitch": frame["pitch"], "pitch_rate": frame["pitch_rate"]})
            previous_phase = phase
        transition = frame.get("phase_transition")
        if transition is not None:
            events.append({"event": "phase_transition", "t": frame["t"],
                           "phase": phase, "transition": _plain(transition)})
    def first(predicate, name):
        for frame in frames:
            if predicate(frame):
                return {"event": name, "t": frame["t"], "phase": (frame.get("jump") or {}).get("phase"),
                        "wheel_gap": _plain(frame["wheel_gap"]),
                        "normal_forces": _plain(frame["normal_forces"]),
                        "contact": _plain(frame["contact_estimator"]),
                        "decision_mode": frame["decision"]["mode"]}
        return {"event": name, "t": None}
    events.extend([
        first(lambda f: (f.get("jump") or {}).get("phase") == "COMPRESS", "enter_compress"),
        first(lambda f: (f.get("jump") or {}).get("phase") == "THRUST", "enter_thrust"),
        first(lambda f: bool(np.any(np.asarray(f["wheel_gap"]) > 1.0e-3)), "wheel_gap_gt_1mm"),
        first(lambda f: (f.get("jump") or {}).get("phase") == "THRUST" and
                    float(np.mean(np.asarray(f["leg_state"])[[0, 2]])) >= release_end,
              "leg_length_ge_release_end"),
        first(lambda f: (f.get("jump") or {}).get("phase") == "FLIGHT", "enter_flight"),
        first(lambda f: bool(any(f["contact_geometry"])), "first_geometry_contact"),
        first(lambda f: bool(any(f["contact_estimator"])), "first_estimated_contact"),
        first(lambda f: bool(f["airborne_active"]), "airborne_active_true"),
        first(lambda f: f["decision"]["mode"] == "ground", "mode_ground"),
    ])
    return events


def _summary(frames, sim, release_end):
    gaps = np.asarray([f["wheel_gap"] for f in frames], dtype=float)
    leg = np.asarray([f["leg_state"] for f in frames], dtype=float)
    qpos = np.asarray([f["qpos"] for f in frames], dtype=float)
    qvel = np.asarray([f["qvel"] for f in frames], dtype=float)
    pitch = np.asarray([f["pitch"] for f in frames], dtype=float)
    pitch_rate = np.asarray([f["pitch_rate"] for f in frames], dtype=float)
    ctrl = np.asarray([f["ctrl"] for f in frames], dtype=float)
    phases = {}
    for frame in frames:
        phase = (frame.get("jump") or {}).get("phase", "NONE")
        phases[phase] = phases.get(phase, 0) + 1
    return {
        "simulated_seconds": float(sim.data.time),
        "stop_reason": sim.stop_reason,
        "frame_count": len(frames),
        "phase_frame_counts": phases,
        "max_abs_pitch_rad": float(np.max(np.abs(pitch))) if len(pitch) else None,
        "max_abs_pitch_rate_rad_s": float(np.max(np.abs(pitch_rate))) if len(pitch_rate) else None,
        "max_forward_displacement_m": float(np.max(np.abs(qpos[:, 0] - qpos[0, 0]))) if len(qpos) else None,
        "max_forward_velocity_m_s": float(np.max(np.abs(qvel[:, 0]))) if len(qvel) else None,
        "max_leg_length_m": float(np.max(leg[:, [0, 2]])) if len(leg) else None,
        "min_leg_length_m": float(np.min(leg[:, [0, 2]])) if len(leg) else None,
        "release_end_m": float(release_end),
        "max_positive_wheel_gap_m": float(np.max(gaps)) if len(gaps) else None,
        "max_abs_ctrl": _plain(np.max(np.abs(ctrl), axis=0)) if len(ctrl) else [],
        "wbc_calls": int(sum(bool(f["wbc_called"]) for f in frames)),
        "output_source_counts": {source: sum(f["output_source"] == source for f in frames)
                                 for source in sorted({f["output_source"] for f in frames})},
        "mpc_calls": int(sum(bool(f["mpc_called"]) for f in frames)),
        "airborne_frames": int(sum(bool(f["airborne_active"]) for f in frames)),
        "phase_transition_count": int(sum(f.get("phase_transition") is not None for f in frames)),
        "phase_transition_assertion_failures": int(sum(
            f.get("phase_transition") is not None
            and not f["phase_transition"]["assertion_passed"] for f in frames)),
        "thrust_command_built_frames": int(sum(bool(f.get("thrust_command_built")) for f in frames)),
        "thrust_applied_frames": int(sum(bool(f.get("thrust_applied")) for f in frames)),
    }


def _disturbance(case):
    if case in ("baseline", "command_velocity"):
        return ()
    if case == "delta_velocity":
        return (Disturbance(time=1.0, delta_velocity=(0.0, 0.0, 0.80)),)
    if case == "qfrc":
        return (Disturbance(time=1.0, force=(40.0, 0.0, 0.0),
                           torque=(0.0, 3.0, 0.0), duration=0.20),)
    raise ValueError(f"unsupported case: {case}")


def _command(case, velocity: float, hold: float,
             yaw_rate: float = 0.0, yaw_hold: float | None = None):
    """CMD（16.3）：与 ``tools/view.py`` 同签名的按时间定值回调，只用于有输入工况。

    ``0 ~ hold`` 给 ``velocity``，``hold`` 之后归零；其它 case 仍返回 ``default_command()``（零指令）。
    这是测试入口，不参与生产代码。
    """
    if case != "command_velocity":
        return default_command()

    yaw_until = float(hold) if yaw_hold is None else float(yaw_hold)

    def command(time_s: float) -> MotionCommand:
        return MotionCommand(velocity=(float(velocity) if float(time_s) < float(hold) else 0.0),
                             yaw_rate=(float(yaw_rate) if float(time_s) < yaw_until else 0.0))

    return command


def run_case(stage, case, run_id, duration, jump_height, jump_stroke, jump_time,
             explicit_jump_reference=False, compression_only=False, freeze_jump_reference=False,
             vmc_feedforward=False, leg_length_compression=False, leg_length_thrust=False,
             thrust_only=False, flight_zero_torque=False, flight_leg_retract=False,
             flight_leg_extend=False, mpc_force_feedforward=False, wheel_torque_limit=None,
             flight_leg_preswing=0.0, flight_dual_mass=False, release_ramp=False,
             flight_hold_momentum=False, jump_lock_horizontal=False,
             thrust_momentum_task=False, thrust_momentum_kh=0.0, thrust_momentum_khi=0.0,
             two_mass_takeoff=False, mpc_nominal_normal=False, thrust_h_t=-0.20,
             command_velocity=0.5, command_hold=3.0,
             command_yaw_rate=0.0, command_yaw_hold=None):
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / f"{stage}_{case}_{run_id}.json.gz"
    log_path = RESULTS / f"{stage}_{case}_{run_id}.log"
    setup_logging(default_level="INFO", log_file=log_path)
    config = SimulationConfig(jump_height=jump_height, jump_stroke=jump_stroke,
                              jump_time=jump_time, wheel_torque_limit=wheel_torque_limit,
                              flight_leg_preswing=flight_leg_preswing,
                              flight_dual_mass=bool(flight_dual_mass),
                              release_ramp=bool(release_ramp),
                              flight_hold_momentum=bool(flight_hold_momentum),
                              jump_lock_horizontal=bool(jump_lock_horizontal),
                              thrust_momentum_task=bool(thrust_momentum_task),
                              thrust_momentum_kh=float(thrust_momentum_kh),
                              thrust_momentum_khi=float(thrust_momentum_khi),
                              two_mass_takeoff=bool(two_mass_takeoff),
                              mpc_nominal_normal=bool(mpc_nominal_normal),
                              thrust_h_t=float(thrust_h_t))
    sim = Gbc486MpcWbcSimulation(config)
    frames = []
    sim.run(_command(case, command_velocity, command_hold,
                     yaw_rate=command_yaw_rate, yaw_hold=command_yaw_hold), duration,
            disturbances=_disturbance(case), trace_sink=frames.append,
            explicit_jump_reference=explicit_jump_reference, compression_only=compression_only,
            freeze_jump_reference=freeze_jump_reference, vmc_feedforward=vmc_feedforward,
            leg_length_compression=leg_length_compression, leg_length_thrust=leg_length_thrust,
            thrust_only=thrust_only, flight_zero_torque=flight_zero_torque,
            flight_leg_retract=flight_leg_retract, flight_leg_extend=flight_leg_extend,
            mpc_force_feedforward=mpc_force_feedforward)
    release_end = float(sim.jump.config.release_end) if sim.jump is not None else float(config.nominal_length)
    document = {
        "metadata": {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "workspace": str(PROJECT),
            "git_commit": _git_commit(),
            "git_status_short_raw": _git_status(),
            "python": sys.version,
            "platform": platform.platform(),
            "mujoco": __import__("mujoco").__version__,
            "numpy": np.__version__,
            "case": case,
            "run_id": run_id,
            "command": ("default_command()" if case != "command_velocity"
                        else f"command_velocity(velocity={command_velocity}, hold={command_hold})"),
            "command_velocity": (None if case != "command_velocity" else float(command_velocity)),
            "command_hold_s": (None if case != "command_velocity" else float(command_hold)),
            "config": {
                "backend": config.backend, "model_variant": config.model_variant,
                "preset": config.preset, "nominal_length_m": config.nominal_length,
                "jump_height_m": jump_height, "jump_stroke_m": jump_stroke,
                "jump_time_s": jump_time, "duration_s": duration,
            },
            "disturbance": _plain(_disturbance(case)),
            "explicit_jump_reference": bool(explicit_jump_reference),
            "vmc_feedforward": bool(vmc_feedforward),
            "leg_length_compression": bool(leg_length_compression),
            "leg_length_thrust": bool(leg_length_thrust),
            "thrust_only": bool(thrust_only),
            "flight_zero_torque": bool(flight_zero_torque),
            "flight_leg_retract": bool(flight_leg_retract),
            "flight_leg_extend": bool(flight_leg_extend),
            "mpc_force_feedforward": bool(mpc_force_feedforward),
            "wheel_torque_limit": (None if wheel_torque_limit is None else float(wheel_torque_limit)),
            "viewer_xfrc_applied_executed": False,
            "viewer_xfrc_applied_note": "GUI viewer path is not executed by this headless tool; xfrc_applied remains separately recorded.",
            "units": {
                "t": "s", "qpos": "model coordinates; base x/y/z in m",
                "qvel": "model coordinates per s", "qacc": "model coordinates per s2",
                "ctrl": "N*m actuator command", "wheel_gap": "m signed lowest wheel point z",
                "normal_forces": "N", "tangent_forces": "N", "leg_state": "[m, rad, m, rad]",
                "leg_rate": "[m/s, rad/s, m/s, rad/s]",
                "com_xz": "[m, m] world coordinates [x, z]",
                "com_vxz": "[m/s, m/s] world coordinates [vx, vz]",
                "wheel_hub_xz": "[[m, m], [m, m]] world coordinates [x, z], right then left",
            },
        },
        "summary": _summary(frames, sim, release_end),
        "events": _events(frames, release_end),
        "frames": _plain(frames),
    }
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, separators=(",", ":"))
    _LOGGER.info("diagnostic result written: %s", output)
    _LOGGER.info("summary: %s", json.dumps(document["summary"], ensure_ascii=False, sort_keys=True))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="P0")
    parser.add_argument("--case", choices=("baseline", "delta_velocity", "qfrc", "command_velocity"),
                        default="baseline")
    # CMD（16.3）：有输入速度指令工况；其它 case 忽略这两个参数。
    parser.add_argument("--command-velocity", type=float, default=0.5,
                        help="CMD：0~hold 期间的速度指令 [m/s]")
    parser.add_argument("--command-hold", type=float, default=3.0,
                        help="CMD：速度指令保持时长 [s]（之后归零）")
    # 第 26 节 CASE-B/C 需要偏航指令入口（25.4 曾加过、后被回退，这里按原样恢复）。
    parser.add_argument("--command-yaw-rate", type=float, default=0.0,
                        help="CMD：0~yaw_hold 期间的偏航角速度指令 [rad/s]，默认 0 保持原行为")
    parser.add_argument("--command-yaw-hold", type=float, default=None,
                        help="CMD：偏航指令保持时长 [s]；缺省沿用 --command-hold")
    parser.add_argument("--run-id", default="run01")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--jump-height", type=float, default=0.10)
    parser.add_argument("--jump-stroke", type=float, default=0.12)
    parser.add_argument("--jump-time", type=float, default=1.0)
    parser.add_argument("--no-jump", action="store_true")
    parser.add_argument("--explicit-jump-reference", action="store_true")
    parser.add_argument("--compression-only", action="store_true",
                        help="WP5：压缩完成即停止，不执行 THRUST/FLIGHT/LANDING")
    parser.add_argument("--freeze-jump-reference", action="store_true",
                        help="B组实验：压缩期 pitch/roll/angle_R/angle_L 用跳跃开始时的冻结参考")
    parser.add_argument("--vmc-feedforward", action="store_true",
                        help="VP1：COMPRESS/COMPRESS_BRAKE 用等效力前馈替代 com_vertical 任务")
    parser.add_argument("--leg-length-compression", action="store_true",
                        help="VL1：COMPRESS/COMPRESS_BRAKE 用腿长行加速度驱动，不创建 com_vertical")
    parser.add_argument("--leg-length-thrust", action="store_true",
                        help="TR1：THRUST 用腿长行加速度驱动，不创建 com_vertical")
    parser.add_argument("--thrust-only", action="store_true",
                        help="TR1：相位将进入 FLIGHT 时停止，不执行 FLIGHT/LANDING 控制")
    parser.add_argument("--flight-zero-torque", action="store_true",
                        help="FL1：FLIGHT 或真离地时输出显式零力矩，不保持地面力矩")
    parser.add_argument("--flight-leg-retract", action="store_true",
                        help="FL2：几何离地时主动收腿（pd = 3000/150），抬高轮毂")
    parser.add_argument("--flight-leg-preswing", type=float, default=0.0,
                        help="[rad] 触地前腿角终端预置（伸腿进度 0->1 生效），0=原行为")
    parser.add_argument("--flight-dual-mass", action="store_true",
                        help="S1：飞行段改用双质量块单轨迹规划（默认关）")
    parser.add_argument("--release-ramp", action="store_true",
                        help="S2：THRUST 末段 25 ms 把 a_z 斜坡降到离地值（默认关）")
    parser.add_argument("--flight-hold-momentum", action="store_true",
                        help="S5-a：飞行段切向参考改为保持离地总角动量（默认关）")
    parser.add_argument("--jump-lock-horizontal", action="store_true",
                        help="S6：压缩段锁定水平参考（质心参考放到轮触地点正上方，默认关）")
    parser.add_argument("--thrust-momentum-task", action="store_true",
                        help="10.48 B1：THRUST 段用角动量行+差模腿角行替换角度/前向速度行")
    parser.add_argument("--thrust-momentum-kh", type=float, default=0.0,
                        help="[1/s] 角动量回路比例 k_H（辨识参数）")
    parser.add_argument("--thrust-momentum-khi", type=float, default=0.0,
                        help="[1/s^2] 角动量回路积分 k_HI（辨识参数）")
    parser.add_argument("--two-mass-takeoff", action="store_true",
                        help="S9（10.54）：THRUST 两块计划 + 飞行段方案 (b)")
    parser.add_argument("--mpc-nominal-normal", action="store_true",
                        help="10.70 A：MPC 逐轮力上限用名义法向力 M*g/2（不用单拍采样 N）")
    parser.add_argument("--thrust-h-t", type=float, default=-0.20,
                        help="10.72 C：离地 H 目标 [kg*m^2/s]，默认 -0.20")
    parser.add_argument("--flight-leg-extend", action="store_true",
                        help="FL3：飞行腿剖面（上升收腿/下落伸腿）+ LANDING 缓冲")
    parser.add_argument("--mpc-force-feedforward", action="store_true",
                        help="P-b：把 MPC 计划轮力经 tau=f*R 前馈进 WBC（默认关闭）")
    parser.add_argument("--wheel-torque-limit", type=float, default=None,
                        help="WLIM：覆盖轮执行器限幅 [N*m]，None=沿 MJCF")
    # SW1：仅覆盖飞行收腿的两个常量，默认 None 表示不改动任何行为。
    parser.add_argument("--flight-retract-floor", type=float, default=None,
                        help="SW1：只覆盖 control.FLIGHT_RETRACT_FLOOR，None=不改")
    parser.add_argument("--flight-retract-torque-limit", type=float, default=None,
                        help="SW1：只覆盖 control.FLIGHT_RETRACT_TORQUE_LIMIT，None=不改")
    args = parser.parse_args()
    if args.flight_retract_floor is not None:
        _control.FLIGHT_RETRACT_FLOOR = float(args.flight_retract_floor)
    if args.flight_retract_torque_limit is not None:
        _control.FLIGHT_RETRACT_TORQUE_LIMIT = float(args.flight_retract_torque_limit)
    # 10.48 B1：角动量任务必须与腿长加速度通道同用；与 S6 水平锁定互斥。
    if args.thrust_momentum_task and not args.leg_length_thrust:
        parser.error("--thrust-momentum-task 需要同时打开 --leg-length-thrust")
    if args.two_mass_takeoff:
        for _flag, _value in (("--flight-dual-mass", args.flight_dual_mass),
                              ("--leg-length-thrust", args.leg_length_thrust)):
            if not _value:
                parser.error("--two-mass-takeoff 需要同时打开 %s" % _flag)
        for _flag, _value in (("--thrust-momentum-task", args.thrust_momentum_task),
                              ("--jump-lock-horizontal", args.jump_lock_horizontal),
                              ("--flight-hold-momentum", args.flight_hold_momentum)):
            if _value:
                parser.error("--two-mass-takeoff 与 %s 互斥" % _flag)
    if args.thrust_momentum_task and args.jump_lock_horizontal:
        parser.error("--thrust-momentum-task 与 --jump-lock-horizontal 不能同时打开")
    jump_height = None if args.no_jump else args.jump_height
    run_case(args.stage, args.case, args.run_id, args.duration, jump_height, args.jump_stroke,
             args.jump_time, explicit_jump_reference=args.explicit_jump_reference,
             flight_leg_preswing=args.flight_leg_preswing,
             flight_dual_mass=args.flight_dual_mass,
             release_ramp=args.release_ramp,
             flight_hold_momentum=args.flight_hold_momentum,
             jump_lock_horizontal=args.jump_lock_horizontal,
             thrust_momentum_task=args.thrust_momentum_task,
             thrust_momentum_kh=args.thrust_momentum_kh,
             thrust_momentum_khi=args.thrust_momentum_khi,
             two_mass_takeoff=args.two_mass_takeoff,
             mpc_nominal_normal=args.mpc_nominal_normal,
             thrust_h_t=args.thrust_h_t,
             compression_only=args.compression_only,
             freeze_jump_reference=args.freeze_jump_reference,
             vmc_feedforward=args.vmc_feedforward,
             leg_length_compression=args.leg_length_compression,
             leg_length_thrust=args.leg_length_thrust,
             thrust_only=args.thrust_only,
             flight_zero_torque=args.flight_zero_torque,
             flight_leg_retract=args.flight_leg_retract,
             flight_leg_extend=args.flight_leg_extend,
             mpc_force_feedforward=args.mpc_force_feedforward,
             wheel_torque_limit=args.wheel_torque_limit,
             command_velocity=args.command_velocity,
             command_yaw_rate=args.command_yaw_rate,
             command_yaw_hold=args.command_yaw_hold,
             command_hold=args.command_hold)


if __name__ == "__main__":
    main()