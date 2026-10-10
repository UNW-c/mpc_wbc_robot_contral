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
    if case == "baseline":
        return ()
    if case == "delta_velocity":
        return (Disturbance(time=1.0, delta_velocity=(0.0, 0.0, 0.80)),)
    if case == "qfrc":
        return (Disturbance(time=1.0, force=(40.0, 0.0, 0.0),
                           torque=(0.0, 3.0, 0.0), duration=0.20),)
    raise ValueError(f"unsupported case: {case}")


def run_case(stage, case, run_id, duration, jump_height, jump_stroke, jump_time,
             explicit_jump_reference=False, compression_only=False, freeze_jump_reference=False,
             vmc_feedforward=False, leg_length_compression=False, leg_length_thrust=False,
             thrust_only=False, flight_zero_torque=False, flight_leg_retract=False,
             flight_leg_extend=False, release_phi_feedforward=False,
             release_phi_force_feedforward=False, release_reanchor=False):
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / f"{stage}_{case}_{run_id}.json.gz"
    log_path = RESULTS / f"{stage}_{case}_{run_id}.log"
    setup_logging(default_level="INFO", log_file=log_path)
    config = SimulationConfig(jump_height=jump_height, jump_stroke=jump_stroke,
                              jump_time=jump_time)
    sim = Gbc486MpcWbcSimulation(config)
    frames = []
    sim.run(default_command(), duration, disturbances=_disturbance(case), trace_sink=frames.append,
            explicit_jump_reference=explicit_jump_reference, compression_only=compression_only,
            freeze_jump_reference=freeze_jump_reference, vmc_feedforward=vmc_feedforward,
            leg_length_compression=leg_length_compression, leg_length_thrust=leg_length_thrust,
            thrust_only=thrust_only, flight_zero_torque=flight_zero_torque,
            flight_leg_retract=flight_leg_retract, flight_leg_extend=flight_leg_extend,
            release_phi_feedforward=release_phi_feedforward,
            release_phi_force_feedforward=release_phi_force_feedforward,
            release_reanchor=release_reanchor)
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
            "command": "default_command()",
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
            "release_phi_feedforward": bool(release_phi_feedforward),
            "release_phi_force_feedforward": bool(release_phi_force_feedforward),
            "release_reanchor": bool(release_reanchor),
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
    parser.add_argument("--case", choices=("baseline", "delta_velocity", "qfrc"), default="baseline")
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
    parser.add_argument("--flight-leg-extend", action="store_true",
                        help="FL3：飞行腿剖面（上升收腿/下落伸腿）+ LANDING 缓冲")
    parser.add_argument("--release-phi-feedforward", action="store_true",
                        help="FL15B1a：THRUST 段给 angle_R/L 参考叠加斜坡腿摆角前馈")
    parser.add_argument("--release-phi-force-feedforward", action="store_true",
                        help="FL15B2：THRUST 段给腿旋量加 phi 力前馈（B1 备用入口）")
    parser.add_argument("--release-reanchor", action="store_true",
                        help="FL17B：THRUST 段双参考重锚定（预倾 pitch 轨迹 + 世界系腿锚定）")
    args = parser.parse_args()
    jump_height = None if args.no_jump else args.jump_height
    run_case(args.stage, args.case, args.run_id, args.duration, jump_height, args.jump_stroke,
             args.jump_time, explicit_jump_reference=args.explicit_jump_reference,
             compression_only=args.compression_only,
             freeze_jump_reference=args.freeze_jump_reference,
             vmc_feedforward=args.vmc_feedforward,
             leg_length_compression=args.leg_length_compression,
             leg_length_thrust=args.leg_length_thrust,
             thrust_only=args.thrust_only,
             flight_zero_torque=args.flight_zero_torque,
             flight_leg_retract=args.flight_leg_retract,
             flight_leg_extend=args.flight_leg_extend,
             release_phi_feedforward=args.release_phi_feedforward,
             release_phi_force_feedforward=args.release_phi_force_feedforward,
             release_reanchor=args.release_reanchor)


if __name__ == "__main__":
    main()