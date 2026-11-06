"""独立的力方向、执行器映射和接触相位提示核查工具。"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from gbc_mpc_wbc.logging_setup import setup_logging
from gbc_mpc_wbc.simulation import Gbc486MpcWbcSimulation, SimulationConfig, default_command
from gbc_mpc_wbc import robot

_LOGGER = logging.getLogger(__name__)
RESULTS = ROOT / "results"


class PhaseHintSpy:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.observations = []

    def update(self, normals, dt, **kwargs):
        phase = kwargs.get("phase")
        self.observations.append({
            "index": len(self.observations),
            "phase": getattr(phase, "value", str(phase)),
            "normals": np.asarray(normals, dtype=float).tolist(),
            "dt_s": float(dt),
        })
        return self.wrapped.update(normals, dt, **kwargs)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def config():
    return SimulationConfig(jump_height=0.10, jump_stroke=0.12, jump_time=1.0,
                            backend="wls", model_variant="held_leg", preset="tuned-agile")


def angle_dict(sim, leg):
    model, data = sim.model, sim.data
    names = (leg.front_joint, leg.rear_joint) + leg.passive_joints
    return {name: float(data.qpos[model.joint(name).qposadr[0]]) for name in names}


def mapping_audit(sim, force):
    dyn = sim.dyn
    dyn.leg_state()
    legs = []
    residuals = []
    tau_all = dyn.thrust_torque((force, force))[0]
    for side, leg in enumerate(robot.LEGS):
        kin = dyn._leg_kinematics[leg.name]
        angles = angle_dict(sim, leg)
        length, phi = kin.lengths(angles)
        wrench = np.array([force * np.cos(phi), -force * length * np.sin(phi)])
        expected = kin.jacobian(angles).T @ wrench
        ids = dyn.leg_actuators[2 * side:2 * side + 2]
        actual = np.asarray(tau_all[list(ids)], dtype=float)
        residuals.append(float(np.max(np.abs(actual - expected))))
        legs.append({
            "side": leg.name,
            "actuator_ids": [int(v) for v in ids],
            "actuator_names": [sim.model.actuator(int(v)).name for v in ids],
            "kinematics_actuated_order": list(kin.actuated),
            "kinematics_passive_order": list(kin.passive),
            "length_m": float(length), "phi_rad": float(phi),
            "jacobian": kin.jacobian(angles), "wrench": wrench,
            "tau_expected": expected, "tau_actual": actual,
        })
    rows = []
    for actuator_id in dyn.leg_actuators:
        actuator_id = int(actuator_id)
        joint_id = int(sim.model.actuator_trnid[actuator_id, 0])
        nz = np.flatnonzero(np.abs(dyn.selection[actuator_id]) > 1.0e-12)
        rows.append({
            "actuator_id": actuator_id,
            "actuator_name": sim.model.actuator(actuator_id).name,
            "joint_id": joint_id,
            "joint_name": sim.model.joint(joint_id).name,
            "dof_indices": nz,
            "selection_values": dyn.selection[actuator_id, nz],
            "gear": float(sim.model.actuator_gear[actuator_id, 0]),
            "gainprm": float(sim.model.actuator_gainprm[actuator_id, 0]),
            "ctrlrange": sim.model.actuator_ctrlrange[actuator_id],
        })
    return {
        "leg_actuators": dyn.leg_actuators,
        "wheel_actuators": dyn.wheel_actuators,
        "selection_rows": rows,
        "legs": legs,
        "max_jacobian_residual_Nm": max(residuals, default=0.0),
    }


def force_measure(force):
    sim = Gbc486MpcWbcSimulation(config())
    model, data, dyn = sim.model, sim.data, sim.dyn
    tau, peak = dyn.thrust_torque((force, force))
    joint_ids = [int(model.actuator(int(a)).trnid[0]) for a in dyn.leg_actuators]
    q0 = np.asarray([data.qpos[model.jnt_qposadr[j]] for j in joint_ids], dtype=float)
    leg0 = dyn.leg_state().copy()
    points0 = robot.wheel_contact_points(model, data)
    base = int(model.body("base").id)
    com0 = np.asarray(data.subtree_com[base], dtype=float).copy()
    dt = float(model.opt.timestep)
    data.ctrl[:] = tau
    states = []
    for _ in range(2):
        mujoco.mj_step(model, data)
        states.append({
            "leg": dyn.leg_state().copy(),
            "points": robot.wheel_contact_points(model, data),
            "com": np.asarray(data.subtree_com[base], dtype=float).copy(),
            "q": np.asarray([data.qpos[model.jnt_qposadr[j]] for j in joint_ids], dtype=float),
        })
    v0 = (states[0]["com"] - com0) / dt
    v1 = (states[1]["com"] - states[0]["com"]) / dt
    return {
        "force_per_leg_N": force,
        "tau_Nm": tau,
        "peak_tau_Nm": peak,
        "leg_delta": states[-1]["leg"] - leg0,
        "leg_rate_after_second_step": (states[-1]["leg"] - states[-2]["leg"]) / dt,
        "wheel_gap_delta_m": states[-1]["points"][:, 2] - points0[:, 2],
        "com_velocity_after_second_step_m_s": v1,
        "com_acceleration_fd_m_s2": (v1 - v0) / dt,
        "actuated_joint_delta_rad": states[-1]["q"] - q0,
        "actuated_joint_rate_rad_s": (states[-1]["q"] - states[-2]["q"]) / dt,
    }


def phase_hint_audit():
    sim = Gbc486MpcWbcSimulation(config())
    spy = PhaseHintSpy(sim.contact_estimator)
    sim.contact_estimator = spy
    sim.run(default_command(), 1.3, explicit_jump_reference=True)
    counts = Counter(item["phase"] for item in spy.observations)
    return {
        "total_update_calls": len(spy.observations),
        "phase_counts": dict(counts),
        "first": spy.observations[:3],
        "last": spy.observations[-3:],
        "all_phase_none": all(item["phase"] == "none" for item in spy.observations),
    }


def run_case(run_id):
    RESULTS.mkdir(exist_ok=True)
    log_path = RESULTS / f"SIGN_MAPPING_{run_id}.log"
    out_path = RESULTS / f"SIGN_MAPPING_{run_id}.json"
    setup_logging(default_level="INFO", log_file=log_path)
    base = Gbc486MpcWbcSimulation(config())
    force = float(base.jump.config.leg_force())
    mapping = mapping_audit(base, force)
    positive = force_measure(force)
    negative = force_measure(-force)
    antisym = float(np.max(np.abs(np.asarray(positive["tau_Nm"]) + np.asarray(negative["tau_Nm"]))))
    phase = phase_hint_audit()
    document = {
        "metadata": {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "workspace": str(PROJECT), "python": sys.version,
            "platform": platform.platform(), "mujoco": mujoco.__version__,
            "numpy": np.__version__, "run_id": run_id,
            "force_source": "JumpConfig.leg_force(), h=0.10 m, stroke=0.12 m",
            "units": {"force": "N", "torque": "N*m", "leg_delta": "[m,rad,m,rad]",
                      "gap_delta": "m", "com_acceleration": "m/s^2",
                      "joint_delta": "rad", "joint_rate": "rad/s"},
        },
        "mapping_audit": plain(mapping),
        "positive_force": plain(positive), "negative_force": plain(negative),
        "tau_antisymmetry_max_Nm": antisym,
        "phase_hint_audit": phase,
    }
    out_path.write_text(json.dumps(plain(document), ensure_ascii=False, indent=2), encoding="utf-8")
    _LOGGER.info("mapping residual max=%.3e N*m", mapping["max_jacobian_residual_Nm"])
    _LOGGER.info("tau antisymmetry max=%.3e N*m", antisym)
    _LOGGER.info("positive leg_delta=%s gap_delta=%s com_acc=%s", positive["leg_delta"], positive["wheel_gap_delta_m"], positive["com_acceleration_fd_m_s2"])
    _LOGGER.info("negative leg_delta=%s gap_delta=%s com_acc=%s", negative["leg_delta"], negative["wheel_gap_delta_m"], negative["com_acceleration_fd_m_s2"])
    _LOGGER.info("phase_hint counts=%s all_none=%s", phase["phase_counts"], phase["all_phase_none"])
    _LOGGER.info("result written: %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run01")
    args = parser.parse_args()
    run_case(args.run_id)


if __name__ == "__main__":
    main()