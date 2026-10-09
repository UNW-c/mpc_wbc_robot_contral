"""E2/WP2: 固定接触下 WBC 质心竖直任务接入自检（A/B 对照）。

只验证一件事：显式 ``desired_com_acceleration_z`` 是否进入 WBC，生成唯一
``com_vertical`` 任务，并产生可记录的 目标/bias/实际值/残差 与力矩输出。

不改控制参数、增益、权重、阈值、MPC 参数；不推进物理；不做完整跳跃。
输出：results/E2_com_vertical_task_<date>_run01.json / .log（绝对路径见日志）。
用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/e2_com_vertical_task.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gbc_mpc_wbc import _circleg  # noqa: F401,E402  (先把 circleg 根目录加入 sys.path)
from gbc_mpc_wbc.control import wbc_stage  # noqa: E402
from gbc_mpc_wbc.sampling import FeedbackSampler  # noqa: E402
from gbc_mpc_wbc.simulation import Gbc486MpcWbcSimulation, SimulationConfig  # noqa: E402
from Algorithm.contact_state import JumpPhaseHint  # noqa: E402
from Algorithm.supervision import ControlMode  # noqa: E402

RESULTS = ROOT / "results"
RESULT_PATH = RESULTS / "E2_com_vertical_task_20261009_run01.json"
LOG_PATH = RESULTS / "E2_com_vertical_task_20261009_run01.log"

# A/B 诊断输入：B 组使用的显式质心竖直加速度目标 [m/s^2]。
# 这是测试输入，不是控制参数/增益。取 +1.0 的理由：明显非零以便判读方向，
# 且远小于 WBC 的 acceleration_limit = 50（不构成限幅测试）。
A_Z_TARGET = 1.0


def build_sim() -> Gbc486MpcWbcSimulation:
    """与正式跳跃同一后端/机型/preset 的静止装配，不启用 jump。"""
    return Gbc486MpcWbcSimulation(SimulationConfig(
        backend="wls", model_variant="held_leg", preset="tuned-agile", nominal_length=0.28))


def make_sampler(sim: Gbc486MpcWbcSimulation) -> FeedbackSampler:
    return FeedbackSampler(
        model=sim.model, data=sim.data, dynamics=sim.dyn, tasks=sim.tasks,
        contact_estimator=sim.contact_estimator, supervisor=sim.supervisor,
        reduced=sim.reduced, nominal_angle=sim.pose.angle,
        pitch_limit=sim.mpc.settings.pitch_limit)


def make_feedback(sim: Gbc486MpcWbcSimulation) -> tuple[object, int]:
    """固定站立位形一拍反馈，并把监督层推进到 GROUND（否则 motion 任务被禁用）。

    推进只用监督层自己的 recovery_dwell（0.10 s）与合成时间戳，**不推进物理**，
    因此位形与接触拓扑保持固定。返回 (feedback, 推进拍数)。
    """
    sampler = make_sampler(sim)
    feedback = sampler.update(distance=0.0, phase_hint=JumpPhaseHint.NONE)
    advances = 0
    if feedback.decision.mode != ControlMode.GROUND:
        dt = float(sim.model.opt.timestep)
        t = float(sim.data.time)
        roll, pitch, _ = sim.dyn.attitude()
        stable = max(abs(roll), abs(pitch)) < sim.mpc.settings.pitch_limit
        for _ in range(int(math.ceil(sim.supervisor.recovery_dwell / dt)) + 2):
            t += dt
            decision = sim.supervisor.update(
                timestamp=t, dt=dt, contact=(True, True), enabled=True,
                command_fresh=True, stable=stable, slipping=False)
            advances += 1
            if decision.mode == ControlMode.GROUND:
                feedback = replace(feedback, decision=decision)
                break
    return feedback, advances


def call_wbc(sim: Gbc486MpcWbcSimulation, feedback, target: float | None):
    """单次 WBC 调用；两次 A/B 使用同一 feedback（同状态、同接触、同参数）。"""
    counters = dict(wbc_calls=0, wbc_fallback=0, wbc_clip=0, wbc_infeasible=0)
    return wbc_stage(
        control=sim.control, tasks=sim.tasks, dynamics=sim.dyn, mpc=sim.mpc,
        config=sim.config, feedback=feedback, solution=None, feedback_accum=sim.feedback,
        time_s=float(sim.data.time), timing=sim.timing, timings=dict(mpc=[], wbc=[]),
        counters=counters, exhausted={}, scaled={}, normal_violation_time=0.0,
        infeasible_active=False, jump=None, desired_com_acceleration_z=target)


def snapshot(sim: Gbc486MpcWbcSimulation, feedback, outcome, target: float | None) -> dict:
    result = outcome.result
    tasks = outcome.task_trace
    names = [task["name"] for task in tasks]
    com = next((task for task in tasks if task["name"] == "com_vertical"), None)
    row, bias, com_z, com_vz = sim.dyn.com_vertical_row()
    qacc = np.asarray(result.qacc, dtype=float) if result is not None else np.zeros(sim.model.nv)
    achieved = float(row @ qacc + float(bias))
    # 分解：硬约束基线（接触+欠驱动力学决定的 qacc_hard）与任务修正量，二者相加得 qacc。
    # 这是判断“目标被硬约束抵消”还是“被其它任务权重分摊”的无参数证据。
    hard_baseline = None
    correction = None
    if result is not None and len(result.hierarchy.increments) >= 2:
        qacc_hard = np.asarray(result.hierarchy.increments[0], dtype=float)
        delta = np.asarray(result.hierarchy.increments[1], dtype=float)
        hard_baseline = float(row @ qacc_hard + float(bias))
        correction = float(row @ delta)
    roll, pitch, yaw = sim.dyn.attitude()
    return {
        "desired_com_acceleration_z_input": target,
        "task_names": names,
        "com_vertical_present": com is not None,
        "com_vertical_desired": None if com is None else np.asarray(com["desired_acceleration"]).tolist(),
        "com_vertical_bias": None if com is None else np.asarray(com["bias_acceleration"]).tolist(),
        "com_vertical_achieved": None if com is None else np.asarray(com["achieved_acceleration"]).tolist(),
        "com_vertical_residual": None if com is None else np.asarray(com["residual"]).tolist(),
        "com_vertical_jacobian": None if com is None else np.asarray(com["jacobian"]).tolist(),
        "J_com_z_shape": list(np.asarray(row, dtype=float).shape),
        "bias_com_z": float(bias),
        "com_z": float(com_z),
        "com_vz": float(com_vz),
        "achieved_com_acceleration_z": achieved,
        "com_acceleration_hard_baseline": hard_baseline,
        "com_acceleration_task_correction": correction,
        "qacc": qacc.tolist(),
        "data_ctrl": np.asarray(outcome.torque, dtype=float).tolist(),
        "wbc_raw_torque": np.asarray(outcome.raw_torque, dtype=float).tolist(),
        "tau_jump": np.asarray(outcome.tau_jump, dtype=float).tolist(),
        "predicted_normal": np.asarray(outcome.predicted_normal, dtype=float).tolist(),
        "measured_normal_forces": np.asarray(feedback.forces, dtype=float)[:, 2].tolist(),
        "feasible": None if result is None else bool(result.feasible),
        "clip_applied": None if result is None else bool(result.clip_applied),
        "scaling": None if result is None else {k: float(v) for k, v in result.scaling.items()},
        "dynamics_residual_norm": None if result is None else float(result.dynamics_residual_norm),
        "contact_residual_norm": None if result is None else float(result.hierarchy.contact_residual_norm),
        "pitch": float(pitch),
        "pitch_rate": float(feedback.state.pitch_rate),
        "contact": [bool(c) for c in feedback.decision.contact],
        "contact_topology": list(feedback.decision.contact),
    }


def run() -> dict:
    sim = build_sim()
    feedback, supervisor_advances = make_feedback(sim)
    # 单工况冒烟：一次 A、一次 B 必须跑通且返回结果对象
    a1 = call_wbc(sim, feedback, None)
    b1 = call_wbc(sim, feedback, A_Z_TARGET)
    # 同输入重复性：同一 feedback 再各调一次
    a2 = call_wbc(sim, feedback, None)
    b2 = call_wbc(sim, feedback, A_Z_TARGET)
    snap = {
        "A_run1": snapshot(sim, feedback, a1, None),
        "A_run2": snapshot(sim, feedback, a2, None),
        "B_run1": snapshot(sim, feedback, b1, A_Z_TARGET),
        "B_run2": snapshot(sim, feedback, b2, A_Z_TARGET),
    }
    repeat_torque = float(max(
        np.max(np.abs(np.asarray(snap["A_run1"]["data_ctrl"]) - np.asarray(snap["A_run2"]["data_ctrl"]))),
        np.max(np.abs(np.asarray(snap["B_run1"]["data_ctrl"]) - np.asarray(snap["B_run2"]["data_ctrl"])))))
    repeat_qacc = float(max(
        np.max(np.abs(np.asarray(snap["A_run1"]["qacc"]) - np.asarray(snap["A_run2"]["qacc"]))),
        np.max(np.abs(np.asarray(snap["B_run1"]["qacc"]) - np.asarray(snap["B_run2"]["qacc"])))))
    a_com = snap["A_run1"]
    b_com = snap["B_run1"]
    checks = {
        "a_has_no_com_vertical": not a_com["com_vertical_present"],
        "b_has_single_com_vertical": sum(1 for n in b_com["task_names"] if n == "com_vertical") == 1,
        "b_target_matches_input": (b_com["com_vertical_desired"] is not None
                                   and abs(b_com["com_vertical_desired"][0] - A_Z_TARGET) < 1e-12),
        "b_jacobian_dim_matches_nv": list(np.asarray(b_com["com_vertical_jacobian"]).shape)
        == [1, int(sim.model.nv)],
        "b_achieved_and_residual_recorded": (b_com["com_vertical_achieved"] is not None
                                             and b_com["com_vertical_residual"] is not None),
        "both_feasible": bool(a_com["feasible"]) and bool(b_com["feasible"]),
        "repeatable": repeat_torque <= 1.0e-15 and repeat_qacc <= 1.0e-15,
        "finite": all(np.all(np.isfinite(np.asarray(snap[k]["data_ctrl"], dtype=float)))
                      and np.all(np.isfinite(np.asarray(snap[k]["qacc"], dtype=float)))
                      for k in snap),
        "same_contact_topology": a_com["contact_topology"] == b_com["contact_topology"],
    }
    checks["passed"] = all(checks.values())
    delta = float(b_com["achieved_com_acceleration_z"] - a_com["achieved_com_acceleration_z"])
    return {
        "metadata": {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "workspace": str(ROOT.parent),
            "python": sys.version,
            "numpy": np.__version__,
            "nominal_length_m": float(sim.config.nominal_length),
            "backend": str(sim.config.backend),
            "model_variant": str(sim.config.model_variant),
            "preset": str(sim.config.preset),
            "a_z_target_m_s2": A_Z_TARGET,
            "reference_source": "explicit diagnostic argument, not MPC sample",
        },
        "checks": checks,
        "decision": {
            "mode": str(feedback.decision.mode.value),
            "motion_enabled": bool(feedback.decision.motion_enabled),
            "contact": list(feedback.decision.contact),
            "supervisor_advances": int(supervisor_advances),
        },
        "repeat_metrics": {"max_abs_torque_diff": repeat_torque, "max_abs_qacc_diff": repeat_qacc},
        "comparison": {
            "achieved_com_acceleration_z_A": a_com["achieved_com_acceleration_z"],
            "achieved_com_acceleration_z_B": b_com["achieved_com_acceleration_z"],
            "delta_achieved_B_minus_A": delta,
            "com_vertical_desired_B": b_com["com_vertical_desired"],
            "com_vertical_residual_B": b_com["com_vertical_residual"],
            "direction_consistent": delta > 0.0,
        },
        "snapshots": snap,
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("e2_com_vertical_task")
    try:
        result = run()
    except Exception:
        logger.exception("E2 WBC com_vertical task check failed")
        return 1
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("E2 result written: %s", RESULT_PATH)
    logger.info("E2 checks: %s", result["checks"])
    logger.info("E2 comparison: %s", result["comparison"])
    return 0 if result["checks"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
