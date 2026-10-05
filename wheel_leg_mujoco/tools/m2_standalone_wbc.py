"""M2：WBC 单独运行（不接 MPC），L0 = 0.42 零速站立 10 s，三个后端对照。

只验证可行性，不调参（AGENTS.md §1）。增益 / 权重为 circleg 默认值的直接搬用。
用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/m2_standalone_wbc.py [--duration 10] [--backends nullspace wls qp]
输出：results/m2_standalone_wbc.md / .json，以及每个后端的 250 Hz 时序 results/m2_<backend>.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gbc_mpc_wbc import robot as R  # noqa: E402
from gbc_mpc_wbc.dynamics import Gbc486Dynamics  # noqa: E402
from gbc_mpc_wbc.simulation import SimulationConfig, Gbc486MpcWbcSimulation  # noqa: E402
from gbc_mpc_wbc.wbc import BACKENDS  # noqa: E402

RESULTS = ROOT / "results"
NOMINAL_L0 = 0.42
WBC_DT = 0.004
# 停机判据（测试保护，不是控制参数）：取自 circleg SimulationConfig 的同名量
ATTITUDE_GUARD = 0.785   # [rad]
FORCE_TOLERANCE = 1e-6   # 预测法向力负值容差 [N]


from gbc_mpc_wbc.simulation import Gbc486MpcWbcSimulation, SimulationConfig  # noqa: E402
from Algorithm.supervision import ControlDecision, ControlMode  # noqa: E402
from Algorithm.wbc import ContactElimination  # noqa: E402

RESULTS = ROOT / "results"
NOMINAL_L0 = 0.42
WBC_DT = 0.004
# 停机判据（测试保护，不是控制参数）：取自 circleg SimulationConfig 的同名量
ATTITUDE_GUARD = 0.785   # [rad]
FORCE_TOLERANCE = 1e-6   # 预测法向力负值容差 [N]


def run(backend: str, duration: float):
    """WBC 单独运行：照 circleg 的调用方式，但不调用 solve_mpc（sample = None）。

    任务构造、接触块、求解器注入、限值都与闭环一致，只有 MPC 参考缺省：
    等价于 circleg ``tasks(..., sample=None)`` 那条分支。
    """
    sim = Gbc486MpcWbcSimulation(SimulationConfig(backend=backend))
    model, data, dyn, control = sim.model, sim.data, sim.dyn, sim.control
    every = int(round(WBC_DT / model.opt.timestep))
    steps = int(round(duration / model.opt.timestep))
    records, stop = [], None
    counters = dict(wbc_calls=0, fallback=0, clip=0, infeasible=0, solver_error=0)
    task_names = tuple(sim.exhausted)
    exhausted = {name: 0 for name in task_names}
    scaled = {name: 0 for name in task_names}
    solve_times, limit_hits = [], set()
    for step in range(steps):
        if step % every == 0:
            mujoco.mj_forward(model, data)
            contact, forces, counts, other = dyn.sensed_contact()
            state, rows = sim.chassis_state(contact, forces)
            roll, pitch, _ = dyn.attitude()
            decision = sim.supervisor.update(
                timestamp=float(data.time), dt=WBC_DT, contact=contact, enabled=True,
                command_fresh=True, stable=max(abs(roll), abs(pitch)) < sim.mpc.settings.pitch_limit,
                slipping=False)
            balance, motion = sim.tasks.tasks(state, rows, None)
            blocks = sim.tasks.contact_blocks(rows, contact)
            _, result = control.solve_wbc(
                mass=dyn.mass(), selection=dyn.selection, dynamics_bias=dyn.dynamics_bias(),
                contacts=blocks, balance_tasks=balance, motion_tasks=motion, limits=dyn.limits,
                decision=decision, dt=WBC_DT)
            torque = np.zeros(model.nu) if result is None else np.asarray(result.torque, dtype=float)
            counters["wbc_calls"] += 1
            reason = None
            predicted = np.zeros(0)
            if result is not None:
                counters["fallback"] += int(result.fallback_used)
                counters["clip"] += int(result.clip_applied)
                counters["infeasible"] += int(not result.feasible)
                exhausted_now = set()
                for outcome in result.hierarchy.outcomes:
                    exhausted[outcome.name] += int(outcome.exhausted)
                    if outcome.exhausted:
                        exhausted_now.add(outcome.name)
                for name, value in result.scaling.items():
                    if name not in exhausted_now:
                        scaled[name] += int(value < 1.0 - 1e-8)
                active = [b for b in blocks if all(contact[i] for i in b.wheels)]
                jc = np.vstack([b.jacobian for b in active])
                predicted = ContactElimination.from_matrices(
                    jc, dyn.selection, dyn.mass()).recover_contact_force(
                    result.qacc, dyn.dynamics_bias(), torque)
                if not result.feasible:
                    reason = "wbc_equality_or_torque_failure"
            normals = predicted[list(sim.tasks.normal_indices(contact))] if predicted.size else np.zeros(0)
            data.ctrl[:] = torque
            if reason is None and other:
                reason = "unexpected_body_contact"
            if reason is None and normals.size and float(normals.min()) < -FORCE_TOLERANCE:
                reason = "negative_predicted_normal_force"
            if reason is None and not dyn.limits.contains(torque):
                reason = "torque_limit"
            roll, pitch, yaw = dyn.attitude()
            leg = dyn.leg_state()
            residual = {} if result is None else {o.name: float(o.residual_norm)
                                                 for o in result.hierarchy.outcomes}
            records.append(dict(
                t=round(float(data.time), 6), contact=list(contact), counts=counts.tolist(),
                roll=float(roll), pitch=float(pitch), yaw=float(yaw),
                x=float(data.qpos[0]), y=float(data.qpos[1]), z=float(data.qpos[2]),
                v=float(np.asarray(np.zeros(1))[0]) if False else float(rows.task[0][0] @ data.qvel),
                leg=leg.tolist(), torque=torque.tolist(), sensed_normal=forces[:, 2].tolist(),
                predicted_normal=normals.tolist(), residual=residual,
                dynamics_residual=None if result is None else float(result.dynamics_residual_norm),
                contact_residual=None if result is None else float(result.hierarchy.contact_residual_norm),
                fallback=None if result is None else bool(result.fallback_used),
                feasible=None if result is None else bool(result.feasible),
                scaling={} if result is None else {k: float(v) for k, v in result.scaling.items()},
                conditioning={} if result is None or backend != "nullspace" else
                {o.name: float(o.conditioning) for o in result.hierarchy.outcomes},
                solve_time=0.0, error=None))
            if reason:
                stop = reason
                break
        mujoco.mj_step(model, data)
        hits = dyn.joint_limit_active()
        if hits:
            limit_hits.update(hits)
            stop = "joint_limit:" + ",".join(sorted(set(hits)))
            break
        if not np.all(np.isfinite(np.r_[data.qpos, data.qvel])):
            stop = "nonfinite_state"
            break
        if max(abs(v) for v in dyn.attitude()[:2]) > ATTITUDE_GUARD:
            stop = "attitude_guard"
            break
    return summarize(backend, duration, data, records, stop, counters, exhausted, scaled, solve_times,
                     limit_hits, sim.pose), records


def summarize(backend, duration, data, records, stop, counters, exhausted, scaled, solve_times, limit_hits, pose):
    arr = lambda key: np.array([r[key] for r in records], dtype=float)  # noqa: E731
    tail = [r for r in records if r["t"] >= records[-1]["t"] - 2.0]
    legs = np.array([r["leg"] for r in records])
    torques = np.abs(np.array([r["torque"] for r in records]))
    predicted = [min(r["predicted_normal"]) for r in records if r["predicted_normal"]]
    dyn_res = [r["dynamics_residual"] for r in records if r["dynamics_residual"] is not None]
    times = np.array(solve_times) * 1e3 if solve_times else np.zeros(1)
    residual_tail = {name: float(np.max([r["residual"].get(name, 0.0) for r in tail]))
                     for name in exhausted}
    return dict(
        backend=backend, completed=stop is None, stop_reason=stop,
        simulated=float(data.time), requested=duration, L0=pose.length, phi0=pose.angle,
        **counters,
        max_abs_pitch=float(np.max(np.abs(arr("pitch")))), max_abs_roll=float(np.max(np.abs(arr("roll")))),
        max_abs_yaw=float(np.max(np.abs(arr("yaw")))),
        max_abs_x=float(np.max(np.abs(arr("x") - records[0]["x"]))), final_x=float(records[-1]["x"] - records[0]["x"]),
        max_abs_v=float(np.max(np.abs(arr("v")))),
        max_leg_length_error=float(np.max(np.abs(legs[:, [0, 2]] - pose.length))),
        max_leg_angle_error=float(np.max(np.abs(legs[:, [1, 3]] - pose.angle))),
        tail_pitch_span=float(np.ptp([r["pitch"] for r in tail])), tail_x_span=float(np.ptp([r["x"] for r in tail])),
        tail_max_abs_v=float(np.max(np.abs([r["v"] for r in tail]))),
        max_torque=torques.max(axis=0).tolist(),
        min_predicted_normal=float(min(predicted)) if predicted else None,
        max_dynamics_residual=float(max(dyn_res)) if dyn_res else None,
        exhausted_counts=exhausted, scaled_counts=scaled, tail_task_residual_max=residual_tail,
        joint_limits_hit=sorted(limit_hits),
        solve_ms=None,
        first_conditioning=records[0]["conditioning"],
        last=dict(t=records[-1]["t"], pitch=records[-1]["pitch"], roll=records[-1]["roll"], x=records[-1]["x"],
                  leg=records[-1]["leg"], torque=records[-1]["torque"], error=records[-1]["error"]),
    )


def to_markdown(summaries) -> str:
    def f(v, n=3, scale=1.0):
        return "—" if v is None else f"{v * scale:.{n}f}"

    def sci(v):
        return "—" if v is None else f"{v:.1e}"
    lines = [
        "# M2：WBC 单独运行（不接 MPC）零速站立", "",
        f"脚本：`tools/m2_standalone_wbc.py`（mujoco {mujoco.__version__}）。L0 = {NOMINAL_L0} m，"
        f"WBC {1 / WBC_DT:.0f} Hz，物理 1 kHz。",
        "增益 / 权重为 circleg 默认值直接搬用（attitude 80/10、height 100/12、regularize 60/6、"
        "速度阻尼 14；权重见 `gbc_mpc_wbc/wbc.py`），**未调参**。", "",
        "| 后端 | 完成 | 停机原因 | 仿真时长 s | max\\|θ\\| mrad | max\\|roll\\| mrad | max\\|x\\| mm | max\\|v\\| mm/s |"
        " 腿长误差 max mm | 腿角误差 max mrad | 末 2 s θ 峰峰 mrad | 末 2 s x 峰峰 mm |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['backend']} | {'是' if s['completed'] else '否'} | {s['stop_reason'] or '—'} | {s['simulated']:.3f} | "
            f"{f(s['max_abs_pitch'], 2, 1e3)} | {f(s['max_abs_roll'], 2, 1e3)} | {f(s['max_abs_x'], 2, 1e3)} | "
            f"{f(s['max_abs_v'], 2, 1e3)} | {f(s['max_leg_length_error'], 2, 1e3)} | {f(s['max_leg_angle_error'], 2, 1e3)} | "
            f"{f(s['tail_pitch_span'], 3, 1e3)} | {f(s['tail_x_span'], 3, 1e3)} |")
    lines += ["", "| 后端 | WBC 次数 | fallback | 力矩越限 clamp | infeasible | 求解异常 | 最小预测法向力 N | "
              "max 动力学残差 | max\\|τ\\| [jAB, jAG, jIJ, jIO, 右轮, 左轮] N·m | 关节限位 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(
            f"| {s['backend']} | {s['wbc_calls']} | {s['fallback']} | {s['clip']} | {s['infeasible']} | "
            f"{s['solver_error']} | {f(s['min_predicted_normal'], 1)} | {sci(s['max_dynamics_residual'])} | "
            f"{[round(v, 2) for v in s['max_torque']]} | {s['joint_limits_hit'] or '无'} |")
    names = list(summaries[0]["exhausted_counts"]) if summaries else []
    lines += ["", "## 任务统计", "",
              "exhausted = 零空间法中该级已无剩余权限的拍数；scaled = 该级增量被力矩限幅缩小的拍数（零空间法，不含 exhausted 级）。"
              "末 2 s 残差 = |J q̈ + bias − 目标| 的最大值 [m/s² 或 rad/s²]。", "",
              "| 后端 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for s in summaries:
        lines.append(f"| {s['backend']} exhausted | " + " | ".join(str(s["exhausted_counts"][n]) for n in names) + " |")
        lines.append(f"| {s['backend']} scaled | " + " | ".join(str(s["scaled_counts"][n]) for n in names) + " |")
        lines.append(f"| {s['backend']} 末 2 s 残差 | " + " | ".join(f"{s['tail_task_residual_max'][n]:.2e}"
                                                                    for n in names) + " |")
    ns = next((s for s in summaries if s["backend"] == "nullspace"), None)
    if ns and ns["first_conditioning"]:
        lines += ["", "零空间法首拍各级条件数：" + "，".join(f"{k} {v:.1f}" for k, v in ns["first_conditioning"].items())]
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    RESULTS.mkdir(exist_ok=True)
    summaries = []
    for backend in args.backends:
        summary, records = run(backend, args.duration)
        summaries.append(summary)
        with gzip.open(RESULTS / f"m2_{backend}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(records, handle)
        print(f"{backend}: completed={summary['completed']} stop={summary['stop_reason']} t={summary['simulated']:.3f}",
              flush=True)
    (RESULTS / "m2_standalone_wbc.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    text = to_markdown(summaries)
    (RESULTS / "m2_standalone_wbc.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
