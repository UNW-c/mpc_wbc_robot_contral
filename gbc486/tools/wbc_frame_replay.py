"""WBC 单帧离线复算（诊断工具）。

用途
----
定位“某个任务在求解器里达不到”的原因。用 ``results/*.json.gz`` 里已经记录的
``qpos`` / ``qvel`` / ``sample`` / ``decision`` / ``distance`` 重建**一拍** WBC 输入，
直接调 ``SupervisedControl.solve_wbc``，并对任务集做消融。

这一级不做什么
--------------
- 不推进物理、不写执行器、不改任何增益 / 权重 / 阈值 / 门控 / 门限。
- 不产生新结果文件；只在标准输出打印。

自检（先跑，再读结论）
----------------------
``--self-check`` 用完整任务集复算，与记录里的 ``wbc_raw_torque``、``wbc_result.qacc``
逐元素比对。不通过就说明本工具重建的输入与实跑不一致，结论作废。

用法
----
    .venv/Scripts/python.exe wheel_leg_mujoco/tools/wbc_frame_replay.py \
        --result results/R1_LAND3_baseline_run01.json.gz --times 2.996 2.000
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gbc_mpc_wbc.logging_setup import setup_logging            # noqa: E402
from gbc_mpc_wbc.sampling import chassis_state                  # noqa: E402
from gbc_mpc_wbc.simulation import Gbc486MpcWbcSimulation, SimulationConfig  # noqa: E402
from Algorithm.mpc.trajectory import MotionSample               # noqa: E402
from Algorithm.supervision import ControlDecision, ControlMode  # noqa: E402

RESULTS = ROOT / "results"


def _load(path: Path) -> dict:
    with gzip.open(path) as handle:
        return json.load(handle)


def _pick_frame(frames: list, t_want: float) -> int:
    """取离 t_want 最近的“真正调过 WBC”的帧。任务轨迹只在求解帧才有内容。"""
    candidates = [k for k, y in enumerate(frames) if y.get("wbc_called")]
    if not candidates:
        raise ValueError("该结果里没有任何 wbc_called 帧")
    return min(candidates, key=lambda k: abs(float(frames[k]["t"]) - t_want))


def _decision_from(frame: dict) -> ControlDecision:
    d = frame["decision"]
    return ControlDecision(mode=ControlMode(d["mode"]),
                           contact=tuple(bool(v) for v in d["contact"]),
                           timestamp=float(frame["t"]),
                           context=int(d.get("context", 0)),
                           reset_reference=bool(d.get("reset_reference", False)))


def _restore(sim, frame: dict) -> None:
    """把仿真状态设回记录的那一拍。只设状态，不推进物理。"""
    import mujoco
    sim.data.qpos[:] = np.asarray(frame["qpos"], dtype=float)
    sim.data.qvel[:] = np.asarray(frame["qvel"], dtype=float)
    sim.data.time = float(frame["t"])
    sim.data.ctrl[:] = np.asarray(frame["ctrl"], dtype=float)
    mujoco.mj_forward(sim.model, sim.data)


def _forces_from(frame: dict) -> np.ndarray:
    forces = np.zeros((2, 3))
    for side in range(2):
        forces[side, 0] = float(frame["tangent_forces"][side])
        forces[side, 2] = float(frame["normal_forces"][side])
    return forces


def _sample_from(frame: dict) -> MotionSample | None:
    raw = frame.get("sample")
    if not isinstance(raw, dict) or raw.get("state") is None:
        return None
    return MotionSample(state=np.asarray(raw["state"], dtype=float),
                        acceleration=np.asarray(raw["acceleration"], dtype=float),
                        force=np.zeros(2 if "force" not in raw else len(raw["force"])),
                        timestamp=float(frame["t"]))


def _call_wbc(sim, *, balance, motion, decision, blocks, sample, motion_override):
    return sim.control.solve_wbc(
        mass=sim.dyn.mass(), selection=sim.dyn.selection,
        dynamics_bias=sim.dyn.dynamics_bias(),
        contacts=blocks, balance_tasks=tuple(balance), motion_tasks=tuple(motion),
        limits=sim.dyn.limits, decision=decision, dt=sim.timing.wbc_dt,
        external_force=None, weight_overrides=None, motion_override=motion_override)


def _zero_activation(tasks, names):
    return tuple(dataclasses.replace(t, activation=0.0) if t.name in names else t for t in tasks)


def _outcomes(result) -> dict:
    if result is None:
        return {}
    return {item.name: item for item in result.hierarchy.outcomes}


def _print_solution(label: str, torque, result, state) -> None:
    names = ("pitch", "angle_R", "angle_L", "forward_velocity", "length_R", "length_L", "yaw", "roll")
    outcomes = _outcomes(result)
    qacc = None if result is None else np.asarray(result.qacc, dtype=float)
    print(f"  [{label}]")
    print("    轮力矩 ctrl[4,5] = %+.5f %+.5f   (限幅 ±6)" % (torque[4], torque[5]))
    print("    髋力矩 ctrl[0:4] = %s" % " ".join("%+8.4f" % v for v in torque[:4]))
    if qacc is not None:
        print("    qacc[0]=%+.5f  qacc[4]=%+.5f  (forward / pitch)" % (qacc[0], qacc[4]))
    if result is not None:
        print("    feasible=%s fallback=%s clip=%s scaling=%s"
              % (getattr(result, "feasible", None), getattr(result, "fallback_used", None),
                 getattr(result, "clip_applied", None),
                 {k: round(float(v), 4) for k, v in (result.scaling or {}).items()}))
    for name in names:
        item = outcomes.get(name)
        if item is None:
            continue
        print("      %-17s desired %+9.5f  achieved %+9.5f  residual %+9.5f  exhausted=%s"
              % (name, float(np.ravel(item.desired)[0]), float(np.ravel(item.achieved)[0]),
                 float(item.residual_norm), getattr(item, "exhausted", None)))


def _row_space_report(balance, motion) -> None:
    """检查 forward_velocity 行与它上面所有任务行的关系。"""
    ordered = list(balance) + list(motion)
    names = [t.name for t in ordered]
    if "forward_velocity" not in names:
        return
    idx = names.index("forward_velocity")
    rows = np.vstack([np.atleast_2d(np.asarray(t.jacobian, dtype=float)) for t in ordered])
    fv = rows[idx]
    print("    任务行矩阵 = %s，秩 = %d（任务数 %d）"
          % (rows.shape, int(np.linalg.matrix_rank(rows)), len(ordered)))
    print("    forward_velocity 行范数 = %.6e" % float(np.linalg.norm(fv)))
    # |行_上面这些张成的空间| 对 fv 行的投影长度；若几乎等于 fv 行范数，
    # 说明 fv 这个方向已经完全被更高优先级的任务行张成，低层没有独立方向可用。
    if idx == 0:
        return
    above = rows[:idx]
    coeff, *_ = np.linalg.lstsq(above.T, fv, rcond=None)
    proj = coeff @ above
    print("    fv 行在“它上面 %d 行”张成空间内的投影范数 = %.6e  (占比 %.4f)"
          % (idx, float(np.linalg.norm(proj)),
             float(np.linalg.norm(proj) / max(np.linalg.norm(fv), 1e-30))))
    print("    正交残差范数 = %.6e" % float(np.linalg.norm(fv - proj)))


def replay_one(sim, frame: dict, *, show_rows: bool, motion_override: bool = True) -> None:
    decision = _decision_from(frame)
    _restore(sim, frame)
    state, rows = chassis_state(dynamics=sim.dyn, data=sim.data, contact=decision.contact,
                               forces=_forces_from(frame), distance=float(frame["distance"]),
                               equilibrium_pitch=float(sim.reduced.metadata["equilibrium_pitch"]),
                               nominal_angle=float(sim.pose.angle))
    sample = _sample_from(frame)
    blocks = sim.tasks.contact_blocks(rows, decision.contact)

    print("=== t=%.4f  mode=%s  com_vx=%+.5f  raw_torque 记录值=%s ==="
          % (float(frame["t"]), decision.mode.value, float(frame["com_vxz"][0]),
             np.array2string(np.asarray(frame["wbc_raw_torque"], dtype=float), precision=5)))

    balance, motion = sim.tasks.tasks(state, rows, sample,
                                     desired_com_acceleration_z=None,
                                     pitch_reference=None, roll_reference=None,
                                     angle_R_reference=None, angle_L_reference=None,
                                     world_angle_anchor=motion_override)
    if show_rows:
        _row_space_report(balance, motion)

    cases = [
        ("full", balance, motion),
        ("full - pitch", _zero_activation(balance, {"pitch"}), motion),
        ("full - angle", balance, _zero_activation(motion, {"angle_R", "angle_L"})),
        ("full - length", balance, _zero_activation(motion, {"length_R", "length_L"})),
        ("full - fv", balance, _zero_activation(motion, {"forward_velocity"})),
        ("pitch + fv", balance, tuple(t for t in motion if t.name == "forward_velocity")),
        ("只留 fv（其余 activation=0）", _zero_activation(balance, {"pitch"}),
         _zero_activation(motion, {"angle_R", "angle_L", "length_R", "length_L", "yaw", "roll"})),
        ("只留 pitch（其余 activation=0）", balance,
         _zero_activation(motion, {"angle_R", "angle_L", "forward_velocity", "length_R", "length_L",
                                   "yaw", "roll"})),
        ("pitch + angle + fv", balance,
         tuple(t for t in motion if t.name in ("angle_R", "angle_L", "forward_velocity"))),
        ("pitch + length + fv", balance,
         tuple(t for t in motion if t.name in ("forward_velocity", "length_R", "length_L"))),
    ]
    first = True
    for label, bal, mot in cases:
        try:
            torque, result = _call_wbc(sim, balance=bal, motion=mot, decision=decision,
                                       blocks=blocks, sample=sample,
                                       motion_override=motion_override)
        except Exception as exc:                                   # noqa: BLE001
            print("  [%s] 求解异常：%s: %s" % (label, type(exc).__name__, exc))
            continue
        torque = np.asarray(torque, dtype=float)
        if first:
            # 自检：完整任务集的复算结果必须与记录一致，否则本工具重建的输入与实跑不符，结论作废
            recorded_torque = np.asarray(frame["wbc_raw_torque"], dtype=float)
            recorded_qacc = np.asarray((frame.get("wbc_result") or {}).get("qacc"), dtype=float)
            d_torque = float(np.max(np.abs(torque - recorded_torque)))
            d_qacc = (float(np.max(np.abs(np.asarray(result.qacc, dtype=float) - recorded_qacc)))
                      if recorded_qacc.size else float("nan"))
            ok = d_torque < 1e-9 and (not recorded_qacc.size or d_qacc < 1e-9)
            print("    [自检] 与记录力矩最大逐元素差 = %.3e ；qacc 最大逐元素差 = %.3e => %s"
                  % (d_torque, d_qacc, "通过" if ok else "不通过（后续结论作废）"))
            first = False
        _print_solution(label, torque, result, state)
    print()


def authority_report(sim, frame: dict, *, motion_override: bool = True) -> None:
    """逐任务给出“在硬约束零空间里剩下的行范数”。

    求解器（``wbc_limited.LimitedWeightedResolver``）在硬约束 ``A q̈ = b``（接触等式 +
    欠驱动动力学等式）的零空间里做加权最小二乘：``projected = J_task @ null``。
    如果某任务的 ``‖J @ null‖ / ‖J‖`` 接近 0，该任务就**几乎完全由硬约束解定死**，
    权重再大、需求再大都改不动它——这正是“需求扫不动解”的机制。
    """
    from gbc_mpc_wbc.wbc_limited import _hard_system                 # noqa: PLC0415
    from Algorithm.wbc.projection import ContactElimination          # noqa: PLC0415

    decision = _decision_from(frame)
    _restore(sim, frame)
    state, rows = chassis_state(dynamics=sim.dyn, data=sim.data, contact=decision.contact,
                               forces=_forces_from(frame), distance=float(frame["distance"]),
                               equilibrium_pitch=float(sim.reduced.metadata["equilibrium_pitch"]),
                               nominal_angle=float(sim.pose.angle))
    sample = _sample_from(frame)
    blocks = sim.tasks.contact_blocks(rows, decision.contact)
    balance, motion = sim.tasks.tasks(state, rows, sample, desired_com_acceleration_z=None,
                                     pitch_reference=None, roll_reference=None,
                                     angle_R_reference=None, angle_L_reference=None,
                                     world_angle_anchor=motion_override)
    tasks = list(balance) + list(motion)

    jacobian = np.vstack([blk.jacobian for blk in blocks])
    bias = np.concatenate([blk.bias for blk in blocks])
    mass = sim.dyn.mass()
    elimination = ContactElimination.from_matrices(jacobian, sim.dyn.selection, mass)
    hard_rows, _ = _hard_system(elimination, jacobian, bias, sim.dyn.dynamics_bias())
    pseudo = np.linalg.pinv(hard_rows, rcond=1e-12)
    null = np.eye(mass.shape[0]) - pseudo @ hard_rows

    print("=== 硬约束零空间权威度  t=%.4f ===" % float(frame["t"]))
    print("  硬约束行数 = %d（含接触等式 %d 行）  nv=%d  零空间维数 = %d  硬约束秩 = %d"
          % (hard_rows.shape[0], jacobian.shape[0], mass.shape[0],
             int(mass.shape[0] - np.linalg.matrix_rank(hard_rows)),
             int(np.linalg.matrix_rank(hard_rows))))
    print("  %-17s %12s %12s %8s" % ("任务", "‖J‖", "‖J@null‖", "剩余占比"))
    for task in tasks:
        row = np.atleast_2d(np.asarray(task.jacobian, dtype=float))
        full = float(np.linalg.norm(row))
        left = float(np.linalg.norm(row @ null))
        print("  %-17s %12.6e %12.6e %7.4f" % (task.name, full, left, left / max(full, 1e-30)))
    # 交叉验证：求解器自己也把 N N^T M 吐回来了，必须与我算的一致，否则上表无意义。
    try:
        _, result = _call_wbc(sim, balance=balance, motion=motion, decision=decision,
                              blocks=blocks, sample=sample, motion_override=motion_override)
        ref_ns = np.asarray(result.hierarchy.null_space, dtype=float)
        mine = null @ null.T @ mass
        delta = float(np.max(np.abs(mine - ref_ns)))
        print("  [交叉验证] 我算的 N·Nᵀ·M 与 result.hierarchy.null_space 最大逐元素差 = %.3e => %s"
              % (delta, "一致" if delta < 1e-9 else "不一致（上表作废）"))
    except Exception as exc:                                       # noqa: BLE001
        print("  [交叉验证] 失败：%s: %s" % (type(exc).__name__, exc))
    # 执行器可达性：选择矩阵的秩就是执行器能直接产生广义力的方向数。
    sel = np.asarray(sim.dyn.selection, dtype=float)
    print("  selection 形状 %s 秩 = %d（执行器数 nu=%d）  nv=%d"
          % (sel.shape, int(np.linalg.matrix_rank(sel)), sel.shape[1], mass.shape[0]))
    print()


def sweep_forward(sim, frame: dict, *, stops, motion_override: bool = True) -> None:
    """把 forward_velocity 的需求从 0 扫到“t_stop 秒内把当前速度收到 0”，
    其余任务不变，记录轮力矩 / 髀用量 / 各项残差。

    这是**单帧快照扫描**，不是闭环结果：它给出“这一拍如果要拿到这么多前向加速度，
    要付出多少髀权限”，即可行域边界。停下来的过程里速度会变小，需求随之变小，
    所以本表是初始时刻的最严一档。
    """
    decision = _decision_from(frame)
    _restore(sim, frame)
    state, rows = chassis_state(dynamics=sim.dyn, data=sim.data, contact=decision.contact,
                               forces=_forces_from(frame), distance=float(frame["distance"]),
                               equilibrium_pitch=float(sim.reduced.metadata["equilibrium_pitch"]),
                               nominal_angle=float(sim.pose.angle))
    sample = _sample_from(frame)
    blocks = sim.tasks.contact_blocks(rows, decision.contact)
    balance, motion = sim.tasks.tasks(state, rows, sample, desired_com_acceleration_z=None,
                                     pitch_reference=None, roll_reference=None,
                                     angle_R_reference=None, angle_L_reference=None,
                                     world_angle_anchor=motion_override)
    kp_att, kd_att = sim.tasks.gains.attitude_gains
    v0 = abs(float(frame["com_vxz"][0]))
    print("=== 扫描 t=%.4f  v0=%.5f m/s  attitude_gains kp=%.4f kd=%.4f ==="
          % (float(frame["t"]), v0, float(kp_att), float(kd_att)))
    print("  t_stop  a_req   轮 ctrl[4,5]        max|髀|  髀饱和  scaling  fv残差  pitch残差  等效 pitch 角误差")
    for t_stop in stops:
        a_req = v0 / t_stop
        mot = tuple(dataclasses.replace(t, desired_acceleration=np.array([a_req]))
                    if t.name == "forward_velocity" else t for t in motion)
        try:
            torque, result = _call_wbc(sim, balance=balance, motion=mot, decision=decision,
                                       blocks=blocks, sample=sample,
                                       motion_override=motion_override)
        except Exception as exc:                                   # noqa: BLE001
            print("  %.2f   %+.4f  求解异常 %s: %s" % (t_stop, a_req, type(exc).__name__, exc))
            continue
        torque = np.asarray(torque, dtype=float)
        out = _outcomes(result)
        hip = float(np.max(np.abs(torque[:4])))
        hip_lim = float(np.max(np.abs(np.asarray(sim.dyn.limits.upper)[:4])))
        res_fv = out["forward_velocity"].residual_norm if "forward_velocity" in out else float("nan")
        res_p = out["pitch"].residual_norm if "pitch" in out else float("nan")
        # 姿态任务的 desired = kp*(ref-pitch) + kd*(ref_rate-pitch_rate)，所以残差 / kp
        # 就是“差多少弧度才能补上这个加速度缺口”（稳态等效角误差）。
        ang = res_p / kp_att if kp_att else float("nan")
        print("  %.2f   %+.4f  %+8.4f %+8.4f   %7.3f  %-6s  %.4f  %7.4f  %8.4f  %+8.4f rad (%+.2f deg)"
              % (t_stop, a_req, torque[4], torque[5], hip, "是" if hip >= hip_lim - 1e-6 else "否",
                 float(result.scaling.get("forward_velocity", float("nan"))), res_fv, res_p,
                 ang, np.degrees(ang)))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="WBC 单帧离线复算（只读，不改控制路径）")
    parser.add_argument("--result", default="results/R1_LAND3_baseline_run01.json.gz")
    parser.add_argument("--times", nargs="+", type=float, default=[2.996, 2.000])
    parser.add_argument("--rows", action="store_true", help="打印任务行空间分析")
    parser.add_argument("--sweep-fv", action="store_true",
                        help="扫描：把 forward_velocity 需求按 t_stop 换算后逐点求解")
    parser.add_argument("--authority", action="store_true",
                        help="逐任务给出在硬约束零空间里剩余的行范数占比")
    parser.add_argument("--stops", nargs="+", type=float,
                        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--no-motion-override", action="store_true",
                        help="关闭落地窗口 motion_override（对照）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging("INFO" if args.verbose else "WARNING")

    path = RESULTS / args.result if not Path(args.result).is_absolute() else Path(args.result)
    payload = _load(path)
    frames = payload["frames"]
    meta = payload.get("metadata", {}).get("config", {})
    config = SimulationConfig(**{k: v for k, v in meta.items()
                                 if k in {f.name for f in dataclasses.fields(SimulationConfig)}})
    sim = Gbc486MpcWbcSimulation(config)
    print("结果文件：%s" % path.name)
    print("配置重建：backend=%s model_variant=%s preset=%s nominal_length=%.4f"
          % (config.backend, config.model_variant, config.preset, config.nominal_length))
    print("wbc_dt=%.6f  equilibrium_pitch=%.6f  nominal_angle=%.6f"
          % (sim.timing.wbc_dt, float(sim.reduced.metadata["equilibrium_pitch"]), float(sim.pose.angle)))
    print()
    for t_want in args.times:
        k = _pick_frame(frames, t_want)
        if args.authority:
            authority_report(sim, frames[k], motion_override=not args.no_motion_override)
        if args.sweep_fv:
            sweep_forward(sim, frames[k], stops=args.stops,
                          motion_override=not args.no_motion_override)
        if not args.authority and not args.sweep_fv:
            replay_one(sim, frames[k], show_rows=args.rows,
                       motion_override=not args.no_motion_override)


if __name__ == "__main__":
    main()
