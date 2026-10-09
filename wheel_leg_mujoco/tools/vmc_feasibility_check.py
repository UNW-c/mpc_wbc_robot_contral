"""VMC 压缩相位可行性自检（问题排查agent，2026-10-09）。

纯离线诊断。读一次已有运行结果里的压缩位形，判定在给定轮权限下：

    (1) 质心竖直加速度 = 目标 与 俯仰角加速度 = 0 能否同时成立（com+pitch）
    (2) 再加上两条腿的摆角加速度 = 0 是否仍可行（com+pitch+angle）
    (3) 在质心目标硬约束下，俯仰角加速度的可达范围（判断 WBC 俯仰 PD 需求是否超权限）

为什么要有它：WP8 已证明把 com_vertical 当软任务会与 pitch/angle 抢权限。在决定投入
VMC（等效力前馈）之前，先用它回答"轮权限恢复后姿态到底能不能压住"，避免又做一轮
注定失败的实机实验。

每个采样帧、每种 selection 解一个线性规划（scipy.optimize.linprog）：

    决策变量 z = [qacc(nv), f(nf), tau(nu)]

    等式：
        Jc qacc = -bias_c                       接触 + 闭链加速度约束
        M qacc + h = S^T tau + Jc^T f           完整动力学（含欠驱动方向）
        <各硬目标>                              com / pitch / phi_R / phi_L 的加速度
    不等式：
        limits.lower <= tau <= limits.upper
        逐轮：f_n >= 0, |f_t| <= mu f_n, |f_lat| <= 2 mu f_n（横向为共模，逐轮各半）

    Q1 = 轮 selection 保留（本方案）；Q0 = 轮 selection 清零（WP8 现状），用于对照。

自检（工作区第 4 条）：用记录的 qacc/ctrl 反解接触力，与记录的 normal_forces/
tangent_forces 逐轮比对；并检查 Jc qacc + bias_c 是否为零。两者通过，工具口径才算成立。

本工具不修改任何控制代码、不写执行器、不推进物理。
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from gbc_mpc_wbc import robot as R  # noqa: E402
from gbc_mpc_wbc.dynamics import Gbc486Dynamics  # noqa: E402
from gbc_mpc_wbc.logging_setup import setup_logging  # noqa: E402
from gbc_mpc_wbc.wbc import Gbc486Tasks, Nominal  # noqa: E402
from Algorithm.wbc.projection import ContactElimination  # noqa: E402

_LOGGER = logging.getLogger(__name__)
RESULTS = ROOT / "results"
COM_TOLERANCE = 1.0e-3        # [m/s^2] com 目标判为命中的容差
ACCEL_TOLERANCE = 1.0e-3      # [rad/s^2] 姿态/腿角目标判为命中的容差
MU_BISECTION_STEPS = 40
_MU_MAX = 2.0


def _load_frames(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _nearest_frame(frames, time_s: float):
    return min(frames, key=lambda f: abs(float(f["t"]) - time_s))


def _state_matrices(model, data, frame):
    """在给定记录帧的位形上重建行、动力学与接触块。"""
    data.qpos[:] = np.asarray(frame["qpos"], dtype=float)
    data.qvel[:] = np.asarray(frame["qvel"], dtype=float)
    mujoco.mj_forward(model, data)
    dynamics = Gbc486Dynamics(model, data)
    rows = dynamics.rowset()
    contact = tuple(bool(c) for c in frame["decision"]["contact"])
    tasks = Gbc486Tasks(dynamics, Nominal(0.28, 0.0, 0.0))
    blocks = tasks.contact_blocks(rows, contact)
    return dict(dynamics=dynamics, rows=rows, contact=contact, blocks=blocks,
                jacobian=np.vstack([b.jacobian for b in blocks]),
                contact_bias=np.concatenate([b.bias for b in blocks]),
                mass=dynamics.mass(), bias=dynamics.dynamics_bias(),
                selection=dynamics.selection, limits=dynamics.limits,
                row_com=dynamics.com_vertical_row()[0],
                bias_com=dynamics.com_vertical_row()[1])


def _linear_targets(state):
    """把每个待考察物理量写成 (row_over_qacc, bias)，含义为 量 = row·qacc + bias。"""
    nf = state["jacobian"].shape[0]
    nu = state["selection"].shape[0]
    leg_rows, leg_bias = state["rows"].leg
    task_jacobian, task_bias = state["rows"].task
    pad = np.zeros(nf + nu)

    def row(vector):
        return np.concatenate([np.asarray(vector, dtype=float), pad])

    return {
        "com": (row(state["row_com"]), float(state["bias_com"])),
        "pitch": (row(task_jacobian[2]), float(task_bias[2])),
        "len_R": (row(leg_rows[0]), float(leg_bias[0])),
        "len_L": (row(leg_rows[2]), float(leg_bias[2])),
        "phi_R": (row(leg_rows[1]), float(leg_bias[1])),
        "phi_L": (row(leg_rows[3]), float(leg_bias[3])),
    }


def _self_check(state, frame):
    """用记录的 qacc/ctrl 反解接触力，与记录的法向/切向力比对；并检查接触约束残差。"""
    count = int(sum(bool(c) for c in state["contact"]))
    elimination = ContactElimination.from_matrices(state["jacobian"], state["selection"],
                                                   state["mass"])
    qacc = np.asarray(frame["qacc"], dtype=float)
    torque = np.asarray(frame["ctrl"], dtype=float)
    recovered = elimination.recover_contact_force(qacc, state["bias"], torque)
    tangent = recovered[0:2 * count:2]
    normal = recovered[1:2 * count:2]
    normal_ref = np.asarray(frame["normal_forces"], dtype=float)[:count]
    tangent_ref = np.asarray(frame["tangent_forces"], dtype=float)[:count]
    return dict(
        max_normal_error=float(np.max(np.abs(normal - normal_ref))),
        max_tangent_error=float(np.max(np.abs(tangent - tangent_ref))),
        contact_residual_norm=float(np.linalg.norm(state["jacobian"] @ qacc + state["contact_bias"])),
        recovered_normal=normal.tolist(), recorded_normal=normal_ref.tolist(),
    )


def _friction_rows(state, mu):
    nv = state["mass"].shape[0]
    nf = state["jacobian"].shape[0]
    nu = state["selection"].shape[0]
    width = nv + nf + nu
    rows, upper, normal_index = [], [], []
    index = 0
    for active in state["contact"]:
        if not active:
            continue
        tangential, normal = index, index + 1
        for sign in (1.0, -1.0):
            row = np.zeros(width)
            row[nv + tangential] = sign
            row[nv + normal] = -mu
            rows.append(row)
            upper.append(0.0)
        row = np.zeros(width)
        row[nv + normal] = -1.0
        rows.append(row)
        upper.append(0.0)
        normal_index.append(normal)
        index += 2
    lateral = index  # 共模横向行的力下标
    for normal in normal_index:
        for sign in (1.0, -1.0):
            row = np.zeros(width)
            row[nv + lateral] = sign
            row[nv + normal] = -2.0 * mu
            rows.append(row)
            upper.append(0.0)
    return np.asarray(rows), np.asarray(upper)


def _solve(state, wheels_enabled, mu, hard, soft=None, soft_value=0.0):
    """通用 LP。

    hard      : dict 名称 -> 目标值，作为硬等式（量 = row·q̈ + bias）
    soft      : 名称或 None，最小化 |量 - soft_value|
    """
    from scipy.optimize import linprog

    targets = _linear_targets(state)
    nv = state["mass"].shape[0]
    nf = state["jacobian"].shape[0]
    nu = state["selection"].shape[0]
    width = nv + nf + nu
    limits = state["limits"]
    friction, friction_upper = _friction_rows(state, mu)

    eq_blocks = [np.hstack([state["jacobian"], np.zeros((nf, nf)), np.zeros((nf, nu))]),
                 np.hstack([state["mass"], -state["jacobian"].T, -state["selection"].T])]
    eq_targets = [np.atleast_1d(-state["contact_bias"]), np.atleast_1d(-state["bias"])]
    for name, value in hard.items():
        row, bias = targets[name]
        eq_blocks.append(row[None, :])
        eq_targets.append(np.atleast_1d(value - bias))

    slack_count = 0
    soft_index = None
    if soft is not None:
        row, bias = targets[soft]
        plus = np.zeros((1, width + 2))
        plus[0, :width] = row
        plus[0, width] = -1.0
        minus = np.zeros((1, width + 2))
        minus[0, :width] = row
        minus[0, width + 1] = 1.0
        eq_blocks.append(plus)
        eq_blocks.append(minus)
        eq_targets.append(np.atleast_1d(soft_value - bias))
        eq_targets.append(np.atleast_1d(soft_value - bias))
        slack_count = 2
        soft_index = np.arange(width, width + 2)

    total = width + slack_count
    matrix_eq = np.vstack([np.hstack([block, np.zeros((block.shape[0], total - block.shape[1]))])
                           for block in eq_blocks])
    target_eq = np.concatenate(eq_targets)
    matrix_ub = np.hstack([friction, np.zeros((friction.shape[0], slack_count))])
    bounds = [(None, None)] * nv + [(None, None)] * nf
    for index in range(nu):
        if not wheels_enabled and index in state["dynamics"].wheel_actuators:
            bounds.append((0.0, 0.0))
        else:
            bounds.append((float(limits.lower[index]), float(limits.upper[index])))
    bounds += [(0.0, None)] * slack_count

    cost = np.zeros(total)
    if soft_index is not None:
        cost[soft_index] = 1.0
    result = linprog(cost, A_ub=matrix_ub, b_ub=friction_upper, A_eq=matrix_eq, b_eq=target_eq,
                     bounds=bounds, method="highs")
    if not result.success:
        return None
    z = result.x
    qacc = z[:nv]
    torque = z[nv + nf:nv + nf + nu]
    achieved = {name: float(row @ z[:width] + bias) for name, (row, bias) in targets.items()}
    return dict(achieved=achieved, max_abs_tau=float(np.max(np.abs(torque))),
                qacc=qacc, torque=torque)


def _feasible(state, wheels_enabled, mu, hard):
    result = _solve(state, wheels_enabled, mu, hard)
    return result is not None


def _min_mu(state, wheels_enabled, hard):
    if not _feasible(state, wheels_enabled, _MU_MAX, hard):
        return None
    low, high = 0.0, _MU_MAX
    for _ in range(MU_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        if _feasible(state, wheels_enabled, mid, hard):
            high = mid
        else:
            low = mid
    return float(high)


def _analyse_frame(model, data, frame, mu_model):
    state = _state_matrices(model, data, frame)
    report = dict(t=float(frame["t"]), phase=frame["jump"]["phase"],
                  com_target=None, self_check=_self_check(state, frame),
                  wheels_on={}, wheels_off={})
    com_target = frame["jump"]["desired_com_acceleration_z"]
    if com_target is None:
        return report
    com_target = float(com_target)
    report["com_target"] = com_target
    for label, enabled in (("wheels_on", True), ("wheels_off", False)):
        com_pitch = {"com": com_target, "pitch": 0.0}
        com_pitch_angle = dict(com_pitch, phi_R=0.0, phi_L=0.0)
        com_best = _solve(state, enabled, mu_model, {"pitch": 0.0}, soft="com",
                          soft_value=com_target)
        pitch_best = _solve(state, enabled, mu_model, {"com": com_target}, soft="pitch")
        exact = _solve(state, enabled, mu_model, com_pitch_angle)
        leg_space = {"len_R": com_target, "len_L": com_target,
                     "pitch": 0.0, "phi_R": 0.0, "phi_L": 0.0}
        leg_exact = _solve(state, enabled, mu_model, leg_space)
        report[label] = dict(
            com_pitch_feasible=_feasible(state, enabled, mu_model, com_pitch),
            com_pitch_angle_feasible=exact is not None,
            com_best_achieved=None if com_best is None else com_best["achieved"]["com"],
            com_best_error=None if com_best is None else abs(com_best["achieved"]["com"] - com_target),
            pitch_best_achieved=None if pitch_best is None else pitch_best["achieved"]["pitch"],
            mu_req=_min_mu(state, enabled, com_pitch_angle),
            com_pitch_angle_max_abs_tau=None if exact is None else exact["max_abs_tau"],
            leg_space_feasible=leg_exact is not None,
            leg_space_mu_req=_min_mu(state, enabled, leg_space),
            leg_space_max_abs_tau=None if leg_exact is None else leg_exact["max_abs_tau"],
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="VMC 压缩相位可行性自检")
    parser.add_argument("--input", type=Path, default=RESULTS / "WP8_baseline_run01.json.gz")
    parser.add_argument("--frames", type=str,
                        default="1.000,1.030,1.060,1.090,1.096,1.121,1.150,1.184,1.220")
    parser.add_argument("--mu", type=float, default=None, help="覆盖摩擦系数；默认取模型值")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    setup_logging()
    payload = _load_frames(arguments.input)
    model, data = R.load()
    if arguments.mu is None:
        wheel = [model.geom(leg.wheel_geom).id for leg in R.LEGS]
        mu_model = float(max(np.max(model.geom_friction[wheel][:, 0]),
                             model.geom_friction[model.geom("floor").id, 0]))
    else:
        mu_model = float(arguments.mu)
    _LOGGER.info("输入 %s，帧数 %d，模型摩擦系数 mu=%.4f",
                 arguments.input, len(payload["frames"]), mu_model)

    reports = []
    for token in arguments.frames.split(","):
        token = token.strip()
        if not token:
            continue
        report = _analyse_frame(model, data, _nearest_frame(payload["frames"], float(token)), mu_model)
        reports.append(report)
        on, off = report.get("wheels_on", {}), report.get("wheels_off", {})
        _LOGGER.info(
            "t=%.3f %s target=%s | 自检 dFn=%.2e dFt=%.2e | "
            "ON com+pitch=%s com+pitch+angle=%s leg_space=%s leg_mu=%s leg_tau=%s mu_req=%s | "
            "OFF com+pitch=%s com+pitch+angle=%s leg_space=%s",
            report["t"], report["phase"], report["com_target"],
            report["self_check"]["max_normal_error"], report["self_check"]["max_tangent_error"],
            on.get("com_pitch_feasible"), on.get("com_pitch_angle_feasible"),
            on.get("leg_space_feasible"), _fmt(on.get("leg_space_mu_req")),
            _fmt(on.get("leg_space_max_abs_tau")), _fmt(on.get("mu_req")),
            off.get("com_pitch_feasible"), off.get("com_pitch_angle_feasible"),
            off.get("leg_space_feasible"),
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    output = arguments.output or RESULTS / f"VMC_feasibility_{stamp}.json"
    output.write_text(json.dumps(dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        input=str(arguments.input), mu_model=mu_model,
        com_tolerance=COM_TOLERANCE, accel_tolerance=ACCEL_TOLERANCE,
        reports=reports), indent=2, ensure_ascii=False), encoding="utf-8")
    _LOGGER.info("结果写入 %s", output)
    return 0


def _fmt(value):
    return "None" if value is None else f"{value:.3e}"


if __name__ == "__main__":
    raise SystemExit(main())
