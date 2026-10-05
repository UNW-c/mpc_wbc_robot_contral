"""M0 模型核验 + 标称腿长 L0 扫描（只测量，不涉及任何控制参数）。

用法（在 wheel_leg_mujoco 目录下）：
    PYTHONDONTWRITEBYTECODE=1 ../.venv/Scripts/python.exe tools/m0_l0_sweep.py

输出：results/m0_l0_sweep.json 与 results/m0_l0_sweep.md
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gbc_mpc_wbc import robot as R  # noqa: E402
from VMC import leg_VMC  # noqa: E402

RESULTS = ROOT / "results"
G = 9.81
LENGTHS = np.round(np.arange(0.12, 0.4401, 0.02), 3)
TILT = math.radians(20.0)
HOLD_TIME = 0.3          # 静态力矩开环保持时长 [s]
TORQUE_LIMIT = 20.0      # 腿电机 ctrlrange [N*m]
# Simulation.py 中的关节零位补偿（VMC 映射用，只做对照）
VMC_OFFSETS = dict(jAB=0.027, jAG=1.3, jIJ=0.003, jIO=-1.3)


def jac_point(model, data, body_id, point):
    jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jac, None, point, body_id)
    return jac


def loop_jacobian(model, data):
    rows = []
    for leg in R.LEGS:
        for s1, s2 in leg.loops:
            j1, j2 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, j1, None, model.site(s1).id)
            mujoco.mj_jacSite(model, data, j2, None, model.site(s2).id)
            rows.append(j1 - j2)
    return np.vstack(rows)  # 12 x nv（世界 x/y/z）


def static_torques(model, data):
    """准静态：S'tau + Jc'f + Jl'lam = qfrc_bias - qfrc_passive，轮力矩为 0。"""
    mujoco.mj_forward(model, data)
    points = R.wheel_contact_points(model, data)
    jc = np.vstack([jac_point(model, data, model.body(leg.wheel_body).id, p)
                    for leg, p in zip(R.LEGS, points)])                      # 6 x nv
    jl = loop_jacobian(model, data)                                          # 12 x nv
    sel = np.zeros((model.nu, model.nv))
    for a in range(model.nu):
        sel[a, model.jnt_dofadr[model.actuator_trnid[a, 0]]] = model.actuator_gear[a, 0]
    leg_act = [model.actuator(n).id for n in ("Right_front_joint_act", "Right_rear_joint_act",
                                              "Left_front_joint_act", "Left_rear_joint_act")]
    A = np.hstack([sel[leg_act].T, jc.T, jl.T])                              # nv x (4+6+12)
    b = data.qfrc_bias - data.qfrc_passive
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = float(np.linalg.norm(A @ x - b))
    # 唯一性：A 的零空间在 tau 分量上的投影
    _, s, vt = np.linalg.svd(A)
    rank = int(np.sum(s > 1e-9 * s[0]))
    null = vt[rank:]
    tau_ambiguity = float(np.max(np.abs(null[:, :4]))) if null.size else 0.0
    tau = x[:4]
    forces = x[4:10].reshape(2, 3)
    return tau, forces, residual, tau_ambiguity, leg_act


def com_and_inertia(model, data):
    mujoco.mj_forward(model, data)
    ids = range(1, model.nbody)
    mass = np.array([model.body_mass[i] for i in ids])
    pos = np.array([data.xipos[i] for i in ids])
    com = (mass[:, None] * pos).sum(0) / mass.sum()
    inertia = np.zeros((3, 3))
    for i, m_i, p in zip(ids, mass, pos):
        rot = data.ximat[i].reshape(3, 3)
        inertia += rot @ np.diag(model.body_inertia[i]) @ rot.T
        r = p - com
        inertia += m_i * (r @ r * np.eye(3) - np.outer(r, r))
    return com, float(mass.sum()), inertia


def clearance(model, data):
    """非轮 geom 到地面的最小距离：(参与碰撞的, 全部 geom) [m]。"""
    floor = model.geom("floor").id
    skip = {floor} | {model.geom(leg.wheel_geom).id for leg in R.LEGS}
    skip |= {g for g in range(model.ngeom)
             if model.body(model.geom_bodyid[g]).name in ("wheel_right", "wheel_left")}
    colliding, overall = np.inf, np.inf
    fromto = np.zeros(6)
    for g in range(model.ngeom):
        if g in skip or model.geom_bodyid[g] == 0:
            continue
        dist = mujoco.mj_geomDistance(model, data, g, floor, 1.0, fromto)
        overall = min(overall, dist)
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            colliding = min(colliding, dist)
    return float(colliding), float(overall)


def tilted_clearance(model, data, qpos, theta):
    """整机绕两轮心连线刚体转动 theta（轮心不动、轮仍着地）后的最小离地间隙。"""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    center = np.mean([data.xpos[model.body(leg.wheel_body).id] for leg in R.LEGS], axis=0)
    rot = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0],
                    [-math.sin(theta), 0, math.cos(theta)]])
    data.qpos[0:3] = center + rot @ (qpos[0:3] - center)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.array([math.cos(theta / 2), 0, math.sin(theta / 2), 0]), qpos[3:7])
    data.qpos[3:7] = quat
    mujoco.mj_forward(model, data)
    result = clearance(model, data)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return result


def vmc_singularity(model, data):
    q = {n: float(data.qpos[model.joint(n).qposadr[0]]) + VMC_OFFSETS[n] for n in VMC_OFFSETS}
    out = []
    for phi1, phi4 in ((q["jAB"] + math.pi, q["jAG"]), (q["jIO"] + math.pi, q["jIJ"])):
        v = leg_VMC()
        v.vmc_calc_pos(phi1=phi1, phi4=phi4, pitch=0.0, gyro=0.0)
        out.append((float(v.L0), float(abs(math.sin(v.phi3 - v.phi2)))))
    return out


def hold_test(model, data, qpos, tau, leg_act):
    """静态力矩开环保持：轮力矩 0，腿力矩取 static_torques，检查漂移与闭链残差。"""
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.ctrl[leg_act] = tau
    mujoco.mj_forward(model, data)
    start = [R.leg_polar(model, data, leg) for leg in R.LEGS]
    z0 = float(data.qpos[2])
    loop_max = 0.0
    for _ in range(int(round(HOLD_TIME / model.opt.timestep))):
        mujoco.mj_step(model, data)
        for leg in R.LEGS:
            inplane, outplane = R.loop_residual(model, data, leg)
            loop_max = max(loop_max, float(np.max(np.abs(np.r_[inplane, outplane]))))
    mujoco.mj_forward(model, data)
    end = [R.leg_polar(model, data, leg) for leg in R.LEGS]
    drift_l = max(abs(e[0] - s[0]) for s, e in zip(start, end))
    drift_phi = max(abs(e[1] - s[1]) for s, e in zip(start, end))
    pitch = 2 * math.asin(max(-1.0, min(1.0, float(data.qpos[5]))))
    others = sum(1 for c in data.contact[:data.ncon]
                 if model.body(model.geom_bodyid[c.geom1]).name not in ("world", "wheel_right", "wheel_left")
                 or model.body(model.geom_bodyid[c.geom2]).name not in ("world", "wheel_right", "wheel_left"))
    return dict(drift_length=drift_l, drift_angle=drift_phi, drift_z=float(data.qpos[2]) - z0,
                pitch_end=pitch, loop_residual_max=loop_max, other_contacts_end=others)


def model_checks(model, data, qpos):
    """M0 验收：接触点、闭链秩、自由度。"""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    names = lambda g: model.body(model.geom_bodyid[g]).name  # noqa: E731
    contacts = [(names(c.geom1), names(c.geom2), c.pos.copy()) for c in data.contact[:data.ncon]]
    predicted = R.wheel_contact_points(model, data)
    errors = []
    for leg, point in zip(R.LEGS, predicted):
        pts = [c[2] for c in contacts if leg.wheel_body in c[:2]]
        errors.append(float(np.linalg.norm(np.mean(pts, axis=0)[[0, 2]] - point[[0, 2]])) if pts else float("nan"))
    jl = loop_jacobian(model, data)
    inplane = jl[[i for i in range(12) if i % 3 != 1]]
    s = np.linalg.svd(inplane, compute_uv=False)
    rank = int(np.sum(s > 1e-6 * s[0]))
    return dict(ncon=len(contacts), pairs=sorted({f"{a}-{b}" for a, b, _ in contacts}),
                per_wheel=[sum(1 for c in contacts if leg.wheel_body in c[:2]) for leg in R.LEGS],
                contact_point_error=errors, loop_inplane_rank=rank,
                loop_inplane_cond=float(s[0] / s[-1]), independent_dof=model.nv - rank)


def sweep():
    model, data = R.load()
    rows, checks = [], None
    for length in LENGTHS:
        row = dict(L0=float(length))
        try:
            pose = R.standing_pose(model, data, float(length))
        except RuntimeError as error:
            row.update(reachable=False, reason=str(error))
            rows.append(row)
            continue
        qpos = pose.qpos
        margins = R.joint_range_margins(model, data, R.ACTUATED_LEG_JOINTS + ("jGH", "jOP"))
        row.update(phi0=pose.angle, reachable=bool(min(margins.values()) >= 0.0),
                   min_joint_margin=float(min(margins.values())),
                   min_margin_joint=min(margins, key=margins.get))
        com, mass, inertia = com_and_inertia(model, data)
        h = float(com[2] - np.mean(R.wheel_contact_points(model, data)[:, 2]))
        row.update(com_height=h, open_loop_pole=math.sqrt(G / h), mass=mass,
                   pitch_inertia_com=float(inertia[1, 1]), yaw_inertia_com=float(inertia[2, 2]))
        tau, forces, residual, ambiguity, leg_act = static_torques(model, data)
        row.update(static_torque=tau.tolist(), static_torque_max=float(np.max(np.abs(tau))),
                   torque_margin_ratio=float(np.max(np.abs(tau)) / TORQUE_LIMIT),
                   static_residual=residual, torque_ambiguity=ambiguity,
                   normal_force=forces[:, 2].tolist(), tangential_force=forces[:, 0].tolist())
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        conds = []
        for leg in R.LEGS:
            jac, passive_cond = R.leg_jacobian(model, data, leg)
            conds.append((float(np.linalg.cond(jac)), passive_cond))
        row.update(leg_jacobian_cond=max(c[0] for c in conds), loop_block_cond=max(c[1] for c in conds))
        vmc = vmc_singularity(model, data)
        row.update(vmc_L0=[v[0] for v in vmc], vmc_sin_phi3_phi2=min(v[1] for v in vmc))
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        c0 = clearance(model, data)
        cp = tilted_clearance(model, data, qpos, +TILT)
        cn = tilted_clearance(model, data, qpos, -TILT)
        row.update(clearance_colliding=c0[0], clearance_all=c0[1],
                   clearance_tilt_colliding=min(cp[0], cn[0]), clearance_tilt_all=min(cp[1], cn[1]))
        row.update(hold=hold_test(model, data, qpos, tau, leg_act))
        if checks is None or abs(length - 0.30) < 1e-9:
            checks = dict(L0=float(length), **model_checks(model, data, qpos))
        rows.append(row)
    return model, rows, checks


def to_markdown(model, rows, checks) -> str:
    def f(v, n=3):
        return "—" if v is None else f"{v:.{n}f}"
    lines = [
        "# M0 模型核验与 L0 扫描结果",
        "",
        "脚本：`tools/m0_l0_sweep.py`（mujoco %s）。只测量，不涉及控制参数。" % mujoco.__version__,
        "",
        "## 1. 模型核验（L0 = %.2f m 站立位形）" % checks["L0"],
        "",
        f"- 接触数 {checks['ncon']}，接触对 {checks['pairs']}，每轮接触点数 {checks['per_wheel']}",
        f"- 圆柱接触点（各轮接触点均值）与解析最低点的 xz 偏差：{[f'{e*1000:.3f} mm' for e in checks['contact_point_error']]}",
        f"- 闭链平面内 8 行：秩 {checks['loop_inplane_rank']}，cond {checks['loop_inplane_cond']:.1f}；"
        f"独立自由度 nv − 秩 = {checks['independent_dof']}",
        "",
        "## 2. L0 扫描",
        "",
        "腿角 φ0 取静平衡值（整机质心在两轮接触点正上方，机身水平，轮力矩 0）。",
        "静态力矩为准静态最小二乘解，顺序 [jAB, jAG, jIJ, jIO]。",
        "",
        "| L0 m | 可达 | φ0 rad | 最小关节余量 rad（关节） | 质心高 h m | √(g/h) rad/s | 静态力矩 max N·m（/20） |"
        " 腿雅可比 cond | 闭链块 cond | VMC min|sin(φ3−φ2)| | 离地间隙 碰撞/全部 mm | ±20° 间隙 碰撞/全部 mm |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "phi0" not in r:
            lines.append(f"| {r['L0']:.2f} | 否（{r['reason']}） | | | | | | | | | | |")
            continue
        lines.append(
            f"| {r['L0']:.2f} | {'是' if r['reachable'] else '否'} | {r['phi0']:+.4f} | "
            f"{r['min_joint_margin']:+.3f}（{r['min_margin_joint']}） | {r['com_height']:.3f} | "
            f"{r['open_loop_pole']:.2f} | {r['static_torque_max']:.2f}（{r['torque_margin_ratio']:.0%}） | "
            f"{r['leg_jacobian_cond']:.1f} | {r['loop_block_cond']:.1f} | {r['vmc_sin_phi3_phi2']:.3f} | "
            f"{r['clearance_colliding']*1000:.1f} / {r['clearance_all']*1000:.1f} | "
            f"{r['clearance_tilt_colliding']*1000:.1f} / {r['clearance_tilt_all']*1000:.1f} |")
    lines += [
        "",
        "## 3. 静态力矩开环保持（%.1f s，轮力矩 0，腿力矩 = 上表静态解）" % HOLD_TIME,
        "",
        "用于交叉验证静态力矩是否正确；未加任何反馈，平衡点本身开环不稳定，漂移随时间增长属预期。",
        "",
        "| L0 m | 静态解残差 | 力矩唯一性(零空间投影) | 法向力 R/L N | ΔL mm | Δφ mrad | Δz mm | 末俯仰 mrad | 闭链残差 max mm | 末时刻非轮接触 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "hold" not in r:
            continue
        h = r["hold"]
        lines.append(
            f"| {r['L0']:.2f} | {r['static_residual']:.1e} | {r['torque_ambiguity']:.1e} | "
            f"{r['normal_force'][0]:.1f} / {r['normal_force'][1]:.1f} | {h['drift_length']*1000:.2f} | "
            f"{h['drift_angle']*1000:.2f} | {h['drift_z']*1000:.2f} | {h['pitch_end']*1000:.2f} | "
            f"{h['loop_residual_max']*1000:.2f} | {h['other_contacts_end']} |")
    first = next(r for r in rows if "mass" in r)
    lines += ["", f"整机质量 {first['mass']:.3f} kg；m·g/2 = {first['mass']*G/2:.1f} N。", ""]
    return "\n".join(lines)


def main():
    # Windows 控制台默认 GBK，表格里有 − / √ 等字符
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    model, rows, checks = sweep()
    RESULTS.mkdir(exist_ok=True)
    payload = dict(mujoco=mujoco.__version__, checks=checks, rows=rows)
    (RESULTS / "m0_l0_sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=lambda o: o.tolist()), encoding="utf-8")
    (RESULTS / "m0_l0_sweep.md").write_text(to_markdown(model, rows, checks), encoding="utf-8")
    print(to_markdown(model, rows, checks))


if __name__ == "__main__":
    main()
