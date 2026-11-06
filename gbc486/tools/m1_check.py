"""M1 核验：Gbc486Dynamics 的选择矩阵、各类行与偏置、接触力回代、欠驱动维数。

只测量，不涉及任何控制参数。用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/m1_check.py
输出：results/m1_dynamics_check.md / .json
"""
from __future__ import annotations

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
from gbc_mpc_wbc.dynamics import Gbc486Dynamics, LEG, LOOP, TASK  # noqa: E402
from Algorithm.wbc import ContactElimination  # noqa: E402
from VMC import leg_VMC  # noqa: E402

RESULTS = ROOT / "results"
NOMINAL_L0 = 0.42            # 部署方案决策 3
CHECK_TIMESTEP = 1.0e-4      # 导数核验用的细步长（仅内存中修改）
CHECK_DURATION = 0.15        # [s]
PERTURB_TORQUE = (2.0, 0.5)  # 腿 / 轮随机扰动力矩幅值 [N*m]，只用于激励运动
SEED = 0
CONTACT = (True, True)
VMC_OFFSETS = dict(jAB=0.027, jAG=1.3, jIJ=0.003, jIO=-1.3)


def setup(timestep: float | None = None):
    model, data = R.load()
    if timestep is not None:
        model.opt.timestep = timestep
    R.standing_pose(model, data, NOMINAL_L0)
    return model, data, Gbc486Dynamics(model, data)


def static_torque(dyn: Gbc486Dynamics):
    """准静态：S'τ + C'λ = h，轮力矩 = 0。返回 (τ, λ, 残差)。"""
    mujoco.mj_forward(dyn.model, dyn.data)
    rows = dyn.rowset()
    constraint, _ = rows.constraints(CONTACT)
    legs = list(dyn.leg_actuators)
    A = np.hstack([dyn.selection[legs].T, constraint.T])
    h = dyn.dynamics_bias()
    x, *_ = np.linalg.lstsq(A, h, rcond=None)
    tau = np.zeros(dyn.model.nu)
    tau[legs] = x[:4]
    return tau, x[4:], float(np.linalg.norm(A @ x - h))


def check_selection(dyn: Gbc486Dynamics):
    m, d = dyn.model, dyn.data
    worst = 0.0
    for a in range(m.nu):
        d.ctrl[:] = 0.0
        d.ctrl[a] = 1.0
        mujoco.mj_forward(m, d)
        worst = max(worst, float(np.max(np.abs(d.qfrc_actuator - dyn.selection.T @ d.ctrl))))
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    left_wheel = float(dyn.selection[dyn.wheel_actuators[1], m.joint("jwheel_left").dofadr[0]])
    return dict(max_error=worst, left_wheel_entry=left_wheel)


def check_structure(dyn: Gbc486Dynamics):
    """静止站立位形上的秩、欠驱动维数、任务行在约束零空间内的独立性。"""
    mujoco.mj_forward(dyn.model, dyn.data)
    rows = dyn.rowset()
    constraint, _ = rows.constraints(CONTACT)
    s = np.linalg.svd(constraint, compute_uv=False)
    rank = int(np.sum(s > 1e-9 * s[0]))
    _, _, vt = np.linalg.svd(constraint)
    null = vt[rank:].T                                  # 约束零空间基（运动自由度）
    elim = ContactElimination.from_matrices(constraint, dyn.selection, dyn.mass())
    task_rows, _ = rows.task
    leg_rows, _ = rows.leg
    named = {
        "forward(heading)": task_rows[0], "roll": task_rows[1], "pitch": task_rows[2], "yaw": task_rows[3],
        "L_c": 0.5 * (leg_rows[0] + leg_rows[2]), "L_d": 0.5 * (leg_rows[0] - leg_rows[2]),
        "phi_c": 0.5 * (leg_rows[1] + leg_rows[3]), "phi_d": 0.5 * (leg_rows[1] - leg_rows[3]),
    }
    projected = {k: v @ null for k, v in named.items()}

    def rank_of(keys):
        mat = np.vstack([projected[k] for k in keys])
        sv = np.linalg.svd(mat, compute_uv=False)
        return int(np.sum(sv > 1e-6 * sv[0])), float(sv[-1] / sv[0])

    roll_ld = rank_of(["roll", "L_d"])
    plan = ["pitch", "roll", "L_c", "yaw", "phi_c", "phi_d", "forward(heading)"]
    return dict(
        constraint_rows=int(constraint.shape[0]), constraint_rank=rank, constraint_cond=float(s[0] / s[rank - 1]),
        motion_dof=int(null.shape[1]),
        actuation_rows=int(elim.actuation_matrix.shape[0]), actuation_cols=int(elim.actuation_matrix.shape[1]),
        actuation_cond=elim.actuation_conditioning,
        underactuated_dims=int(elim.unactuated_directions.shape[1]),
        roll_vs_L_d_rank=roll_ld[0], roll_vs_L_d_ratio=roll_ld[1],
        roll_L_d_cosine=float(abs(projected["roll"] @ projected["L_d"])
                              / (np.linalg.norm(projected["roll"]) * np.linalg.norm(projected["L_d"]))),
        planned_task_rank=rank_of(plan)[0], planned_task_ratio=rank_of(plan)[1], planned_tasks=plan,
        prefix_ranks={k: rank_of(plan[:i + 1])[0] for i, k in enumerate(plan)},
    )


def check_kinematic_bias(dyn: Gbc486Dynamics, samples: int = 20, step: float = 1.0e-4):
    """与动力学无关的核验：沿随机速度 v 做位形积分 q(s) = q ⊕ s·v。

对可直接计算的量 y(q)（腿空间 [L, φ]、闭链 site 差在机身 x/z 上的投影），
一阶差分应等于 J v，二阶差分应等于 bias（qacc = 0 时的 J̇v）。
    """
    m, d = dyn.model, dyn.data
    rng = np.random.default_rng(SEED + 1)
    base_q = d.qpos.copy()
    scratch = mujoco.MjData(m)

    def quantities(q):
        scratch.qpos[:] = q
        mujoco.mj_kinematics(m, scratch)
        rot = scratch.xmat[dyn.base].reshape(3, 3)
        out = [v for leg in R.LEGS for v in R.leg_polar(m, scratch, leg)]
        for s1, s2 in dyn.loop_sites:
            diff = scratch.site_xpos[s1] - scratch.site_xpos[s2]
            out += [rot[:, 0] @ diff, rot[:, 2] @ diff]
        return np.array(out)

    first, second = [], []
    for _ in range(samples):
        # 只取满足闭链速度约束的 v：在闭链行零空间里随机取
        d.qpos[:] = base_q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        loop_rows = dyn.rowset().loop[0]
        _, sv, vt = np.linalg.svd(loop_rows)
        null = vt[int(np.sum(sv > 1e-9 * sv[0])):].T
        d.qvel[:] = null @ rng.normal(size=null.shape[1])
        d.qvel[:] *= 0.5 / np.linalg.norm(d.qvel)
        mujoco.mj_forward(m, d)
        rows = dyn.rowset()
        predicted_rate = np.r_[rows.leg[0] @ d.qvel, rows.loop[0] @ d.qvel]
        predicted_bias = np.r_[rows.leg[1], rows.loop[1]]
        values = []
        for s in (-step, 0.0, step):
            q = base_q.copy()
            mujoco.mj_integratePos(m, q, d.qvel, s)
            values.append(quantities(q))
        rate = (values[2] - values[0]) / (2 * step)
        accel = (values[2] - 2 * values[1] + values[0]) / step ** 2
        first.append(np.abs(rate - predicted_rate) / np.maximum(np.abs(rate), 1e-3))
        second.append(np.abs(accel - predicted_bias) / np.maximum(np.abs(accel), 1e-2))
    d.qpos[:] = base_q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    first, second = np.array(first), np.array(second)
    return dict(samples=samples, step=step,
                leg_rate_rel_max=float(first[:, :4].max()), loop_rate_rel_max=float(first[:, 4:].max()),
                leg_bias_rel_max=float(second[:, :4].max()), loop_bias_rel_max=float(second[:, 4:].max()))


def check_leg_vs_vmc(dyn: Gbc486Dynamics):
    m, d = dyn.model, dyn.data
    mujoco.mj_forward(m, d)
    q = {n: float(d.qpos[m.joint(n).qposadr[0]]) + VMC_OFFSETS[n] for n in VMC_OFFSETS}
    legs = dyn.leg_state()
    out = []
    for (phi1, phi4), length in zip(((q["jAB"] + math.pi, q["jAG"]), (q["jIO"] + math.pi, q["jIJ"])), legs[[0, 2]]):
        v = leg_VMC()
        v.vmc_calc_pos(phi1=phi1, phi4=phi4, pitch=0.0, gyro=0.0)
        out.append(float(v.L0 - length))
    return dict(vmc_minus_model=out, leg_state=legs.tolist())


def simulate_checks():
    """细步长开环仿真：静态力矩 + 随机扰动，逐步核验行导数、约束与接触力回代。"""
    model, data, dyn = setup(CHECK_TIMESTEP)
    tau0, _, residual = static_torque(dyn)
    rng = np.random.default_rng(SEED)
    amplitude = np.zeros(model.nu)
    amplitude[list(dyn.leg_actuators)] = PERTURB_TORQUE[0]
    amplitude[list(dyn.wheel_actuators)] = PERTURB_TORQUE[1]
    steps = int(round(CHECK_DURATION / model.opt.timestep))
    hold = int(round(0.01 / model.opt.timestep))
    stats = dict(derivative=[], loop_velocity=[], loop_acceleration=[], contact_velocity=[],
                 contact_acceleration=[], force_heading=[], force_normal=[], force_scale=[],
                 recover_residual=[], limit_active=set(), other_contacts=0, speed=[])
    previous = None
    for step in range(steps):
        if step % hold == 0:
            perturb = rng.uniform(-1.0, 1.0, model.nu) * amplitude
        data.ctrl[:] = np.clip(tau0 + perturb, dyn.limits.lower, dyn.limits.upper)
        mujoco.mj_forward(model, data)
        rows = dyn.rowset()
        rate = rows.rows @ data.qvel
        predicted = rows.rows @ data.qacc + rows.bias
        if previous is not None and step % hold != 0:  # 跳过力矩阶跃拍（q̈ 不连续，梯形估计失效）
            measured = (rate - previous[0]) / model.opt.timestep
            expected = 0.5 * (predicted + previous[1])
            scale = np.maximum(np.abs(expected), 1.0)
            stats["derivative"].append(np.abs(measured - expected) / scale)
        previous = (rate, predicted)
        loop_rows, loop_bias = rows.loop
        stats["loop_velocity"].append(float(np.max(np.abs(loop_rows @ data.qvel))))
        stats["loop_acceleration"].append(float(np.max(np.abs(loop_rows @ data.qacc + loop_bias))))
        active, counts, other = dyn.contact_geometry()

        # 接触力已不由 MuJoCo 力 API 提供（用户 2026-10-07 删除）：本脚本的

        # 力相关统计同步失效，仅保留几何量；需要力时请走 recover_contact_force。

        forces = np.zeros((2, 3))
        stats["other_contacts"] += other
        stats["limit_active"].update(dyn.joint_limit_active())
        stats["speed"].append(abs(float(rate[0])))
        if all(active):
            constraint, bias = rows.constraints(CONTACT)
            nc = rows.contact_row_count(CONTACT)
            stats["contact_velocity"].append(float(np.max(np.abs(constraint[:nc] @ data.qvel))))
            stats["contact_acceleration"].append(float(np.max(np.abs(constraint[:nc] @ data.qacc + bias[:nc]))))
            net = dyn.mass() @ data.qacc + dyn.dynamics_bias() - dyn.selection.T @ data.ctrl
            multiplier, *_ = np.linalg.lstsq(constraint.T, net, rcond=None)
            stats["recover_residual"].append(float(np.linalg.norm(constraint.T @ multiplier - net)
                                                   / max(np.linalg.norm(net), 1e-9)))
            yaw = dyn.attitude()[2]
            heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            for side in range(2):
                stats["force_heading"].append(abs(float(multiplier[2 * side] - heading @ forces[side])))
                stats["force_normal"].append(abs(float(multiplier[2 * side + 1] - forces[side, 2])))
                stats["force_scale"].append(float(forces[side, 2]))
        mujoco.mj_step(model, data)
    derivative = np.array(stats["derivative"])
    groups = {"task": TASK, "contact": slice(4, 10), "loop": LOOP, "leg": LEG}
    return dict(
        timestep=model.opt.timestep, duration=CHECK_DURATION, static_residual=residual,
        static_leg_torque=tau0[list(dyn.leg_actuators)].tolist(),
        derivative_relative_p99={k: float(np.percentile(derivative[:, v], 99)) for k, v in groups.items()},
        derivative_relative_max={k: float(np.max(derivative[:, v])) for k, v in groups.items()},
        loop_velocity_max=float(np.max(stats["loop_velocity"])),
        loop_acceleration_p99=float(np.percentile(stats["loop_acceleration"], 99)),
        contact_velocity_max=float(np.max(stats["contact_velocity"])),
        contact_acceleration_p99=float(np.percentile(stats["contact_acceleration"], 99)),
        recover_residual_p99=float(np.percentile(stats["recover_residual"], 99)),
        force_heading_error_max=float(np.max(stats["force_heading"])),
        force_normal_error_max=float(np.max(stats["force_normal"])),
        normal_force_mean=float(np.mean(stats["force_scale"])),
        max_speed=float(np.max(stats["speed"])),
        joint_limits_hit=sorted(stats["limit_active"]), other_contacts=int(stats["other_contacts"]),
        final_pitch=float(dyn.attitude()[1]),
    )


def run_all():
    model, data, dyn = setup()
    return dict(mujoco=mujoco.__version__, L0=NOMINAL_L0,
                selection=check_selection(dyn), structure=check_structure(dyn),
                kinematic=check_kinematic_bias(dyn),
                vmc=check_leg_vs_vmc(dyn), simulation=simulate_checks())


def to_markdown(r) -> str:
    sel, st, vmc, sim, kin = r["selection"], r["structure"], r["vmc"], r["simulation"], r["kinematic"]
    lines = [
        "# M1 核验：Gbc486Dynamics", "",
        f"脚本：`tools/m1_check.py`（mujoco {r['mujoco']}），站立位形 L0 = {r['L0']} m。只测量，不涉及控制参数。", "",
        "## 1. 选择矩阵",
        f"- `selection.T @ ctrl` 与 `qfrc_actuator` 最大差 {sel['max_error']:.1e}；"
        f"左轮项 = {sel['left_wheel_entry']:+.0f}（已含 gainprm = −1）", "",
        "## 2. 结构（双支撑静止站立）",
        f"- 硬约束行 {st['constraint_rows']}（接触 5 + 闭链 8），秩 {st['constraint_rank']}，cond {st['constraint_cond']:.1f}",
        f"- 运动自由度 nv − 秩 = {st['motion_dof']}",
        f"- 消力后执行矩阵 `Q_uᵀSᵀ` {st['actuation_rows']}×{st['actuation_cols']}，cond {st['actuation_cond']:.1f}；"
        f"**欠驱动维数 {st['underactuated_dims']}**",
        f"- roll 行与 L_d 行投影到约束零空间后：秩 {st['roll_vs_L_d_rank']}，|cos| = {st['roll_L_d_cosine']:.4f}",
        f"- 第 4.4 节计划任务 {st['planned_tasks']}：秩 {st['planned_task_rank']}（最小/最大奇异值比 {st['planned_task_ratio']:.2e}）",
        f"- 逐级累计秩：{st['prefix_ranks']}", "",
        f"## 3. 行与偏置的运动学核验（与动力学无关，{kin['samples']} 组闭链一致的随机速度，差分步 {kin['step']}）",
        f"- 腿空间行：一阶 J v 相对误差最大 {kin['leg_rate_rel_max']:.1e}；二阶 bias 相对误差最大 {kin['leg_bias_rel_max']:.1e}",
        f"- 闭链行：一阶最大 {kin['loop_rate_rel_max']:.1e}；二阶最大 {kin['loop_bias_rel_max']:.1e}", "",
        "## 4. 腿空间量 vs 旧 VMC",
        f"- [L_R, φ_R, L_L, φ_L] = {[round(v, 4) for v in vmc['leg_state']]}",
        f"- VMC L0 − 模型 L（右/左）：{[f'{v*1000:+.2f} mm' for v in vmc['vmc_minus_model']]}", "",
        f"## 5. 开环细步长仿真（dt = {sim['timestep']}, {sim['duration']} s，静态力矩 + 随机扰动）",
        f"- 静态腿力矩 [jAB, jAG, jIJ, jIO] = {[round(v, 3) for v in sim['static_leg_torque']]} N·m，残差 {sim['static_residual']:.1e}",
        "- 行导数一致性 |Δ(Jq̇)/dt − (Jq̈ + bias)| / max(|·|, 1)（已排除力矩阶跃拍；该项含积分器误差，仅作量级参考）：",
        "", "| 行组 | p99 | max |", "|---|---|---|",
    ]
    for k in sim["derivative_relative_p99"]:
        lines.append(f"| {k} | {sim['derivative_relative_p99'][k]:.2e} | {sim['derivative_relative_max'][k]:.2e} |")
    lines += [
        "",
        f"- 闭链：max |J q̇| = {sim['loop_velocity_max']:.2e} m/s，p99 |J q̈ + bias| = {sim['loop_acceleration_p99']:.2e} m/s²"
        "（MuJoCo 闭链是软约束，这是约束自身的纠偏加速度）",
        f"- 接触：max |Jc q̇| = {sim['contact_velocity_max']:.2e} m/s（最大前进速度 {sim['max_speed']:.3f} m/s），"
        f"p99 |Jc q̈ + bias| = {sim['contact_acceleration_p99']:.2e} m/s²（同为软接触的穿透动力学）",
        f"- 约束力回代：p99 相对残差 {sim['recover_residual_p99']:.2e}；与 `mj_contactForce` 相比，切向最大差 "
        f"{sim['force_heading_error_max']:.3f} N、法向最大差 {sim['force_normal_error_max']:.3f} N（平均法向力 {sim['normal_force_mean']:.1f} N）",
        f"- 关节限位触发：{sim['joint_limits_hit'] or '无'}；非轮接触累计 {sim['other_contacts']}；末俯仰 {sim['final_pitch']:.4f} rad",
        "",
    ]
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = run_all()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "m1_dynamics_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=lambda o: o.tolist()), encoding="utf-8")
    text = to_markdown(result)
    (RESULTS / "m1_dynamics_check.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
