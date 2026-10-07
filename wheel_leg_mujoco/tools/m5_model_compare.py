"""M5：6 状态 held-leg 与 8 状态 leg-input 模型对比（同工况，指标逐项列出，不做取舍结论）。

内容：
1. 两个模型的 A / B / 极点 / 配平点，以及 8 状态模型的线性化自洽性核验
   （用非线性降阶 ODE 积分一步，与 A·δz、B·δu 比较）；
2. 同一批工况在两个模型下的闭环指标（与 M3 / M4 的工况保持一致）。

只验证可行性，不调参。用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/m5_model_compare.py [--variants held_leg leg_input] [--cases stand30 ...]
输出：results/m5_model_compare.md / .json，时序 results/m5_<variant>_<case>.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gbc_mpc_wbc import robot as R  # noqa: E402
from gbc_mpc_wbc.dynamics import Gbc486Dynamics  # noqa: E402
from gbc_mpc_wbc.model_builder import HeldLegModelBuilder, LegInputModelBuilder  # noqa: E402
from gbc_mpc_wbc.simulation import Disturbance, Gbc486MpcWbcSimulation, SimulationConfig  # noqa: E402
from Algorithm.mpc import MotionCommand  # noqa: E402
from m3_mpc_wbc import PUSH_TIME, case_metrics, circleg_revision  # noqa: E402
from m4_command_tracking import (CRUISE_WINDOW, DURATION, HOLD_END, HOLD_START,  # noqa: E402
                                 STOP_SPEED, metrics as command_metrics)
from gbc_mpc_wbc.logging_setup import get_logger, setup_logging  # noqa: E402

RESULTS = ROOT / "results"
VARIANTS = ("held_leg", "leg_input")
# (工况名, 速度指令, 偏航指令, 推扰速度增量, 时长)
CASES = {
    "stand30": (0.0, 0.0, None, 30.0),
    "push_fwd": (0.0, 0.0, (0.20, 0.0, 0.0), 12.0),
    "push_lat": (0.0, 0.0, (0.0, 0.20, 0.0), 12.0),
    "v1.0": (1.0, 0.0, None, DURATION),
    "v2.0": (2.0, 0.0, None, DURATION),
    "w1.0": (0.0, 1.0, None, DURATION),
    "v1.0w0.5": (1.0, 0.5, None, DURATION),
}


def make_command(v, w):
    def command(t):
        active = HOLD_START <= t < HOLD_END
        return MotionCommand(velocity=v if active else 0.0, yaw_rate=w if active else 0.0)
    return command


def linearization_checks(builder, is8: bool, dt: float = 0.02):
    """线性化自洽性：把非线性降阶 ODE 积分一步，与 ``A·δz``、``B·δu`` 比较。

    做法：以微小扰动做中心差分得到线性预测，再与同样大小的非线性步进比较。
    把扰动缩小 10 倍后误差应降约 100 倍（二阶余项），否则说明线性化有错。
    扰动取得很小（腿角偏差 1e-3 rad、输入 0.2），以免把真实非线性算成误差。
    """
    if is8:
        builder.set_legs(0.0)
        theta, tau = builder.trim(0.0)
        point = np.zeros(8)
        point[2], point[6] = theta, 0.0
        base = np.array([0.0, 0.0, tau])
    else:
        point = np.zeros(6)
        point[2] = 0.0
        base = np.zeros(2)
    dimension = 8 if is8 else 6
    ninput = 3 if is8 else 2

    def step(z, u):                                        # 四阶 Runge-Kutta
        k1 = builder.derivative(z, u)
        k2 = builder.derivative(z + 0.5 * dt * k1, u)
        k3 = builder.derivative(z + 0.5 * dt * k2, u)
        k4 = builder.derivative(z + dt * k3, u)
        return z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    rng = np.random.default_rng(0)
    directions = []
    for _ in range(4):
        dz = np.zeros(dimension)
        dz[rng.integers(dimension)] = rng.uniform(-1.0, 1.0)
        du = np.zeros(ninput)
        du[rng.integers(ninput)] = rng.uniform(-1.0, 1.0)
        directions.append((dz, du))

    def error_at(scale: float) -> float:
        worst = 0.0
        for dz, du in directions:
            nonlinear = step(point + scale * dz, base + scale * du) - step(point, base)
            epsilon = 1.0e-4
            linear = (step(point + epsilon * scale * dz, base + epsilon * scale * du)
                      - step(point, base)) / epsilon
            worst = max(worst, float(np.max(np.abs(nonlinear - linear))))
        return worst

    coarse, fine = error_at(1.0), error_at(0.1)
    tiny = error_at(0.01)
    return dict(dt=dt, perturbation="dz 1 rad-ish / du 1 unit，逐级缩小 10 倍",
                error_coarse=coarse, error_fine=fine, error_tiny=tiny,
                ratio_coarse_to_fine=coarse / max(fine, 1e-30), ratio_fine_to_tiny=fine / max(tiny, 1e-30))


def run_case(variant: str, case: str):
    v_cmd, w_cmd, push, duration = CASES[case]
    sim = Gbc486MpcWbcSimulation(SimulationConfig(backend="nullspace", model_variant=variant))
    disturbances = [] if push is None else [Disturbance(PUSH_TIME, delta_velocity=push)]
    summary = sim.run(make_command(v_cmd, w_cmd), duration, disturbances)
    model = summary.pop("model")
    summary.update(case=case, variant=variant, v_cmd=v_cmd, w_cmd=w_cmd)
    summary["metrics"] = (case_metrics(sim.records, push is not None) if push is not None or case == "stand30"
                          else command_metrics(sim.records, v_cmd, w_cmd))
    if variant == "leg_input":
        legs = np.array([r["leg"] for r in sim.records])
        phi = 0.5 * (legs[:, 1] + legs[:, 3])
        summary["leg_angle_tracking"] = dict(
            max_abs_dev=float(np.max(np.abs(phi - sim.pose.angle))),
            ref_max_abs_dev=float(np.max([abs(r["ref"][6]) for r in sim.records if r["ref"] is not None] or [0.0])),
            max_abs_error=float(np.max([abs(r["ref"][6] - (0.5 * (r["leg"][1] + r["leg"][3]) - sim.pose.angle))
                                        for r in sim.records if r["ref"] is not None] or [0.0])))
    return summary, sim.records, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup_logging(default_level="WARNING")
    RESULTS.mkdir(exist_ok=True)
    revision = circleg_revision()

    # 1. 模型矩阵与线性化自洽性
    model, data = R.load()
    pose = R.standing_pose(model, data, 0.42)
    dyn = Gbc486Dynamics(model, data)
    models = {}
    for variant in VARIANTS:
        builder = (LegInputModelBuilder(dyn, pose.length) if variant == "leg_input"
                   else HeldLegModelBuilder(dyn))
        reduced = builder.build()
        models[variant] = dict(
            state_dim=reduced.state_dim, input_dim=reduced.input_dim,
            A=reduced.A.tolist(), B=reduced.B.tolist(),
            eigen_real=[float(v.real) for v in np.linalg.eigvals(reduced.A)],
            unstable_pole=float(reduced.unstable_pole), half_track=reduced.half_track,
            yaw_inertia=reduced.yaw_inertia, wheel_radius=reduced.wheel_radius,
            metadata={k: v for k, v in reduced.metadata.items() if k != "leg_state"},
            linearization=linearization_checks(builder, variant == "leg_input"))
        print(f"{variant}: dim {reduced.state_dim}/{reduced.input_dim} pole {reduced.unstable_pole:.3f} "
              f"lin err {models[variant]['linearization']['error_tiny']:.2e} "
              f"ratio {models[variant]['linearization']['ratio_fine_to_tiny']:.0f}x", flush=True)

    # 2. 闭环
    runs = []
    for variant in args.variants:
        for case in args.cases:
            summary, records, _ = run_case(variant, case)
            runs.append(summary)
            with gzip.open(RESULTS / f"m5_{variant}_{case}.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(records, handle)
            print(f"{variant:10s} {case:9s} completed={summary['completed']} stop={summary['stop_reason']} "
                  f"t={summary['simulated']:.2f}", flush=True)
    payload = dict(mujoco=mujoco.__version__, circleg=revision, models=models, runs=runs)
    (RESULTS / "m5_model_compare.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    text = to_markdown(payload)
    (RESULTS / "m5_model_compare.md").write_text(text, encoding="utf-8")
    print(text)


def to_markdown(p) -> str:
    def f(v, n=2, scale=1.0):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * scale:.{n}f}"
    variants = list(p["models"])
    lines = [
        "# M5：6 状态 held-leg 与 8 状态 leg-input 对比", "",
        f"脚本：`tools/m5_model_compare.py`（mujoco {p['mujoco']}；circleg HEAD {p['circleg'].get('head')}，"
        f"`controller.py` = {p['circleg'].get('imported_sha256', {}).get('Algorithm/mpc/controller.py')}）。",
        "L0 = 0.42 m，零空间法 WBC，增益 / 权重为 circleg 默认值，**未调参**。", "",
        "## 1. 模型", "",
    ]
    for name in variants:
        m = p["models"][name]
        A, B = np.array(m["A"]), np.array(m["B"])
        lines += [f"### {name}（状态 {m['state_dim']}，输入 {m['input_dim']}）", "",
                  "```", "A =", np.array2string(A, precision=4, suppress_small=True), "B =",
                  np.array2string(B, precision=4, suppress_small=True), "```", "",
                  f"- 特征值实部：{np.round(m['eigen_real'], 4).tolist()}；不稳定极点 **{m['unstable_pole']:.3f} rad/s**",
                  f"- 轮半径 {m['wheel_radius']}，半轮距 {m['half_track']:.4f}，偏航惯量 {m['yaw_inertia']:.4f}",
                  f"- metadata：{ {k: (round(v, 8) if isinstance(v, float) else v) for k, v in m['metadata'].items()} }",
                  f"- 线性化自洽性（非线性降阶 ODE 一步 vs 线性预测，dt = {m['linearization']['dt']} s）："
                  f"扰动 1/0.1/0.01 倍时误差 {m['linearization']['error_coarse']:.2e} / "
                  f"{m['linearization']['error_fine']:.2e} / {m['linearization']['error_tiny']:.2e}，"
                  f"缩小比 {m['linearization']['ratio_coarse_to_fine']:.0f}x / "
                  f"{m['linearization']['ratio_fine_to_tiny']:.0f}x（二阶余项应为 ~100x/次）", ""]
    keys = ["completed", "stop_reason", "max_abs_pitch", "max_abs_roll", "max_leg_length_error", "max_leg_torque",
            "max_torque", "mpc_fallbacks", "wbc_fallback", "wbc_clip", "wbc_infeasible"]
    lines += ["## 2. 闭环对比（同工况）", ""]
    for case in dict.fromkeys(r["case"] for r in p["runs"]):
        lines += [f"### {case}", "",
                  "| 模型 | 完成 | 停机 | max\\|θ\\| ° | max\\|roll\\| mrad | 腿长误差 mm | max 腿力矩 N·m | "
                  "MPC fallback | WBC fallback/clip/infeasible |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in p["runs"] if x["case"] == case]:
            lines.append(
                f"| {r['variant']} | {'是' if r['completed'] else '否'} | {r['stop_reason'] or '—'} | "
                f"{f(r['max_abs_pitch'], 2, 57.29578)} | {f(r['max_abs_roll'], 2, 1e3)} | "
                f"{f(r['max_leg_length_error'], 2, 1e3)} | {f(max(r['max_torque'][:4]))} | {r['mpc_fallbacks']} | "
                f"{r['wbc_fallback']}/{r['wbc_clip']}/{r['wbc_infeasible']} |")
        m = [x for x in p["runs"] if x["case"] == case]
        if case == "stand30":
            lines += ["", "| 模型 | 末 2 s s 峰峰 mm | max\\|v\\| mm/s |", "|---|---|---|"]
            for r in m:
                lines.append(f"| {r['variant']} | {f(r['metrics']['tail_s_span'], 3, 1e3)} | "
                             f"{f(r['max_abs_v'], 2, 1e3)} |")
        elif case.startswith("push_"):
            lines += ["", "| 模型 | 峰值 s mm | 峰值 θ mrad | s 回位 s | θ 回位 s | 末 2 s s 峰峰 mm |",
                      "|---|---|---|---|---|---|"]
            for r in m:
                k = r["metrics"]
                lines.append(f"| {r['variant']} | {f(k['peak_s'], 1, 1e3)} | {f(k['peak_pitch'], 1, 1e3)} | "
                             f"{f(k['settle_s'])} | {f(k['settle_pitch'])} | {f(k['tail_s_span'], 2, 1e3)} |")
        elif "v_cmd" in m[0] and m[0]["v_cmd"]:
            lines += ["", "| 模型 | 稳态 v / 误差 m/s | 90% 上升 s | 停车 s / 距离 m | 刹车反向 mm/s | 轮力矩饱和拍数 |",
                      "|---|---|---|---|---|---|"]
            for r in m:
                k = r["metrics"]
                lines.append(f"| {r['variant']} | {f(k.get('cruise_v_mean'), 3)} / {f(k.get('cruise_v_error'), 3)} | "
                             f"{f(k.get('rise_90'))} | {f(k.get('stop_time'))} / {f(k.get('stop_distance'), 3)} | "
                             f"{f(k.get('brake_reverse'), 0, 1e3)} | {k['wheel_torque_saturated_steps']} |")
        if m[0]["w_cmd"]:
            lines += ["", "| 模型 | 稳态 ψ̇ / 误差 rad/s | 90% 上升 s | 末航向 rad |", "|---|---|---|---|"]
            for r in m:
                k = r["metrics"]
                lines.append(f"| {r['variant']} | {f(k.get('cruise_w_mean'), 3)} / {f(k.get('cruise_w_error'), 3)} | "
                             f"{f(k.get('rise_w_90'))} | {f(k.get('final_heading'), 3)} |")
        extra = [r for r in m if "leg_angle_tracking" in r]
        if extra:
            lines += ["", "8 状态模型：共模腿角（偏差）实测最大 %.2f mrad，参考最大 %.2f mrad，跟踪误差最大 %.2f mrad"
                      % (extra[0]["leg_angle_tracking"]["max_abs_dev"] * 1e3,
                         extra[0]["leg_angle_tracking"]["ref_max_abs_dev"] * 1e3,
                         extra[0]["leg_angle_tracking"]["max_abs_error"] * 1e3)]
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
