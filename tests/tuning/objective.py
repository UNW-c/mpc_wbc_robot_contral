"""参数寻优的评价工具（第 7 次排查 P0）。

替代旧的代价函数（`tests/full_state_eval.py` 保留作历史对照，已知问题见 #20～#22：
髋项用 mean 恒为 0、`_ALLOWED_HIP=0.10` 会重罚新结构的主执行器、权重尺度多为占位）。
本模块只复用 `tests/acceptance_compare.py::_metrics` 的口径（F1 修正后），
早停在目标函数里自己实现（Optuna 的多目标不支持 pruner）。

对外接口：
    evaluate(params, scenarios, horizon_after_push) -> EvaluationResult
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MotionCommand, MPCSettings
from APP.mujoco_mpc_wbc import Disturbance, MuJoCoMpcWbcSimulation, SimulationConfig

_PUSH_TIME = 5.0

# 训练工况（P1）：推扰后只跑 horizon_after_push 秒
_TRAIN_SCENARIOS = {
    "T-a": dict(disturbances=[Disturbance(time=_PUSH_TIME, delta_velocity=(0.2, 0.0, 0.0))],
                push_speed=0.2, tail=4.0),
    "T-b": dict(disturbances=[Disturbance(time=_PUSH_TIME, delta_velocity=(0.5, 0.0, 0.0))],
                push_speed=0.5, tail=4.0),
    "T-c": dict(disturbances=[Disturbance(time=_PUSH_TIME, delta_velocity=(-0.5, 0.0, 0.0))],
                push_speed=0.5, tail=4.0),
    "T-d": dict(disturbances=[Disturbance(time=_PUSH_TIME, delta_velocity=(0.8, 0.0, 0.0))],
                push_speed=0.8, tail=4.0),
    "T-e": dict(disturbances=[Disturbance(time=_PUSH_TIME, force=(40.0, 0.0, 0.0),
                                          torque=(0.0, 3.0, 0.0), duration=0.5)],
                push_speed=None, tail=5.0, event_time=_PUSH_TIME + 0.5),
    "T-f": dict(disturbances=[Disturbance(time=_PUSH_TIME, delta_velocity=(0.5, 0.0, 0.0)),
                              Disturbance(time=_PUSH_TIME + 0.8, delta_velocity=(0.5, 0.0, 0.0))],
                push_speed=0.5, tail=4.8),
}
# 最易失败的工况先跑（早停顺序）
_SCENARIO_ORDER = ("T-d", "T-f", "T-e", "T-b", "T-c", "T-a")

# 硬约束门槛（P1）
_CONSTRAINTS = {
    "pitch_peak_deg": 10.0,
    "pitch_cross": 3.0,
    "hip_speed_peak": 20.0,
    "hip_torque_occupation": 0.8,
    "solve_p95_ms": 5.0,
}

# E4 基线（= 2026-09-28 部署的默认值）
E4_PARAMS = {
    "position": 80000.0,
    "velocity": 2000.0,
    "pitch": 51200.0,
    "pitch_rate": 960.0,
    "hip_angle": 50.0,
    "hip_rate": 10.0,
    "terminal_scale": 5.0,
    "tracking_bandwidth": 10.0,
    "hip_tracking": 100.0,
    "hip_angle_limit": 0.55,
}

_PENALTY = {"O1": 10.0, "O2": 10.0, "O3": 1.0}

_AMAX_CACHE: dict[float, float] = {}


def amax_for_limit(hip_angle_limit: float) -> float:
    """髋角限值对应的持续加速度上限 [m/s^2]（B1 稳态族的非线性插值，P0 要求）。"""
    key = round(float(hip_angle_limit), 4)
    if key not in _AMAX_CACHE:
        from APP.mujoco_balance_model import HipInputModelBuilder
        from APP.mujoco_mpc_wbc import MuJoCoDynamics
        from APP.mujoco_robot import MuJoCoRobot

        robot = MuJoCoRobot.load("plane")
        builder = HipInputModelBuilder(MuJoCoDynamics(robot))
        accelerations, hip_angles = [], []
        for a in np.linspace(0.2, 1.8, 9):
            solution = builder.steady_state(float(a))
            accelerations.append(float(a))
            hip_angles.append(abs(float(solution["q_c"])))
        _AMAX_CACHE[key] = float(np.interp(key, hip_angles, accelerations))
    return _AMAX_CACHE[key]


@dataclass
class EvaluationResult:
    metrics: dict
    objectives: tuple
    violations: dict
    feasible: bool
    failure: str | None
    wall_time: float
    per_scenario: dict


def _build(params: dict, horizon_after_push: float):
    base = MPCSettings(compensate_hip_torque=False)
    weights = replace(
        base.weights,
        position=float(params["position"]),
        velocity=float(params["velocity"]),
        pitch=float(params["pitch"]),
        pitch_rate=float(params["pitch_rate"]),
        hip_angle=float(params["hip_angle"]),
        hip_rate=float(params["hip_rate"]),
        terminal_scale=float(params["terminal_scale"]),
    )
    settings = replace(base, weights=weights, hip_angle_limit=float(params["hip_angle_limit"]))
    omega = float(params["tracking_bandwidth"])
    config = replace(
        SimulationConfig(model_variant="hip_input", forward_mode="mpc", preview_lead=0.0),
        tracking_velocity_gain=2.0 * omega,
        tracking_position_gain=omega * omega,
    )
    del horizon_after_push
    return config, settings


def _constraint_violations(metric: dict, solve_p95_ms: float) -> dict:
    violations = {
        "guard": 0.0 if metric["stop"] in (None, "viewer_closed") else 1.0,
        "fallback": float(metric["mpc_fb"] + metric["wbc_fb"]),
        "pitch_peak_deg": float(metric["peak_pitch"]) - _CONSTRAINTS["pitch_peak_deg"],
        "pitch_cross": float(metric["pitch_cross"]) - _CONSTRAINTS["pitch_cross"],
        "hip_speed_peak": float(metric["hip_speed_peak"]) - _CONSTRAINTS["hip_speed_peak"],
        "hip_torque_occupation": float(metric["hip_occ"]) - _CONSTRAINTS["hip_torque_occupation"],
        "solve_p95_ms": float(solve_p95_ms) - _CONSTRAINTS["solve_p95_ms"],
    }
    return violations


def evaluate(params: dict, scenarios=None, horizon_after_push: float = 4.0) -> EvaluationResult:
    """评价一组参数：按早停顺序跑训练工况，返回约束违反量与三个目标（各工况取最差）。"""
    from tests.acceptance_compare import _metrics

    start = time.perf_counter()
    names = [name for name in _SCENARIO_ORDER if scenarios is None or name in scenarios]
    config, settings = _build(params, horizon_after_push)
    metrics_by_scenario: dict = {}
    solve_times: list = []
    failure = None

    import APP.mujoco_mpc_wbc as app_module

    previous_weight = app_module._TASK_QP_WEIGHTS.get("hip_tracking")
    app_module._TASK_QP_WEIGHTS["hip_tracking"] = float(params["hip_tracking"])
    try:
        for name in names:
            spec = _TRAIN_SCENARIOS[name]
            duration = _PUSH_TIME + float(
                spec["tail"] if horizon_after_push is None else max(horizon_after_push, spec.get("tail", 0.0)))
            sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)
            sim.solution_log = []
            summary = sim.run(MotionCommand(velocity=0.0, yaw_rate=0.0), duration, None,
                              spec["disturbances"])
            event_time = spec.get("event_time", _PUSH_TIME)
            metric = _metrics(sim, summary, event_time)
            solve_times += [float(e["qp_solve_time"]) for e in sim.solution_log]
            metrics_by_scenario[name] = metric
            violations = _constraint_violations(
                metric, float(np.percentile(solve_times, 95)) * 1000.0 if solve_times else 0.0)
            if any(value > 0.0 for value in violations.values()):
                failure = f"{name}: " + ", ".join(
                    f"{key}={value:+.3f}" for key, value in violations.items() if value > 0.0)
                break
        wall = time.perf_counter() - start
        if failure is not None:
            return EvaluationResult(metrics=metrics_by_scenario,
                                    objectives=(_PENALTY["O1"], _PENALTY["O2"], _PENALTY["O3"]),
                                    violations=violations, feasible=False, failure=failure,
                                    wall_time=wall, per_scenario=metrics_by_scenario)
        # 目标（各工况取最差）
        o1 = max(float(m["settle"]) if np.isfinite(m["settle"]) else _PENALTY["O1"]
                 for m in metrics_by_scenario.values())
        a_max = amax_for_limit(float(params["hip_angle_limit"]))
        o2 = 0.0
        for name, metric in metrics_by_scenario.items():
            push_speed = _TRAIN_SCENARIOS[name]["push_speed"]
            if push_speed is None or metric["peak_position"] != metric["peak_position"]:
                o2 = max(o2, float(metric["peak_position"]) / 0.09)  # 持续推用名义下限代理
                continue
            lower = push_speed ** 2 / (2.0 * max(a_max, 1e-9))
            o2 = max(o2, float(metric["peak_position"]) / max(lower, 1e-9))
        o3 = max(float(m["hip_angle_peak"]) for m in metrics_by_scenario.values())
        return EvaluationResult(metrics=metrics_by_scenario, objectives=(o1, o2, o3),
                                violations=violations, feasible=True, failure=None,
                                wall_time=wall, per_scenario=metrics_by_scenario)
    finally:
        if previous_weight is None:
            app_module._TASK_QP_WEIGHTS.pop("hip_tracking", None)
        else:
            app_module._TASK_QP_WEIGHTS["hip_tracking"] = previous_weight


if __name__ == "__main__":
    result = evaluate(E4_PARAMS, horizon_after_push=4.0)
    print(f"feasible={result.feasible} failure={result.failure}")
    print(f"objectives (O1, O2, O3) = {tuple(round(v, 4) for v in result.objectives)}")
    print(f"violations: { {k: round(v, 4) for k, v in result.violations.items()} }")
    print(f"wall time = {result.wall_time:.1f} s（{len(result.metrics)} 个工况）")
