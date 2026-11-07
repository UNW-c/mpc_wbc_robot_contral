"""P3：零输入抗扰参数寻优（阶段 A QMC 铺点 → B NSGA-II → C 局部细化）。

用法：
    .venv/Scripts/python.exe -m tests.tuning.run_study                 # 全量（12 进程）
    .venv/Scripts/python.exe -m tests.tuning.run_study --phase-a 4 --phase-b 0 --phase-c 0 --jobs 2
    # 中断后重新运行同一条命令：JournalStorage + load_if_exists 会自动续跑

输出（artifacts/tuning/<时间戳>/）：journal.log、trials.csv、pareto.png、importance.csv、summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import optuna
from optuna.samplers import NSGAIISampler, QMCSampler
from optuna.storages import JournalFileOpenLock, JournalFileStorage, JournalStorage

from tests.tuning.objective import E5_PARAMS, evaluate

optuna.logging.set_verbosity(optuna.logging.WARNING)

_SEED = 20260928
_CONSTRAINT_KEYS = ("guard", "fallback", "pitch_peak_deg", "pitch_cross",
                    "hip_speed_peak", "hip_torque_occupation", "solve_p95_ms", "zeta")
# (name, low, high, scale)；scale ∈ {log, linear, categorical}
_SEARCH = (
    ("position", 1.0e4, 1.0e6, "log"),
    ("velocity", 50.0, 2.0e4, "log"),
    ("pitch", 5.0e3, 5.0e5, "log"),
    ("pitch_rate", 1.0e2, 1.0e4, "log"),
    ("hip_angle", 5.0, 500.0, "log"),
    ("hip_rate", 0.1, 30.0, "log"),
    ("terminal_scale", 1.0, 20.0, "log"),
    ("tracking_bandwidth", 7.0, 25.0, "log"),
    ("hip_tracking", 20.0, 500.0, "log"),
    ("hip_angle_limit", 0.55, 0.80, "linear"),
)


def _sample(trial: optuna.Trial) -> dict:
    params = {}
    terminal_mode = trial.suggest_categorical("terminal_mode", ["scale", "lqr"])
    params["terminal_mode"] = terminal_mode
    for name, low, high, scale in _SEARCH:
        if name == "terminal_scale" and terminal_mode != "scale":
            params[name] = 5.0
            continue
        if scale == "log":
            params[name] = float(trial.suggest_float(name, low, high, log=True))
        else:
            params[name] = float(trial.suggest_float(name, low, high))
    return params


def _objective(trial: optuna.Trial) -> tuple:
    params = _sample(trial)
    result = evaluate(params, horizon_after_push=_TAIL[0])
    trial.set_user_attr("params", params)
    trial.set_user_attr("feasible", bool(result.feasible))
    trial.set_user_attr("failure", result.failure)
    trial.set_user_attr("violations", {k: float(v) for k, v in result.violations.items()})
    trial.set_user_attr("wall_time", float(result.wall_time))
    trial.set_user_attr("per_scenario", {k: {
        "settle": float(v["settle"]), "peak_position": float(v["peak_position"]),
        "peak_pitch": float(v["peak_pitch"]), "pitch_cross": int(v["pitch_cross"]),
        "hip_speed_peak": float(v["hip_speed_peak"]), "hip_occ": float(v["hip_occ"]),
        "final_position_abs": float(v["final_position_abs"]),
    } for k, v in result.per_scenario.items()})
    return tuple(float(v) for v in result.objectives)


def _constraints(trial: optuna.trial.FrozenTrial) -> list:
    violations = trial.user_attrs.get("violations", {})
    return [float(violations.get(key, 0.0)) for key in _CONSTRAINT_KEYS]


def _feasible_trials(*studies: optuna.Study):
    """合并多个 study 的可行 trial（阶段 A/B 分开存储）。"""
    trials = [t for study in studies for t in study.trials]
    return [t for t in trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("feasible")]


def _candidates(feasible) -> dict:
    """按 C9"平衡"规则挑 3 组候选（fast / gentle / balanced）。"""
    if not feasible:
        return {}
    values = np.array([t.values for t in feasible])
    names = ["fast", "gentle", "balanced"]
    picked = {}
    best_fast = min(feasible, key=lambda t: t.values[0])
    picked["tuned-fast"] = best_fast.user_attrs["params"]
    base_o1 = best_fast.values[0]
    gentle_pool = [t for t in feasible if t.values[0] <= base_o1]
    if not gentle_pool:
        gentle_pool = feasible
    picked["tuned-gentle"] = min(gentle_pool, key=lambda t: t.values[2]).user_attrs["params"]
    low, high = values.min(axis=0), values.max(axis=0)
    span = np.where(high - low > 1e-12, high - low, 1.0)
    normalized = (values - low) / span
    index = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    picked["tuned-balanced"] = feasible[index].user_attrs["params"]
    del names
    return picked


def _pareto_front(feasible):
    if not feasible:
        return []
    values = np.array([t.values for t in feasible])
    keep = []
    for i in range(len(values)):
        dominated = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        if not np.any(dominated):
            keep.append(feasible[i])
    return keep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a", type=int, default=128, help="QMC 铺点次数")
    parser.add_argument("--phase-b", type=int, default=384, help="NSGA-II 次数（8 代 x 48）")
    parser.add_argument("--phase-c", type=int, default=32, help="每组候选的局部扰动次数")
    parser.add_argument("--jobs", type=int, default=12, help="并行进程数（默认 12，避免吃满 CPU）")
    parser.add_argument("--tail", type=float, default=4.0, help="推扰后仿真时长 [s]")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    _TAIL[0] = args.tail

    output = args.output or (Path("artifacts/tuning") / time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    # 实测：JournalStorage 在同机 12 进程下锁竞争严重（每个事件都要抢锁，
    # 总 CPU 只有 ~1 核）。改用 RDBStorage(SQLite)——官方支持多进程并行，
    # 锁粒度更好，配 120 s 的 busy timeout 重试。
    database = output / "optuna.db"
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{database.as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 120.0}})
    directions = ["minimize", "minimize", "minimize"]
    # Optuna 4.x 不能在 optimize() 里换 sampler，所以阶段 A/B 用两个 study 共享同一个 storage。
    study_a = optuna.create_study(
        study_name="p3-qmc", directions=directions, storage=storage, load_if_exists=True,
        sampler=QMCSampler(qmc_type="sobol", scramble=True, seed=_SEED))
    if len(study_a.trials) == 0:
        baseline = dict(E5_PARAMS)
        baseline["terminal_mode"] = "scale"
        study_a.enqueue_trial(baseline)
        print(f"输出目录：{output}")
        print(f"E5 基线已放入第 1 个 trial：{ {k: round(v, 4) if isinstance(v, float) else v for k, v in baseline.items()} }")
    else:
        print(f"续跑：阶段 A 已有 {len(study_a.trials)} 个 trial（{output}）")

    def progress(study_, trial):
        if trial.number % 10 == 0:
            feasible = len(_feasible_trials(study_a, study_b))
            best = min((t.values[0] for t in _feasible_trials(study_a, study_b)),
                       default=float("nan"))
            print(f"  [{study_.study_name}] trial {trial.number}: 可行 {feasible}，best O1={best:.2f}")

    print(f"阶段 A：QMC {args.phase_a} 次（{args.jobs} 进程）")
    study_b = optuna.create_study(
        study_name="p3-nsga", directions=directions, storage=storage, load_if_exists=True,
        sampler=NSGAIISampler(population_size=48, constraints_func=_constraints, seed=_SEED))
    study_a.optimize(_objective, n_trials=args.phase_a, n_jobs=args.jobs, callbacks=[progress])
    if len(study_b.trials) == 0:
        for trial in [t for t in study_a.trials
                      if t.state == optuna.trial.TrialState.COMPLETE][:48]:
            study_b.enqueue_trial(trial.user_attrs["params"])
    if args.phase_b > 0:
        print(f"阶段 B：NSGA-II {args.phase_b} 次")
        study_b.optimize(_objective, n_trials=args.phase_b, n_jobs=args.jobs, callbacks=[progress])
    picked = _candidates(_feasible_trials(study_a, study_b))
    if args.phase_c > 0 and picked:
        print(f"阶段 C：局部细化 {len(picked)} 组 x {args.phase_c} 次")
        rng = np.random.default_rng(_SEED)
        for label, params in picked.items():
            print(f"  候选 {label}: { {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()} }")
            for _ in range(args.phase_c):
                perturbed = dict(params)
                for key, value in params.items():
                    if key == "hip_angle_limit":
                        perturbed[key] = float(np.clip(value * (1.0 + rng.uniform(-0.1, 0.1)), 0.55, 0.80))
                    elif isinstance(value, float) and key != "terminal_mode":
                        perturbed[key] = float(value * 10 ** rng.uniform(-0.08, 0.08))
                study_b.enqueue_trial(perturbed)
        study_b.optimize(_objective, n_trials=args.phase_c * len(picked), n_jobs=args.jobs,
                         callbacks=[progress])

    # 保存（阶段 A/B 合并；只用标准库，不新增 pandas 依赖）
    import csv

    rows = []
    for study in (study_a, study_b):
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = {"study": study.study_name, "number": trial.number,
                   "O1": trial.values[0], "O2": trial.values[1], "O3": trial.values[2],
                   "feasible": trial.user_attrs.get("feasible", False),
                   "failure": trial.user_attrs.get("failure") or "",
                   "wall_time": trial.user_attrs.get("wall_time", 0.0)}
            for key, value in trial.user_attrs.get("violations", {}).items():
                row[f"violation_{key}"] = value
            for key, value in trial.user_attrs.get("params", {}).items():
                row[f"param_{key}"] = value
            rows.append(row)
    if rows:
        with (output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    feasible = _feasible_trials(study_a, study_b)
    front = _pareto_front(feasible)
    summary = {
        "trials": len(study_a.trials) + len(study_b.trials),
        "feasible": len(feasible),
        "pareto_size": len(front),
        "candidates": {label: params for label, params in _candidates(feasible).items()},
        "baseline_e5": {k: (v if not isinstance(v, float) else round(v, 5))
                        for k, v in dict(E5_PARAMS, terminal_mode="scale").items()},
        "jobs": args.jobs, "tail": args.tail,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    if front:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        values = np.array([t.values for t in front])
        colors = values[:, 2]
        figure, axis = plt.subplots(figsize=(7, 5))
        scatter = axis.scatter(values[:, 0], values[:, 1], c=colors, cmap="viridis", s=36)
        axis.set_xlabel("O1 return time [s] (lower better)")
        axis.set_ylabel("O2 peak / physical lower bound")
        axis.set_title("Pareto front (color = O3 hip swing [rad])")
        figure.colorbar(scatter, label="O3")
        figure.tight_layout()
        figure.savefig(output / "pareto.png", dpi=120)
    try:
        importances = {}
        for index, objective_name in enumerate(("O1", "O2", "O3")):
            importances[objective_name] = {
                key: float(value) for key, value in
                optuna.importance.get_param_importances(
                    study_b, target=lambda t, i=index: t.values[i]).items()}
        (output / "importance.json").write_text(json.dumps(importances, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    except Exception as error:  # 重要性分析失败不影响主结果
        print(f"重要性分析跳过：{error}")
    print(f"完成：trial={len(study_a.trials) + len(study_b.trials)} 可行={len(feasible)} 前沿={len(front)}")
    print(f"输出：{output}")
    return 0


_TAIL = [4.0]

if __name__ == "__main__":
    raise SystemExit(main())
