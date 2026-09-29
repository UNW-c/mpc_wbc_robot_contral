"""补充扫描：hip_rate 沿敏感性提示方向（更小）扩展，并更新 candidates.json。

运行：
    .venv/Scripts/python.exe -m tests.tuning.refine_hip_rate
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.tuning.grid_sweep import _cost, _worker_init, _evaluate_one

_OUT = Path("artifacts/tuning/grid/20260929")


def main() -> int:
    candidates = json.loads((_OUT / "candidates.json").read_text(encoding="utf-8"))
    best = dict(candidates["tuned-best"])
    variants = []
    for value in (0.5, 1.0, 1.5, 2.0, 2.4):
        params = dict(best)
        params["hip_rate"] = value
        variants.append(params)
    print(f"hip_rate 补充扫描（当前 tuned-best 为 {best['hip_rate']}）：")
    with Pool(len(variants), initializer=_worker_init) as pool:
        results = list(pool.imap_unordered(_evaluate_one, variants, chunksize=1))
    best_cost = _cost_for(candidates)
    improved = None
    for params, result in results:
        cost = _cost(result)
        print(f"  hip_rate={params['hip_rate']}: J={cost:.3f} 可行={result.feasible} "
              f"O1={result.objectives[0]:.2f} O2={result.objectives[1]:.2f} "
              f"O3={result.objectives[2]:.2f}")
        if result.feasible and (improved is None or cost < improved[0]):
            improved = (cost, params)
    if improved is not None and improved[0] < best_cost:
        candidates["tuned-best"] = improved[1]
        (_OUT / "candidates.json").write_text(
            json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"tuned-best 已更新：J {best_cost:.3f} -> {improved[0]:.3f}"
              f"（hip_rate={improved[1]['hip_rate']}）")
    else:
        print(f"无改善（当前 best J={best_cost:.3f}）")
    return 0


def _cost_for(candidates: dict) -> float:
    from tests.tuning.objective import evaluate
    result = evaluate(candidates["tuned-best"], horizon_after_push=4.0)
    return _cost(result)


if __name__ == "__main__":
    raise SystemExit(main())
