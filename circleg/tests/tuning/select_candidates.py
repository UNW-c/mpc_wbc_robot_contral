"""第 13 次排查：边界补跑 + 候选选择。

读阶段 1~4 的 CSV -> 若最优点落在网格边界则扩一档补跑 -> 按 C9 规则挑 3 组候选。

运行：
    .venv/Scripts/python.exe -m tests.tuning.select_candidates
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.tuning.grid_sweep import (_PARAM_KEYS, _cost, _read_rows, _worker_init, _evaluate_one)

_OUT = Path("artifacts/tuning/grid/20260929")


def _row_params(row: dict) -> dict:
    return {key: (float(row[f"param_{key}"]) if key != "terminal_mode" else row[f"param_{key}"])
            for key in _PARAM_KEYS}


def main() -> int:
    all_rows = [row for stage in (1, 2, 3, 4) for row in _read_rows(_OUT / f"stage{stage}.csv")]
    all_rows = [row for row in all_rows if row.get("J")]
    all_rows.sort(key=lambda row: float(row["J"]))
    best = all_rows[0]
    best_params = _row_params(best)
    print(f"全局最优：J={float(best['J']):.3f} O1={float(best['O1']):.2f} "
          f"O2={float(best['O2']):.2f} O3={float(best['O3']):.2f}")
    print("  params:", {k: round(v, 3) if isinstance(v, float) else v
                        for k, v in best_params.items()})

    # 边界补跑：position（阶段 1 的下边界档 2e4 的最优被细化到 2e4，扩一档）
    extra_params = []
    for value in (25000.0, 30000.0, 40000.0):
        params = dict(best_params)
        params["position"] = value
        extra_params.append(params)
    # 最优点还落在 pitch（上边界 1.5e5）、pitch_rate（下边界 300）、
    # tracking_bandwidth（下边界 7）上——每个边界方向扩一档（各 3 组）。
    for name, values in (("pitch", (225000.0, 300000.0, 450000.0)),
                         ("pitch_rate", (150.0, 100.0, 50.0)),
                         ("tracking_bandwidth", (5.0, 4.0, 3.0))):
        for value in values:
            params = dict(best_params)
            params[name] = value
            extra_params.append(params)
    with Pool(12, initializer=_worker_init) as pool:
        extra = list(pool.imap_unordered(_evaluate_one, extra_params, chunksize=1))
    print("\n边界补跑（position）：")
    extra_rows = []
    for params, result in extra:
        cost = _cost(result)
        row = {"params": params, "J": cost, "feasible": bool(result.feasible),
               "O1": float(result.objectives[0]), "O2": float(result.objectives[1]),
               "O3": float(result.objectives[2])}
        extra_rows.append(row)
        print(f"  position={params['position']:.0f}: J={cost:.3f} 可行={result.feasible} "
              f"O1={row['O1']:.2f} O2={row['O2']:.2f} O3={row['O3']:.2f}")

    combined = []
    for row in all_rows:
        if row.get("feasible") != "True":
            continue
        combined.append({**_row_params(row), "_J": float(row["J"]), "_O1": float(row["O1"]),
                         "_O2": float(row["O2"]), "_O3": float(row["O3"])})
    for row in extra_rows:
        if row["feasible"]:
            combined.append({**row["params"], "_J": row["J"], "_O1": row["O1"],
                             "_O2": row["O2"], "_O3": row["O3"]})
    feasible_count = len(combined)
    best_j = min(item["_J"] for item in combined)
    candidates = {
        "tuned-best": min(combined, key=lambda item: item["_J"]),
        "tuned-fast": min(combined, key=lambda item: item["_O1"]),
        "tuned-gentle": min([item for item in combined if item["_J"] <= best_j * 1.10],
                            key=lambda item: item["_O3"]),
    }
    print(f"\n可行解 {feasible_count} 组；候选（J 最优 {best_j:.3f}）：")
    saved = {}
    for label, item in candidates.items():
        params = {key: value for key, value in item.items() if not key.startswith("_")}
        saved[label] = params
        print(f"  {label}: J={item['_J']:.3f} O1={item['_O1']:.2f} O2={item['_O2']:.2f} "
              f"O3={item['_O3']:.2f}")
        print("    ", {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in params.items()})
    target = _OUT / "candidates.json"
    target.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n候选已保存：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
