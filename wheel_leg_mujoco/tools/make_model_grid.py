"""离线构建腿长网格上的降阶模型参数（方案 A 的数据基础）。

用户 2026-10-06 指定：
- **8 点足够**（本脚本用 0.14~0.42、步长 0.04，覆盖压缩终点到伸展终点）；
- **最简化的方式**：逐元素取值，运行时按腿长查表/线性内插；
- 运行时腿长由**高频解算**更新（已与控制器解耦），**MPC 每次运行时读取当前腿长并据此填参数**。

本脚本只做离线采样与落盘，**不改任何控制逻辑**。

输出：``gbc_mpc_wbc/model_grid_leg_length.json``
    { "grid": [...], "entries": { "0.14": {A,B,metadata...}, ... } }

注意：metadata 里的能力表（accel_capacity*）按腿长单值烘焙，腿长不符时 model_builder
会拒绝加载（model_builder.py:61）。所以每个网格点单独生成并各存一份。

用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/make_model_grid.py
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gbc_mpc_wbc import robot as R  # noqa: E402
from gbc_mpc_wbc.dynamics import Gbc486Dynamics  # noqa: E402
from gbc_mpc_wbc.model_builder import LegInputModelBuilder  # noqa: E402
from gbc_mpc_wbc.logging_setup import get_logger, setup_logging  # noqa: E402

GRID = (0.140, 0.180, 0.220, 0.260, 0.300, 0.340, 0.380, 0.420)   # [m] 8 点
OUTPUT = ROOT / "gbc_mpc_wbc" / "model_grid_leg_length.json"

logger = get_logger("make_model_grid")


def _jsonable(value):
    """把 numpy / tuple / dataclass 递归转成可 JSON 化的对象（不改语义）。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def sample(length: float) -> dict:
    model, data = R.load()
    R.standing_pose(model, data, length)
    dynamics = Gbc486Dynamics(model, data)
    reduced = LegInputModelBuilder(dynamics, length).build()
    entry = {name: _jsonable(value) for name, value in vars(reduced).items()}
    # 显式补齐：用户 2026-10-06 正推实测“随腿长变化”的量。它们是**属性**，vars() 抓不到，
    # 必须是拟合清单的一部分：A(8x8) max|dA|=65.99、B(8x3) max|dB|=6.60、
    # unstable_pole 12.098->10.466、yaw_inertia 0.4928->0.4511。其余（state_dim/input_dim/
    # wheel_radius/half_track/pitch_inertia）实测不随腿长变，一并记录以便运行时校验。
    for name in ("A", "B", "unstable_pole", "yaw_inertia", "state_dim", "input_dim",
                 "wheel_radius", "half_track", "pitch_inertia"):
        entry[name] = _jsonable(getattr(reduced, name, None))
    meta = entry.get("metadata", {})
    a_mat = np.asarray(getattr(reduced, "A", np.zeros((0, 0))), dtype=float)
    logger.info("L=%.3f m: A(8x8) max|A|=%.2f | B max|B|=%.2f | 极点 %.4f | 偏航惯量 %.4f | "
                "平衡俯仰 %.6f | tau_trim %.3f | 能力表=%s",
                length, float(np.max(np.abs(a_mat))) if a_mat.size else float("nan"),
                float(np.max(np.abs(np.asarray(getattr(reduced, "B", 0.0), dtype=float)))),
                float(getattr(reduced, "unstable_pole", float("nan"))),
                float(getattr(reduced, "yaw_inertia", float("nan"))),
                float(meta.get("equilibrium_pitch", float("nan"))),
                float(meta.get("tau_trim", float("nan"))),
                "有" if meta.get("accel_capacity") else "无")
    return entry


def main():
    setup_logging(default_level="INFO")
    entries = {}
    logger.info("开始采样：%d 个腿长点", len(GRID))
    for length in GRID:
        entries[f"{length:.3f}"] = sample(float(length))

    OUTPUT.write_text(json.dumps({"grid": list(GRID), "entries": entries},
                                 ensure_ascii=False), encoding="utf-8")
    logger.info("已写入 %s（%.1f MB）", OUTPUT, OUTPUT.stat().st_size / 1e6)

    # 打印关键标量随腿长的变化，供人工核对单调/合理性
    logger.info("键参数随腿长变化：")
    logger.info("  %-7s %-10s %-9s %-9s %-13s %-10s %-6s", "L[m]", "max|A|", "极点",
                "偏航惯量", "平衡俯仰[rad]", "tau_trim", "能力表")
    for length in GRID:
        e = entries[f"{length:.3f}"]
        meta = e.get("metadata", {})
        a_mat = np.asarray(e.get("A") or np.zeros(0), dtype=float)
        logger.info("  %-7.3f %-10.2f %-9.4f %-9.4f %-13.6f %-10.3f %-6s", length,
                    float(np.max(np.abs(a_mat))) if a_mat.size else float("nan"),
                    float(e.get("unstable_pole", float("nan"))),
                    float(e.get("yaw_inertia", float("nan"))),
                    float(meta.get("equilibrium_pitch", float("nan"))),
                    float(meta.get("tau_trim", float("nan"))),
                    "有" if meta.get("accel_capacity") else "无")


if __name__ == "__main__":
    main()
