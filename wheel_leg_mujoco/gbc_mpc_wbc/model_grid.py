"""腿长 -> 降阶模型参数的查表（方案 A）。

本模块做什么
------------
降阶模型原本只在标称腿长 ``L0`` 处线性化一次，而实测 A/B 随腿长变化极大
（``max|dA|`` 在 L=0.14~0.42 上是 65.99 量级）。本模块把 8 个腿长点的参数
（``model_grid_leg_length.json``，由 ``../tools/make_model_grid.py`` 离线生成）
载入，并对任意腿长做**分段线性插值**，供 MPC 换挡使用。

输入 / 输出
-----------
输入：腿长 ``L`` [m]（调用方给实测值）
输出：``{"A": (6x6 或 8x8), "B": ..., "metadata": {...}}`` 参数字典

本模块不做什么
--------------
- 不建模型、不做线性化（那是 ``model_builder.py`` 与离线生成脚本的事）
- 不持有 MPC、不改 MPC 对象（换挡由 ``simulation.py::_schedule_mpc_leg_length`` 负责）
- 不配置日志（见 ``logging_setup.py``）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from .logging_setup import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)

_HERE = Path(__file__).with_name("model_grid_leg_length.json")
CAPACITY_PREFIX = "capacity_gbc486_"
_WHEEL_RADIUS_TOLERANCE = 1.0e-9


def capacity_path(leg_length: float) -> Path:
    """按腿长取能力表路径；没有对应文件时退回通用名。"""
    exact = _HERE.with_name(f"{CAPACITY_PREFIX}{leg_length:.3f}.json")
    return exact if exact.exists() else _HERE.with_name(_CAPACITY_PREFIX + ".json")


class ModelGrid:
    """腿长 -> 降阶模型参数（分段线性）。线程/多进程下只读，无状态。"""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else _HERE
        if not self.path.exists():
            raise FileNotFoundError(f"缺少腿长网格数据：{self.path}（先跑 tools/make_model_grid.py）")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.grid = np.asarray(payload["grid"], dtype=float)
        self.entries = payload["entries"]
        # 预取成数组，避免每拍做字符串拼键
        self.A = np.asarray([self.entries[f"{l:.3f}"]["A"] for l in self.grid], dtype=float)
        self.B = np.asarray([self.entries[f"{l:.3f}"]["B"] for l in self.grid], dtype=float)
        self.pole = np.asarray([float(self.entries[f"{l:.3f}"]["unstable_pole"]) for l in self.grid])
        self.yaw_inertia = np.asarray([float(self.entries[f"{l:.3f}"]["yaw_inertia"])
                                       for l in self.grid])
        self._capacity_cache: dict[float, dict] = {}

    # ------------------------------------------------------------------ 查询
    def bracket(self, length: float) -> tuple[int, int, float]:
        """返回 (低点下标, 高点下标, 权重)。超出网格时钳到端点（不外推）。"""
        x = float(np.clip(length, self.grid[0], self.grid[-1]))
        upper = int(min(np.searchsorted(self.grid, x), len(self.grid) - 1))
        lower = max(upper - 1, 0)
        span = float(self.grid[upper] - self.grid[lower])
        weight = 0.0 if span <= 0.0 else (x - float(self.grid[lower])) / span
        return lower, upper, weight

    def interpolate(self, length: float) -> dict:
        """给定腿长返回该点的模型参数（A、B、极点、偏航惯量、能力表）。"""
        lower, upper, w = self.bracket(length)
        entry = self.entries[f"{self.grid[lower]:.3f}"]
        metadata = dict(entry.get("metadata", {}))
        # 能力表按“最近网格点”取（各点单独烘焙，不做插值，避免拼出非物理的表）
        nearest = float(self.grid[lower] if w < 0.5 else self.grid[upper])
        metadata.update(self.capacity(nearest))
        return {
            "A": (1.0 - w) * self.A[lower] + w * self.A[upper],
            "B": (1.0 - w) * self.B[lower] + w * self.B[upper],
            "unstable_pole": float((1.0 - w) * self.pole[lower] + w * self.pole[upper]),
            "yaw_inertia": float((1.0 - w) * self.yaw_inertia[lower] + w * self.yaw_inertia[upper]),
            "leg_length": float(np.clip(length, self.grid[0], self.grid[-1])),
            "metadata": metadata,
        }

    def capacity(self, length: float) -> dict:
        """取该腿长的能力表（accel_capacity / accel_capacity_tilted[_brake]）。"""
        key = round(float(length), 3)
        if key not in self._capacity_cache:
            path = capacity_path(key)
            if not path.exists():
                _LOGGER.warning("腿长 %.3f 的能力表不存在：%s", key, path)
                self._capacity_cache[key] = {}
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._capacity_cache[key] = dict(payload)
        return self._capacity_cache[key]
