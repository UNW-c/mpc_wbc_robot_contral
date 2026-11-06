"""跳跃飞行段"路径预筛"（只读，不进控制路径）。

对应文档：`docs/agent交互文档_双质量块规划_方案部署.txt` 第 10.22 节。
目的：在把候选路径送进 MuJoCo 仿真之前，先按圆周坐标（半径 L / 极角 phi_w）
算出它需要多大的径向力与切向力矩，据此筛掉越限的候选。

本工具：
  - 不 import 控制层（只 import 纯函数 `leg_angle_anchor`）；
  - 不跑仿真、不写 results/；
  - 只输出峰值、限幅比较与切向冲量（角动量传递的代理量）。

圆周坐标（m1 = 下块质量，两块距离 L，极角 phi_w）：
    v_r = L'          v_t = L*phi'
    a_r = L'' - L*phi'^2          (含向心项)
    a_t = L*phi'' + 2*L'*phi'     (含科氏项)
    F_r  = m1*(a_r + g*cos(phi_w))          与 L 共轭
    tau_t = m1*L*(a_t + g*sin(phi_w))       与 phi 共轭

执行器估计（保守上界，用实测雅可比行的幅值）：
    |tau_act| <= 0.5*(|F_r|*|dL/dq|_max + |tau_t|*|dphi/dq|_max)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gbc_mpc_wbc.leg_angle_anchor import phi_eq  # noqa: E402
from gbc_mpc_wbc.logging_setup import setup_logging  # noqa: E402

_LOGGER = logging.getLogger(__name__)

GRAVITY = 9.81
LOWER_MASS = 3.924      # [kg] 两轮 + GH/OP，来源：MJCF 逐 body 质量
L_MIN, L_MAX = 0.114, 0.430
F_R_LIMIT = 320.0       # [N] 单腿径向力上限（36 N·m / 最深位形 |dL/dq|）
TAU_T_LIMIT = 40.0      # [N·m] 单腿切向力矩上限（MJCF ctrlrange）
TAU_ACT_LIMIT = 40.0    # [N·m] 单执行器上限
DLDQ_MAX = 0.1495       # [m/rad] 实测 L 行雅可比幅值
DPHI_DQ_MAX = 0.506     # [rad/rad] 实测 phi 行雅可比幅值


@dataclass(frozen=True)
class FlightPath:
    """一条候选路径：半径与极角的时间函数（离散采样）。"""

    name: str
    t: np.ndarray
    radius: np.ndarray        # [m]
    angle: np.ndarray         # [rad] 世界系极角


def _deriv(y: np.ndarray, t: np.ndarray, order: int = 1) -> np.ndarray:
    """对采样序列求一阶/二阶导（中心差分，端点用单边）。"""
    out = y
    for _ in range(order):
        out = np.gradient(out, t)
    return out


def _bump(u: np.ndarray) -> np.ndarray:
    """两端值/一阶/二阶导均为 0 的鼓包，峰值 1（与方案 3.2 的 f(u) 相同）。"""
    return 64.0 * u ** 3 * (1.0 - u) ** 3


def _flight_time(v0: float, l0: float, l_td: float) -> float:
    k = LOWER_MASS / (14.546 - LOWER_MASS)
    a = (1.0 + k) * GRAVITY
    disc = v0 * v0 - 2.0 * a * (l_td - l0)
    return (v0 + float(np.sqrt(max(disc, 0.0)))) / a


def path_radial_dominant(v0: float = 2.29, l0: float = 0.283,
                         l_tuck: float = 0.16, l_td: float = 0.28,
                         samples: int = 2001) -> FlightPath:
    """P-L：半径按双质量块规划（收腿到 l_tuck），极角跟随静态平衡 phi_eq。"""
    duration = _flight_time(v0, l0, l_td)
    t = np.linspace(0.0, duration, samples)
    k = LOWER_MASS / (14.546 - LOWER_MASS)
    a = (1.0 + k) * GRAVITY
    h = (l0 + v0 * v0 / (2.0 * a) - l_tuck) / (1.0 + k)
    u = t / duration
    radius = l0 + (v0 * t - 0.5 * a * t * t) - (1.0 + k) * h * _bump(u)
    angle = np.array([phi_eq(float(r)) for r in radius])
    return FlightPath("P-L 径向主导", t, radius, angle)


def path_tangential_dominant(v0: float = 2.29, l0: float = 0.283,
                             l_dip: float = 0.020, sweep: float = 0.12,
                             l_td: float = 0.28, samples: int = 2001) -> FlightPath:
    """P-phi：半径只做很小的起伏，极角按 sweep 幅度摆动。"""
    duration = _flight_time(v0, l0, l_td)
    t = np.linspace(0.0, duration, samples)
    u = t / duration
    radius = l0 - l_dip * _bump(u)
    base = float(phi_eq(l0))
    angle = base + sweep * _bump(u)
    return FlightPath("P-phi 切向主导", t, radius, angle)


def screen(path: FlightPath) -> dict:
    """按圆周坐标算出力/力矩峰值并与限幅比较。"""
    t = path.t
    radius = path.radius
    angle = path.angle
    r_d = _deriv(radius, t, 1)
    r_dd = _deriv(radius, t, 2)
    a_d = _deriv(angle, t, 1)
    a_dd = _deriv(angle, t, 2)
    # 每个量都是"两腿合计"，除以 2 得到单腿
    half = 0.5 * LOWER_MASS
    radial = half * (r_dd - radius * a_d ** 2 + GRAVITY * np.cos(angle))
    tangential = half * radius * (radius * a_dd + 2.0 * r_d * a_d + GRAVITY * np.sin(angle))
    torque_est = 0.5 * (np.abs(radial) * DLDQ_MAX + np.abs(tangential) * DPHI_DQ_MAX)
    return dict(
        name=path.name,
        duration=float(t[-1]),
        radius_min=float(radius.min()),
        radius_max=float(radius.max()),
        radial_peak=float(np.abs(radial).max()),
        tangential_peak=float(np.abs(tangential).max()),
        torque_peak=float(np.abs(torque_est).max()),
        radial_over=float(np.abs(radial).max()) > F_R_LIMIT,
        tangential_over=float(np.abs(tangential).max()) > TAU_T_LIMIT,
        torque_over=float(np.abs(torque_est).max()) > TAU_ACT_LIMIT,
        radius_over=bool(radius.min() < L_MIN or radius.max() > L_MAX),
        angle_sweep=float(angle.max() - angle.min()),
        tangential_impulse=float(np.trapezoid(tangential, t)),
    )


def _report(row: dict) -> None:
    _LOGGER.info(
        "%s: 时长 %.3f s | L=[%.4f,%.4f] | F_r峰 %.1f N | tau_t峰 %.1f N·m | "
        "tau_act估 %.1f N·m | 角度摆幅 %.3f rad",
        row["name"], row["duration"], row["radius_min"], row["radius_max"],
        row["radial_peak"], row["tangential_peak"], row["torque_peak"], row["angle_sweep"])
    flags = [label for label, key in (("F_r 越限", "radial_over"),
                                      ("tau_t 越限", "tangential_over"),
                                      ("tau_act 越限", "torque_over"),
                                      ("半径越限", "radius_over")) if row[key]]
    _LOGGER.info("  限幅：%s；切向冲量 ∫tau_t dt = %+.4f N·m·s",
                 ("、".join(flags) if flags else "全部通过"), row["tangential_impulse"])


def main() -> int:
    parser = argparse.ArgumentParser(description="跳跃飞行段路径预筛（只读）")
    parser.add_argument("--v0", type=float, default=2.29, help="离地时机体竖直速度 [m/s]")
    parser.add_argument("--l0", type=float, default=0.283, help="离地腿长 [m]")
    parser.add_argument("--l-tuck", type=float, default=0.16, help="P-L 的收腿下限 [m]")
    parser.add_argument("--l-dip", type=float, default=0.020, help="P-phi 的半径起伏 [m]")
    parser.add_argument("--sweep", type=float, default=0.12, help="P-phi 的极角摆幅 [rad]")
    parser.add_argument("--log-level", default=os.environ.get("PATH_PRESCREEN_LOG_LEVEL", "INFO"))
    args = parser.parse_args()
    setup_logging(args.log_level)

    rows = [screen(path_radial_dominant(args.v0, args.l0, args.l_tuck)),
            screen(path_tangential_dominant(args.v0, args.l0, args.l_dip, args.sweep))]
    for row in rows:
        _report(row)
    passed = [row["name"] for row in rows
              if not (row["radial_over"] or row["tangential_over"]
                      or row["torque_over"] or row["radius_over"])]
    _LOGGER.info("可送仿真的候选：%s", "、".join(passed) if passed else "无")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
