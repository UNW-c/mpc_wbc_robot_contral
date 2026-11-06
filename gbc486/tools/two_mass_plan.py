"""两质量块飞行段规划器（只读，离线；不进控制路径）。

对应文档：`docs/agent交互文档_双质量块规划_方案部署.txt` 第 1、3 节原口径：
  **规划量为下块（轮）的绝对高度 z1(t)**，一条 C2 鼓包；
  上块的 z2(t) 由动量关系给出；腿长 L = z2 - z1 是差值；
  T 由触地腿长解出，h 由收腿下限解出 —— 都不是调出来的。
本工具在该口径上补两维：水平位置（接触点与质心偏差 d_n）与上块姿态 theta2。

物理（文档 1.2 / 1.4）：
  m1*z1'' = N - m1*g - F ,   N = m1*(z1''+g) + F
  分离条件  F = -m1*g = -38.5 N（两腿）
  z2'' = -k*z1'' - (1+k)*g ,  k = m1/m2
  F(t) = -m1*(z1''(t) + g) = m2*(z2''(t) + g)
  tau_c = I2*theta2''          （F 是中心力，不产生力矩）

导出量（不是输入）：L(t)、F(t)、tau_c(t)、离地间隙、行程、基座上升、触地速度。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gbc_mpc_wbc.logging_setup import setup_logging  # noqa: E402

_LOGGER = logging.getLogger(__name__)

GRAVITY = 9.81
LOWER_MASS = 3.924          # [kg] 两轮 + GH/OP
UPPER_MASS = 10.622         # [kg] 整机 14.546 - 3.924
UPPER_INERTIA = 0.1142      # [kg*m^2] 上块绕自身质心的俯仰惯量（MJCF 静位形近似）
L_MIN, L_MAX = 0.114, 0.430
F_LIMIT = 320.0             # [N] 单侧腿力上限
TAU_LIMIT = 40.0            # [N*m] 切向力矩上限


def shape(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f = 64*u^3*(1-u)^3 及其一、二阶导（端点 f=f'=f''=0）。"""
    f = 64.0 * u ** 3 * (1.0 - u) ** 3
    d1 = 64.0 * 3.0 * u ** 2 * (1.0 - u) ** 2 * (1.0 - 2.0 * u)
    d2 = 64.0 * 6.0 * u * (1.0 - u) * ((1.0 - 2.0 * u) ** 2 - u * (1.0 - u))
    return f, d1, d2


def solve_timing(v0: float, l0: float, l_td: float, l_tuck: float,
                 samples: int = 4001) -> tuple[float, float]:
    """由触地腿长解 T（文档 3.3），由收腿下限解 h。"""
    k = LOWER_MASS / UPPER_MASS
    acc = (1.0 + k) * GRAVITY
    disc = v0 * v0 - 2.0 * acc * (l_td - l0)
    if disc < 0.0:
        raise ValueError("触地腿长不可达：v0 不足以支撑到 l_td")
    duration = (v0 + float(np.sqrt(disc))) / acc
    u = np.linspace(0.0, 1.0, samples)
    f, _, _ = shape(u)
    free = l0 + v0 * (u * duration) - 0.5 * acc * (u * duration) ** 2
    # L(u) = free - (1+k)*h*f(u)；h 越大 L 越小，故取内部最小比值即为解
    interior = (u > 0.02) & (u < 0.98)
    ratio = (free[interior] - l_tuck) / ((1.0 + k) * f[interior])
    gap = float(np.min(ratio))
    return duration, gap


def build(v0: float, l0: float, l_td: float, l_tuck: float, dn_mm: float,
          theta_deg: float, z1_offset: float = 0.0,
          samples: int = 4001) -> dict:
    """生成飞行段两块轨迹与力。z1 以离地时刻为 0。"""
    k = LOWER_MASS / UPPER_MASS
    acc = (1.0 + k) * GRAVITY
    duration, h = solve_timing(v0, l0, l_td, l_tuck, samples)
    u = np.linspace(0.0, 1.0, samples)
    t = u * duration
    f, _, f_dd = shape(u)
    z1 = h * f + z1_offset
    z1_dd = h * f_dd / duration ** 2
    z2 = l0 + v0 * t - 0.5 * acc * t ** 2 - k * h * f + z1_offset
    z2_dd = -acc - k * h * f_dd / duration ** 2
    leg = z2 - z1
    x1 = np.zeros_like(t)
    x2 = x1 - dn_mm * 1e-3
    theta = np.deg2rad(theta_deg) * (6.0 * u ** 2 - 8.0 * u ** 3 + 3.0 * u ** 4)
    return dict(t=t, u=u, z1=z1, z2=z2, leg=leg, x1=x1, x2=x2, theta=theta,
                duration=duration, height=h,
                force=-LOWER_MASS * (z1_dd + GRAVITY),
                force_check=UPPER_MASS * (z2_dd + GRAVITY),
                torque=UPPER_INERTIA * np.gradient(np.gradient(theta, t), t),
                dn_mm=dn_mm)


def report(plan: dict) -> None:
    t, leg, force, torque = plan["t"], plan["leg"], plan["force"], plan["torque"]
    dn = plan["x1"] - plan["x2"]
    _LOGGER.info(
        "T=%.4f s | h=%.4f m | L=[%.4f, %.4f] | 离地间隙 %.4f m | 基座上升 %.4f m | "
        "触地速度 %.2f m/s",
        plan["duration"], plan["height"], leg.min(), leg.max(),
        plan["z1"].max(), float(plan["z2"].max() - plan["z2"][0]),
        float(np.gradient(plan["z2"], t)[-1]))
    _LOGGER.info(
        "F 峰 %.1f N（单腿 %.1f）| tau_c 峰 %.2f N*m | F(0)=%.1f F(T)=%.1f | "
        "d_n=%.1f mm | 力校验最大偏差 %.2e N",
        np.abs(force).max(), np.abs(force).max() / 2, np.abs(torque).max(),
        force[0], force[-1], float(np.abs(dn).mean() * 1e3),
        np.abs(force - plan["force_check"]).max())
    flags = []
    if leg.min() < L_MIN or leg.max() > L_MAX:
        flags.append("腿长越界")
    if np.abs(force).max() / 2 > F_LIMIT:
        flags.append("腿力越限")
    if np.abs(torque).max() > TAU_LIMIT:
        flags.append("切向力矩越限")
    _LOGGER.info("检查：%s", "、".join(flags) if flags else "全部通过")


def table(plan: dict, rows: int = 11) -> None:
    idx = np.linspace(0, len(plan["t"]) - 1, rows).astype(int)
    _LOGGER.info("  u     z1      z2      L        z1''     F       tau_c")
    for i in idx:
        _LOGGER.info("  %.1f  %+.4f  %+.4f  %.4f  %+7.1f  %+7.1f  %+6.2f",
                     plan["u"][i], plan["z1"][i], plan["z2"][i], plan["leg"][i],
                     plan["force"][i] / -LOWER_MASS - GRAVITY, plan["force"][i],
                     plan["torque"][i])


def main() -> int:
    parser = argparse.ArgumentParser(description="两质量块飞行段规划（只读）")
    parser.add_argument("--v0", type=float, default=2.29, help="离地竖直速度 [m/s]")
    parser.add_argument("--l0", type=float, default=0.283, help="离地腿长 [m]")
    parser.add_argument("--l-td", type=float, default=0.280, help="触地腿长 [m]")
    parser.add_argument("--l-tuck", type=float, default=0.160, help="收腿下限 [m]")
    parser.add_argument("--dn-mm", type=float, default=0.0)
    parser.add_argument("--theta-deg", type=float, default=0.0)
    parser.add_argument("--log-level", default=os.environ.get("TWO_MASS_LOG_LEVEL", "INFO"))
    args = parser.parse_args()
    setup_logging(args.log_level)

    base = build(args.v0, args.l0, args.l_td, args.l_tuck, 0.0, 0.0)
    report(base)
    table(base)
    for tag, dn, th in (("d_n=14.7 mm", 14.7, 0.0), ("theta=20 deg", 0.0, 20.0)):
        plan = build(args.v0, args.l0, args.l_td, args.l_tuck, dn, th)
        _LOGGER.info("[%s]", tag)
        report(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
