"""P2-b：闭环极点分析（给参数搜索区间，不跑仿真）。

对无约束 LTI 问题，MPC（终端 Riccati）的闭环行为等价于无限时域 LQR，
所以用 LQR 闭环极点 `eig(A - B K)` 评估各参数对收敛速度的影响。
离散极点换算为连续等效：`s = ln(lambda) / dt`。

运行：
    .venv/Scripts/python.exe -m tests.tuning.pole_analysis
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algorithm.mpc import MPCSettings
from APP.mujoco_balance_model import HipInputModelBuilder
from APP.mujoco_mpc_wbc import MuJoCoDynamics
from APP.mujoco_robot import MuJoCoRobot

_DT = 0.02

# P3 的初始搜索区间（对数空间）
_RANGES = {
    "position": (1.0e4, 1.0e6),
    "velocity": (2.0e2, 2.0e4),
    "pitch": (5.0e3, 5.0e5),
    "pitch_rate": (1.0e2, 1.0e4),
    "hip_angle": (5.0, 500.0),
    "hip_rate": (1.0, 100.0),
    "terminal_scale": (1.0, 20.0),
    "hip_angle_limit": (0.45, 0.65),
    "tracking_bandwidth": (5.0, 25.0),
    "hip_tracking": (20.0, 500.0),
}


def _model_and_scale():
    robot = MuJoCoRobot.load("plane")
    model = HipInputModelBuilder(MuJoCoDynamics(robot)).build_reduced(0.0)
    discrete = model.discretize(_DT)
    base_norm = float(np.linalg.norm(discrete.B[:, 0]))
    scale = np.ones(model.input_dim)
    for j in range(2, model.input_dim):
        scale[j] = base_norm / max(float(np.linalg.norm(discrete.B[:, j])), 1e-9)
    return model, discrete, scale


_STATE_NAMES = ("s", "v", "theta", "theta_dot", "psi", "psi_dot", "q_c", "q_c_dot")
_LONGITUDINAL_INDICES = (0, 1, 2, 3, 6, 7)


def closed_loop_analysis(values: dict) -> dict:
    """完整闭环分析：极点、特征向量归属、纵向子块的极点与阻尼比（#26）。

    无约束 LTI 下 MPC（终端 Riccati）等价于无限时域 LQR，所以用 eig(A-BK)。
    """
    from scipy.linalg import solve_discrete_are

    model, discrete, scale = _model_and_scale()
    weights = replace(MPCSettings().weights,
                      position=values.get("position", 80000.0),
                      velocity=values.get("velocity", 2000.0),
                      pitch=values.get("pitch", 51200.0),
                      pitch_rate=values.get("pitch_rate", 960.0),
                      hip_angle=values.get("hip_angle", 50.0),
                      hip_rate=values.get("hip_rate", 10.0),
                      terminal_scale=values.get("terminal_scale", 5.0))
    A = discrete.A
    B = discrete.B * scale[None, :]
    Q = np.diag(weights.state_diagonal(A.shape[0]))
    R = np.diag(np.maximum(weights.input_diagonal(model.input_dim) * scale ** 2, 1e-9))
    P = np.asarray(solve_discrete_are(A, B, Q, R), dtype=float)
    gain = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    eigenvalues, eigenvectors = np.linalg.eig(A - B @ gain)
    continuous = np.log(eigenvalues) / _DT
    entries = []
    for index, pole in enumerate(continuous):
        vector = np.abs(eigenvectors[:, index])
        total = float(np.sum(vector)) or 1.0
        shares = vector / total
        order = np.argsort(shares)[::-1]
        components = [(int(i), float(shares[i])) for i in order if shares[i] > 0.05]
        longitudinal_share = float(np.sum(shares[list(_LONGITUDINAL_INDICES)]))
        damping = (-pole.real / abs(pole)) if abs(pole) > 1e-9 else float("nan")
        entries.append(dict(pole=pole, components=components,
                            longitudinal_share=longitudinal_share, zeta=float(damping)))
    longitudinal = [entry for entry in entries if entry["longitudinal_share"] >= 0.5]
    dominant = max(longitudinal, key=lambda entry: entry["pole"].real) if longitudinal else None
    return dict(entries=entries, longitudinal=longitudinal, dominant=dominant)


def longitudinal_damping(values: dict) -> tuple:
    """纵向子块的最慢极点实部与阻尼比（供 #29d 的 ζ 预检使用）。"""
    analysis = closed_loop_analysis(values)
    dominant = analysis["dominant"]
    if dominant is None:
        return float("nan"), float("nan")
    return float(dominant["pole"].real), float(dominant["zeta"])


def continuous_poles(values: dict) -> np.ndarray:
    """给定权重参数，返回连续等效的闭环极点（实部）。"""
    from scipy.linalg import solve_discrete_are

    model, discrete, scale = _model_and_scale()
    weights = replace(MPCSettings().weights,
                      position=values.get("position", 80000.0),
                      velocity=values.get("velocity", 2000.0),
                      pitch=values.get("pitch", 51200.0),
                      pitch_rate=values.get("pitch_rate", 960.0),
                      hip_angle=values.get("hip_angle", 50.0),
                      hip_rate=values.get("hip_rate", 10.0),
                      terminal_scale=values.get("terminal_scale", 5.0))
    A = discrete.A
    B = discrete.B * scale[None, :]
    Q = np.diag(weights.state_diagonal(A.shape[0]))
    R = np.diag(np.maximum(weights.input_diagonal(model.input_dim) * scale ** 2, 1e-9))
    P = np.asarray(solve_discrete_are(A, B, Q, R), dtype=float)
    gain = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    discrete_poles = np.linalg.eigvals(A - B @ gain)
    return np.log(np.abs(discrete_poles)) / _DT


def main() -> int:
    print("P2-b 闭环极点分析（连续等效实部；越负越快）")
    base_poles = continuous_poles({})
    base_dominant = float(np.max(base_poles))
    print(f"当前默认（E4）：主导极点实部 = {base_dominant:+.2f} rad/s "
          f"（目标 -4 ~ -8），全部极点：{np.round(np.sort(base_poles), 2)}")
    print()
    print(f"{'参数':>20}{'档位':>12}{'主导极点':>10}{'是否在 -4~-8':>14}")
    for name, (low, high) in _RANGES.items():
        if name in ("hip_angle_limit", "tracking_bandwidth", "hip_tracking"):
            continue  # 与 MPC 极点无关（限值/执行器带宽）
        for factor, label in ((1.0, "低端"), (1.0, "中值"), (1.0, "高端")):
            if label == "低端":
                value = low
            elif label == "中值":
                value = float(np.sqrt(low * high))
            else:
                value = high
            poles = continuous_poles({name: value})
            dominant = float(np.max(poles))
            flag = "在" if -8.0 <= dominant <= -4.0 else "超出"
            print(f"{name:>20}{label:>10}({value:>10.3g}){dominant:>12.2f}{flag:>14}")
    print()
    print("WBC 跟踪带宽检查（需求：ω ≥ 2 × |MPC 主导极点|）：")
    need = 2.0 * abs(base_dominant)
    print(f"  |主导极点| = {abs(base_dominant):.2f} → 需要 ω ≥ {need:.2f} rad/s；"
          f"P3 区间 [5, 25] {'满足' if 25.0 >= need else '不足'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
