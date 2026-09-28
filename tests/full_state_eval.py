"""Full-state robustness cost for zero-command disturbance tests.

Every deviation is normalized by the square of an allowed value (Bryson rule)
so the terms are comparable. Allowances marked TODO have no measured basis yet
and are placeholders for trend comparison only.
"""
from __future__ import annotations

import logging

import numpy as np

from Algorithm.mpc import MotionCommand
from APP.mujoco_mpc_wbc import Disturbance, MuJoCoMpcWbcSimulation

_LOGGER = logging.getLogger(__name__)

_ALLOWED_POSITION = 0.10  # [m]
_ALLOWED_SPEED = 0.20  # [m/s]
# 姿态峰值：用户 2026-09-25 指定的验收上限 10 deg；作为积分与峰值项的归一化尺度。
_ALLOWED_PITCH = 0.1745  # [rad] = 10 deg
_ALLOWED_PITCH_RATE = 0.20  # [rad/s]  TODO 无实测依据
_ALLOWED_ROLL = 0.05  # [rad]
_ALLOWED_ROLL_RATE = 0.20  # [rad/s]  TODO 无实测依据
_ALLOWED_YAW = 0.10  # [rad]  TODO 无实测依据
_ALLOWED_YAW_RATE = 0.20  # [rad/s]  TODO 无实测依据
_ALLOWED_HIP = 0.10  # [rad]  TODO 无实测依据
_ALLOWED_HEIGHT = 0.01  # [m]  TODO 无实测依据
_WHEEL_TORQUE_LIMIT = 5.0  # [N*m] wheel ctrlrange in URDF_Model_plane.xml
_FAIL_PENALTY = 1000.0  # TODO 无实测依据

# 收敛判据与目标（与报告口径一致）：进入阈值并保持即视为收敛。
_SETTLE_PITCH = 0.02  # [rad] 姿态收敛判据
_SETTLE_POSITION = 0.02  # [m] 位移收敛判据
_SETTLE_TARGET = 2.0  # [s] 收敛时间目标（占位："越快越好"，暂用 2 s，可收紧）
_ALLOWED_POSITION_PEAK = 0.10  # [m] 位移峰值归一化尺度（占位：用户未定，暂用 0.10 m）
_ALLOWED_OVERSHOOT = 0.0175  # [rad] 反向过冲归一化尺度（1 deg，占位）

_PUSH_TIME = 5.0  # [s]
_EVALUATION_SECONDS = 12.0  # [s]

_BASE_CASES: dict[str, tuple[float, float, float]] = {
    "forward": (0.20, 0.0, 0.0),
    "backward": (-0.20, 0.0, 0.0),
    "lateral": (0.0, 0.20, 0.0),
}


def _integral(values: np.ndarray, times: np.ndarray) -> float:
    """Trapezoidal integral, independent of the numpy trapz/trapezoid rename."""
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(times)))


def _settle_time(times: np.ndarray, values: np.ndarray, threshold: float,
                 start: float) -> float:
    """从 ``start`` 后首次进入阈值并保持到窗口结束的时刻（相对 ``start`` 的秒数）。

    取“最后一次超出阈值”之后的第一个采样；窗口结束时仍超出则返回窗口长度。
    """
    mask = times >= start
    tt = times[mask]
    vv = values[mask]
    if tt.size == 0:
        return 0.0
    ok = np.abs(vv) < threshold
    if np.all(ok):
        return 0.0
    last_bad = np.flatnonzero(~ok)[-1]
    if last_bad >= tt.size - 1:
        return float(tt[-1] - start)
    return float(tt[last_bad + 1] - start)


def _overshoot(times: np.ndarray, values: np.ndarray, start: float) -> float:
    """主峰值方向的反向最大值（反向过冲）。"""
    vv = values[times >= start]
    if vv.size == 0:
        return 0.0
    peak_sign = float(np.sign(vv[int(np.argmax(np.abs(vv)))]))
    if peak_sign == 0.0:
        return 0.0
    opposite = vv[np.sign(vv) == -peak_sign]
    return float(np.max(np.abs(opposite))) if opposite.size else 0.0


def full_state_cost(summary: dict, records: list[dict], hip_columns: tuple[int, int]) -> tuple[float, dict[str, float]]:
    """Cost over every tracked state; smaller is better."""
    times = np.array([record["time"] for record in records])
    states = np.array([record["state"] for record in records])
    attitudes = np.array([record["attitude"] for record in records])
    qpos = np.array([record["qpos"] for record in records])
    qvel = np.array([record["qvel"] for record in records])
    torques = np.array([record["torque"] for record in records])
    # Height is a deviation from the standing height, not an absolute altitude.
    nominal_height = float(qpos[0, 2])
    series = {
        "position": (states[:, 0], _ALLOWED_POSITION),
        "speed": (states[:, 1], _ALLOWED_SPEED),
        "pitch": (states[:, 2], _ALLOWED_PITCH),
        "pitch_rate": (states[:, 3], _ALLOWED_PITCH_RATE),
        "yaw": (states[:, 4], _ALLOWED_YAW),
        "yaw_rate": (states[:, 5], _ALLOWED_YAW_RATE),
        "roll": (attitudes[:, 0], _ALLOWED_ROLL),
        "roll_rate": (qvel[:, 3], _ALLOWED_ROLL_RATE),
        "hip": (np.mean(qpos[:, list(hip_columns)], axis=1), _ALLOWED_HIP),
        "height": (qpos[:, 2] - nominal_height, _ALLOWED_HEIGHT),
    }
    terms = {
        name: _integral(values**2, times) / allowed**2
        for name, (values, allowed) in series.items()
    }
    terms["effort"] = _integral(np.sum(torques**2, axis=1), times) / _WHEEL_TORQUE_LIMIT**2
    terms["terminal_position"] = float(states[-1, 0] ** 2) / _ALLOWED_POSITION**2
    terms["terminal_speed"] = float(states[-1, 1] ** 2) / _ALLOWED_SPEED**2
    terms["terminal_pitch"] = float(states[-1, 2] ** 2) / _ALLOWED_PITCH**2
    terms["failure"] = 0.0 if summary["stop_reason"] is None else _FAIL_PENALTY
    # 收敛时间与峰值：姿态、位移各自显式参与（用户 2026-09-25 要求）。
    after = times >= _PUSH_TIME
    pitch_peak = float(np.max(np.abs(states[after, 2]))) if np.any(after) else 0.0
    position_peak = float(np.max(np.abs(states[after, 0]))) if np.any(after) else 0.0
    terms["pitch_peak"] = pitch_peak**2 / _ALLOWED_PITCH**2
    terms["position_peak"] = position_peak**2 / _ALLOWED_POSITION_PEAK**2
    terms["overshoot"] = _overshoot(times, states[:, 2], _PUSH_TIME)**2 / _ALLOWED_OVERSHOOT**2
    terms["pitch_settle"] = (_settle_time(times, states[:, 2], _SETTLE_PITCH, _PUSH_TIME)
                             / _SETTLE_TARGET) ** 2
    terms["position_settle"] = (_settle_time(times, states[:, 0], _SETTLE_POSITION, _PUSH_TIME)
                                / _SETTLE_TARGET) ** 2
    return float(sum(terms.values())), terms


def run_case(delta_velocity: tuple[float, float, float], seconds: float = _EVALUATION_SECONDS,
             config=None):
    simulation = MuJoCoMpcWbcSimulation(config=config)
    push = Disturbance(time=_PUSH_TIME, delta_velocity=tuple(float(v) for v in delta_velocity))
    summary = simulation.run(MotionCommand(velocity=0.0, yaw_rate=0.0), seconds, None, [push])
    return summary, simulation.records, simulation.hip_q


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    total = 0.0
    for name, delta_velocity in _BASE_CASES.items():
        summary, records, hip_columns = run_case(delta_velocity)
        cost, terms = full_state_cost(summary, records, hip_columns)
        total += cost
        share = {
            key: round(100.0 * value / cost, 2)
            for key, value in sorted(terms.items(), key=lambda item: -item[1])
            if value > 1.0e-9
        }
        _LOGGER.info("%-8s J_full=%.2f stop=%s share%%=%s", name, cost, summary["stop_reason"], share)
    _LOGGER.info("TOTAL J_full=%.2f", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
