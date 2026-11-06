"""轮腿参数寻优的评价工具（LQR 型：误差 + 出力 + 每项收敛时间）。

设计依据（用户 2026-10-04 指定）：
- 代价函数参考 LQR 的 ``x'Qx + u'Ru``：既算误差，也算出力；
- **每个状态通道都同时包含角度与角速度**：pitch(+rate)、腿角 φ_c(+φ̇_c)、
  位移 s(+v)、yaw(+yaw_rate)、roll(+roll_rate)、高度 L_c(+L̇_c)；
- 状态耦合不写进 Q 的非对角项，而是由**仿真闭环**体现（本机 A 的 A[1,2]=9.31、
  A[3,6]=-29.6、A[7,2]=-134、A[7,6]=106，pitch/腿角/位移强耦合）；
- 收敛时间**不设强制阈值**，越短越好；这里用"误差进入 5%·ref 并持续 1 s"作为
  **测量口径**（未收敛时折算为窗口长度 + 末端残余/ref，保持连续）；
- 优先级是**弱优先级**：优化低优先级状态时不得让高优先级状态变差（见
  :func:`degradation`），不靠权重硬压。

标量：
    可行   ``J = Σ_i w_i·(峰值误差_i/ref_i + 收敛时间_i/1s) + w_u·出力 + 弱优先级罚项``
    不可行 ``J = 100 + Σ max(0, 硬约束违反量)``
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from .simulation import (DRIVE_HIP_ANGLE_LIMIT, Disturbance, Gbc486MpcWbcSimulation, MotionCommand,
                         SimulationConfig, default_command)
from . import wbc
from .wbc import WbcGains

PUSH_TIME = 4.0
TAIL = 8.0
T_REF = 1.0              # 收敛时间归一化 [s]
BAND_FRACTION = 0.05     # 收敛判据的测量带宽 = 该通道 ref 的比例
SETTLE_HOLD = 1.0        # 进入带宽后需保持 [s]
W_EFFORT = 0.0           # 2026-10-05 用户决定：迭代参数时不用出力作为比较标准。
                         # effort 仍照常计算并写入报告，只退出 J 的排序。
                         # 历史（0.05 = 6 个执行器各自按力矩限归一化的出力项）供参考。
PRIORITY_TOL = 0.10      # 弱优先级校验容差

# MPC 降阶模型变体（用户 2026-10-04 确认）：回到 leg_input（8 状态）。
# 只用它 + **小腿角限幅**（hip_angle_limit 是 MPC 的硬状态约束）：
# 腿只承担一部分质心偏移（限幅封顶），剩下交给俯仰 → 小摆腿 + 小俯仰共同平衡。
# 实测（T-b +0.5）: limit=0.15 时腿角峰 150 mrad、俯仰峰 0.69°、位移 1.17 s 收敛；
#                   limit=0.25 时腿角峰 250 mrad、俯仰峰 0.28°、位移 0.79 s 收敛。
MODEL_VARIANT = "leg_input"

# 硬约束：**只把 pitch 对做成硬约束**（用户要求 pitch 优先；腿角由限幅天然限制，
# 不再参与硬约束，只进 J 且权重排第二）。
TIER1_CONSTRAINTS = {
    "pitch_peak": 0.10,       # 归一化峰值上限（×0.175 rad = 17.5 mrad = 1.0°）
    "pitch_settle": 0.5,      # [s]
}
TIER1_CHANNELS = ("pitch", "pitch_rate")

# 通道：(名称, ref, 所属优先级对的权重)。同一对的角/角速度共用权重。
CHANNELS = (
    ("pitch", 0.175, 1.00),
    ("pitch_rate", 1.00, 1.00),
    ("leg_angle", 0.65, 0.50),
    ("leg_rate", 5.00, 0.50),
    ("position", 0.104, 0.25),
    ("velocity", 0.50, 0.25),
    ("yaw", 0.10, 0.12),
    ("yaw_rate", 0.50, 0.12),
    ("roll", 0.10, 0.08),
    ("roll_rate", 1.00, 0.08),
    ("height", 0.05, 0.05),
    ("height_rate", 0.30, 0.05),
)
CHANNEL_NAMES = tuple(name for name, _, _ in CHANNELS)
CHANNEL_ORDER = {name: index for index, name in enumerate(CHANNEL_NAMES)}
# 执行器限幅（与 MJCF 一致）：腿 ±20 N·m、轮 ±6 N·m。
# 轮的 ±4 → ±6 由用户 2026-10-04 指定（原值 ±4 为模型自带，6 为待硬件规格确认的值）。
TORQUE_LIMITS = np.array([20.0, 20.0, 20.0, 20.0, 6.0, 6.0])

SCENARIOS = {
    "T-a": dict(disturbance=dict(delta_velocity=(0.2, 0.0, 0.0)), push_speed=0.2),
    "T-b": dict(disturbance=dict(delta_velocity=(0.5, 0.0, 0.0)), push_speed=0.5),
    "T-c": dict(disturbance=dict(delta_velocity=(-0.5, 0.0, 0.0)), push_speed=0.5),
    "T-d": dict(disturbance=dict(delta_velocity=(0.6, 0.0, 0.0)), push_speed=0.6),
    "T-e": dict(disturbance=dict(force=(40.0, 0.0, 0.0), torque=(0.0, 3.0, 0.0), duration=0.2),
                push_speed=None),
    "T-f": dict(disturbance=dict(delta_velocity=(0.5, 0.0, 0.0)), push_speed=0.5,
                second_velocity=(0.5, 0.0, 0.0)),
}
SCENARIO_ORDER = ("T-d", "T-f", "T-e", "T-b", "T-c", "T-a")   # 最易失败先跑（早停）
SCENARIO_NAMES = tuple(SCENARIO_ORDER)

VALIDATION = {
    "V-0.65": (0.65, 0.0, 0.0),
    "V-0.70": (0.70, 0.0, 0.0),
    "V--0.70": (-0.70, 0.0, 0.0),
    "V-0.80": (0.80, 0.0, 0.0),
}

_CONSTRAINTS = {
    "pitch_peak_deg": 10.0,
    "pitch_cross": 3.0,
    "leg_speed_peak": 20.0,          # [rad/s] 共模腿角速率峰
    "leg_torque_occupation": 0.8,
    "solve_p95_ms": 5.0,
}
PITCH_DEADBAND_DEG = 0.1
_PENALTY = 10.0

DEFAULTS = dict(
    position=20000.0, velocity=1333.3333333333333, pitch=150000.0, pitch_rate=300.0,
    yaw=60.0, yaw_rate=8.0, hip_angle=50.0, hip_rate=3.0, terminal_scale=5.0,
    attitude_kp=80.0, attitude_kd=10.0, height_kp=100.0, height_kd=12.0,
    regularize_kp=60.0, regularize_kd=6.0, tracking_bandwidth=7.0,
    # 腿角限幅（MPC 硬状态约束）："分给腿的质心偏移份额"旋钮，见 MODEL_VARIANT 注释
    hip_angle_limit=0.15,
    # 执行器限幅（备选参数）：源 MJCF 轮 ±4（用户 2026-10-04 改为 6）、腿 ±20
    wheel_torque_limit=6.0, leg_torque_limit=20.0,
    # WLS/QP 任务权重（按优先级 pitch > 腿角 > 位移速度 > 腿长 > yaw > roll）
    w_pitch=100.0, w_angle=60.0, w_forward=40.0, w_length=25.0, w_yaw=15.0, w_roll=10.0,
)
# WLS/QP 任务名 -> 参数名（resolver_factory 在构造仿真时读模块级 TASK_WEIGHTS）
_TASK_WEIGHT_KEYS = {
    "pitch": "w_pitch", "angle_R": "w_angle", "angle_L": "w_angle",
    "forward_velocity": "w_forward", "length_R": "w_length", "length_L": "w_length",
    "yaw": "w_yaw", "roll": "w_roll",
}


def _apply_task_weights(params: dict) -> None:
    for name, key in _TASK_WEIGHT_KEYS.items():
        wbc.TASK_WEIGHTS[name] = float(params[key])
# 当前部署（用户 2026-10-05 选定）：leg_input + 腿角预算 0.125。
# 依据：grid10 阶段 5（hip_angle_limit × position 二维联合扫描，15 组 8 可行）。
# 全训练集实测（最差工况）：腿角峰 125.1 mrad（7.2°）、俯仰峰 0.769°、位移峰 192.4 mm、位移收敛 1.98 s。
# 相对上一版（0.15）：腿角峰 -17%、位移峰 +15%、位移收敛 +11%、俯仰峰 -9%、俯仰收敛 +29%
# （0.432 s，仍 <0.5 s 硬约束）。
# 取舍依据：腿角峰恒等于 hip_angle_limit（与 position 无关），压摆幅只有降预算一条路；
# 再降到 0.10 时位移峰 +46%、收敛 +71%（见 grid10 CSV），代价不成比例。
# 2026-10-05 用户定档：**零输入抗扰取位移最小的那组**。腿角预算扫描（当前配置，位置权重 40000）：
#   0.125 -> 位移峰 192.4 mm / 收敛 1.98 s / 腿角 125 mrad / 俯仰 0.769 deg
#   0.200 -> 129.5 mm / 1.52 s / 200 mrad / 0.722 deg
#   0.250 -> 108.5 mm / 1.36 s / 250 mrad / 0.704 deg   <-- 采用（位移最小且可行）
#   0.300 -> 不可行（俯仰峰 4.27 deg，超 tier1 1.0 deg 判据）
# 注意：驱动段预算由 DRIVE_HIP_ANGLE_LIMIT 单独给（0.20），不再与保持段绑定；
# 驱动段"姿态倾角尽量小"那档的加速度限值仍由 θ=0 能力表在 0.20 处反查，不受这里影响。
TUNED = dict(DEFAULTS, position=40000.0, velocity=2666.7, pitch=600000.0, hip_angle_limit=0.25)

# 驱动阶段（DRIVE 及其后的 BRAKE）的共模腿角预算 [rad]。用户 2026-10-05 选定（方案"乙"）：
# 照 circleg 的"驱动放开、松键按 hip_limit_relax_rate 回落"逻辑。
# 依据：本机共模腿角机械可达范围 [-0.784, +0.674] rad（部署方案 §3.2 实测），
# 取 0.50 ≈ 下界的 74%，给关节限位留余量。circleg 的 1.2 rad 超出本机可达范围，不能照搬。
# 适用条件：**只作用于 DRIVE 及其后的 BRAKE 段**；HOLD/受扰段仍用 hip_angle_limit（当前 0.125），
# 所以零指令受扰工况（T-a..T-f）的既有结论不受影响。
# 待实测：驱动工况下该值是否够用（腿角是否长期顶住）——见部署方案 §12。
# 驱动阶段的腿角预算（DRIVE_HIP_ANGLE_LIMIT）从 simulation 导入：单一真值来源。

_MPC_WEIGHT_FIELDS = ("position", "velocity", "pitch", "pitch_rate", "yaw", "yaw_rate",
                      "hip_angle", "hip_rate", "terminal_scale")
_TEMPLATE: dict = {}


@dataclass
class EvaluationResult:
    metrics: dict
    channels: dict          # name -> (归一化峰值误差, 收敛时间 s)
    objectives: tuple       # (误差+收敛总分, 出力项, 弱优先级罚项)
    violations: dict
    violation_text: str
    feasible: bool
    failure: str | None
    wall_time: float
    per_scenario: dict


def _template(variant: str | None = None):
    """按模型变体构建模板仿真（各变体各自缓存），读出带本机物理量覆盖的 MPCSettings。

    必须按变体分开：6 状态模型没有腿角状态，ZMP 参考要自动关掉
    （``gbc486_mpc_settings`` 里有这个保护，但直接传 mpc_settings 会绕过它，之前就崩过）。
    """
    variant = variant or MODEL_VARIANT
    if variant not in _TEMPLATE:
        sim = Gbc486MpcWbcSimulation(SimulationConfig(backend="wls", model_variant=variant,
                                                      preset="tuned-agile"))
        # 自检（工作区 AGENTS.md §4）：DEFAULTS 必须与 preset 实际值一致。
        for key in _MPC_WEIGHT_FIELDS:
            actual = float(getattr(sim.mpc.settings.weights, key))
            if abs(float(DEFAULTS[key]) - actual) > 1e-9:
                raise ValueError(
                    f"objective.DEFAULTS['{key}']={DEFAULTS[key]} 与 preset(tuned-agile) 实际值 {actual} 不一致")
        _TEMPLATE[variant] = dict(settings=sim.mpc.settings, phi0=float(sim.pose.angle),
                                  length=float(sim.pose.length))
    entry = _TEMPLATE[variant]
    return entry["settings"], entry["phi0"], entry["length"]


def _settings(params: dict):
    base, phi0, length = _template(params.get("model_variant"))
    weights = replace(base.weights, **{key: float(params[key]) for key in _MPC_WEIGHT_FIELDS})
    settings = replace(base, weights=weights)
    if params.get("hip_angle_limit") is not None:
        limit = float(params["hip_angle_limit"])
        # 保持段用扫描出来的预算；驱动段用 DRIVE_HIP_ANGLE_LIMIT（用户 2026-10-05 选定，来自 simulation）
        settings = replace(settings, hip_angle_limit=limit,
                           drive_hip_angle_limit=DRIVE_HIP_ANGLE_LIMIT,
                           zmp_crank_limit=0.9 * DRIVE_HIP_ANGLE_LIMIT)
    # 限幅被 config 覆盖时，MPC 侧的力矩限也必须跟着改；否则模型允许 ±8
    # 而 MPC 仍按 ±6 规划（自检抓到的 bug）。
    if params.get("wheel_torque_limit") is not None:
        settings = replace(settings, wheel_torque_limit=float(params["wheel_torque_limit"]))
    if params.get("leg_torque_limit") is not None:
        settings = replace(settings, hip_torque_limit=float(params["leg_torque_limit"]))
    return settings, phi0, length


def _gains(params: dict) -> WbcGains:
    omega = float(params["tracking_bandwidth"])
    return WbcGains(
        attitude_gains=(float(params["attitude_kp"]), float(params["attitude_kd"])),
        height_gains=(float(params["height_kp"]), float(params["height_kd"])),
        regularize_gains=(float(params["regularize_kp"]), float(params["regularize_kd"])),
        tracking_velocity_gain=2.0 * omega, tracking_position_gain=omega * omega,
    )


def _make(params: dict, overrides: dict | None = None):
    settings, _, _ = _settings(params)
    if overrides:
        # 仅评测用的临时覆盖（如 ZMP 开关 / split），不改 TUNED
        settings = replace(settings, **overrides)
    gains = _gains(params)
    _apply_task_weights(params)
    config = SimulationConfig(backend=str(params.get("backend", "wls")),
                              model_variant=str(params.get("model_variant", MODEL_VARIANT)),
                              preset="tuned-agile", wbc_gains=gains,
                              wheel_torque_limit=params.get("wheel_torque_limit"),
                              leg_torque_limit=params.get("leg_torque_limit"))
    sim = Gbc486MpcWbcSimulation(config=config, mpc_settings=settings)
    sim.tasks.gains = gains    # simulation 会用 preset 带宽覆盖 wbc_gains，这里显式接管
    return sim


def _channel_errors(sim) -> dict:
    """从记录提取每个通道的误差时间序列（角度与角速度都在内）。"""
    records = sim.records
    _, phi0, length = _template(getattr(sim.config, "model_variant", MODEL_VARIANT))
    time = np.array([r["t"] for r in records], dtype=float)
    leg = np.array([r["leg"] for r in records], dtype=float)
    leg_angle = 0.5 * (leg[:, 1] + leg[:, 3]) - phi0
    height = 0.5 * (leg[:, 0] + leg[:, 2]) - length
    return time, {
        "pitch": np.array([r["pitch"] for r in records], dtype=float),
        "pitch_rate": np.array([r["pitch_rate"] for r in records], dtype=float),
        "leg_angle": leg_angle,
        "leg_rate": np.array([r["leg_rate"] for r in records], dtype=float),   # 雅可比口径
        "position": np.array([r["s"] for r in records], dtype=float),
        "velocity": np.array([r["v"] for r in records], dtype=float),
        "yaw": np.array([r["yaw"] for r in records], dtype=float),
        "yaw_rate": np.array([r["yaw_rate"] for r in records], dtype=float),
        "roll": np.array([r["roll"] for r in records], dtype=float),
        "roll_rate": np.array([r["roll_rate"] for r in records], dtype=float),
        "height": height,
        "height_rate": np.array([r["length_rate"] for r in records], dtype=float),  # 雅可比口径
    }


def channel_values(sim, event_time: float) -> dict:
    """每个通道的 (归一化峰值误差, 收敛时间 s)。收敛口径见模块说明。"""
    time, errors = _channel_errors(sim)
    window = time >= event_time
    if not np.any(window):
        return {name: (float("inf"), float("inf")) for name, _, _ in CHANNELS}
    dt = float(np.mean(np.diff(time))) if len(time) > 1 else 0.004
    hold = max(1, int(round(SETTLE_HOLD / dt)))
    span = float(time[-1] - event_time)
    out = {}
    for name, ref, _ in CHANNELS:
        error = errors[name]
        peak = float(np.max(np.abs(error[window]))) / ref
        band, settled = BAND_FRACTION * ref, None
        for index in range(len(time)):
            if time[index] >= event_time and index + hold <= len(time) \
                    and np.all(np.abs(error[index:index + hold]) < band):
                settled = float(time[index] - event_time)
                break
        if settled is None:      # 未收敛：窗口长度 + 末端残余/ref（连续，不分档）
            settled = span + float(np.min(np.abs(error[time >= time[-1] - SETTLE_HOLD]))) / ref
        out[name] = (peak, settled)
    return out


def _metrics(sim, summary, event_time: float) -> dict:
    records = sim.records
    window = np.array([r["t"] for r in records], dtype=float) >= event_time
    if not np.any(window):
        return dict(peak_pitch=float("nan"), pitch_cross=0, leg_speed_peak=float("nan"),
                    leg_occupation=float("nan"), effort=float("nan"),
                    mpc_fb=summary["mpc_fallbacks"], wbc_fb=summary["wbc_fallback"],
                    wbc_infeasible=summary["wbc_infeasible"], stop=f"early:{summary['stop_reason']}")
    torque = np.array([r["torque"] for r in records], dtype=float)
    pitch = np.array([r["pitch"] for r in records], dtype=float)
    _, errors = _channel_errors(sim)
    pitch_deg = np.degrees(pitch[window])
    significant = np.where(np.abs(pitch_deg) > PITCH_DEADBAND_DEG, np.sign(pitch_deg), 0.0)
    nonzero = significant[significant != 0.0]
    return dict(
        peak_pitch=float(np.max(np.abs(pitch_deg))),
        pitch_cross=int(np.sum(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0,
        leg_speed_peak=float(np.max(np.abs(errors["leg_rate"][window]))),
        leg_occupation=float(np.max(np.abs(torque[window, :4]))) / 20.0,
        effort=float(np.mean(np.sum((torque[window] / TORQUE_LIMITS) ** 2, axis=1))),
        mpc_fb=summary["mpc_fallbacks"], wbc_fb=summary["wbc_fallback"],
        wbc_infeasible=summary["wbc_infeasible"], stop=summary["stop_reason"],
    )


def _violations(metric: dict, solve_p95_ms: float) -> dict:
    return {
        "guard": 0.0 if metric["stop"] in (None, "viewer_closed") else 1.0,
        "fallback": float(metric["mpc_fb"] + metric["wbc_fb"] + metric["wbc_infeasible"]),
        "pitch_peak_deg": float(metric["peak_pitch"]) - _CONSTRAINTS["pitch_peak_deg"],
        "pitch_cross": float(metric["pitch_cross"]) - _CONSTRAINTS["pitch_cross"],
        "leg_speed_peak": float(metric["leg_speed_peak"]) - _CONSTRAINTS["leg_speed_peak"],
        "leg_torque_occupation": float(metric["leg_occupation"]) - _CONSTRAINTS["leg_torque_occupation"],
        "solve_p95_ms": float(solve_p95_ms) - _CONSTRAINTS["solve_p95_ms"],
    }


def worst_channels(per_scenario: dict) -> dict:
    """各通道取全部工况的最差值：(峰值, 收敛时间)。"""
    out = {}
    for name in CHANNEL_NAMES:
        peaks = [channels[name][0] for channels in per_scenario.values() if name in channels]
        settles = [channels[name][1] for channels in per_scenario.values() if name in channels]
        out[name] = (max(peaks) if peaks else float("inf"), max(settles) if settles else float("inf"))
    return out


def score(channels: dict) -> dict:
    """逐通道分数 S_i = w_i·(峰值 + 收敛时间/T_REF)。"""
    weights = {name: weight for name, _, weight in CHANNELS}
    return {name: weights[name] * (values[0] + values[1] / T_REF) for name, values in channels.items()}


def degradation(candidate: dict, baseline: dict) -> tuple[float, list]:
    """弱优先级校验：低优先级优化不得让高优先级变差。

    违反条件：存在 i 优先级高于 j，使得 S_j 改善（< 基线）而 S_i 恶化（> 基线·(1+tol)）。
    返回 (罚项, 违反说明列表)。
    """
    scores_c, scores_b = score(candidate), score(baseline)
    penalty, events = 0.0, []
    for high in CHANNEL_NAMES:
        worsened = scores_c[high] > scores_b[high] * (1.0 + PRIORITY_TOL)
        if not worsened:
            continue
        improved_low = [low for low in CHANNEL_NAMES
                        if CHANNEL_ORDER[low] > CHANNEL_ORDER[high]
                        and scores_c[low] < scores_b[low]
                        and scores_c[low] < scores_b[low] * (1.0 - PRIORITY_TOL)]
        if improved_low:
            delta = scores_c[high] - scores_b[high]
            penalty += delta
            events.append(f"{high} 恶化 {delta:+.3f}（同时改善：{','.join(improved_low)}）")
    return penalty, events


def _converged_gap(params: dict, first: Disturbance, margin: float = 0.5) -> float:
    """第一次扰动后位移基本收敛所需时间 + margin（用户要求：等收敛再加下一次扰动）。"""
    probe = _make(params)
    summary = probe.run(default_command(), PUSH_TIME + TAIL, [first])
    settle = channel_values(probe, PUSH_TIME)["position"][1]
    return settle + margin if np.isfinite(settle) else 2.0 + margin


def evaluate(params: dict, scenarios=None, baseline: dict | None = None,
             horizon_after_push: float | None = None) -> EvaluationResult:
    """评价一组参数：跑训练工况，返回通道分数、出力项与弱优先级罚项。"""
    start = time.perf_counter()
    names = [name for name in SCENARIO_ORDER if scenarios is None or name in scenarios]
    metrics_by_scenario, channels_by_scenario = {}, {}
    solve_p95_ms, failure, violations = 0.0, None, {}
    for name in names:
        spec = SCENARIOS[name]
        first = Disturbance(PUSH_TIME, **spec["disturbance"])
        if spec.get("second_velocity") is not None:
            gap = _converged_gap(params, first)
            event_time = PUSH_TIME + gap          # T-f 的收敛/峰值从第二次推扰起算
            disturbances = [first, Disturbance(event_time, delta_velocity=spec["second_velocity"])]
            duration = PUSH_TIME + gap + TAIL
        else:
            event_time = PUSH_TIME
            disturbances = [first]
            duration = PUSH_TIME + (TAIL if horizon_after_push is None
                                    else max(horizon_after_push, TAIL))
        sim = _make(params)
        summary = sim.run(default_command(), duration, disturbances)
        metric = _metrics(sim, summary, event_time)
        channels_by_scenario[name] = channel_values(sim, event_time)
        solve_p95_ms = max(solve_p95_ms, float(summary["timing_ms"]["mpc"]["p95"]))
        metrics_by_scenario[name] = metric
        violations = _violations(metric, solve_p95_ms)
        if any(value > 0.0 for value in violations.values()):
            failure = f"{name}: " + ", ".join(f"{k}={v:+.3f}" for k, v in violations.items() if v > 0.0)
            break
    wall = time.perf_counter() - start
    channels = worst_channels(channels_by_scenario)
    # 硬约束由 TIER1_CONSTRAINTS 驱动（键名形如 "<通道>_peak" / "<通道>_settle"），
    # 增减项只需改常量，不再两处同步。
    index_of = {"peak": 0, "settle": 1}
    tier1 = {}
    for key, limit in TIER1_CONSTRAINTS.items():
        channel, metric = key.rsplit("_", 1)
        tier1[key] = channels[channel][index_of[metric]] - float(limit)
    if failure is None:
        violations = {**violations, **tier1}
        if any(value > 0.0 for value in tier1.values()):
            failure = "tier1: " + ", ".join(f"{k}={v:+.3f}" for k, v in tier1.items() if v > 0.0)
    if failure is not None:
        return EvaluationResult(metrics_by_scenario, channels, (_PENALTY, _PENALTY, _PENALTY),
                                violations, failure, False, failure, wall, channels_by_scenario)
    scores = score(channels)
    # J 只含低优先级通道：位移/速度、yaw、roll、腿长（tier1 已由硬约束保证）
    error_term = float(sum(v for k, v in scores.items() if k not in TIER1_CHANNELS))
    effort = float(np.nanmax([m["effort"] for m in metrics_by_scenario.values()]))
    priority_penalty, events = (0.0, [])
    if baseline is not None:
        priority_penalty, events = degradation(channels, baseline)
    return EvaluationResult(metrics_by_scenario, channels, (error_term, effort, priority_penalty),
                            violations, "; ".join(events), True, None, wall, channels_by_scenario)


def scalar(result: EvaluationResult) -> float:
    if not result.feasible:
        return 100.0 + float(sum(max(0.0, v) for v in result.violations.values()))
    error_term, effort, priority_penalty = result.objectives
    return float(error_term) + W_EFFORT * float(effort) + float(priority_penalty)


def validate(params: dict, cases=None, tail: float = TAIL) -> dict:
    names = [name for name in VALIDATION if cases is None or name in cases]
    report = {}
    for name in names:
        sim = _make(params)
        summary = sim.run(default_command(), PUSH_TIME + tail,
                          [Disturbance(PUSH_TIME, delta_velocity=VALIDATION[name])])
        metric = _metrics(sim, summary, PUSH_TIME)
        report[name] = dict(passed=summary["stop_reason"] is None, stop=summary["stop_reason"],
                            peak_pitch_deg=round(metric["peak_pitch"], 2),
                            leg_speed_peak=round(metric["leg_speed_peak"], 2),
                            leg_occupation=round(metric["leg_occupation"], 3))
    return report


# ---------------------------------------------------------------- 驱动工况（2026-10-05）
# 现有 T-a..T-f 全是零指令受扰，**没有任何速度指令工况**；而 ZMP 参考按设计只在有加速度
# 指令时才起作用（部署方案 §5.6）。所以驱动工况是验证"速度输入 + ZMP"的前置条件。
#
# ``schedule = ((t0, v0), (t1, v1), ...)`` 为分段常值速度指令，t=0 起生效；
# ``duration`` 要给足，让"释放后的刹车 + 回位"走完。
DRIVE_CASES = {
    "D-a": dict(schedule=((0.0, 0.3), (2.0, 0.0)), duration=8.0),               # 阶跃 0.3，保持 2 s
    "D-b": dict(schedule=((0.0, 0.5), (2.0, 0.0)), duration=8.0),               # 阶跃 0.5，保持 2 s
    "D-c": dict(schedule=((0.0, 0.3), (0.6, 0.0)), duration=8.0),               # 中途释放（刹车段）
    "D-d": dict(schedule=((0.0, 0.2), (0.6, 0.4), (1.6, 0.0)), duration=8.0),   # 两级速度
    "D-e": dict(schedule=((0.0, 0.2), (1.2, -0.2), (2.6, 0.0)), duration=9.0),  # 中途反向
}
DRIVE_ORDER = tuple(DRIVE_CASES)


def drive_command(schedule):
    """分段常值指令表 -> ``command(t)``（与 circleg 的 callable command 同接口）。"""
    steps = tuple((float(start), float(value)) for start, value in schedule)

    def command(timestamp: float) -> MotionCommand:
        level = 0.0
        for start, value in steps:
            if timestamp + 1e-12 >= start:
                level = value
        return MotionCommand(velocity=level, yaw_rate=0.0)

    return command


def _steady_levels(schedule):
    """取出"非零指令的保持段"：(起点, 终点, 指令值)。"""
    out = []
    for index, (start, value) in enumerate(schedule):
        if abs(float(value)) < 1e-9:
            continue
        end = float(schedule[index + 1][0]) if index + 1 < len(schedule) else float("inf")
        out.append((float(start), end, float(value)))
    return out


def drive_metrics(sim, schedule) -> dict:
    """驱动工况指标：跟踪、摆幅、前倾、停止距离、占用、保护。"""
    record = sim.records
    time = np.array([r["t"] for r in record], dtype=float)
    velocity = np.array([r["v"] for r in record], dtype=float)
    position = np.array([r["s"] for r in record], dtype=float)
    pitch = np.degrees(np.array([r["pitch"] for r in record], dtype=float))
    leg = 0.5 * (np.array([r["leg"][1] for r in record])
                 + np.array([r["leg"][3] for r in record])) - float(sim.pose.angle)
    torque = np.array([r["torque"] for r in record], dtype=float)
    acceleration = np.gradient(velocity, time) if len(time) > 2 else np.zeros_like(velocity)

    steady_error, rise_time, steady_pitch = [], [], []
    for start, end, level in _steady_levels(schedule):
        window = (time >= end - 0.5) & (time <= end)
        if window.any():
            steady_error.append(abs(float(np.mean(velocity[window])) - level))
            steady_pitch.append(abs(float(np.mean(pitch[window]))))
        reached = np.where((time >= start) & (np.abs(velocity) >= 0.9 * abs(level))
                           & (np.sign(velocity) == np.sign(level)))[0]
        if reached.size:
            rise_time.append(float(time[reached[0]] - start))

    brakes = []
    for index in range(len(schedule) - 1):
        level = float(schedule[index][1])
        next_start, next_level = schedule[index + 1]
        if abs(level) < 1e-9 or abs(float(next_level)) >= abs(level):
            continue
        release = float(next_start)
        window = time >= release
        if not window.any():
            continue
        settled = float("nan")
        for i in range(len(time)):
            if time[i] < release:
                continue
            after = (time >= time[i]) & (time <= time[i] + 0.4)
            if after.any() and float(np.max(np.abs(velocity[after]))) < 0.05:
                settled = float(time[i] - release)
                break
        brakes.append(dict(release=release,
                           overshoot_mm=float(np.max(np.abs(position[window] - position[window][0]))) * 1e3,
                           brake_pitch_peak_deg=float(np.max(np.abs(pitch[window]))),
                           settle_s=settled))

    return dict(
        stop=sim.stop_reason,
        speed_peak=float(np.max(np.abs(velocity))),
        v_err_steady=float(max(steady_error)) if steady_error else float("nan"),
        rise_time_s=float(max(rise_time)) if rise_time else float("nan"),
        pitch_steady_deg=float(max(steady_pitch)) if steady_pitch else float("nan"),
        leg_peak_mrad=float(np.max(np.abs(leg))) * 1e3,
        pitch_peak_deg=float(np.max(np.abs(pitch))),
        wheel_occ=float(np.max(np.abs(torque[:, 4:6]))) / 6.0,
        leg_occ=float(np.max(np.abs(torque[:, :4]))) / 20.0,
        drift_mm=float(position[-1]) * 1e3,
        # 实测加减速峰值：用于校核 planner 手填的起步/刹车限值（AGENTS.md §5：那两个数目前无依据）
        a_peak_up=float(np.max(acceleration)),
        a_peak_down=float(np.min(acceleration)),
        fallback=int(sim.counters["wbc_fallback"]),
        infeasible=int(sim.counters["wbc_infeasible"]),
        brakes=brakes,
    )


def evaluate_drive(params: dict, cases=None, **overrides) -> dict:
    """跑驱动工况。``overrides`` 透传给 ``_make`` 的 MPCSettings（如 zmp_reference / zmp_split）。"""
    names = [name for name in DRIVE_ORDER if cases is None or name in cases]
    report = {}
    for name in names:
        spec = DRIVE_CASES[name]
        sim = _make(params, overrides or None)
        summary = sim.run(drive_command(spec["schedule"]), spec["duration"])
        report[name] = drive_metrics(sim, spec["schedule"])
        report[name]["guard"] = summary["stop_reason"]
    return report


if __name__ == "__main__":
    result = evaluate(DEFAULTS, scenarios=("T-a", "T-b"))
    print(f"feasible={result.feasible} failure={result.failure}")
    print(f"objectives (误差+收敛, 出力, 弱优先级罚) = {tuple(round(v, 4) for v in result.objectives)}")
    print(f"J = {scalar(result):.3f}")
    scores = score(result.channels)
    for name in CHANNEL_NAMES:
        peak, settled = result.channels[name]
        print(f"  {name:12s} peak/ref={peak:8.3f}  settle={settled:6.3f} s  S={scores[name]:.4f}")
    print(f"wall = {result.wall_time:.1f} s")
