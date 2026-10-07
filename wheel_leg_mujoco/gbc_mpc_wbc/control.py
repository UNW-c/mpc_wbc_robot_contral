"""控制级：MPC 与 WBC 两个**互相异步**的求解级。

为什么分成两级
--------------
两者运行频率本来就不同（默认 MPC 50 Hz、WBC 250 Hz），所以这里不把谁嵌在谁里面：

- 各自用自己的 ``due(step)`` 判断该不该算（见 ``Algorithm/timing.py``）；
- WBC 取用的是**带时间戳的轨迹采样** ``sample_at(t)``，所以 MPC 本拍有没有更新都不影响它；
- 两级共同的输入只有采样级的 ``Feedback``，两级之间只通过 ``solution`` 单向传递。

这一级不做什么
--------------
- 不采样（见 ``sampling.py``）、不写执行器（见 ``output.py``）、不推进物理
- 不配置日志（见 ``logging_setup.py``）
- **不缓存 MPC 对象**：腿长换挡会整体重建 MPC，缓存会拿到旧对象（踩过的坑见
  ``docs/修改日志.md`` 第 32 次排查第 4 条）

副作用约定：WBC 级会**就地修改**传进来的 ``counters`` / ``exhausted`` / ``scaled``
三个统计字典（它们是报告口径的持有者，由 simulation 持有并最终交给 ``summary()``）。
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from . import _circleg  # noqa: F401
from .logging_setup import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class WbcOutcome:
    """WBC 级一拍的产出：力矩、停机原因，以及记录所需的中间量。"""

    torque: np.ndarray
    reason: str | None
    sample: object
    result: object
    predicted_normal: np.ndarray
    normal_violation_time: float
    infeasible_active: bool
    normal_warned: bool
    limit_warned: bool


def mpc_stage(*, control, feedback_accum, time_s: float, feedback, command,
              schedule_leg_length, timings: dict, counters: dict):
    """控制级 A：求解一次 MPC，返回 ``(solution, 本拍指令)``。

    与 WBC 无关：不等 WBC、不读 WBC 的力矩。腿长拟合参数调度在求解前执行（幂等，
    仅在最近网格点变化时才重建 MPC）。
    """
    current = command(time_s) if callable(command) else command
    start = time.perf_counter()
    schedule_leg_length()
    solution = control.solve_mpc(feedback.state, current, feedback.decision,
                                 feedback_accum.consume(feedback.state))
    timings["mpc"].append(time.perf_counter() - start)
    if solution is not None:
        counters["mpc_calls"] += 1
        counters["mpc_fallbacks"] += int(solution.fallback)
        counters["mpc_relaxed"] += int(bool(solution.qp.relaxed))
    return solution, current


def wbc_stage(*, control, tasks, dynamics, mpc, config, feedback, solution, feedback_accum,
              time_s: float, timing, timings: dict, counters: dict, exhausted: dict,
              scaled: dict, normal_violation_time: float,
              infeasible_active: bool, normal_warned: bool = False,
              limit_warned: bool = False) -> WbcOutcome:
    """控制级 B：求解一次 WBC，返回力矩与停机原因。

    ``mpc`` 由调用方**每拍传入**（不缓存）：腿长换挡会重建 MPC 对象，缓存会用到旧模型。
    """
    sample = (None if solution is None
              else solution.trajectory.sample_at(time_s + config.preview_lead))
    balance, motion = tasks.tasks(feedback.state, feedback.rows, sample)
    blocks = tasks.contact_blocks(feedback.rows, feedback.decision.contact)
    start = time.perf_counter()
    torque, result = control.solve_wbc(
        mass=dynamics.mass(), selection=dynamics.selection, dynamics_bias=dynamics.dynamics_bias(),
        contacts=blocks, balance_tasks=balance, motion_tasks=motion, limits=dynamics.limits,
        decision=feedback.decision, dt=timing.wbc_dt)
    timings["wbc"].append(time.perf_counter() - start)
    counters["wbc_calls"] += 1
    predicted_normal = np.zeros(0)
    if result is not None:
        counters["wbc_fallback"] += int(result.fallback_used)
        counters["wbc_clip"] += int(result.clip_applied)
        counters["wbc_infeasible"] += int(not result.feasible)
        # WBC -> MPC 反馈：把任务残差与饱和/越限回传给 planner（circleg APP:593 的同一位置）。
        # 缺这一句时 ReferenceFeedback.saturated 恒 False、motion_residual 恒 0，
        # planner.plan 的 blocked 永不成立，capability 不会因 WBC 受限而降级，
        # MPC 会持续规划 WBC 做不到的运动。
        feedback_accum.observe(result, dynamics.limits)
        exhausted_names = set()
        for item in result.hierarchy.outcomes:
            exhausted[item.name] = exhausted.get(item.name, 0) + int(item.exhausted)
            if item.exhausted:
                exhausted_names.add(item.name)
        for name, value in result.scaling.items():
            if name not in exhausted_names:
                scaled[name] = scaled.get(name, 0) + int(value < 1.0 - 1e-8)
        # 预测用反解：与采样级的实测反解是**同一个入口**，只是输入换成"控制器解出来的 q̈ + 解出的力矩"
        predicted_normal = dynamics.recover_contact_force(
            blocks, result.qacc, torque, external=dynamics.data.qfrc_applied)
    normal_threshold = -max(config.force_tolerance,
                            config.normal_force_margin_ratio * mpc.settings.nominal_normal_force)
    normal_min = (float(np.min(tasks.contact_normals(predicted_normal, feedback.decision.contact)))
                  if predicted_normal.size else float("inf"))
    # 关节限位顶住时反解不可信（限位约束力不在 h 里），本拍不累计越界时间
    if dynamics.contact_untrusted():
        normal_min = float("inf")
    normal_violation_time = (normal_violation_time + timing.wbc_dt
                             if normal_min < normal_threshold else 0.0)
    # WBC 求解不可行（强扰下 WLS 无法精确满足硬约束，力矩会被 clamp）：
    # 用户 2026-10-04 指示，这里不判失败、不停机，只记一条警告后继续；
    # 发散后的处置方案另行安排。力矩仍然是求解器输出的（已 clamp 到限幅内）。
    if result is not None and not result.feasible and not infeasible_active:
        _LOGGER.warning("WBC 求解不可行（动力学残差 %.3e，torque_clipped=%s）：仅警告，继续运行",
                        result.dynamics_residual_norm, result.clip_applied)
    infeasible_active = bool(result is not None and not result.feasible)
    # 以下两条属"控制器输出检查"，不是机体发散（用户 2026-10-07 要求）：
    # WBC 解出负的预测法向力、或力矩超出限幅，都**降级为警告 + 计数，不停机**——
    # 否则一次正常扰动就会直接终止仿真。限幅本身由执行器层 clamp 兜底（MuJoCo 会裁到 ctrlrange）。
    normal_active = normal_violation_time + 1e-12 >= config.normal_force_duration
    if normal_active:
        counters["negative_normal_force"] = counters.get("negative_normal_force", 0) + 1
        if not normal_warned:
            _LOGGER.warning("WBC 预测法向力为负已持续 %.3f s（阈值 %.3f s）：降级为警告，继续运行",
                            normal_violation_time, config.normal_force_duration)
    limit_active = not dynamics.limits.contains(torque)
    if limit_active:
        counters["torque_limit"] = counters.get("torque_limit", 0) + 1
        if not limit_warned:
            _LOGGER.warning("WBC 解出力矩超出执行器限幅：降级为警告，继续运行（执行器层会 clamp）")
    reason = None
    if feedback.other:
        reason = "unexpected_body_contact"
    elif mpc.degraded:
        reason = "persistent_mpc_fallback"
    return WbcOutcome(torque=torque, reason=reason, sample=sample, result=result,
                      predicted_normal=predicted_normal,
                      normal_violation_time=normal_violation_time,
                      infeasible_active=infeasible_active,
                      normal_warned=normal_active, limit_warned=limit_active)
