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

from dataclasses import dataclass, replace
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
    sample_raw: object
    result: object
    predicted_normal: np.ndarray
    normal_violation_time: float
    infeasible_active: bool
    normal_warned: bool
    limit_warned: bool
    raw_torque: np.ndarray
    tau_jump: np.ndarray
    task_trace: tuple[dict, ...]


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
              limit_warned: bool = False, jump=None, sample_override=None,
              desired_com_acceleration_z: float | None = None,
              jump_holds: dict | None = None) -> WbcOutcome:
    """控制级 B：求解一次 WBC，返回力矩与停机原因。

    ``mpc`` 由调用方**每拍传入**（不缓存）：腿长换挡会重建 MPC 对象，缓存会用到旧模型。
    """
    sample_raw = (None if solution is None
                  else solution.trajectory.sample_at(time_s + config.preview_lead))
    # WP5（2026-10-09）：跳跃期彻底切断 MPC sample 到 WBC 的输入。
    # - sample_raw 仍从 MPC solution 生成并写入 trace，作为 shadow 诊断；
    # - 普通 GROUND（jump 不活动）保持原行为：sample = sample_raw；
    # - jump.active 时：有固定 jump_reference 才用 sample_override，否则 sample=None，
    #   tasks.tasks() 必须走 sample=None 分支（pitch 用当前误差回零，不吃 MPC 参考）。
    if jump is not None and jump.active:
        sample = sample_override
    else:
        sample = sample_raw
    # E2/WP2：显式质心竖直目标直接进入 WBC 任务，None 时普通路径不变。
    # 新部署（2026-10-09）：jump_holds 为非空时，pitch/roll/angle_R/angle_L 改用冻结参考。
    holds = {} if jump_holds is None else jump_holds
    balance, motion = tasks.tasks(feedback.state, feedback.rows, sample,
                                  desired_com_acceleration_z=desired_com_acceleration_z,
                                  pitch_reference=holds.get("pitch"),
                                  roll_reference=holds.get("roll"),
                                  angle_R_reference=holds.get("angle_R"),
                                  angle_L_reference=holds.get("angle_L"))
    # (b) 腿长任务参考跟跳跃：wbc.py::tasks 里该任务参考标称 L0，跳跃期必须换成跳跃给出的目标腿长，
    #     否则 WBC 会把正在伸长的腿拉回去（与蹬伸力对撞，第 32 次排查 bug#1 的机理）。
    if jump is not None and jump.active and jump.leg_length_target is not None:
        leg = dynamics.leg_state()
        leg_rows = feedback.rows.leg[0]
        kp_h, kd_h = tasks.gains.height_gains
        rebuilt = []
        for task in motion:
            length_index = {"length_R": 0, "length_L": 2}.get(task.name)
            if length_index is None:
                rebuilt.append(task)
                continue
            rate = float(leg_rows[length_index] @ dynamics.data.qvel)
            target = kp_h * (jump.leg_length_target - float(leg[length_index])) - kd_h * rate
            rebuilt.append(replace(task, desired_acceleration=np.array([target])))
        motion = tuple(rebuilt)
    # (d) 跳跃期 WBC **不再是轮的执行权**（用户 2026-10-07 排查结论：压缩段轮力矩经地面摩擦产生
    #     水平力，实测 0.46 N*m 即造成 -15.6 mm 横移与 0.334 m/s 侧速）。两处一起改：
    #     ① 摘掉依赖轮的任务（位移/偏航）；② 轮执行器行从选择矩阵里清零——求解器因此把轮子
    #     当作**自由滚动**（角加速度由接触摩擦决定），从根上切断"用地面对抗竖直蹬伸"的通道。
    if jump is not None and jump.active:
        # 新部署（2026-10-09）：不再过滤任何任务，保留 tasks.tasks() 返回的全部任务对象与原顺序。
        # 只在 COMPRESS/COMPRESS_BRAKE/THRUST 用 activation=0 关闭两条腿长任务的输出权限，
        # 不改变它们的 Jacobian、bias、desired；LANDING/RECOVER 恢复原有腿长任务权限。
        if jump.phase.value in ("COMPRESS", "COMPRESS_BRAKE", "THRUST"):
            motion = tuple(replace(task, activation=0.0)
                           if task.name in ("length_R", "length_L") else task
                           for task in motion)
    blocks = tasks.contact_blocks(feedback.rows, feedback.decision.contact)
    # (c) WP3（2026-10-09）：正式路径不再调用 thrust_torque，也不再向 WBC 注入跳跃外部力。
    #     跳跃力统一由 com_vertical 任务求解；external_force 只保留真实外部扰动通道（当前为空）。
    external = None
    tau_jump = None
    task_trace = []
    for task in tuple(balance) + tuple(motion):
        entry = {
            "name": task.name,
            "desired_acceleration": np.asarray(task.desired_acceleration, dtype=float).copy(),
            "bias_acceleration": np.asarray(task.bias_acceleration, dtype=float).copy(),
            "activation": float(task.activation),
        }
        if task.name == "com_vertical":
            entry["jacobian"] = np.asarray(task.jacobian, dtype=float).copy()
        task_trace.append(entry)
    start = time.perf_counter()
    selection = dynamics.selection
    if jump is not None and jump.active:
        selection = np.array(dynamics.selection, dtype=float, copy=True)
        selection[list(dynamics.wheel_actuators), :] = 0.0
    torque, result = control.solve_wbc(
        mass=dynamics.mass(), selection=selection, dynamics_bias=dynamics.dynamics_bias(),
        contacts=blocks, balance_tasks=balance, motion_tasks=motion, limits=dynamics.limits,
        decision=feedback.decision, dt=timing.wbc_dt, external_force=external)
    raw_torque = np.asarray(torque, dtype=float).copy()
    timings["wbc"].append(time.perf_counter() - start)
    # 新部署（2026-10-09）：逐任务写出 name/desired/bias/activation/achieved/residual/
    # scaling/exhausted，不能只记录 com_vertical。
    if result is not None:
        outcome_by_name = {item.name: item for item in result.hierarchy.outcomes}
        for entry in task_trace:
            item = outcome_by_name.get(entry["name"])
            if item is None:
                continue
            entry["achieved_acceleration"] = np.asarray(item.achieved, dtype=float).copy()
            entry["residual"] = np.asarray(item.residual, dtype=float).copy()
            entry["scaling"] = float(result.scaling.get(entry["name"], float("nan")))
            entry["exhausted"] = bool(item.exhausted)
    task_trace = tuple(task_trace)
    # WP3：tau_jump 只保留为诊断字段（恒为0），不再参与求和；data.ctrl 就是 WBC 总力矩。
    tau_jump_trace = np.zeros_like(raw_torque)
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
    return WbcOutcome(torque=torque, reason=reason, sample=sample, sample_raw=sample_raw, result=result,
                      predicted_normal=predicted_normal,
                      normal_violation_time=normal_violation_time,
                      infeasible_active=infeasible_active,
                      normal_warned=normal_active, limit_warned=limit_active,
                      raw_torque=raw_torque, tau_jump=tau_jump_trace,
                      task_trace=task_trace)
