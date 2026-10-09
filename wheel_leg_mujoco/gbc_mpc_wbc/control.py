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
from .jump import GRAVITY  # noqa: E402
from Algorithm.wbc import ActuatorLimits  # noqa: E402

_LOGGER = get_logger(__name__)

# WP8（2026-10-09）：压缩/制动/释放相位让质心竖直任务在 WLS 中占主导的具名权重。
# 来源：用户 2026-10-09 批准（WP8 部署文档第三节）。只在跳跃相位覆盖，GROUND 不生效。
JUMP_COM_VERTICAL_WEIGHT = 5000.0


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
    # VP1（2026-10-09）：压缩/制动相位的等效力前馈（VMC）诊断字段。
    # feedforward_applied=False 时 tau_ff 全零、feedforward_force=0，表示本拍未启用前馈。
    tau_ff: np.ndarray | None = None
    feedforward_force: float = 0.0
    feedforward_applied: bool = False
    wheel_selection_cleared: bool = False


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
              jump_holds: dict | None = None, vmc_feedforward: bool = False,
              jump_config=None, leg_length_compression: bool = False,
              leg_length_thrust: bool = False) -> WbcOutcome:
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
    # VP1（2026-10-09）：VMC 前馈仅在 COMPRESS/COMPRESS_BRAKE 且开关打开时生效；
    # 这两个相位不创建 com_vertical 任务，由等效力前馈接管竖向目标。
    vmc_phase = bool(vmc_feedforward and jump is not None and jump.active
                     and jump.phase.value in ("COMPRESS", "COMPRESS_BRAKE"))
    # VL1（2026-10-09）：腿长空间加速度驱动。COMPRESS/COMPRESS_BRAKE 把加速度目标挂到
    # length_R/length_L 行，不创建 com_vertical；开关默认关时行为与当前完全一致。
    leg_length_phase = bool(leg_length_compression and jump is not None and jump.active
                            and jump.phase.value in ("COMPRESS", "COMPRESS_BRAKE"))
    # TR1（2026-10-09）：THRUST 也用腿长行加速度驱动（与压缩对称）；开关默认关时 THRUST
    # 行为与当前完全一致（com_vertical 权重 5000 路径）。
    leg_length_thrust_phase = bool(leg_length_thrust and jump is not None and jump.active
                                   and jump.phase.value == "THRUST")
    com_target = (None if (vmc_phase or leg_length_phase or leg_length_thrust_phase)
                  else desired_com_acceleration_z)
    balance, motion = tasks.tasks(feedback.state, feedback.rows, sample,
                                  desired_com_acceleration_z=com_target,
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
        # VL1（2026-10-09）：leg_length_compression 打开且相位为 COMPRESS/COMPRESS_BRAKE 时，
        # 不对 length_R/L 置 activation=0，而是把加速度目标换成 a_L（=jump.desired_com_acceleration_z）；
        # 其余相位与默认路径保持 activation=0（新部署 2026-10-09）。
        if leg_length_phase or leg_length_thrust_phase:
            a_L = float(jump.desired_com_acceleration_z)
            motion = tuple(replace(task, desired_acceleration=np.array([a_L]))
                           if task.name in ("length_R", "length_L") else task
                           for task in motion)
        elif jump.phase.value in ("COMPRESS", "COMPRESS_BRAKE", "THRUST"):
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
    # VP1（2026-10-09）：轮 selection 只在“无任何轮几何接触”时清零；任一接地就用完整 selection。
    selection = dynamics.selection
    wheel_selection_cleared = False
    if jump is not None and jump.active and not any(feedback.decision.contact):
        selection = np.array(dynamics.selection, dtype=float, copy=True)
        selection[list(dynamics.wheel_actuators), :] = 0.0
        wheel_selection_cleared = True
    # WP8（2026-10-09）：COMPRESS/COMPRESS_BRAKE/THRUST 期间给 com_vertical 传显式权重重载，
    # 其余任务权重不变；GROUND（jump 不活动）不传，行为与改动前一致。
    jump_weight_overrides = None
    if (jump is not None and jump.active
            and jump.phase.value in ("COMPRESS", "COMPRESS_BRAKE", "THRUST")):
        jump_weight_overrides = {"com_vertical": JUMP_COM_VERTICAL_WEIGHT}
    # VP1（2026-10-09）：等效力前馈。把 tau_ff 折进 dynamics_bias 与执行器限幅，
    # 而不是在 solve_wbc 之后直接叠加（否则限幅不自洽）。
    tau_ff = np.zeros(dynamics.model.nu)
    feedforward_force = 0.0
    dynamics_bias_solve = dynamics.dynamics_bias()
    limits_solve = dynamics.limits
    if vmc_phase:
        mass_total = (float(np.sum(dynamics.model.body_mass)) if jump_config is None
                      else float(jump_config.total_mass))
        a_z = float(jump.desired_com_acceleration_z)
        feedforward_force = 0.5 * mass_total * (GRAVITY + a_z)
        tau_ff, _peak = dynamics.thrust_torque(np.full(2, feedforward_force))
        tau_ff = np.asarray(tau_ff, dtype=float)
        dynamics_bias_solve = dynamics_bias_solve - np.asarray(selection, dtype=float).T @ tau_ff
        limits_solve = ActuatorLimits(dynamics.limits.lower - tau_ff, dynamics.limits.upper - tau_ff)
    torque, result = control.solve_wbc(
        mass=dynamics.mass(), selection=selection, dynamics_bias=dynamics_bias_solve,
        contacts=blocks, balance_tasks=balance, motion_tasks=motion, limits=limits_solve,
        decision=feedback.decision, dt=timing.wbc_dt, external_force=external,
        weight_overrides=jump_weight_overrides)
    if vmc_phase:
        torque = np.asarray(torque, dtype=float) + tau_ff
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
            # WP8：记录 com_vertical 的有效权重与传入的 overrides，便于核对 5000.0 是否生效。
            if jump_weight_overrides and entry["name"] == "com_vertical":
                entry["effective_weight"] = float(jump_weight_overrides["com_vertical"])
                entry["weight_overrides"] = dict(jump_weight_overrides)
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
                      task_trace=task_trace,
                      tau_ff=tau_ff, feedforward_force=float(feedforward_force),
                      feedforward_applied=bool(vmc_phase),
                      wheel_selection_cleared=bool(wheel_selection_cleared))
