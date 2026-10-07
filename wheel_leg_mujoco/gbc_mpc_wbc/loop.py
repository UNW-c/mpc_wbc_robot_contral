"""运行循环：把「采样 / 控制 / 输出」四段按各自节拍串起来（本工作区唯一的时序权威）。

四段与节拍（用户 2026-10-07 要求：采样、输出、控制计算三者解耦；MPC 与 WBC 也解耦，
因为两者频率不同，按异步运行）：

    ==========  ==================  ==========================================
    阶段        节拍                做什么（实现在哪）
    ==========  ==================  ==========================================
    采样级      每物理步（1 kHz）   反馈：运动学 / 接触力 / 离地判定 / 状态 / 监督层
                                   见 ``sampling.py``
    控制级 A    MPC 节拍（50 Hz）   自己的 ``mpc_due``，与 WBC 异步，见 ``control.py``
    控制级 B    WBC 节拍（250 Hz）  自己的 ``wbc_due``，取带时间戳轨迹采样，见 ``control.py``
    输出级      每物理步（1 kHz）   写执行器（零阶保持）+ 推进物理 + 输出侧护栏
                                   见 ``output.py``
    ==========  ==================  ==========================================

节拍之间只用**局部变量**传递（``torque`` / ``feedback`` / ``solution``），不造包装类：
任何一级改频率都不影响其它级的口径。

本模块不做什么：不定义控制算法（见 ``control.py``）、不采样细节（见 ``sampling.py``）、
不碰执行器细节（见 ``output.py``）、不配置日志（见 ``logging_setup.py``）。
"""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from .logging_setup import get_logger  # noqa: E402

# viewer 刷新周期（仿真时间），约 60 fps。见本文件循环尾部 viewer.sync() 处的说明：
# 每物理步都 sync 会让 viewer 比实时慢 9 倍（渲染 91441 顶点的 base_link 等）。
VIEWER_SYNC_DT = 1.0 / 60.0
_LOGGER = get_logger(__name__)


def run_loop(sim, command, duration: float, disturbances=(), viewer=None,
             frame_callback=None, frame_interval: float = 0.0) -> None:
    """积分情景；传入 viewer 时按实时步长同步（与 circleg ``run`` 相同）。

    ``sim`` 是 ``Gbc486MpcWbcSimulation``：本函数只读它的装配好的零件
    （``dyn`` / ``control`` / ``tasks`` / ``mpc`` / ``timing`` / ``config`` ...），
    并把运行统计写回（``records`` / ``counters`` / ``distance`` / ``stop_reason``）。
    """
    m, d = sim.model, sim.data
    from .sampling import FeedbackSampler
    from .control import mpc_stage, wbc_stage
    from .output import advance_physics, write_actuator_output

    infinite = not math.isfinite(duration)
    if infinite and viewer is None:
        raise ValueError("无限时长需要 viewer 才能停下来")
    pending = sorted(disturbances, key=lambda item: item.time)
    steps = None if infinite else int(round(duration / m.opt.timestep))
    sampler = FeedbackSampler(
        model=m, data=d, dynamics=sim.dyn, tasks=sim.tasks,
        contact_estimator=sim.contact_estimator,
        supervisor=sim.supervisor, reduced=sim.reduced, nominal_angle=sim.pose.angle,
        pitch_limit=sim.mpc.settings.pitch_limit)
    force_until, wrench = 0.0, np.zeros(6)
    sim._command = None
    torque = np.zeros(m.nu)      # 输出级：最近一次控制解（零阶保持）
    feedback = None              # 采样级：最近一拍反馈
    solution = None              # MPC 级：最近一次解（WBC 与记录共用）
    reason = None
    normal_violation_time = 0.0
    infeasible_active = False
    airborne_active = False       # 离地档标志（由采样级的离地检测驱动）
    airborne_since = 0.0
    normal_warned = limit_warned = False
    wbc_warning = guard_warning = None   # 告警去抖：只在状态变化的那一拍打日志
    last_frame_time = -1.0e9      # 帧回调节流（离线录像用；frame_callback=None 时无开销）
    step = 0
    while steps is None or step < steps:
        step_wall_start = time.perf_counter()
        while pending and d.time >= pending[0].time - 1e-12:
            push = pending.pop(0)
            d.qvel[0:3] += np.asarray(push.delta_velocity, dtype=float)
            d.qvel[3:6] += np.asarray(push.delta_omega, dtype=float)
            if push.duration > 0.0:
                force_until = float(d.time) + float(push.duration)
                wrench = np.concatenate([np.asarray(push.force, dtype=float),
                                         np.asarray(push.torque, dtype=float)])
        # 持续扰动：在时长内每步施加广义外力（viewer 鼠标拖住不放）
        d.qfrc_applied[:] = 0.0
        if d.time < force_until:
            d.qfrc_applied[:6] = wrench
        if viewer is not None:
            # 被动 viewer 只记录鼠标 MjvPerturb；物理在本循环里跑，所以每步在这里刷新
            d.xfrc_applied[:] = 0.0
            mujoco.mjv_applyPerturbPose(m, d, viewer.perturb, 0)
            mujoco.mjv_applyPerturbForce(m, d, viewer.perturb)

        # ------------------------------------------------------ 采样级（每物理步）
        feedback = sampler.update(distance=sim.distance)
        if not feedback.decision.motion_enabled:
            solution = None

        # ---------------------------- 控制级 A：MPC（自己的节拍，与 WBC 异步）
        if sim.timing.mpc_due(step):
            solution, sim._command = mpc_stage(
                control=sim.control, feedback_accum=sim.feedback, time_s=float(d.time),
                feedback=feedback, command=command,
                schedule_leg_length=sim._schedule_mpc_leg_length,
                timings=sim.timings, counters=sim.counters)

        # ---- 离地档（用户 2026-10-07 要求）：离地检测生效后脱离控制器 ----
        # 判据用算法层估计器的结论（feedback.decision.contact），两轮**全部**判为离地才进入；
        # 进入后 WBC 不求解、也不把它的力矩写进执行器（输出置零），直到重新检测到接触。
        # 依据：WBC 的硬约束建立在"轮子接地"这一假设上，离地后该假设失效，继续输出没有物理意义。
        airborne = feedback is not None and not any(feedback.decision.contact)
        if airborne and not airborne_active:
            airborne_since = float(d.time)
            _LOGGER.info("离地检测生效：脱离控制器（t=%.3f s）", airborne_since)
        elif airborne_active and not airborne:
            _LOGGER.info("重新检测到接触：恢复控制器（脱离持续 %.3f s）", float(d.time) - airborne_since)
        airborne_active = airborne
        # ---------------------------- 控制级 B：WBC（自己的节拍，与 MPC 异步）
        if sim.timing.wbc_due(step) and not airborne_active:
            outcome = wbc_stage(
                control=sim.control, tasks=sim.tasks, dynamics=sim.dyn, mpc=sim.mpc,
                config=sim.config, feedback=feedback, solution=solution,
                feedback_accum=sim.feedback, time_s=float(d.time), timing=sim.timing,
                timings=sim.timings, counters=sim.counters, exhausted=sim.exhausted,
                scaled=sim.scaled, normal_violation_time=normal_violation_time,
                infeasible_active=infeasible_active)
            torque = outcome.torque
            reason = outcome.reason
            normal_violation_time = outcome.normal_violation_time
            infeasible_active = outcome.infeasible_active
            normal_warned = outcome.normal_warned
            limit_warned = outcome.limit_warned
            if step % sim.config.record_stride == 0:
                sim.records.append(sim._record(
                    feedback.state, feedback.decision, outcome.sample, torque,
                    outcome.result, outcome.predicted_normal, feedback.rows))
            # 控制器检查告警**不停机**（用户 2026-10-07：不要因为保护定值终止仿真）。
            # 去抖：只在告警出现/变化的那一拍打日志，计数每一拍累加，便于事后统计占比。
            if reason and reason != wbc_warning:
                _LOGGER.warning("控制器检查告警（不停机）：%s", reason)
            wbc_warning = reason
            if reason:
                key = reason.split(":")[0]
                sim.counters[key] = sim.counters.get(key, 0) + 1

        # -------------------------------------------- 输出级（每物理步）+ 物理推进
        if airborne_active:
            torque = np.zeros_like(torque)      # 脱离期间不输出控制解（零力矩）
        write_actuator_output(d, torque)
        if frame_callback is not None and (frame_interval <= 0.0
                                           or float(d.time) - last_frame_time >= frame_interval - 1e-12):
            frame_callback(d)
            last_frame_time = float(d.time)
        delta, guard = advance_physics(model=m, data=d, dynamics=sim.dyn, config=sim.config)
        sim.distance += delta
        # 输出侧护栏同样降级为告警（关节限位 / 姿态 / 速度）：不再停机。
        # 唯一例外是状态非有限（NaN/Inf）：物理已经死了，继续积分没有任何意义，
        # 这一条仍然停机，并在日志里说明（如需一并降级请明确说）。
        if guard == "nonfinite_state":
            _LOGGER.error("状态出现 NaN/Inf：物理已失效，停机（此条未降级）")
            sim.stop_reason = guard
            break
        if guard and guard != guard_warning:
            _LOGGER.warning("输出侧护栏告警（不停机）：%s", guard)
        guard_warning = guard
        if guard:
            key = guard.split(":")[0]
            sim.counters[key] = sim.counters.get(key, 0) + 1
        if viewer is not None:
            # 被动 viewer 刷新 + 实时限速；关窗口即停。
            # 2026-10-05：sync() 不再每物理步调用。实测每步 sync+渲染约 8.2 ms（物理步长 1 ms），
            # 会使 viewer 比实时慢 9.16 倍——用户看到的“收敛慢/效果差”有相当一部分是显示器上的
            # 时间被拉长 9 倍造成的假象（headless 同配置为 0.95 倍）。现在按渲染帧率（约 60 Hz
            # 仿真时间）刷新，物理与控制在 1 kHz 照常推进。渲染负担主因：base_link 单网格 91441 顶点。
            if float(d.time) - sim._viewer_sync_time >= VIEWER_SYNC_DT:
                viewer.sync()
                sim._viewer_sync_time = float(d.time)
            remaining = m.opt.timestep - (time.perf_counter() - step_wall_start)
            if remaining > 0.0:
                time.sleep(remaining)
            if not viewer.is_running():
                sim.stop_reason = "viewer_closed"
                break
        step += 1
