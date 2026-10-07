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

# viewer 刷新周期（仿真时间），约 60 fps。见本文件循环尾部 viewer.sync() 处的说明：
# 每物理步都 sync 会让 viewer 比实时慢 9 倍（渲染 91441 顶点的 base_link 等）。
VIEWER_SYNC_DT = 1.0 / 60.0


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
        model=m, data=d, dynamics=sim.dyn, contact_estimator=sim.contact_estimator,
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

        # ---------------------------- 控制级 B：WBC（自己的节拍，与 MPC 异步）
        if sim.timing.wbc_due(step):
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
            if step % sim.config.record_stride == 0:
                sim.records.append(sim._record(
                    feedback.state, feedback.decision, outcome.sample, torque,
                    outcome.result, outcome.predicted_normal, feedback.rows))
            if reason:
                sim.stop_reason = reason
                break

        # -------------------------------------------- 输出级（每物理步）+ 物理推进
        write_actuator_output(d, torque)
        if frame_callback is not None and (frame_interval <= 0.0
                                           or float(d.time) - last_frame_time >= frame_interval - 1e-12):
            frame_callback(d)
            last_frame_time = float(d.time)
        delta, guard = advance_physics(model=m, data=d, dynamics=sim.dyn, config=sim.config)
        sim.distance += delta
        if guard:
            sim.stop_reason = guard
            break
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
