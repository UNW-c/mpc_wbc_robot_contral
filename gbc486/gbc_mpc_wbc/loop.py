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
# 落地窗口（原 ANCHOR 锤定帧数，2026-10-17 用户确认保留 20 帧）。来源：用户指定 20 帧；
# 作用是避掉 t=1.893 那个单帧 mode 抖动。现在只做纯帧计时，不再有任何锚定语义。
LANDING_WINDOW_FRAMES = 20
_LOGGER = get_logger(__name__)


def run_loop(sim, command, duration: float, disturbances=(), viewer=None,
             frame_callback=None, frame_interval: float = 0.0, trace_sink=None,
             explicit_jump_reference: bool = False,
             compression_only: bool = False,
             freeze_jump_reference: bool = False,
             vmc_feedforward: bool = False,
             leg_length_compression: bool = False,
             leg_length_thrust: bool = False,
             thrust_only: bool = False,
             flight_zero_torque: bool = False,
             flight_leg_retract: bool = False,
             flight_leg_extend: bool = False,
             mpc_force_feedforward: bool = False,
             reset_key=None) -> None:
    """积分情景；传入 viewer 时按实时步长同步（与 circleg ``run`` 相同）。

    ``sim`` 是 ``Gbc486MpcWbcSimulation``：本函数只读它的装配好的零件
    （``dyn`` / ``control`` / ``tasks`` / ``mpc`` / ``timing`` / ``config`` ...），
    并把运行统计写回（``records`` / ``counters`` / ``distance`` / ``stop_reason``）。
    """
    m, d = sim.model, sim.data
    from .sampling import FeedbackSampler
    from .control import (mpc_stage, wbc_stage, flight_leg_retract_torque,
                          flight_leg_profile_torque, landing_buffer_torque,
                          dual_mass_flight_torque)
    from .jump import JumpObservation, LIFTOFF_GAP
    from Algorithm.contact_state import JumpPhaseHint
    from Algorithm.mpc.trajectory import MotionSample
    from . import robot as R
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
    jump_command = None           # 跳跃级：最近一拍跳跃指令（飞行段用于"力矩保持"）
    jump_reference = None         # Diagnostic explicit fixed WBC reference
    jump_holds = None             # 新部署：跳跃开始时捕获一次的姿态/腿角冻结参考
    previous_phase = "IDLE"     # Diagnostic-only state-flow audit
    last_thrust_applied = False
    thrust_applied_last = False
    previous_com_xz = None
    previous_com_time = None
    flight_leg_ref = None        # FL4：飞行剖面状态（跨帧保持）
    flight_dual_ref = None       # S1：双质量块飞行计划状态（跨帧保持）
    momentum_state = None        # 10.48 B3：THRUST 角动量任务的积分/记录状态（每跳重建）
    two_mass_state = None        # 10.54 C1：THRUST 两块计划的时钟与计划对象（每跳重建）
    landing_state = None         # 10.68：LANDING 两块计划的时钟与计划对象（每跳重建）
    s9_plan_end = False          # 10.58 D4：计划时钟到点（上一拍算，本拍进观测）
    s9_plan_verified = False     # 10.58 D4：到点核验结果
    jump_landed = False          # LAND2：这一跳已落地的锁存（LANDING/RECOVER 置位，COMPRESS 清零）
    landed_frames = 0            # 落地后 ground 模式连续帧数（只做 landing_window 的帧计时）
    landing_reanchor_pending = False   # FL9：本跳是否已离地、等待落地重锚
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
        step_time = float(d.time)
        # PERF（第 22 节）：缓存生命周期 = 一个物理步；这里清空，保证不跨步复用。
        sim.dyn.begin_step()
        # 第 27 节：跳跃激活期间斜坡会被冻结（motion_enabled=False 不推进），
        # 落地后的软重锚必须改用实测速度重新播种，不能用冻结前的旧速度。
        if jump_command is not None and jump_command.active:
            sim.mpc.planner.mark_reference_frozen()
        # 手动复位（第 24 节）：view 里按 R。实体与控制栈由 sim.reset_run 负责，
        # 本函数自己持有的跳跃/帧计时状态在这里一并清掉（它们不属于 sim）。
        if reset_key is not None and reset_key():
            _LOGGER.info("用户按 R 手动复位：t=%.3f s", float(d.time))
            sim.reset_run(viewer)
            jump_command = None
            jump_reference = None
            jump_holds = None
            previous_phase = "IDLE"
            last_thrust_applied = thrust_applied_last = False
            previous_com_xz = previous_com_time = None
            flight_leg_ref = None
            flight_dual_ref = None
            jump_landed = False
            landed_frames = 0
            landing_reanchor_pending = False
            feedback = None
            solution = None
            normal_violation_time = 0.0
            infeasible_active = False
            airborne_active = False
            airborne_since = 0.0
            torque = np.zeros(m.nu)
        mpc_called = False
        wbc_called = False
        # 第 33 节（LAND-REFRESH）：落地重锚那一拍强制 MPC/WBC 重算，只在本拍生效。
        landing_refresh = False
        outcome = None
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

        # -------------------------------------------------- 起跳触发（跳跃级）
        # 第 24 节：定时触发与外部请求（view 空格）两条来源并存。
        # - 定时：仍受 ``_jump_started`` 的"整轮一次"约束，语义不变；
        # - 请求：只多一个 OR 分支，相位判据在 ``sim.request_jump()`` 里（只在 IDLE 受理），
        #   这里**不加** ``phase == "IDLE"``（上一版就是在这里改坏的）。
        _jump_requested = bool(sim._jump_requested)
        _timed_due = not sim._jump_started and float(d.time) >= sim.config.jump_time
        if sim.jump is not None and (_timed_due or _jump_requested):
            sim._jump_requested = False          # 请求已被消费
            if _timed_due:
                sim._jump_started = True         # 只由定时来源推进该锁
            sim._precheck_jump()
            # 第 32 节（c）：起跳那一刻是否存在非零速度指令（用现成的 velocity_deadband 判据，
            # 不新增阈值）。与跳跃锁存同生命周期，下一跳 COMPRESS 时在 reset_landing_reanchor() 里清。
            _jump_cmd = command(float(d.time)) if callable(command) else command
            sim.mpc.planner.mark_jump_velocity_command(
                abs(float(getattr(_jump_cmd, "velocity", 0.0) or 0.0))
                > sim.mpc.planner.limits.velocity_deadband)
            jump_command = sim.jump.start(float(d.time))
            flight_leg_ref = None                # 每跳复位飞行腿剖面状态
            flight_dual_ref = None               # 每跳复位双质量块飞行计划
            momentum_state = ({"integral": 0.0, "last_t": None}
                              if bool(sim.config.thrust_momentum_task) else None)
            two_mass_state = ({"plan": None, "t0": None}
                              if bool(sim.config.two_mass_takeoff) else None)
            landing_state = ({"plan": None, "t0": None}
                             if bool(sim.config.two_mass_takeoff) else None)
            # S6（部署文档 10.32 D1）：压缩段锁水平参考。
            # 轮水平位置参考锁在进入压缩帧的实测轮触地点，质心水平参考也放到该点正上方
            # （即 d_n 目标 = 0），飞行/落地段的参考不在此改动。只动参考，不动增益。
            if bool(sim.config.jump_lock_horizontal):
                from . import robot as _lock_robot
                _points = np.asarray(_lock_robot.wheel_contact_points(m, d), dtype=float)
                _contact_x = float(np.mean(_points[:, 0]))
                _base_x = float(d.qpos[0])
                _target = float(sim.distance) + (_contact_x - _base_x)
                sim.mpc.planner.soft_reanchor(
                    _target, float(sim.dyn.attitude()[2]), reset_position=True)
                _LOGGER.info(
                    "S6 水平参考锁定：轮触地 x=%.4f m（相对参考坐标 %.4f），"
                    "机体质心偏置 d_n=%.1f mm，参考位置 %.4f -> %.4f",
                    _contact_x, _target - float(sim.distance), 1.0e3 * (_base_x - _contact_x),
                    float(sim.distance), _target)

        # ------------------------------------------------------ 采样级（每物理步）
        if jump_command is None:
            phase_hint = JumpPhaseHint.NONE
        else:
            phase_hint = {
                "THRUST": JumpPhaseHint.THRUST,
                "FLIGHT": JumpPhaseHint.FLIGHT,
                "LANDING": JumpPhaseHint.LANDING,
            }.get(jump_command.phase.value, JumpPhaseHint.NONE)
        feedback = sampler.update(distance=sim.distance, phase_hint=phase_hint)
        if not feedback.decision.motion_enabled:
            solution = None
        # FL9（2026-10-09）：落地重锚定修正。只对“跳过一次后落地”的重锚生效（跳过启动首帧
        # 的 GROUND 脉冲），且每跳只触发一次。重锚 MPC planner + GroundContactEstimator +
        # ControlSupervisor 的内部状态（BodyStateKalman 不在本仿真路径）。
        if jump_command is not None and jump_command.phase.value == "FLIGHT":
            landing_reanchor_pending = True
        if (flight_leg_extend and landing_reanchor_pending
                and feedback.decision.reset_reference):
            from dataclasses import replace as _replace
            land_state = _replace(feedback.state, position=sim.distance)
            sim.mpc.reset(land_state)
            sim.contact_estimator.reset()
            sim.supervisor.reset()
            landing_reanchor_pending = False
            # 第 33 节（a）：丢弃落地前那次 MPC 解（WBC 不再采落地前轨迹），并在本拍强制重算。
            solution = None
            landing_refresh = True
            _LOGGER.info("落地重锚定：reset_reference=True @ t=%.3f s，mpc/estimator/supervisor 已重锚",
                         float(d.time))
        # ---------------- 跳跃级：每物理步推进一次（纯计算，只用测量量）----------------
        previous_jump_phase = None if jump_command is None else jump_command.phase.value
        jump_wheel_gap = None
        if sim.jump is not None:
            leg = sim.dyn.leg_state()
            leg_rows = feedback.rows.leg[0]
            jump_wheel_gap = float(np.min(R.wheel_contact_points(m, d)[:, 2]))
            # 10.62 E4：S9 计划终点核验——**当拍**算，判据用两轮法向力（N<=0），
            # gap/接触估计器/腿长只记录。上一轮的"上一拍值"实现已删除。
            s9_plan_end = False
            s9_plan_verified = False
            if two_mass_state is not None and two_mass_state.get("plan") is not None:
                _s9 = two_mass_state["plan"]
                s9_plan_end = bool(float(d.time) - float(two_mass_state["t0"]) >= float(_s9.T))
                _n_now = np.asarray(feedback.forces[:, 2], dtype=float)
                s9_plan_verified = bool(np.all(_n_now <= 0.0))
                if s9_plan_end and previous_jump_phase == "THRUST":   # 10.70：只在 THRUST 末拍记一次
                    _LOGGER.info(
                        "S9 计划终点（当拍）：elapsed=%.4f T=%.4f N=[%.1f, %.1f] gap=%.4f mm L=%.4f 核验=%s",
                        float(d.time) - float(two_mass_state["t0"]), float(_s9.T),
                        float(_n_now[0]), float(_n_now[-1]),
                        1.0e3 * float(jump_wheel_gap), float(leg[0]), s9_plan_verified)
            # 10.68：上一拍已在 LANDING 时，首拍用实测状态生成落地计划；之后按计划时钟判到点
            landing_plan_end = False
            if (landing_state is not None and jump_command is not None
                    and jump_command.phase.value == "LANDING"):
                if landing_state.get("plan") is None:
                    from . import two_mass as _lp
                    _row, _bias, _zc_abs, _ = sim.dyn.com_vertical_row()
                    _low = float(np.min(R.wheel_contact_points(m, d)[:, 2]))
                    landing_state["plan"] = _lp.LandingPlan(
                        sim.two_mass_tables, float(_zc_abs - _low), float(_row @ d.qvel),
                        float(sim.config.landing_length_end), float(sim.two_mass_tables["M"]))
                    landing_state["t0"] = float(d.time)
                    _LOGGER.info("S10 落地计划：T=%.4f s、N 峰 %.1f N、zc0=%.4f vcz0=%+.3f -> L_end %.3f",
                                 landing_state["plan"].T, landing_state["plan"].n_peak,
                                 landing_state["plan"].zc0, landing_state["plan"].vcz0,
                                 float(sim.config.landing_length_end))
                landing_plan_end = bool(float(d.time) - float(landing_state["t0"])
                                        >= float(landing_state["plan"].T))
            jump_command = sim.jump.update(JumpObservation(
                time=float(d.time),
                leg_length=0.5 * (float(leg[0]) + float(leg[2])),
                leg_length_rate=0.5 * (float(leg_rows[0] @ d.qvel) + float(leg_rows[2] @ d.qvel)),
                leg_length_R=float(leg[0]), leg_length_rate_R=float(leg_rows[0] @ d.qvel),
                leg_length_L=float(leg[2]), leg_length_rate_L=float(leg_rows[2] @ d.qvel),
                wheel_gap=jump_wheel_gap,
                geometry_contact=tuple(bool(v) for v in sim.dyn.contact_geometry()[0]),
                contact=feedback.decision.contact, pitch=float(feedback.state.pitch),
                pitch_rate=float(feedback.state.pitch_rate),
                thrust_applied=bool(thrust_applied_last),
                plan_end=bool(s9_plan_end), plan_verified=bool(s9_plan_verified),
                landing_plan_end=bool(landing_plan_end)))
            if (explicit_jump_reference and jump_command is not None
                    and jump_command.active and jump_reference is None):
                state = feedback.state.vector(sim.reduced.state_dim)
                state[1] = 0.0
                state[2] = 0.0
                state[3] = 0.0
                state[5] = 0.0
                if state.size > 7:
                    state[6] = 0.0
                    state[7] = 0.0
                jump_reference = MotionSample(
                    state=np.asarray(state, dtype=float),
                    acceleration=np.zeros(sim.reduced.state_dim, dtype=float),
                    force=np.zeros(sim.reduced.input_dim, dtype=float),
                    timestamp=float(d.time),
                )

        # 新部署：B组实验用“跳跃开始时捕获一次”的姿态/腿角冻结参考。
        if (freeze_jump_reference and jump_command is not None and jump_command.active
                and jump_holds is None):
            roll_now, pitch_now, _yaw_now = sim.dyn.attitude()
            leg_now = sim.dyn.leg_state()
            jump_holds = dict(
                pitch=float(pitch_now), roll=float(roll_now),
                angle_R=float(sim.pose.angle - leg_now[1]),
                angle_L=float(sim.pose.angle - leg_now[3]))


        # WP5（compression_only）：压缩完成或立即停止条件都在写 THRUST / 推进物理前生效。
        if compression_only and jump_command is not None:
            if jump_command.phase.value == "THRUST":
                # 在把 THRUST 写入 data.ctrl 之前停止；本轮不得执行释放/飞行/落地。
                sim.stop_reason = ("compression_complete" if previous_jump_phase == "COMPRESS_BRAKE"
                                   else "compression_thrust_without_brake")
                _LOGGER.info("compression_only：压缩完成（%s -> THRUST），在写入 THRUST 前停止本轮"
                             "（t=%.3f s）", previous_jump_phase, float(d.time))
                break
            if jump_command.phase.value == "ABORT":
                sim.stop_reason = f"compression_abort:{jump_command.abort_reason}"
                _LOGGER.warning("compression_only：进入 ABORT，立即停止本轮（原因 %s，t=%.3f s）",
                                jump_command.abort_reason, float(d.time))
                break
            if jump_command.active:
                geometry_flags, _counts, _other = sim.dyn.contact_geometry()
                normals = np.asarray(feedback.forces[:, 2], dtype=float)
                if not all(bool(v) for v in geometry_flags) or not bool(np.all(normals > 0.0)):
                    sim.stop_reason = "compression_contact_lost"
                    _LOGGER.warning("compression_only：几何接触或实际法向力丢失，立即停止本轮"
                                    "（geometry=%s normals=%s t=%.3f s）",
                                    tuple(bool(v) for v in geometry_flags), normals.tolist(),
                                    float(d.time))
                    break

        # TR1（thrust_only）：相位将进入 FLIGHT 时停止，不执行 FLIGHT/LANDING 控制。
        if thrust_only and jump_command is not None and jump_command.phase.value == "FLIGHT":
            sim.stop_reason = "liftoff_confirmed"
            _LOGGER.info("thrust_only：确认离地（FLIGHT），在写入 FLIGHT 控制前停止本轮（t=%.3f s）",
                         float(d.time))
            break

        # FL16B（2026-10-10）：释放段前馈斜坡计时（与 FL17B 共用同一相位计时）。
        # ---------------------------- 控制级 A：MPC（自己的节拍，与 WBC 异步）
        if landing_refresh or sim.timing.mpc_due(step):
            mpc_called = True
            _mpc_feedback = feedback
            if bool(sim.config.mpc_nominal_normal):
                # 10.70 A：normal_forces=None -> WheelMPC.wheel_force_limit 取 nominal_normal_force
                from dataclasses import replace as _replace_a
                _mpc_feedback = _replace_a(feedback, state=_replace_a(feedback.state, normal_forces=None))
            solution, sim._command = mpc_stage(
                control=sim.control, feedback_accum=sim.feedback, time_s=float(d.time),
                feedback=_mpc_feedback, command=command,
                timings=sim.timings, counters=sim.counters)

        # ---- 离地档（用户 2026-10-07 要求）：离地检测生效后脱离控制器 ----
        # 判据用算法层估计器的结论（feedback.decision.contact），两轮**全部**判为离地才进入；
        # 进入后 WBC 不求解、也不把它的力矩写进执行器（输出置零），直到重新检测到接触。
        # 依据：WBC 的硬约束建立在"轮子接地"这一假设上，离地后该假设失效，继续输出没有物理意义。
        # 跳跃是独立状态：跳跃期由跳跃规划器（几何真离地）决定是否脱离，离地档不再插手。
        # 依据（用户 2026-10-07）：力的判定会在轮子仍压地时误报离地，此时断开控制器会让
        # 机体在最需要控姿的时刻失控（实测俯仰冲到 -19.6 度）。
        jump_active_now = jump_command is not None and jump_command.active
        # LAND2（2026-10-16，用户同意）：落地锁存。只看瞬时轮地间隙会把落地后任何一次
        # 反弹（gap>1mm）当成再次离地，导致飞行档/离地档重入。ABORT 不置位（中止后
        # 自由下落仍走离地档）。
        _landed_phase = None if jump_command is None else jump_command.phase.value
        if _landed_phase in ("LANDING", "RECOVER"):
            jump_landed = True
        elif _landed_phase == "COMPRESS":
            jump_landed = False          # 新一跳开始，解锁
            landed_frames = 0            # 新一跳开始，落地帧计时清零
            # 第 31 节：新一跳开始 -> 允许下一次落地再做一次播种重锚（每次落地恰好一次）。
            sim.mpc.planner.reset_landing_reanchor()
        # 落地后 ground 模式的连续帧数：只用于 landing_window 的帧计时（NOANCHOR，2026-10-17）。
        landed_frames = (landed_frames + 1
                         if jump_landed and feedback.decision.mode == "ground" else 0)
        # LAND4（2026-10-17，用户同意）：离地判据改为“估计器无接触”且“几何也无接触”双确认，
        # 避免“离地档清零力矩 -> 法向力 0 -> 估计器持续报无接触”的自锁。几何来源与 jump.py 一致。
        _geometry_contact = tuple(bool(v) for v in sim.dyn.contact_geometry()[0])
        airborne = (feedback is not None and not any(feedback.decision.contact)
                    and not any(_geometry_contact)
                    and not jump_active_now)
        if airborne and not airborne_active:
            airborne_since = float(d.time)
            _LOGGER.info("离地检测生效：脱离控制器（t=%.3f s）", airborne_since)
        elif airborne_active and not airborne:
            _LOGGER.info("重新检测到接触：恢复控制器（脱离持续 %.3f s）", float(d.time) - airborne_since)
        airborne_active = airborne
        # ---------------------------- 控制级 B：WBC（自己的节拍，与 MPC 异步）
        # WP3（2026-10-09）：FLIGHT 或无有效接触时不调用要求接触块的 WBC；
        # COMPRESS/THRUST 有接触时持续调用 WBC。disconnect 字段仅作状态记录，不再改变 WBC 调用。
        jump_phase_now = None if jump_command is None else jump_command.phase.value
        contact_available = bool(feedback is not None and any(feedback.decision.contact))
        # 腿角目标回到 _apply_zmp_reference 写入的 reference[:,6]/[:,7]（NOANCHOR 删去覆盖）。
        wbc_holds = jump_holds
        if ((landing_refresh or sim.timing.wbc_due(step)) and not airborne_active
                and jump_phase_now != "FLIGHT" and contact_available):
            wbc_called = True
            outcome = wbc_stage(
                control=sim.control, tasks=sim.tasks, dynamics=sim.dyn, mpc=sim.mpc,
                config=sim.config, feedback=feedback, solution=solution,
                feedback_accum=sim.feedback, time_s=float(d.time), timing=sim.timing,
                timings=sim.timings, counters=sim.counters, exhausted=sim.exhausted,
                scaled=sim.scaled, normal_violation_time=normal_violation_time,
                infeasible_active=infeasible_active, jump=jump_command,
                desired_com_acceleration_z=(None if jump_command is None
                                            else jump_command.desired_com_acceleration_z),
                jump_holds=wbc_holds,
                vmc_feedforward=vmc_feedforward,
                leg_length_compression=leg_length_compression,
                leg_length_thrust=leg_length_thrust,
                jump_config=(None if sim.jump is None else sim.jump.config),
                sample_override=(jump_reference if explicit_jump_reference
                                 and jump_command is not None and jump_command.active
                                 else None),
                landing_window=(jump_landed and landed_frames < LANDING_WINDOW_FRAMES),
                thrust_momentum=bool(sim.config.thrust_momentum_task),
                k_momentum=float(sim.config.thrust_momentum_kh),
                k_momentum_integral=float(sim.config.thrust_momentum_khi),
                momentum_state=momentum_state,
                two_mass_state=two_mass_state,
                two_mass_tables=getattr(sim, "two_mass_tables", None),
                landing_state=landing_state,
                mpc_force_feedforward=mpc_force_feedforward,
                wheel_radius=float(sim.reduced.wheel_radius))
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
        # FL1（2026-10-09）：flight_zero_torque 打开时，相位为 FLIGHT 或“真几何离地”
        # （min wheel_gap > LIFTOFF_GAP）则输出显式零力矩，不再 zero_order_hold 保持地面力矩。
        # 默认 False 时行为与当前完全一致。
        geometrically_airborne = bool(jump_wheel_gap is not None and jump_wheel_gap > LIFTOFF_GAP
                                      and not jump_landed)
        flight_zero = bool(flight_zero_torque and jump_command is not None
                           and (jump_command.phase.value == "FLIGHT" or geometrically_airborne))
        jump_phase_value = None if jump_command is None else jump_command.phase.value
        # FL3（2026-10-09）：飞行腿剖面（上升收腿/下落伸腿）与落地缓冲。
        leg_now = None
        leg_rate_now = None
        landing_buffer = False
        if sim.jump is not None and flight_leg_extend and (geometrically_airborne
                                                           or jump_phase_value in ("LANDING", "FLIGHT")):
            leg_now = sim.dyn.leg_state()
            leg_rate_now = np.asarray(feedback.rows.leg[0] @ d.qvel, dtype=float)
            compression_target = (sim.jump.config.leg_length_nominal
                                  - (sim.jump.config.leg_length_nominal
                                     - sim.jump.config.leg_length_min) * 0.5)
            landing_buffer = bool(jump_phase_value == "LANDING"
                                  and not bool(sim.config.two_mass_takeoff)
                                  and float(d.qvel[2]) < 0.0
                                  and min(float(leg_now[0]), float(leg_now[2])) > compression_target)
        # FL5（2026-10-09）：飞行剖面门控改为“FLIGHT 相位或几何离地”（FL4 因只靠 gap>1mm，
        # 推离完成时 gap≈0.03 mm 剖面不启动，腿被动伸到上限）。
        flight_profile = bool(flight_leg_extend and sim.jump is not None
                              and (jump_phase_value == "FLIGHT" or geometrically_airborne))
        # FL2（2026-10-09）：仅收腿（无伸腿开关时），几何离地时用收腿力矩代替零力矩。
        flight_retract = bool(flight_leg_retract and not flight_leg_extend
                              and geometrically_airborne and sim.jump is not None)
        if landing_buffer:
            torque = landing_buffer_torque(sim.dyn, leg_now, leg_rate_now,
                                           sim.jump.config.leg_length_min,
                                           sim.jump.config.leg_length_max,
                                           sim.jump.config.leg_length_nominal,
                                           pitch=float(feedback.state.pitch))
        elif flight_profile and bool(sim.config.flight_dual_mass):
            # S1：飞行段单轨迹（部署文档 v6 第 3 节）。开关关时不走这里，行为不变。
            torque, flight_dual_ref = dual_mass_flight_torque(
                sim.dyn, leg_now, leg_rate_now, float(d.qvel[2]),
                dt=float(m.opt.timestep), pitch=float(feedback.state.pitch),
                plan=flight_dual_ref,
                leg_length_nominal=sim.jump.config.leg_length_nominal,
                hold_momentum=bool(sim.config.flight_hold_momentum),
                two_mass_table=getattr(sim, "two_mass_tables", None),
                landing_length=(float(sim.config.landing_length_td)
                                if bool(sim.config.two_mass_takeoff) else None))
        elif flight_profile:
            torque, flight_leg_ref = flight_leg_profile_torque(
                sim.dyn, leg_now, leg_rate_now, float(d.qvel[2]),
                sim.jump.config.leg_length_min, sim.jump.config.leg_length_max,
                sim.jump.config.leg_length_nominal,
                retract_state=flight_leg_ref, dt=float(m.opt.timestep),
                pitch=float(feedback.state.pitch),
                preswing=float(sim.config.flight_leg_preswing))
        elif flight_retract:
            leg_now = sim.dyn.leg_state()
            leg_rate_now = np.asarray(feedback.rows.leg[0] @ d.qvel, dtype=float)
            torque = flight_leg_retract_torque(sim.dyn, leg_now, leg_rate_now,
                                               sim.jump.config.leg_length_min,
                                               sim.jump.config.leg_length_max)
        elif flight_zero:
            torque = np.zeros_like(torque)
        elif airborne_active:
            # 飞行段（用户 D4）：跳跃规划器接管时**保持上一拍力矩**（零阶保持，力矩连续）；
            # 非跳跃的意外离地仍按原逻辑置零。
            if jump_command is not None and jump_command.hold_torque:
                pass
            else:
                torque = np.zeros_like(torque)
        write_actuator_output(d, torque)
        if landing_buffer:
            output_source = "landing_buffer"
        elif flight_profile:
            output_source = "flight_leg_profile"
        elif flight_retract:
            output_source = "flight_leg_retract"
        elif flight_zero:
            output_source = "flight_zero_torque"
        elif airborne_active:
            output_source = "jump_hold" if jump_command is not None and jump_command.hold_torque else "airborne_zero"
        else:
            output_source = "wbc_update" if wbc_called else "zero_order_hold"
        # WP3：释放是否执行改由“WBC 已求解且总力矩已写入”确认，不再用 tau_jump 是否非零。
        thrust_applied_now = bool(
            jump_command is not None and jump_command.phase.value == "THRUST"
            and wbc_called and output_source == "wbc_update" and outcome is not None
            and outcome.result is not None)
        thrust_applied_last = thrust_applied_now
        if trace_sink is not None:
            from . import robot as _trace_robot
            wheel_points = _trace_robot.wheel_contact_points(m, d)
            geometry_flags, geometry_counts, geometry_other = sim.dyn.contact_geometry()
            jump_payload = None if jump_command is None else {
                "phase": jump_command.phase.value,
                "active": bool(jump_command.active),
                "disconnect": bool(jump_command.disconnect),
                "hold_torque": bool(jump_command.hold_torque),
                "leg_force": float(jump_command.leg_force),
                "leg_length_target": (None if jump_command.leg_length_target is None
                                       else float(jump_command.leg_length_target)),
                "wheel_torque": float(jump_command.wheel_torque),
                "desired_com_acceleration_z": (None if jump_command.desired_com_acceleration_z is None
                                               else float(jump_command.desired_com_acceleration_z)),
                # 10.48 B3：角动量任务的运行记录（H、H 的目标、积分状态、单拍耗时）
                "momentum": (None if momentum_state is None else {
                    "h": float(momentum_state.get("h", 0.0)),
                    "h_des": float(momentum_state.get("h_des", 0.0)),
                    "integral": float(momentum_state.get("integral", 0.0)),
                    "row_ms": float(momentum_state.get("row_ms", 0.0))}),
                "flight_leg_ref": (None if flight_leg_ref is None
                                   else [str(flight_leg_ref[0])]
                                        + [float(v) for v in flight_leg_ref[1:]]),
            }
            sample = None if outcome is None else outcome.sample
            sample_raw = None if outcome is None else outcome.sample_raw
            result = None if outcome is None else outcome.result
            base_id = int(m.body("base").id)
            com = np.asarray(d.subtree_com[base_id], dtype=float).copy()
            com_xz = com[[0, 2]]
            if previous_com_xz is None or previous_com_time is None or step_time <= previous_com_time:
                com_velocity_xz = np.zeros(2, dtype=float)
            else:
                com_velocity_xz = (com_xz - previous_com_xz) / (step_time - previous_com_time)
            previous_com_xz = com_xz.copy()
            previous_com_time = step_time
            hub_xz = np.asarray(d.xpos[list(sim.dyn.wheel_bodies)], dtype=float)[:, [0, 2]].copy()
            tau_jump_current = (None if outcome is None
                                else np.asarray(outcome.tau_jump, dtype=float).copy())
            current_phase = ("IDLE" if jump_payload is None
                             else str(jump_payload["phase"]))
            trace_leg = np.asarray(sim.dyn.leg_state(), dtype=float).copy()
            trace_leg_rate = np.asarray(feedback.rows.leg[0] @ d.qvel, dtype=float).copy()
            # WP7：触发条件与是否被临时隔离都取本拍 JumpPlanner 的判定，
            # 不再用 compression_only 充当隔离标志。侧别与 robot.LEGS=(RIGHT, LEFT) 一致。
            planner = sim.jump
            compression_direction_lost_R = bool(planner is not None and planner.direction_lost_R)
            compression_direction_lost_L = bool(planner is not None and planner.direction_lost_L)
            compression_direction_lost_isolated_R = bool(
                planner is not None and planner.direction_lost_isolated_R)
            compression_direction_lost_isolated_L = bool(
                planner is not None and planner.direction_lost_isolated_L)
            compression_direction_lost_isolation_active = bool(
                planner is not None and planner.direction_lost_isolation_active)
            compression_direction_lost_isolated = bool(
                compression_direction_lost_isolated_R or compression_direction_lost_isolated_L)
            predicted_vec = (np.zeros(0, dtype=float) if outcome is None
                             else np.asarray(outcome.predicted_normal, dtype=float))
            predicted_normal_R = None
            predicted_normal_L = None
            contact_flags = tuple(bool(v) for v in feedback.decision.contact)
            contact_count = int(sum(contact_flags))
            if predicted_vec.size >= 2 * contact_count and contact_count > 0:
                wheel_forces = sim.tasks.contact_forces(predicted_vec, contact_flags)
                for row, side in enumerate(i for i, flag in enumerate(contact_flags) if flag):
                    normal = float(wheel_forces[row, 1])
                    if side == 0:
                        predicted_normal_R = normal
                    elif side == 1:
                        predicted_normal_L = normal
            thrust_command_built = bool(
                jump_payload is not None
                and current_phase == "THRUST"
                and float(jump_payload["leg_force"]) > 0.0)
            thrust_applied = bool(thrust_applied_now)
            trace_previous_phase = previous_phase
            trace_transition_reason = None
            phase_transition = None
            if current_phase != previous_phase:
                allowed_predecessors = {
                    "IDLE": {"RECOVER", "ABORT", "LANDING"},   # 10.70：S10 落地计划终点直接交还（10.68 B3.2）
                    "COMPRESS": {"IDLE"},
                    "COMPRESS_BRAKE": {"COMPRESS"},
                    "THRUST": {"COMPRESS_BRAKE"},
                    "FLIGHT": {"THRUST"},
                    "LANDING": {"FLIGHT"},
                    "RECOVER": {"LANDING"},
                    "ABORT": {"IDLE", "COMPRESS", "COMPRESS_BRAKE", "THRUST", "FLIGHT", "LANDING", "RECOVER", "ABORT"},
                }
                legal_predecessor = previous_phase in allowed_predecessors.get(current_phase, set())
                geometry_contact_candidate = bool(any(geometry_flags))
                estimator_contact_confirmed = bool(all(feedback.contact))
                normal_force_positive = bool(np.all(np.asarray(feedback.forces[:, 2], dtype=float) > 0.0))
                reasons = []
                assertion_passed = legal_predecessor
                if not legal_predecessor:
                    reasons.append("previous_phase_not_allowed")
                if previous_phase == "THRUST" and current_phase == "FLIGHT" and not last_thrust_applied:
                    assertion_passed = False
                    reasons.append("thrust_not_applied")
                if previous_phase == "FLIGHT" and current_phase == "LANDING":
                    # FL6（2026-10-09）：与 jump.py 纯几何判据一致，不再要求估计器接触/法向力。
                    if not bool(all(geometry_flags)):
                        assertion_passed = False
                        reasons.append("geometry_contact_not_confirmed")
                    gap_min = float(np.min(np.asarray(wheel_points, dtype=float)[:, 2]))
                    if not (gap_min <= 0.0):
                        assertion_passed = False
                        reasons.append("wheel_gap_not_closed")
                if previous_phase == "LANDING" and current_phase == "RECOVER" and not estimator_contact_confirmed:
                    assertion_passed = False
                    reasons.append("dual_contact_not_confirmed")
                if previous_phase == "RECOVER" and current_phase == "IDLE" and (not estimator_contact_confirmed or airborne_active):
                    assertion_passed = False
                    reasons.append("control_handoff_not_confirmed")
                phase_transition = {
                    "previous_phase": previous_phase,
                    "requested_phase": current_phase,
                    "legal_predecessor": bool(legal_predecessor),
                    "assertion_passed": bool(assertion_passed),
                    "blocking_reasons": tuple(reasons),
                    "prior_thrust_applied": bool(last_thrust_applied),
                    "thrust_command_built": bool(thrust_command_built),
                    "thrust_applied": bool(thrust_applied),
                    "geometry_contact_candidate": geometry_contact_candidate,
                    "estimator_contact_confirmed": estimator_contact_confirmed,
                    "normal_force_positive": normal_force_positive,
                }
                trace_transition_reason = None if planner is None else planner.transition_reason
                if current_phase == "FLIGHT":
                    gap_now = float(np.max(np.asarray(wheel_points, dtype=float)[:, 2]))
                    phase_transition["flight_by_geometry"] = bool(gap_now > LIFTOFF_GAP)
                    phase_transition["flight_max_wheel_gap"] = gap_now
                    _LOGGER.info("THRUST->FLIGHT 触发判据：%s（max wheel_gap=%.6f m，阈值 %.6f）",
                                 ("s9_plan_end_force" if bool(sim.config.two_mass_takeoff)
                                  else ("geometry" if gap_now > LIFTOFF_GAP else "release_end_fallback")),
                                 gap_now, LIFTOFF_GAP)
                previous_phase = current_phase
            last_thrust_applied = thrust_applied_now
            trace_sink({
                "step": int(step), "t": step_time, "t_after": float(d.time),
                "qpos": np.asarray(d.qpos, dtype=float).copy(),
                "qvel": np.asarray(d.qvel, dtype=float).copy(),
                "qacc": np.asarray(d.qacc, dtype=float).copy(),
                "ctrl": np.asarray(d.ctrl, dtype=float).copy(),
                "com_xz": com[[0, 2]],
                "distance": float(sim.distance),
                "com_vxz": com_velocity_xz.copy(),
                "wheel_hub_xz": hub_xz,
                "qfrc_applied": np.asarray(d.qfrc_applied[:6], dtype=float).copy(),
                "xfrc_applied": np.asarray(d.xfrc_applied, dtype=float).copy(),
                "jump": jump_payload,
                "phase_transition": phase_transition,
                "previous_phase": trace_previous_phase,
                "transition_reason": trace_transition_reason,
                "thrust_command_built": bool(thrust_command_built),
                "thrust_applied": bool(thrust_applied),
                "pitch": float(feedback.state.pitch),
                "pitch_rate": float(feedback.state.pitch_rate),
                # 偏航与 state 系速度需要进轨迹（判据与反馈表用实测值，不用 MPC 节点）。
                "yaw": float(feedback.state.yaw),
                "yaw_rate": float(feedback.state.yaw_rate),
                "state_velocity": float(feedback.state.velocity),
                "leg_state": np.asarray(sim.dyn.leg_state(), dtype=float).copy(),
                "leg_rate": np.asarray(feedback.rows.leg[0] @ d.qvel, dtype=float).copy(),
                "leg_length_R": float(trace_leg[0]),
                "leg_length_L": float(trace_leg[2]),
                "leg_length_rate_R": float(trace_leg_rate[0]),
                "leg_length_rate_L": float(trace_leg_rate[2]),
                "compression_direction_lost_R": compression_direction_lost_R,
                "compression_direction_lost_L": compression_direction_lost_L,
                "compression_direction_lost_isolated": compression_direction_lost_isolated,
                "compression_direction_lost_isolated_R": compression_direction_lost_isolated_R,
                "compression_direction_lost_isolated_L": compression_direction_lost_isolated_L,
                "compression_direction_lost_isolation_active": compression_direction_lost_isolation_active,
                "wheel_points": np.asarray(wheel_points, dtype=float).copy(),
                "wheel_gap": np.asarray(wheel_points[:, 2], dtype=float).copy(),
                "contact_geometry": tuple(bool(v) for v in geometry_flags),
                "contact_geometry_counts": tuple(int(v) for v in geometry_counts),
                "contact_geometry_other": int(geometry_other),
                "normal_forces": np.asarray(feedback.forces[:, 2], dtype=float).copy(),
                "tangent_forces": np.asarray(feedback.forces[:, 0], dtype=float).copy(),
                "contact_estimator": tuple(bool(v) for v in feedback.contact),
                "decision": {
                    "mode": feedback.decision.mode.value,
                    "contact": tuple(bool(v) for v in feedback.decision.contact),
                    "motion_enabled": bool(feedback.decision.motion_enabled),
                    "balance_enabled": bool(feedback.decision.balance_enabled),
                    "reset_reference": bool(feedback.decision.reset_reference),
                    "context": int(feedback.decision.context),
                },
                "airborne_active": bool(airborne_active),
                "geometrically_airborne": bool(geometrically_airborne),
                "flight_zero_torque": bool(flight_zero_torque),
                "flight_leg_retract": bool(flight_leg_retract),
                "flight_leg_extend": bool(flight_leg_extend),
                "mpc_called": bool(mpc_called),
                "mpc_due": bool(sim.timing.mpc_due(step)),
                "landing_refresh": bool(landing_refresh),
                "wbc_due": bool(sim.timing.wbc_due(step)),
                "mpc_solution": None if solution is None else {
                    "fallback": bool(solution.fallback),
                    "relaxed": bool(solution.qp.relaxed),
                    "reference": np.asarray(solution.reference.states(solution.trajectory.state_dim)[0], dtype=float).copy(),
                    "force": np.asarray(solution.force, dtype=float).copy(),
                },
                "sample_raw": None if sample_raw is None else {
                    "state": np.asarray(sample_raw.state, dtype=float).copy(),
                    "acceleration": np.asarray(sample_raw.acceleration, dtype=float).copy(),
                },
                "sample": None if sample is None else {
                    "state": np.asarray(sample.state, dtype=float).copy(),
                    "acceleration": np.asarray(sample.acceleration, dtype=float).copy(),
                },
                # 第 28.8 节判别实验：只读记录，不参与任何控制量计算。
                "planner_forward_pos": float(sim.mpc.planner._forward.position),  # noqa: SLF001
                "planner_forward_vel": float(sim.mpc.planner._forward.velocity),  # noqa: SLF001
                "planner_mode": sim.mpc.planner.mode.value,
                "ref_v": None if sample is None else float(sample.state[1]),
                "explicit_jump_reference_active": bool(explicit_jump_reference
                                                        and jump_command is not None
                                                        and jump_command.active
                                                        and jump_reference is not None),
                "explicit_jump_reference": (None if not (explicit_jump_reference
                                                          and jump_command is not None
                                                          and jump_command.active
                                                          and jump_reference is not None) else {
                    "state": np.asarray(jump_reference.state, dtype=float).copy(),
                    "acceleration": np.asarray(jump_reference.acceleration, dtype=float).copy(),
                    "timestamp": float(jump_reference.timestamp),
                }),
                "wbc_called": bool(wbc_called),
                "wbc_raw_torque": None if outcome is None else np.asarray(outcome.raw_torque, dtype=float).copy(),
                "tau_jump": None if outcome is None else np.asarray(outcome.tau_jump, dtype=float).copy(),
                "tau_ff": None if outcome is None else np.asarray(outcome.tau_ff, dtype=float).copy(),
                "vmc_feedforward": None if outcome is None else {
                    "applied": bool(outcome.feedforward_applied),
                    "force_per_leg": float(outcome.feedforward_force),
                    "a_z": (None if jump_command is None
                            else jump_command.desired_com_acceleration_z),
                    "tau_ff": np.asarray(outcome.tau_ff, dtype=float).copy(),
                },
                "wheel_selection_cleared": bool(outcome is not None
                                                and outcome.wheel_selection_cleared),
                "leg_length_compression": bool(leg_length_compression),
                "leg_length_thrust": bool(leg_length_thrust),
                "wbc_task_trace": None if outcome is None else outcome.task_trace,
                "wbc_result": None if result is None else {
                    "feasible": bool(result.feasible),
                    "fallback_used": bool(result.fallback_used),
                    "clip_applied": bool(result.clip_applied),
                    "dynamics_residual_norm": float(result.dynamics_residual_norm),
                    "qacc": np.asarray(result.qacc, dtype=float).copy(),
                    "scaling": {str(k): float(v) for k, v in result.scaling.items()},
                    "outcomes": tuple({
                        "name": item.name,
                        "residual_norm": float(item.residual_norm),
                        "scaling": float(result.scaling.get(item.name, item.activated)),
                        "exhausted": bool(item.exhausted),
                    } for item in result.hierarchy.outcomes),
                },
                "predicted_normal": predicted_vec.copy(),
                "predicted_normal_R": predicted_normal_R,
                "predicted_normal_L": predicted_normal_L,
                "wbc_output_written": bool(wbc_called),
                "output_source": output_source,
                "mpc_object_id": int(id(sim.mpc)),
                "mpc_model_A": np.asarray(sim.mpc.model.A, dtype=float).copy(),
                "mpc_model_B": np.asarray(sim.mpc.model.B, dtype=float).copy(),
            })
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
        if compression_only and guard:
            sim.stop_reason = f"compression_guard:{guard}"
            _LOGGER.warning("compression_only：输出侧护栏 %s，立即停止本轮", guard)
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
    # PERF（第 22 节）：缓存统计写进日志（22.4 要求"每步 rowset 重算次数 = 1"可核对）。
    stats = sim.dyn.cache_stats
    if stats["steps"]:
        _LOGGER.info("PERF 缓存统计：步数=%d | rowset 重算/步=%.3f（命中 %d）| "
                     "joint_limit 重算/步=%.3f（命中 %d）| lengths 重算/步=%.3f（命中 %d）",
                     stats["steps"], stats["rowset_recompute"] / stats["steps"], stats["rowset_hit"],
                     stats["joint_limit_recompute"] / stats["steps"], stats["joint_limit_hit"],
                     stats["lengths_recompute"] / stats["steps"], stats["lengths_hit"])
