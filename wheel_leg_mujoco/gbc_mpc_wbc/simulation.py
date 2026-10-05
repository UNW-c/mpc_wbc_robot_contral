"""GBC486 MPC + WBC 闭环。

**结构与 circleg ``APP/mujoco_mpc_wbc.py::MuJoCoMpcWbcSimulation.run`` 一致**：
``ControlTiming`` 分频 → ``ControlSupervisor`` 判模式 → ``SupervisedControl.solve_mpc``
→ ``solve_wbc`` → 写 ``data.ctrl``。跳跃、hip_input 专用分支、viewer 相关部分本阶段不用。

只做"换模型必须改"的部分：模型（``Gbc486Dynamics`` + 两个降阶模型 Builder）、
接触块里的闭链行、以及本机器人的 MPC 物理量覆盖（部署方案第 7 节）。

``model_variant``：``held_leg`` = 6 状态（对应 circleg held_hip）；``leg_input`` = 8 状态
（对应 circleg hip_input，共模腿摆角为状态、共模腿力矩为输入）。
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from . import _circleg  # noqa: F401
from . import robot as R
from .dynamics import Gbc486Dynamics, leg_combination
from .model_builder import HeldLegModelBuilder, LegInputModelBuilder
from .wbc import Gbc486Tasks, Nominal, WbcGains, resolver_factory
from Algorithm.geometry import wrap_angle  # noqa: E402
from Algorithm.contact_state import (GroundContactConfig,  # noqa: E402
                                     GroundContactEstimator, JumpPhaseHint)
from Algorithm.mpc import ChassisState, MotionCommand, MPCSettings, WheelMPC  # noqa: E402
from APP.mujoco_mpc_wbc import _PRESETS, default_app_mpc_settings  # noqa: E402
from Algorithm.supervision import (ControlSupervisor, SupervisedControl,  # noqa: E402
                                   WbcFeedbackAccumulator)
from Algorithm.timing import ControlTiming  # noqa: E402
from Algorithm.wbc import ContactElimination  # noqa: E402

G = 9.81
# viewer 刷新周期（仿真时间），约 60 fps。见 run() 中 viewer.sync() 的说明：
# 每物理步都 sync 会让 viewer 比实时慢 9 倍（渲染 91441 顶点的 base_link 等）。
VIEWER_SYNC_DT = 1.0 / 60.0
_LOGGER = logging.getLogger(__name__)

# 驱动阶段（DRIVE 及其后的 BRAKE）的共模腿角预算 [rad]。单一真值来源：
# objective._settings 与本模块的 gbc486_mpc_settings 都读它。
# 2026-10-05 用户选定 **方案 B**（保姿态、压摆幅的折中，取代方案 A 的"俯仰严格为 0"）：
# 驱动段腿角封顶 0.20 rad，超出的质心偏移交给俯仰（允许加减速瞬态前倾约 14°），
# 于是"巡航姿态水平（a=0 -> theta=0）+ 加减速摆腿小 + 加速度够"能同时成立。
# 依据：Delta = z*a/g = r_eff*sin(phi) + z*sin(theta)，本机 r_eff/z = 0.8064。
#   phi=0.20 时：a=2.0 -> theta=2.5deg；a=3.0 -> 8.4deg；a=4.0 -> 14.3deg；a=5.0 -> 20.5deg（超限）。
# 所以本预算下 a <= 约 4.0 m/s^2（配合 zmp_split 与 15deg 倾角能力表）。
DRIVE_HIP_ANGLE_LIMIT = 0.20
# 方案 B 允许的加减速前倾上限 [rad]（15 deg）。必须与倾角能力表的生成角度一致：
# A1 用它反查"前倾这么多时能持续多大加速度"。
PITCH_FEEDFORWARD_MAX = 0.2618
# 2026-10-05 用户指定"控制速度提高一倍"（circleg 原值 2.5 / 3.0）。
# SPEED_LIMIT 是 MPC 状态带，必须 >= MAX_SPEED，否则内部先被卡死。
# 2026-10-05 用户定稿：最大速度 **1.5 m/s**（先前的 2.5 / 5.0 尝试均作废）。
# 用户原话：只需求 1.5 就够，不再追求高速。
# 副带好处：1.5 m/s 下制动时间 = 1.5/3.37 = 0.45 s < MPC 视界 0.5 s，
# 所以“视界短于制动时间”的问题在当前速度下不存在（不需要稀疏 QP）。
# SPEED_LIMIT 是 MPC 状态带，必须 >= MAX_SPEED，否则内部先被卡死；
# speed_guard（SimulationConfig）同样必须 > MAX_SPEED，否则一超就停机。
MAX_SPEED = 1.5
SPEED_LIMIT = 3.0
# MPC 步长与视界（2026-10-05 尝试加长，**失败，已回退**）。
#
# 为什么失败（实测）：加长视界后 OSQP 在**凝聚型 QP** 上不收敛，日志为
# ``MPC 求解未成功(maximum iterations reached)``，每拍走兜底通路，控制迅速崩：
#   N=25 (0.50 s) 俯仰峰  0.73 deg / 位移 188 mm   —— 已验证配置
#   N=40 (0.80 s) 俯仰峰  1.48 deg（已退化）
#   N=50 (1.00 s) 俯仰峰  5.45 deg
#   N=60 (1.20 s) 俯仰峰 38.43 deg，负法向力保护停机
#   N=75 (1.50 s) 俯仰峰 44.91 deg，attitude_guard 停机
# 把 ``osqp_settings`` 的 max_iter 提到 2e5、放宽 eps 到 1e-4 都救不回来（病态，非迭代数）。
#
# 代价标度（同一台机、同一工况）：
#   dt=0.02 N=25  -> 1.45/2.18 ms      dt=0.02 N=75 -> 7.32/11.36 ms
#   dt=0.02 N=100 -> 12.66/17.81 ms    dt=0.01 N=200 -> 146/453 ms（不可行）
# 结论：代价由 N（决策变量 2N 个）主导，想“更长视界 + 更高频率”必须先换
# **稀疏 QP 公式**（状态也当决策变量、动力学作为稀疏约束），而不是调这两个数。
# 未完成项：稀疏 QP 尚未实现。
MPC_DT = 0.02
MPC_HORIZON = 25


def preset_settings(preset: str | None):
    """按 circleg ``_build_from_args`` 的方式组装 ``MPCSettings``（单一真值来源）。

    ``preset = None`` 时用 ``default_app_mpc_settings()``（circleg viewer 默认 = tuned-fast-accel）；
    给定 preset 名时以 ``MPCSettings(compensate_hip_torque=False)`` 为底，再叠加 preset 的
    weights / hip_angle_limit / use_jerk_limit / tilt_assisted_limits / pitch_feedforward / mpc_overrides。
    本函数不包含任何本机器人专属量，那些在 :func:`gbc486_mpc_settings` 里叠。

    返回 ``(preset 条目, settings)``。
    """
    entry = dict(_PRESETS.get(preset, {})) if preset else {}
    if not preset:
        return entry, default_app_mpc_settings()
    base = MPCSettings(compensate_hip_torque=False)
    weights = replace(base.weights, **entry.get("weights", {})) if entry.get("weights") else base.weights
    settings = replace(base, weights=weights,
                       hip_angle_limit=float(entry.get("hip_angle_limit", base.hip_angle_limit)))
    if entry.get("use_jerk_limit"):
        settings = replace(settings, limits=replace(settings.limits, use_jerk_limit=True))
    if entry.get("tilt_assisted_limits") or entry.get("pitch_feedforward"):
        settings = replace(settings,
                           tilt_assisted_limits=bool(entry.get("tilt_assisted_limits")),
                           pitch_feedforward=bool(entry.get("pitch_feedforward")))
    if entry.get("mpc_overrides"):
        settings = replace(settings, **entry["mpc_overrides"])
    return entry, settings


def gbc486_mpc_settings(model, dynamics: Gbc486Dynamics, reduced, leg_angle_limit: float | None = None,
                        leg_length: float | None = None, preset: str | None = None) -> MPCSettings:
    """preset（circleg） + 本机器人物理量（部署方案第 7 节）。

    控制参数（权重、规划限值、jerk、倾角辅助、ZMP split 等）全部来自 circleg preset；
    这里只覆盖“换模型必须换”的物理量：轮力矩限、标称法向力、偏航力矩预算、scrub 系数、
    共模腿角可达范围、腿力矩限、ZMP 的质心高与“曲柄半径”等效值。
    """
    entry, base = preset_settings(preset)
    wheel_limit = float(model.actuator_ctrlrange[dynamics.wheel_actuators[0], 1])
    mass = float(sum(model.body_mass))
    force_limit = wheel_limit / reduced.wheel_radius
    com_height = float(reduced.metadata.get("com_height", 0.0))
    overrides = dict(
        wheel_torque_limit=wheel_limit,                            # 4.0（circleg 5.0）
        nominal_normal_force=mass * G / 2.0,                       # 71.3（circleg 46.6）
        yaw_moment_limit=2.0 * force_limit * reduced.half_track,   # 2·(τ/R)·d（circleg 7.5）
        scrub_moment_coefficient=0.0,                              # 对称姿态下两轮不前后错开
        compensate_hip_torque=False,                               # 腿反作用已在模型里；8 状态的 τ_c 本身就是输入
        # 2026-10-05 用户指定"控制速度提高一倍"：circleg 的 2.5 / 3.0 一律翻倍。
        limits=replace(base.limits, equilibrium_pitch=0.0, max_speed=MAX_SPEED),
        speed_limit=SPEED_LIMIT,
        dt=MPC_DT,
        horizon=MPC_HORIZON,
    )
    if com_height > 0.0:
        # ZMP：Δ* = z_com·a/g。两个长度都必须是**实测值**：
        # - z_com 用接触点以上的实测质心高；
        # - r_crank 用 d(CoM_x − 接触点_x)/d(δφ_c)，**不是标称腿长**（质心不在腿顶端，
        #   本机 0.2133 vs L0 0.280，差 31%，直接用 L0 会让曲柄参考偏小）。
        overrides["zmp_com_height"] = com_height
        offset_radius = reduced.metadata.get("leg_offset_radius")
        if offset_radius is None:
            _LOGGER.warning("模型 %s 没有 leg_offset_radius，ZMP 曲柄半径沿用 preset 值"
                            "（不能拿标称腿长代替）", reduced.source)
        else:
            overrides["zmp_crank_radius"] = float(offset_radius)
    # 2026-10-05 用户重新定档（方案 C：姿态优先，允许降低加速度上限）：
    #   优先级 = 姿态倾角尽量小 > 腿部摆角小 > 加速度上限（可降）。
    # 1) zmp_split = 1.0：质心偏移**全部由摆腿承担**，俯仰参考严格为 0。
    #    物理依据 a = g*(r_eff*sin(phi) + z*sin(theta))/z：theta=0 时腿角
    #    phi = asin(z*a/(g*r_eff)) = asin(0.1264*a)，所以倾角/摆幅都小就必须把 a 压低。
    # 2) tilt_assisted_limits = False：不再用"机身前倾"换能力，规划限值改由
    #    theta=0 的 accel_capacity 表反查（A1），与本档的"不用倾角"一致。
    # 3) 驱动腿角预算 DRIVE_HIP_ANGLE_LIMIT = 0.20 rad：该点 theta=0 能力 1.59 m/s^2，
    #    A1 取 0.85/0.95 -> 起步 1.35 / 刹车 1.51 m/s^2。
    overrides["zmp_split"] = 1.0
    overrides["tilt_assisted_limits"] = False
    overrides["pitch_feedforward_max"] = PITCH_FEEDFORWARD_MAX
    overrides["brake_pitch_max"] = PITCH_FEEDFORWARD_MAX
    if leg_angle_limit is not None:
        # 预设给的 0.8 / drive 1.2 rad 超出本机共模腿角可达范围（M0 实测约 ±0.55），换成实测值
        overrides["hip_angle_limit"] = float(leg_angle_limit)
        if base.drive_hip_angle_limit is not None:
            overrides["drive_hip_angle_limit"] = DRIVE_HIP_ANGLE_LIMIT
        overrides["hip_torque_limit"] = float(model.actuator_ctrlrange[dynamics.leg_actuators[0], 1])
        # 曲柄参考上限跟**驱动段**预算（原实现用静态 hip_angle_limit，见 MPCSettings.zmp_crank_limit）
        overrides["zmp_crank_limit"] = 0.9 * DRIVE_HIP_ANGLE_LIMIT
    if base.zmp_reference and reduced.state_dim < 8:
        # ZMP 参考会把摆腿分量写入腿角状态那两列（`_apply_zmp_reference` 写 [:, 6] / [:, 7]），
        # 6 状态模型没有这两列，所以自动关闭。circleg 也是把 zmp_reference 与 hip_input（8 状态）
        # 捆在同一个 preset 里的。
        overrides["zmp_reference"] = False
    return replace(base, **overrides)


@dataclass(frozen=True)
class Disturbance:
    """与 circleg 同义：瞬时速度增量（世界系线速度 / 机体系角速度）+ 持续外力。

    持续分量（``force`` / ``torque`` / ``duration``）在 ``duration`` 内作为广义外力
    （``qfrc_applied``）每步施加，对应 viewer 里鼠标拖住不放的情形。
    """

    time: float
    delta_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)   # [m/s] world-frame
    delta_omega: tuple[float, float, float] = (0.0, 0.0, 0.0)      # [rad/s] body-frame
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)            # [N] world-frame
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)           # [N*m] body-frame
    duration: float = 0.0                                          # [s]


@dataclass(frozen=True)
class SimulationConfig:
    """对应 circleg ``SimulationConfig`` 中本阶段用到的字段（同名、同默认值）。"""

    backend: str = "wls"          # 2026-10-04 用户指定：默认后端改为 WLS（nullspace 会饿死末级任务）
    model_variant: str = "held_leg"        # held_leg（6 状态）| leg_input（8 状态）
    # circleg preset 名（取自 APP._PRESETS）；"none" = circleg viewer 默认（tuned-fast-accel，无 ZMP）
    preset: str = "tuned-agile"            # 含 ZMP 参考（zmp_reference / zmp_split 0.3）
    # 标称腿长 [m]。M0 实测机构伸展上限 0.430 m、缩腿下限 0.14 m，所以取行程带中点
    # （0.43−L0 ≈ L0−0.14 → L0 ≈ 0.285），取 0.28：双向各约 0.15 m 行程。
    # 原值 0.42 贴在伸展上限，只乘 10 mm 余量，机身太高且高度/横滚任务没权限。
    nominal_length: float = 0.28
    # 初始俯仰扰动 [rad]（circleg ``--scenario pitch`` 用的 ±0.02）。
    initial_pitch: float = 0.0
    attitude_guard: float = 0.785
    speed_guard: float = 3.00
    force_tolerance: float = 1.0e-6
    # WBC 采样 MPC 轨迹时的时间偏移 [s]（circleg ``SimulationConfig.preview_lead``）。
    # 与 ``MPCSettings.prediction_lead`` 不是一回事：后者是 MPC 内部把反馈状态前推的量
    # （Algorithm/mpc/controller.py:585），前者只管 WBC 从哪一时刻取参考。
    # circleg 历史故障“相对 MPC 参考的 PD 失效（J 9.03->29.01）”的触发条件就是
    # preview_lead=0.02（APP/mujoco_mpc_wbc.py:421 注释），所以两者必须分开。
    preview_lead: float = 0.0
    # 预测法向力保护：实测静止时反算准确（右 71.4 / 左 71.3 N），但瞬态会出现单拍跳变
    # （实测 +47.3 -> -9.1 N，一拍 4 ms）。原先用 force_tolerance=1e-6 当阀值，等于零容差，
    # 会被单拍跳变打穿（grid3 三阶段 41 例不可行里 38 例都是它）。现改为：
    #   阀值 = -max(force_tolerance, margin_ratio * 名义单轮法向力)   0.05 * 71.3 = 3.6 N
    #   并且要求连续 duration 秒为负才停机（与离地检测同一思路的时间确认）。
    normal_force_margin_ratio: float = 0.05
    normal_force_duration: float = 0.012   # [s] 连续 3 个 WBC 周期
    record_stride: int = 4
    wbc_gains: WbcGains = field(default_factory=WbcGains)
    # 8 状态模型的共模腿角状态限值 [rad]（偏差量）。依据 M0 扫描：L = 0.28 m 时共模腿角
    # 可达 [−0.784, +0.674]，相对 φ0 = −0.1263 即 −0.658 / +0.800，取 0.65（不超可达范围）。
    leg_angle_limit: float = 0.65
    # 执行器限幅覆盖 [N*m]（None = 沿 MJCF 的 actuator ctrlrange）。
    # 用户 2026-10-04：限幅力矩纳入优化备选范围，所以做成可覆盖，而不是写死读模型。
    wheel_torque_limit: float | None = None
    leg_torque_limit: float | None = None


class Gbc486MpcWbcSimulation:
    def __init__(self, config: SimulationConfig = SimulationConfig(), mpc_settings: MPCSettings | None = None):
        self.config = config
        self._viewer_sync_time = -1.0e9      # 上次 viewer.sync() 的仿真时刻
        self.model, self.data = R.load()
        self._apply_torque_limits()
        self.pose = R.standing_pose(self.model, self.data, config.nominal_length)
        self.dyn = Gbc486Dynamics(self.model, self.data)
        self.reduced = (LegInputModelBuilder(self.dyn, self.pose.length).build()
                        if config.model_variant == "leg_input" else
                        HeldLegModelBuilder(self.dyn).build())
        mujoco.mj_forward(self.model, self.data)
        settings = mpc_settings or gbc486_mpc_settings(
            self.model, self.dyn, self.reduced,
            config.leg_angle_limit if config.model_variant == "leg_input" else None,
            leg_length=self.pose.length,
            preset=None if config.preset in (None, "", "none") else config.preset)
        self.mpc = WheelMPC(self.reduced, settings)
        self.timing = ControlTiming(self.model.opt.timestep, 0.004, self.mpc.settings.dt)
        self.supervisor = ControlSupervisor(self.mpc.settings.limits.touchdown_dwell)
        self.control = SupervisedControl(self.mpc, wbc_resolver_factory=resolver_factory(config.backend))
        # 离地判定放在算法层（Algorithm.contact_state），与轮圈腿同一套；适配层不再自己拿
        # 仿真接触力做裸阈值。off_duration_idle 依据本机实测：瞬时卸载（轮子仍接地）低法向力
        # 持续 0.008~0.020 s，真实腾空 0.204 s；取 0.050 s = 瞬时卸载的 2.5 倍、腾空的 0.25 倍。
        # 不用 specific_force_z / lift_height：本机仿真没有加速度计传感器，用速度差分代替
        # 噪声过大（实测瞬时卸载时低至 -4.12 m/s²，会把旁证误放行）。待实车有真实 IMU 加速度
        # 计时再启用。
        self.contact_estimator = GroundContactEstimator(
            GroundContactConfig(off_duration_idle=0.050))
        # preset 的 tracking_bandwidth -> WBC 跟踪带宽（circleg: kv = 2ω、kp = ω²）
        omega = _PRESETS.get(config.preset, {}).get("tracking_bandwidth")
        gains = (WbcGains(tracking_velocity_gain=2.0 * float(omega),
                          tracking_position_gain=float(omega) ** 2) if omega else config.wbc_gains)
        self.tasks = Gbc486Tasks(self.dyn, Nominal(self.pose.length, self.pose.angle, 0.0),
                                 gains, leg_input=config.model_variant == "leg_input",
                                 state_dim=self.reduced.state_dim)
        # circleg: WbcFeedbackAccumulator(("forward_velocity", "yaw"))
        self.feedback = WbcFeedbackAccumulator(("forward_velocity", "yaw"))
        # 初始俯仰扰动：必须在降阶模型建好之后施加（circleg 也是这个顺序），
        # 否则模型会绕一个已经倾斜的位形线性化。
        if config.initial_pitch:
            self._set_initial_pitch(config.initial_pitch)
        self.distance = 0.0
        self.solution = None
        self._normal_violation_time = 0.0
        self._infeasible_active = False
        self.records: list[dict] = []
        self.stop_reason = None
        self.counters = dict(mpc_calls=0, mpc_fallbacks=0, mpc_relaxed=0, wbc_calls=0, wbc_fallback=0,
                             wbc_clip=0, wbc_infeasible=0)
        # 任务名随 wbc.py 的任务集变化（现在是每条腿单独的 length_*/angle_*），
        # 用动态字典而不是写死的键，避免 KeyError。
        self.exhausted: dict[str, int] = {}
        self.scaled: dict[str, int] = {}
        self.timings = dict(mpc=[], wbc=[])

    def _apply_torque_limits(self) -> None:
        """按 config 覆盖执行器 ctrlrange。必须在建 dynamics / MPC 之前调用，
        否则 ``ActuatorLimits`` 与 ``wheel_torque_limit`` 读到的还是 MJCF 的旧值。"""
        groups = (
            (("Right_Wheel_act", "Left_Wheel_act"), self.config.wheel_torque_limit),
            (("Right_front_joint_act", "Right_rear_joint_act", "Left_front_joint_act",
              "Left_rear_joint_act"), self.config.leg_torque_limit),
        )
        for names, limit in groups:
            if limit is None:
                continue
            bound = abs(float(limit))
            for name in names:
                self.model.actuator_ctrlrange[self.model.actuator(name).id] = (-bound, bound)

    def _set_initial_pitch(self, pitch: float) -> None:
        """与 circleg ``MuJoCoRobot.set_initial_pitch`` + ``_place_wheels_on_ground`` 相同：
        在站立位形上给机身一个俯仰角，然后整体下移使轮子重新落地。"""
        half = 0.5 * float(pitch)
        self.data.qpos[3:7] = np.array([math.cos(half), 0.0, math.sin(half), 0.0])
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[2] -= float(np.min(R.wheel_contact_points(self.model, self.data)[:, 2]))
        mujoco.mj_forward(self.model, self.data)

    # ----------------------------------------------------------------- state
    def chassis_state(self, contact, forces):
        rows = self.dyn.rowset()
        rates = rows.task[0] @ self.data.qvel
        roll, pitch, yaw = self.dyn.attitude()
        leg = self.dyn.leg_state()
        leg_rows, _ = leg_combination(rows, +1.0)
        return ChassisState(
            position=self.distance, velocity=float(rates[0]),
            pitch=float(pitch) - self.reduced.metadata["equilibrium_pitch"],
            pitch_rate=float(rates[2]), yaw=float(yaw), yaw_rate=float(rates[3]),
            # 降阶模型第 6 个状态是共模腿角偏差 phi_c，反馈也取共模 0.5*(phi_R+phi_L)-phi0。
            # 用单腿（leg[1]）会让 MPC 与 WBC 的腿角反馈带上一份 phi_d 的系统误差。
            # 符号见 model_builder.LegInputModelBuilder.set_legs：δφ_c := −(极角 − φ0)。
            # 算法层依赖 ∂v̇/∂φ_c > 0，所以这里也要取负，否则状态与参考不同号。
            hip_angle=float(self.pose.angle - 0.5 * (leg[1] + leg[3])),
            hip_rate=-float(leg_rows[1] @ self.data.qvel),
            normal_forces=tuple(float(v) for v in forces[:, 2]), contact=contact), rows

    # ------------------------------------------------------------------- run
    def run(self, command, duration: float, disturbances=(), viewer=None):
        """积分情景；传入 viewer 时按实时步长同步（与 circleg ``run`` 相同）。"""
        m, d = self.model, self.data
        infinite = not math.isfinite(duration)
        if infinite and viewer is None:
            raise ValueError("无限时长需要 viewer 才能停下来")
        pending = sorted(disturbances, key=lambda item: item.time)
        steps = None if infinite else int(round(duration / m.opt.timestep))
        decision, self._command = None, None
        force_until, wrench = 0.0, np.zeros(6)
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
            if self.timing.wbc_due(step):
                mujoco.mj_forward(m, d)
                _measured, forces, counts, other = self.dyn.sensed_contact()
                # 逐轮法向力交给算法层估计器（滞回 + 确认时间）再给监督层；直接拿裸阈值会把
                # MuJoCo 在几十微米间隙下关闭接触误判成离地（实测轮地间隙仅 +0.02~0.04 mm）。
                contact = self.contact_estimator.update(
                    tuple(forces[:, 2]), self.timing.wbc_dt, phase=JumpPhaseHint.NONE).contact
                state, rows = self.chassis_state(contact, forces)
                roll, pitch, _ = self.dyn.attitude()
                decision = self.supervisor.update(
                    timestamp=d.time, dt=self.timing.wbc_dt, contact=contact, enabled=True,
                    command_fresh=True, stable=max(abs(roll), abs(pitch)) < self.mpc.settings.pitch_limit,
                    slipping=False)
                if not decision.motion_enabled:
                    self.solution = None
                if self.timing.mpc_due(step):
                    current = command(d.time) if callable(command) else command
                    self._command = current
                    start = time.perf_counter()
                    self.solution = self.control.solve_mpc(state, current, decision, self.feedback.consume(state))
                    self.timings["mpc"].append(time.perf_counter() - start)
                    if self.solution is not None:
                        self.counters["mpc_calls"] += 1
                        self.counters["mpc_fallbacks"] += int(self.solution.fallback)
                        self.counters["mpc_relaxed"] += int(bool(self.solution.qp.relaxed))
                sample = (None if self.solution is None
                          else self.solution.trajectory.sample_at(d.time + self.config.preview_lead))
                balance, motion = self.tasks.tasks(state, rows, sample)
                blocks = self.tasks.contact_blocks(rows, decision.contact)
                start = time.perf_counter()
                torque, result = self.control.solve_wbc(
                    mass=self.dyn.mass(), selection=self.dyn.selection, dynamics_bias=self.dyn.dynamics_bias(),
                    contacts=blocks, balance_tasks=balance, motion_tasks=motion, limits=self.dyn.limits,
                    decision=decision, dt=self.timing.wbc_dt)
                self.timings["wbc"].append(time.perf_counter() - start)
                self.counters["wbc_calls"] += 1
                reason = None
                predicted_normal = np.zeros(0)
                if result is not None:
                    self.counters["wbc_fallback"] += int(result.fallback_used)
                    self.counters["wbc_clip"] += int(result.clip_applied)
                    self.counters["wbc_infeasible"] += int(not result.feasible)
                    # WBC -> MPC 反馈：把任务残差与饱和/越限回传给 planner（circleg APP:593
                    # 的同一位置）。缺这一句时 ReferenceFeedback.saturated 恒 False、
                    # motion_residual 恒 0，planner.plan 的 blocked 永不成立，capability
                    # 不会因 WBC 受限而降级，MPC 会持续规划 WBC 做不到的运动。
                    self.feedback.observe(result, self.dyn.limits)
                    exhausted = set()
                    for outcome in result.hierarchy.outcomes:
                        self.exhausted[outcome.name] = (self.exhausted.get(outcome.name, 0)
                                                        + int(outcome.exhausted))
                        if outcome.exhausted:
                            exhausted.add(outcome.name)
                    for name, value in result.scaling.items():
                        if name not in exhausted:
                            self.scaled[name] = (self.scaled.get(name, 0)
                                                 + int(value < 1.0 - 1e-8))
                    active = [b for b in blocks if all(decision.contact[i] for i in b.wheels)]
                    jc = np.vstack([b.jacobian for b in active])
                    predicted_normal = ContactElimination.from_matrices(
                        jc, self.dyn.selection, self.dyn.mass()).recover_contact_force(
                        result.qacc, self.dyn.dynamics_bias(), torque)
                d.ctrl[:] = torque
                normal_threshold = -max(self.config.force_tolerance,
                                        self.config.normal_force_margin_ratio
                                        * self.mpc.settings.nominal_normal_force)
                normal_min = (float(np.min(predicted_normal[list(
                    self.tasks.normal_indices(decision.contact))]))
                    if predicted_normal.size else float("inf"))
                self._normal_violation_time = (self._normal_violation_time + self.timing.wbc_dt
                                               if normal_min < normal_threshold else 0.0)
                # WBC 求解不可行（强扰下 WLS 无法精确满足硬约束，力矩会被 clamp）：
                # 用户 2026-10-04 指示，这里不判失败、不停机，只记一条警告后继续；
                # 发散后的处置方案另行安排。力矩仍然是求解器输出的（已 clamp 到限幅内）。
                if result is not None and not result.feasible and not self._infeasible_active:
                    _LOGGER.warning("WBC 求解不可行（动力学残差 %.3e，torque_clipped=%s）：仅警告，继续运行",
                                    result.dynamics_residual_norm, result.clip_applied)
                self._infeasible_active = bool(result is not None and not result.feasible)
                if other:
                    reason = "unexpected_body_contact"
                elif self.mpc.degraded:
                    reason = "persistent_mpc_fallback"
                elif self._normal_violation_time + 1e-12 >= self.config.normal_force_duration:
                    reason = "negative_predicted_normal_force"
                elif not self.dyn.limits.contains(torque):
                    reason = "torque_limit"
                if step % self.config.record_stride == 0:
                    self.records.append(self._record(state, decision, sample, torque, result,
                                                     predicted_normal, rows))
                if reason:
                    self.stop_reason = reason
                    break
            previous = d.qpos[:3].copy()
            yaw = self.dyn.attitude()[2]
            mujoco.mj_step(m, d)
            middle = yaw + 0.5 * wrap_angle(self.dyn.attitude()[2] - yaw)
            self.distance += float((d.qpos[:3] - previous) @ np.array([math.cos(middle), math.sin(middle), 0.0]))
            hits = self.dyn.joint_limit_active()
            if hits:
                self.stop_reason = "joint_limit:" + ",".join(sorted(set(hits)))
                break
            if not np.all(np.isfinite(np.r_[d.qpos, d.qvel])):
                self.stop_reason = "nonfinite_state"
                break
            if max(abs(v) for v in self.dyn.attitude()[:2]) > self.config.attitude_guard:
                self.stop_reason = "attitude_guard"
                break
            if float(np.linalg.norm(d.qvel[:2])) > self.config.speed_guard:
                self.stop_reason = "speed_guard"
                break
            if viewer is not None:
                # 被动 viewer 刷新 + 实时限速；关窗口即停。
                # 2026-10-05：``sync()`` **不再每物理步调用**。实测每步 ``sync()+渲染`` 约 8.2 ms
                # （物理步长 1 ms），使 viewer 比实时慢 **9.16 倍**：用户看到的“收敛慢/效果差”
                # 有相当一部分是显示器上的时间被拉长 9 倍造成的假象（headless 同配置为 0.95 倍）。
                # 现在改为按渲染帧率（约 60 Hz 仿真时间）刷新，物理与控制在 1 kHz 照常推进。
                # 渲染负担主因：``base_link`` 单个网格 91441 顶点。
                if float(d.time) - self._viewer_sync_time >= VIEWER_SYNC_DT:
                    viewer.sync()
                    self._viewer_sync_time = float(d.time)
                remaining = m.opt.timestep - (time.perf_counter() - step_wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
                if not viewer.is_running():
                    self.stop_reason = "viewer_closed"
                    break
            step += 1
        return self.summary(duration)

    def _record(self, state, decision, sample, torque, result, predicted_normal, rows) -> dict:
        legs = self.dyn.leg_state()
        common_rows, _ = leg_combination(rows, +1.0)
        roll_rate = float(rows.task[0][1] @ self.data.qvel)
        return dict(
            t=round(float(self.data.time), 6), mode=decision.mode.value,
            s=state.position, v=state.velocity, pitch=state.pitch, pitch_rate=state.pitch_rate,
            yaw=state.yaw, yaw_rate=state.yaw_rate, roll=float(self.dyn.attitude()[0]),
            roll_rate=roll_rate,
            # 腿长 / 腿角速率走雅可比（共模行 @ qvel），不用差分：差分有采样延迟与噪声
            length_rate=float(common_rows[0] @ self.data.qvel),
            leg_rate=float(common_rows[1] @ self.data.qvel),
            leg=legs.tolist(), torque=np.asarray(torque, dtype=float).tolist(),
            cmd_v=None if self._command is None else float(self._command.velocity or 0.0),
            cmd_w=None if self._command is None else float(self._command.yaw_rate or 0.0),
            plan_v=None if self.solution is None else float(self.solution.reference.velocity[0]),
            plan_w=None if self.solution is None else float(self.solution.reference.yaw_rate[0]),
            plan_yaw=None if self.solution is None else float(self.solution.reference.yaw[0]),
            ref=None if sample is None else sample.state.tolist(),
            ref_acc=None if sample is None else sample.acceleration.tolist(),
            mpc_force=None if self.solution is None else self.solution.force.tolist(),
            planner=self.mpc.planner.mode.value,
            predicted_normal=predicted_normal.tolist(),
            dynamics_residual=None if result is None else float(result.dynamics_residual_norm),
            fallback=None if result is None else bool(result.fallback_used))

    def summary(self, duration: float) -> dict:
        rec = self.records
        arr = lambda key: np.array([r[key] for r in rec], dtype=float)  # noqa: E731
        legs = np.array([r["leg"] for r in rec])
        times = {k: np.array(v) * 1e3 for k, v in self.timings.items() if v}
        modes: dict[str, int] = {}
        for r in rec:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        return dict(
            backend=self.config.backend, model_variant=self.config.model_variant,
            completed=self.stop_reason is None, stop_reason=self.stop_reason,
            simulated=float(self.data.time),
            requested_seconds=None if not math.isfinite(duration) else duration, **self.counters,
            exhausted=self.exhausted, scaled=self.scaled, modes=modes,
            max_abs_pitch=float(np.max(np.abs(arr("pitch")))), max_abs_roll=float(np.max(np.abs(arr("roll")))),
            max_abs_s=float(np.max(np.abs(arr("s")))), final_s=float(rec[-1]["s"]),
            max_abs_v=float(np.max(np.abs(arr("v")))), final_v=float(rec[-1]["v"]),
            max_leg_length_error=float(np.max(np.abs(legs[:, [0, 2]] - self.pose.length))),
            max_torque=np.max(np.abs(np.array([r["torque"] for r in rec])), axis=0).tolist(),
            timing_ms={k: dict(p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)),
                               max=float(v.max())) for k, v in times.items()},
            model=dict(state_dim=self.reduced.state_dim, input_dim=self.reduced.input_dim,
                       A=self.reduced.A.tolist(), B=self.reduced.B.tolist(),
                       unstable_pole=float(self.reduced.unstable_pole), half_track=self.reduced.half_track,
                       yaw_inertia=self.reduced.yaw_inertia, wheel_radius=self.reduced.wheel_radius,
                       metadata=self.reduced.metadata),
        )


def default_command() -> MotionCommand:
    return MotionCommand(velocity=0.0, yaw_rate=0.0)
