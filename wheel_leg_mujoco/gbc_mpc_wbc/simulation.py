"""GBC486 轮腿的闭环装配与运行循环（MPC + WBC）。

本模块做什么
------------
把「模型 / 动力学 / 降阶模型 / WBC 求解器」这些零件装成一条可运行控制链，
并给出按物理步推进的 ``run()`` 循环。

``run()`` 内的四段是**刻意分开**的（用户 2026-10-07 要求：采样、输出、控制计算解耦；
MPC 与 WBC 也解耦，因为两者频率不同，按异步运行）：

    ============  ==================  ===================================
    阶段          节拍                职责
    ============  ==================  ===================================
    采样级        每物理步（1 kHz）   运动学、接触力、离地判定、状态、监督层模式
    控制级 A      MPC 节拍（50 Hz）   自己的 ``mpc_due(step)``，与 WBC 异步
    控制级 B      WBC 节拍（250 Hz）  自己的 ``wbc_due(step)``，取带时间戳轨迹采样
    输出级        每物理步（1 kHz）   把最近一次控制解写进 ``data.ctrl``（零阶保持）
    ============  ==================  ===================================

节拍之间只用循环内局部变量传递（``torque`` / ``state`` / ``rows`` / ``contact`` /
``decision`` / ``sample``），不额外造包装类：改某一级的频率不影响其它级的口径。

本模块不做什么
--------------
- 不定义 WBC 任务与接触块（见 ``wbc.py``）、不实现 WBC 求解器（见 ``wbc_limited.py``）
- 不建模型、不做线性化（见 ``robot.py`` / ``model_builder.py``）
- 不配置日志（见 ``logging_setup.py``）；这里只 ``get_logger`` 后写日志

时序说明：物理步长取模型 ``opt.timestep``；``wbc_dt``/``mpc_dt`` 必须是它的整数倍
（见 ``Algorithm/timing.py``）。viewer 的实时同步见 ``VIEWER_SYNC_DT`` 处的注释。
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from . import _circleg  # noqa: F401
from .logging_setup import get_logger  # noqa: E402
from .loop import VIEWER_SYNC_DT, run_loop  # noqa: E402
from .jump import JumpPhase  # noqa: E402
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
_LOGGER = get_logger(__name__)

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


def _leg_length_limits(model, data, angle: float, margin: float = 0.0,
                       low: float = 0.100, high: float = 0.470, step: float = 0.002):
    """扫描腿长机械可用范围：返回所有**受限**关节距限位 >= margin 的腿长区间 [m]。

    用模型自带的逆运动学 ``R.solve_leg`` 取位形，只读关节角与限位，不碰控制链路。
    受限关节为 jAB/jAG/jGH（右腿）/jIJ/jIO/jOP（左腿）；被动 jBE/jEC/jCF 在 MJCF 里没有限位
    （``jnt_range`` 为 [0,0]），若参与判定余量恒为负（本次踩过的坑，见方案 9.3 节）。
    """
    leg = R.RIGHT
    joints = [name for name in (leg.front_joint, leg.rear_joint) + leg.passive_joints
              if model.joint(name).limited]
    accepted = []
    for value in np.arange(low, high + 1e-9, step):
        R.solve_leg(model, data, leg, float(value), angle)
        worst = min(min(float(data.qpos[model.joint(name).qposadr[0]]) - model.jnt_range[model.joint(name).id][0],
                        model.jnt_range[model.joint(name).id][1] - float(data.qpos[model.joint(name).qposadr[0]]))
                    for name in joints)
        if worst >= margin:
            accepted.append(float(value))
    return (min(accepted), max(accepted)) if accepted else (float("nan"), float("nan"))


def _com_rise_ratio(model, data, angle: float, length_hi: float, length_lo: float) -> float:
    """kappa = 质心相对**轮地接触点**的抬升 / 腿长行程。

    固定机身位形下分别取两个腿长，比较 "子树质心 z - 轮心 z" 的差值——等价于轮子压在地面上时
    机体抬升多少（用户口径：跳跃高度算的是质心抬升）。
    """
    heights = []
    for length in (length_hi, length_lo):
        for leg in R.LEGS:
            R.solve_leg(model, data, leg, length, angle)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com = float(data.subtree_com[model.body("base").id][2])
        wheel = float(data.xpos[model.body(R.RIGHT.wheel_body).id][2])
        heights.append(com - wheel)
    return (heights[0] - heights[1]) / (length_hi - length_lo)


@dataclass(frozen=True)
class SimulationConfig:
    """对应 circleg ``SimulationConfig`` 中本阶段用到的字段（同名、同默认值）。"""

    backend: str = "wls"          # 2026-10-04 用户指定：默认后端改为 WLS（nullspace 会饿死末级任务）
    model_variant: str = "leg_input"        # leg_input（8 状态，含腿摆角 φ）| held_leg（6 状态）
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
    # ---------------- 跳跃（方案 v3，用户 2026-10-07 审阅）----------------
    # 目标质心抬升 [m]；None = 不启用跳跃。压缩行程 [m] 由用户给定（= 加速行程）。
    jump_height: float | None = None
    jump_stroke: float = 0.12
    jump_time: float = 1.0                 # [s] 起跳指令时刻
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
        self.model, self.data = R.load()
        self._apply_torque_limits()
        self.pose = R.standing_pose(self.model, self.data, config.nominal_length)
        # 开局出生位形（站立解算结果）。手动复位要用它，不能用 model.qpos0：
        # 实测 qpos0 不是站立位形（腿长 0.3138 m vs 站立 0.2801 m），复位后要多解一次才能站住。
        self._initial_qpos = self.pose.qpos.copy()
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
        self._viewer_sync_time = -1.0e9      # 上次 viewer.sync() 的仿真时刻
        self._jump_requested = False         # 外部（view 空格）请求起跳；由 loop 消费
        self.records: list[dict] = []
        self.stop_reason = None
        self.counters = dict(mpc_calls=0, mpc_fallbacks=0, mpc_relaxed=0, wbc_calls=0, wbc_fallback=0,
                             wbc_clip=0, wbc_infeasible=0, mpc_ff_frames=0)
        # 任务名随 wbc.py 的任务集变化（现在是每条腿单独的 length_*/angle_*），
        # 用动态字典而不是写死的键，避免 KeyError。
        self.exhausted: dict[str, int] = {}
        # ---------------- 跳跃装配（方案 v3）----------------
        # 跳跃是独立状态：规划器只吃测量量（腿长/腿长率/接触/姿态），不持有模型与控制器对象。
        self.jump = None
        self._jump_started = False
        if config.jump_height is not None:
            from .jump import JumpConfig, JumpPlanner
            scratch = mujoco.MjData(self.model)
            limits = _leg_length_limits(self.model, scratch, self.pose.angle, margin=0.0)
            ratio = _com_rise_ratio(self.model, scratch, self.pose.angle,
                                    config.nominal_length, config.nominal_length - config.jump_stroke)
            self.jump = JumpPlanner(JumpConfig(
                target_height=float(config.jump_height),
                compression_stroke=float(config.jump_stroke),
                leg_length_nominal=float(config.nominal_length),
                com_rise_ratio=float(ratio),
                total_mass=float(np.sum(self.model.body_mass)),
                leg_length_min=float(limits[0]), leg_length_max=float(limits[1]),
                pitch_limit=float(self.mpc.settings.pitch_limit)))
        self.scaled: dict[str, int] = {}
        self.timings = dict(mpc=[], wbc=[])
        # 第 0 步自检（2026-10-17，用户要求永久保留）：MPC 实际模型维数与 ZMP 摆腿通道的真实状态。
        # 目的是定死“6 状态还是 8 状态”、“zmp_reference 是否打开”、“曲柄半径是否退回 preset”。
        # 只读属性，不参与控制，也不改变任何初始化顺序。
        _LOGGER.info(
            "模型自检: variant=%s reduced.state_dim=%d mpc.model.state_dim=%d input_dim=%d "
            "zmp_reference=%s zmp_split=%.3f zmp_crank_radius=%.4f zmp_com_height=%.4f",
            config.model_variant, self.reduced.state_dim, self.mpc.model.state_dim,
            self.mpc.model.input_dim, self.mpc.settings.zmp_reference,
            self.mpc.settings.zmp_split, self.mpc.settings.zmp_crank_radius,
            self.mpc.settings.zmp_com_height,
        )

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
        """薄委托：状态组装已移到 ``sampling.chassis_state``（下列文件仍引用本方法：wheel_leg_mujoco/tools/m2_standalone_wbc.py, wheel_leg_mujoco/gbc_mpc_wbc/sampling.py）。"""
        from .sampling import chassis_state as _build
        return _build(dynamics=self.dyn, data=self.data, contact=contact, forces=forces,
                      distance=self.distance,
                      equilibrium_pitch=self.reduced.metadata["equilibrium_pitch"],
                      nominal_angle=self.pose.angle)

    # ------------------------------------------------------------------- run
    def run(self, command, duration: float, disturbances=(), viewer=None,
            frame_callback=None, frame_interval: float = 0.0, trace_sink=None,
            explicit_jump_reference: bool = False, compression_only: bool = False,
            freeze_jump_reference: bool = False, vmc_feedforward: bool = False,
            leg_length_compression: bool = False, leg_length_thrust: bool = False,
            thrust_only: bool = False, flight_zero_torque: bool = False,
            flight_leg_retract: bool = False, flight_leg_extend: bool = False,
            mpc_force_feedforward: bool = False, reset_key=None):
        """积分情景；四段时序在 ``loop.run_loop``（采样 / MPC / WBC / 输出 + 推进）。

        本方法只做委托：装配在 ``__init__``，时序在 ``loop.py``，控制算法在
        ``control.py``，反馈在 ``sampling.py``，执行器与护栏在 ``output.py``。
        """
        return run_loop(self, command, duration, disturbances=disturbances, viewer=viewer,
                        frame_callback=frame_callback, frame_interval=frame_interval,
                        trace_sink=trace_sink,
                        explicit_jump_reference=explicit_jump_reference,
                        compression_only=compression_only,
                        freeze_jump_reference=freeze_jump_reference,
                        vmc_feedforward=vmc_feedforward,
                        leg_length_compression=leg_length_compression,
                        leg_length_thrust=leg_length_thrust,
                        thrust_only=thrust_only,
                        flight_zero_torque=flight_zero_torque,
                        flight_leg_retract=flight_leg_retract,
                        flight_leg_extend=flight_leg_extend,
                        mpc_force_feedforward=mpc_force_feedforward,
                        reset_key=reset_key)

    def reset_run(self, viewer=None) -> None:
        """手动复位（view 按 R）：回到出生位形并重新锚定整条控制栈。

        借 circleg ``APP/mujoco_mpc_wbc.py::_recover_from_guard`` 的口径：
        实体复位（qpos=出生位形、qvel/外力/ctrl 清零）+ 控制栈各自 reset +
        MPC 全量重锚（position=None，下次 plan 从实测值锚定）。
        **不重置 data.time**：仿真时间继续走，便于在日志里看出"第几秒按了 R"。
        """
        data, model = self.data, self.model
        data.qpos[:] = self._initial_qpos
        data.qvel[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        if viewer is not None:
            # 清掉鼠标拖拽残留在 viewer 里的扰动，否则复位后会被立刻再推翻
            viewer.perturb.active = 0
            viewer.perturb.select = 0
        mujoco.mj_forward(model, data)
        self.supervisor.reset()
        if getattr(self, "contact_estimator", None) is not None:
            self.contact_estimator.reset()
        self.feedback.reset()
        if self.jump is not None:
            self.jump.phase = JumpPhase.IDLE
            self.jump.reason = None
            self.jump._airborne_confirmed = False                       # noqa: SLF001 - 与 start() 同口径
            self.jump._direction_lost_seen = {"R": False, "L": False}   # noqa: SLF001
        self._jump_started = False
        self._jump_requested = False
        self.mpc.reset(None)
        self.solution = None
        self.distance = 0.0
        self._normal_violation_time = 0.0
        self._infeasible_active = False

    def _jump_takeoff_ready(self) -> tuple[bool, str]:
        """第 29 节"姿态稳定"判据：只引用现成判据，不新建任何阈值。

        - ``planner.mode == HOLD``：仓库里现成的"停稳"定义，已包含 hold_dwell / hold_speed /
          hold_yaw_rate / hold_pitch / hold_pitch_rate / 全接触 / 不打滑；
        - 且两腿腿长在 ``[leg_length_min, leg_length_max]`` 内：否则下一次越界检查会立刻再次中止。
        """
        mode = self.mpc.planner.mode.value
        if mode != "hold":
            return False, "planner.mode=%s（需 hold）" % mode
        config = self.jump.config
        legs = self.dyn.leg_state()
        for side, index in (("R", 0), ("L", 2)):
            length = float(legs[index])
            if not (config.leg_length_min <= length <= config.leg_length_max):
                return False, "leg_length_out_of_range:%s(%.4f m, range [%.3f, %.3f])" % (
                    side, length, config.leg_length_min, config.leg_length_max)
        return True, ""

    def request_jump(self) -> bool:
        """外部请求起跳（view 按空格）。返回是否受理。

        相位判据放在**请求侧**，不放进 loop 的定时触发路径：
        定时触发是"整轮一次"的既有语义，不能被请求侧的状态判断改写。
        第 29 节：IDLE 直接受理；ABORT 且姿态已稳定时先复位到 IDLE 再受理，
        使中止后不需要按 R 就能重新起跳；其余相位拒绝并记录原因。
        没有跳跃对象（``jump_height is None``）时不受理并记 WARN。
        """
        if self.jump is None:
            _LOGGER.warning("跳跃请求被忽略：未配置 jump_height（需 --jump-height）")
            return False
        phase = self.jump.phase.value
        if phase == "IDLE":
            self._jump_requested = True
            return True
        if phase == "ABORT":
            ready, reason = self._jump_takeoff_ready()
            if not ready:
                _LOGGER.info("跳跃请求被拒绝：相位 ABORT 且未停稳（%s，t=%.3f s）",
                             reason, float(self.data.time))
                return False
            self.jump.reset_to_idle()
            _LOGGER.info("跳跃请求受理：相位 ABORT 已复位到 IDLE（t=%.3f s）", float(self.data.time))
            self._jump_requested = True
            return True
        _LOGGER.info("跳跃请求被拒绝：当前相位 %s（只在 IDLE 或已停稳的 ABORT 受理，t=%.3f s）",
                     phase, float(self.data.time))
        return False

    def _precheck_jump(self) -> None:
        """起跳前可行性预检：把跳跃等效力经力雅可比解算成关节力矩，与执行器限幅比较。

        用户要求"不暴力迭代、不可行就停"：这里只算一次并如实报告（超限也不擅自改目标）。
        用当前实测位形 + 压缩终点两个位形各算一次，取较大者。
        """
        config = self.jump.config
        _LOGGER.info("起跳前预检：等效力 %.1f N/腿 | 压缩终点 %.4f m | 当前腿长 %.4f m",
                     config.leg_force(), config.compression_end, float(self.dyn.leg_state()[0]))
        leg_force = config.leg_force()
        _, peak_now = self.dyn.thrust_torque((leg_force, leg_force))
        limit = float(np.max(self.dyn.limits.upper[list(self.dyn.leg_actuators)]))
        if peak_now > limit:
            _LOGGER.warning("起跳前预检：所需关节力矩峰值 %.2f N*m 超过限幅 %.2f N*m（%.0f%%）——"
                            "按方案要求如实报告，不迭代", peak_now, limit, 100.0 * peak_now / limit)
        else:
            _LOGGER.info("起跳前预检通过：关节力矩峰值 %.2f N*m <= 限幅 %.2f N*m",
                         peak_now, limit)

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
