"""GBC486 跳跃规划器：状态机 + 闭式力反解。

设计依据：``wheel_leg_mujoco/跳跃功能方案.md``（v3，用户 2026-10-07 审阅通过）。

本模块**只做纯计算**（第 15.1 条解耦）：

- 不持有 ``model`` / ``data``，不读 qpos/qvel（观测由调用方填好）；
- 不持有 MPC / WBC / 采样器 / 输出级对象；不写执行器、不推进物理、不配日志；
- **不含任何等待型定时器**：状态切换只由腿长测量量、接触测量量、或由模型闭式算出的量触发
  （用户 2026-10-07 第 7 条）。

状态机：``IDLE -> COMPRESS -> THRUST -> FLIGHT -> LANDING -> RECOVER -> IDLE``，异常进 ``ABORT``。

力的反解（用户把关系式确认为）：高度给定后

    v_takeoff = sqrt(2 g h)                      离地速度
    a         = v_takeoff^2 / (2 * com_rise)     质心加速度；com_rise = kappa * compression_stroke
    F_total   = m (g + a)                        施加在机体上的竖直等效力（整机）
    F_leg     = F_total / 2                      对称两腿，单腿

**固定压缩行程即可解出所需的力；固定力即可反解出压缩行程**（``stroke_for_force``）。

竖直力到关节力矩的换算（力雅可比）不在本模块：那需要模型几何，由
``dynamics.Gbc486Dynamics.thrust_torque`` 承担（见方案第 5.2 节）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .logging_setup import get_logger

_LOGGER = get_logger(__name__)

GRAVITY = 9.81          # [m/s^2] 标准重力，常数
# 真离地判据：轮最低点离地间隙超过 1 mm。**力的判定（接触力估计器）不能单独触发离地**——
# 蹬伸期法向力会在轮子仍压在地面时降到零（实测腿只伸了 2 mm 就被判离地），若据此断开控制器，
# 机体在最需要控姿的时刻失去控制，实测俯仰直冲到 -19.6 度（用户 2026-10-07 指出）。
# 仿真侧用几何间隙作为真值；真车上对应的旁证是 IMU 竖直比力（本适配层尚未接）。
LIFTOFF_GAP = 1.0e-3
# 落地/恢复判据的腿长邻域宽度 [m]。来源：与腿长换挡滞回常量 _LEG_LENGTH_GEAR_HYSTERESIS = 0.008 m
# 同量级（该值取网格间距 0.04 m 的 1/5），这里取 0.010 m = 腿长范围 0.29 m 的 3.4%。
LANDING_BAND = 0.010


class JumpPhase(Enum):
    """相位。字符串值与日志一致，便于事后 grep。"""

    IDLE = "IDLE"
    COMPRESS = "COMPRESS"
    COMPRESS_BRAKE = "COMPRESS_BRAKE"
    THRUST = "THRUST"
    FLIGHT = "FLIGHT"
    LANDING = "LANDING"
    RECOVER = "RECOVER"
    ABORT = "ABORT"


@dataclass(frozen=True)
class JumpConfig:
    """一跳的全部输入。模型派生的量（质量、κ、腿长范围、姿态限）由调用方算好传进来。"""

    target_height: float            # [m] 目标质心抬升高度（用户给定）
    compression_stroke: float       # [m] 压缩行程 = 加速行程（用户给定）
    leg_length_nominal: float       # [m] 标称腿长 L0
    com_rise_ratio: float           # [-] κ = 质心抬升 / 腿长行程（模型实算）
    total_mass: float               # [kg] 整机质量（模型汇总）
    leg_length_min: float           # [m] 腿长机械下限（关节余量=0 处）
    leg_length_max: float           # [m] 腿长机械上限
    pitch_limit: float              # [rad] 姿态包线（沿用 MPC 配置值，不新造阈值）
    release_ceiling: float | None = None    # [m] 释放段腿长上限，默认取 L0
    thrust_wheel_gain: float = 0.0          # [N*m/rad] 蹬伸段轮俯仰 PD 增益；0 = 不给轮力矩
    # 压缩参考（部署文档第九节；用户 2026-10-09 批准的建议值）：
    # COMPRESS 把支撑力降到半重以产生受控下压；COMPRESS_BRAKE 把支撑力升到 1.5 倍重
    # 把下降速度刹回。两者都满足接触可行域 -g < a_com_z；数值来源为用户批准，非自行试数。
    compress_acceleration_ratio: float = -0.5    # [-] a_com_z = ratio * g（负=向下）
    brake_acceleration_ratio: float = 0.5        # [-] a_com_z = ratio * g（正=支撑增强）

    @property
    def compression_end(self) -> float:
        """压缩终点腿长：``L0 - 压缩行程``。"""
        return self.leg_length_nominal - self.compression_stroke

    @property
    def release_end(self) -> float:
        """释放段腿长上限（兜底判据；正常应由接触丢失触发离地）。"""
        return self.leg_length_nominal if self.release_ceiling is None else float(self.release_ceiling)

    def takeoff_speed(self) -> float:
        """离地速度 [m/s]，由目标高度反解。"""
        return (2.0 * GRAVITY * self.target_height) ** 0.5

    def com_rise(self) -> float:
        """加速段内质心抬升量 [m]。"""
        return self.com_rise_ratio * self.compression_stroke

    def com_acceleration(self) -> float:
        """质心竖直加速度 [m/s^2]。"""
        rise = self.com_rise()
        return float("inf") if rise <= 0.0 else self.takeoff_speed() ** 2 / (2.0 * rise)

    def compress_acceleration(self) -> float:
        """COMPRESS 阶段的质心竖直加速度 [m/s^2]（负=向下）。来源：用户 2026-10-09 批准的
        profile，a = -0.5 g，即 F_normal_total = 0.5 m g，满足 -g < a < 0。"""
        return self.compress_acceleration_ratio * GRAVITY

    def brake_acceleration(self) -> float:
        """COMPRESS_BRAKE 阶段的质心竖直加速度 [m/s^2]（正=支撑增强，减速）。来源：用户
        2026-10-09 批准的 profile，a = +0.5 g，即 F_normal_total = 1.5 m g。"""
        return self.brake_acceleration_ratio * GRAVITY

    def total_force(self) -> float:
        """施加在机体上的竖直等效力（整机）[N]。"""
        return self.total_mass * (GRAVITY + self.com_acceleration())

    def leg_force(self) -> float:
        """单腿竖直等效力 [N]。"""
        return 0.5 * self.total_force()

    def thrust_time(self) -> float:
        """加速时间 [s]——导出量，不是设定值。"""
        acceleration = self.com_acceleration()
        return 0.0 if acceleration <= 0.0 else self.takeoff_speed() / acceleration

    def stroke_for_force(self, leg_force: float) -> float:
        """固定力 -> 反解所需的压缩行程 [m]（与上面互为反函数）。"""
        acceleration = 2.0 * float(leg_force) / self.total_mass - GRAVITY
        if acceleration <= 0.0 or self.com_rise_ratio <= 0.0:
            return float("inf")
        return self.takeoff_speed() ** 2 / (2.0 * acceleration * self.com_rise_ratio)


@dataclass(frozen=True)
class JumpObservation:
    """一跳观测：全部是测量量。"""

    time: float                 # [s]
    leg_length: float           # [m] 两腿腿长均值（共模，仅供非压缩相位使用）
    leg_length_rate: float      # [m/s]
    # 新部署（2026-10-09）：压缩方向/终点/机械范围判断必须用逐腿量，禁止用均值或 min 替代。
    leg_length_R: float         # [m] 右腿腿长
    leg_length_rate_R: float    # [m/s] 右腿腿长率（负=变短）
    leg_length_L: float         # [m] 左腿腿长
    leg_length_rate_L: float    # [m/s] 左腿腿长率（负=变短）
    contact: tuple              # 逐轮接触判定（算法层估计器）
    pitch: float                # [rad]
    pitch_rate: float           # [rad/s]
    # 轮最低点离地间隙 [m]（几何真值）。默认 0 = 未提供，此时**不会**判定真离地（安全方向）。
    wheel_gap: float = 0.0
    geometry_contact: tuple[bool, bool] = (False, False)
    thrust_applied: bool = False


@dataclass(frozen=True)
class JumpCommand:
    """一跳输出。调用方按这些字段装配 WBC 与输出级。"""

    phase: JumpPhase
    leg_force: float                    # [N] 施加在机体上的竖直等效力（单腿）
    leg_length_target: float | None      # [m] 腿长任务参考；None = 保持当前由 WBC 自行处理
    wheel_owned: bool                   # True = 蹬伸段轮力矩由本模块独占（WBC 轮任务失效）
    wheel_torque: float                 # [N*m] 轮俯仰指令（左右对称）
    hold_torque: bool                   # True = 输出级保持上一拍力矩（飞行段）
    disconnect: bool                    # True = 断开控制器（飞行段）
    wheel_torque: float                # [N*m] 轮俯仰指令（左右对称）
    abort_reason: str | None = None
    # WP3（2026-10-09）：本拍 WBC 质心竖直加速度目标 [m/s^2]。None = 本相位不向 WBC
    # 提供质心竖直任务（例如 FLIGHT 无接触）。来源：现有 JumpConfig 推导量 com_acceleration()。
    desired_com_acceleration_z: float | None = None

    @property
    def active(self) -> bool:
        """跳跃是否正在进行（用于"跳跃期关闭冲突功能"的判据）。"""
        return self.phase not in (JumpPhase.IDLE, JumpPhase.ABORT)


class JumpPlanner:
    """状态机 + 闭式反解。线程/节拍无关：调用方决定多快调 ``update``。"""

    def __init__(self, config: JumpConfig) -> None:
        self.config = config
        self.phase = JumpPhase.IDLE
        self.reason: str | None = None
        self.started_at = 0.0
        self.phase_since = 0.0
        # WP7（2026-10-09 部署文档）临时隔离：COMPRESS 里
        # compression_direction_lost 仍按原条件判定并记录，但不再立即 ABORT。
        # 这不是阈值修改，也不是最终修复；测试结束后必须由问题排查 agent 决定是否恢复。
        self.direction_lost_isolation_active = True
        self.direction_lost_R = False
        self.direction_lost_L = False
        self.direction_lost_isolated_R = False
        self.direction_lost_isolated_L = False
        self.transition_reason: str | None = None
        # FL4（2026-10-09）：进入 FLIGHT 后“曾真实几何离地”latch；仅 latch 置位后才允许 LANDING，
        # 避免 release_end 在仍接地时触发 FLIGHT 后立即被几何接触误判落地。
        self._airborne_confirmed = False
        self._direction_lost_seen = {"R": False, "L": False}

    # ------------------------------------------------------------------ 事件
    def start(self, time: float) -> JumpCommand:
        """收到起跳指令：进入压缩段。"""
        self.phase = JumpPhase.COMPRESS
        self.reason = None
        self.started_at = float(time)
        self.phase_since = float(time)
        self.transition_reason = "跳跃开始"
        self._airborne_confirmed = False
        config = self.config
        _LOGGER.info(
            "跳跃开始：目标高度 %.3f m | 压缩行程 %.3f m | 压缩终点 %.4f m | "
            "加速行程(质心) %.4f m | 离地速度 %.3f m/s | 等效力 %.1f N/腿 | 加速时间 %.1f ms",
            config.target_height, config.compression_stroke, config.compression_end,
            config.com_rise(), config.takeoff_speed(), config.leg_force(),
            1.0e3 * config.thrust_time())

    def _abort(self, time: float, reason: str) -> JumpCommand:
        if self.phase is not JumpPhase.ABORT:
            _LOGGER.warning("跳跃中止（t=%.3f s）：%s", time, reason)
        self.phase = JumpPhase.ABORT
        self.reason = reason
        return self._command(JumpPhase.ABORT, abort=reason)

    # ------------------------------------------------------------------ 主循环
    def update(self, obs: JumpObservation) -> JumpCommand:
        """推进一拍。返回本拍应施加的指令（纯函数，无副作用除状态机本身）。"""
        self.direction_lost_R = False
        self.direction_lost_L = False
        self.direction_lost_isolated_R = False
        self.direction_lost_isolated_L = False
        if self.phase is JumpPhase.IDLE:
            return self._command(JumpPhase.IDLE)
        if self.phase is JumpPhase.ABORT:
            return self._command(JumpPhase.ABORT, abort=self.reason)
        config = self.config
        # 关节余量硬约束（方案 3.4 节）：逐腿判断，越出机械范围即主动收手。
        for side, length in (("R", obs.leg_length_R), ("L", obs.leg_length_L)):
            if not (config.leg_length_min <= length <= config.leg_length_max):
                return self._abort(obs.time, "leg_length_out_of_range:%s(%.4f m, range [%.3f, %.3f])"
                                   % (side, length, config.leg_length_min, config.leg_length_max))
        if self.phase is JumpPhase.COMPRESS:
            # 逐腿压缩方向：条件与原保护相同（腿长率>=0 且未到终点）。
            # 约定：leg_length_rate<0 = 腿长变短、正在压缩；>=0 = 压缩已停止或反转。
            # WP7 临时隔离：条件成立时只记录侧别，不 return _abort。机械范围越界仍走上面的
            # leg_length_out_of_range 中止，不改名。
            for side, length, rate in (("R", obs.leg_length_R, obs.leg_length_rate_R),
                                       ("L", obs.leg_length_L, obs.leg_length_rate_L)):
                lost = bool(length > config.compression_end and rate >= 0.0)
                isolated = bool(lost and self.direction_lost_isolation_active)
                if side == "R":
                    self.direction_lost_R = lost
                    self.direction_lost_isolated_R = isolated
                else:
                    self.direction_lost_L = lost
                    self.direction_lost_isolated_L = isolated
                if lost and not self._direction_lost_seen[side]:
                    _LOGGER.warning(
                        "WP7 temporary isolation: compression_direction_lost:%s "
                        "t=%.3f R length=%.4f rate=%+.5f L length=%.4f rate=%+.5f; "
                        "abort suppressed, compression continues",
                        side, obs.time,
                        obs.leg_length_R, obs.leg_length_rate_R,
                        obs.leg_length_L, obs.leg_length_rate_L)
                self._direction_lost_seen[side] = lost
            # 制动触发：逐腿算停止距离，任一侧接近终点即进入制动（保护性，避免任一侧越过）。
            brake = config.brake_acceleration()
            trigger = False
            for length, rate in ((obs.leg_length_R, obs.leg_length_rate_R),
                                 (obs.leg_length_L, obs.leg_length_rate_L)):
                remaining = length - config.compression_end
                descent_speed = max(0.0, -float(rate))    # [m/s] 向下为正
                stop_distance = (descent_speed ** 2) / (2.0 * brake) if brake > 0.0 else float("inf")
                if remaining <= max(stop_distance, 0.0):
                    trigger = True
            if trigger:
                self._enter(JumpPhase.COMPRESS_BRAKE, obs,
                            "进入压缩制动（右 %.4f/%.3f，左 %.4f/%.3f）"
                            % (obs.leg_length_R, obs.leg_length_rate_R,
                               obs.leg_length_L, obs.leg_length_rate_L))
            return self._command(self.phase)
        if self.phase is JumpPhase.COMPRESS_BRAKE:
            # 双腿都满足"到达终点且下降已制动"才切 THRUST（右且左）。
            if (obs.leg_length_R <= config.compression_end and obs.leg_length_rate_R >= 0.0
                    and obs.leg_length_L <= config.compression_end and obs.leg_length_rate_L >= 0.0):
                self._enter(JumpPhase.THRUST, obs,
                            "压缩完成且下降已制动（右 %.4f/%+.3f，左 %.4f/%+.3f）"
                            % (obs.leg_length_R, obs.leg_length_rate_R,
                               obs.leg_length_L, obs.leg_length_rate_L))
                return self._command(JumpPhase.THRUST)
            return self._command(self.phase)
        if self.phase is JumpPhase.THRUST:
            # 状态保险：蹬伸命令必须已经实际写入执行器，之后才允许评估离地。
            if not obs.thrust_applied:
                return self._command(self.phase)
            # 真离地判据：几何间隙张开（不是"法向力降到零"）。力的判定会误报——蹬伸期轮子仍
            # 压在地面时法向力也会掉到零；据此断开控制器会让机体在最需要控姿时失控（已实测）。
            if obs.wheel_gap > LIFTOFF_GAP:
                self._enter(JumpPhase.FLIGHT, obs,
                            "轮地间隙张开到 %.1f mm（几何离地，腿长 %.4f m，腿长率 %+.3f m/s）"
                            % (1.0e3 * obs.wheel_gap, obs.leg_length, obs.leg_length_rate))
            elif obs.leg_length >= config.release_end:
                # FL3 更新版（2026-10-09）：恢复 release_end 触发。它是推离完成的标志，也是轮真正
                # 离开地面的起点（实测：释放段腿伸展量≈base 上升量，wheel_gap 全程＜0.1 mm）。
                self._enter(JumpPhase.FLIGHT, obs,
                            "推离完成（腿到达释放上限 %.4f m，gap=%.2f mm，轮即将离地）"
                            % (config.release_end, 1.0e3 * obs.wheel_gap))
            return self._command(self.phase)
        if self.phase is JumpPhase.FLIGHT:
            # FL4：先置位“曾真实几何离地”latch；只有 latch 置位后，几何接触才允许转 LANDING。
            if obs.wheel_gap > LIFTOFF_GAP:
                self._airborne_confirmed = True
            if self._airborne_confirmed and all(obs.geometry_contact) and obs.wheel_gap <= 0.0:
                self._enter(JumpPhase.LANDING, obs,
                            "几何接触恢复（落地腿长 %.4f m，间隙 %.2f mm，姿态 %.2f deg）"
                            % (obs.leg_length, 1.0e3 * obs.wheel_gap, obs.pitch * 57.2957795))
            return self._command(self.phase)
        if self.phase is JumpPhase.LANDING:
            if (all(obs.geometry_contact) and all(obs.contact)
                    and abs(obs.leg_length - config.leg_length_nominal) <= LANDING_BAND):
                self._enter(JumpPhase.RECOVER, obs, "腿长回到标称邻域（%.4f m）" % obs.leg_length)
            return self._command(self.phase)
        # RECOVER
        if (all(obs.geometry_contact) and all(obs.contact)
                and abs(obs.pitch) <= config.pitch_limit
                and abs(obs.leg_length - config.leg_length_nominal) <= LANDING_BAND):
            self._enter(JumpPhase.IDLE, obs,
                        "姿态与腿长进入正常包线（俯仰 %.2f deg），交还控制器"
                        % (obs.pitch * 57.2957795))
        return self._command(self.phase)

    def _enter(self, phase: JumpPhase, obs: JumpObservation, why: str) -> None:
        _LOGGER.info("跳跃相位 %s -> %s（t=%.3f s）：%s",
                     self.phase.value, phase.value, obs.time, why)
        self.transition_reason = why
        self.phase = phase
        self.phase_since = obs.time

    # ------------------------------------------------------------------ 装配
    def _command(self, phase: JumpPhase, abort: str | None = None) -> JumpCommand:
        """按相位装配输出。**力与腿长目标只在这里定义**，别处不得再拼。"""
        config = self.config
        common = dict(phase=phase, abort_reason=abort)
        acceleration = config.com_acceleration()
        if not math.isfinite(acceleration):
            acceleration = None
        if phase is JumpPhase.COMPRESS:
            # 阶段九：压缩目标改为用户批准的半重支撑参考（满足 -g < a < 0），不再取负 THRUST 加速度。
            return JumpCommand(leg_force=-config.leg_force(), leg_length_target=None,
                               wheel_owned=False, wheel_torque=0.0, hold_torque=False,
                               disconnect=False,
                               desired_com_acceleration_z=config.compress_acceleration(), **common)
        if phase is JumpPhase.COMPRESS_BRAKE:
            # 阶段九：制动把支撑力升到 1.5 倍重，把下降速度刹回，再切 THRUST。
            return JumpCommand(leg_force=-config.leg_force(), leg_length_target=None,
                               wheel_owned=False, wheel_torque=0.0, hold_torque=False,
                               disconnect=False,
                               desired_com_acceleration_z=config.brake_acceleration(), **common)
        if phase in (JumpPhase.THRUST,):
            # WP3：释放由 WBC 统一求解（质心竖直任务），不再叠加 tau_jump。
            # 目标来源：JumpConfig.com_acceleration()，符号为正表示质心向上。
            return JumpCommand(leg_force=config.leg_force(), leg_length_target=None,
                               wheel_owned=True,
                               wheel_torque=config.thrust_wheel_gain, hold_torque=False,
                               disconnect=False,
                               desired_com_acceleration_z=acceleration, **common)
        if phase is JumpPhase.FLIGHT:
            # 飞行段：断开控制器，输出级保持上一拍力矩（用户 D4：力矩保持）
            return JumpCommand(leg_force=0.0, leg_length_target=None,
                               wheel_owned=False, wheel_torque=0.0, hold_torque=True,
                               disconnect=True, **common)
        if phase in (JumpPhase.LANDING, JumpPhase.RECOVER):
            # 落地与恢复：腿长回标称，其余交还 WBC（跳跃期"关闭冲突功能"在此结束）
            return JumpCommand(leg_force=0.0, leg_length_target=config.leg_length_nominal,
                               wheel_owned=False, wheel_torque=0.0, hold_torque=False,
                               disconnect=False, **common)
        if phase is JumpPhase.ABORT:
            return JumpCommand(leg_force=0.0, leg_length_target=config.leg_length_nominal,
                               wheel_owned=False, wheel_torque=0.0, hold_torque=False,
                               disconnect=False, **common)
        # IDLE：不干预
        return JumpCommand(leg_force=0.0, leg_length_target=None, wheel_owned=False,
                           wheel_torque=0.0, hold_torque=False, disconnect=False, **common)
