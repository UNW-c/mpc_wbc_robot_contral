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
import math
import time

import numpy as np

from . import _circleg  # noqa: F401
from .logging_setup import get_logger  # noqa: E402
from .leg_angle_anchor import WORLD_TARGET_DEFAULT, phi_body_reference, phi_eq  # noqa: E402
from .dynamics import leg_combination  # noqa: E402
from . import two_mass as _two_mass  # noqa: E402
from .jump import GRAVITY  # noqa: E402
from Algorithm.wbc import ActuatorLimits  # noqa: E402
from Algorithm.wbc.projection import TaskLevel  # noqa: E402

_LOGGER = get_logger(__name__)

# WP8（2026-10-09）：压缩/制动/释放相位让质心竖直任务在 WLS 中占主导的具名权重。
# 来源：用户 2026-10-09 批准（WP8 部署文档第三节）。只在跳跃相位覆盖，GROUND 不生效。
JUMP_COM_VERTICAL_WEIGHT = 5000.0

# FL2（2026-10-09）：飞行段主动收腿的 PD 参数与收腿下限余量。
# 来源：用户 2026-10-09 批准（FL2 部署文档第三节）：kp=3000 N/m、kd=150 N*s/m、
# 收腿下限 = leg_length_min + 0.005 m。
FLIGHT_RETRACT_KP = 3000.0        # [N/m]
FLIGHT_RETRACT_KD = 150.0         # [N*s/m]
# FL19（2026-10-10 用户指定）：飞行段收腿下限改为绝对腿长 0.20 m（每腿）。
FLIGHT_RETRACT_FLOOR = 0.16       # [m] LAND2：0.20 -> 0.16（SW1 唯一过全硬约束且 T_settle 收敛组）
# FL4（2026-10-09，用户批准）：上升收腿的参考速率限幅与力矩上限。
FLIGHT_RETRACT_RATE_LIMIT = 1.5   # [m/s] 收腿参考速率上限
FLIGHT_EXTEND_RATE_LIMIT = 1.5    # [m/s] 伸腿参考速率上限（FL9，与收腿同值同理由）
FLIGHT_RETRACT_TORQUE_LIMIT = 36.0  # [N*m] 上升收腿力矩上限（=40 的 90%）
# FL10（2026-10-10）：phi 参考改世界系。物理含义：与站立静平衡一致（质心落在轮接触点上）。
# R1（2026-10-17）：世界系摆角常量与 phi_eq 已并入 leg_angle_anchor.py（唯一来源），此处不再定义。
KP_PHI = 20.9             # [N*m/rad] FL12 新增益（用户批准）
KD_PHI = 1.67             # [N*m*s/rad] FL12 新增益（用户批准）
# FL13（2026-10-10）：启用耦合前馈，c 取测试agent专项实测值（对称双施加、共模 phidd 过零）。
# 实测 c_opt 区间 -0.037~-0.063，取 -0.052；文档原 +0.15（单腿口径）作废。
FEEDFORWARD_COUPLING = -0.052    # [-] F_phi += c*F_L（仅飞行下落段）
# FL21（2026-10-12 用户批准）：上升->下落交接处的 F_phi 启用斜坡（单点）。
# 依据：phi 环带宽 omega=25 rad/s -> 时间常数 1/25 = 40 ms（部署文档 FL21 三）。
# 计时起点为“进入 falling 的那一刻”；切换后前 T_ramp 内 F_phi 乘 (0->1)。FL22 起上升段也启用。
FLIGHT_PHI_RAMP = 0.040    # [s]


def _leg_axis_torque(dynamics, force_per_leg) -> np.ndarray:
    """逐腿轴向力（F_phi=0）-> 执行器力矩；见 ``_leg_wrench_torque``。"""
    return _leg_wrench_torque(dynamics, force_per_leg, np.zeros(2))


def _leg_wrench_torque(dynamics, force_L, force_phi) -> np.ndarray:
    """逐腿 (F_L, F_phi) 力旋量 -> 执行器力矩：``tau = J^T @ [F_L, F_phi]``，
    并 clamp 到执行器限幅。FL6：飞行段用 F_phi 稳定摆角。"""
    from . import robot as R  # 延迟导入，避免模块级循环依赖
    dynamics.leg_state()  # 惰性建立解析运动学
    torque = np.zeros(dynamics.model.nu)
    for side, leg in enumerate(R.LEGS):
        kin = dynamics._leg_kinematics[leg.name]
        angles = {name: float(dynamics.data.qpos[dynamics.model.joint(name).qposadr[0]])
                  for name in (leg.front_joint, leg.rear_joint) + leg.passive_joints}
        tau = kin.jacobian(angles).T @ np.array([float(force_L[side]), float(force_phi[side])])
        for offset, actuator in enumerate(dynamics.leg_actuators[2 * side:2 * side + 2]):
            torque[actuator] = tau[offset]
    return np.clip(torque, dynamics.limits.lower, dynamics.limits.upper)


def _pd_force(l_ref, leg_position, leg_velocity) -> np.ndarray:
    """同一组腿长 PD：``F_L = kp*(l_ref - L) - kd*Ldot``（腿长顺序 [L_R,phi_R,L_L,phi_L]）。"""
    forces = []
    for side in range(2):
        index = 2 * side
        forces.append(FLIGHT_RETRACT_KP * (float(l_ref) - float(leg_position[index]))
                      - FLIGHT_RETRACT_KD * float(leg_velocity[index]))
    return np.asarray(forces, dtype=float)


def _pd_force_v(l_ref, v_ref, leg_position, leg_velocity) -> np.ndarray:
    """FL7：带参考腿速的腿长 PD：``F_L = kp*(l_ref - L) + kd*(v_ref - Ldot)``。"""
    forces = []
    for side in range(2):
        index = 2 * side
        forces.append(FLIGHT_RETRACT_KP * (float(l_ref) - float(leg_position[index]))
                      + FLIGHT_RETRACT_KD * (float(v_ref) - float(leg_velocity[index])))
    return np.asarray(forces, dtype=float)


def _phi_force(leg_position, leg_velocity, pitch, l_ref=None, world_offset=0.0) -> np.ndarray:
    """FL9/FL11/FL12 下落段摆角 PD（世界系参考）：
    ``F_phi = kp_phi*(phi_body_ref - phi) - kd_phi*phi_dot``；
    ``phi_body_ref = world_ref + pitch``；``world_ref = phi_eq(l_ref)``（FL12 角度前馈），
    ``l_ref=None`` 时退回常数 ``WORLD_TARGET_DEFAULT``。"""
    world_ref = (WORLD_TARGET_DEFAULT if l_ref is None else phi_eq(l_ref)) + float(world_offset)
    phi_body_ref = phi_body_reference(world_ref, pitch)
    forces = []
    for side in range(2):
        index = 2 * side + 1
        forces.append(KP_PHI * (phi_body_ref - float(leg_position[index]))
                      - KD_PHI * float(leg_velocity[index]))
    return np.asarray(forces, dtype=float)


def flight_leg_retract_torque(dynamics, leg_position, leg_velocity,
                              leg_length_min, leg_length_max) -> np.ndarray:
    """FL2 飞行段主动收腿力矩：``l_ref = FLIGHT_RETRACT_FLOOR``（FL19 起为绝对 0.20 m）。"""
    l_ref = min(max(FLIGHT_RETRACT_FLOOR,
                    float(leg_length_min)), float(leg_length_max))
    return _leg_axis_torque(dynamics, _pd_force(l_ref, leg_position, leg_velocity))


def flight_leg_profile_torque(dynamics, leg_position, leg_velocity, vertical_velocity,
                              leg_length_min, leg_length_max, nominal_length,
                              retract_state=None, dt=0.0, pitch=0.0, preswing=0.0):
    """FL3/4/7/8/9 飞行腿剖面：上升（vz>0）收腿；下落（vz<0）速率受限伸腿 + phi 回复。

    状态为六元组 ``(mode, L_ref, v_ref, L_entry, ref_accel, phi_elapsed)``，``mode`` ∈ {"rising","falling"}。
    - 上升：参考自首步起按 RATE 下收（无上穿挡位）；力矩 clamp 36。
    - 下落（FL9）：伸腿参考速率受限（L_ref 以 v_ref 向 nominal 收敛，v_ref 由 ref_accel 斜坡）；
      模式切换时捕获上一状态保持 (L_ref,v_ref) 连续。
    - FL21/FL22：``phi_elapsed`` 自飞行剖面首次进入起累计；F_phi 乘启用斜坡因子（``FLIGHT_PHI_RAMP`` 秒内 0->1），
      **上升与下落两段都启用**（FL22：把阻尼提前到上升段）。
    返回 ``(torque, new_state)``。
    """
    lo = float(leg_length_min)
    hi = float(leg_length_max)
    target_retract = min(max(FLIGHT_RETRACT_FLOOR, lo), hi)
    target_extend = min(max(float(nominal_length), lo), hi)
    falling = float(vertical_velocity) < 0.0
    want_mode = "falling" if falling else "rising"
    larger = 0 if float(leg_position[0]) >= float(leg_position[2]) else 2
    if retract_state is not None and str(retract_state[0]) == want_mode:
        mode = want_mode
        l_ref = float(retract_state[1])
        v_ref = float(retract_state[2])
        l_entry = float(retract_state[3])
        ref_accel = float(retract_state[4])
        phi_elapsed = float(retract_state[5]) if len(retract_state) > 5 else 0.0
    else:
        # 捕获：首次进飞行，或上升↔下落切换。保持 (L_ref,v_ref) 连续以消除单步阶跃。
        l_entry = max(float(leg_position[0]), float(leg_position[2]))
        if retract_state is not None:
            l_ref = float(retract_state[1])
            v_ref = float(retract_state[2])
            ref_accel = float(retract_state[4])
            # 模式切换：phi 计数延续（FL22 下 phi 全程启用，不因切换重新起斜坡）。
            phi_elapsed = float(retract_state[5]) if len(retract_state) > 5 else 0.0
        else:
            l_ref = l_entry
            v_ref = float(leg_velocity[larger])
            t_apex = max(abs(float(vertical_velocity)) / GRAVITY, 1.0e-6)
            ref_accel = (abs(v_ref) + FLIGHT_RETRACT_RATE_LIMIT) / t_apex
            phi_elapsed = 0.0     # 飞行剖面首次进入：phi 启用斜坡从此计时（FL22）
        mode = want_mode
    step_limit = ref_accel * float(dt)
    if mode == "rising":
        l_ref = max(target_retract, l_ref - FLIGHT_RETRACT_RATE_LIMIT * float(dt))
        # RAMP1：速度参考必须等于位置参考的斜率，否则 kd 项与 kp 项对消（OBS1 实测净力均值 −0.9 N）
        v_ref = (-FLIGHT_RETRACT_RATE_LIMIT
                 if l_ref > target_retract + 1.0e-9 else 0.0)
    else:
        delta = FLIGHT_EXTEND_RATE_LIMIT - v_ref
        if abs(delta) <= step_limit:
            v_ref = FLIGHT_EXTEND_RATE_LIMIT
        else:
            v_ref += step_limit * (1.0 if delta > 0 else -1.0)
        l_ref = min(target_extend, l_ref + v_ref * float(dt))
    l_ref = min(max(l_ref, lo), hi)
    forces = _pd_force_v(l_ref, v_ref, leg_position, leg_velocity)
    # FL22（2026-10-12 用户批准）：上升/下落**两段都算 phi**（把阻尼提前到上升段），
    # 沿用 FL21 的启用斜坡（自飞行剖面进入起 FLIGHT_PHI_RAMP 秒内 0->1），削掉启用阶跃。
    phi_ramp = min(1.0, max(0.0, phi_elapsed / FLIGHT_PHI_RAMP))
    # P24：触地前腿角终端预置。伸腿进度 frac 0->1 时把世界角目标摆到 preswing，
    # 让触地点从质心前方移到质心后方（低头力矩变抬头力矩）。上升段 frac=0，不生效。
    _span = max(target_extend - target_retract, 1.0e-6)
    _frac = min(1.0, max(0.0, (l_ref - target_retract) / _span))
    phi = (_phi_force(leg_position, leg_velocity, pitch, l_ref,
                      world_offset=float(preswing) * _frac)
           + FEEDFORWARD_COUPLING * forces) * phi_ramp
    phi_elapsed += float(dt)
    torque = _leg_wrench_torque(dynamics, forces, phi)
    if mode == "rising":
        torque = np.clip(torque, -FLIGHT_RETRACT_TORQUE_LIMIT, FLIGHT_RETRACT_TORQUE_LIMIT)
    return torque, (mode, l_ref, v_ref, l_entry, ref_accel, phi_elapsed)


# ---------------------------------------------------------------------------
# 双质量块飞行规划（部署文档 v6 第 3.2/3.3 节）
# ---------------------------------------------------------------------------
# 为什么用一条轨迹：两块之间只有一条腿力 F，任何"收腿段/伸腿段"的分段都会在
# 段边界引入力或加速度不连续，再用下一段去补偿（v5 的错误，见文档 3.6）。
# 这里只规划轮子绝对高度 z1(t) 的一条 C2 曲线，腿长 L = z2 - z1 是差值。
DUAL_MASS_NORM = 64.0        # f(u) = 64*u^3*(1-u)^3，使 f(0.5)=1（文档 3.2 节给定）


@dataclass(frozen=True)
class DualMassFlightPlan:
    """飞行段计划的捕获量（每跳算一次，逐拍只推进 ``t``）。"""
    t: float            # [s] 距捕获帧的时间
    T: float            # [s] 飞行总时长（由触地腿长解出）
    h: float            # [m] z1 轨迹幅值（由最小腿长解出）
    L0: float           # [m] 捕获帧腿长
    v0: float           # [m/s] 捕获帧机体竖直速度
    ratio: float        # [-] m1/m2
    phi_elapsed: float  # [s] phi 启用斜坡计时
    flight: object | None = None   # 10.54 D：方案 (b) 的飞行计划（T/h/l_ref/phi_theta）
    phi_ref: float = 0.0      # [rad] H-hold 的切向参考（世界系，增量积分）
    h_target: float = 0.0     # [kg*m^2/s] 角动量保持目标（捕获帧实测总角动量 y 分量）


def dual_mass_masses(dynamics) -> tuple[float, float]:
    """(m1, m2)：m1 = 两轮 + 关节 jGH/jOP 所在体（GH/OP），m2 = 整机 - m1。

    数值取自 MJCF（0.322*2 + 1.640*2 = 3.924 kg，整机 14.546 kg），与文档 1.1 一致。
    """
    from . import robot as R
    model = dynamics.model
    lower = 0.0
    for leg in R.LEGS:
        lower += float(model.body_mass[model.body(leg.wheel_body).id])
        lower += float(model.body_mass[int(model.joint(leg.passive_joints[0]).bodyid[0])])
    total = float(model.body_mass.sum())
    return lower, total - lower


def _dual_mass_shape(u: float) -> tuple[float, float, float]:
    """返回 (f, f', f'')：f = 64*u^3*(1-u)^3，端点 f=f'=f''=0。"""
    u = min(1.0, max(0.0, float(u)))
    cubic = u * u * u * (1.0 - u) ** 3
    first = 3.0 * u * u * (1.0 - u) ** 2 * (1.0 - 2.0 * u)
    second = 6.0 * u * (1.0 - u) * ((1.0 - 2.0 * u) ** 2 - u * (1.0 - u))
    return DUAL_MASS_NORM * cubic, DUAL_MASS_NORM * first, DUAL_MASS_NORM * second


def total_angular_momentum_y(dynamics) -> float:
    """整机绕总质心的角动量 y 分量 [kg*m^2/s]（飞行段无外力矩，守恒）。

    ``mj_objectVelocity`` 的线速度部分已按刚体质心给出（实测该口径下飞行段
    H_y = -0.349 全程不变），所以直接用它，不再额外叠加 ω×(xipos-xpos)。
    """
    import mujoco
    model, data = dynamics.model, dynamics.data
    centre = data.subtree_com[0]
    total = np.zeros(3)
    for index in range(1, model.nbody):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, index, velocity, 0)
        omega = velocity[:3]
        rot = data.ximat[index].reshape(3, 3)
        inertia = rot @ np.diag(model.body_inertia[index]) @ rot.T
        total += inertia @ omega + model.body_mass[index] * np.cross(
            data.xipos[index] - centre, velocity[3:])
    return float(total[1])


def dual_mass_flight_plan(m1: float, m2: float, L0: float, v0: float,
                          L_td: float, L_tuck: float) -> tuple[float, float]:
    """解 (T, h)（文档 3.3 节）：T 由触地腿长，h 由最小腿长。无实数解返回 (nan, nan)。"""
    ratio = m1 / m2
    accel = (1.0 + ratio) * GRAVITY
    disc = v0 * v0 - 2.0 * accel * (L_td - L0)
    if disc < 0.0:
        return float("nan"), float("nan")
    T = (v0 + math.sqrt(disc)) / accel
    half = v0 * T / 2.0 - accel * T * T / 8.0
    h = (L0 + half - L_tuck) / (1.0 + ratio)
    return T, h


def dual_mass_flight_torque(dynamics, leg_position, leg_velocity, vertical_velocity,
                            dt, pitch=0.0, plan=None,
                            leg_length_nominal=0.28, hold_momentum=False,
                            two_mass_table=None, landing_length=None):
    """飞行段出力矩：``F = -m1*(z1''+g)`` + 轴向 PD，``tau = J^T [F_L, F_phi]``。

    phi 通道沿用 KP_PHI/KD_PHI + FEEDFORWARD_COUPLING 与 40 ms 启用斜坡（世界系参考
    = ``phi_eq(L_ref)``）。返回 ``(torque, new_plan)``。
    """
    m1, m2 = dual_mass_masses(dynamics)
    two_flight = None
    if plan is None and two_mass_table is not None:
        # 10.54 D：方案 (b)。T 由质心弹道与 D_td 解出，h 由最小腿长解出；
        # l_ref 由 D(L) 表反查，world_ref 用流形 phi_theta。
        two = _two_mass.TwoMassModel(dynamics.model, dynamics.data)
        com_row, _com_bias, com_z_abs, _v = dynamics.com_vertical_row()
        lowest = float(np.min(_two_mass.R.wheel_contact_points(dynamics.model,
                                                              dynamics.data)[:, 2]))
        vc = float(com_row @ dynamics.data.qvel)
        two_flight = _two_mass.FlightPlan(
            two_mass_table, vc, com_z_abs - lowest, FLIGHT_RETRACT_FLOOR,
            D_td_flight=(None if landing_length is None
                         else _two_mass.D_of_L(two_mass_table, float(landing_length))))
        plan = DualMassFlightPlan(0.0, two_flight.T, two_flight.h, 0.0, vc,
                                  two.mass_lower / two.mass_upper, 0.0,
                                  flight=two_flight)
    if plan is None:
        L0 = max(float(leg_position[0]), float(leg_position[2]))
        v0 = float(vertical_velocity)
        T, h = dual_mass_flight_plan(m1, m2, L0, v0, float(leg_length_nominal),
                                     FLIGHT_RETRACT_FLOOR)
        # S5-a（文档 10.24 D）：切向参考改为"保持离地时的总角动量"：
        #   phi' = H/(m1*L^2)，phi_ref 每拍增量积分（F2：不一次性积整段）。
        _seed = 0 if float(leg_position[0]) >= float(leg_position[2]) else 2
        plan = DualMassFlightPlan(
            0.0, T, h, L0, v0, m1 / m2, 0.0,
            phi_ref=float(leg_position[_seed + 1]) - float(pitch),
            h_target=total_angular_momentum_y(dynamics) if hold_momentum else 0.0)
    ratio = plan.ratio
    u = 0.0 if plan.T <= 0.0 else min(1.0, plan.t / plan.T)
    shape, shape_d, shape_dd = _dual_mass_shape(u)
    z1 = plan.h * shape                                  # [m] 轮子相对捕获帧高度
    z1_d = (plan.h * shape_d / plan.T) if plan.T > 0.0 else 0.0
    z1_dd = (plan.h * shape_dd / (plan.T * plan.T)) if plan.T > 0.0 else 0.0
    force_plan = -m1 * (z1_dd + GRAVITY)                 # [N] 两腿合力计划值，正 = 撑开
    travel = plan.v0 * plan.t - (1.0 + ratio) * GRAVITY * plan.t * plan.t / 2.0
    l_ref = plan.L0 + travel - (1.0 + ratio) * z1        # [m] L = z2 - z1
    v_ref = plan.v0 - (1.0 + ratio) * GRAVITY * plan.t - (1.0 + ratio) * z1_d  # [m/s] = dL_ref/dt
    world_offset_plan = 0.0
    if getattr(plan, "flight", None) is not None:
        # 10.54 D：方案 (b) 的计划量（l_ref 由 D(L) 表反查，world_ref 用流形 phi_theta）
        sample = plan.flight.sample(plan.t)
        l_ref = sample["l_ref"]
        v_ref = sample["v_ref"]
        force_plan = -m1 * (sample["z1_dd"] + GRAVITY)
        _k_theta = float(plan.flight.table.get("dtheta_dphi", 1.0))   # 10.68
        world_offset_plan = (sample["phi_theta"] - phi_eq(float(l_ref))
                             - float(pitch) * (1.0 - 1.0 / _k_theta))
    # 第 10.6 节：纯前馈会把误差积分成腿长漂移，补上沿用现有增益的反馈项
    # （3000/150 即 flight 剖面的 FLIGHT_RETRACT_KP/KD，按文档字面：F_cmd 为两腿合力，
    # 用较大腿的 L 与 Ldot 计算，再 /2 分给两腿）。
    larger = 0 if float(leg_position[0]) >= float(leg_position[2]) else 2
    force_total = (force_plan
                   + FLIGHT_RETRACT_KP * (float(l_ref) - float(leg_position[larger]))
                   + FLIGHT_RETRACT_KD * (float(v_ref) - float(leg_velocity[larger])))
    forces = np.full(2, force_total / 2.0)
    phi_ramp = min(1.0, max(0.0, plan.phi_elapsed / FLIGHT_PHI_RAMP))
    # S5-a：切向参考由"保持 H"解算（phi' = H/(m1*L_ref^2)），PD 增益不动。
    world_offset = (plan.phi_ref - phi_eq(float(l_ref))) if hold_momentum else world_offset_plan
    phi = (_phi_force(leg_position, leg_velocity, pitch, float(l_ref), world_offset=world_offset)
           + FEEDFORWARD_COUPLING * forces) * phi_ramp
    torque = _leg_wrench_torque(dynamics, forces, phi)
    next_phi_ref = plan.phi_ref
    if hold_momentum:
        radius = max(float(l_ref), 1.0e-6)
        next_phi_ref = plan.phi_ref + plan.h_target / (m1 * radius * radius) * float(dt)
    return torque, replace(plan, t=plan.t + float(dt), phi_elapsed=plan.phi_elapsed + float(dt),
                           phi_ref=next_phi_ref)


def landing_buffer_torque(dynamics, leg_position, leg_velocity,
                          leg_length_min, leg_length_max, nominal_length,
                          pitch=0.0) -> np.ndarray:
    """FL3/FL9 落地缓冲：同一组腿长 PD 对 L0 做弹簧阻尼（受压时向 L0 推回）。

    说明：文档只给“从 L0 压缩到目标压缩量”与同一组 PD，未明写 l_ref；按弹簧阻尼
    缓冲的物理，l_ref 取标称 L0（若取 0.197 会产生向下的腿力，与缓冲相反）。
    FL11：叠加 `F_phi`（同一组 PD、同一世界系参考），避免接地后 phi 无控被地面带着转。
    """
    l_ref = min(max(float(nominal_length), float(leg_length_min)), float(leg_length_max))
    return _leg_wrench_torque(dynamics, _pd_force(l_ref, leg_position, leg_velocity),
                              _phi_force(leg_position, leg_velocity, pitch))


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
              timings: dict, counters: dict):
    """控制级 A：求解一次 MPC，返回 ``(solution, 本拍指令)``。

    与 WBC 无关：不等 WBC、不读 WBC 的力矩。
    """
    current = command(time_s) if callable(command) else command
    start = time.perf_counter()
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
              leg_length_thrust: bool = False, landing_window: bool = False,
              mpc_force_feedforward: bool = False, wheel_radius: float | None = None,
              thrust_momentum: bool = False, k_momentum: float = 0.0,
              k_momentum_integral: float = 0.0,
              momentum_state: dict | None = None,
              two_mass_state: dict | None = None,
              two_mass_tables: dict | None = None,
              landing_state: dict | None = None) -> WbcOutcome:
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
    # ANCHOR（2026-10-17，用户同意）：sample_override 优先。落地后 jump 已不活动，但锚定目标值
    # 仍需注入 WBC，所以放行；sample_override is None 时两条分支与改动前完全一致
    # （jump.active -> None；否则 sample_raw），零副作用由 t<t_anchor 与 S1 逐位全 0 验收。
    if sample_override is not None:
        sample = sample_override
    elif jump is not None and jump.active:
        sample = None
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
                                  angle_L_reference=holds.get("angle_L"),
                                  world_angle_anchor=landing_window)
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
    # ---- 文档 10.54 C2 / 10.56：THRUST 段换成两块水平计划 + 竖向 com_z ----
    if (two_mass_state is not None and two_mass_tables is not None
            and jump is not None and jump.active and jump.phase.value == "THRUST"):
        table = two_mass_tables
        two = _two_mass.TwoMassModel(dynamics.model, dynamics.data)
        p1, p2 = two.block_state()
        j1, j2 = two.block_jacobians()
        com_row, com_bias, com_z_abs, _com_vz = dynamics.com_vertical_row()
        lowest = float(np.min(_two_mass.R.wheel_contact_points(
            dynamics.model, dynamics.data)[:, 2]))
        plan = two_mass_state.get("plan")
        if plan is None:
            # 10.62 E1：竖向五次 + 分离端点；T 由一维求解给出（不是设定值）。
            # 入口加速度取**实测** a_com（row@qacc + bias），不是常数目标。
            _acc_meas = float(com_row @ dynamics.data.qacc + com_bias)
            plan = _two_mass.TakeoffPlan(
                table, float(com_z_abs - lowest), float(com_row @ dynamics.data.qvel), _acc_meas,
                float(p2[0] - p1[0]), float((j2[0] - j1[0]) @ dynamics.data.qvel),
                float(table["M"]),
                float(np.sqrt(2.0 * GRAVITY * float(jump_config.target_height))))
            two_mass_state["plan"] = plan
            two_mass_state["t0"] = float(time_s)
            _LOGGER.info("S9 THRUST 计划：T=%.4f s、N 峰 %.1f N（计划时钟起点 t=%.3f）"
                         "L'(T)=%+.3f phi'(T)=%+.3f",
                         plan.T, plan.n_peak, float(time_s), plan.ldot_end, plan.phidot_end)
        elapsed = min(float(time_s) - float(two_mass_state["t0"]), plan.T)
        zc_plan, vcz_plan, acc_plan = plan.zc(elapsed), plan.vcz(elapsed), plan.acc(elapsed)
        dx_plan, dx_rate_plan, dx_acc_plan = plan.dx(elapsed), plan.dx_rate(elapsed), plan.dx_acc(elapsed)
        kp_h, kd_h = tasks.gains.height_gains
        kp_r, kd_r = tasks.gains.regularize_gains
        zc_now = float(com_z_abs - lowest)
        vcz_now = float(com_row @ dynamics.data.qvel)
        dx_now = float(p2[0] - p1[0])
        dx_rate_now = float((j2[0] - j1[0]) @ dynamics.data.qvel)
        dx_bias = float(_two_mass.dx_rel_bias(dynamics))
        com_task = TaskLevel(np.atleast_2d(np.asarray(com_row, dtype=float)),
                             np.array([acc_plan + kp_h * (zc_plan - zc_now)
                                       + kd_h * (vcz_plan - vcz_now)]),
                             np.array([float(com_bias)]), name="com_z")
        dx_task = TaskLevel(np.atleast_2d((j2[0] - j1[0])),
                            np.array([dx_acc_plan + kp_h * (dx_plan - dx_now)
                                      + kd_h * (dx_rate_plan - dx_rate_now)]),
                            np.array([dx_bias]), name="dx_rel")
        leg = dynamics.leg_state()
        leg_rows_raw, leg_bias_raw = feedback.rows.leg
        diff_rows, diff_bias = leg_combination(feedback.rows, -1.0)
        common_rows, common_bias = leg_combination(feedback.rows, +1.0)
        # D1（10.58）：L_d 是**差模**腿长（leg_combination(rows,-1)[0]），目标把两块/两腿的腿长差拉回 0。
        # 第一版误写成共模腿长回 0.28（腿长当规划量），与 com_z 抢同一纵向自由度。
        length_diff = 0.5 * (float(leg[0]) - float(leg[2]))
        length_diff_rate = float(diff_rows[0] @ dynamics.data.qvel)
        phi_diff = 0.5 * (float(leg[1]) - float(leg[3]))
        phi_diff_rate = float(diff_rows[1] @ dynamics.data.qvel)
        # D2：phi_d 行取负（与 wbc.py 的 angle 任务同一符号约定），目标同样把差模拉到 0。
        l_d_task = TaskLevel(np.atleast_2d(diff_rows[0]),
                             np.array([-kp_h * length_diff - kd_h * length_diff_rate]),
                             np.array([float(diff_bias[0])]), name="L_d")
        phi_d_task = TaskLevel(np.atleast_2d(-diff_rows[1]),
                               np.array([kp_r * phi_diff - kd_r * (-phi_diff_rate)]),
                               np.array([float(-diff_bias[1])]), name="phi_d")
        kept = tuple(task for task in motion
                     if task.name not in ("length_R", "length_L", "angle_R", "angle_L",
                                          "forward_velocity"))
        motion = kept + (com_task, dx_task, l_d_task, phi_d_task)
    # ---- 文档 10.68：LANDING 段两块计划（与 THRUST 任务组对称，经 WBC 执行到计划终点）----
    if (landing_state is not None and two_mass_tables is not None
            and landing_state.get("plan") is not None
            and jump is not None and jump.active and jump.phase.value == "LANDING"):
        lplan = landing_state["plan"]
        two = _two_mass.TwoMassModel(dynamics.model, dynamics.data)
        p1, p2 = two.block_state()
        j1, j2 = two.block_jacobians()
        com_row, com_bias, com_z_abs, _com_vz = dynamics.com_vertical_row()
        lowest = float(np.min(_two_mass.R.wheel_contact_points(
            dynamics.model, dynamics.data)[:, 2]))
        elapsed = float(time_s) - float(landing_state["t0"])
        kp_h, kd_h = tasks.gains.height_gains
        kp_r, kd_r = tasks.gains.regularize_gains
        zc_now = float(com_z_abs - lowest)
        vcz_now = float(com_row @ dynamics.data.qvel)
        dx_now = float(p2[0] - p1[0])
        dx_rate_now = float((j2[0] - j1[0]) @ dynamics.data.qvel)
        h_step = 1.0e-3          # [s] dx 计划速度用中心差分步长（= 物理步长 model.opt.timestep）
        dx_plan = lplan.dx_eq_at(elapsed)
        dx_rate_plan = ((lplan.dx_eq_at(elapsed + h_step) - lplan.dx_eq_at(max(elapsed - h_step, 0.0)))
                        / (h_step + min(elapsed, h_step)))
        com_task = TaskLevel(np.atleast_2d(np.asarray(com_row, dtype=float)),
                             np.array([lplan.acc(elapsed) + kp_h * (lplan.zc(elapsed) - zc_now)
                                       + kd_h * (lplan.vcz(elapsed) - vcz_now)]),
                             np.array([float(com_bias)]), name="com_z")
        dx_task = TaskLevel(np.atleast_2d((j2[0] - j1[0])),
                            np.array([kp_h * (dx_plan - dx_now) + kd_h * (dx_rate_plan - dx_rate_now)]),
                            np.array([float(_two_mass.dx_rel_bias(dynamics))]), name="dx_rel")
        leg = dynamics.leg_state()
        diff_rows, diff_bias = leg_combination(feedback.rows, -1.0)
        length_diff = 0.5 * (float(leg[0]) - float(leg[2]))
        phi_diff = 0.5 * (float(leg[1]) - float(leg[3]))
        l_d_task = TaskLevel(np.atleast_2d(diff_rows[0]),
                             np.array([-kp_h * length_diff
                                       - kd_h * float(diff_rows[0] @ dynamics.data.qvel)]),
                             np.array([float(diff_bias[0])]), name="L_d")
        phi_d_task = TaskLevel(np.atleast_2d(-diff_rows[1]),
                               np.array([kp_r * phi_diff
                                         + kd_r * float(diff_rows[1] @ dynamics.data.qvel)]),
                               np.array([float(-diff_bias[1])]), name="phi_d")
        kept = tuple(task for task in motion
                     if task.name not in ("length_R", "length_L", "angle_R", "angle_L",
                                          "forward_velocity"))
        motion = kept + (com_task, dx_task, l_d_task, phi_d_task)

    # ---- 文档 10.48 B2：THRUST 段把角度行/前向速度行换成"角动量 + 差模腿角" ----
    # 只替换本段任务集：pitch/length/yaw/roll 与几何/相位判据一律不动。
    if (thrust_momentum and jump is not None and jump.active
            and jump.phase.value == "THRUST"):
        _t_perf = time.perf_counter()
        momentum_row, momentum_bias, momentum_h = dynamics.pitch_momentum_row()
        rate = 0.0 if momentum_state is None else float(momentum_state.get("integral", 0.0))
        if momentum_state is not None:
            last = momentum_state.get("last_t")
            if last is not None:
                rate = float(momentum_state.get("integral", 0.0)) + momentum_h * max(
                    0.0, float(time_s) - float(last))
            momentum_state["integral"] = rate
            momentum_state["last_t"] = float(time_s)
            momentum_state["h"] = momentum_h
            momentum_state["h_des"] = (-float(k_momentum) * momentum_h
                                       - float(k_momentum_integral) * rate)
            momentum_state["row_ms"] = (time.perf_counter() - _t_perf) * 1000.0
        desired = (-float(k_momentum) * momentum_h - float(k_momentum_integral) * rate)
        momentum_task = TaskLevel(
            np.atleast_2d(np.asarray(momentum_row, dtype=float)),
            np.array([desired]), np.array([float(momentum_bias)]), name="momentum_pitch")
        # 差模腿角：摘掉 angle_R/L 后保持两腿腿角一致（沿用现有 angle 任务的半差与 kp_r/kd_r）
        leg_rows_raw, leg_bias_raw = feedback.rows.leg
        kp_r, kd_r = tasks.gains.regularize_gains
        angle_row_r, angle_row_l = -leg_rows_raw[1], -leg_rows_raw[3]
        angle_bias_r, angle_bias_l = -leg_bias_raw[1], -leg_bias_raw[3]
        phi_d_row = 0.5 * (angle_row_r - angle_row_l)
        phi_d_bias = 0.5 * (angle_bias_r - angle_bias_l)
        leg_now = dynamics.leg_state()
        phi_d_meas = 0.5 * ((0.0 - float(leg_now[1])) - (0.0 - float(leg_now[3])))
        phi_d_rate = 0.5 * (float(angle_row_r @ dynamics.data.qvel)
                            - float(angle_row_l @ dynamics.data.qvel))
        phi_d_desired = 0.0 - kp_r * phi_d_meas - kd_r * phi_d_rate
        phi_d_task = TaskLevel(np.atleast_2d(phi_d_row), np.array([phi_d_desired]),
                               np.array([float(phi_d_bias)]), name="phi_d")
        kept = tuple(task for task in motion
                     if task.name not in ("angle_R", "angle_L", "forward_velocity"))
        rebuilt = []
        inserted = False
        for task in kept:
            rebuilt.append(task)
            if not inserted and task.name in ("length_R", "length_L"):
                rebuilt.extend((momentum_task, phi_d_task))
                inserted = True
        if not inserted:
            rebuilt.extend((momentum_task, phi_d_task))
        motion = tuple(rebuilt)
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
    # N1（2026-10-17，用户同意）：落地窗口内强制打开 motion 组（腿长/腿角/roll/yaw/前向速度），
    # 使 single_support / recovery 下腿长仍有人管。landing_window=False 时行为与改动前完全一致。
    motion_override = True if landing_window else None
    # VP1（2026-10-09）：等效力前馈。把 tau_ff 折进 dynamics_bias 与执行器限幅，
    # 而不是在 solve_wbc 之后直接叠加（否则限幅不自洽）。
    tau_ff = np.zeros(dynamics.model.nu)
    feedforward_force = 0.0
    dynamics_bias_solve = dynamics.dynamics_bias()
    limits_solve = dynamics.limits
    feedforward_active = False
    if vmc_phase:
        mass_total = (float(np.sum(dynamics.model.body_mass)) if jump_config is None
                      else float(jump_config.total_mass))
        a_z = float(jump.desired_com_acceleration_z)
        feedforward_force = 0.5 * mass_total * (GRAVITY + a_z)
        tau_ff, _peak = dynamics.thrust_torque(np.full(2, feedforward_force))
        tau_ff = np.asarray(tau_ff, dtype=float)
        feedforward_active = True
    # P-b / Pb2（2026-10-17，用户同意）：MPC 计划力前馈接线。默认关闭；与 VMC 前馈走同一条路径。
    # Pb2：加 landing_window 门控（只在落地锁存期内生效），且 tau_ff 真折叠进限幅与 dynamics_bias。
    # R 取 sim.reduced.wheel_radius，不写死。
    if (mpc_force_feedforward and landing_window and solution is not None
            and wheel_radius is not None and float(wheel_radius) > 0.0):
        plan_force = np.asarray(solution.force, dtype=float)
        radius = float(wheel_radius)
        tau_ff[dynamics.wheel_actuators[0]] = radius * float(plan_force[0])
        tau_ff[dynamics.wheel_actuators[1]] = radius * float(plan_force[1])
        feedforward_active = True
        counters["mpc_ff_frames"] = counters.get("mpc_ff_frames", 0) + 1
        _LOGGER.info(
            "P-b MPC 计划力前馈：f=[%.4f, %.4f] tau_ff_wheel=[%.4f, %.4f] 生效帧=%d",
            float(plan_force[0]), float(plan_force[1]),
            float(tau_ff[dynamics.wheel_actuators[0]]),
            float(tau_ff[dynamics.wheel_actuators[1]]), counters["mpc_ff_frames"])
    if feedforward_active:
        # Pb2（2026-10-17，用户同意）：真折叠。把 tau_ff 折进 WBC 的执行器限幅与 dynamics_bias，
        # 使求解器对已施加力矩记账，再在求解后叠加 tau_ff（限幅自洽）。与 VMC 同一段代码。
        limits_solve = replace(limits_solve,
                               lower=limits_solve.lower - tau_ff,
                               upper=limits_solve.upper - tau_ff)
        dynamics_bias_solve = dynamics_bias_solve - dynamics.selection.T @ tau_ff
    torque, result = control.solve_wbc(
        mass=dynamics.mass(), selection=selection, dynamics_bias=dynamics_bias_solve,
        contacts=blocks, balance_tasks=balance, motion_tasks=motion, limits=limits_solve,
        decision=feedback.decision, dt=timing.wbc_dt, external_force=external,
        weight_overrides=jump_weight_overrides, motion_override=motion_override)
    if feedforward_active:
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
