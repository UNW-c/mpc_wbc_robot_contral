"""Headless, truth-state MuJoCo integration for control-logic validation.

This is an experimental entry point, independent of the legacy application.
No estimator/adaptive compensation is enabled. Contact is a flat-ground nominal
rolling model, not a claim of equivalence to MuJoCo's compliant contact solver.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, replace
import gzip
import json
import logging
import time
from pathlib import Path

import mujoco
import numpy as np

from Algorithm.geometry import quaternion_pitch, quaternion_yaw, wrap_angle
from Algorithm.mpc import ChassisState, MotionCommand, WheelMPC, MPCSettings
from Algorithm.supervision import (ContactBlock, ControlSupervisor, SupervisedControl,
                                   WbcFeedbackAccumulator)
from Algorithm.timing import ControlTiming
from Algorithm.wbc import (ActuatorLimits, ContactElimination, TaskLevel,
                           height_task_activation, height_task_activation_from_factor,
                           roll_gain_scale, roll_gain_scale_from_factor)
from Algorithm.wbc.task_qp import task_qp_resolver_factory
from APP.mujoco_robot import MuJoCoRobot
from APP.config import ROOT
from APP.mujoco_balance_model import HeldHipModelBuilder, HipInputModelBuilder

_LOGGER = logging.getLogger(__name__)
_DIFFERENCE_DT = 1e-6
# Headless duration when --duration is omitted; matches the original 60 s default.
_DEFAULT_HEADLESS_DURATION = 60.0  # [s]
# Tail statistics window; the same value bounds records memory for unbounded runs.
_RECENT_RECORD_WINDOW = 5.0  # [s]

# 单级 QP 的任务权重。height=50：2026-09-27 用户要求补齐横滚/高度类任务后恢复
# （多工况扫描曾显示 0 略优 1-2%，但那是仅前后向工况下的结论）。
_TASK_QP_WEIGHTS = {
    "pitch": 100.0,
    "roll": 100.0,
    "yaw": 50.0,
    "height": 50.0,
    "forward_velocity": 50.0,
    "regularization": 1.0,
    # B3（A4-6）：8 状态模型下 regularization 拆成两项——共模跟踪 MPC（100）、差模保持名义（1）。
    "hip_tracking": 100.0,
    "hip_differential": 1.0,
}


@dataclass(frozen=True)
class SimulationConfig:
    # Existing PlanarWbc defaults; fixed for this validation, not tuned on MuJoCo.
    attitude_gains: tuple[float, float] = (80.0, 10.0)
    height_gains: tuple[float, float] = (100.0, 12.0)
    # 速度反馈增益；2026-09-25 与位置环联合扫描后提高到 320（见修改日志）。
    velocity_gain: float = 320.0
    regularize_gains: tuple[float, float] = (60.0, 6.0)
    wbc_dt: float = 0.004
    residual_tolerance: float = 1e-7
    force_tolerance: float = 1e-6
    speed_guard: float = 1.60
    # 实验性 |roll|/|pitch| 停机线，取自用户的 45° 上限。
    # 它是测试保护，不是控制器参数；超过即停机并记为失败。
    attitude_guard: float = 0.785
    # WBC 采样 MPC 轨迹的前瞻时长 [s]。0（T4 默认）= 按实际时刻采样：MPC 刷新的
    # 那一拍轨迹第 0 节点就是实测状态，此时 PD 误差为 0 是正确的；两次刷新之间
    # 误差自然增长，由 WBC 的 PD 纠正（真正的反馈）；前馈只来自 sample.acceleration。
    # 旧默认 0.02 是把 PD 变成了 20 ms 前瞻的前馈注入（见排查记录 #6）。
    preview_lead: float = 0.0
    # WBC 求解后端："qp"（单级 QP，2026-09-25 后的主路径）或
    # "nullspace"（旧零空间法，保留作对照）。
    wbc_backend: str = "qp"
    # 模型变体。"hip_input" = 8 状态含髋输入模型（2026-09-28 部署为默认，C7）；
    # "held_hip" = 6 状态握髋模型（旧结构，保留作对照，见 --preset old）。
    model_variant: str = "hip_input"
    # B3：共模髋角 guard [rad]。机构极限暂定 0.9（C3 待用户给真实值），只作为停机保护。
    hip_guard: float = 0.9
    # 记录间隔（物理步）：1 = 1 kHz 全物理步（调试/调参），4 = 250 Hz（与 WBC 同步）。
    record_stride: int = 4
    # 无限（viewer）运行的记录裁剪：False（默认）保留全量供事后分析；
    # 超长时间运行可设 True 只保留最近窗口。
    trim_infinite_records: bool = False
    # WBC 位置反馈增益 [1/s^2 per m]：>0 时把位置误差叠加到前向速度任务目标，
    # 让 WBC 参与位移回位。2026-09-25 联合扫描最优（pg=320/vg=320，
    # 见 docs/修改日志.md）；置 0 可关闭位置环。仅 forward_mode="legacy" 使用。
    position_gain: float = 320.0
    # 纵向任务目标来源（T1，2026-09-27）：
    # "mpc"    = 跟踪 MPC 优化轨迹：a_mpc(t) + kv·(v_mpc(t) − v) + kp·(s_mpc(t) − s)，
    #            按实际时刻采样，不引用键盘指令与 planner 参考位置；
    # "legacy" = 旧的绝对回零 PD（vg·指令速度 − v − pg·位置误差），仅作对照。
    forward_mode: str = "mpc"  # 2026-09-28 部署为默认（C7）；旧结构用 --preset old
    # 跟踪 MPC 轨迹的误差增益，初值按 ω=10 rad/s 临界阻尼给出（kv = 2ω，kp = ω²），T5 再定。
    tracking_velocity_gain: float = 20.0   # [1/s]
    tracking_position_gain: float = 100.0  # [1/s^2]
    # WBC 高度任务激活下限：死点排程因子为 0 时用此下限替代。
    # 实测（2026-09-25）：前向 0.2 m/s 扰动下 0/0.5/1.0 无差异（J 9.03，无副作用），
    # 为后续侧向/偏心工况保留高度反馈，默认激活。
    height_activation_floor: float = 1.0
    # 横滚任务激活下限（2026-09-27 补）：死点处 roll_gain_scale=0 会让横滚完全
    # 无主动控制；用此下限保留最小权限（0 = 历史行为）。
    roll_activation_floor: float = 1.0
    # WBC 前向任务目标加速度的限幅 [m/s^2]。QP 的力矩/接触硬约束已限制可实现值，
    # 目标只是"愿望"：实测限幅 50 与不限幅等价（J=9.03），限幅 5 会把正常抗扰需求
    # （vg·速度误差≈24）削掉导致性能退化（J=15.32）。50 仅防大误差时目标爆炸。
    acceleration_limit: float = 50.0

    def __post_init__(self) -> None:
        if int(self.record_stride) < 1:
            raise ValueError("record_stride 必须 >= 1（物理步为单位）")
        if self.forward_mode not in ("mpc", "legacy"):
            raise ValueError(f"未知 forward_mode: {self.forward_mode}")
        if self.model_variant not in ("held_hip", "hip_input"):
            raise ValueError(f"未知 model_variant: {self.model_variant}")


@dataclass(frozen=True)
class Disturbance:
    """Deterministic push on the floating base.

    瞬时分量（``delta_velocity``/``delta_omega``）立即加到 ``qvel``；
    持续分量（``force``/``torque``/``duration``）在 ``duration`` 内作为广义外力
    （``qfrc_applied``）每步施加，用于复现鼠标拖动这类持续推力。
    """

    time: float                                  # [s]
    delta_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)  # [m/s] world-frame
    delta_omega: tuple[float, float, float] = (0.0, 0.0, 0.0)     # [rad/s] body-frame
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)           # [N] world-frame
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)          # [N*m] body-frame
    duration: float = 0.0                                         # [s] 持续时长


class MuJoCoDynamics:
    """Named actuator mapping and geometric rolling-contact matrices."""

    def __init__(self, robot: MuJoCoRobot):
        self.robot = robot
        self.model = robot.model
        self.wheels = tuple(int(self.model.geom(n).id) for n in
                            ("right_wheel_collision", "left_wheel_collision"))
        self.floor = int(self.model.geom("floor").id)
        self.selection = np.zeros((self.model.nu, self.model.nv))
        for actuator in range(self.model.nu):
            joint = self.model.actuator_trnid[actuator, 0]
            self.selection[actuator, self.model.jnt_dofadr[joint]] = self.model.actuator_gear[actuator, 0]
        self.limits = ActuatorLimits(self.model.actuator_ctrlrange[:, 0],
                                     self.model.actuator_ctrlrange[:, 1])
        self.scratch = mujoco.MjData(self.model)

    def mass(self):
        matrix = np.empty((self.model.nv, self.model.nv))
        if tuple(map(int, mujoco.__version__.split(".")[:2])) >= (3, 13):
            mujoco.mj_fullM(self.model, self.robot.data, matrix)
        else:
            mujoco.mj_fullM(self.model, matrix, self.robot.data.qM)
        return matrix

    def sensed_contact(self):
        """Sum every wheel-floor contact; forces are on the wheel in world axes."""
        forces = np.zeros((2, 3))
        counts = np.zeros(2, dtype=int)
        other = 0
        data = self.robot.data
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = (int(contact.geom1), int(contact.geom2))
            if self.floor not in pair:
                other += 1
                continue
            geom = pair[1] if pair[0] == self.floor else pair[0]
            if geom not in self.wheels:
                other += 1
                continue
            side = self.wheels.index(geom)
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, wrench)
            sign = 1.0 if pair[1] == geom else -1.0
            forces[side] += sign * contact.frame.reshape(3, 3).T @ wrench[:3]
            counts[side] += 1
        contact = tuple(bool(n > 0 and f[2] > 0) for n, f in zip(counts, forces))
        return contact, forces, counts, other

    @staticmethod
    def attitude(data):
        w, x, y, z = data.qpos[3:7]
        roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
        return np.array([roll, quaternion_pitch(data.qpos[3:7]), quaternion_yaw(data.qpos[3:7])])

    def kinematics(self, data):
        """Rows for heading velocity, Euler rates, and moving contact geometry."""
        roll, pitch, yaw = self.attitude(data)
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        rows = np.zeros((4, self.model.nv))
        rows[0, :3] = heading
        # Free-joint angular qvel is expressed in the local body frame.
        rows[1:4, 3:6] = np.array([
            [1, np.sin(roll)*np.tan(pitch), np.cos(roll)*np.tan(pitch)],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll)/np.cos(pitch), np.cos(roll)/np.cos(pitch)]])
        contacts = []
        for geom in self.wheels:
            axle = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            down = np.array([0., 0., -1.])
            radial = down - axle * (axle @ down)
            radial /= np.linalg.norm(radial)
            point = data.geom_xpos[geom] + self.model.geom_size[geom, 0] * radial
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, data, jacobian, None, point, self.model.geom_bodyid[geom])
            contacts.append(np.vstack((heading @ jacobian, jacobian[2], lateral @ jacobian)))
        return rows, np.asarray(contacts)

    def matrices(self, contact):
        data = self.robot.data
        rows, jacobians = self.kinematics(data)
        samples = []
        for direction in (-1., 1.):
            self.scratch.qpos[:] = data.qpos
            self.scratch.qvel[:] = data.qvel
            mujoco.mj_integratePos(self.model, self.scratch.qpos, data.qvel,
                                  direction * _DIFFERENCE_DT)
            mujoco.mj_forward(self.model, self.scratch)
            samples.append(self.kinematics(self.scratch))
        task_bias = (samples[1][0] - samples[0][0]) @ data.qvel / (2 * _DIFFERENCE_DT)
        # Differentiate the moving geometric point, not a fixed material point.
        contact_bias = (samples[1][1] - samples[0][1]) @ data.qvel / (2 * _DIFFERENCE_DT)
        blocks = [ContactBlock(jacobians[i, :2], contact_bias[i, :2], (i,)) for i in range(2)]
        active = tuple(i for i, enabled in enumerate(contact) if enabled)
        if active:
            blocks.append(ContactBlock(np.mean(jacobians[list(active), 2], axis=0)[None, :],
                                       np.array([np.mean(contact_bias[list(active), 2])]), active))
        return rows, task_bias, tuple(blocks)

    def hip_pitch_moment(self):
        # Reaction on the base is opposite each joint's world-axis motor torque.
        data = self.robot.data
        heading_y = np.array([-np.sin(quaternion_yaw(data.qpos[3:7])),
                              np.cos(quaternion_yaw(data.qpos[3:7])), 0.])
        moment = 0.0
        for name in ("j1_ctrl", "j3_ctrl"):
            actuator = int(self.model.actuator(name).id)
            joint = self.model.actuator_trnid[actuator, 0]
            moment -= float(heading_y @ data.xaxis[joint]) * float(data.actuator_force[actuator]) * self.model.actuator_gear[actuator, 0]
        return moment


class MuJoCoMpcWbcSimulation:
    """50 Hz MPC -> 250 Hz WBC -> 1 kHz unmodified MuJoCo physics."""

    def __init__(self, config: SimulationConfig | None = None, initial_pitch: float = 0.0,
                 mpc_settings: MPCSettings | None = None):
        self.config = config or SimulationConfig()
        self.robot = MuJoCoRobot.load("plane")
        self.dynamics = MuJoCoDynamics(self.robot)
        # Hip reactions already belong to the held-hip reduced dynamics. The
        # old isolated-body tau/I correction would count them a second time.
        resolved_settings = mpc_settings or MPCSettings(compensate_hip_torque=False)
        if self.config.model_variant == "hip_input":
            if resolved_settings.compensate_hip_torque:
                # 髋力矩已经是模型输入，再补一次会重复计入（A4-6 的断言）。
                raise ValueError("hip_input 模型下 compensate_hip_torque 必须为 False")
            reduced = HipInputModelBuilder(self.dynamics).build_reduced(0.0)
        else:
            reduced = HeldHipModelBuilder(self.dynamics).build()
        self.mpc = WheelMPC(reduced, resolved_settings)
        self.timing = ControlTiming(self.robot.timestep, self.config.wbc_dt, self.mpc.settings.dt)
        if self.config.wbc_backend not in ("nullspace", "qp"):
            raise ValueError(f"未知 WBC 后端: {self.config.wbc_backend}")
        control_kwargs = {}
        if self.config.wbc_backend == "qp":
            control_kwargs["wbc_resolver_factory"] = task_qp_resolver_factory(_TASK_QP_WEIGHTS)
        self.control = SupervisedControl(self.mpc, **control_kwargs)
        self.supervisor = ControlSupervisor(self.mpc.settings.limits.touchdown_dwell)
        self.feedback = WbcFeedbackAccumulator(("forward_velocity", "yaw"))
        self.height = float(self.robot.data.qpos[2])
        self.hip_q = [int(self.robot.model.joint(n).qposadr[0]) for n in ("joint1", "joint3")]
        self.hip_v = [int(self.robot.model.joint(n).dofadr[0]) for n in ("joint1", "joint3")]
        self._hip_axis_sign = self._hip_axis_sign_value()
        self.nominal_hips = self.robot.data.qpos[self.hip_q].copy()
        self.robot.set_initial_pitch(initial_pitch)
        if initial_pitch:
            self.robot._place_wheels_on_ground()
        self.distance = 0.0
        self.solution = None
        self.command = MotionCommand(velocity=0., yaw_rate=0.)
        self.records = []
        # 最近一次 WBC 求解的诊断量，供高采样率记录复用（步间保持）。
        self._wbc_diagnostics = None
        self.stop_reason = None
        self.mpc_calls = 0
        self.wbc_calls = 0
        self.mpc_fallbacks = 0
        self.wbc_fallbacks = 0
        # R2 诊断用：非 None 时逐拍记录 MPC 求解摘要（默认不记录，无开销）。
        self.solution_log: list | None = None

    def _hip_axis_sign_value(self) -> float:
        """两髋轴的符号关系 dot(axis1, axis3)（当前模型 -1），共模/差模定义共用。"""
        axis1 = np.asarray(self.robot.model.joint("joint1").axis, dtype=float)
        axis3 = np.asarray(self.robot.model.joint("joint3").axis, dtype=float)
        sign = float(np.dot(axis1, axis3))
        if not np.isclose(sign, -1.0):
            raise ValueError(f"髋轴方向关系变化（dot={sign}），共模/差模定义需要复核")
        return sign

    def _hip_common(self, values: np.ndarray) -> float:
        """共模髋角 0.5*(q1 + sign*q3)：腿的前后摆动（纵向，T7 用）。

        两髋轴 -y/+y 相反，同相位摆动时 q1 = -q3，平均值恒为 0；不要用 mean 代替
        （viewer-test6 实测 q1=+0.374、q3=-0.374 时 mean 恒为 0，排查记录 #10）。
        """
        return 0.5 * float(values[0] + self._hip_axis_sign * values[1])

    def _hip_differential(self, values: np.ndarray) -> float:
        """差模髋角 0.5*(q1 - sign*q3)：决定两轮前后错开 0.12*sin(q_diff)（#13）。"""
        return 0.5 * float(values[0] - self._hip_axis_sign * values[1])

    def _hip_authority(self, values: np.ndarray) -> float:
        """单腿权限因子 mean(|sin q1|, |sin q3|)：横滚/高度排程用（#13）。"""
        return 0.5 * (abs(float(np.sin(values[0]))) + abs(float(np.sin(values[1]))))

    def state(self, contact, forces):
        data = self.robot.data
        rows, _ = self.dynamics.kinematics(data)
        rates = rows @ data.qvel
        attitude = self.dynamics.attitude(data)
        return ChassisState(position=self.distance, velocity=float(rates[0]),
                            pitch=float(attitude[1])-self.mpc.model.metadata["equilibrium_pitch"],
                            pitch_rate=float(rates[2]),
                            yaw=float(attitude[2]), yaw_rate=float(rates[3]),
                            hip_angle=self._hip_common(data.qpos[self.hip_q]),
                            hip_rate=self._hip_common(data.qvel[self.hip_v]),
                            hip_differential=self._hip_differential(data.qpos[self.hip_q]),
                            hip_authority=self._hip_authority(data.qpos[self.hip_q]),
                            hip_torque=self.dynamics.hip_pitch_moment(),
                            normal_forces=tuple(forces[:, 2]), contact=contact)

    def tasks(self, state, rows, bias, sample):
        data = self.robot.data
        nv = self.robot.model.nv
        kp, kd = self.config.attitude_gains
        reference = state.vector().copy() if sample is None else sample.state
        acceleration = np.zeros(3) if sample is None else sample.acceleration
        if sample is None:
            reference[2:6] = 0.0
        pitch = acceleration[1] + kp*(reference[2]-state.pitch) + kd*(reference[3]-state.pitch_rate)
        roll = -kp*self.dynamics.attitude(data)[0] - kd*(rows[1] @ data.qvel)
        authority = (state.hip_authority if state.hip_authority is not None
                     else roll_gain_scale(state.hip_angle))
        balance = (
            TaskLevel(rows[[2]], np.array([pitch]), bias[[2]], name="pitch"),
            TaskLevel(rows[[1]], np.array([roll]), bias[[1]], name="roll",
                      activation=max(roll_gain_scale_from_factor(authority),
                                     self.config.roll_activation_floor)))
        kp_h, kd_h = self.config.height_gains
        kp_r, kd_r = self.config.regularize_gains
        hip_target = (kp_r*(self.nominal_hips - data.qpos[self.hip_q])
                      - kd_r*data.qvel[self.hip_v])
        # 纵向任务目标（T1，2026-09-27）：
        # - mpc 模式：MPC 是纵向唯一规划者。前馈取 sample.acceleration（MPC 规划的
        #   纵向加速度），误差项只修正 MPC 两次刷新之间的偏差；不引用键盘指令
        #   （command.velocity）与 planner 参考位置，避免两套参考互相抵消。
        # - legacy 模式：旧的绝对回零 PD（保留作对照）。历史结论“相对 MPC 参考的
        #   PD 失效（J 9.03→29.01）”是在 preview_lead=0.02（PD 变前馈，见 #6）和
        #   双控制器结构（键盘指令直进 WBC，见 #3）下得出的，不代表跟踪结构本身有问题。
        if self.config.forward_mode == "mpc":
            kv = self.config.tracking_velocity_gain
            kp = self.config.tracking_position_gain
            if sample is None:
                # 非 GROUND 模式没有 MPC 轨迹：只做速度阻尼，不引用任何位置参考
                velocity_target = -kv * state.velocity
            else:
                velocity_target = (sample.acceleration[0]
                                   + kv * (sample.state[1] - state.velocity)
                                   + kp * (sample.state[0] - state.position))
        else:
            velocity_target = self.config.velocity_gain * (float(self.command.velocity or 0.0) - state.velocity)
            if self.config.position_gain:
                # 位置项叠加到速度任务目标；参考位置取 MPC 规划器的参考位置
                # （HOLD 时为原点、DRIVE 时跟随实测），不能用硬编码 0，否则 DRIVE 被拉回。
                reference_position = (0.0 if self.solution is None
                                      else float(self.solution.reference.position[0]))
                velocity_target -= self.config.position_gain * (float(state.position) - reference_position)
        velocity_target = float(np.clip(velocity_target, -self.config.acceleration_limit,
                                        self.config.acceleration_limit))
        self._forward_target = velocity_target
        if self.config.model_variant == "hip_input":
            # B3（A4-6）：共模髋跟踪 MPC 轨迹（含 q_c_ddot 前馈），差模髋保持名义。
            q_common = self._hip_common(data.qpos[self.hip_q])
            qd_common = self._hip_common(data.qvel[self.hip_v])
            q_diff = self._hip_differential(data.qpos[self.hip_q])
            qd_diff = self._hip_differential(data.qvel[self.hip_v])
            if sample is None:
                common_target = -kp_r * q_common - kd_r * qd_common
            else:
                common_target = (sample.acceleration[3]
                                 + kp_r * (sample.state[6] - q_common)
                                 + kd_r * (sample.state[7] - qd_common))
            diff_target = -kp_r * q_diff - kd_r * qd_diff
            common_row = 0.5 * (np.eye(nv)[self.hip_v[0]]
                                + self._hip_axis_sign * np.eye(nv)[self.hip_v[1]])
            diff_row = 0.5 * (np.eye(nv)[self.hip_v[0]]
                              - self._hip_axis_sign * np.eye(nv)[self.hip_v[1]])
            hip_tasks = (
                TaskLevel(common_row[None, :], np.array([common_target]), name="hip_tracking"),
                TaskLevel(diff_row[None, :], np.array([diff_target]), name="hip_differential"),
            )
        else:
            hip_tasks = (TaskLevel(np.eye(nv)[self.hip_v], hip_target, name="regularization"),)
        motion = (
            TaskLevel(rows[[3]], np.array([acceleration[2] + kp*wrap_angle(reference[4]-state.yaw)
                                           + kd*(reference[5]-state.yaw_rate)]), bias[[3]], name="yaw"),
            TaskLevel(np.eye(nv)[[2]], np.array([kp_h*(self.height-data.qpos[2])-kd_h*data.qvel[2]]),
                      name="height", activation=max(height_task_activation_from_factor(authority),
                                                   self.config.height_activation_floor)),
            TaskLevel(rows[[0]], np.array([velocity_target]),
                      bias[[0]], name="forward_velocity"),
        ) + hip_tasks
        # Parking must not spend the hip-posture authority on a redundant base
        # velocity task. A drifting crank changes the reduced model itself.
        # Pitch remains first; hip posture also stays active during recovery.
        return (balance[0], motion[-1], balance[1]), motion[:-1]

    def run(self, command, duration: float, viewer=None, disturbances=None, allow_driving=False):
        """Integrate the scenario; a supplied passive viewer is synced at real-time pace."""
        if np.isnan(duration) or duration <= 0:
            raise ValueError("duration must be positive")
        infinite = not np.isfinite(duration)
        if infinite and viewer is None:
            raise ValueError("an infinite duration needs a viewer to stop it")
        if self.records:
            raise ValueError("Create a new simulation for each scenario")
        data = self.robot.data
        pending = sorted(disturbances or (), key=lambda item: item.time)
        force_until = 0.0
        wrench = np.zeros(6)
        total_steps = None if infinite else int(np.ceil(duration / self.robot.timestep))
        step = 0
        while total_steps is None or step < total_steps:
            step_wall_start = time.perf_counter()
            while pending and self.robot.time >= pending[0].time:
                push = pending.pop(0)
                data.qvel[0:3] += np.asarray(push.delta_velocity, dtype=float)
                data.qvel[3:6] += np.asarray(push.delta_omega, dtype=float)
                if push.duration > 0.0:
                    force_until = self.robot.time + float(push.duration)
                    wrench = np.concatenate([np.asarray(push.force, dtype=float),
                                             np.asarray(push.torque, dtype=float)])
            # 持续扰动：在时长内每步施加广义外力（headless 复现鼠标拖动）。
            data.qfrc_applied[:] = 0.0
            if self.robot.time < force_until:
                data.qfrc_applied[:6] = wrench
            if viewer is not None:
                # The passive viewer only tracks MjvPerturb from mouse input; it never
                # applies it because physics runs in this loop, so refresh the force here.
                data.xfrc_applied[:] = 0.0
                mujoco.mjv_applyPerturbPose(self.robot.model, data, viewer.perturb, 0)
                mujoco.mjv_applyPerturbForce(self.robot.model, data, viewer.perturb)
            if self.timing.wbc_due(step):
                # mj_step advances qpos after computing dynamics; refresh all sensors/matrices.
                self.robot.forward()
                contact, forces, counts, other = self.dynamics.sensed_contact()
                state = self.state(contact, forces)
                decision = self.supervisor.update(timestamp=self.robot.time, dt=self.timing.wbc_dt,
                    contact=contact, enabled=True, command_fresh=True,
                    # HOLD thresholds classify a stopped robot, not a safely moving one.
                    stable=max(abs(self.dynamics.attitude(data)[:2])) < self.mpc.settings.pitch_limit,
                    slipping=False)
                if not decision.motion_enabled:
                    self.solution = None
                if self.timing.mpc_due(step):
                    current = command(self.robot.time) if callable(command) else command
                    if not allow_driving and (current.position is not None or current.heading is not None
                                              or current.velocity not in (None, 0.)
                                              or current.yaw_rate not in (None, 0.)):
                        raise ValueError("This validation stage accepts zero velocity/yaw-rate commands only")
                    self.command = current
                    self.solution = self.control.solve_mpc(state, current, decision, self.feedback.consume(state))
                    if self.solution is not None:
                        self.mpc_calls += 1
                        self.mpc_fallbacks += int(self.solution.fallback)
                        if self.solution_log is not None:
                            solution = self.solution
                            self.solution_log.append(dict(
                                time=self.robot.time,
                                predicted_states=solution.predicted_states.tolist(),
                                reference_position=float(solution.reference.position[0]),
                                planner_mode=self.mpc.planner.mode.value,
                                planner_reset_reason=self.mpc.planner.last_reset_reason,
                                force_peak=float(np.max(np.linalg.norm(solution.force_trajectory, axis=1))),
                                qp_relaxed=bool(solution.qp.relaxed),
                                yaw_moment_limit=float(np.max(self.mpc.stage_bounds(state).yaw_moment)),
                                qp_solve_time=float(solution.qp.solve_time),
                            ))
                sample = (None if self.solution is None else self.solution.trajectory.sample_at(
                    self.robot.time + self.config.preview_lead))
                rows, bias, blocks = self.dynamics.matrices(contact)
                balance, motion = self.tasks(state, rows, bias, sample)
                mass = self.dynamics.mass()
                # MuJoCo passive forces are on the RHS of its equations.
                h = data.qfrc_bias - data.qfrc_passive
                torque, result = self.control.solve_wbc(mass=mass, selection=self.dynamics.selection,
                    dynamics_bias=h, contacts=blocks, balance_tasks=balance, motion_tasks=motion,
                    limits=self.dynamics.limits, decision=decision, dt=self.timing.wbc_dt)
                self.wbc_calls += 1
                predicted_force = np.zeros(0)
                predicted_qacc = np.zeros(self.robot.model.nv)
                if result is not None:
                    self.wbc_fallbacks += int(result.fallback_used)
                    self.feedback.observe(result, self.dynamics.limits)
                    active = [b for b in blocks if all(contact[i] for i in b.wheels)]
                    jc = np.vstack([b.jacobian for b in active])
                    predicted_force = ContactElimination.from_matrices(jc, self.dynamics.selection, mass).recover_contact_force(
                        result.qacc, h, torque)
                    predicted_qacc = result.qacc
                data.ctrl[:] = torque
                self.robot.forward()
                _, applied_forces, _, _ = self.dynamics.sensed_contact()
                # 诊断量缓存：物理步后的记录按 record_stride 采样时复用（步间保持）。
                self._wbc_diagnostics = dict(
                    torque=torque.tolist(),
                    applied_force=applied_forces.tolist(),
                    predicted_force=predicted_force.tolist(),
                    predicted_qacc=predicted_qacc.tolist(),
                    wbc_input_contact=list(contact),
                    wbc_input_force=forces.tolist(),
                    mode=decision.mode.value,
                    pitch_reference=None if sample is None else float(sample.state[2]),
                    reference_velocity=None if sample is None else float(sample.state[1]),
                    velocity_error=None if sample is None else float(sample.state[1] - state.velocity),
                    forward_target=float(self._forward_target),
                    mpc_acceleration=None if sample is None else float(sample.acceleration[0]),
                    planner_velocity=None if self.solution is None else float(self.solution.reference.velocity[1]),
                    mpc_velocity=None if self.solution is None else float(self.solution.predicted_states[1][1]),
                    position_reference=None if self.solution is None else float(self.solution.reference.position[0]),
                    mpc_force=None if sample is None else sample.force.tolist(),
                    contact_residual=0. if result is None else result.hierarchy.contact_residual_norm,
                    dynamics_residual=0. if result is None else result.dynamics_residual_norm,
                    scaling={} if result is None else result.scaling,
                    task_residuals={} if result is None else {o.name: o.residual_norm for o in result.hierarchy.outcomes},
                    task_goal={} if result is None else {
                        o.name: [float(np.linalg.norm(o.desired)), float(np.linalg.norm(o.achieved))]
                        for o in result.hierarchy.outcomes},
                    task_activation={} if result is None else {
                        o.name: float(o.activated) for o in result.hierarchy.outcomes},
                    mpc_fallback=False if self.solution is None else self.solution.fallback,
                    wbc_fallback=False if result is None else result.fallback_used)
                reason = None
                if other:
                    reason = "unexpected_body_contact"
                elif result is not None and not result.feasible:
                    reason = "wbc_equality_or_torque_failure"
                elif self.mpc.degraded:
                    reason = "persistent_mpc_fallback"
                elif predicted_force.size and min(predicted_force[1:2*sum(contact):2]) < -self.config.force_tolerance:
                    reason = "negative_predicted_normal_force"
                elif not self.dynamics.limits.contains(torque):
                    reason = "torque_limit"
                if reason:
                    self.stop_reason = reason
                    break
            previous = data.qpos[:3].copy()
            yaw = quaternion_yaw(data.qpos[3:7])
            self.robot.step()
            middle_yaw = yaw + 0.5*wrap_angle(quaternion_yaw(data.qpos[3:7])-yaw)
            self.distance += float((data.qpos[:3]-previous) @ np.array([np.cos(middle_yaw), np.sin(middle_yaw), 0.]))
            if not np.all(np.isfinite(np.r_[data.qpos, data.qvel, data.ctrl])):
                self.stop_reason = "nonfinite_state"
                break
            if step % self.config.record_stride == 0:
                self.records.append(self._sample_record())
            attitude = self.dynamics.attitude(data)
            if max(abs(attitude[:2])) > self.config.attitude_guard:
                self.stop_reason = "attitude_guard"
                break
            if np.linalg.norm(data.qvel[:2]) > self.config.speed_guard:
                self.stop_reason = "speed_guard"
                break
            if self.config.hip_guard > 0.0:
                if abs(self._hip_common(data.qpos[self.hip_q])) > self.config.hip_guard:
                    self.stop_reason = "hip_guard"
                    break
            if viewer is not None:
                # Passive viewer pump plus real-time pacing; closing the window stops the run.
                viewer.sync()
                remaining = self.robot.timestep - (time.perf_counter() - step_wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
                if not viewer.is_running():
                    self.stop_reason = "viewer_closed"
                    break
            if infinite and self.config.trim_infinite_records:
                self._trim_records()
            step += 1
        summary = dict(completed=self.stop_reason is None, stop_reason=self.stop_reason,
            simulated_seconds=self.robot.time, requested_seconds=duration,
            mpc_calls=self.mpc_calls, wbc_calls=self.wbc_calls,
            mpc_fallbacks=self.mpc_fallbacks, wbc_fallbacks=self.wbc_fallbacks,
            final_qpos=data.qpos.tolist(), final_qvel=data.qvel.tolist(),
            max_abs_pitch=max(abs(r["state"][2]) for r in self.records),
            max_torque=np.max(np.abs([r["torque"] for r in self.records]), axis=0).tolist(),
            max_dynamics_residual=max(r["dynamics_residual"] for r in self.records),
            max_contact_residual=max(r["contact_residual"] for r in self.records))
        # Finishing the integration is not the same as passing a balance check.
        tail = [r for r in self.records if r["time"] >= self.robot.time-_RECENT_RECORD_WINDOW]
        summary.update(max_abs_position=max(abs(r["state"][0]) for r in self.records),
                       tail_max_abs_speed=max(abs(r["state"][1]) for r in tail),
                       tail_position_span=float(np.ptp([r["state"][0] for r in tail])),
                       tail_max_abs_pitch=max(abs(r["attitude"][1]) for r in tail),
                       max_abs_hip=float(np.max(np.abs([np.array(r["qpos"])[self.hip_q] for r in self.records]))),
                       final_distance=self.distance,
                       model_source=self.mpc.model.source, model_metadata=self.mpc.model.metadata)
        data.ctrl[:] = 0.
        _LOGGER.info("MuJoCo logic test: %s", summary)
        return summary

    def _trim_records(self) -> None:
        """Keep a bounded recent window so an unbounded run does not grow memory."""
        interval = self.robot.timestep * self.config.record_stride
        window_records = max(1, int(round(_RECENT_RECORD_WINDOW / interval)))
        if len(self.records) > 2 * window_records:
            del self.records[:len(self.records) - window_records]

    def _sample_record(self) -> dict:
        """物理步后的记录采样；WBC 诊断取自最近一次求解（步间保持）。

        每个记录点先 forward，使 qacc / 接触力与当前 qpos/qvel/ctrl 一致，
        避免记到半步前的旧值。
        """
        data = self.robot.data
        self.robot.forward()
        contact, forces, counts, _ = self.dynamics.sensed_contact()
        state = self.state(contact, forces)
        record = dict(
            time=self.robot.time,
            state=state.vector().tolist(),
            qpos=data.qpos.tolist(),
            qvel=data.qvel.tolist(),
            # 施加之扰动（viewer 鼠标扰动用 xfrc_applied；持续扰动用 qfrc_applied）
            xfrc_applied=data.xfrc_applied.tolist(),
            qfrc_applied=data.qfrc_applied[:6].tolist(),
            torque=(self._wbc_diagnostics or {}).get(
                "torque", np.zeros(self.dynamics.limits.size).tolist()),
            ctrl=data.ctrl.tolist(),
            contact=list(contact),
            contact_counts=counts.tolist(),
            actual_force=forces.tolist(),
            attitude=self.dynamics.attitude(data).tolist(),
            actual_qacc=data.qacc.tolist(),
            reference_mode=self.mpc.planner.mode.value,
            reference_valid=self.mpc.planner.valid,
            hip_pitch_moment=state.hip_torque,
        )
        record.update(self._wbc_diagnostics or {})
        return record


_NEW_PRESET = {
    # 新方案（2026-09-28 部署为默认，C7）：与 SimulationConfig / MPCSettings 的默认值一致。
    "model_variant": "hip_input",
    "forward_mode": "mpc",
    "preview_lead": 0.0,
    "hip_weight": 50.0,
    "hip_angle_limit": 0.55,
    "hip_guard": 0.9,
}
_OLD_PRESET = {
    # 旧结构对照（6 状态握髋模型 + legacy 纵向，preview_lead 0.0 为 R1 基线）。
    "model_variant": "held_hip",
    "forward_mode": "legacy",
    "preview_lead": 0.0,
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("stand", "pitch", "pitch_negative"), default="stand")
    parser.add_argument("--duration", type=float,
                        help="Simulated seconds; omit for an unbounded viewer run "
                             "or the headless default of 60 s.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/zero-balance/final")
    parser.add_argument("--viewer", action="store_true",
                        help="Open the passive MuJoCo viewer; runs at real-time pace.")
    parser.add_argument("--attitude-guard", type=float,
                        help="Experimental |roll|/|pitch| stop line in rad; default 0.785.")
    parser.add_argument("--speed-guard", type=float,
                        help="Experimental base speed stop line in m/s; default 1.60.")
    parser.add_argument("--wbc-backend", choices=("nullspace", "qp"), default=None,
                        help="WBC solver backend; omit to use SimulationConfig default (qp).")
    # E5：新方案的运行入口（全部默认 None，不传时行为与旧默认完全一致）。
    parser.add_argument("--model-variant", choices=("held_hip", "hip_input"), default=None,
                        help="Model variant; omit for the default (held_hip).")
    parser.add_argument("--forward-mode", choices=("legacy", "mpc"), default=None,
                        help="Longitudinal task source; omit for the default (legacy).")
    parser.add_argument("--preview-lead", type=float, default=None,
                        help="MPC trajectory preview lead [s]; omit for the default (0.0).")
    parser.add_argument("--hip-weight", type=float, default=None,
                        help="QPWeights.hip_angle for the hip_input model.")
    parser.add_argument("--hip-angle-limit", type=float, default=None,
                        help="MPC common-mode hip angle limit [rad] for the hip_input model.")
    parser.add_argument("--hip-guard", type=float, default=None,
                        help="Common-mode hip stop line [rad] (guard).")
    parser.add_argument("--preset", choices=("new", "old"), default=None,
                        help="One-shot preset: new (default scheme) or old (legacy comparison).")
    return parser


def _build_from_args(argv=None):
    """解析参数并构造 ``(SimulationConfig, MPCSettings | None, args)``。

    单独给出的参数优先于 ``--preset``；不传任何新参数时与旧默认完全一致（E5）。
    """
    args = _build_arg_parser().parse_args(argv)
    if args.preset == "new":
        preset = dict(_NEW_PRESET)
    elif args.preset == "old":
        preset = dict(_OLD_PRESET)
    else:
        preset = {}

    def pick(explicit, key, fallback):
        if explicit is not None:
            return explicit
        if key is not None and key in preset:
            return preset[key]
        return fallback

    base = SimulationConfig()
    config = replace(
        base,
        attitude_guard=pick(args.attitude_guard, None, base.attitude_guard),
        speed_guard=pick(args.speed_guard, None, base.speed_guard),
        wbc_backend=pick(args.wbc_backend, None, base.wbc_backend),
        model_variant=pick(args.model_variant, "model_variant", base.model_variant),
        forward_mode=pick(args.forward_mode, "forward_mode", base.forward_mode),
        preview_lead=pick(args.preview_lead, "preview_lead", base.preview_lead),
        hip_guard=pick(args.hip_guard, "hip_guard", base.hip_guard),
    )
    hip_weight = pick(args.hip_weight, "hip_weight", None)
    hip_limit = pick(args.hip_angle_limit, "hip_angle_limit", None)
    settings = None
    if hip_weight is not None or hip_limit is not None or config.model_variant == "hip_input":
        mpc_base = MPCSettings(compensate_hip_torque=False)
        weights = mpc_base.weights
        if hip_weight is not None:
            weights = replace(weights, hip_angle=float(hip_weight))
        settings = replace(
            mpc_base,
            weights=weights,
            hip_angle_limit=(float(hip_limit) if hip_limit is not None
                             else mpc_base.hip_angle_limit),
        )
    return config, settings, args


def main():
    config, settings, args = _build_from_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise SystemExit("Output must stay inside the workspace")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / f"{args.scenario}.log", encoding="utf-8")])
    fallback_settings = MPCSettings()
    effective_weights = settings.weights if settings is not None else fallback_settings.weights
    effective_limit = settings.hip_angle_limit if settings is not None else fallback_settings.hip_angle_limit
    _LOGGER.info("生效配置：model_variant=%s forward_mode=%s preview_lead=%.3f hip_angle=%.1f "
                 "hip_angle_limit=%.2f hip_guard=%.2f wbc_backend=%s",
                 config.model_variant, config.forward_mode, config.preview_lead,
                 effective_weights.hip_angle, effective_limit, config.hip_guard, config.wbc_backend)
    pitch = {"stand": 0., "pitch": 0.02, "pitch_negative": -0.02}[args.scenario]
    duration = args.duration
    if duration is None:
        duration = float("inf") if args.viewer else _DEFAULT_HEADLESS_DURATION
    simulation = MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings, initial_pitch=pitch)
    command = MotionCommand(velocity=0., yaw_rate=0.)
    if args.viewer:
        from mujoco import viewer as mujoco_viewer
        from APP.keyboard_task import KeyboardCommandSource

        keyboard = KeyboardCommandSource()
        _LOGGER.info("Viewer mode: real-time pacing; close the window to stop. "
                     "Keyboard up/down = speed, left/right = yaw; available=%s", keyboard.available)

        def driving(_time):
            return MotionCommand(velocity=keyboard.read(), yaw_rate=keyboard.read_yaw_rate())

        with mujoco_viewer.launch_passive(simulation.robot.model, simulation.robot.data) as viewer:
            summary = simulation.run(driving, duration, viewer, allow_driving=True)
    else:
        summary = simulation.run(command, duration)
    if summary["requested_seconds"] is not None and not np.isfinite(summary["requested_seconds"]):
        # 无限（viewer）运行的 requested_seconds 是 inf，JSON 拒绝；写为 null。
        summary["requested_seconds"] = None
    if simulation.records:
        payload = {"summary": summary, "records": simulation.records,
                   "simulation_config": asdict(simulation.config), "mpc_settings": asdict(simulation.mpc.settings),
                   "model_A": simulation.mpc.model.A.tolist(), "model_B": simulation.mpc.model.B.tolist()}
        with gzip.open(output / f"{args.scenario}.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False)
        (output / f"{args.scenario}.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    # A window close is a user stop, not a controller failure.
    return 0 if summary["completed"] or summary["stop_reason"] == "viewer_closed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
