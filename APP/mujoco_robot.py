"""Object-oriented MuJoCo model/data access for the application layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from Algorithm.geometry import quaternion_pitch, quaternion_yaw

from .config import DEFAULT_CONFIG, AppConfig


def joint_position(model: Any, data: Any, name: str) -> float:
    joint = model.joint(name)
    return float(data.qpos[int(joint.qposadr[0])])


def joint_velocity(model: Any, data: Any, name: str) -> float:
    joint = model.joint(name)
    return float(data.qvel[int(joint.dofadr[0])])


def set_joint_position(model: Any, data: Any, name: str, value: float) -> None:
    joint = model.joint(name)
    data.qpos[int(joint.qposadr[0])] = float(value)


def body_pitch(data: Any) -> float:
    return quaternion_pitch(np.asarray(data.qpos[3:7], dtype=float))


def body_pitch_rate(data: Any) -> float:
    return float(data.qvel[4])


def body_yaw(data: Any) -> float:
    return quaternion_yaw(np.asarray(data.qpos[3:7], dtype=float))


GROUND_GEOM_PREFIX = "field_"
LEGACY_GROUND_GEOM_NAME = "floor"


def ground_geom_ids(model: Any) -> tuple[int, ...]:
    """所有构成“地面”的 geom id。

    地形不止一种：``plane`` / ``rough`` 用一个名为 ``floor`` 的平面，``field`` 用
    RMUC 2026 场地（41 个 ``field_*`` 棱柱 + 4 个 ``field_boundary_*`` 挡板）。
    ``MuJoCoDynamics.sensed_contact`` 原来只认 ``model.geom("floor")``，在 ``field``
    地形上轮子压到中心高地 / 斜坡 / 台阶时不会被判成着地，因此改为按名字集合识别。

    判定条件：名字等于 ``floor``（兼容平面地形）或以 ``field_`` 开头，且参与碰撞
    （``contype`` 或 ``conaffinity`` 非零，用来排除不参与碰撞的 ``field_visual``）。

    新增或重命名地形 geom 时必须同步更新本函数，否则着地判定会失效。
    """
    ids = []
    for index in range(int(model.ngeom)):
        name = model.geom(index).name or ""
        if name != LEGACY_GROUND_GEOM_NAME and not name.startswith(GROUND_GEOM_PREFIX):
            continue
        if not (model.geom_contype[index] or model.geom_conaffinity[index]):
            continue
        ids.append(index)
    return tuple(ids)


@dataclass
class MuJoCoRobot:
    """Owns all simulator-facing reads and writes used by the controller."""

    model: Any
    data: Any
    model_path: Path | None = None

    @classmethod
    def load(cls, terrain: str, config: AppConfig = DEFAULT_CONFIG) -> "MuJoCoRobot":
        path = config.model_paths[terrain]
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        robot = cls(model, data, path)
        robot.initialize_pose(config.leg.angle_zero)
        return robot

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def joint_position(self, name: str) -> float:
        return joint_position(self.model, self.data, name)

    def joint_velocity(self, name: str) -> float:
        return joint_velocity(self.model, self.data, name)

    def joint_load(self, name: str) -> float:
        dof = int(self.model.joint(name).dofadr[0])
        return float(self.data.qfrc_bias[dof] + self.data.qfrc_constraint[dof])

    def set_joint_position(self, name: str, value: float) -> None:
        set_joint_position(self.model, self.data, name, value)

    def sensor(self, name: str) -> np.ndarray:
        sensor_id = int(self.model.sensor(name).id)
        start = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        return np.asarray(self.data.sensordata[start : start + dimension], dtype=float).copy()

    def wheel_angles(self) -> np.ndarray:
        return np.array(
            [self.joint_position("joint2"), self.joint_position("joint4")],
            dtype=float,
        )

    def hip_angles(self) -> np.ndarray:
        return np.array(
            [self.joint_position("joint1"), self.joint_position("joint3")],
            dtype=float,
        )

    def initialize_pose(self, hip_angle: float) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.set_joint_position("joint1", hip_angle)
        self.set_joint_position("joint2", 0.0)
        self.set_joint_position("joint3", hip_angle)
        self.set_joint_position("joint4", 0.0)
        mujoco.mj_forward(self.model, self.data)
        self._place_wheels_on_ground()
        # MuJoCo's viewer reset restores model.qpos0.  Keep it equal to the
        # loaded contact pose used to derive the LQR model.
        self.model.qpos0[:] = self.data.qpos

    def _place_wheels_on_ground(self) -> None:
        wheel_geom_ids = (
            int(self.model.geom("right_wheel_collision").id),
            int(self.model.geom("left_wheel_collision").id),
        )
        lowest_wheel_z = max(
            float(self.data.geom_xpos[geom_id, 2] - self.model.geom_size[geom_id, 0])
            for geom_id in wheel_geom_ids
        )
        self.data.qpos[2] -= lowest_wheel_z
        mujoco.mj_forward(self.model, self.data)

    def set_initial_pitch(self, pitch: float) -> None:
        if abs(float(pitch)) <= 0.0:
            return
        half = 0.5 * float(pitch)
        self.data.qpos[3:7] = np.array(
            [np.cos(half), 0.0, np.sin(half), 0.0],
            dtype=float,
        )
        mujoco.mj_forward(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def apply_torques(
        self,
        right_wheel: float,
        left_wheel: float,
        right_hip: float,
        left_hip: float,
    ) -> None:
        self.data.ctrl[self.model.actuator("j1_ctrl").id] = float(right_hip)
        self.data.ctrl[self.model.actuator("j3_ctrl").id] = float(left_hip)
        self.data.ctrl[self.model.actuator("j2_ctrl").id] = float(right_wheel)
        self.data.ctrl[self.model.actuator("j4_ctrl").id] = float(left_wheel)


def build_model(
    terrain: str,
    config: AppConfig = DEFAULT_CONFIG,
) -> tuple[Any, Any, Path]:
    """Compatibility factory returning the traditional model/data tuple."""
    robot = MuJoCoRobot.load(terrain, config)
    assert robot.model_path is not None
    return robot.model, robot.data, robot.model_path
