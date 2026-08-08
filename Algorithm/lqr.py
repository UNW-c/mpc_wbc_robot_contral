"""LQR design and independent wheel control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are


def lqr_gain_from_linearization(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Compute a continuous-time LQR gain from a local linear model."""
    riccati = solve_continuous_are(A, B, Q, R)
    return np.asarray(np.linalg.solve(R, B.T @ riccati), dtype=float)


@dataclass(frozen=True)
class PendulumParameters:
    pendulum_mass: float
    wheel_mass: float
    wheel_inertia: float
    com_height: float
    pendulum_inertia: float

    @classmethod
    def from_model(cls, model: Any, data: Any) -> "PendulumParameters":
        pendulum_names = ("base_link", "joint1_Link", "joint3_Link")
        wheel_names = ("joint2_Link", "joint4_Link")
        pendulum_ids = [int(model.body(name).id) for name in pendulum_names]
        wheel_ids = [int(model.body(name).id) for name in wheel_names]

        masses = np.asarray(model.body_mass[pendulum_ids], dtype=float)
        coms = np.asarray(data.xipos[pendulum_ids], dtype=float)
        mass = float(np.sum(masses))
        com = np.sum(masses[:, None] * coms, axis=0) / mass

        wheel_mass = float(np.mean(model.body_mass[wheel_ids]))
        wheel_inertia = float(np.mean([model.body_inertia[i][1] for i in wheel_ids]))
        wheel_z = float(np.mean(data.xipos[wheel_ids, 2]))
        height = float(com[2] - wheel_z)
        if height <= 1.0e-4:
            raise ValueError(f"q=0 is not upright: pendulum COM height={height:.6f} m")

        inertia = 0.0
        for body_mass, body_id, body_com in zip(masses, pendulum_ids, coms):
            offset = body_com - com
            inertia += float(model.body_inertia[body_id][1])
            inertia += float(body_mass) * (offset[0] ** 2 + offset[2] ** 2)
        return cls(mass, wheel_mass, wheel_inertia, height, inertia)


@dataclass(frozen=True)
class LQRDesign:
    """One shared local linearization used by both wheel controllers."""

    A: np.ndarray
    B: np.ndarray
    K: np.ndarray

    @classmethod
    def from_model(
        cls,
        model: Any,
        data: Any,
        parameters: PendulumParameters,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> "LQRDesign":
        # ``parameters`` is retained as an explicit design input and for
        # diagnostics, even though the current gain is contact-linearized.
        del parameters
        A, B = cls._measure_contact_linearization(model, data)
        K = lqr_gain_from_linearization(A, B, Q, R)
        return cls(A, B, K)

    @staticmethod
    def _measure_contact_linearization(model: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
        """Numerically linearize [x, pitch, x_dot, pitch_dot] in rolling contact."""
        qpos0 = np.asarray(data.qpos, dtype=float).copy()
        qvel0 = np.asarray(data.qvel, dtype=float).copy()
        qpos_eps = 1.0e-5
        qvel_eps = 1.0e-5

        rolling_coefficients: dict[int, tuple[float, float, float]] = {}
        wheel_body_ids = {
            int(model.body("joint2_Link").id): int(model.joint("joint2").dofadr[0]),
            int(model.body("joint4_Link").id): int(model.joint("joint4").dofadr[0]),
        }
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            wheel_body = body2 if body1 == 0 else body1
            wheel_dof = wheel_body_ids.get(wheel_body)
            if wheel_dof is None:
                continue
            jacobian_position = np.zeros((3, model.nv), dtype=float)
            jacobian_rotation = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                contact.pos,
                wheel_body,
            )
            rolling_coefficients[wheel_dof] = (
                float(jacobian_position[0, 0]),
                float(jacobian_position[0, 4]),
                float(jacobian_position[0, wheel_dof]),
            )
        if len(rolling_coefficients) != 2:
            raise ValueError("Expected both wheel-ground rolling contacts for LQR linearization")

        def rolling_qvel(x_dot: float, pitch_dot: float) -> np.ndarray:
            velocity = qvel0.copy()
            velocity[0] = float(x_dot)
            velocity[4] = float(pitch_dot)
            for wheel_dof, (x_jacobian, pitch_jacobian, wheel_jacobian) in rolling_coefficients.items():
                velocity[wheel_dof] = -(
                    x_jacobian * x_dot + pitch_jacobian * pitch_dot
                ) / wheel_jacobian
            return velocity

        def place_wheels_on_ground(qpos: np.ndarray) -> None:
            scratch = mujoco.MjData(model)
            scratch.qpos[:] = qpos
            scratch.qvel[:] = 0.0
            mujoco.mj_forward(model, scratch)
            wheel_geoms = (
                int(model.geom("right_wheel_collision").id),
                int(model.geom("left_wheel_collision").id),
            )
            lowest_wheel_z = max(
                float(scratch.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0])
                for geom_id in wheel_geoms
            )
            qpos[2] -= lowest_wheel_z

        def derivative(
            qpos: np.ndarray,
            qvel: np.ndarray,
            right_torque: float = 0.0,
        ) -> np.ndarray:
            scratch = mujoco.MjData(model)
            scratch.qpos[:] = qpos
            scratch.qvel[:] = qvel
            scratch.ctrl[:] = 0.0
            scratch.ctrl[model.actuator("j2_ctrl").id] = right_torque
            scratch.ctrl[model.actuator("j4_ctrl").id] = -right_torque
            mujoco.mj_forward(model, scratch)
            return np.array(
                [scratch.qvel[0], scratch.qvel[4], scratch.qacc[0], scratch.qacc[4]],
                dtype=float,
            )

        A = np.zeros((4, 4), dtype=float)
        A[0, 2] = 1.0
        A[1, 3] = 1.0

        plus_q = qpos0.copy()
        minus_q = qpos0.copy()
        mujoco.mju_quatIntegrate(plus_q[3:7], np.array([0.0, qpos_eps, 0.0]), 1.0)
        mujoco.mju_quatIntegrate(minus_q[3:7], np.array([0.0, -qpos_eps, 0.0]), 1.0)
        place_wheels_on_ground(plus_q)
        place_wheels_on_ground(minus_q)
        A[:, 1] = (derivative(plus_q, qvel0) - derivative(minus_q, qvel0)) / (2.0 * qpos_eps)

        A[:, 2] = (
            derivative(qpos0, rolling_qvel(qvel_eps, 0.0))
            - derivative(qpos0, rolling_qvel(-qvel_eps, 0.0))
        ) / (2.0 * qvel_eps)
        A[:, 3] = (
            derivative(qpos0, rolling_qvel(0.0, qvel_eps))
            - derivative(qpos0, rolling_qvel(0.0, -qvel_eps))
        ) / (2.0 * qvel_eps)

        B = np.zeros((4, 1), dtype=float)
        B[:, 0] = (
            derivative(qpos0, qvel0, 1.0) - derivative(qpos0, qvel0, -1.0)
        ) / 2.0
        return A, B


class WheelLQR:
    """One independent wheel controller using a shared fixed LQR design."""

    def __init__(
        self,
        design: LQRDesign,
        local_sign: float,
        torque_limit: float,
    ) -> None:
        self.A = design.A
        self.B = design.B
        self.K = design.K
        self.local_sign = float(local_sign)
        self.torque_limit = float(torque_limit)

    def torque(
        self,
        x: float,
        x_reference: float,
        pitch: float,
        pitch_reference: float,
        velocity: float,
        velocity_reference: float,
        pitch_rate: float,
        equilibrium_torque: float,
    ) -> float:
        local_state = np.array(
            [
                self.local_sign * (x - x_reference),
                self.local_sign * (pitch - pitch_reference),
                self.local_sign * (velocity - velocity_reference),
                self.local_sign * pitch_rate,
            ],
            dtype=float,
        )
        local_equilibrium_torque = self.local_sign * equilibrium_torque
        command = local_equilibrium_torque - (self.K @ local_state).item()
        return float(np.clip(command, -self.torque_limit, self.torque_limit))
