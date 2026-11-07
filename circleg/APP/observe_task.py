"""Controller status reporting."""

from __future__ import annotations

from typing import Any

from .chassis_controller import ChassisController


class StatusObserver:
    def format(
        self,
        data: Any,
        controller: ChassisController,
        torques: tuple[float, float, float, float],
    ) -> str:
        robot = controller.robot
        return (
            f"[t={data.time:6.3f}] pitch_est={controller.estimator.pitch: .5f} "
            f"pitch_rate_est={controller.estimator.pitch_rate: .5f} "
            f"v_est={controller.estimator.forward_velocity: .3f} "
            f"v_ref={controller.physical_velocity_reference: .2f} "
            f"v_lqr_ref={controller.velocity_reference: .2f} "
            f"q1={robot.joint_position('joint1'): .5f} "
            f"q3={robot.joint_position('joint3'): .5f} "
            f"yaw_est={controller.estimator.yaw: .5f} yawRef={controller.yaw_reference: .5f} "
            f"yawRateRef={controller.yaw_rate_reference: .2f} tauYaw={controller.yaw_torque: .3f} "
            f"tauR={torques[0]: .3f} tauL={torques[1]: .3f} "
            f"tauHip=({torques[2]: .3f},{torques[3]: .3f})"
        )

    def print(
        self,
        data: Any,
        controller: ChassisController,
        torques: tuple[float, float, float, float],
    ) -> None:
        print(self.format(data, controller, torques))


def print_status(
    data: Any,
    controller: ChassisController,
    torques: tuple[float, float, float, float],
) -> None:
    StatusObserver().print(data, controller, torques)
