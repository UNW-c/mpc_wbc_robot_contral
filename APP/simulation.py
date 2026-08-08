"""Viewer/headless simulation lifecycle."""

from __future__ import annotations

import time

import numpy as np

from Algorithm.geometry import PlanarLegGeometry

from .chassis_controller import ChassisController
from .config import DEFAULT_CONFIG, AppConfig
from .keyboard_task import KeyboardCommandSource
from .mujoco_robot import MuJoCoRobot
from .observe_task import StatusObserver
from .xbox_task import PreferredGamepadCommandSource, XboxCommandSource


class SimulationApp:
    def __init__(
        self,
        terrain: str,
        duration: float,
        headless: bool,
        initial_pitch: float = 0.0,
        config: AppConfig = DEFAULT_CONFIG,
    ) -> None:
        self.terrain = terrain
        self.duration = max(float(duration), 0.0)
        self.headless = bool(headless)
        self.initial_pitch = float(initial_pitch)
        self.config = config
        self.robot = MuJoCoRobot.load(terrain, config)
        self.robot.set_initial_pitch(self.initial_pitch)
        self.controller = ChassisController(self.robot.model, self.robot.data, config)
        self.command_source = None
        if not self.headless:
            keyboard = KeyboardCommandSource(config)
            gamepad = XboxCommandSource(config)
            self.command_source = PreferredGamepadCommandSource(keyboard, gamepad)
        self.observer = StatusObserver()
        self.last_print = -0.2

    def print_startup(self) -> None:
        geometry = PlanarLegGeometry(
            self.config.leg.base_length,
            self.config.leg.link_length,
        )
        controller = self.controller
        print(f"Model: {self.robot.model_path}")
        print(
            f"Leg geometry: L(q)={geometry.base_length:.3f}-{geometry.link_length:.3f}*cos(q), "
            f"L(0)={geometry.length(0.0):.3f} m"
        )
        print(
            f"Pendulum: mass={controller.parameters.pendulum_mass:.4f} kg, "
            f"COM height={controller.parameters.com_height:.4f} m, "
            f"inertia={controller.parameters.pendulum_inertia:.6f} kg m^2"
        )
        print(f"K_R={controller.right_lqr.K.ravel()}")
        print(f"K_L={controller.left_lqr.K.ravel()}")
        print(
            "LQR state: [x_est, pitch_est, v_forward_est, pitch_rate_est], "
            "reference=[x_ref, pitch_ref(v_ref), v_ref, 0]"
        )
        print(
            f"LQR weights: Q={np.asarray(self.config.lqr.q_diagonal)} "
            f"R={self.config.lqr.r:.3f}; fixed over leg-length range; "
            f"reference speed rate={self.config.lqr.velocity_reference_rate:.2f} m/s^2; "
            f"pitch reference limit={self.config.lqr.pitch_reference_limit:.3f} rad"
        )
        print("Wheel convention: right positive forward, left negative forward")
        print("Hip convention: direct q1/q3 encoder coordinates, both targets q=0")
        if not self.headless:
            print(
                f"Input: SDL gamepad preferred; keyboard fallback. Max speed = "
                f"+/-{self.config.keyboard.speed_step:.2f} m/s"
            )
            print("Gamepad: B = restore default leg posture")

    def step_once(self) -> tuple[float, float, float, float]:
        if self.robot.time < self.last_print:
            self.last_print = self.robot.time - 0.2
        torques = self.controller.apply(self.robot.timestep, self.command_source)
        self.robot.step()
        if self.robot.time - self.last_print >= 0.2:
            self.observer.print(self.robot.data, self.controller, torques)
            self.last_print = self.robot.time
        return torques

    def run(self) -> None:
        self.print_startup()
        if self.headless:
            while self.robot.time < self.duration:
                self.step_once()
            return

        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(self.robot.model, self.robot.data) as viewer:
            while viewer.is_running() and self.robot.time < self.duration:
                start = time.perf_counter()
                self.step_once()
                viewer.sync()
                remaining = self.robot.timestep - (time.perf_counter() - start)
                if remaining > 0.0:
                    time.sleep(remaining)


def run(
    terrain: str,
    duration: float,
    headless: bool,
    initial_pitch: float,
    config: AppConfig = DEFAULT_CONFIG,
) -> None:
    SimulationApp(terrain, duration, headless, initial_pitch, config).run()
