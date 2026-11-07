"""SDL gamepad commands and keyboard/gamepad arbitration."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Callable

from .config import DEFAULT_CONFIG, AppConfig
from .keyboard_task import KeyboardCommandSource, MotionCommand


XINPUT_GAMEPAD_B = 0x2000
XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE = 7_849
XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE = 8_689
XINPUT_GAMEPAD_TRIGGER_THRESHOLD = 30


@dataclass(frozen=True)
class XboxGamepadState:
    """Portable representation of the normalized SDL controller state."""

    left_x: int = 0
    left_y: int = 0
    right_x: int = 0
    right_y: int = 0
    left_trigger: int = 0
    right_trigger: int = 0
    buttons: int = 0


class _NativeGamepad(ctypes.Structure):
    _fields_ = [
        ("buttons", wintypes.WORD),
        ("left_trigger", ctypes.c_ubyte),
        ("right_trigger", ctypes.c_ubyte),
        ("thumb_lx", ctypes.c_short),
        ("thumb_ly", ctypes.c_short),
        ("thumb_rx", ctypes.c_short),
        ("thumb_ry", ctypes.c_short),
    ]


class _NativeState(ctypes.Structure):
    _fields_ = [
        ("packet_number", wintypes.DWORD),
        ("gamepad", _NativeGamepad),
    ]


class WindowsXInputReader:
    """Use XInput as a dependency-free Windows fallback."""

    _DLL_NAMES = ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll")

    def __init__(self) -> None:
        self._get_state: Any | None = None
        self._library: Any | None = None
        if os.name != "nt":
            return
        for dll_name in self._DLL_NAMES:
            try:
                library = ctypes.WinDLL(dll_name)
                get_state = library.XInputGetState
                get_state.argtypes = [wintypes.DWORD, ctypes.POINTER(_NativeState)]
                get_state.restype = wintypes.DWORD
                self._library = library
                self._get_state = get_state
                return
            except (AttributeError, OSError):
                continue

    def read(self, controller_index: int) -> XboxGamepadState | None:
        if self._get_state is None:
            return None
        native_state = _NativeState()
        if self._get_state(int(controller_index), ctypes.byref(native_state)) != 0:
            return None
        gamepad = native_state.gamepad
        return XboxGamepadState(
            left_x=int(gamepad.thumb_lx),
            left_y=int(gamepad.thumb_ly),
            right_x=int(gamepad.thumb_rx),
            right_y=int(gamepad.thumb_ry),
            left_trigger=int(gamepad.left_trigger),
            right_trigger=int(gamepad.right_trigger),
            buttons=int(gamepad.buttons),
        )


class SDLGamepadReader:
    """Read a standard SDL game controller on Windows or Linux.

    SDL's game-controller mapping normalizes Xbox-compatible pads across both
    platforms.  ``pygame`` is imported lazily so headless simulations do not
    require a connected display or controller.
    """

    _LEFT_X = 0
    _LEFT_Y = 1
    _RIGHT_X = 2
    _RIGHT_Y = 3
    _LEFT_TRIGGER = 4
    _RIGHT_TRIGGER = 5
    _B_BUTTON = 1

    def __init__(self) -> None:
        self.available = False
        self._api: Any | None = None
        self._controllers: dict[int, Any] = {}
        try:
            from pygame._sdl2 import controller

            controller.init()
            self._api = controller
            self.available = True
        except Exception:
            # Input is optional: a missing SDL backend simply selects keyboard.
            self._api = None

    @staticmethod
    def _trigger_to_xinput(value: int) -> int:
        value = int(value)
        if value < 0:
            return int(max(0, min(255, round((value + 32_768) * 255 / 32_768))))
        return int(max(0, min(255, round(value * 255 / 32_767))))

    def _controller(self, controller_index: int) -> Any | None:
        if self._api is None:
            return None
        try:
            if not self._api.is_controller(controller_index):
                self._controllers.pop(controller_index, None)
                return None
            controller = self._controllers.get(controller_index)
            if controller is None:
                controller = self._api.Controller(controller_index)
                self._controllers[controller_index] = controller
            return controller
        except Exception:
            self._controllers.pop(controller_index, None)
            return None

    def read(self, controller_index: int) -> XboxGamepadState | None:
        controller = self._controller(int(controller_index))
        if controller is None:
            return None
        try:
            return XboxGamepadState(
                left_x=int(controller.get_axis(self._LEFT_X)),
                left_y=-int(controller.get_axis(self._LEFT_Y)),
                right_x=int(controller.get_axis(self._RIGHT_X)),
                right_y=-int(controller.get_axis(self._RIGHT_Y)),
                left_trigger=self._trigger_to_xinput(
                    controller.get_axis(self._LEFT_TRIGGER)
                ),
                right_trigger=self._trigger_to_xinput(
                    controller.get_axis(self._RIGHT_TRIGGER)
                ),
                buttons=XINPUT_GAMEPAD_B if controller.get_button(self._B_BUTTON) else 0,
            )
        except Exception:
            self._controllers.pop(int(controller_index), None)
            return None


class GamepadReader:
    """Prefer SDL mapping and fall back to native XInput on Windows."""

    def __init__(self) -> None:
        self._sdl = SDLGamepadReader()
        self._xinput = WindowsXInputReader() if os.name == "nt" else None

    def read(self, controller_index: int) -> XboxGamepadState | None:
        state = self._sdl.read(controller_index)
        if state is not None:
            return state
        return None if self._xinput is None else self._xinput.read(controller_index)


# Kept as an import-compatible name for existing Windows scripts.
XInputReader = WindowsXInputReader


class XboxCommandSource:
    """Continuously map an SDL-compatible gamepad to the command interface.

    Left stick vertical controls forward speed.  Right stick horizontal controls
    yaw.  Left/right triggers shorten/lengthen the legs and B restores the
    configured default leg posture.  A disconnected pad reports ``None`` from
    :meth:`command`, allowing the arbiter to fall back to the keyboard.
    """

    def __init__(
        self,
        config: AppConfig = DEFAULT_CONFIG,
        controller_index: int = 0,
        state_reader: Callable[[int], XboxGamepadState | None] | None = None,
    ) -> None:
        if int(controller_index) < 0:
            raise ValueError("controller_index must be non-negative")
        self.controller_index = int(controller_index)
        self.maximum_speed = float(config.keyboard.speed_step)
        self.maximum_yaw_rate = float(config.yaw.command_rate)
        self.maximum_leg_rate = float(config.leg.command_rate)
        reader = GamepadReader()
        self._state_reader = state_reader if state_reader is not None else reader.read
        self.connected = False
        self._last_command = MotionCommand()
        self._sampled = False

    @staticmethod
    def _stick(value: int, deadzone: int) -> float:
        value = int(value)
        magnitude = abs(value)
        if magnitude <= deadzone:
            return 0.0
        maximum = 32_767 if value >= 0 else 32_768
        normalized = (magnitude - deadzone) / (maximum - deadzone)
        return float(normalized if value >= 0 else -normalized)

    @staticmethod
    def _trigger(value: int) -> float:
        value = int(value)
        if value <= XINPUT_GAMEPAD_TRIGGER_THRESHOLD:
            return 0.0
        return float(
            min(
                1.0,
                (value - XINPUT_GAMEPAD_TRIGGER_THRESHOLD)
                / (255 - XINPUT_GAMEPAD_TRIGGER_THRESHOLD),
            )
        )

    def command(self) -> MotionCommand | None:
        """Poll once and return a continuous command, or ``None`` if disconnected."""
        state = self._state_reader(self.controller_index)
        self.connected = state is not None
        if state is None:
            return None
        forward = self._stick(state.left_y, XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE)
        yaw = self._stick(state.right_x, XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE)
        leg_rate = self._trigger(state.right_trigger) - self._trigger(state.left_trigger)
        return MotionCommand(
            forward_speed=forward * self.maximum_speed,
            yaw_rate=-yaw * self.maximum_yaw_rate,
            leg_length_rate=leg_rate * self.maximum_leg_rate,
            reset_posture=bool(state.buttons & XINPUT_GAMEPAD_B),
        )

    def read(self) -> float:
        command = self.command()
        self._last_command = MotionCommand() if command is None else command
        self._sampled = True
        return self._last_command.forward_speed

    def read_yaw_rate(self) -> float:
        if not self._sampled:
            command = self.command()
            self._last_command = MotionCommand() if command is None else command
        return self._last_command.yaw_rate

    def read_leg_length_rate(self) -> float:
        if not self._sampled:
            command = self.command()
            self._last_command = MotionCommand() if command is None else command
        self._sampled = False
        return self._last_command.leg_length_rate

    def read_reset_posture(self) -> bool:
        if not self._sampled:
            command = self.command()
            self._last_command = MotionCommand() if command is None else command
        return self._last_command.reset_posture


class PreferredGamepadCommandSource:
    """Use a connected gamepad; otherwise pass through keyboard commands."""

    def __init__(
        self,
        keyboard: KeyboardCommandSource,
        gamepad: XboxCommandSource,
    ) -> None:
        self.keyboard = keyboard
        self.gamepad = gamepad
        self.active_source = "keyboard"
        self._last_command = MotionCommand()
        self._sampled = False

    def _sample(self) -> MotionCommand:
        gamepad_command = self.gamepad.command()
        if gamepad_command is not None:
            self.active_source = "gamepad"
            return gamepad_command
        self.active_source = "keyboard"
        return self.keyboard.command()

    def read(self) -> float:
        self._last_command = self._sample()
        self._sampled = True
        return self._last_command.forward_speed

    def read_yaw_rate(self) -> float:
        if not self._sampled:
            self._last_command = self._sample()
        return self._last_command.yaw_rate

    def read_leg_length_rate(self) -> float:
        if not self._sampled:
            self._last_command = self._sample()
        self._sampled = False
        return self._last_command.leg_length_rate

    def read_reset_posture(self) -> bool:
        if not self._sampled:
            self._last_command = self._sample()
        return self._last_command.reset_posture


# Xbot is commonly used as a shorthand for an Xbox-compatible controller.
XbotCommandSource = XboxCommandSource
