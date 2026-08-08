"""Interactive keyboard command source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_CONFIG, AppConfig


@dataclass(frozen=True)
class MotionCommand:
    forward_speed: float = 0.0
    yaw_rate: float = 0.0
    leg_length_rate: float = 0.0
    reset_posture: bool = False


class KeyboardCommandSource:
    """Cross-platform keyboard listener used by the interactive viewer.

    The listener is provided by :mod:`pynput`, which supports Windows and
    Linux desktop sessions without platform-specific key codes.  In a
    headless session where no keyboard backend is available, all commands
    safely remain zero.
    """

    SPEED_STEP = 1.5

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self.speed_step = float(config.keyboard.speed_step)
        self.yaw_command_rate = float(config.yaw.command_rate)
        self.leg_length_command_rate = float(config.leg.command_rate)
        self._pressed_keys: set[str] = set()
        self.available = False
        self._listener: Any | None = None
        self._start_listener()

    @staticmethod
    def _key_name(key: Any) -> str | None:
        char = getattr(key, "char", None)
        if isinstance(char, str) and char:
            return char.lower()
        name = getattr(key, "name", None)
        return str(name).lower() if name is not None else None

    def _on_press(self, key: Any) -> None:
        name = self._key_name(key)
        if name is not None:
            self._pressed_keys.add(name)

    def _on_release(self, key: Any) -> None:
        name = self._key_name(key)
        if name is not None:
            self._pressed_keys.discard(name)

    def _start_listener(self) -> None:
        try:
            from pynput import keyboard as pynput_keyboard

            self._listener = pynput_keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()
            self.available = True
        except (ImportError, OSError, RuntimeError):
            self._listener = None

    def _pressed(self, name: str) -> bool:
        return name in self._pressed_keys

    def read(self) -> float:
        up = self._pressed("up")
        down = self._pressed("down")
        if up and not down:
            return self.speed_step
        if down and not up:
            return -self.speed_step
        return 0.0

    def read_yaw_rate(self) -> float:
        left = self._pressed("left")
        right = self._pressed("right")
        if left and not right:
            return self.yaw_command_rate
        if right and not left:
            return -self.yaw_command_rate
        return 0.0

    def read_leg_length_rate(self) -> float:
        shorter = self._pressed("q")
        longer = self._pressed("e")
        if shorter and not longer:
            return -self.leg_length_command_rate
        if longer and not shorter:
            return self.leg_length_command_rate
        return 0.0

    def read_reset_posture(self) -> bool:
        return False

    def command(self) -> MotionCommand:
        return MotionCommand(
            self.read(),
            self.read_yaw_rate(),
            self.read_leg_length_rate(),
            self.read_reset_posture(),
        )


# Existing scripts and documentation use this name.
KeyboardVelocityStep = KeyboardCommandSource
