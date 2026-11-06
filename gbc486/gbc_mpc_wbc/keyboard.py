"""交互式键盘指令源（只读按键状态）。

``pynput`` 负责跨平台按键监听；无图形会话或未安装时，所有按键保持未按下，
不会阻塞仿真。速度步长由调用方给定，偏航与腿长命令率沿用项目默认值。
"""
from __future__ import annotations

from typing import Any


class KeyboardCommandSource:
    """读取方向键 / WASD / 空格 / R 的按下状态。"""

    def __init__(self, speed_step: float = 2.5, yaw_rate: float = 9.6,
                 leg_length_rate: float = 0.120) -> None:
        self.speed_step = float(speed_step)
        self.yaw_command_rate = float(yaw_rate)
        self.leg_length_command_rate = float(leg_length_rate)
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

    def pressed_key(self, name: str) -> bool:
        """某个键当前是否按下（只读）。空格键名为 ``"space"``。"""
        return self._pressed(name)
