"""输出级（每个物理步一次）：写执行器 + 推进物理 + 输出侧护栏。

这一级做什么
------------
- 把**最近一次**控制解写进 ``data.ctrl``（零阶保持：WBC 未到期的步写的是同一份解）
- 推进一个物理步，并更新里程计（用前后机身偏航的平均方向投影水平位移）
- 输出侧护栏：关节限位、状态非有限、姿态越界、速度越界 -> 返回停机原因

为什么单独成级
--------------
改前 ``d.ctrl`` 是在 WBC 分支里写的，写入时机被绑在 WBC 节拍上。现在写入与推进
每物理步各做一次，因此 WBC 换频率既不影响执行器更新时机，也不牵动采样级。

这一级不做什么
--------------
- 不算控制量（见 ``control.py``）、不采样（见 ``sampling.py``）、不配置日志
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from . import _paths  # noqa: F401
from Algorithm.geometry import wrap_angle  # noqa: E402


def write_actuator_output(data, torque: np.ndarray) -> None:
    """输出：把最近一次控制解写进执行器（零阶保持）。"""
    data.ctrl[:] = torque


def advance_physics(*, model, data, dynamics, config) -> tuple[float, str | None]:
    """推进一个物理步并做输出侧护栏，返回 ``(本步里程增量, 停机原因或 None)``。"""
    previous = data.qpos[:3].copy()
    yaw = dynamics.attitude()[2]
    mujoco.mj_step(model, data)
    middle = yaw + 0.5 * wrap_angle(dynamics.attitude()[2] - yaw)
    delta = float((data.qpos[:3] - previous) @ np.array([math.cos(middle), math.sin(middle), 0.0]))
    hits = dynamics.joint_limit_active()
    if hits:
        return delta, "joint_limit:" + ",".join(sorted(set(hits)))
    if not np.all(np.isfinite(np.r_[data.qpos, data.qvel])):
        return delta, "nonfinite_state"
    if max(abs(v) for v in dynamics.attitude()[:2]) > config.attitude_guard:
        return delta, "attitude_guard"
    if float(np.linalg.norm(data.qvel[:2])) > config.speed_guard:
        return delta, "speed_guard"
    return delta, None
