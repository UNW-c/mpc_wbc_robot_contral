"""FL20（2026-10-12）：腿摆角世界系锚定的唯一换算来源。

本模块**只做一个换算**，纯函数式、不持有任何状态：

    phi_body_ref = world_target + pitch

含义：机体坐标系下的腿摆角参考 = 世界系目标摆角 + 机体俯仰角。这样腿在**世界系**里被
锚定到 ``world_target``（默认 -0.1263 = 站立静平衡 phi_world），与机体俯仰解耦。

来源：部署文档 FL20 任务 1。原 ``control.py::_phi_force`` 里直接写的 ``world_ref + pitch``
一并改为调用本函数，保证全工程只有这一处换算。
"""

# [rad] 世界系腿摆角锚定目标（站立静平衡 phi_world，静力矩自平衡）。来源：用户/部署文档 FL17B 第一节。
WORLD_TARGET_DEFAULT = -0.1263


def phi_body_reference(world_target: float, pitch: float) -> float:
    """机体系腿摆角参考：``world_target + pitch``（世界系锚定）。"""
    return float(world_target) + float(pitch)
