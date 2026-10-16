"""腿摆角世界系锚定的唯一换算来源（R1 起）。

本模块纯函数式、不持有状态，集中提供全部“世界系腿角”换算：

    phi_eq(l_ref)                     静平衡世界系摆角随腿长
    phi_body_reference(world_target, pitch)   世界系目标 -> 机体系参考
    world_from_body(phi_body, pitch)          机体系角 -> 世界系角（上式逆）

来源：部署文档 FL20 任务 1。R1（2026-10-17，用户同意）把 ``control.py`` 的
``_phi_eq`` 与重影常量 ``PHI_WORLD_REFERENCE`` / ``PHI_EQ_SLOPE`` / ``PHI_EQ_ANCHOR_LENGTH``
一并并入本模块，数值一字不改，使“世界系腿角”只有这一处来源。
"""

# [rad] 世界系腿摆角锚定目标（站立静平衡 phi_world，静力矩自平衡）。
# 来源：用户/部署文档 FL17B 第一节；与站立求解的 pose.angle 同值。
WORLD_TARGET_DEFAULT = -0.1263

# [rad/m] 静平衡摆角对腿长的斜率；[m] 锚点腿长。
# 来源：部署文档离线拟合（原 control.py 常量，R1 搬入本模块，数值未改）。
PHI_EQ_SLOPE = 0.97
PHI_EQ_ANCHOR_LENGTH = 0.28


def phi_eq(l_ref) -> float:
    """静平衡摆角（世界系）随腿长参考：``WORLD_TARGET_DEFAULT + PHI_EQ_SLOPE*(L-0.28)``。
    拟合区间只到 L=0.14；L=0.119 属外推（已知限制）。"""
    return WORLD_TARGET_DEFAULT + PHI_EQ_SLOPE * (float(l_ref) - PHI_EQ_ANCHOR_LENGTH)


def phi_body_reference(world_target: float, pitch: float) -> float:
    """世界系目标摆角 -> 机体系参考：``world_target + pitch``（世界系锚定）。"""
    return float(world_target) + float(pitch)


def world_from_body(phi_body: float, pitch: float) -> float:
    """机体系摆角 -> 世界系摆角：``phi_body - pitch``（``phi_body_reference`` 的逆）。"""
    return float(phi_body) - float(pitch)
