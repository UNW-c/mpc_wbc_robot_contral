"""GBC486 模型加载、腿空间量与站立位形（M0）。

只依赖 mujoco / numpy，不依赖上层 circleg 的任何代码。

坐标约定：
- 机身系 x 前、y 左、z 上（与 MJCF 中 base 的坐标系一致）。
- 腿空间量以髋轴点 A 为极点，在机身系 xz 平面内取轮心：
  ``L = |r_xz|``，``phi = atan2(r_x, -r_z)``，即"从竖直向下量起，轮心在前为正"。
- 右腿：髋点为 AG 原点（jAB / jAG 同轴）；左腿：髋点为 IO 原点（jIJ / jIO 同轴）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_XML = PROJECT_ROOT / "MJCF" / "env.xml"

# 放置到地面时给的接触穿透量 [m]：保证新建 MjData 一次 forward 就能建立接触
# （circleg 方案 §7.3 记录过 ncon=0 陷阱）。
GROUND_PENETRATION = 1.0e-4
_NEWTON_TOL = 1.0e-11
_NEWTON_MAX_ITER = 60
_FD_STEP = 1.0e-7


@dataclass(frozen=True)
class LegSpec:
    name: str
    hip_body: str
    front_joint: str
    rear_joint: str
    passive_joints: tuple[str, ...]
    wheel_body: str
    wheel_joint: str
    wheel_geom: str
    loops: tuple[tuple[str, str], ...]

    @property
    def solve_joints(self) -> tuple[str, ...]:
        """站立位形求解的未知关节（不含轮自旋）。"""
        return (self.front_joint, self.rear_joint) + self.passive_joints


RIGHT = LegSpec("right", "AG", "jAB", "jAG", ("jGH", "jBE", "jEC", "jCF"),
                "wheel_right", "jwheel_right", "right_wheel_collision",
                (("EC-D", "AG-D"), ("CF-F", "GH-F")))
LEFT = LegSpec("left", "IO", "jIJ", "jIO", ("jOP", "jJM", "jMK", "jKN"),
               "wheel_left", "jwheel_left", "left_wheel_collision",
               (("IO-L", "MK-L"), ("OP-N", "KN-N")))
LEGS = (RIGHT, LEFT)
ACTUATED_LEG_JOINTS = ("jAB", "jAG", "jIJ", "jIO")


def load(path: Path | str = ENV_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, mujoco.MjData(model)


def _base_rotation(model, data) -> np.ndarray:
    return data.xmat[model.body("base").id].reshape(3, 3)


def leg_polar(model, data, leg: LegSpec) -> tuple[float, float]:
    """腿长 L 与腿角 phi（机身系，见模块说明）。需要 xpos 已更新。"""
    hip = data.xpos[model.body(leg.hip_body).id]
    wheel = data.xpos[model.body(leg.wheel_body).id]
    r = _base_rotation(model, data).T @ (wheel - hip)
    return float(math.hypot(r[0], r[2])), float(math.atan2(r[0], -r[2]))


def loop_residual(model, data, leg: LegSpec) -> tuple[np.ndarray, np.ndarray]:
    """闭链 site 差（机身系）：返回 (平面内 x/z 共 4 项, 平面外 y 共 2 项)。"""
    rot = _base_rotation(model, data).T
    inplane, outplane = [], []
    for s1, s2 in leg.loops:
        diff = rot @ (data.site_xpos[model.site(s1).id] - data.site_xpos[model.site(s2).id])
        inplane.extend((diff[0], diff[2]))
        outplane.append(diff[1])
    return np.asarray(inplane), np.asarray(outplane)


def qpos_addresses(model, joints) -> np.ndarray:
    """关节名列表 -> ``qpos`` 下标数组（供位形读写）。"""
    return np.array([int(model.joint(name).qposadr[0]) for name in joints])


def _qpos_addresses(model, joints) -> np.ndarray:
    return qpos_addresses(model, joints)


def _dof_addresses(model, joints) -> np.ndarray:
    return np.array([int(model.joint(name).dofadr[0]) for name in joints])


def solve_leg(model, data, leg: LegSpec, length: float, angle: float) -> int:
    """牛顿法求腿关节角，使闭链平面内闭合且 (L, phi) = 目标。返回迭代次数。

    机身位姿保持不变。未收敛时抛出 RuntimeError（调用方据此判定"不可达"）。
    """
    adr = _qpos_addresses(model, leg.solve_joints)
    target = np.array([length, angle])

    def residual(x):
        data.qpos[adr] = x
        mujoco.mj_kinematics(model, data)
        inplane, _ = loop_residual(model, data, leg)
        return np.r_[inplane, np.array(leg_polar(model, data, leg)) - target]

    x = data.qpos[adr].copy()
    for iteration in range(1, _NEWTON_MAX_ITER + 1):
        r = residual(x)
        if np.max(np.abs(r)) < _NEWTON_TOL:
            return iteration
        jac = np.empty((r.size, x.size))
        for i in range(x.size):
            step = np.zeros_like(x)
            step[i] = _FD_STEP
            jac[:, i] = (residual(x + step) - residual(x - step)) / (2 * _FD_STEP)
        delta = np.linalg.lstsq(jac, -r, rcond=None)[0]
        scale = min(1.0, 0.2 / max(float(np.max(np.abs(delta))), 1e-12))  # 单步不超过 0.2 rad
        x = x + scale * delta
    residual(x)
    raise RuntimeError(f"{leg.name} 腿 (L={length:.3f}, phi={angle:.3f}) 牛顿法未收敛")


def wheel_contact_points(model, data) -> np.ndarray:
    """两轮（右、左）圆柱碰撞体的最低点（世界系），需要 geom_xpos 已更新。"""
    points = []
    for leg in LEGS:
        g = model.geom(leg.wheel_geom).id
        axle = data.geom_xmat[g].reshape(3, 3)[:, 2]
        down = np.array([0.0, 0.0, -1.0])
        radial = down - axle * (axle @ down)
        radial /= np.linalg.norm(radial)
        points.append(data.geom_xpos[g] + model.geom_size[g, 0] * radial)
    return np.asarray(points)


def robot_com(model, data) -> np.ndarray:
    mujoco.mj_comPos(model, data)
    return data.subtree_com[model.body("base").id].copy()


@dataclass(frozen=True)
class StandingPose:
    length: float
    angle: float
    iterations: int
    qpos: np.ndarray


def standing_pose(model, data, length: float, angle: float | None = None,
                  penetration: float = GROUND_PENETRATION) -> StandingPose:
    """两腿对称 (L, phi)、机身水平、轮子落地的位形，写入 data 并 mj_forward。

    ``angle=None`` 时用割线法求静平衡腿角：整机质心 x = 两轮接触点 x 的均值
    （机身水平、零速度下俯仰静平衡，轮力矩为 0）。
    """
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[0:3] = (0.0, 0.0, 0.5)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_kinematics(model, data)
    iterations = 0

    def place(phi):
        nonlocal iterations
        for leg in LEGS:
            iterations += solve_leg(model, data, leg, length, phi)
        com = robot_com(model, data)
        return float(com[0] - np.mean(wheel_contact_points(model, data)[:, 0]))

    if angle is None:
        a0, a1 = 0.0, 0.05
        f0, f1 = place(a0), place(a1)
        for _ in range(30):
            if abs(f1) < 1e-10:
                break
            a0, a1, f0 = a1, a1 - f1 * (a1 - a0) / (f1 - f0), f1
            f1 = place(a1)
        else:
            raise RuntimeError(f"L={length:.3f} 静平衡腿角未收敛")
        angle = a1
    else:
        place(angle)
    lowest = float(np.min(wheel_contact_points(model, data)[:, 2]))
    data.qpos[2] -= lowest + penetration
    mujoco.mj_forward(model, data)
    return StandingPose(float(length), float(angle), iterations, data.qpos.copy())


def leg_jacobian(model, data, leg: LegSpec) -> tuple[np.ndarray, float]:
    """主动关节角速度 [front, rear] -> [L_dot, phi_dot] 的 2x2 雅可比（含闭链约束）。

    返回 (J, cond(J_passive))，J_passive 是平面内闭链行对 4 个被动关节的 4x4 块，
    它的条件数是闭链自身的奇异度指标。需要 mj_kinematics / mj_comPos 已更新。
    """
    nv = model.nv
    rot = _base_rotation(model, data).T
    active = _dof_addresses(model, (leg.front_joint, leg.rear_joint))
    passive = _dof_addresses(model, leg.passive_joints)
    rows = []
    for s1, s2 in leg.loops:
        j1, j2 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacSite(model, data, j1, None, model.site(s1).id)
        mujoco.mj_jacSite(model, data, j2, None, model.site(s2).id)
        diff = rot @ (j1 - j2)
        rows.extend((diff[0], diff[2]))
    loop = np.asarray(rows)
    jp, ja = loop[:, passive], loop[:, active]
    passive_rate = -np.linalg.solve(jp, ja)  # 4x2
    jw = np.zeros((3, nv))
    mujoco.mj_jacBody(model, data, jw, None, model.body(leg.wheel_body).id)
    jw = rot @ jw
    cartesian = jw[[0, 2]][:, active] + jw[[0, 2]][:, passive] @ passive_rate  # [x_dot, z_dot]
    hip = data.xpos[model.body(leg.hip_body).id]
    r = rot @ (data.xpos[model.body(leg.wheel_body).id] - hip)
    length2 = r[0] ** 2 + r[2] ** 2
    polar = np.array([[r[0], r[2]], [-r[2], r[0]]]) / np.array([[math.sqrt(length2)], [length2]])
    return polar @ cartesian, float(np.linalg.cond(jp))


def joint_range_margins(model, data, joints) -> dict[str, float]:
    """有限位关节到最近边界的余量 [rad]，负值表示越界。"""
    margins = {}
    for name in joints:
        j = model.joint(name)
        if not model.jnt_limited[j.id]:
            continue
        q = float(data.qpos[j.qposadr[0]])
        lo, hi = model.jnt_range[j.id]
        margins[name] = min(q - lo, hi - q)
    return margins
