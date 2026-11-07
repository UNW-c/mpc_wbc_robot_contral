"""M1~M3：髋角极限的几何 / 动力学分析（不跑闭环仿真）。

M1：曲柄 / 轮与机身的几何干涉（凸包判断，排除铰链附近的顶点）
M2：离地间隙（机身网格最低点随髋角 / 俯仰的变化）
M3：控制权限饱和（稳态族的 da/dq_c）

运行：
    .venv/Scripts/python.exe -m tests.tuning.hip_limit_geometry
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
from scipy.spatial import ConvexHull

from APP.mujoco_balance_model import HipInputModelBuilder
from APP.mujoco_mpc_wbc import MuJoCoDynamics
from APP.mujoco_robot import MuJoCoRobot

_HINGE_EXCLUSION = 0.015   # [m] 铰链轴线附近排除半径
_CLEARANCE_MIN = 0.015     # [m] 离地间隙门槛（M2）


def _vertices_world(model, data, body_name: str) -> np.ndarray:
    """收集某个 body 下所有 mesh geom 的世界系顶点。"""
    points = []
    for geom in range(model.ngeom):
        mesh_id = int(model.geom_dataid[geom])
        if mesh_id < 0:
            continue
        if model.body(int(model.geom_bodyid[geom])).name != body_name:
            continue
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[start:start + count]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        points.append(vertices @ rotation.T + data.geom_xpos[geom])
    return np.vstack(points) if points else np.zeros((0, 3))


def _axis_distance(points: np.ndarray, anchor: np.ndarray, axis: np.ndarray) -> np.ndarray:
    offset = points - anchor
    along = offset @ axis
    perpendicular = offset - np.outer(along, axis)
    return np.linalg.norm(perpendicular, axis=1)


def _triangles_world(model, data, body_name: str) -> np.ndarray:
    """某个 body 下所有 mesh geom 的三角形（世界系，形状 (N, 3, 3)）。"""
    triangles = []
    for geom in range(model.ngeom):
        mesh_id = int(model.geom_dataid[geom])
        if mesh_id < 0:
            continue
        if model.body(int(model.geom_bodyid[geom])).name != body_name:
            continue
        vert_start = int(model.mesh_vertadr[mesh_id])
        vert_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = model.mesh_vert[vert_start:vert_start + vert_count]
        faces = model.mesh_face[face_start:face_start + face_count]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        world = vertices @ rotation.T + data.geom_xpos[geom]
        triangles.append(world[faces])
    return np.vstack(triangles) if triangles else np.zeros((0, 3, 3))


def _inside_mesh(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """射线法（+x 方向）判断点是否在多面体网格内部（凹形也适用）。

    Möller-Trumbore；统计每个点与所有三角形的交点个数，奇数 = 内部。
    """
    if points.size == 0 or triangles.size == 0:
        return np.zeros(len(points), dtype=bool)
    direction = np.array([1.0, 0.0, 0.0])
    counts = np.zeros(len(points), dtype=int)
    for triangle in triangles:
        edge1 = triangle[1] - triangle[0]
        edge2 = triangle[2] - triangle[0]
        h = np.cross(direction, edge2)
        a = float(edge1 @ h)
        if abs(a) < 1e-12:
            continue
        f = 1.0 / a
        s = points - triangle[0]
        u = f * (s @ h)
        q = np.cross(s, edge1)
        v = f * (q @ direction)
        t = f * (q @ edge2)
        hit = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-9)
        counts += hit.astype(int)
    return (counts % 2) == 1


def m1_interference(model, data) -> dict:
    """M1：扫描髋角，找曲柄 / 轮顶点进入机身凸包的首次角度。"""
    hip_joints = (model.joint("joint1").id, model.joint("joint3").id)
    hip_qpos = [int(model.joint(n).qposadr[0]) for n in ("joint1", "joint3")]
    legs = (("joint1_Link", "joint2_Link"), ("joint3_Link", "joint4_Link"))
    base_names = ("base_link",)

    # 机身固定：三角形与凸包只需算一次
    base_triangles = _triangles_world(model, data, "base_link")
    base_hull = ConvexHull(base_triangles.reshape(-1, 3))
    results = {"positive": None, "negative": None, "detail": []}
    for leg_index, leg_bodies in enumerate(legs):
        for direction, angles in (("positive", range(0, 181)),
                                  ("negative", range(0, -181, -1))):
            first = None
            for degree in angles:
                q = math.radians(degree)
                data.qpos[hip_qpos[leg_index]] = q
                mujoco.mj_kinematics(model, data)
                leg_points = np.vstack([_vertices_world(model, data, name) for name in leg_bodies])
                # 排除铰链附近（本来就重叠）
                joint = hip_joints[leg_index]
                anchor = np.asarray(data.xanchor[joint], dtype=float)
                axis = np.asarray(data.xaxis[joint], dtype=float)
                keep = _axis_distance(leg_points, anchor, axis) > _HINGE_EXCLUSION
                query = leg_points[keep]
                if query.size == 0:
                    continue
                # 凸包粗筛（凹形机身只是候选，不直接判干涉）-> 射线法精确判断
                candidates = query[np.all(
                    query @ base_hull.equations[:, :3].T + base_hull.equations[:, 3] <= 1e-9,
                    axis=1)]
                if candidates.size == 0:
                    continue
                inside = _inside_mesh(candidates, base_triangles)
                count = int(np.sum(inside))
                if count > 0:
                    first = dict(degrees=degree, count=count, max_depth=0.0)
                    break
            key = direction
            if results[key] is None or (first is not None and first["degrees"] < results[key]["degrees"]):
                results[key] = first
            results["detail"].append((leg_index, direction, first))
    return results


def m2_clearance(model, data, robot) -> dict:
    """M2：机身网格最低点离地高度（q x theta 网格）。"""
    hip_qpos = [int(model.joint(n).qposadr[0]) for n in ("joint1", "joint3")]
    home = data.qpos.copy()
    contact_z = float(data.geom_xpos[int(model.geom("right_wheel_collision").id), 2])
    rows = []
    for q_deg in range(0, 110, 5):
        q = math.radians(q_deg)
        for theta_deg in (-10.0, 0.0, 10.0):
            data.qpos[:] = home
            data.qpos[hip_qpos[0]] = q
            data.qpos[hip_qpos[1]] = -q
            # theta：机身绕自身原点倾斜（纯几何，不做接触点补偿）
            mujoco.mju_quatIntegrate(data.qpos[3:7],
                                     np.array([0.0, math.radians(theta_deg), 0.0]), 1.0)
            # 落轮：把机身沿 z 平移，让两轮最低点贴地（摆腿后机身会真正下沉）
            robot._place_wheels_on_ground()
            mujoco.mj_kinematics(model, data)
            lowest = float(np.min(_vertices_world(model, data, "base_link")[:, 2]))
            rows.append((q_deg, theta_deg, lowest))
    # 找出满足门槛的最大 q（对每个 theta）
    summary = {}
    for theta_deg in (-10.0, 0.0, 10.0):
        candidates = [q for q, th, low in rows if th == theta_deg and low >= _CLEARANCE_MIN]
        summary[theta_deg] = max(candidates) if candidates else 0
    return {"summary_degrees": summary, "rows": rows, "contact_z": contact_z}


def m3_authority(robot) -> dict:
    """M3：稳态族 q_c(a) 与 da/dq_c，找权限降到 50% 的角度。"""
    # M1/M2 会改 robot 的 qpos；HipInputModelBuilder 以当前位形为名义工作点，
    # 所以这里用干净的实例（否则稳态族整体偏移）。
    del robot
    robot = MuJoCoRobot.load("plane")
    builder = HipInputModelBuilder(MuJoCoDynamics(robot))
    table = []
    for a in np.arange(0.0, 2.55, 0.1):
        solution = builder.steady_state(float(a))
        table.append((float(a), abs(float(solution["q_c"]))))
    accelerations = np.array([row[0] for row in table])
    hip_angles = np.array([row[1] for row in table])
    gradient = np.gradient(accelerations, hip_angles)
    base = float(gradient[1]) if gradient.size > 1 else float("nan")
    threshold = 0.5 * base
    q_auth = None
    a_auth = None
    for index in range(1, len(table)):
        if abs(gradient[index]) < abs(threshold):
            q_auth = float(hip_angles[index])
            a_auth = float(accelerations[index])
            break
    return {"table": table, "gradient": gradient, "base_gradient": base,
            "q_auth": q_auth, "a_auth": a_auth}


def main() -> int:
    robot = MuJoCoRobot.load("plane")
    model, data = robot.model, robot.data

    print("=== M1 几何干涉（凸包粗筛 + 射线法，排除铰链 15 mm 内的顶点）===")
    # 自检：机身内部点应为 True、远处点应为 False
    check_triangles = _triangles_world(model, data, "base_link")
    center = check_triangles.reshape(-1, 3).mean(axis=0)
    inside_center = bool(_inside_mesh(center[None, :], check_triangles)[0])
    outside = bool(_inside_mesh((center + np.array([5.0, 0.0, 0.0]))[None, :], check_triangles)[0])
    print(f"  自检：机身中心点在内部={inside_center}（期望 True）、远点={outside}（期望 False）")
    interference = m1_interference(model, data)
    for direction, result in (("正向", interference["positive"]), ("负向", interference["negative"])):
        if result is None:
            print(f"  {direction}: 全范围（±180°）无干涉")
        else:
            print(f"  {direction}: 首次干涉 {result['degrees']:+d}°"
                  f"（{result['count']} 个顶点，最大穿入 {result['max_depth'] * 1000:.1f} mm）")

    print("\n=== M2 离地间隙（门槛 15 mm）===")
    clearance = m2_clearance(model, data, robot)
    print(f"{'q(°)':>6}{'θ=-10°':>10}{'θ=0°':>10}{'θ=+10°':>10}   机身最低点 (m)")
    for q_deg, theta_deg, lowest in clearance["rows"]:
        if theta_deg == 0.0:
            value_neg = next(low for qq, th, low in clearance["rows"]
                             if qq == q_deg and th == -10.0)
            value_pos = next(low for qq, th, low in clearance["rows"]
                             if qq == q_deg and th == 10.0)
            print(f"{q_deg:>6}{value_neg:>10.4f}{lowest:>10.4f}{value_pos:>10.4f}")
    print("  满足 15 mm 的最大 q：" + ", ".join(
        f"θ={th:+.0f}°→{q}°" for th, q in clearance["summary_degrees"].items()))

    print("\n=== M3 控制权限（da/dq_c）===")
    authority = m3_authority(robot)
    print(f"  q=0 处的 da/dq_c = {authority['base_gradient']:.3f} (m/s²)/rad")
    for a, q_c in authority["table"][::5]:
        print(f"  a={a:.1f} m/s² -> q_c={q_c:.3f} rad")
    if authority["q_auth"] is None:
        print("  权限未降到 50%（扫描范围内）")
    else:
        print(f"  权限降到 50% 的角度 q_auth = {authority['q_auth']:.3f} rad"
              f"（此时 a = {authority['a_auth']:.2f} m/s²）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
