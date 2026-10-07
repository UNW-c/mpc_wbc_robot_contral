"""腿长 / 腿摆角的**解析推导**（含闭链雅可比）——运行期不调用 MuJoCo 位置 API。

为什么要它
----------
改前 ``robot.leg_polar`` 是每拍去问 MuJoCo 要 ``data.xpos[hip]`` / ``data.xpos[wheel]``，
``robot.leg_jacobian`` 走 ``mj_jacSite`` / ``mj_jacBody``。那等于把腿长、腿摆角、以及
力矩分配用的雅可比都外包给仿真器的位置接口。本模块自己推：

几何常数（只读一次，__init__）
    - 各连杆相对父体的偏置（在**参考位形**下转换到机身系，取 xz 平面分量）
    - 每个肢体的关节转轴在机身系的 **y 分量符号**（决定平面内转角的正负）
    - 两条闭链 site 相对所在体的平面偏置
    来源全是模型常量：``model.body_pos`` / ``model.body_quat`` / ``model.jnt_axis`` /
    ``model.site_pos`` + 一次 ``mj_kinematics``（只在参考位形上跑一次，不参与运行期）。

运行期（纯解析）
    1) 腿长与腿摆角由**全部腿关节角**（含被动关节，状态由仿真积分给出）经平面链式
       组合得到：``P = sum_k R(Theta_k) @ p_k``；
       ``L = |P_wheel - P_hip|``，``phi = atan2(rx, -rz)``（沿用 robot.py 的约定）。
    2) 平面雅可比解析求导：对角度求导即把该段偏置转 90 度再累加。
    3) 力矩分配用的雅可比由**闭链速度约束**消去被动关节速率：
       ``dC/dq_pass @ q_pass_dot + dC/dq_act @ q_act_dot = 0``
       ``J = A_all @ [I ; -(dC/dq_pass)^-1 dC/dq_act]``，``A_all = d(L, phi)/dq_all``。

本模块不做什么
--------------
- 不读 ``data.xpos`` / ``xmat`` / ``mj_jacSite``（运行期）；不修改仿真状态
- 不做控制、不配置日志（见 ``logging_setup.py``）

单位：长度 m，角度 rad。平面坐标一律为机身系 (x, z)。
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from . import robot as R
from .logging_setup import get_logger

_LOGGER = get_logger(__name__)


def _rot(angle: float) -> np.ndarray:
    """机身系 xz 平面内"绕 +y 转 angle"的 2x2 矩阵（与 robot.leg_polar 的约定一致）。"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s], [-s, c]])


def _drot(angle: float) -> np.ndarray:
    """上面的导数：d/dangle R(angle) = R(angle + 90°)。"""
    return _rot(angle + 0.5 * math.pi)


class LegKinematics:
    """一条腿的平面解析运动学（几何常数在构造时从模型读出）。"""

    def __init__(self, model, leg: R.LegSpec) -> None:
        self.model = model
        self.leg = leg
        # 参考位形只跑一次，用来把"父体系偏置"换算到机身系常量
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = 0.0
        mujoco.mj_kinematics(model, scratch)

        # 收集本腿涉及的全部刚体：两条链 + 闭链 site 所在体
        involved = [leg.hip_body, leg.wheel_body]
        for joint_name in (leg.front_joint, leg.rear_joint) + leg.passive_joints:
            involved.append(model.body(int(model.joint(joint_name).bodyid[0])).name)
        for site_a, site_b in leg.loops:
            for site_name in (site_a, site_b):
                involved.append(model.body(int(model.site(site_name).bodyid[0])).name)

        # 去重并按"父在子前"排序（保证链式组合时父已算好）
        ordered: list[str] = []
        for name in involved:
            chain, cursor = [], name
            while cursor not in ordered:
                chain.append(cursor)
                parent = int(model.body(cursor).parentid[0])
                if parent == 0:
                    break
                cursor = model.body(parent).name
            for item in reversed(chain):
                if item not in ordered:
                    ordered.append(item)
        self.body_order = ordered
        self.parent: dict[str, str | None] = {}
        self.offset: dict[str, tuple[float, float]] = {}
        self.joint_of: dict[str, str | None] = {}
        self.sign: dict[str, float] = {}
        body_of_joint = {joint_name: model.body(int(model.joint(joint_name).bodyid[0])).name
                         for joint_name in (leg.front_joint, leg.rear_joint) + leg.passive_joints}
        for name in ordered:
            body_id = int(model.body(name).id)
            parent_id = int(model.body(body_id).parentid[0])
            parent_rot = scratch.xmat[parent_id].reshape(3, 3)
            offset = parent_rot @ np.asarray(model.body_pos[body_id], dtype=float)
            self.parent[name] = model.body(parent_id).name if parent_id else None
            self.offset[name] = (float(offset[0]), float(offset[2]))
            joint_name = next((j for j, b in body_of_joint.items() if b == name), None)
            self.joint_of[name] = joint_name
            if joint_name is None:
                self.sign[name] = 0.0
            else:
                # jnt_axis 是**关节所属刚体**坐标系的量（不是父体系），必须用它自己的参考旋转映射；
                # 用父体的旋转去映射会把符号弄反（右腿 jAG/jAB 全反，左腿因父体是 base 而碰巧正确）。
                body_rot = scratch.xmat[body_id].reshape(3, 3)
                axis = body_rot @ np.asarray(model.jnt_axis[model.joint(joint_name).id], dtype=float)
                self.sign[name] = 1.0 if axis[1] >= 0.0 else -1.0

        # 闭链 site：site 名 -> (所在体名, 平面偏置)
        self.site: dict[str, tuple[str, tuple[float, float]]] = {}
        for site_a, site_b in leg.loops:
            for site_name in (site_a, site_b):
                site_id = int(model.site(site_name).id)
                body_id = int(model.site(site_id).bodyid[0])
                rot = scratch.xmat[body_id].reshape(3, 3)
                point = rot @ np.asarray(model.site_pos[site_id], dtype=float)
                self.site[site_name] = (model.body(body_id).name,
                                        (float(point[0]), float(point[2])))
        self.actuated = (leg.front_joint, leg.rear_joint)
        self.passive = tuple(leg.passive_joints)
        _LOGGER.debug("LegKinematics(%s)：%d 个刚体、%d 个闭链 site、驱动关节 %s",
                      leg.name, len(ordered), len(self.site), self.actuated)

    # ---------------------------------------------------------------- 链式组合
    def _chain(self, angles: dict[str, float] | None):
        """返回 (body -> 平面坐标, body -> 累计角)。angles 为"关节名 -> 角"。"""
        angles = angles or {}
        pos: dict[str, tuple[float, float]] = {}
        theta: dict[str, float] = {}
        for name in self.body_order:
            parent = self.parent[name]
            base_pos = (0.0, 0.0) if parent is None else pos[parent]
            base_theta = 0.0 if parent is None else theta[parent]
            # 根刚体（无父体）的自身偏置不计入：本链的坐标一律相对根刚体原点，
            # 否则会叠加一个常数量（实测 +200 mm，因腿长取差值所以此前没暴露）。
            offset = (np.zeros(2) if parent is None
                      else np.asarray(self.offset[name], dtype=float))
            pos[name] = tuple(base_pos + _rot(base_theta) @ offset)
            joint_name = self.joint_of[name]
            theta[name] = base_theta + self.sign[name] * float(angles.get(joint_name or "", 0.0))
        return pos, theta

    def _path(self, body: str) -> list[str]:
        """body 到根的路径（根在前）。"""
        chain = []
        cursor: str | None = body
        while cursor is not None:
            chain.append(cursor)
            cursor = self.parent[cursor]
        return list(reversed(chain))

    def _point_and_derivatives(self, body: str, site: str | None, angles: dict[str, float]):
        """返回 (平面坐标, {关节名 -> d(坐标)/d(角)})。全解析，无位置 API。"""
        pos, theta = self._chain(angles)
        if site is None:
            point = np.asarray(pos[body], dtype=float)
            chain_body = body
            extra: np.ndarray | None = None
        else:
            site_body, offset = self.site[site]
            extra = np.asarray(offset, dtype=float)
            point = np.asarray(pos[site_body], dtype=float) + _rot(theta[site_body]) @ extra
            chain_body = site_body
        derivatives: dict[str, np.ndarray] = {}
        # 沿路径逐段累加：某段偏置对"位于其上方(含)的关节"求导为一个九十度旋转项
        chain = self._path(chain_body)
        for index, name in enumerate(chain):
            parent = self.parent[name]
            parent_theta = 0.0 if parent is None else theta[parent]
            offset = np.asarray(self.offset[name], dtype=float)
            term = _drot(parent_theta) @ offset
            # 该段偏置是被**父体**累计角旋转的，所以对它求导的关节只到 parent(name)，
            # 不包含 name 自己的关节（否则会把自身关节算两次）。
            for upper in chain[:index]:
                joint_name = self.joint_of[upper]
                if joint_name is None:
                    continue
                derivatives.setdefault(joint_name, np.zeros(2))
                derivatives[joint_name] = derivatives[joint_name] + self.sign[upper] * term
        if extra is not None:
            site_body, _ = self.site[site]
            term = _drot(theta[site_body]) @ extra
            for upper in self._path(site_body):
                joint_name = self.joint_of[upper]
                if joint_name is None:
                    continue
                derivatives.setdefault(joint_name, np.zeros(2))
                derivatives[joint_name] = derivatives[joint_name] + self.sign[upper] * term
        return point, derivatives

    # ------------------------------------------------------------------ 对外接口
    def lengths(self, angles: dict[str, float]) -> tuple[float, float]:
        """(L, phi)：腿长与腿摆角。输入是**全部**腿关节角（含被动）。"""
        pos, _ = self._chain(angles)
        hip = np.asarray(pos[self.leg.hip_body], dtype=float)
        wheel = np.asarray(pos[self.leg.wheel_body], dtype=float)
        r = wheel - hip
        return float(np.hypot(r[0], r[1])), float(math.atan2(r[0], -r[1]))

    def jacobian(self, angles: dict[str, float]) -> np.ndarray:
        """J = d(L, phi)/d(q_act)：已用闭链速度约束消去被动关节速率（2x2）。"""
        hip, d_hip = self._point_and_derivatives(self.leg.hip_body, None, angles)
        wheel, d_wheel = self._point_and_derivatives(self.leg.wheel_body, None, angles)
        r = wheel - hip
        length = float(np.hypot(r[0], r[1]))
        # dL/dq = (r/|r|) · d(r)/dq
        rows = [r / max(length, 1e-12)]
        # dphi/dq = d/dq atan2(rx, -rz) = ( -rz*drx + rx*drz ) / |r|^2
        denom = max(length * length, 1e-24)
        rows.append(np.array([-r[1], r[0]]) / denom)
        all_deriv = []
        for joint_name in self.actuated + self.passive:
            all_deriv.append((d_wheel.get(joint_name, np.zeros(2))
                              - d_hip.get(joint_name, np.zeros(2))))
        a_all = np.array([[row @ d for d in all_deriv] for row in rows])   # 2 x 6
        n_act = len(self.actuated)
        c_act = self._closure_matrix(angles, self.actuated)
        c_pass = self._closure_matrix(angles, self.passive)
        coupling = -np.linalg.solve(c_pass, c_act)          # 被动速率 = coupling @ 驱动速率
        transform = np.vstack([np.eye(n_act), coupling])    # 6 x 2
        return a_all @ transform

    def closure_residual(self, angles: dict[str, float]) -> np.ndarray:
        """闭链残差（两对 site 的平面坐标差，4 项）：与 _closure_matrix 同一套解析式。"""
        out: list[float] = []
        for site_a, site_b in self.leg.loops:
            body_a, _ = self.site[site_a]
            body_b, _ = self.site[site_b]
            point_a, _ = self._point_and_derivatives(body_a, site_a, angles)
            point_b, _ = self._point_and_derivatives(body_b, site_b, angles)
            difference = point_a - point_b
            out.extend((float(difference[0]), float(difference[1])))
        return np.asarray(out)

    def _closure_matrix(self, angles: dict[str, float], joints: tuple[str, ...]) -> np.ndarray:
        """闭链残差对 joints 的雅可比（4 x len(joints)）：两对 site 的平面坐标差。"""
        rows = []
        for site_a, site_b in self.leg.loops:
            body_a, _ = self.site[site_a]
            body_b, _ = self.site[site_b]
            point_a, deriv_a = self._point_and_derivatives(body_a, site_a, angles)
            point_b, deriv_b = self._point_and_derivatives(body_b, site_b, angles)
            difference = point_a - point_b
            del difference  # 残差本身不参与雅可比，这里只取导数
            for joint_name in joints:
                rows.append(deriv_a.get(joint_name, np.zeros(2))
                            - deriv_b.get(joint_name, np.zeros(2)))
        # rows 的排布：每对 site 先 x 后 z，按 joints 顺序展开
        matrix = np.zeros((2 * len(self.leg.loops), len(joints)))
        for loop_index in range(len(self.leg.loops)):
            for joint_index, _ in enumerate(joints):
                vector = rows[loop_index * len(joints) + joint_index]
                matrix[2 * loop_index, joint_index] = vector[0]
                matrix[2 * loop_index + 1, joint_index] = vector[1]
        return matrix
