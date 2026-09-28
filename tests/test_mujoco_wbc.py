"""可选 MuJoCo 矩阵检查：单拍等式动力学验证，不宣称闭环或摩擦可行。"""

from pathlib import Path

import numpy as np
import pytest

from Algorithm.wbc.projection import ContactElimination, NullSpaceHierarchy, TaskLevel
from Algorithm.wbc.sns import ActuatorLimits, SaturationResolver

mujoco = pytest.importorskip("mujoco")
MODEL_PATH = Path(__file__).resolve().parents[1] / "URDF_Model/mjcf/URDF_Model_plane.xml"


@pytest.mark.parametrize("angle", [0.0, 25.0, 60.0])
def test_real_model_matrices_close_full_dynamics(angle):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    q = np.deg2rad(angle)
    for name in ("joint1", "joint3"):
        data.qpos[model.joint(name).qposadr[0]] = q
    radius = float(model.geom_size[model.geom("right_wheel_collision").id, 0])
    data.qpos[2] = radius + 0.093022 - 0.06 * np.cos(q) - 1e-4
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    mass = np.empty((model.nv, model.nv))
    # MuJoCo 3.13 调整了 Python fullM 签名；兼容项目现有最低依赖版本。
    if tuple(map(int, mujoco.__version__.split(".")[:2])) >= (3, 13):
        mujoco.mj_fullM(model, data, mass)
    else:
        mujoco.mj_fullM(model, mass, data.qM)
    selection = np.zeros((model.nu, model.nv))
    for actuator in range(model.nu):
        joint = model.actuator_trnid[actuator, 0]
        selection[actuator, model.jnt_dofadr[joint]] = model.actuator_gear[actuator, 0]
    point_jacobians = []
    for name in ("right_wheel_collision", "left_wheel_collision"):
        geom = model.geom(name).id
        assert any(geom in (c.geom1, c.geom2) for c in data.contact[:data.ncon])
        # 平地名义模型：每轮中心线几何切点，加集中横向约束，共五行。
        point = data.geom_xpos[geom] + np.array([0.0, 0.0, -radius])
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, np.zeros_like(jacobian), point, model.geom_bodyid[geom])
        point_jacobians.append(jacobian)
    right, left = point_jacobians
    contact = np.vstack((right[0], right[2], left[0], left[2], 0.5 * (right[1] + left[1])))
    elimination = ContactElimination.from_matrices(contact, selection, mass)
    limits = ActuatorLimits(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    tasks = [TaskLevel(np.eye(model.nv)[[i]], np.array([value]), name=name)
             for i, value, name in ((4, 1.0, "pitch"), (3, 0.0, "roll"), (5, 0.0, "yaw"),
                                    (2, 0.0, "height"), (0, 0.0, "velocity"))]
    result = SaturationResolver(NullSpaceHierarchy(mass), elimination).resolve(
        contact, np.zeros(5), tasks, limits, dynamics_bias=data.qfrc_bias)
    force = elimination.recover_contact_force(result.qacc, data.qfrc_bias, result.torque)
    np.testing.assert_allclose(mass @ result.qacc + data.qfrc_bias,
                               selection.T @ result.torque + contact.T @ force, atol=1e-8)
    np.testing.assert_allclose(contact @ result.qacc, 0.0, atol=1e-8)
    assert limits.contains(result.torque)
    assert result.feasible
