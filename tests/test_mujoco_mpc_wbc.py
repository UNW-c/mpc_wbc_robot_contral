"""MuJoCo evidence for the zero-command parking integration."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation, MuJoCoDynamics
from APP.mujoco_robot import MuJoCoRobot
from Algorithm.mpc import MotionCommand


def test_named_transmission_and_hip_reaction_sign_match_simulator():
    robot = MuJoCoRobot.load("plane")
    adapter = MuJoCoDynamics(robot)
    robot.data.ctrl[:] = [0.4, -0.3, 0.2, -0.1]
    robot.forward()
    np.testing.assert_allclose(adapter.selection.T @ robot.data.ctrl, robot.data.qfrc_actuator, atol=1e-12)
    assert adapter.hip_pitch_moment() == pytest.approx(0.7)
    robot.data.ctrl[:] = [0.4, 0.4, 0., 0.]
    robot.forward()
    assert adapter.hip_pitch_moment() == pytest.approx(0., abs=1e-12)


def test_rolling_bias_does_not_add_material_point_centripetal_acceleration():
    robot = MuJoCoRobot.load("plane")
    adapter = MuJoCoDynamics(robot)
    robot.data.qvel[0] = 0.8
    robot.data.qvel[int(robot.model.joint("joint2").dofadr[0])] = 5.
    robot.data.qvel[int(robot.model.joint("joint4").dofadr[0])] = -5.
    robot.forward()
    _, _, blocks = adapter.matrices((True, True))
    for block in blocks:
        np.testing.assert_allclose(block.jacobian @ robot.data.qvel, 0., atol=1e-9)
        np.testing.assert_allclose(block.bias, 0., atol=1e-8)


def test_euler_rates_match_independent_quaternion_integration():
    robot = MuJoCoRobot.load("plane")
    adapter = MuJoCoDynamics(robot)
    mujoco.mju_quatIntegrate(robot.data.qpos[3:7], np.array([0.12, -0.08, 0.3]), 1.)
    robot.data.qvel[3:6] = [0.4, -0.3, 0.2]
    robot.forward()
    before = adapter.attitude(robot.data)
    rows, _ = adapter.kinematics(robot.data)
    mujoco.mj_integratePos(robot.model, robot.data.qpos, robot.data.qvel, 1e-7)
    observed = (adapter.attitude(robot.data)-before)/1e-7
    np.testing.assert_allclose(rows[1:] @ robot.data.qvel, observed, atol=1e-8)


def test_held_hip_model_input_response_matches_independent_mujoco_reaction_solve():
    # 2026-09-28 默认切换为 hip_input（C7）；本测试核验 6 状态握髋模型，显式指定。
    from APP.mujoco_mpc_wbc import SimulationConfig
    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(model_variant="held_hip"))
    data = sim.robot.data
    # MuJoCo's compliant solver is independent of the model builder's rigid KKT.
    def response(torque):
        data.ctrl[:] = torque
        sim.robot.forward()
        return data.qacc.copy()
    offset = response(np.zeros(4))
    columns = np.column_stack([response(np.eye(4)[i])-offset for i in range(4)])
    torque = np.array([0., 0., 0.1, -0.1])
    torque[:2] = np.linalg.solve(columns[sim.hip_v, :2], -(offset+columns @ torque)[sim.hip_v])
    actual = response(torque)
    np.testing.assert_allclose(actual[sim.hip_v], 0., atol=1e-8)
    expected = sim.mpc.model.A @ sim.state((True, True), np.ones((2, 3))).vector()
    expected += sim.mpc.model.B @ np.array([0.1/0.16, 0.1/0.16])
    # Compliant contact differs by about 4.1% in x and 0.2% in pitch at this
    # operating point. This checks sign/scale; it is not an exact plant match.
    np.testing.assert_allclose(actual[[0, 4]], expected[[1, 3]], rtol=0.05, atol=1e-4)
    assert not sim.mpc.settings.compensate_hip_torque


@pytest.mark.parametrize("pitch", [0., 0.02, -0.02])
def test_zero_command_balance_recovers_without_crank_runaway(pitch):
    sim = MuJoCoMpcWbcSimulation(initial_pitch=pitch)
    result = sim.run(MotionCommand(velocity=0., yaw_rate=0.), 8.)
    assert result["completed"], result
    assert result["mpc_fallbacks"] == result["wbc_fallbacks"] == 0
    assert result["max_abs_pitch"] < 0.021
    # 0.05：2026-09-28 默认切换为 hip_input 后，零平衡会主动用髋（实测 0.040-0.0404 rad），
    # 这是新方案"用髋平衡"的正常行为（旧结构固有 0.0065）；断言保留"曲柄不跑飞"的意图。
    assert result["max_abs_hip"] < 0.05
    assert abs(result["final_qvel"][0]) < 0.01
    assert max(abs(result["final_qvel"][i]) for i in sim.hip_v) < 1e-3
    assert abs(result["final_qvel"][4]) < 0.001
    assert result["max_abs_position"] < 0.05
    assert result["max_dynamics_residual"] < 1e-7
    # 1e-5：接触残差由 OSQP 求解精度与问题条件数决定。旧结构实测 1.198e-7（R7 曾放宽到 1e-6）；
    # 2026-09-28 默认切换为 hip_input 后，髋前馈使 WBC 解的量级更大，实测升到 7.4-8.2e-6。
    # 与 WBC 自身的 feasible 判据（dynamics_norm < 1e-5）保持一致。
    assert result["max_contact_residual"] < 1e-5
    assert set(r["mode"] for r in sim.records) <= {"ground", "recovery"}
    assert all(r["mode"] == "ground" for r in sim.records[25:])
    assert np.all(sim.robot.data.ctrl == 0.)


def test_hip_angle_uses_common_mode_definition():
    """R3（#10）：共模髆角 = 0.5*(q1 − q3)，不是两髆平均。"""
    sim = MuJoCoMpcWbcSimulation()
    contact = (True, True)
    _, forces, _, _ = sim.dynamics.sensed_contact()
    sim.robot.data.qpos[sim.hip_q] = [0.3, -0.3]
    sim.robot.forward()
    state = sim.state(contact, forces)
    assert state.hip_angle == pytest.approx(0.3)
    sim.robot.data.qpos[sim.hip_q] = [0.3, 0.3]  # 差模（反向摆动）应为 0
    sim.robot.forward()
    state = sim.state(contact, forces)
    assert state.hip_angle == pytest.approx(0.0)


def test_nonzero_driving_command_is_not_enabled_in_balance_stage():
    sim = MuJoCoMpcWbcSimulation()
    with pytest.raises(ValueError, match="zero velocity"):
        sim.run(MotionCommand(velocity=0.2), 1.)


# ---- T1：纵向任务目标来源（forward_mode）-----------------------------------

def _forward_target(sim, sample):
    """提取 tasks() 里的 forward_velocity 目标加速度与当前状态。"""
    contact = (True, True)
    _, forces, _, _ = sim.dynamics.sensed_contact()
    state = sim.state(contact, forces)
    rows, bias, _ = sim.dynamics.matrices(contact)
    balance, motion = sim.tasks(state, rows, bias, sample)
    for task in (*balance, *motion):
        if task.name == "forward_velocity":
            return float(np.asarray(task.desired_acceleration)[0]), state
    raise AssertionError("forward_velocity task missing")


def test_forward_mode_mpc_ignores_command_velocity():
    """mpc 模式下 fv 目标不随键盘指令变化。"""
    from Algorithm.mpc.trajectory import MotionSample
    from APP.mujoco_mpc_wbc import SimulationConfig

    # 本测试面向 6 状态模型的 mpc 分支（8 状态用 4 维 acceleration），显式配置。
    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(model_variant="held_hip", forward_mode="mpc"))
    contact = (True, True)
    _, forces, _, _ = sim.dynamics.sensed_contact()
    state = sim.state(contact, forces)
    sample = MotionSample(state.vector(), np.zeros(3), np.zeros(2), 0.0)
    sim.command = MotionCommand(velocity=1.0)
    first, _ = _forward_target(sim, sample)
    sim.command = MotionCommand(velocity=0.3)
    second, _ = _forward_target(sim, sample)
    assert first == second


def test_forward_mode_mpc_zero_tracking_error_is_zero():
    """mpc 模式下 sample 等于当前状态且加速度为零时，fv 目标为零。"""
    from Algorithm.mpc.trajectory import MotionSample
    from APP.mujoco_mpc_wbc import SimulationConfig

    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(model_variant="held_hip", forward_mode="mpc"))
    contact = (True, True)
    _, forces, _, _ = sim.dynamics.sensed_contact()
    state = sim.state(contact, forces)
    sample = MotionSample(state.vector(), np.zeros(3), np.zeros(2), 0.0)
    target, _ = _forward_target(sim, sample)
    assert target == pytest.approx(0.0, abs=1e-12)


def test_forward_mode_legacy_matches_historical_formula():
    """legacy 模式保持历史公式：vg*(v_cmd − v) − pg*(s − s_ref)，限幅 ±50。"""
    from APP.mujoco_mpc_wbc import SimulationConfig

    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(forward_mode="legacy"))
    sim.command = MotionCommand(velocity=0.1)
    target, state = _forward_target(sim, None)
    expected = 320.0 * (0.1 - state.velocity) - 320.0 * (state.position - 0.0)
    expected = float(np.clip(expected, -50.0, 50.0))
    assert target == pytest.approx(expected)
    sim.command = MotionCommand(velocity=0.05)
    changed, _ = _forward_target(sim, None)
    assert changed != target


def test_hip_differential_matches_wheel_offsets():
    """#13（A2）：差模髋角决定两轮前后错开 0.12*sin(q_diff)。"""
    sim = MuJoCoMpcWbcSimulation()
    contact = (True, True)
    _, forces, _, _ = sim.dynamics.sensed_contact()
    data = sim.robot.data
    right = int(sim.robot.model.geom("right_wheel_collision").id)
    left = int(sim.robot.model.geom("left_wheel_collision").id)
    data.qpos[sim.hip_q] = [0.3, 0.3]  # 差模
    sim.robot.forward()
    state = sim.state(contact, forces)
    dx = float(data.geom_xpos[right, 0] - data.geom_xpos[left, 0])
    assert abs(0.12 * float(np.sin(state.hip_differential)) - abs(dx)) < 1e-3
    data.qpos[sim.hip_q] = [0.3, -0.3]  # 共模
    sim.robot.forward()
    state = sim.state(contact, forces)
    assert state.hip_differential == pytest.approx(0.0, abs=1e-12)
    assert abs(float(data.geom_xpos[right, 0] - data.geom_xpos[left, 0])) < 1e-6


def test_yaw_budget_uses_differential_hip_angle():
    """#13（A2）：共模摆髋不产生 scrub 折扣；差模才折扣。"""
    sim = MuJoCoMpcWbcSimulation()
    contact = (True, True)
    _, forces, _, _ = sim.dynamics.sensed_contact()
    settings = sim.mpc.settings
    data = sim.robot.data
    cases = (([0.4, -0.4], 0.0),
             ([0.4, 0.4], settings.scrub_moment_coefficient * float(np.sin(0.4))))
    for angles, scrub in cases:
        data.qpos[sim.hip_q] = angles
        sim.robot.forward()
        state = sim.state(contact, forces)
        limit = sim.mpc.wheel_force_limit(state)
        budget = sim.mpc.yaw_moment_limit(state, limit)
        achievable = sim.mpc.model.half_track * float(np.sum(limit))
        expected = min(max(settings.yaw_moment_limit - scrub, settings.yaw_moment_floor), achievable)
        assert budget == pytest.approx(expected, rel=1e-9)


# ---- B3：8 状态模型（hip_input）接入 --------------------------------------

def test_hip_input_model_variant_prediction_dimensions():
    """B3：hip_input 时预测轨迹 (N+1, 8)、sample 加速度 4 维、力 3 维。"""
    from APP.mujoco_mpc_wbc import SimulationConfig

    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(model_variant="hip_input"))
    assert sim.mpc.model.state_dim == 8
    assert sim.mpc.model.input_dim == 3
    contact = (True, True)
    _, forces, _, _ = sim.dynamics.sensed_contact()
    state = sim.state(contact, forces)
    solution = sim.mpc.solve(state, MotionCommand(velocity=0.0))
    assert solution.predicted_states.shape == (sim.mpc.horizon + 1, 8)
    sample = solution.trajectory.sample_at(solution.trajectory.start_time)
    assert sample.state.shape == (8,)
    assert sample.acceleration.shape == (4,)
    assert sample.force.shape == (3,)


def test_hip_input_model_zero_state_outputs_zero():
    """B3：零状态、零参考时 MPC 输出为零（与握髋模型行为一致）。

    用显式零状态：MuJoCo 站立位形的 pitch 是 0.000215，而新模型在 q_c=0 的
    theta_trim 是 0.000430（双条件平衡点），直接拿初始位形会带入 0.0002 rad 的
    微小偏差（对应约 -0.012 N 的稳态小力，无实际影响，B4 报告已记录）。
    """
    from APP.mujoco_mpc_wbc import SimulationConfig
    from Algorithm.mpc import ChassisState

    sim = MuJoCoMpcWbcSimulation(config=SimulationConfig(model_variant="hip_input"))
    solution = sim.mpc.solve(ChassisState(), MotionCommand(velocity=0.0))
    np.testing.assert_allclose(solution.force, 0.0, atol=1e-9)
    np.testing.assert_allclose(solution.force_trajectory, 0.0, atol=1e-9)


def test_hip_input_requires_compensation_off():
    """B3：hip_input 与 compensate_hip_torque=True 同时开启必须抛 ValueError。"""
    from APP.mujoco_mpc_wbc import SimulationConfig
    from Algorithm.mpc import MPCSettings

    with pytest.raises(ValueError, match="compensate_hip_torque"):
        MuJoCoMpcWbcSimulation(config=SimulationConfig(model_variant="hip_input"),
                               mpc_settings=MPCSettings(compensate_hip_torque=True))


# ---- E5：CLI 参数（--preset new 等，不改默认值） ---------------------------

def test_cli_defaults_match_config_defaults():
    from APP.mujoco_mpc_wbc import _build_from_args, SimulationConfig
    from Algorithm.mpc import MPCSettings

    config, settings, _ = _build_from_args([])
    assert config == SimulationConfig()
    # 2026-09-28 默认切换为 hip_input（C7）：不传参数时会构造对应的 MPCSettings。
    assert config.model_variant == "hip_input"
    assert config.forward_mode == "mpc"
    assert settings is not None
    assert settings.compensate_hip_torque is False
    assert settings.weights.hip_angle == 50.0
    assert settings.hip_angle_limit == 0.55


def test_cli_preset_new_builds_hip_input():
    from APP.mujoco_mpc_wbc import _build_from_args, _PRESETS

    config, settings, _ = _build_from_args(["--preset", "new"])
    assert config.model_variant == "hip_input"
    assert config.forward_mode == "mpc"
    assert config.preview_lead == 0.0
    assert config.hip_guard == _PRESETS["new"]["hip_guard"]
    assert settings is not None
    assert settings.weights.hip_angle == 50.0
    assert settings.hip_angle_limit == 0.55
    assert settings.compensate_hip_torque is False
    # 不跑仿真也能构建仿真对象
    from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation
    MuJoCoMpcWbcSimulation(config=config, mpc_settings=settings)


def test_cli_individual_argument_overrides_preset():
    from APP.mujoco_mpc_wbc import _build_from_args

    _, settings, _ = _build_from_args(["--preset", "new", "--hip-weight", "200"])
    assert settings.weights.hip_angle == 200.0
