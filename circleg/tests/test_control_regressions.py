"""独立物理/数学反例：不以实现自身的中间量作为正确性的唯一依据。"""

import numpy as np
import pytest

from Algorithm.mpc import ChassisState, MotionCommand, MPCSettings, OptimizedTrajectory, WheelMPC
from Algorithm.timing import ControlTiming
from Algorithm.wbc.projection import ContactElimination, NullSpaceHierarchy, TaskLevel
from Algorithm.wbc.sns import ActuatorLimits, SaturationResolver, maximum_scale
from tests.joint_harness import JointLoop, PlanarWbc, build_mpc, find_static_equilibrium, rolling_velocity
from tests.planar_robot import PlanarTwoWheelRobot


def resolver(selection):
    mass = np.eye(3)
    contact = np.array([[1.0, 0.0, 0.0]])
    elimination = ContactElimination.from_matrices(contact, selection, mass)
    return SaturationResolver(NullSpaceHierarchy(mass), elimination), contact, elimination


def test_disjoint_scale_intervals_have_no_feasible_intersection():
    assert maximum_scale(np.zeros(2), np.ones(2), np.array([0.8, 0.0]), np.array([1.0, 0.2])) == 0.0


def test_unactuated_acceleration_cannot_be_reported_as_achieved():
    solve, contact, elimination = resolver(np.array([[0.0, 0.0, 1.0]]))
    h = np.array([0.0, 2.0, 0.0])
    result = solve.resolve(contact, None, [TaskLevel(np.array([[0.0, 1.0, 0.0]]), np.array([1.0]), name="free")],
                           ActuatorLimits.symmetric([10.0]), dynamics_bias=h)
    assert result.qacc[1] == pytest.approx(-2.0)
    assert result.hierarchy.outcome("free").residual_norm == pytest.approx(3.0)
    force = elimination.recover_contact_force(result.qacc, h, result.torque)
    np.testing.assert_allclose(result.qacc + h, elimination.selection.T @ result.torque + contact.T @ force, atol=1e-10)
    assert result.feasible


def test_saturating_lower_task_does_not_reduce_achievable_higher_task():
    solve, contact, _ = resolver(np.eye(3)[[1, 2]])
    top = TaskLevel(np.array([[0.0, 1.0, 0.0]]), np.array([1.0]), name="pitch")
    low = TaskLevel(np.array([[0.0, 0.0, 1.0]]), np.array([100.0]), name="height")
    limits = ActuatorLimits.symmetric(np.ones(2))
    alone = solve.resolve(contact, None, [top], limits)
    together = solve.resolve(contact, None, [top, low], limits)
    np.testing.assert_allclose(together.torque, [1.0, 1.0], atol=1e-10)
    assert together.qacc[1] == pytest.approx(alone.qacc[1])
    assert together.scale_of("pitch") == 1.0
    assert together.scale_of("height") == pytest.approx(0.01)


def test_damped_task_is_preserved_in_final_solution_and_diagnostics():
    hierarchy = NullSpaceHierarchy(np.eye(3))
    contact = np.array([[1.0, 0.0, 0.0]])
    top = TaskLevel(np.array([[0.0, 1.0, 0.0]]), np.array([1.0]), damping=0.1, name="top")
    low = TaskLevel(np.array([[0.0, 1.0, 1.0]]), np.array([100.0]), name="low")
    alone = hierarchy.solve(contact, None, [top])
    result = hierarchy.solve(contact, None, [top, low])
    assert result.qacc[1] == pytest.approx(alone.qacc[1])
    np.testing.assert_allclose(result.outcome("top").achieved, top.jacobian @ result.qacc)


def test_inactive_task_releases_space_and_exhausted_budget_keeps_its_slot():
    solve, contact, _ = resolver(np.eye(3)[[1, 2]])
    top = TaskLevel(np.array([[0.0, 1.0, 0.0]]), np.array([2.0]), activation=0.0, name="off")
    low = TaskLevel(top.jacobian, np.array([3.0]), name="low")
    exhausted = TaskLevel(top.jacobian, np.array([10.0]), name="exhausted")
    result = solve.resolve(contact, None, [top, low, exhausted], ActuatorLimits.symmetric([10.0, 10.0]),
                           budgets={"exhausted": (-np.ones(2), np.ones(2))})
    assert result.qacc[1] == pytest.approx(3.0)
    assert result.hierarchy.outcome("exhausted").exhausted
    np.testing.assert_allclose(result.attribution[3], 0.0)
    assert result.attribution.shape == (4, 2)


@pytest.mark.parametrize("damping", [0.0, 0.1])
@pytest.mark.parametrize("contact", [np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]]),
                                     np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                                     np.zeros((0, 3))])
def test_contact_rank_and_force_recovery_share_one_factorization(contact, damping):
    elimination = ContactElimination.from_matrices(contact, np.eye(3), np.eye(3), damping)
    rank = np.linalg.matrix_rank(contact) if contact.size else 0
    assert elimination.q_u.shape[1] == 3 - rank
    np.testing.assert_allclose(elimination.q_c @ elimination.r_factor, contact.T, atol=1e-12)
    force = np.arange(contact.shape[0], dtype=float) + 1.0
    qacc = contact.T @ force
    recovered = elimination.recover_contact_force(qacc, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(contact.T @ recovered, qacc, atol=1e-12)


def test_fallback_recomputes_acceleration_for_applied_torque():
    solve, contact, elimination = resolver(np.eye(3)[[1, 2]])
    bias = np.array([0.0, 5.0, -4.0])
    result = solve.resolve(contact, None, [TaskLevel(np.array([[0.0, 1.0, 0.0]]), np.array([1.0]), name="task")],
                           ActuatorLimits.symmetric([1.0, 1.0]), dynamics_bias=bias)
    assert result.fallback_used and result.feasible
    np.testing.assert_allclose(result.qacc, [0.0, -4.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(result.qacc + bias, elimination.selection.T @ result.torque, atol=1e-12)


def test_interpolated_acceleration_is_derivative_of_interpolated_velocity():
    # p=t^3, v=3t^2, a=6t，三个通道均为同一个可手算多项式。
    times = np.array([0.0, 0.2, 0.4])
    states = np.column_stack([times**3, 3 * times**2] * 3)
    trajectory = OptimizedTrajectory(states, np.array([[1.0, 2.0], [3.0, 4.0]]), 0.2, 10.0)
    for t in (0.0, 0.05, 0.2, 0.35, 0.4):
        sample = trajectory.sample(t)
        np.testing.assert_allclose(sample.state[::2], t**3, atol=1e-12)
        np.testing.assert_allclose(sample.state[1::2], 3 * t**2, atol=1e-12)
        np.testing.assert_allclose(sample.acceleration, 6 * t, atol=1e-12)
        np.testing.assert_allclose(trajectory.sample_at(10.0 + t).state, sample.state, atol=1e-12)
    np.testing.assert_allclose(trajectory.sample(0.2).force, [3.0, 4.0])
    with pytest.raises(ValueError, match="有效期"):
        trajectory.sample(0.401)
    with pytest.raises(ValueError, match="有效期"):
        trajectory.sample_at(9.9)


def test_mpc_limit_change_reaches_wbc_through_optimized_trajectory():
    robot = PlanarTwoWheelRobot()
    q = robot.standing_pose()
    torque, _, _ = find_static_equilibrium(robot, q)
    solutions, outputs = [], []
    for bound in (5.0, 0.01):
        mpc, _, _ = build_mpc(robot, q, torque, wheel_torque_limit=bound)
        sol = mpc.solve(ChassisState(), MotionCommand(velocity=0.3))
        sample = sol.trajectory.sample(0.004)
        outputs.append(PlanarWbc(robot).solve(q, np.zeros(7), sample.force,
                       (sample.state[2], sample.state[3], sample.acceleration[1]),
                       (0.0, sample.state[1]), forward_acceleration=sample.acceleration[0])["torque"])
        solutions.append(sol)
    np.testing.assert_allclose(solutions[0].reference.states(), solutions[1].reference.states())
    assert np.linalg.norm(solutions[0].force - solutions[1].force) > 0.1
    assert np.linalg.norm(outputs[0] - outputs[1]) > 1e-3


def test_fallback_and_prediction_lead_use_the_returned_force_trajectory(measured_model, monkeypatch):
    from dataclasses import replace
    mpc = WheelMPC(measured_model, MPCSettings(prediction_lead=0.003))
    original = mpc.qp.solve

    def failed(*args, **kwargs):
        force, info = original(*args, **kwargs)
        return force, replace(info, status_value=7, status="forced failure")

    monkeypatch.setattr(mpc.qp, "solve", failed)
    solution = mpc.solve(ChassisState(pitch=0.01), MotionCommand(velocity=0.2), timestamp=2.0)
    assert solution.fallback
    assert solution.trajectory.start_time == pytest.approx(2.003)
    expected = mpc.predict(solution.state, solution.force_trajectory, mpc.hip_drift(ChassisState(pitch=0.01)))
    np.testing.assert_allclose(solution.trajectory.states, expected)


def test_linearization_preserves_selected_state_rates_and_geometry():
    robot = PlanarTwoWheelRobot()
    q = robot.standing_pose()
    np.testing.assert_allclose(robot.contact_points(q)[:, 1], 0.0, atol=1e-12)
    for velocity, pitch_rate in ((1.0, 0.0), (0.0, 1.0), (0.4, -0.2)):
        rate = rolling_velocity(robot, q, velocity, pitch_rate)
        np.testing.assert_allclose(rate[[0, 2]], [velocity, pitch_rate], atol=1e-12)
        np.testing.assert_allclose(robot.contact_jacobian(q) @ rate, 0.0, atol=1e-12)
    torque, _, _ = find_static_equilibrium(robot, q)
    _, model, _ = build_mpc(robot, q, torque)
    np.testing.assert_allclose(model.A[0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(model.A[2], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-10)
    spun = q.copy()
    spun[[4, 6]] += [1.7, -2.3]
    np.testing.assert_allclose(robot.mass_matrix(spun), robot.mass_matrix(q), atol=1e-12)
    np.testing.assert_allclose(robot.gravity_bias(spun), robot.gravity_bias(q), atol=1e-12)


def test_multirate_loop_recomputes_wbc_five_times_per_mpc_tick():
    robot = PlanarTwoWheelRobot()
    q = robot.standing_pose()
    torque, _, _ = find_static_equilibrium(robot, q)
    mpc, _, _ = build_mpc(robot, q, torque)
    result = JointLoop(robot, mpc, PlanarWbc(robot)).run(MotionCommand(velocity=0.2), 5, q)
    assert result["mpc_calls"] == 5 and result["wbc_calls"] == 25
    np.testing.assert_allclose(np.diff(result["control_history"]["time"]), 0.004, atol=1e-12)
    assert np.max(result["control_history"]["dynamics_residual"]) < 1e-8
    assert result["minimum_normal_force"] > 0.0
    for kwargs in ({"wbc_dt": 0.003}, {"physics_dt": 0.0}, {"wbc_dt": 0.0005}):
        with pytest.raises(ValueError):
            ControlTiming(**kwargs)
