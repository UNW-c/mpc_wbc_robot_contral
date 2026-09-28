"""Mode transitions, contact ownership and multi-rate feedback contracts."""
import numpy as np
import pytest
from Algorithm.mpc import WheelMPC, ChassisState, MotionCommand, ReferenceFeedback
from Algorithm.supervision import (ContactDetectionConfig, ContactDetector, ControlMode,
    ControlDecision, ControlSupervisor, SupervisedControl, ContactBlock, WbcFeedbackAccumulator)
from Algorithm.wbc.projection import ContactElimination, NullSpaceHierarchy, TaskLevel
from Algorithm.wbc.sns import ActuatorLimits, SaturationResolver
from Algorithm.observers import CompensationRouter, DisturbanceEstimate, DisturbanceSpace


def test_contact_hysteresis_debounce_and_force_budget_are_separate():
    detector = ContactDetector(ContactDetectionConfig(10., 3., 0.02, 0.03))
    assert detector.update([11., 0.], [True, True], 0.01)[0] == (False, False)
    assert detector.update([9., 0.], [True, True], 0.01)[0] == (False, False)
    detector.update([11., 0.], [True, True], 0.01)
    assert detector.update([11., 0.], [True, True], 0.01)[0] == (True, False)
    contact, budget = detector.update([0., 0.], [True, True], 0.01)
    assert contact == (True, False) and budget == (0., 0.)
    detector.update([0., 0.], [True, True], 0.01)
    assert detector.update([0., 0.], [True, True], 0.01)[0] == (False, False)
    detector.update([11., 11.], [True, True], 0.03)
    assert detector.update([11., 11.], [False, True], 0.01)[0] == (False, True)


def test_supervisor_arbitrates_once_and_resets_only_on_return_to_ground():
    supervisor = ControlSupervisor(0.03)
    def step(k, **kw):
        args = dict(timestamp=k * 0.01, dt=0.01, contact=(True, True), enabled=True,
                    command_fresh=True, stable=True, slipping=False)
        args.update(kw)
        return supervisor.update(**args)
    assert step(0).mode == ControlMode.RECOVERY
    assert step(1).mode == ControlMode.RECOVERY
    ground = step(2)
    assert ground.mode == ControlMode.GROUND and ground.reset_reference
    assert not step(3).reset_reference
    single = step(4, contact=(True, False))
    assert single.balance_enabled and not single.motion_enabled
    assert single.context != ground.context
    assert step(5, contact=(False, False)).mode == ControlMode.AIRBORNE
    # Disabled takes precedence even when all other signals request motion.
    assert step(6, enabled=False).mode == ControlMode.DISABLED
    assert step(7).mode == ControlMode.RECOVERY
    assert step(8).mode == ControlMode.RECOVERY
    assert step(9).reset_reference
    with pytest.raises(ValueError):
        step(9)


def test_ground_reference_pause_prevents_mpc_solving_and_resume_reanchors(measured_model, monkeypatch):
    mpc = WheelMPC(measured_model)
    chain = SupervisedControl(mpc)
    original = mpc.solve
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(mpc, "solve", counted)
    state = ChassisState(position=0.1, velocity=0.2)
    paused = ControlDecision(ControlMode.AIRBORNE, (False, False), 0., 1, False)
    assert chain.solve_mpc(state, MotionCommand(velocity=0.5), paused, ReferenceFeedback()) is None
    assert not calls and not mpc.planner.valid
    resumed = ControlDecision(ControlMode.GROUND, (True, True), 1., 2, True)
    out = chain.solve_mpc(state, MotionCommand(velocity=0.5), resumed, ReferenceFeedback())
    assert out.reference.position[0] == state.position
    assert out.reference.velocity[0] == state.velocity
    assert len(calls) == 1
    # Stale command independently prevents any solve, even with an outdated decision.
    assert chain.solve_mpc(state, MotionCommand(), resumed, ReferenceFeedback(command_fresh=False)) is None
    assert len(calls) == 1


def test_single_support_keeps_balance_and_releases_other_contacts_and_motion(measured_model):
    chain = SupervisedControl(WheelMPC(measured_model))
    mass, selection = np.eye(3), np.eye(3)
    contacts = (ContactBlock([[0., 1., 0.]], [0.], (0,)),
                ContactBlock([[0., 0., 1.]], [0.], (1,)))
    balance = (TaskLevel([[0., 0., 1.]], [2.], name="balance"),)
    motion = (TaskLevel([[1., 0., 0.]], [10.], name="motion"),)
    decision = ControlDecision(ControlMode.SINGLE_SUPPORT, (True, False), 0., 1, False)
    torque, output = chain.solve_wbc(mass=mass, selection=selection, dynamics_bias=np.zeros(3),
        contacts=contacts, balance_tasks=balance, motion_tasks=motion, limits=ActuatorLimits.symmetric([20.]*3),
        decision=decision, dt=0.004)
    np.testing.assert_allclose(output.qacc, [0., 0., 2.], atol=1e-12)
    assert [item.name for item in output.hierarchy.outcomes] == ["balance"]
    assert np.linalg.norm(torque) > 0.
    disabled = ControlDecision(ControlMode.DISABLED, (True, False), 0.004, 2, False)
    torque, output = chain.solve_wbc(mass=mass, selection=selection, dynamics_bias=np.zeros(3),
        contacts=contacts, balance_tasks=balance, motion_tasks=motion, limits=ActuatorLimits.symmetric([20.]*3),
        decision=disabled, dt=0.004)
    np.testing.assert_array_equal(torque, 0.)
    assert output is None


def test_wbc_final_motion_residual_drives_feedback_not_regularizer_scale():
    mass, selection, contact = np.eye(2), np.eye(2), np.array([[0., 1.]])
    solver = SaturationResolver(NullSpaceHierarchy(mass), ContactElimination.from_matrices(contact, selection, mass))
    limits = ActuatorLimits.symmetric([1., 1.])
    def solve(target):
        return solver.resolve(contact, [0.], [TaskLevel([[1., 0.]], [target], name="motion"),
            TaskLevel([[1., 0.]], [100.], name="regularization")], limits)
    collector = WbcFeedbackAccumulator(("motion",))
    nominal = solve(0.2)
    assert nominal.scale_of("regularization") == 0.
    collector.observe(nominal, limits)
    feedback = collector.consume(ChassisState())
    assert not feedback.saturated and feedback.motion_residual < 1e-12
    collector.observe(solve(10.), limits)
    collector.observe(nominal, limits)  # One good final tick must not erase the interval's failure.
    feedback = collector.consume(ChassisState())
    assert feedback.saturated and feedback.motion_residual >= 9. - 1e-10
    assert collector.consume(ChassisState()).motion_residual == 0.


def test_one_compensation_router_owns_only_one_layer(measured_model):
    space = DisturbanceSpace.REDUCED_ACCELERATION
    router = CompensationRouter("eso", space, [1.]*3, [100.]*3, 0.04)
    chain = SupervisedControl(WheelMPC(measured_model), router)
    decision = ControlDecision(ControlMode.GROUND, (True, True), 0., 1, True)
    estimate = DisturbanceEstimate(np.array([0.2, 0., 0.]), space, "eso", 0., 1, True)
    out = chain.solve_mpc(ChassisState(), MotionCommand(), decision, ReferenceFeedback(), estimate)
    assert np.linalg.norm(out.force) > 0.
    torque, result = chain.solve_wbc(mass=np.eye(2), selection=np.eye(2), dynamics_bias=np.zeros(2),
        contacts=(ContactBlock([[0., 1.]], [0.], (0, 1)),),
        balance_tasks=(TaskLevel([[1., 0.]], [0.], name="balance"),), motion_tasks=(),
        limits=ActuatorLimits.symmetric([1., 1.]), decision=decision, dt=0.004, estimate=estimate)
    np.testing.assert_allclose(torque, 0., atol=1e-12)
    assert result.feasible
