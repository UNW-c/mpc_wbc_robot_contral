"""Perturbed analytic dynamics verify compensation sign and constraint ownership."""
import numpy as np
import pytest
from Algorithm.mpc import WheelMPC, MPCSettings, ChassisState, MotionCommand
from Algorithm.observers import VelocityESO, GeneralizedMomentumObserver
from Algorithm.wbc.projection import ContactElimination, NullSpaceHierarchy, TaskLevel
from Algorithm.wbc.sns import SaturationResolver, ActuatorLimits


def resolver():
    mass, contact, selection = np.diag([2., 1.]), np.array([[0., 1.]]), np.array([[1., 0.]])
    elimination = ContactElimination.from_matrices(contact, selection, mass)
    return SaturationResolver(NullSpaceHierarchy(mass), elimination), mass, contact, selection


@pytest.mark.parametrize("limit", [0.3, 2.0])
def test_wbc_external_force_is_inside_complete_dynamics_and_torque_limits(limit):
    solver, mass, contact, selection = resolver()
    h, external = np.array([0., 9.81]), np.array([0.6, 0.])
    output = solver.resolve(contact, [0.], [TaskLevel([[1., 0.]], [0.], name="motion")],
                            ActuatorLimits.symmetric([limit]), dynamics_bias=h, external_force=external)
    force = solver.elimination.recover_contact_force(output.qacc, h - external, output.torque)
    np.testing.assert_allclose(mass @ output.qacc + h, selection.T @ output.torque + contact.T @ force + external, atol=1e-12)
    assert abs(output.torque[0]) <= limit + 1e-12
    assert output.torque[0] < 0.0
    assert output.feasible


def test_mpc_disturbance_lead_and_predicted_trajectory_use_same_additive_model(measured_model):
    settings = MPCSettings(prediction_lead=0.004)
    mpc = WheelMPC(measured_model, settings)
    disturbance = np.array([0.3, -0.1, 0.2])
    state = ChassisState()
    out = mpc.solve(state, MotionCommand(), disturbance_acceleration=disturbance)
    continuous = np.zeros(6)
    continuous[[1, 3, 5]] = disturbance
    np.testing.assert_allclose(out.state, measured_model.drift_step_map(0.004) @ continuous, atol=1e-12)
    discrete = measured_model.discretize(settings.dt, continuous)
    x = out.state.copy()
    for i, force in enumerate(out.force_trajectory):
        x = discrete.A @ x + discrete.B @ force + discrete.w
        np.testing.assert_allclose(out.trajectory.states[i + 1], x, atol=1e-10)
    assert np.all(np.abs(out.force_trajectory) <= mpc.wheel_force_limit(state) + 1e-9)
    plain = WheelMPC(measured_model, settings).solve(state, MotionCommand())
    assert np.linalg.norm(out.force - plain.force) > 1e-4


@pytest.mark.parametrize("kind", ["eso", "momentum"])
def test_observed_external_force_reduces_error_without_post_solve_torque_patch(kind):
    errors = []
    for enabled in (False, True):
        solver, mass, contact, _ = resolver()
        eso, momentum = VelocityESO(1, 10., 0.1), GeneralizedMomentumObserver([10., 10.], 0.1)
        h = np.array([0., 9.81])
        velocity = np.zeros(2)
        previous_torque = 0.
        for i in range(800):
            d = eso.update(velocity[:1], [previous_torque / 2.], 0.005, trusted=True)
            # Contact is an independent known support in this exactly vertical fixture.
            r = momentum.update(mass, velocity, [previous_torque, 0.], h, [0., 0.], h, 0.005,
                                trusted=True, contact_known=True)
            estimated = np.array([2. * d[0], 0.]) if kind == "eso" else r
            ready = eso.valid if kind == "eso" else momentum.valid
            output = solver.resolve(contact, [0.], [TaskLevel([[1., 0.]], [-2. * velocity[0]], name="motion")],
                ActuatorLimits.symmetric([2.]), dynamics_bias=h,
                external_force=estimated if enabled and ready else None)
            # Applied output drives the independent plant; the observer never gets the true external force.
            external = 1.0 if i >= 100 else 0.
            velocity[0] += 0.005 * (output.torque[0] + external) / 2.
            previous_torque = output.torque[0]
            assert abs(previous_torque) <= 2. + 1e-12
            assert output.dynamics_residual_norm < 1e-10
        errors.append(abs(velocity[0]))
    assert errors[0] > 0.2
    assert errors[1] < 0.002
