"""Independent mathematical fixtures; gains here are not robot tuning."""
import numpy as np
import pytest

from Algorithm.observers import (RLSConfig, RecursiveLeastSquares, VelocityESO,
    GeneralizedMomentumObserver, CompensationRouter, DisturbanceEstimate, DisturbanceSpace,
    AdaptiveAccelerationObserver)


def regression(size=2):
    return RecursiveLeastSquares(np.zeros(size), 100 * np.eye(size), -5 * np.ones(size),
                                 5 * np.ones(size), RLSConfig(1.0, 0.01, 0.1, 0.995))


def test_rls_identifies_independent_features_and_preserves_covariance():
    model = regression()
    for i in range(300):
        phi = np.array([np.sin(i * 0.2), np.cos(i * 0.37)])
        model.update(phi, phi @ [1.7, -0.4], learning_allowed=True)
    np.testing.assert_allclose(model.parameters, [1.7, -0.4], atol=1e-5)
    assert model.usable
    assert np.linalg.eigvalsh(model.covariance).min() > 0.0
    np.testing.assert_allclose(model.covariance, model.covariance.T, atol=1e-14)


def test_rls_unexcited_or_prohibited_learning_does_not_change_model_or_covariance():
    model = regression()
    for _ in range(100):
        model.update([1., 1.], 3., learning_allowed=True)
    assert not model.excited
    np.testing.assert_array_equal(model.parameters, [0., 0.])
    p = model.covariance
    for i in range(100):
        model.update([1., (-1.)**i], 300., learning_allowed=False)
    np.testing.assert_array_equal(model.covariance, p)
    assert not model.updated


def test_rls_projection_does_not_claim_trusted_parameter_estimate():
    model = regression(1)
    model.update([1.], 500., learning_allowed=True)
    assert model.parameters[0] == 5.
    assert model.projected and not model.usable


@pytest.mark.parametrize("dt", [0.001, 0.02, 0.1])
def test_eso_constant_disturbance_sign_and_sample_rate(dt):
    observer = VelocityESO(2, bandwidth=8., warmup=0.1)
    velocity = np.array([0.4, -0.2])
    known, disturbance = np.array([1., -0.3]), np.array([0.7, -1.2])
    observer.update(velocity, known, dt, trusted=True)
    for _ in range(round(5 / dt)):
        velocity += dt * (known + disturbance)
        observer.update(velocity, known, dt, trusted=True)
    np.testing.assert_allclose(observer.disturbance, disturbance, atol=1e-9)
    assert observer.valid


def test_eso_uses_applied_input_and_resets_on_context_or_missing_data():
    observer = VelocityESO(1, 10., 0.1)
    v = np.zeros(1)
    for _ in range(300):
        observer.update(v, [0.5], 0.01, trusted=True, context=1)
        v += 0.005  # Requested acceleration might be 5; applied is only 0.5.
    np.testing.assert_allclose(observer.disturbance, 0., atol=1e-10)
    observer.update(v + 10, [0.5], 0.01, trusted=True, context=2)
    assert not observer.valid
    np.testing.assert_array_equal(observer.disturbance, 0.)
    observer.update(v, [0.5], 0.01, trusted=False)
    assert not observer.valid


def test_momentum_known_contacts_are_not_external_and_external_sign_is_correct():
    observer = GeneralizedMomentumObserver([12., 8.], warmup=0.1)
    mass = np.diag([2., 3.])
    velocity = np.zeros(2)
    applied, h, contact, external = np.array([1., -2.]), np.array([0., 9.]), np.array([0., 9.]), np.array([0.6, -0.4])
    acceleration = np.linalg.solve(mass, applied + contact + external - h)
    for _ in range(1000):
        observer.update(mass, velocity, applied, h, [0., 0.], contact, 0.005,
                        trusted=True, contact_known=True)
        velocity += acceleration * 0.005
    np.testing.assert_allclose(observer.residual, external, atol=1e-9)
    observer.update(mass, velocity, applied, h, [0., 0.], contact, 0.005,
                    trusted=True, contact_known=False)
    assert not observer.valid
    np.testing.assert_array_equal(observer.residual, 0.)


def test_momentum_variable_mass_includes_mass_derivative_and_contact_reset():
    observer = GeneralizedMomentumObserver([5.], 0.1)
    # p(t) = (2+t)*3; known Mdot*v=3, no unknown force despite changing momentum.
    for i in range(400):
        observer.update([[2 + i * 0.01]], [3.], [0.], [0.], [3.], [0.], 0.01,
                        trusted=True, contact_known=True)
    np.testing.assert_allclose(observer.residual, 0., atol=1e-12)
    observer.update([[7.]], [20.], [0.], [0.], [3.], [0.], 0.01,
                    trusted=True, contact_known=True, context=1)
    np.testing.assert_array_equal(observer.residual, 0.)
    assert not observer.valid


def test_momentum_does_not_misclassify_held_actuation_steps_as_external_impulses():
    observer = GeneralizedMomentumObserver([20.], 0.1)
    velocity = 0.0
    observer.update([[2.]], [velocity], [0.], [0.], [0.], [0.], 0.01,
                    trusted=True, contact_known=True)
    for torque in [0., 3., -2., 1., 0.] * 30:
        velocity += torque / 2. * 0.01
        observer.update([[2.]], [velocity], [torque], [0.], [0.], [0.], 0.01,
                        trusted=True, contact_known=True)
        np.testing.assert_allclose(observer.residual, 0., atol=1e-12)


def test_routing_enforces_age_context_units_limits_and_exclusive_source():
    space = DisturbanceSpace.REDUCED_ACCELERATION
    router = CompensationRouter("eso", space, [1., 2., 3.], [10., 10., 10.], 0.05)
    sample = DisturbanceEstimate(np.array([20., -20., 1.]), space, "eso", 0., 1, True)
    np.testing.assert_allclose(router.route(sample, timestamp=0., dt=0.02, context=1, enabled=True), [0.2, -0.2, 0.2])
    np.testing.assert_array_equal(router.route(sample, timestamp=0.1, dt=0.1, context=1, enabled=True), 0.)
    wrong = DisturbanceEstimate(np.ones(3), DisturbanceSpace.GENERALIZED_FORCE, "momentum", 0.2, 1, True)
    with pytest.raises(ValueError, match="mismatch"):
        router.route(wrong, timestamp=0.2, dt=0.1, context=1, enabled=True)
    fresh = DisturbanceEstimate(np.ones(3), space, "eso", 0.3, 1, True)
    np.testing.assert_array_equal(router.route(fresh, timestamp=0.3, dt=0.1, context=2, enabled=True), 0.)


def test_rls_and_eso_composition_compensates_total_once_and_model_commit_restarts():
    regressions = tuple(regression(1) for _ in range(3))
    bank = AdaptiveAccelerationObserver(regressions, VelocityESO(3, 10., 0.1))
    features = ([1.], [1.], [1.])
    slow, unknown = np.array([0.2, -0.4, 0.1]), np.array([0.3, 0.1, -0.2])
    for _ in range(300):
        bank.learn(features, slow, learning_allowed=True)
    assert bank.commit(steady_supported=True)
    velocity = np.zeros(3)
    for k in range(400):
        sample = bank.update(velocity, np.zeros(3), features, features, 0.01,
                             timestamp=k * 0.01, context=1, trusted=True)
        velocity += 0.01 * (slow + unknown)
    np.testing.assert_allclose(sample.value, slow + unknown, atol=1e-9)
    assert sample.valid
    assert bank.commit(steady_supported=True)
    sample = bank.update(velocity, np.zeros(3), features, features, 0.01,
                         timestamp=4., context=1, trusted=True)
    assert not sample.valid


@pytest.mark.parametrize("factory", [lambda: VelocityESO(1, 0, 1),
    lambda: GeneralizedMomentumObserver([-1], 0.1), lambda: RLSConfig(1.1, 1, 1, 1)])
def test_invalid_observer_configuration_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()
