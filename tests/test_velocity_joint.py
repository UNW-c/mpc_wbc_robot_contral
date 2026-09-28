"""Multi-rate braking, reversal and HOLD acceptance on the analytic fixture."""
import numpy as np
import pytest

from Algorithm.mpc import MotionCommand
from tests.joint_harness import JointLoop, PlanarWbc, build_mpc, find_static_equilibrium
from tests.planar_robot import PlanarTwoWheelRobot


@pytest.mark.parametrize("reverse", [False, True])
def test_velocity_release_and_reverse(reverse):
    robot = PlanarTwoWheelRobot()
    q = robot.standing_pose()
    torque, _, _ = find_static_equilibrium(robot, q)
    mpc, _, _ = build_mpc(robot, q, torque)
    loop = JointLoop(robot, mpc, PlanarWbc(robot))
    def command(time):
        speed = 0.4 if time < 2.0 else -0.3 if reverse and time < 4.0 else 0.0
        return MotionCommand(velocity=speed)
    result = loop.run(command, ticks=600, q=q)
    h = result["history"]
    assert abs(h["velocity"][-1]) < 0.02
    assert np.max(np.abs(h["pitch"])) < 0.25
    assert np.all(h["reference_mode"][-50:] == "hold")
    # User acceptance: bounded, recovering HOLD error; no fixed 10 mm deadline.
    error = np.abs(h["position"] - h["position_reference"])
    assert np.max(error) < mpc.settings.limits.position_error_limit
    assert error[-1] < error[399]
    assert np.max(np.diff(error[400:])) <= 1e-9
    assert np.max(np.abs(h["velocity"][-50:])) < 0.02
    assert mpc.planner.reset_count == 2  # Enable and one HOLD capture only.
    assert np.ptp(h["position_reference"][-50:]) < 1e-12
    if reverse:
        assert np.min(h["velocity"]) < -0.25
    assert result["minimum_normal_force"] > 0.0
    assert np.max(result["control_history"]["dynamics_residual"]) < 1e-8
    assert not np.any(result["control_history"]["fallback"])
    assert not np.any(h["mpc_fallback"])
