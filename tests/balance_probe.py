"""Read-only numerical evidence for the zero-speed balance interface."""
import logging
import numpy as np
import mujoco
from APP.mujoco_mpc_wbc import MuJoCoMpcWbcSimulation
from Algorithm.mpc import ReducedModel

_LOGGER = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    sim = MuJoCoMpcWbcSimulation()
    adapter, data, model = sim.dynamics, sim.robot.data, sim.robot.model
    mass = adapter.mass()
    _, _, blocks = adapter.matrices((True, True))
    contact = np.vstack([b.jacobian for b in blocks])
    for locked in (False, True):
        constraint = np.vstack((contact, np.eye(model.nv)[sim.hip_v])) if locked else contact
        kkt = np.block([[mass, -constraint.T], [constraint, np.zeros((len(constraint), len(constraint)))]])
        for name, torque in (("wheel_drive", [0, 0, 1, -1]), ("hip_pitch", [0.5, -0.5, 0, 0])):
            response = np.linalg.solve(kkt, np.r_[adapter.selection.T @ torque, np.zeros(len(constraint))])[:model.nv]
            _LOGGER.info("locked=%s input=%s xacc=%.6f pitchacc=%.6f hipacc=%s", locked, name, response[0], response[4], response[sim.hip_v])
    free = ReducedModel.from_mujoco(model, data)
    _LOGGER.info("Free-hip MPC wheel xacc=%.6f pitchacc=%.6f; legacy hip pitchacc=%.6f", *(free.B[[1,3]] @ np.array([1/.16,1/.16])), free.hip_torque_drift(1.)[3])
    _LOGGER.info("Held-hip MPC wheel response=%s; trim=%s", sim.mpc.model.B[[1,3]] @ np.array([1/.16,1/.16]), sim.mpc.model.metadata)
    sim.robot.data.ctrl[:] = [0.5, -0.5, 0, 0]
    mujoco.mj_forward(model, data)
    _LOGGER.info("MuJoCo applied hip moment=%s actual xacc=%.6f pitchacc=%.6f", adapter.hip_pitch_moment(), data.qacc[0], data.qacc[4])


if __name__ == "__main__":
    main()
