"""E1: verify the center-of-mass vertical observation interface."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gbc_mpc_wbc import robot as robot_model  # noqa: E402
from gbc_mpc_wbc.dynamics import Gbc486Dynamics  # noqa: E402

RESULTS = ROOT / "results"
RESULT_PATH = RESULTS / "E1_com_vertical_check_20261009_run02.json"
LOG_PATH = RESULTS / "E1_com_vertical_check_20261009_run02.log"
NOMINAL_LENGTH = 0.42
FINITE_DIFFERENCE_STEP = 1.0e-6


def setup() -> tuple[mujoco.MjModel, mujoco.MjData, Gbc486Dynamics]:
    model, data = robot_model.load()
    robot_model.standing_pose(model, data, NOMINAL_LENGTH)
    return model, data, Gbc486Dynamics(model, data)


def com_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    return np.asarray(data.subtree_com[model.body("base").id], dtype=float).copy()


def finite_difference_com(
    model: mujoco.MjModel,
    base_qpos: np.ndarray,
    qvel: np.ndarray,
    step: float,
) -> tuple[float, float, float]:
    scratch = mujoco.MjData(model)
    values = []
    for scale in (-step, 0.0, step):
        qpos = base_qpos.copy()
        mujoco.mj_integratePos(model, qpos, qvel, scale)
        scratch.qpos[:] = qpos
        values.append(float(com_position(model, scratch)[2]))
    rate = (values[2] - values[0]) / (2.0 * step)
    bias = (values[2] - 2.0 * values[1] + values[0]) / (step * step)
    return rate, bias, values[1]


def run() -> dict:
    model, data, dynamics = setup()
    rng = np.random.default_rng(20261009)
    samples = []
    repeat_diffs = []
    max_row_rate_error = 0.0
    max_bias_error = 0.0
    max_com_error = 0.0
    max_acceleration_error = 0.0

    base_qpos = data.qpos.copy()
    for sample_index in range(8):
        data.qpos[:] = base_qpos
        data.qvel[:] = rng.normal(0.0, 0.2, size=model.nv)
        mujoco.mj_forward(model, data)
        first = dynamics.com_vertical_row()
        second = dynamics.com_vertical_row()
        repeat_diffs.append(float(max(np.max(np.abs(a - b)) for a, b in zip(first, second))))
        row, bias, com_z, com_vz = first
        fd_rate, fd_bias, fd_com_z = finite_difference_com(
            model, data.qpos.copy(), data.qvel.copy(), FINITE_DIFFERENCE_STEP)
        row_rate = float(row @ data.qvel)
        row_rate_error = abs(row_rate - fd_rate)
        bias_error = abs(float(bias) - fd_bias)
        com_error = abs(float(com_z) - fd_com_z)
        acceleration_error = abs(float(bias) - fd_bias)
        max_row_rate_error = max(max_row_rate_error, row_rate_error)
        max_bias_error = max(max_bias_error, bias_error)
        max_com_error = max(max_com_error, com_error)
        max_acceleration_error = max(max_acceleration_error, acceleration_error)
        samples.append({
            "index": sample_index,
            "row_shape": list(row.shape),
            "nv": int(model.nv),
            "com_z": float(com_z),
            "com_vz": float(com_vz),
            "row_qvel": row_rate,
            "fd_rate": fd_rate,
            "bias": float(bias),
            "fd_bias": fd_bias,
            "fd_com_z": fd_com_z,
            "com_acceleration_at_qacc_zero": float(bias),
            "fd_acceleration_at_qacc_zero": fd_bias,
            "row_rate_abs_error": row_rate_error,
            "bias_abs_error": bias_error,
            "com_z_abs_error": com_error,
            "acceleration_abs_error": acceleration_error,
        })

    result = {
        "metadata": {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "workspace": str(ROOT.parent),
            "python": sys.version,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "model": str(robot_model.ENV_XML),
            "nominal_length_m": NOMINAL_LENGTH,
            "finite_difference_step_s": FINITE_DIFFERENCE_STEP,
        },
        "interface": {
            "method": "Gbc486Dynamics.com_vertical_row",
            "return_order": ["J_com_z", "bias_com_z", "com_z", "com_vz"],
            "com_definition": "base subtree COM, same as robot.robot_com",
            "qacc_dimension": int(model.nv),
            "row_dimension": int(model.nv),
        },
        "checks": {
            "com_definition_consistent": max_com_error <= 1.0e-12,
            "velocity_jacobian_consistent": max_row_rate_error <= 1.0e-7,
            "bias_finite_difference_consistent": max_bias_error <= 1.0e-5,
            "repeatable": max(repeat_diffs, default=0.0) <= 1.0e-15,
            "finite": all(np.isfinite(v) for v in (
                max_row_rate_error, max_bias_error, max_com_error,
                max_acceleration_error)),
        },
        "metrics": {
            "max_row_rate_abs_error": max_row_rate_error,
            "max_bias_abs_error": max_bias_error,
            "max_com_z_abs_error": max_com_error,
            "max_acceleration_abs_error": max_acceleration_error,
            "max_repeat_abs_difference": max(repeat_diffs, default=0.0),
        },
        "samples": samples,
    }
    result["checks"]["passed"] = all(result["checks"].values())
    return result


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("e1_com_vertical_check")
    try:
        result = run()
    except Exception:
        logger.exception("E1 interface test failed")
        return 1
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("E1 result written: %s", RESULT_PATH)
    logger.info("E1 checks: %s", result["checks"])
    return 0 if result["checks"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
