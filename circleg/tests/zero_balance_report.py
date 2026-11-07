"""Summarize saved zero-command traces without running or tuning the controller."""
import gzip
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_LOGGER = logging.getLogger(__name__)


class BalanceReport:
    def __init__(self, directory):
        self.directory = directory

    def generate(self):
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        audit = {}
        for name in ("stand", "pitch", "pitch_negative"):
            with gzip.open(self.directory / f"{name}.json.gz", "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            records = payload["records"]
            force = np.array([r["predicted_force"] for r in records])
            actual = np.array([r["applied_force"] for r in records])
            normal = force[:, [1, 3]]
            tangent = force[:, [0, 2]]
            mu = payload["mpc_settings"]["friction_coefficient"]
            longitudinal_margin = mu*normal-np.abs(tangent)
            # The nominal model lumps lateral force at the midpoint. Check
            # whether a per-wheel lateral-force split can exist inside cones.
            lateral_capacity = np.sqrt(np.maximum((mu*normal)**2-tangent**2, 0.)).sum(axis=1)
            lateral_margin = lateral_capacity-np.abs(force[:, 4])
            actual_margin = mu*actual[:, :, 2]-np.linalg.norm(actual[:, :, :2], axis=2)
            time = np.array([r["time"] for r in records])
            state = np.array([r["state"] for r in records])
            attitude = np.array([r["attitude"] for r in records])
            audit[name] = dict(payload["summary"], min_predicted_normal=float(normal.min()),
                min_longitudinal_friction_margin=float(longitudinal_margin.min()),
                min_lateral_allocation_margin=float(lateral_margin.min()),
                min_actual_normal=float(actual[:, :, 2].min()),
                min_actual_friction_margin=float(actual_margin.min()),
                max_abs_roll=float(np.max(np.abs(attitude[:, 0]))),
                max_abs_yaw=float(np.max(np.abs(attitude[:, 2]))),
                contact_modes=sorted(set(r["mode"] for r in records)))
            axes[0].plot(time, state[:, 0]*1000, label=name)
            axes[1].plot(time, state[:, 1]*1000, label=name)
            axes[2].plot(time, np.rad2deg(attitude[:, 1]), label=name)
            _LOGGER.info("%s normal min=%.4f N, friction margin min=%.4f N", name, normal.min(), longitudinal_margin.min())
        for axis, ylabel in zip(axes, ("Position (mm)", "Velocity (mm/s)", "Pitch (deg)")):
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
            axis.legend()
        axes[-1].set_xlabel("Simulation time (s)")
        figure.suptitle("Zero velocity command: balance and remaining position offset")
        figure.tight_layout()
        figure.savefig(self.directory / "balance.png", dpi=150)
        plt.close(figure)
        (self.directory / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    BalanceReport(Path(__file__).resolve().parents[1] / "artifacts/zero-balance/final").generate()
