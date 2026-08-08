"""CirclegRobot command-line entry point."""

from __future__ import annotations

import argparse
import os

from APP.config import MODEL_PATHS
from APP.simulation import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CirclegRobot longitudinal LQR controller")
    parser.add_argument(
        "--terrain",
        choices=tuple(MODEL_PATHS),
        default=os.getenv("CIRCLEG_TERRAIN", "rough"),
    )
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--initial-pitch", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.terrain, max(float(args.duration), 0.0), args.headless, float(args.initial_pitch))


if __name__ == "__main__":
    main()
