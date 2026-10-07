"""离线生成 GBC486 ``leg_input`` 模型的稳态加速度能力表（A1 / A3）。

用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/make_capacity.py

产物：``gbc_mpc_wbc/capacity_gbc486.json``，键：
    leg_length / generated_at / preset / pitch_max / brake_max / step / maximum /
    wheel_radius / mass / com_height / friction /
    accel_capacity / accel_capacity_tilted / accel_capacity_tilted_brake

为什么必须离线：表里是几十次非线性最小二乘 + fsolve，每次调用要几百次 ``derivative``
（每次含一次逆运动学 + 两次 ``mj_forward`` + 一次 KKT 解）。放进 ``build()`` 会让
每次建模仿真都多等数秒，而这几张表只随几何/质量/腿长变化。

算法与 circleg ``APP/mujoco_balance_model.py`` 一致：
- ``accel_capacity``      ：θ = 0 的稳态族 ``(a, q_c)``（带权正则压向 θ=0）；
- ``accel_capacity_tilted``：固定前倾 ``pitch_max`` 的 ``(q_c, a)``；
- ``..._tilted_brake``     ：刹车后仰 ``brake_max`` 的同一张表。
本实现额外用**执行器力矩限 + 摩擦锥**截断（circleg 原版只靠 q_c 单调性饱和截断）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gbc_mpc_wbc import robot as R  # noqa: E402
from gbc_mpc_wbc.dynamics import Gbc486Dynamics  # noqa: E402
from gbc_mpc_wbc.model_builder import (ACCEL_CAPACITY_MAX, ACCEL_CAPACITY_STEP,  # noqa: E402
                                       GRAVITY, LegInputModelBuilder, capacity_path)
from gbc_mpc_wbc.simulation import gbc486_mpc_settings  # noqa: E402
from gbc_mpc_wbc.logging_setup import get_logger, setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg-length", type=float, default=0.28, help="标称腿长 [m]")
    parser.add_argument("--preset", default="tuned-agile", help="取 pitch_feedforward_max / brake_pitch_max 的 preset")
    parser.add_argument("--step", type=float, default=ACCEL_CAPACITY_STEP, help="a 轴步长 [m/s^2]")
    parser.add_argument("--maximum", type=float, default=ACCEL_CAPACITY_MAX, help="a 轴上限 [m/s^2]")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # 能力表的截断原因（力矩/摩擦/单调性）走 INFO，必须打开才能看到
    setup_logging(default_level="INFO")

    model, data = R.load()
    pose = R.standing_pose(model, data, args.leg_length)
    data.qpos[:] = pose.qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    dynamics = Gbc486Dynamics(model, data)
    builder = LegInputModelBuilder(dynamics, pose.length)

    # 必须用**本机生效**的倾角上限，不能用 circleg preset 的 20deg：
    # A1 拿同一条 pitch_feedforward_max 去反查能力，两处口径必须一致。
    reduced = builder.build()
    settings = gbc486_mpc_settings(model, dynamics, reduced, leg_length=pose.length,
                                   preset=args.preset)
    pitch_max = float(settings.pitch_feedforward_max)
    brake_max = None if settings.brake_pitch_max is None else float(settings.brake_pitch_max)
    print("标称腿长 %.3f m，质心高 %.4f m，质量 %.3f kg，轮半径 %.4f m，摩擦 %.2f"
          % (pose.length, builder._capacity_com_height, builder._capacity_mass,
             builder.radius, builder._capacity_friction))
    print("倾角表角度：起步 %.4f rad (%.1f°)，刹车 %s"
          % (pitch_max, np.degrees(pitch_max),
             "—" if brake_max is None else "%.4f rad (%.1f°)" % (brake_max, np.degrees(brake_max))))

    start = time.perf_counter()
    tables = builder.capacity_tables(pitch_max, brake_max, step=args.step, maximum=args.maximum)
    wall = time.perf_counter() - start

    base = tables["accel_capacity"]
    print("\n== accel_capacity（θ=0 稳态族）实测 %d 点，耗时 %.1f s ==" % (len(base["a"]), wall))
    print("%10s %12s %12s %10s %12s" % ("a [m/s^2]", "q_c [mrad]", "解析 q_c", "偏差", "单轮力 [N]"))
    z, length = builder._capacity_com_height, pose.length
    for a, q in zip(base["a"], base["q_c"]):
        analytic = np.arcsin(min(a * z / (GRAVITY * length), 0.999))
        print("%10.2f %12.1f %12.1f %9.1f%% %12.1f"
              % (a, q * 1e3, analytic * 1e3, 100.0 * (q - analytic) / max(analytic, 1e-9),
                 builder._capacity_mass * a / 2.0))
    for key in ("accel_capacity_tilted", "accel_capacity_tilted_brake"):
        if key in tables:
            table = tables[key]
            print("\n== %s（θ=%.3f rad）%d 点 ==" % (key, table["theta"], len(table["q_c"])))
            print("   q_c [mrad]: " + " ".join("%.0f" % (q * 1e3) for q in table["q_c"]))
            print("   a   [m/s^2]: " + " ".join("%.2f" % a for a in table["a"]))

    payload = dict(
        leg_length=float(pose.length), generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        preset=args.preset, pitch_max=pitch_max, brake_max=brake_max,
        step=float(args.step), maximum=float(args.maximum),
        wheel_radius=float(builder.radius), mass=float(builder._capacity_mass),
        com_height=float(builder._capacity_com_height), friction=float(builder._capacity_friction),
        **tables,
    )
    out = args.out or capacity_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写入 %s" % out)


if __name__ == "__main__":
    main()
