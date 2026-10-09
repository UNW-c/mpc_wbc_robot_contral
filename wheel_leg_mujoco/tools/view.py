"""viewer 交互入口：零输入平衡 + 鼠标推扰，可选键盘速度 / 偏航指令。

与 circleg ``APP/mujoco_mpc_wbc.py`` 的 viewer 分支同一套：被动 viewer、相机跟随车体、
实时限速、鼠标推扰经 ``mjv_applyPerturb*`` 注入、关窗口即停。

用法（在 wheel_leg_mujoco 目录下）：
    ../.venv/Scripts/python.exe tools/view.py                          # 6 状态 held_leg + 零空间法
    ../.venv/Scripts/python.exe tools/view.py --model-variant leg_input
    ../.venv/Scripts/python.exe tools/view.py --backend wls --duration 30

操作：
    - 鼠标左键拖拽机身：推扰（零输入平衡用这个试）
    - 方向键上/下：速度指令，左/右：偏航指令（**注意：有速度输入的模式按用户要求尚未测试**）
    - 关闭窗口结束

只验证可行性，不调参。
"""
from __future__ import annotations

# 必须在导入 numpy / mujoco 之前设好：BLAS 线程数会改变浮点归约顺序，本机开环极点
# 11.6 rad/s，4 秒内足以把 1e-16 放大到宏观量级（实测同一组参数 T-d 从 3.6° 变 44°）。
# 评价与 viewer 必须用同一设置，否则实测与扫描对不上。
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gbc_mpc_wbc.simulation import Gbc486MpcWbcSimulation, SimulationConfig  # noqa: E402
from Algorithm.mpc import MotionCommand  # noqa: E402
from APP.mujoco_mpc_wbc import _PRESETS  # noqa: E402
from gbc_mpc_wbc.simulation import MAX_SPEED
from gbc_mpc_wbc.logging_setup import setup_logging  # noqa: E402

RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="wls", choices=["nullspace", "wls", "qp"])
    parser.add_argument("--model-variant", default="leg_input", choices=["held_leg", "leg_input"],
                        help="默认 leg_input（8 状态，对应 circleg 的 hip_input）")
    parser.add_argument("--preset", default="tuned-agile",
                        choices=list(_PRESETS) + ["none"],
                        help="circleg preset；默认 tuned-agile（含 ZMP 参考）；none = circleg viewer 默认")
    parser.add_argument("--tuned", action="store_true",
                        help="用本轮扫描选出的参数（gbc_mpc_wbc.objective.TUNED）覆盖 MPC 权重与 WBC 增益")
    parser.add_argument("--hip-angle-limit", type=float, default=None,
                        help="覆盖 TUNED 的 MPC 共模腿角预算 [rad]（仅本次运行，不改默认）")
    parser.add_argument("--position-weight", type=float, default=None,
                        help="覆盖 TUNED 的 position 权重（仅本次运行，不改默认）")
    parser.add_argument("--duration", type=float, default=float("inf"),
                        help="仿真时长 [s]；默认无限，关窗口结束")
    parser.add_argument("--no-keyboard", action="store_true", help="不启用键盘指令源")
    parser.add_argument("--jump-height", type=float, default=None,
                        help="启用跳跃并给定目标质心抬升 [m]（例如 0.10）")
    parser.add_argument("--jump-stroke", type=float, default=0.12, help="压缩行程 [m]")
    parser.add_argument("--jump-time", type=float, default=2.0, help="起跳指令时刻 [s]")
    parser.add_argument("--leg-length", type=float, default=None,
                        help="标称腿长 [m]，默认 0.28（行程带中点）；机构范围约 0.14–0.43")
    parser.add_argument("--output", type=Path, default=None, help="结束时写入的记录（JSON.gz）")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup_logging(default_level="INFO")

    from mujoco import viewer as mujoco_viewer
    from APP.keyboard_task import KeyboardCommandSource

    config = SimulationConfig(jump_height=args.jump_height, jump_stroke=args.jump_stroke, jump_time=args.jump_time, backend=args.backend, model_variant=args.model_variant, preset=args.preset,
                              **({} if args.leg_length is None else {"nominal_length": args.leg_length}))
    if args.tuned:
        from gbc_mpc_wbc.objective import MODEL_VARIANT, TUNED, _apply_task_weights, _gains, _settings
        # 单项覆盖只作用于本次运行的副本，不改 objective.TUNED（默认配置保持不动）
        params = dict(TUNED)
        if args.hip_angle_limit is not None:
            params["hip_angle_limit"] = float(args.hip_angle_limit)
        if args.position_weight is not None:
            params["position"] = float(args.position_weight)
        variant = str(params.get("model_variant", MODEL_VARIANT))
        settings, _, _ = _settings(params)
        gains = _gains(params)
        _apply_task_weights(params)             # WLS/QP 任务权重（resolver 构造时读模块级 TASK_WEIGHTS）
        config = SimulationConfig(jump_height=args.jump_height, jump_stroke=args.jump_stroke, jump_time=args.jump_time, backend=args.backend, model_variant=variant,
                                  preset=args.preset, wbc_gains=gains,
                                  wheel_torque_limit=params.get("wheel_torque_limit"),
                                  leg_torque_limit=params.get("leg_torque_limit"),
                                  **({} if args.leg_length is None else {"nominal_length": args.leg_length}))
        simulation = Gbc486MpcWbcSimulation(config=config, mpc_settings=settings)
        simulation.tasks.gains = gains          # simulation 会用 preset 的带宽覆盖 wbc_gains，这里显式接管
        print("生效参数: 模型=" + variant + " " + " ".join(
            f"{k}={params[k]}" for k in ("hip_angle_limit", "position", "velocity", "pitch", "pitch_rate")))
    else:
        simulation = Gbc486MpcWbcSimulation(config=config)
    model = simulation.model
    settings = simulation.mpc.settings
    print(f"模型线 {simulation.reduced.state_dim} 状态 / {simulation.reduced.input_dim} 输入，"
          f"不稳定极点 {simulation.reduced.unstable_pole:.3f} rad/s，"
          f"WBC 后端 {args.backend}，preset {args.preset}")
    print(f"标称腿长 {simulation.pose.length:.3f} m，腿角 {simulation.pose.angle:+.4f} rad，"
          f"质心高 {simulation.reduced.metadata['com_height']:.3f} m")
    print(f"ZMP 参考 {settings.zmp_reference}（split {settings.zmp_split}，z_com {settings.zmp_com_height:.3f} m，"
          f"r_crank {settings.zmp_crank_radius:.3f} m），"
          f"共模腿角限 {settings.hip_angle_limit} rad，S 曲线 {settings.limits.use_jerk_limit}")

    keyboard = None
    if not args.no_keyboard:
        from dataclasses import replace as _replace
        from APP.config import DEFAULT_CONFIG
        # 键盘步长跟本工作区的速度上限（circleg 共享配置里是 2.5；不改那个文件）
        keyboard = KeyboardCommandSource(_replace(
            DEFAULT_CONFIG, keyboard=_replace(DEFAULT_CONFIG.keyboard, speed_step=MAX_SPEED)))
        print(f"键盘指令源可用：{keyboard.available}（方向键上/下=速度，左/右=偏航；"
              f"有速度输入的模式尚未测试）")

    def command(_time):
        if keyboard is None:
            return MotionCommand(velocity=0.0, yaw_rate=0.0)
        return MotionCommand(velocity=keyboard.read(), yaw_rate=keyboard.read_yaw_rate())

    with mujoco_viewer.launch_passive(model, simulation.data) as viewer:
        # 2026-10-05 用户要求 viewer 时间轴贴近实时。实测：headless 0.95 倍、
        # viewer 3.56 倍（已把 sync 从每物理步改为 60 Hz 之后），剩余差距是渲染开销
        # 与仿真线程抢 CPU —— 主因是 base_link 单个网格 91441 顶点。
        # 这里关掉该网格下最贵的三项（阴影/反射/skybox）与接触点力可视化。
        for _flag in ("mjVIS_SHADOW", "mjVIS_REFLECTION", "mjVIS_SKYBOX",
                      "mjVIS_CONTACTFORCE", "mjVIS_CONTACTPOINT", "mjVIS_TRANSPARENT"):
            try:
                viewer.opt.flags[getattr(mujoco.mjtVisFlag, _flag)] = 0
            except (AttributeError, KeyError, TypeError):
                pass
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = int(model.body("base").id)
        summary = simulation.run(command, args.duration, viewer=viewer)

    print(json.dumps({k: v for k, v in summary.items() if k != "model"}, ensure_ascii=False, indent=1))
    if args.output is not None:
        import gzip
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.output, "wt", encoding="utf-8") as handle:
            json.dump(simulation.records, handle)
        print(f"记录已写入 {args.output}")


if __name__ == "__main__":
    main()
