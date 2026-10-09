"""离线录像：headless 跑一个工况，用 ``mujoco.Renderer`` 出帧、ffmpeg 编码 MP4。

只在用户明确要求时录制（用户 2026-10-06 定的策略）：迭代过程用 headless 出数据，
视频只用于给人看现象。输出一律写到 ``circleg-robot/videos/``。

用法：
    python tools/video.py --tag balance --duration 3.0 --fps 50

时序说明：录像是**按物理步**取帧、按 ``--fps`` 节流（``frame_interval = 1/fps``），
所以录出来的时间轴就是真实物理时间（viewer 里的时间是被渲染拖慢的，不能当参考）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from gbc_mpc_wbc.logging_setup import get_logger, setup_logging  # noqa: E402
from gbc_mpc_wbc.simulation import (Disturbance, Gbc486MpcWbcSimulation,  # noqa: E402
                                    SimulationConfig, default_command)

_LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GBC486 离线录像")
    parser.add_argument("--tag", default="balance", help="输出文件名（不含扩展名）")
    parser.add_argument("--duration", type=float, default=3.0, help="仿真时长 [s]")
    parser.add_argument("--fps", type=int, default=50, help="输出帧率")
    # 分辨率上限受模型离屏缓冲限制（MJCF 里 <visual><global offwidth/offheight>，默认 640x480）；
    # 超了 mujoco.Renderer 会直接报 "Image width > framebuffer width"。
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--push-time", type=float, default=1.0, help="推扰时刻 [s]，<0 表示不加")
    parser.add_argument("--push-speed", type=float, default=0.35, help="推扰水平速度 [m/s]")
    parser.add_argument("--push-speed-z", type=float, default=0.0, help="推扰竖直速度 [m/s]（>0 用于触发离地）")
    parser.add_argument("--backend", default="wls")
    parser.add_argument("--variant", default="held_leg")
    parser.add_argument("--preset", default="tuned-agile")
    parser.add_argument("--explicit-jump-reference", action="store_true")
    parser.add_argument("--compression-only", action="store_true",
                        help="WP5：压缩完成/ABORT即停止，不执行 THRUST/FLIGHT/LANDING")
    parser.add_argument("--leg-length-compression", action="store_true",
                        help="VL1：COMPRESS/COMPRESS_BRAKE 用腿长行加速度驱动，不创建 com_vertical")
    parser.add_argument("--leg-length-thrust", action="store_true",
                        help="TR1：THRUST 用腿长行加速度驱动，不创建 com_vertical")
    parser.add_argument("--thrust-only", action="store_true",
                        help="TR1：相位将进入 FLIGHT 时停止，不执行 FLIGHT/LANDING 控制")
    parser.add_argument("--flight-zero-torque", action="store_true",
                        help="FL1：FLIGHT 或真离地时输出显式零力矩，不保持地面力矩")
    parser.add_argument("--flight-leg-retract", action="store_true",
                        help="FL2：几何离地时主动收腿（pd = 3000/150），抬高轮毂")
    parser.add_argument("--jump-height", type=float, default=None,
                        help="启用跳跃并给定目标质心抬升 [m]")
    parser.add_argument("--jump-stroke", type=float, default=0.12, help="压缩行程 [m]")
    parser.add_argument("--jump-time", type=float, default=1.0, help="起跳指令时刻 [s]")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(default_level="INFO")
    sim = Gbc486MpcWbcSimulation(SimulationConfig(
        jump_height=args.jump_height, jump_stroke=args.jump_stroke, jump_time=args.jump_time,
        backend=args.backend, model_variant=args.variant, preset=args.preset))
    model, data = sim.model, sim.data
    renderer = mujoco.Renderer(model, args.height, args.width)
    frames: list = []

    def grab(current) -> None:
        renderer.update_scene(current)
        frames.append(renderer.render().copy())

    disturbances = ()
    if args.push_time >= 0.0:
        disturbances = (Disturbance(time=args.push_time,
                                     delta_velocity=(args.push_speed, 0.0, args.push_speed_z)),)
    _LOGGER.info("开始录制：%.2f s / %d fps / 先平衡后推扰 %.2f m/s @ %.2f s",
                 args.duration, args.fps, args.push_speed, args.push_time)
    sim.run(default_command(), args.duration, disturbances=disturbances,
            frame_callback=grab, frame_interval=1.0 / args.fps,
            explicit_jump_reference=args.explicit_jump_reference,
            compression_only=args.compression_only,
            leg_length_compression=args.leg_length_compression,
            leg_length_thrust=args.leg_length_thrust,
            thrust_only=args.thrust_only,
            flight_zero_torque=args.flight_zero_torque,
            flight_leg_retract=args.flight_leg_retract)
    summary = sim.summary(args.duration)
    _LOGGER.info("仿真结束：完成=%s 停止原因=%s 里程=%.4f m",
                 summary["completed"], summary["stop_reason"], sim.distance)

    output_dir = REPO_ROOT / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.tag}_{args.fps}fps.mp4"
    # 必须用**当前解释器**（sys.executable）取 ffmpeg：裸 "python" 指向系统解释器，
    # 它不一定装了 imageio_ffmpeg（本工作区的依赖装在 .venv 里）。
    ffmpeg = subprocess.run([sys.executable, "-c",
                             "import imageio_ffmpeg,sys;sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())"],
                            capture_output=True, text=True, check=True).stdout.strip()
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()
    _LOGGER.info("完成：%s（%d 帧，%.1f MB）", output, len(frames), output.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
