"""离线录像：headless 跑一个工况，用 ``mujoco.Renderer`` 出帧、ffmpeg 编码 MP4。

按物理步取帧、按 ``--fps`` 节流，时间轴与仿真时间一致。输出到 ``circleg/videos/``。

用法：
    python tools/video.py --tag forward --duration 6 --command-velocity 1.5
    python tools/video.py --tag jump --duration 5 --jump-time 1.0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from APP.mujoco_mpc_wbc import (MuJoCoMpcWbcSimulation, SimulationConfig,  # noqa: E402
                                default_app_mpc_settings)
from Algorithm.mpc import MotionCommand  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circleg 离线录像")
    parser.add_argument("--tag", default="stand", help="输出文件名（不含扩展名）")
    parser.add_argument("--duration", type=float, default=5.0, help="仿真时长 [s]")
    parser.add_argument("--fps", type=int, default=50, help="输出帧率")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--terrain", choices=("plane", "rough", "field"), default="plane")
    parser.add_argument("--camera", choices=("free", "tracking"), default="tracking")
    parser.add_argument("--camera-distance", type=float, default=1.6)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    parser.add_argument("--camera-azimuth", type=float, default=140.0)
    parser.add_argument("--command-velocity", type=float, default=0.0,
                        help="0~hold 期间的速度指令 [m/s]")
    parser.add_argument("--command-hold", type=float, default=3.0,
                        help="速度指令保持时长 [s]")
    parser.add_argument("--command-yaw-rate", type=float, default=0.0,
                        help="0~hold 期间的偏航角速度指令 [rad/s]")
    parser.add_argument("--command-yaw-hold", type=float, default=None,
                        help="偏航指令保持时长 [s]（默认与 --command-hold 相同）")
    parser.add_argument("--jump-time", type=float, default=None,
                        help="起跳触发时刻 [s]；不给则不跳")
    parser.add_argument("--no-jump", action="store_true", help="关闭跳跃模块")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SimulationConfig(terrain=args.terrain, jump_enabled=not args.no_jump)
    sim = MuJoCoMpcWbcSimulation(config=config, mpc_settings=default_app_mpc_settings())
    model, data = sim.robot.model, sim.robot.data

    renderer = mujoco.Renderer(model, args.height, args.width)
    camera = None
    if args.camera == "tracking":
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = int(model.body("base_link").id)
        camera.distance = float(args.camera_distance)
        camera.azimuth = float(args.camera_azimuth)
        camera.elevation = float(args.camera_elevation)

    frames: list = []

    def grab(current) -> None:
        renderer.update_scene(current, camera=camera)
        frames.append(renderer.render().copy())

    hold = float(args.command_hold)
    yaw_hold = hold if args.command_yaw_hold is None else float(args.command_yaw_hold)
    velocity = float(args.command_velocity)
    yaw_rate = float(args.command_yaw_rate)

    def command(time_s: float) -> MotionCommand:
        return MotionCommand(velocity=(velocity if time_s < hold else 0.0),
                             yaw_rate=(yaw_rate if time_s < yaw_hold else 0.0))

    jump_times = None if args.jump_time is None else [float(args.jump_time)]
    summary = sim.run(command, args.duration, allow_driving=True, jump_times=jump_times,
                      frame_callback=grab, frame_interval=1.0 / args.fps)
    print({key: summary[key] for key in ("completed", "stop_reason", "simulated_seconds",
                                         "max_abs_pitch", "final_distance")})

    output_dir = ROOT / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.tag}_{args.fps}fps.mp4"
    ffmpeg = subprocess.run(
        [sys.executable, "-c", "import imageio_ffmpeg,sys;sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())"],
        capture_output=True, text=True, check=True).stdout.strip()
    encode = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
              "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
              "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output)]
    process = subprocess.Popen(encode, stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()
    print(f"written {output} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
