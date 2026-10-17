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
import bisect
import gzip
import json
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
from Algorithm.mpc import MotionCommand  # noqa: E402

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
    parser.add_argument("--flight-leg-extend", action="store_true",
                        help="FL3：飞行腿剖面（上升收腿/下落伸腿）+ LANDING 缓冲")
    parser.add_argument("--jump-height", type=float, default=None,
                        help="启用跳跃并给定目标质心抬升 [m]")
    parser.add_argument("--jump-stroke", type=float, default=0.12, help="压缩行程 [m]")
    parser.add_argument("--jump-time", type=float, default=1.0, help="起跳指令时刻 [s]")
    # CMD（16.3）：有输入速度指令录像。默认 0.0 = 保持原行为（零指令）。
    parser.add_argument("--command-velocity", type=float, default=0.0,
                        help="CMD：0~hold 期间的速度指令 [m/s]，0 = 不加指令")
    # 回放一次真实会话的键盘输入（第 25 节）：读 view 的 --output records，按 cmd_v/cmd_w 复刻；
    # --jump-times 给出每次空格起跳的仿真时刻（配合 --jump-time inf 关掉定时触发）。
    parser.add_argument("--replay", type=Path, default=None,
                        help="回放文件：view --output 写出的 records json.gz（按 cmd_v/cmd_w 时序）")
    parser.add_argument("--jump-times", type=str, default="",
                        help="空格起跳时刻列表，逗号分隔 [s]（配合 --jump-time inf）")
    parser.add_argument("--command-hold", type=float, default=3.0,
                        help="CMD：速度指令保持时长 [s]（之后归零）")
    # 相机：默认自由相机（原行为）；tracking 时跟随 body("base")，否则机体一走远就出画。
    parser.add_argument("--camera", choices=("free", "tracking"), default="free",
                        help="free = 原来的自由相机；tracking = 跟随机体质心")
    parser.add_argument("--camera-distance", type=float, default=1.6, help="tracking 相机距离 [m]")
    parser.add_argument("--camera-elevation", type=float, default=-12.0, help="tracking 相机俯仰 [deg]")
    parser.add_argument("--camera-azimuth", type=float, default=90.0, help="tracking 相机方位 [deg]")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(default_level="INFO")
    sim = Gbc486MpcWbcSimulation(SimulationConfig(
        jump_height=args.jump_height, jump_stroke=args.jump_stroke, jump_time=args.jump_time,
        backend=args.backend, model_variant=args.variant, preset=args.preset))
    model, data = sim.model, sim.data
    renderer = mujoco.Renderer(model, args.height, args.width)
    camera = None
    if args.camera == "tracking":
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = int(model.body("base").id)
        camera.distance = float(args.camera_distance)
        camera.azimuth = float(args.camera_azimuth)
        camera.elevation = float(args.camera_elevation)
    frames: list = []

    def grab(current) -> None:
        if camera is None:
            renderer.update_scene(current)
        else:
            renderer.update_scene(current, camera=camera)
        frames.append(renderer.render().copy())

    disturbances = ()
    if args.push_time >= 0.0:
        disturbances = (Disturbance(time=args.push_time,
                                     delta_velocity=(args.push_speed, 0.0, args.push_speed_z)),)
    # 回放：把 records 里的指令时序压成"值变化点"分段，回放时按时间取当段值。
    replay_segments = None
    if args.replay is not None:
        with gzip.open(args.replay, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload["records"] if isinstance(payload, dict) else payload
        segments = []
        previous = None
        for item in records:
            key = (round(float(item["cmd_v"]), 6), round(float(item["cmd_w"]), 6))
            if key != previous:
                segments.append((float(item["t"]), key[0], key[1]))
                previous = key
        replay_segments = segments
        _LOGGER.info("回放输入：%d 段指令 %s", len(segments),
                     [(round(t, 3), v, w) for t, v, w in segments])
    jump_times = [float(x) for x in args.jump_times.split(",") if x.strip()]
    jump_done = [False] * len(jump_times)
    _LOGGER.info("回放起跳时刻：%s", jump_times)

    # CMD：与 view.py / jump_diagnostic 同签名的按时间定值回调；velocity=0 时退回零指令对象。
    if replay_segments is not None:
        segment_times = [seg[0] for seg in replay_segments]

        def command(time_s: float) -> MotionCommand:
            index = bisect.bisect_right(segment_times, float(time_s) + 1e-9) - 1
            velocity, yaw_rate = ((replay_segments[index][1], replay_segments[index][2])
                                  if index >= 0 else (0.0, 0.0))
            for k, when in enumerate(jump_times):
                if not jump_done[k] and float(time_s) >= when:
                    jump_done[k] = True
                    sim.request_jump()
            return MotionCommand(velocity=velocity, yaw_rate=yaw_rate)
    elif args.command_velocity != 0.0:
        hold = float(args.command_hold)
        value = float(args.command_velocity)

        def command(time_s: float) -> MotionCommand:
            return MotionCommand(velocity=(value if float(time_s) < hold else 0.0), yaw_rate=0.0)
    else:
        command = default_command()
    _LOGGER.info("开始录制：%.2f s / %d fps / 先平衡后推扰 %.2f m/s @ %.2f s / camera=%s",
                 args.duration, args.fps, args.push_speed, args.push_time, args.camera)
    sim.run(command, args.duration, disturbances=disturbances,
            frame_callback=grab, frame_interval=1.0 / args.fps,
            explicit_jump_reference=args.explicit_jump_reference,
            compression_only=args.compression_only,
            leg_length_compression=args.leg_length_compression,
            leg_length_thrust=args.leg_length_thrust,
            thrust_only=args.thrust_only,
            flight_zero_torque=args.flight_zero_torque,
            flight_leg_retract=args.flight_leg_retract,
            flight_leg_extend=args.flight_leg_extend)
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
