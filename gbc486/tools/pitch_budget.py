"""落地 pitch 预算分解（只读后处理）。

对应文档：`docs/agent交互文档_落地pitch优化_测试方案部署.txt` 第 8 节。

本工具只读 `results/*.json.gz`，不 import 控制层、不改变任何控制量，也不写入 results/。
允许 import `gbc_mpc_wbc.leg_angle_anchor`（纯函数模块），用于还原世界系摆角参考。
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gbc_mpc_wbc.leg_angle_anchor import WORLD_TARGET_DEFAULT, phi_body_reference, phi_eq
from gbc_mpc_wbc.logging_setup import setup_logging

_LOGGER = logging.getLogger(__name__)

# [kg*m^2] 机体俯仰等效惯量。来源：历史辨识结果（权限实测 I_eff(俯仰)=0.1142）。
I_EFF_PITCH = 0.1142

# [N*m] 髋关节执行器限幅。来源：MJCF/robot.xml:176-182。
HIP_TORQUE_LIMIT = 40.0

# 摆角在两腿 leg_state 中的下标：leg_state = [l_R, angle_R, l_L, angle_L]。
PHI_INDEX = (1, 3)

# 分段所需字段；缺任何一个都必须报错退出，不允许跳过。
REQUIRED_FRAME_KEYS = (
    "t", "ctrl", "qvel", "pitch", "pitch_rate", "leg_state", "leg_rate",
    "leg_length_R", "leg_length_L", "output_source", "jump",
)

FLIGHT_SOURCE = "flight_leg_profile"
FLIGHT_PHASE = "FLIGHT"
START_PHASE = "COMPRESS"

MIN_FRAMES_PER_SEGMENT = 50


class BudgetError(RuntimeError):
    """输入数据不满足分段或自检前提。"""


def _load(path: Path) -> tuple[dict, list[dict]]:
    if not path.exists():
        raise BudgetError(f"结果文件不存在: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise BudgetError("结果文件没有 frames")
    missing = [key for key in REQUIRED_FRAME_KEYS if key not in frames[0]]
    if missing:
        raise BudgetError(f"frames 缺少字段: {missing}（口径与部署文档 8.1 节不一致）")
    return document, frames


def _jump_phase(frame: dict) -> str | None:
    jump = frame.get("jump")
    return None if not isinstance(jump, dict) else jump.get("phase")


def _flight_leg_ref(frame: dict) -> list | None:
    jump = frame.get("jump")
    if not isinstance(jump, dict):
        return None
    return jump.get("flight_leg_ref")


def _find_start(frames: list[dict]) -> int:
    for index, frame in enumerate(frames):
        if _jump_phase(frame) == START_PHASE:
            return index
    for index, frame in enumerate(frames):
        jump = frame.get("jump")
        if isinstance(jump, dict) and jump.get("active"):
            return index
    raise BudgetError(f"本结果没有跳跃（既无 {START_PHASE} 相位也无 active 跳跃）")


def _find_k0(frames: list[dict], start: int) -> tuple[int, str]:
    for index in range(start, len(frames)):
        if _jump_phase(frames[index]) == FLIGHT_PHASE:
            return index, f"jump.phase 首次为 {FLIGHT_PHASE}"
    for index in range(start, len(frames)):
        if frames[index].get("output_source") == FLIGHT_SOURCE:
            return index, f"output_source 首次为 {FLIGHT_SOURCE}（相位机未出现 {FLIGHT_PHASE}）"
    raise BudgetError(f"找不到离地帧（{FLIGHT_PHASE} 与 {FLIGHT_SOURCE} 都未出现）")


def _find_k2(frames: list[dict], k0: int) -> int:
    for index in range(k0 + 1, len(frames)):
        if frames[index].get("output_source") != FLIGHT_SOURCE:
            return index
    raise BudgetError(f"{FLIGHT_SOURCE} 一直持续到结果末尾，无法界定伸腿段终点")


def _find_k1(frames: list[dict], k0: int, k2: int) -> tuple[int, int, str, int, str]:
    """返回 (选用的 k1, 候选一, 说明一, 候选二, 说明二)。"""
    mode_candidate = None
    for index in range(k0 + 1, k2 + 1):
        previous = _flight_leg_ref(frames[index - 1]) or []
        current = _flight_leg_ref(frames[index]) or []
        if previous and current and previous[0] != current[0]:
            mode_candidate = index
            break

    lengths = np.array([float(frames[index]["leg_length_R"]) for index in range(k0, k2 + 1)])
    trough = int(np.argmin(lengths))
    length_candidate = None
    for offset in range(trough + 1, len(lengths)):
        if lengths[offset] > lengths[trough]:
            length_candidate = k0 + offset
            break

    candidates = [(index, label) for index, label in (
        (mode_candidate, "flight_leg_ref[0] 模式切换"),
        (length_candidate, "leg_length_R 触底回升"),
    ) if index is not None]
    if not candidates:
        raise BudgetError("找不到收腿段与伸腿段的分界（模式未切换且腿长未回升）")
    chosen, label = min(candidates, key=lambda item: item[0])
    return (chosen, mode_candidate, "flight_leg_ref[0] 模式切换",
            length_candidate, "leg_length_R 触底回升")


def _phi_world(frame: dict, sign: int) -> tuple[float, float]:
    pitch = float(frame["pitch"])
    leg = np.asarray(frame["leg_state"], dtype=float)
    return (float(leg[PHI_INDEX[0]]) + sign * pitch,
            float(leg[PHI_INDEX[1]]) + sign * pitch)


def _ground_sign(frames: list[dict], start: int) -> tuple[int, float, float]:
    """用起跳前的落地静止帧确定 phi_world = leg_state +/- pitch 的符号。

    地面静止时 phi_world 应接近 phi_eq(nominal)=WORLD_TARGET_DEFAULT。
    """
    candidates = []
    for index in range(0, start):
        frame = frames[index]
        contact = frame.get("contact_geometry")
        if not contact or not all(bool(flag) for flag in contact):
            continue
        if abs(float(frame["pitch_rate"])) > 1.0e-3:
            continue
        candidates.append(index)
    if not candidates:
        raise BudgetError("起跳前没有找到地面静止帧，无法确定 phi_world 的符号约定")
    index = candidates[-1]
    errors = {}
    for sign in (+1, -1):
        phi_r, phi_l = _phi_world(frames[index], sign)
        errors[sign] = 0.5 * (abs(phi_r - WORLD_TARGET_DEFAULT) + abs(phi_l - WORLD_TARGET_DEFAULT))
    sign = min(errors, key=errors.get)
    return sign, errors[sign], float(frames[index]["t"])


def _trapezoid(values: np.ndarray, times: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(times)))


def _segment_metrics(frames: list[dict], start: int, end: int, sign: int) -> dict:
    if end <= start:
        raise BudgetError(f"段过短：帧区间 {start}..{end}")
    count = end - start + 1
    times = np.array([float(frames[i]["t"]) for i in range(start, end + 1)])
    ctrl = np.array([np.asarray(frames[i]["ctrl"], dtype=float) for i in range(start, end + 1)])
    pitches = np.array([float(frames[i]["pitch"]) for i in range(start, end + 1)])
    rates = np.array([float(frames[i]["pitch_rate"]) for i in range(start, end + 1)])
    phi_world = np.array([_phi_world(frames[i], sign) for i in range(start, end + 1)])

    tau_hip = ctrl[:, PHI_INDEX[0]] - ctrl[:, PHI_INDEX[1]]   # 左右角度关节轴向镜像，净俯仰反作用取差
    tau_wheel = ctrl[:, 4] + ctrl[:, 5]
    int_tau_hip = _trapezoid(tau_hip, times)
    int_tau_wheel = _trapezoid(tau_wheel, times)
    wheel_zero_frames = int(np.sum(np.abs(ctrl[:, 4]) <= 1.0e-12)
                            + np.sum(np.abs(ctrl[:, 5]) <= 1.0e-12))
    return {
        "frames": count,
        "index": (start, end),
        "t": (float(times[0]), float(times[-1])),
        "d_pitch": float(pitches[-1] - pitches[0]),
        "d_pitch_rate": float(rates[-1] - rates[0]),
        "d_phi_world_R": float(phi_world[-1][0] - phi_world[0][0]),
        "d_phi_world_L": float(phi_world[-1][1] - phi_world[0][1]),
        "int_tau_hip": int_tau_hip,
        "int_tau_wheel": int_tau_wheel,
        "max_abs_tau_hip": float(np.max(np.abs(tau_hip))),
        "max_abs_tau_wheel": float(np.max(np.abs(ctrl[:, 4:6]))),
        "wheel_zero_frames": wheel_zero_frames,
        "saturated_frames": int(np.sum(np.abs(ctrl[:, PHI_INDEX[0]]) >= HIP_TORQUE_LIMIT * 0.999)
                                + np.sum(np.abs(ctrl[:, PHI_INDEX[1]]) >= HIP_TORQUE_LIMIT * 0.999)),
    }


def _consistency(segments: list[dict]) -> dict:
    """驱动源归属检查。

    机体俯仰的力矩-动量一致性不能当硬阈值判据：腿在摆动时基座俯仰与腿摆角之间存在动量交换，
    ``I_eff`` 是腿锁定条件下辨识的，单独用它预测会系统性偏高。
    因此这里只判三件事：哪一组执行器在出力（轮力矩是否恒零）、符号是否一致、解释比例多少。
    """
    int_tau_hip = sum(segment["int_tau_hip"] for segment in segments)
    int_tau_wheel = sum(segment["int_tau_wheel"] for segment in segments)
    d_rate = sum(segment["d_pitch_rate"] for segment in segments)
    hip_predicted = int_tau_hip / I_EFF_PITCH
    wheel_predicted = int_tau_wheel / I_EFF_PITCH
    scale = max(abs(d_rate), 1.0e-9)
    wheels_silent = all(segment["wheel_zero_frames"] == 2 * segment["frames"]
                        for segment in segments)
    hip_sign_agrees = (hip_predicted * d_rate) > 0.0
    attributed = wheels_silent and hip_sign_agrees and abs(int_tau_hip) > 1.0e-6
    return {
        "int_tau_hip": int_tau_hip,
        "int_tau_wheel": int_tau_wheel,
        "d_pitch_rate": d_rate,
        "hip_predicted": hip_predicted,
        "wheel_predicted": wheel_predicted,
        "explain_ratio": hip_predicted / scale,
        "wheel_share": abs(wheel_predicted) / scale,
        "wheels_silent": wheels_silent,
        "hip_sign_agrees": hip_sign_agrees,
        "attributed": attributed,
    }


def _write_csv(path: Path, frames: list[dict], start: int, end: int, sign: int) -> None:
    header = ("t,phase,output_source,pitch,pitch_rate,phi_R,phi_L,phi_world_R,phi_world_L,"
              "l_ref,phi_ref_world,phi_ref_body,tau_hip_R,tau_hip_L,tau_wheel_R,tau_wheel_L,"
              "leg_length_R,leg_length_L")
    lines = [header]
    for index in range(start, end + 1):
        frame = frames[index]
        leg = np.asarray(frame["leg_state"], dtype=float)
        ctrl = np.asarray(frame["ctrl"], dtype=float)
        pitch = float(frame["pitch"])
        ref = _flight_leg_ref(frame)
        l_ref = None if not ref or len(ref) < 2 else float(ref[1])
        world = None if l_ref is None else float(phi_eq(l_ref))
        body = None if world is None else float(phi_body_reference(world, pitch))
        phi_world = _phi_world(frame, sign)
        lines.append(",".join(str(value) for value in (
            f"{float(frame['t']):.6f}", _jump_phase(frame) or "", frame.get("output_source", ""),
            f"{pitch:.8f}", f"{float(frame['pitch_rate']):.8f}",
            f"{float(leg[PHI_INDEX[0]]):.8f}", f"{float(leg[PHI_INDEX[1]]):.8f}",
            f"{phi_world[0]:.8f}", f"{phi_world[1]:.8f}",
            "" if l_ref is None else f"{l_ref:.6f}",
            "" if world is None else f"{world:.8f}",
            "" if body is None else f"{body:.8f}",
            f"{float(ctrl[PHI_INDEX[0]]):.6f}", f"{float(ctrl[PHI_INDEX[1]]):.6f}",
            f"{float(ctrl[4]):.6f}", f"{float(ctrl[5]):.6f}",
            f"{float(frame['leg_length_R']):.6f}", f"{float(frame['leg_length_L']):.6f}",
        )))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _LOGGER.info("逐帧 CSV 已写出: %s（%d 行）", path, len(lines) - 1)


def _render(path: Path, document: dict, frames: list[dict], parts: dict) -> str:
    metadata = document.get("metadata", {})
    out = []
    out.append("=" * 72)
    out.append("落地 pitch 预算分解")
    out.append("=" * 72)
    out.append(f"结果文件      : {path.name}")
    out.append(f"git_commit    : {metadata.get('git_commit')}")
    out.append(f"command       : {metadata.get('command')}")
    out.append(f"config        : {metadata.get('config')}")
    out.append("")
    out.append(f"符号约定      : phi_world = leg_state[i] {'+' if parts['sign'] > 0 else '-'} pitch")
    out.append(f"                  地面静止帧 t={parts['sign_t']:.3f} s，"
               f"与 phi_eq(0.28)={WORLD_TARGET_DEFAULT:.4f} 的平均偏差 {parts['sign_error']:.4f} rad")
    out.append("")
    out.append(f"起点 k_start  : 帧 {parts['start']}，t={parts['t_start']:.3f} s")
    out.append(f"k0 离地       : 帧 {parts['k0']}，t={parts['t_k0']:.3f} s，依据：{parts['k0_rule']}")
    out.append(f"k1 收腿终     : 帧 {parts['k1']}，t={parts['t_k1']:.3f} s，依据：{parts['k1_rule']}"
               f"（候选一 {parts['k1_a']} / {parts['k1_a_label']}；"
               f"候选二 {parts['k1_b']} / {parts['k1_b_label']}）")
    out.append(f"k2 剖面终     : 帧 {parts['k2']}，t={parts['t_k2']:.3f} s")
    out.append("")
    head = (f"{'段':<6}{'帧区间':<16}{'t 区间':<22}{'d_pitch':>10}{'占比':>8}"
            f"{'d_phiW_R':>10}{'d_phiW_L':>10}{'int_tau_hip':>13}{'max|tH|':>9}{'max|tW|':>9}{'饱和帧':>8}")
    out.append(head)
    out.append("-" * len(head))
    total = sum(segment["d_pitch"] for segment in parts["segments"])
    for name, segment in zip(("T0", "T1", "T2"), parts["segments"]):
        share = "n/a" if abs(total) < 1.0e-12 else f"{100.0 * segment['d_pitch'] / total:.1f}%"
        frames_text = f"{segment['index'][0]}..{segment['index'][1]}"
        time_text = f"{segment['t'][0]:.3f}..{segment['t'][1]:.3f}"
        out.append(f"{name:<6}{frames_text:<16}{time_text:<22}{segment['d_pitch']:>10.4f}"
                   f"{share:>8}{segment['d_phi_world_R']:>10.4f}{segment['d_phi_world_L']:>10.4f}"
                   f"{segment['int_tau_hip']:>13.4f}{segment['max_abs_tau_hip']:>9.2f}"
                   f"{segment['max_abs_tau_wheel']:>9.3f}{segment['saturated_frames']:>8d}")
    out.append("-" * len(head))
    out.append(f"{'合计':<6}{'':<16}{'':<22}{total:>10.4f}{'100.0%':>8}")
    out.append("")
    check = parts["consistency"]
    out.append("驱动源归属（部署文档 8.5 第 4 条，只统计飞行段 T1+T2；T0 在地上，接触力才是主项）")
    out.append("  髋力矩积分口径             : ctrl[1] - ctrl[3]（左右角度关节轴向镜像，净俯仰反作用取差）")
    out.append(f"  实测 d_pitch_rate          : {check['d_pitch_rate']:+.6f} rad/s")
    out.append(f"  髋力矩积分                 : {check['int_tau_hip']:+.6f} N*m*s")
    out.append(f"  髋力矩预测                 : {check['hip_predicted']:+.6f} rad/s"
               f"（解释比例 {100.0 * check['explain_ratio']:+.1f}%，"
               f"符号{'一致' if check['hip_sign_agrees'] else '不一致'}）")
    out.append(f"  轮力矩积分                 : {check['int_tau_wheel']:+.6f} N*m*s"
               f"（{'飞行段恒为零' if check['wheels_silent'] else '飞行段有出力'}）")
    out.append("  归属结论                   : "
               + ("腿部执行器为唯一驱动源" if check["attributed"]
                  else "不成立，存在其他驱动源，必须单列"))
    out.append("  说明                       : 解释比例不等于 1 属正常现象（腿摆动时基座俯仰与腿摆角"
               "存在动量交换，I_eff 为腿锁定条件辨识值），本判据不设硬阈值。")
    out.append("")
    out.append("自检")
    out.append(f"  三段角度之和 - 总变化      : {parts['sum_residual']:+.3e} rad"
               f"  -> {'通过' if abs(parts['sum_residual']) <= 1.0e-9 else '不通过'}")
    for name, segment in zip(("T0", "T1", "T2"), parts["segments"]):
        flag = "通过" if segment["frames"] >= MIN_FRAMES_PER_SEGMENT else "段过短"
        out.append(f"  {name} 帧数 {segment['frames']:<6} -> {flag}")
    return "\n".join(out)


def build_budget(path: Path) -> tuple[str, dict]:
    document, frames = _load(path)
    start = _find_start(frames)
    k0, k0_rule = _find_k0(frames, start)
    k2 = _find_k2(frames, k0)
    k1, k1_a, k1_a_label, k1_b, k1_b_label = _find_k1(frames, k0, k2)
    sign, sign_error, sign_t = _ground_sign(frames, start)

    segments = [
        _segment_metrics(frames, start, k0, sign),
        _segment_metrics(frames, k0, k1, sign),
        _segment_metrics(frames, k1, k2, sign),
    ]
    total = sum(segment["d_pitch"] for segment in segments)
    measured = float(frames[k2]["pitch"]) - float(frames[start]["pitch"])
    parts = {
        "start": start, "k0": k0, "k1": k1, "k2": k2,
        "t_start": float(frames[start]["t"]), "t_k0": float(frames[k0]["t"]),
        "t_k1": float(frames[k1]["t"]), "t_k2": float(frames[k2]["t"]),
        "k0_rule": k0_rule, "k1_rule": k1_a_label if k1 == k1_a else k1_b_label,
        "k1_a": k1_a, "k1_a_label": k1_a_label, "k1_b": k1_b, "k1_b_label": k1_b_label,
        "sign": sign, "sign_error": sign_error, "sign_t": sign_t,
        "segments": segments, "consistency": _consistency(segments[1:]),
        "sum_residual": total - measured, "measured_total": measured,
    }
    return _render(path, document, frames, parts), parts


def main() -> int:
    parser = argparse.ArgumentParser(description="落地 pitch 预算分解（只读）")
    parser.add_argument("--result", type=Path, required=True, help="results/*.json.gz 路径")
    parser.add_argument("--dump-csv", type=Path, default=None, help="逐帧 CSV 输出路径（可选）")
    parser.add_argument("--report-out", type=Path, default=None, help="报告文本输出路径（可选）")
    parser.add_argument("--log-level", default="INFO", help="日志级别（DEBUG/INFO/WARNING/ERROR）")
    parser.add_argument("--log-file", type=Path, default=None, help="日志文件路径（缺省只输出到控制台）")
    args = parser.parse_args()

    setup_logging(default_level=args.log_level, log_file=args.log_file)
    try:
        report, parts = build_budget(args.result)
    except BudgetError as error:
        _LOGGER.error("预算分解失败：%s", error)
        return 2

    sys.stdout.write(report + "\n")
    if args.report_out is not None:
        args.report_out.write_text(report + "\n", encoding="utf-8")
        _LOGGER.info("报告已写出: %s", args.report_out)
    if args.dump_csv is not None:
        sign = int(parts["sign"])
        _write_csv(args.dump_csv, _load(args.result)[1], parts["start"], parts["k2"], sign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
