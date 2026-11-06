"""从 ``results/`` 的工况记录生成文档用的曲线图。

只做可视化，不参与控制；读数口径与 ``tools/jump_diagnostic.py`` 的记录字段一致：

- ``wheel_gap``        两轮离地间隙 [m]（负值表示压入地面）
- ``pitch``            机身俯仰角 [rad]
- ``leg_state``        ``[L_R, phi_R, L_L, phi_L]``，腿长 [m]
- ``normal_forces``    两轮法向力 [N]
- ``com_vxz``          质心速度 [m/s]
- ``yaw`` / ``yaw_rate`` 偏航角 [rad] / 偏航角速度 [rad/s]

用法：
    python tools/make_figures.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "docs" / "images"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

JUMP_FLIGHT = (1.429, 1.722)
JUMP_LANDING = (1.722, 1.903)


def load(run_id: str) -> dict:
    with gzip.open(RESULTS / f"{run_id}.json.gz", "rt", encoding="utf-8") as handle:
        return json.load(handle)


def arrays(payload: dict) -> dict:
    frames = payload["frames"]
    return {
        "t": np.array([f["t"] for f in frames]),
        "gap": np.array([max(f["wheel_gap"]) for f in frames]),
        "pitch": np.degrees([f["pitch"] for f in frames]),
        "leg": np.array([f["leg_state"] for f in frames]),
        "force": np.array([sum(f["normal_forces"]) for f in frames]),
        "vx": np.array([f["com_vxz"][0] for f in frames]),
        "yaw": np.degrees([f["yaw"] for f in frames]),
        "yaw_rate": np.array([f["yaw_rate"] for f in frames]),
    }


def save(fig, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"written {path}")


def jump_figure() -> None:
    data = arrays(load("T1_baseline_s12c_h0"))
    t = data["t"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 8.2), sharex=True)

    axes[0].plot(t, data["gap"] * 1000.0, color="#1f77b4")
    axes[0].axhline(200.0, color="#d62728", linestyle="--", linewidth=1.0, label="判据 200 mm")
    axes[0].set_ylabel("轮离地间隙 [mm]")
    axes[0].legend(loc="upper right")
    axes[0].annotate(f"峰值 {data['gap'].max()*1000:.0f} mm", xy=(1.55, data["gap"].max() * 1000),
                     xytext=(2.15, 170), arrowprops=dict(arrowstyle="->", color="0.35"))

    axes[1].plot(t, data["pitch"], color="#ff7f0e")
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].set_ylabel("俯仰角 [deg]")
    axes[1].annotate(f"全程 max|pitch| {np.abs(data['pitch']).max():.1f}°",
                     xy=(2.2, 8.0), fontsize=9)

    axes[2].plot(t, data["leg"][:, 0], label="右腿", color="#2ca02c")
    axes[2].plot(t, data["leg"][:, 2], label="左腿", color="#9467bd", linestyle="--")
    axes[2].set_ylabel("腿长 [m]")
    axes[2].legend(loc="upper right")

    axes[3].plot(t, data["force"], color="#8c564b")
    axes[3].set_ylabel("法向力合计 [N]")
    axes[3].set_xlabel("时间 [s]")

    for ax in axes:
        ax.axvspan(*JUMP_FLIGHT, color="#1f77b4", alpha=0.08)
        ax.axvspan(*JUMP_LANDING, color="#ff7f0e", alpha=0.08)
        ax.grid(alpha=0.25)
    axes[0].set_title("原地跳跃（s12c_h0：触地腿长 0.34 m，离地角动量目标 0）", fontsize=12)
    fig.align_ylabels(axes)
    save(fig, "jump_two_mass.png")


def forward_figure() -> None:
    data = arrays(load("T1_command_velocity_forward_ref"))
    t = data["t"]
    command = np.where(t < 3.0, 1.5, 0.0)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)

    axes[0].plot(t, command, color="0.55", linestyle="--", label="速度指令")
    axes[0].plot(t, data["vx"], color="#1f77b4", label="实测质心速度")
    axes[0].set_ylabel("速度 [m/s]")
    axes[0].legend(loc="upper right")
    axes[0].annotate(f"峰值 {data['vx'].max():.2f} m/s", xy=(1.8, data["vx"].max()),
                     xytext=(3.3, 1.25), arrowprops=dict(arrowstyle="->", color="0.35"))

    axes[1].plot(t, data["pitch"], color="#ff7f0e")
    axes[1].set_ylabel("俯仰角 [deg]")
    axes[1].set_ylim(-1.5, 1.5)

    axes[2].plot(t, data["force"], color="#8c564b")
    axes[2].set_ylabel("法向力合计 [N]")
    axes[2].set_xlabel("时间 [s]")

    for ax in axes:
        ax.axvline(3.0, color="0.7", linewidth=0.8, linestyle=":")
        ax.grid(alpha=0.25)
    axes[0].set_title("前行：0→1.5 m/s 加速、匀速、减速停车", fontsize=12)
    fig.align_ylabels(axes)
    save(fig, "forward_speed.png")


def spin_figure() -> None:
    data = arrays(load("T1_command_velocity_spin_ref"))
    t = data["t"]
    command = np.where(t < 4.2, 1.5, 0.0)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)

    axes[0].plot(t, command, color="0.55", linestyle="--", label="偏航角速度指令")
    axes[0].plot(t, data["yaw_rate"], color="#1f77b4", label="实测偏航角速度")
    axes[0].set_ylabel("偏航角速度 [rad/s]")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, data["yaw"], color="#2ca02c")
    axes[1].set_ylabel("偏航角 [deg]")
    axes[1].set_yticks([-180, -90, 0, 90, 180])

    axes[2].plot(t, data["vx"], color="#d62728")
    axes[2].set_ylabel("纵向速度 [m/s]")
    axes[2].set_ylim(-0.15, 0.15)
    axes[2].set_xlabel("时间 [s]")

    for ax in axes:
        ax.axvline(4.2, color="0.7", linewidth=0.8, linestyle=":")
        ax.grid(alpha=0.25)
    axes[0].set_title("原地转圈：1.5 rad/s 保持 4.2 s（约一整圈）", fontsize=12)
    fig.align_ylabels(axes)
    save(fig, "spin_yaw.png")


def forward_jump_figure() -> None:
    data = arrays(load("T1_command_velocity_forward_jump_ref"))
    t = data["t"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)

    axes[0].plot(t, data["gap"] * 1000.0, color="#1f77b4")
    axes[0].set_ylabel("轮离地间隙 [mm]")
    axes[0].annotate(f"峰值 {data['gap'].max()*1000:.0f} mm", xy=(2.05, data["gap"].max() * 1000),
                     xytext=(3.0, 150), arrowprops=dict(arrowstyle="->", color="0.35"))

    axes[1].plot(t, data["pitch"], color="#ff7f0e")
    axes[1].set_ylabel("俯仰角 [deg]")

    axes[2].plot(t, data["vx"], color="#2ca02c")
    axes[2].set_ylabel("纵向速度 [m/s]")
    axes[2].set_xlabel("时间 [s]")

    for ax in axes:
        ax.axvspan(1.961, 2.253, color="#1f77b4", alpha=0.08)
        ax.grid(alpha=0.25)
    axes[0].set_title("边前进边跳跃：1.0 m/s 指令，1.5 s 起跳", fontsize=12)
    fig.align_ylabels(axes)
    save(fig, "forward_jump.png")


def main() -> int:
    jump_figure()
    forward_figure()
    spin_figure()
    forward_jump_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
