"""轮圈腿曲柄传动演示动画。

把髋关节从上死点 q=0 摆到下死点 q=pi 再摆回来，动画展示：

1. 髋电机不是"抬腿"，而是一个**曲柄**：它让轮心绕髋轴画一个半径 60 mm 的圆。
2. 轮心被地面锁在 z=R=160 mm，所以曲柄转动的结果是**机身在轮圈里被顶起/放下**。
3. 曲柄的水平分量让两个轮子朝**相反**方向滚（两条腿是同一零件旋转 180° 安装）。
4. 重力所需的髋力矩正比于水平力臂 0.06*sin(q)，在两个死点为零、行程中点最大。

运行::

    python demo_mechanism.py                 # 弹窗动画
    python demo_mechanism.py --save demo.gif # 存成 gif
    python demo_mechanism.py --period 6      # 一个来回 6 秒

对应文档：docs/构型与控制逻辑.md §2（传动本质）与 §5（髋部增益排程）。
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

# ---- 全部来自 URDF_Model/mjcf/URDF_Model_plane.xml ------------------------
CRANK = 0.060          # joint2 相对 joint1_Link 的 z 偏置 = 曲柄长
WHEEL_R = 0.160        # right/left_wheel_collision 圆柱半径
HIP_X = -0.019944      # 髋轴在机身坐标系里的 x
HIP_Z = -0.093022      # 髋轴在机身坐标系里的 z
WHEEL_Y_R, WHEEL_Y_L = -0.12, 0.12   # 两轮所在平面（轮距 0.24）
RIGHT_SIGN, LEFT_SIGN = -1.0, +1.0   # joint1 轴 (0,-1,0) / joint3 轴 (0,+1,0)

# ---- 来自 APP/config.py --------------------------------------------------
BASE_LENGTH, LINK_LENGTH = 0.080, 0.060      # LegConfig
KP_MIN, KP_MAX, KP_BUMP = 2.0, 10.0, 40.0    # HipConfig

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
# NSimSun / MS Gothic 是等宽且带中文字形的，信息框要靠它对齐数字列
plt.rcParams["font.monospace"] = ["SimSun", "SimHei", "DejaVu Sans Mono"]
plt.rcParams["axes.unicode_minus"] = False

def hub_offset(q, sign):
    """轮心相对髋轴的 (dx, dz)。这两行就是整个传动的全部数学。"""
    return sign * CRANK * np.sin(q), CRANK * np.cos(q)


def leg_length(q):
    """代码里的腿长定义 L(q)（Algorithm/geometry.py）。"""
    return BASE_LENGTH - LINK_LENGTH * np.cos(q)


def progress(q):
    """AdaptiveHipPD 的归一化行程，恒等于 (1-cos q)/2。"""
    return (1.0 - np.cos(q)) / 2.0


def hip_kp(q):
    """髋 PD 的比例增益排程；bump 项 4p(1-p) 恒等于 sin^2(q)。"""
    p = progress(q)
    return KP_MIN + (KP_MAX - KP_MIN) * p + KP_BUMP * 4.0 * p * (1.0 - p)


def body_origin_z(q):
    """轮心贴地时 base_link 原点的高度。"""
    return WHEEL_R - CRANK * np.cos(q) - HIP_Z


def sweep(t, period):
    """0 -> pi -> 0 平滑往复，两端速度为零，正好停在两个死点上。"""
    return 0.5 * np.pi * (1.0 - np.cos(2.0 * np.pi * t / period))


BODY_W, BODY_H = 0.17, 0.15          # 机身外形（示意，不是真实网格）
GREY, BLUE, RED, GREEN = "#5b6570", "#2f6fb3", "#c8402f", "#2e8b57"


def build_side(ax):
    """侧视图（x-z）：曲柄 + 轮圈 + 机身。返回需要每帧更新的 artist。"""
    ax.set_title("侧视图：髋电机是曲柄，把机身在轮圈里顶起来", fontsize=11)
    ax.set_xlim(-0.30, 0.30)
    ax.set_ylim(-0.04, 0.42)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("z / m")
    ax.axhspan(-0.04, 0.0, color="#d9d9d9", zorder=0)
    ax.axhline(0.0, color="k", lw=1.2, zorder=1)

    art = {}
    # 轮心画的那个圆（半径 = 曲柄长），跟着髋轴一起上下平移
    art["trace"] = Circle((0, 0), CRANK, fill=False, ls="--", lw=1.0,
                          ec=GREEN, zorder=3)
    ax.add_patch(art["trace"])
    # 左右轮圈：左轮画淡一点，用来对比两侧相反的滚动方向
    art["wheel_l"] = Circle((0, WHEEL_R), WHEEL_R, fill=False, lw=1.6,
                            ec=GREY, alpha=0.45, zorder=2)
    art["wheel_r"] = Circle((0, WHEEL_R), WHEEL_R, fill=False, lw=2.6,
                            ec=GREY, zorder=4)
    ax.add_patch(art["wheel_l"])
    ax.add_patch(art["wheel_r"])
    art["spokes"] = [ax.plot([], [], color=GREY, lw=1.0, zorder=4)[0]
                     for _ in range(2)]
    art["body"] = Rectangle((0, 0), BODY_W, BODY_H, fc="#9fd6dc", ec=BLUE,
                            lw=1.8, alpha=0.85, zorder=5)
    ax.add_patch(art["body"])
    art["crank_l"], = ax.plot([], [], color=RED, lw=2.0, alpha=0.4, zorder=6)
    art["crank_r"], = ax.plot([], [], color=RED, lw=3.4, zorder=7,
                              solid_capstyle="round")
    art["hip"], = ax.plot([], [], "o", color="k", ms=7, zorder=8)
    art["hub"], = ax.plot([], [], "o", color=RED, ms=6, zorder=8)
    # 力臂画在轴线下方一点，免得 q=pi/2 时和水平的曲柄完全重叠
    art["lever"], = ax.plot([], [], color=GREEN, lw=2.2, marker="|", ms=9,
                            zorder=8)
    art["tie"], = ax.plot([], [], color=GREEN, lw=0.8, ls=":", zorder=8)
    art["start"] = ax.axvline(HIP_X, color=GREY, ls=":", lw=1.0, zorder=1,
                              label="髋轴 x（机身不动）")
    art["info"] = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                          ha="left", fontsize=10, family="monospace",
                          bbox=dict(fc="w", ec=GREY, alpha=0.9, pad=0.5))
    ax.legend([art["crank_r"], art["crank_l"], art["trace"], art["lever"],
               art["start"]],
              ["右腿曲柄（髋轴→轮心，60 mm）", "左腿曲柄（方向相反）",
               "轮心轨迹圆", "重力力臂 0.06·sin q", "髋轴 x（机身不平移）"],
              loc="lower right", fontsize=8.5, framealpha=0.92)
    return art


def build_top(ax):
    """俯视图（x-y）：两条腿是同一个零件转 180° 装的，所以前后错开。"""
    ax.set_title("俯视图：两侧轮心前后错开（C2 装配的直接后果）", fontsize=11)
    ax.set_xlim(-0.26, 0.26)
    ax.set_ylim(-0.21, 0.21)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.add_patch(Rectangle((-0.085, -0.10), BODY_W, 0.20, fc="#9fd6dc",
                           ec=BLUE, lw=1.5, alpha=0.7, zorder=2))
    art = {}
    for side, y in (("r", WHEEL_Y_R), ("l", WHEEL_Y_L)):
        art[f"tire_{side}"] = Rectangle((0, y - 0.012), 2 * WHEEL_R, 0.024,
                                       fc=GREY, alpha=0.75, zorder=3)
        ax.add_patch(art[f"tire_{side}"])
        art[f"hub_{side}"], = ax.plot([], [], "o", color=RED, ms=6, zorder=4)
    art["gap"], = ax.plot([], [], color=GREEN, lw=2.0, marker="|", ms=10,
                          zorder=5)
    art["gap_txt"] = ax.text(0.0, 0.0, "", ha="center", va="bottom",
                             fontsize=9, color=GREEN, zorder=6)
    return art


def build_curves(ax, period):
    """右下角：机身高度、力臂、髋部 kp 随 q 的变化。"""
    q = np.linspace(0.0, np.pi, 400)
    ax.set_title("同一个 q 决定三件事", fontsize=11)
    ax.set_xlabel("髋关节角 q / rad")
    ax.set_ylabel("mm")
    ax.set_xlim(0.0, np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels(["0\n上死点·机身最低", "π/2\n行程中点", "π\n下死点·机身最高"])
    ax.plot(q, body_origin_z(q) * 1e3, color=BLUE, lw=1.8, label="机身高度")
    ax.plot(q, CRANK * np.sin(q) * 1e3, color=GREEN, lw=1.8, label="重力力臂")
    ax.set_ylim(0, 340)
    ax2 = ax.twinx()
    ax2.plot(q, hip_kp(q), color=RED, lw=1.6, ls="--", label="髋部 kp")
    ax2.set_ylabel("kp", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.set_ylim(0, 55)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, framealpha=0.9)

    art = {"mark": ax.axvline(0.0, color="k", lw=1.0),
           "dot_h": ax.plot([], [], "o", color=BLUE, ms=6)[0],
           "dot_a": ax.plot([], [], "o", color=GREEN, ms=6)[0],
           "dot_k": ax2.plot([], [], "o", color=RED, ms=6)[0]}
    return art


def build_figure(period):
    fig = plt.figure(figsize=(13.0, 7.4))
    fig.suptitle("轮圈腿传动：髋电机就是一根曲柄，转动它等于把整台车举起来",
                 fontsize=14)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], hspace=0.42,
                          wspace=0.26, left=0.06, right=0.93, top=0.91,
                          bottom=0.10)
    art = {"side": build_side(fig.add_subplot(gs[:, 0])),
           "top": build_top(fig.add_subplot(gs[0, 1])),
           "cur": build_curves(fig.add_subplot(gs[1, 1]), period)}
    return fig, art


def make_update(art, period):
    side, top, cur = art["side"], art["top"], art["cur"]

    def update(frame_t):
        q = sweep(frame_t, period)
        bz = body_origin_z(q)              # base_link 原点高度
        hip_z = bz + HIP_Z                 # 髋轴高度 = 0.16 - 0.06*cos q
        dx_r, _ = hub_offset(q, RIGHT_SIGN)
        dx_l, _ = hub_offset(q, LEFT_SIGN)
        hub_r, hub_l = HIP_X + dx_r, HIP_X + dx_l

        # --- 侧视图 --------------------------------------------------------
        side["trace"].center = (HIP_X, hip_z)
        side["wheel_r"].center = (hub_r, WHEEL_R)
        side["wheel_l"].center = (hub_l, WHEEL_R)
        # 轮心水平走了 dx，机身不动，所以轮子必须滚过 dx/R 的角度
        theta = -dx_r / WHEEL_R
        for k, spoke in enumerate(side["spokes"]):
            a = theta + k * np.pi / 2.0
            ux, uz = WHEEL_R * np.cos(a), WHEEL_R * np.sin(a)
            spoke.set_data([hub_r - ux, hub_r + ux], [WHEEL_R - uz, WHEEL_R + uz])
        side["body"].set_xy((-BODY_W / 2.0, bz - 0.0133 - BODY_H / 2.0))
        side["crank_r"].set_data([HIP_X, hub_r], [hip_z, WHEEL_R])
        side["crank_l"].set_data([HIP_X, hub_l], [hip_z, WHEEL_R])
        side["hip"].set_data([HIP_X], [hip_z])
        side["hub"].set_data([hub_r, hub_l], [WHEEL_R, WHEEL_R])
        z_lever = WHEEL_R - 0.045
        side["lever"].set_data([HIP_X, hub_r], [z_lever, z_lever])
        side["tie"].set_data([hub_r, hub_r], [z_lever, WHEEL_R])
        side["info"].set_text(
            f"髋角 q：{np.degrees(q):.1f}°\n"
            f"机身高度：{bz * 1e3:.1f} mm\n"
            f"代码腿长 L(q)：{leg_length(q) * 1e3:.1f} mm\n"
            f"重力力臂：{CRANK * np.sin(q) * 1e3:.1f} mm\n"
            f"左右轮心错开：{abs(dx_l - dx_r) * 1e3:.1f} mm\n"
            f"髋部 kp：{hip_kp(q):.2f}")

        # --- 俯视图 --------------------------------------------------------
        top["tire_r"].set_xy((hub_r - WHEEL_R, WHEEL_Y_R - 0.012))
        top["tire_l"].set_xy((hub_l - WHEEL_R, WHEEL_Y_L - 0.012))
        top["hub_r"].set_data([hub_r], [WHEEL_Y_R])
        top["hub_l"].set_data([hub_l], [WHEEL_Y_L])
        top["gap"].set_data([hub_r, hub_l], [0.0, 0.0])
        top["gap_txt"].set_position(((hub_r + hub_l) / 2.0, 0.16))
        top["gap_txt"].set_text(f"{abs(dx_l - dx_r) * 1e3:.0f} mm")

        # --- 曲线 ----------------------------------------------------------
        cur["mark"].set_xdata([q, q])
        cur["dot_h"].set_data([q], [bz * 1e3])
        cur["dot_a"].set_data([q], [CRANK * np.sin(q) * 1e3])
        cur["dot_k"].set_data([q], [hip_kp(q)])
        return []

    return update


def main() -> None:
    p = argparse.ArgumentParser(description="轮圈腿曲柄传动演示动画")
    p.add_argument("--period", type=float, default=6.0, help="一个来回的秒数")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--save", metavar="FILE", help="存成 gif/mp4 而不是弹窗")
    args = p.parse_args()

    period = max(float(args.period), 0.5)
    fps = max(int(args.fps), 5)
    fig, art = build_figure(period)
    frames = np.linspace(0.0, period, int(period * fps), endpoint=False)
    anim = FuncAnimation(fig, make_update(art, period), frames=frames,
                         interval=1000.0 / fps, blit=False)
    if args.save:
        anim.save(args.save, writer=PillowWriter(fps=fps))
        print(f"已保存到 {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()




