# Circleg 轮圈腿机器人仿真

偏心曲柄式轮腿机器人的 MuJoCo 仿真项目。每条腿只有一个髋关节，通过 60 mm 曲柄把轮心挂在髋轴上；轮毂电机直驱轮圈。轮圈腿的机械模型、MJCF 与初始仿真框架来自 CirclegRobot 开源项目；本仓库在其基础上完成 8 状态 MPC + WBC 控制、跳跃与测试整理。

![原地跳跃](docs/images/jump_preview.gif)

> 原地跳跃：压缩、蹬伸、飞行、落地。完整视频见 [videos/serial_jump_50fps.mp4](videos/serial_jump_50fps.mp4)。

## 构型

```
机身 base_link
├── 右摆臂 joint1_Link   hinge joint1（髋）
│   └── 右轮圈 joint2_Link hinge joint2
└── 左摆臂 joint3_Link   hinge joint3（髋）
    └── 左轮圈 joint4_Link hinge joint4
```

- 四个关节全部是串联开链，没有闭链约束，`nq = 11`、`nv = 10`。
- 轮心在髋轴上方 60 mm；`q = 0` 时曲柄竖直向上，机身吊在轮心之下。
- 腿长定义 `L(q) = 0.080 − 0.060·cos q`，`q ∈ [0, π]` 对应机身高度约 0.193–0.313 m。
- 整机 9.5015 kg：机身 8.0119 kg，两条摆臂各 0.3785 kg，两个轮圈各 0.36628 kg。质心只在轮轴上方约 17 mm。

执行器：

| 通道 | 电机 | 力矩上限 |
|---|---|---|
| 纵向 / 俯仰平衡 | 两轮差模 | 5 N·m |
| 偏航 | 两轮共模 | 0.8 N·m |
| 高度 / 腿长 | 两髋同向 | 4→10 N·m（按行程排程） |

曲柄带来两个死点：`q = 0` 和 `q = π` 处所需髋力矩为零，`dq/dL` 趋于无穷。默认姿态放在上死点，机身最低、髋电机零负载；腿长指令必须限速，髋角要进 MPC 状态约束并做独立保护。

曲柄传动动画见 [docs/images/demo_mechanism.gif](docs/images/demo_mechanism.gif)。

## 控制栈

入口 `APP/mujoco_mpc_wbc.py`。上层 8 状态 MPC（`model_variant=hip_input`）以两轮纵向力和共模髋力矩为输入，下层 WBC 以 250 Hz 分配任务。默认 `tuned-fast-accel` preset、QP 后端。

```powershell
python -m APP.mujoco_mpc_wbc --duration 10 --terrain plane
python -m APP.mujoco_mpc_wbc --viewer
```

现场地模型 `URDF_Model/mjcf/URDF_Model_field.xml` 可用 `--terrain field` 加载。

## 当前实测

默认配置、headless 记录：

| 工况 | 条件 | 结果 |
|---|---|---|
| 前行 | 0→1.5 m/s，保持 3 s，随后停车 | 稳态 1.50 m/s，里程 4.33 m，max\|pitch\| 0.263 rad |
| 原地转圈 | 1.5 rad/s，保持 4.2 s | 峰值 1.519 rad/s，约一整圈，纵向漂移 <0.1 mm，max\|pitch\| 0.0002 rad |
| 原地跳跃 | 1.0 s 触发 | 底座高度 0.193→0.585 m，飞行约 0.53 s，max\|pitch\| 0.627 rad |
| 前进+跳跃 | 1.0 m/s，1.5 s 起跳 | 里程 4.61 m，max\|pitch\| 0.625 rad |

四个工况的 640×480 / 50 fps 视频：

- [前行](videos/serial_forward_50fps.mp4)
- [原地转圈](videos/serial_spin_50fps.mp4)
- [原地跳跃](videos/serial_jump_50fps.mp4)
- [前进+跳跃](videos/serial_forward_jump_50fps.mp4)

前行和跳跃的姿态指标明显弱于串腿项目。跳跃仍是早期的单质量/腿长方案，尚未改用双质量块规划。

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` 包含 MuJoCo、NumPy、SciPy、OSQP、Ruckig、pygame、pynput、matplotlib、imageio-ffmpeg 和 pytest。Linux 下的手柄输入需要图形会话；Wayland 需要启用 X11/XWayland 后端。

## 目录结构

```text
circleg/
├── Algorithm/               算法库：MPC、WBC、观测器、状态估计
├── APP/                     应用层：MPC/WBC 入口、MuJoCo 适配
├── URDF_Model/              MJCF 模型、网格、场地
├── tests/                   单元测试、回归测试、参数扫描
├── docs/
│   ├── reference/           设计文档与调试记录
│   └── images/              演示图
├── tools/                   曲柄动画、离线录像、工况指标
├── conftest.py
├── pytest.ini
└── requirements.txt
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/testing.md](docs/testing.md) | 四个标准工况的命令、指标与复现步骤 |
| [docs/reference/轮圈腿技术文档.md](docs/reference/轮圈腿技术文档.md) | 构型约束、控制栈分层、每层的物理依据、跳跃方案 |
| [docs/reference/构型与控制逻辑.md](docs/reference/构型与控制逻辑.md) | 拓扑、尺寸、质量、通道权限与限幅排程 |
| [docs/reference/控制器技术文档_MPC与WBC.md](docs/reference/控制器技术文档_MPC与WBC.md) | 8 状态 MPC、ZMP 参考、WBC 任务与三种求解后端 |
| [docs/reference/MPC-WBC方案.md](docs/reference/MPC-WBC方案.md) | 早期 MPC + 零空间 WBC 方案与构型硬限制 |
| [docs/reference/算法部署方案与代码地图.md](docs/reference/算法部署方案与代码地图.md) | 代码结构与模块职责 |

原始调试记录已归档，不随仓库发布。

## 测试

```powershell
python -m pytest -q
```

当前测试套件有 4 个既有失败，均为断言与代码演进不一致，不是运行链路崩溃：

| 测试 | 现象 |
|---|---|
| `test_control_regressions::test_multirate_loop_recomputes_wbc_five_times_per_mpc_tick` | 传入非法 `wbc_dt/physics_dt` 时不再抛 `ValueError` |
| `test_jerk_ramp::test_soft_reanchor_keeps_velocity_acceleration_and_command_flag` | `soft_reanchor` 不再把位置改到 5.0，测试仍按旧行为断言 |
| `test_velocity_joint::test_velocity_release_and_reverse[False/True]` | 收敛后误差单调性断言 `≤1e-9` 过严，实测有 `2.3e-5` 的微幅回升 |

修改这些断言等于改变评价口径，未自行处理。

## 已知限制

- **跳跃姿态**：`max|pitch|` 约 0.63 rad，落地后需要较长恢复；当前跳跃仍是早期方案，未按双质量块重做。
- **前行俯仰**：1.5 m/s 工况 `max|pitch|` 约 0.263 rad，姿态裕量小于串腿。
- **时间轴**：早期 viewer 调试的时间轴与物理时间不一致，本文指标以 headless 记录为准；viewer 里的时间不能直接当参考。
- **模型参数**：质量、惯量和几何来自 `URDF_Model/mjcf/*.xml` 和 CAD 导出；换轮或加配重后需要重测。
- **场地**：`field` 地形可用，但尚未做完整通过性验证。

## 开源来源

轮圈腿的机械模型、MJCF 与初始仿真框架来自 CirclegRobot 开源项目。原始框架的目录结构、模型文件和许可证声明予以保留；本仓库在其基础上完成 8 状态 MPC + WBC 控制、跳跃、测试与文档整理。

## 许可证与致谢

本项目采用 [PolyForm Noncommercial License 1.0.0](LICENSE)，仅限个人学习、学术研究与技术交流。

控制栈的早期版本与 GBC486 项目同源；两个项目现在各自维护 `Algorithm/` 副本。曲柄传动与离地检测的部分结论参考了《RM2026-串联腿步兵底盘控制系统设计与实现开源》（同济大学 SuperPower 战队）。
