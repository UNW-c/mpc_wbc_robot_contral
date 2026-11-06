# 系统架构与代码地图

本文说明 GBC486 的机械构型、状态定义、软件分层和单步数据流。控制律的推导见 [mpc_wbc.md](mpc_wbc.md)，跳跃见 [jump.md](jump.md)。

## 1. 机械构型

GBC486 是双轮腿机器人，每条腿是一套闭链五连杆：

```
base（机身，free joint）
├── AG 大腿      jAG 主动
│   └── GH 小腿  jGH 被动
│       └── wheel_right  jwheel 主动
└── AB 曲柄      jAB 主动
    └── BE 连杆  jBE 被动
        └── EC 三角连杆 jEC 被动
            └── CF 连杆 jCF 被动

闭链：EC 与 AG 在 D 点铰接；CF 与 GH 在 F 点铰接。
ABED 与 DCFG 都是平行四边形，所以小腿方向由曲柄 AB 刚性决定。
```

两条腿镜像，左腿对应 `IJ / JM / MK / KN / IO / OP`。MuJoCo 只支持树结构，闭链用 8 条 `connect` 约束表达（每条腿 2 个点，每个点约束机身 x、z 两个方向）。

主要几何量（米）：

| 杆 | 长度 | 杆 | 长度 |
|---|---|---|---|
| AB / IJ 曲柄 | 0.0980 | AG / IO 大腿 | 0.2150 |
| BE / JM | 0.1176 | GH / OP 小腿到轮心 | 0.2580 |
| ED / MK | 0.1172 | DC / MK 三角边 | 0.0664 |
| DA / IO | 0.0982 | CF / KN | 0.1168 |
| GD | 0.1168 | FG | 0.0663 |

质量（kg）：

| 部件 | 单个 | 数量 | 合计 |
|---|---|---|---|
| base | 8.400 | 1 | 8.400 |
| 大腿 AG / IO | 0.667 | 2 | 1.334 |
| 小腿 GH / OP | 1.640 | 2 | 3.280 |
| 曲柄 AB / IJ | 0.202 | 2 | 0.404 |
| 连杆 BE / CF | 0.057 | 4 | 0.228 |
| 三角连杆 EC / MK | 0.128 | 2 | 0.256 |
| 轮 | 0.322 | 2 | 0.644 |
| 整机 | | | **14.546** |

跳跃用的两块划分：

- 下块 `m1 = 3.924 kg`：两轮 + 小腿 GH/OP（含轮毂电机）。
- 上块 `m2 = 10.622 kg`：其余全部。
- 质量比 `k = m1/m2 = 0.3694`，等效质量 `μ = m1·m2/(m1+m2) = 2.865 kg`。

执行器与传感器：

| ctrl | 名称 | 关节 | 限幅 |
|---|---|---|---|
| 0 | Right_front | jAB | ±20 N·m |
| 1 | Right_rear | jAG | ±20 N·m |
| 2 | Left_front | jIJ | ±20 N·m |
| 3 | Left_rear | jIO | ±20 N·m |
| 4 | Right_Wheel | jwheel_right | ±4 N·m |
| 5 | Left_Wheel | jwheel_left | ±4 N·m（gain = −1，抵消镜像轴） |

传感器：机身四元数与角速度、四个髋关节角、两个轮转角。轮速由轮转角差分得到。

腿长工作范围约 0.14–0.43 m，标称站立腿长 0.28 m。仿真步长 1 ms，积分器 `implicitfast`。

## 2. 状态与坐标约定

### 2.1 MPC 状态

8 状态模型：

```
x = [s, v, θ, θ̇, ψ, ψ̇, δφ_c, δφ̇_c]
u = [f_R, f_L, τ_c]
```

| 符号 | 含义 | 单位 |
|---|---|---|
| s | 机身沿世界 x 的前进位移 | m |
| v | 前进速度 | m/s |
| θ | 俯仰角 | rad |
| θ̇ | 俯仰角速度 | rad/s |
| ψ | 偏航角 | rad |
| ψ̇ | 偏航角速度 | rad/s |
| δφ_c | 共模腿角相对标称值的偏差 | rad |
| δφ̇_c | 共模腿角速度偏差 | rad/s |
| f_R / f_L | 右 / 左轮纵向地面力 | N |
| τ_c | 共模髋力矩 | N·m |

共模腿角定义：

```
φ_c = (φ_R + φ_L) / 2
δφ_c = −(φ_polar − φ0)
```

负号来自代码的位形约定：`model_builder.py::set_legs` 用 `L = L0、φ_polar = φ0 − phi_c` 设置位形，速度映射必须与位形同号，否则同一个状态会出现自相矛盾。差模腿角 `φ_d` 不作为 MPC 状态，由 WBC 的左右腿任务各自约束。

### 2.2 腿空间量

`robot.py` 用平面五连杆正运动学把关节角换成四个量：

```
[L_R, φ_R, L_L, φ_L]
```

`L` 是髋轴到轮心的距离，`φ` 是这条连线相对机身竖直方向的角度。它们不是规划输入，而是 WBC 的任务空间；跳跃规划里同样是导出量。

### 2.3 接触与相位

- 接触估计：`Algorithm/contact_state.py`，用轮地几何间隙、法向力、连续帧确认组合判断。
- 跳跃相位：`gbc_mpc_wbc/jump.py`，`IDLE → COMPRESS → COMPRESS_BRAKE → THRUST → FLIGHT → LANDING → IDLE`。
- 相位切换由计划时钟或几何接触触发，物理量用于核验。`phase_transition_assertion_failures` 会记录核验失败的次数。

## 3. 软件分层

```
tools/                入口与工具（view / jump_diagnostic / video / make_figures / 扫描）
    │
simulation.py         装配：配置、模型、MPC、WBC、相位机、run()
    │
loop.py               时序：1 kHz 推进、4 ms WBC、20 ms MPC、记录
    │
control.py            控制段：MPC 段、WBC 段、跳跃力矩、飞行力位混合
    │
wbc.py                任务集与权重（要满足什么）
wbc_limited.py        限幅自洽 WLS（怎么解）
    │
dynamics.py           选择矩阵、接触行、闭链行、腿空间行与偏置
robot.py              模型加载、正运动学、站立位形、接触点
    │
MuJoCo + Algorithm/   物理引擎与通用算法库
```

依赖方向是单向的：`tools → simulation → loop → control → wbc → dynamics → robot`。`Algorithm/` 不反向依赖 `gbc_mpc_wbc/`。

### 3.1 各模块职责

| 文件 | 职责 |
|---|---|
| `robot.py` | 加载 `MJCF/env.xml`，读传感器，算 `L/φ`、站立位形、轮接触点、质心与雅可比 |
| `dynamics.py` | 组装 22 行：4 条任务行、6 条轮接触行、8 条闭链行、4 条腿空间行；提供加速度偏置 |
| `model_builder.py` | 在当前位形上用 KKT 求约束加速度，中心差分得到 6 / 8 状态线性模型 |
| `wbc.py` | 构造任务、参考、权重和激活因子；选择求解器后端 |
| `wbc_limited.py` | 接触消去 + 动力学硬约束 + 加权任务；力矩越限时按一维区间缩放，保证 `τ` 与 `q̈` 一致 |
| `control.py` | 把 MPC 解和跳跃计划翻译成 WBC 任务或关节力矩 |
| `loop.py` | 分频调度、相位切换、记录、viewer 同步 |
| `simulation.py` | 配置默认值、装配对象、`run()` 委托、汇总 |
| `two_mass.py` | 两块质量模型、静平衡流形表、起跳/飞行/落地计划 |
| `jump.py` | 相位机、起跳可行性、压缩与蹬伸的参考量 |
| `presets.py` | MPC 权重与限值档位 |

## 4. 单步数据流

每个 1 ms 物理步：

1. `loop.py` 推进 MuJoCo，读 `qpos/qvel` 与传感器。
2. 每 4 ms（250 Hz）：
   - 估计接触与相位；
   - 若在跳跃相位，取 `two_mass` 计划；
   - 组装任务行与硬约束；
   - 调 WBC 求 `ctrl`。
3. 每 20 ms（50 Hz）：
   - 取当前状态，调 MPC；
   - 得到未来 25 步的参考轨迹与轮地力；
   - 更新 WBC 的参考。
4. 力矩限幅、越界保护、写入 `data.ctrl`。
5. 记录帧（默认每 4 个物理步一条）。

WBC 频率是 MPC 的 5 倍，MPC 的参考在两次求解之间由 WBC 采样保持。`preview_lead` 控制 WBC 从轨迹的哪一时刻取参考，默认 0。

## 5. 两套模型

| 模型 | 状态 | 输入 | 用途 |
|---|---|---|---|
| `held_leg`（6 状态） | `[s, v, θ, θ̇, ψ, ψ̇]` | `[f_R, f_L]` | 早期对照；腿角固定为标称值 |
| `leg_input`（8 状态） | 6 状态 + `[δφ_c, δφ̇_c]` | 6 状态输入 + `τ_c` | 默认；允许 MPC 用摆腿辅助加减速 |

6 状态模型保留在代码里用于回归对照和教学，默认入口使用 8 状态。切换方式：

```powershell
python tools/view.py --model-variant held_leg
```

两套模型的 WBC 任务也不同：6 状态只有“腿长回标称 + 腿角保持”，8 状态多了“腿角跟踪 MPC 的共模腿角参考”。

## 6. 关键常量

| 常量 | 值 | 位置 |
|---|---|---|
| 物理步长 | 1 ms | `MJCF/env.xml` |
| WBC 周期 | 4 ms | `Algorithm/timing.py` |
| MPC 周期 | 20 ms | `simulation.py::MPC_DT` |
| MPC 视界 | 25 步（0.5 s） | `simulation.py::MPC_HORIZON` |
| 最大前进速度 | 1.5 m/s | `simulation.py::MAX_SPEED` |
| MPC 速度状态带 | 3.0 m/s | `simulation.py::SPEED_LIMIT` |
| 标称腿长 | 0.28 m | `SimulationConfig.nominal_length` |
| 轮力矩限 | 4.0 N·m | MJCF `actuator ctrlrange` |
| 髋力矩限 | 20.0 N·m | MJCF `actuator ctrlrange` |

这些值都是模型或部署常量，不是调参入口。改它们会改变控制行为，需要重新做回归和判据。

## 7. 扩展点

- **换机器人**：替换 `MJCF/`，改 `robot.py` 的关节名、执行器名和腿空间定义，重跑 `tools/make_capacity.py` 生成能力表。
- **换 WBC 后端**：`wbc.py::resolver_factory` 已支持 `nullspace / wls / qp`。默认 `wls`，`nullspace` 用于复现早期的任务饿死现象。
- **换 MPC 模型**：`model_builder.py` 里两个 builder 同接口，新增 builder 后改 `simulation.py` 的选择分支。
- **新增工况**：在 `tools/jump_diagnostic.py` 加 case，或在 `tools/video.py` 用 `--command-velocity / --command-yaw-rate` 组合。

## 8. 目录与运行边界

`gbc486/` 是自包含目录：`Algorithm/` 在项目内，`_paths.py` 只把本项目根加入 `sys.path`，不读取上层工作区。所有入口都从 `gbc486/` 目录运行：

```powershell
cd gbc486
python tools/view.py
```

如果在仓库根目录用 `python -m gbc486.tools.view` 运行，Python 的模块搜索路径会与脚本方式不同，不保证导入到同一份 `Algorithm`。文档中的命令一律以 `gbc486/` 为工作目录。
