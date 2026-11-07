# wheel_leg_mujoco（GBC486）MPC + WBC 部署方案

日期：2026-10-01
状态：方案文档，未实现，未修改任何代码、模型和参数。

## 0. 决策记录（2026-10-01，用户确认）

| # | 事项 | 决定 |
|---|---|---|
| 1 | 修改 `MJCF/robot.xml`（圆柱轮碰撞体、腿内 `<exclude>`、站立位形） | 同意 |
| 2 | 代码放置 | 方案 A：复用上层 `Algorithm/` |
| 3 | 标称腿长 L0、`φ_c` 范围 | **L0 = 0.42 m**（2026-10-01）→ **2026-10-03 改为 0.28 m**（用户提出站高太高）。M0 实测机构伸展上限 0.430 m、缩腿下限 0.14 m，取行程带中点；实测见第 3.3 节 |
| 4 | 先做 6 状态 held-leg 模型，8 状态留到 M5 | 同意 |
| 5 | 运行环境 | circleg `.venv`（mujoco 3.13.0、osqp 1.1.3） |
| 6 | WBC 后端 | **先用零空间法（`NullSpaceHierarchy` + `SaturationResolver`）或 WLS（`WeightedSaturationResolver`）**；效果不好再切 QP（第 4.4 节） |
| 7 | 文件边界 | 本项目的文档、代码、结果**只写在 `wheel_leg_mujoco/` 下**；上层 circleg 的 `docs/`、`Algorithm/`、`APP/`、`tests/`、`artifacts/` 只读。运行时设 `PYTHONDONTWRITEBYTECODE=1`，避免在上层生成 `__pycache__` |

## 0.1 进度

| 阶段 | 状态 | 产物 |
|---|---|---|
| M0 模型修改 | 已完成（2026-10-01） | `MJCF/robot.xml`（备份 `MJCF/robot.xml.bak`） |
| M0 L0 扫描 | 已完成；用户选 L0 = 0.42 m | `tools/m0_l0_sweep.py`、`results/m0_l0_sweep.md/.json` |
| M1 适配层 + 核验 | 已完成（2026-10-01），见第 5.1 节 | `gbc_mpc_wbc/dynamics.py`、`gbc_mpc_wbc/_circleg.py`、`tools/m1_check.py`、`results/m1_dynamics_check.md/.json`；`robot.xml` 闭链 site 修正 |
| M2 WBC 单独运行 | 已完成（2026-10-01），验收通过，见第 5.2 节 | `gbc_mpc_wbc/wbc.py`、`tools/m2_standalone_wbc.py`、`results/m2_standalone_wbc.md/.json`、`results/m2_<backend>.json.gz` |
| M3 6 状态 MPC 接入 | 已完成（2026-10-01），验收通过，见第 5.3 节 | `gbc_mpc_wbc/model_builder.py`、`gbc_mpc_wbc/simulation.py`、`tools/m3_mpc_wbc.py`、`results/m3_mpc_wbc.md/.json`、`results/m3_<backend>_<case>.json.gz` |
| M4 指令跟踪 | **暂缓**（用户 2026-10-03：先只要求零输入下能正常平衡、姿态不发散，有速度输入的模式暂不测试）。现有的 `results/m4_*` 是重构前的旧代码跑出来的，**已过期** | — |
| M5 8 状态模型 | 零输入工况已对比完成（2026-10-03），速度工况随 M4 一并暂缓 | `gbc_mpc_wbc/model_builder.py::LegInputModelBuilder`、`tools/m5_model_compare.py`、`results/m5_*` |
| M5 起 | 未开始 | — |

参考来源：上层工作区 circleg 轮圈腿的现行方案

- `docs/算法部署方案与代码地图.md`
- `docs/MPC-WBC方案.md`
- `APP/mujoco_mpc_wbc.py`
- `APP/mujoco_balance_model.py`
- `Algorithm/mpc/*`
- `Algorithm/wbc/task_qp.py`
- `Algorithm/supervision.py`

本机模型现状见 `PROJECT_ANALYSIS.md`。

> 2026-10-03 范围收敛（用户指定）：**当前只要求零输入下能正常平衡、姿态不发散**。
> 有速度 / 偏航输入的模式（M4）暂不测试；不纠结离线验证，部署后直接接仿真测试修正。
> 控制方式直接照搬 circleg，只改建模，见第 5.5 节。

本文所有"实测"数据，都是在内存里加载 `MJCF/env.xml` 得到的（circleg `.venv`，mujoco 3.13.0，osqp 1.1.3）。circleg 本身的回归测试我没有运行，它"基本功能正常"这一点以用户说明和 circleg 文档为准。

遵守 `AGENTS.md`：文中所有增益和权重都是从 circleg 搬过来的初值，只用于可行性验证。任何调参都要等用户明确要求。

---

## 1. 目标与总体思路

目标：把 circleg 已经跑通的 **"50 Hz 降阶线性 MPC → 250 Hz 单级 QP WBC → 1 kHz MuJoCo"** 架构部署到 GBC486 双轮腿上。WBC 后端先用零空间法 / WLS，QP 作为后备（决策 6）。原则是尽量复用 circleg 中与机器人无关的算法层，只重写和机器人相关的适配层。

```
MotionCommand(velocity, yaw_rate) ──► 监督层 ControlSupervisor / SupervisedControl   （复用）
        │ 50 Hz                                   │ 250 Hz
        ▼                                         ▼
WheelMPC + VelocityPlanner（复用）        SaturationResolver（零空间）/ WLS（复用；QP 后备）
  模型：GBC486 实测线性化（新写 Builder）    任务：GBC486 任务集（新写 tasks()）
        │ OptimizedTrajectory                     │ τ ∈ R⁶
        └──────────────► sample_at(t) ────────────┘
                                                  ▼
                               data.ctrl = [jAB, jAG, jIJ, jIO, 右轮, 左轮]
```

## 2. 两台机器人的关键差异（决定哪些能直接搬）

| 项 | circleg 轮圈腿 | GBC486（实测/读码） | 影响 |
|---|---|---|---|
| nq / nv / nu | 11 / 10 / 4 | 21 / 20 / 6 | 所有矩阵维数变化；算法层按 `nv`、`nu` 泛化，不受影响 |
| 腿机构 | 每侧 1 个偏心曲柄，无闭链 | 每侧 2 电机 + 双平行四边形，4 个 `connect` 闭链 | **WBC 必须加入闭链约束**（第 4.2 节） |
| 闭链约束 | 无 | 12 行 equality；平面内 8 行，秩 8，cond 24；y 向 4 行奇异值 1e-4 量级（近冗余） | 只取平面内 8 行，否则等式约束秩亏 |
| 独立自由度 | 浮动基 6 + 4 | 浮动基 6 + 腿 4 + 轮 2 = 12 | — |
| 控制权限 | 4 维，需求 5 维，`v_x` 被牺牲（circleg §0） | 6 维：两侧腿长 L、腿角 φ 都能直接控制，两轮各一个 | 高度、横滚有直接执行器，circleg 的"死点双重奇异"和"单支撑横滚失效"的结论**不适用** |
| 轮半径 | 0.16（圆柱碰撞体） | ≈0.077（**网格**碰撞，AABB 推得） | 接触点公式和力矩换算都要改；见第 3 节 |
| 半轮距 | 0.12 | 0.25 | 偏航力臂约为 2 倍 |
| 整机质量 | 9.50 kg | 14.546 kg | 标称单轮法向力 71.3 N |
| 轮力矩限 | ±5 N·m | ±4 N·m | 单轮最大纵向力 4/0.077 ≈ 51.9 N |
| 腿/髋力矩限 | ±10 N·m | ±20 N·m | — |
| 关节名 | joint1–4、`j*_ctrl` | jAB/jAG/jIJ/jIO/jwheel_*，`*_act` | 适配层全部改名 |
| 轴向 | 髋 joint1/joint3 轴相反 | 右腿 jAB/jAG 为 −y，左腿 jIJ/jIO 为 +y；右轮 +y，左轮 −y（世界系） | 共模/差模符号要重新推导，并用断言守住 |

结论：GBC486 的执行权限比 circleg 宽裕。circleg 文档中很多绕开构型缺陷的设计（死点排程、scrub 与腿行程强耦合、`v_x` 让位）在这里**不需要照搬**。但多了闭链约束和网格轮接触两个新问题。

## 3. 前置：模型准备（M0，已获批准修改 `robot.xml`）

实测发现 3 个问题。它们会直接让 circleg 的适配层失效：

1. **轮子是网格碰撞**（`geom type=7`）。落地后每个轮子有 4 个接触点。circleg 的 `MuJoCoDynamics.kinematics` 按"圆柱 geom 的 z 轴 = 轮轴、`geom_size[0]` = 半径"来计算接触点，而 GBC486 轮子网格 geom 的 z 轴并不是轮轴（实测 `[0.97, 0, 0.23]`），`rbound` 是 0.1096，不等于轮半径。
   - 方案：给每个轮加一个圆柱碰撞体（半径以 M0 测量为准，暂按 0.077），把网格改成只显示，与 circleg 的 `right_wheel_collision`/`left_wheel_collision` 做法一致。
2. **腿内部自碰撞**。实测零力矩塌落后出现 GH–BE、GH–EC、GH–CF、JM–OP、MK–OP、KN–OP 等自碰撞。初始姿态下 GH–CF、KN–OP 也已经有轻微穿透（−0.48 mm）。circleg 的 `unexpected_body_contact` 保护会因此误停。
   - 方案：用 `<exclude>` 屏蔽同一条腿内部的连杆对。机身和腿对地面的接触仍保留，作为保护信号。
3. **没有站立初始位形**。现在 base 从 z = 0.2 m 处释放，腿在重力下塌落。
   - 方案：加一个 `keyframe`，或者写一个初始化函数（参考 circleg `MuJoCoRobot.initialize_pose` / `_place_wheels_on_ground`），把腿放到标称 L0、φ0，并让轮子刚好落地。**标称 L0 需要用户指定**，可行范围约 0.14–0.43 m（悬空实测）。

可选项：

- 闭链 site 的初始装配误差 4.5 mm，可以修正 site 坐标。
- 修掉 `PROJECT_ANALYSIS.md` 第 6 节的 `d_x` 单位错误和 `set_joint_positions` 索引 bug。新链路不依赖这两处，修不修都行。

### 3.1 L0 扫描测试（为决策 3 提供数据，不是调参）

在 M0 模型修改完成后，对候选标称腿长逐点做静态核验。腿角取 `φ` 使轮心位于髋正下方，两腿对称。

- 候选：L0 = 0.16 ~ 0.42 m，步长 0.02 m（端点以关节 range 可达为准）。
- 每个点的输出（全部是测量值，不涉及控制参数）：

| 指标 | 计算方法 | 用途 |
|---|---|---|
| 可达性 | 4 个主动关节是否在 `range` 内，以及到边界的余量 | 排除机构不可达的点 |
| 静态支撑力矩 | 站立配平时 4 个腿电机的力矩（`Jᵀ` 映射 m·g/2） | 与 ±20 N·m 对比，算出余量 |
| 雅可比条件数 | 腿空间雅可比 `∂(L,φ)/∂(q_front,q_rear)` 的 cond，以及 VMC 中 `sin(φ3−φ2)` | 远离奇异 |
| 质心高度 h / 开环极点 | MuJoCo 实测质心；`√(g/h)` | 平衡难度参考 |
| 离地间隙 | 机身和腿各 geom 到地面的最小距离；`θ = ±20°` 时是否触地 | 俯仰安全余量 |
| 闭链残差 | 站立后 1 s 的 `connect` 残差 | 模型健康检查 |

- 交付：完整结果表，不做取舍，由用户选择 L0。`φ_c` 允许范围用同样方法扫描。

M0 验收（只做核验，不调参）：

- 站立位形下 `ncon = 2`，每轮一个点，且没有其他接触。
- 平面内闭链行秩为 8，独立自由度为 12。
- 用轮心、半径和轮轴算出的接触点，与 MuJoCo 接触点的偏差小于 1 mm。
- 实测站立位形的总质量、质心高度和俯仰惯量，写进第 7 节参数表。

### 3.2 M0 实施记录（2026-10-01）

**模型改动（`MJCF/robot.xml`，可用 `robot.xml.bak` 还原）**：

1. 轮网格改为 `class="untouch"`，只用于显示和质量，质量 0.322 kg 不变。新增 `right_wheel_collision` / `left_wheel_collision` 圆柱体，`density="0"`，不改变质量。
   - 半径 0.077：网格在轮 body 系 xy 平面内的最大径向距离。
   - 半宽 0.011：胎面 z 范围 −0.0161 ~ 0.0061（右轮）。**M4 已改为 0.001**（用户选方案 A，见第 5.4 节）。
   - 中心偏置 ∓ 0.005：取胎面中点。
   - 轮轴 = body z = 关节轴。
2. 新增 `<contact><exclude>` 6 对：GH–BE/EC/CF、OP–JM/MK/KN。
   - 依据：车身悬空、主动关节施加 ±6 N·m 随机力矩，跑 20 s，只在这 6 对之间出现过接触（CF–GH 43244 步，KN–OP 48457 步，其余每对 896–3134 步）。
   - 腿与机身、腿与地面的碰撞保留。
3. 站立位形没有写成 keyframe（keyframe 会把 L0 固定下来），而是由 `gbc_mpc_wbc/robot.py::standing_pose(L0)` 在运行时生成：
   - 用牛顿法闭合闭链并达到目标 (L, φ)；
   - 用割线法求静平衡腿角；
   - 最后放到地面，穿透量 1e-4 m。
   - 选定 L0 后可以再决定是否固化成 keyframe。

闭链 site 的 4.5 mm 初始装配误差在 M0 未修，M1 已修（见第 5.1 节）。

**M0 验收结果（L0 = 0.30 站立位形）**：

| 项 | 结果 | 评价 |
|---|---|---|
| 接触 | 4 个点，全部是地面–轮；每轮 2 点（圆柱–平面的两个边缘点） | 无其他接触 ✓。原定的"每轮 1 点"不成立，WBC 按每轮 1 个几何接触点建模（与 circleg 相同），力取两点之和 |
| 接触点偏差 | 两点均值与解析最低点相差 0.05 mm | < 1 mm ✓ |
| 闭链平面内 8 行 | 秩 8，cond 11.9 | ✓ |
| 独立自由度 | 20 − 8 = 12 | ✓ |
| 质量 | 14.546 kg；静态法向力 71.4 / 71.3 N | 与 m·g/2 一致 ✓ |
| 旧 VMC 一致性回归 | 右腿 L0 误差最大 4.9 mm，左腿 2.0 mm | 与改模型前相同 ✓ |

**L0 扫描结果摘要**（完整表见 `results/m0_l0_sweep.md`）：

- 机构可达：L0 ∈ [0.14, 0.42] m；0.12 是 jAG 越界，0.44 是 jOP 越界。
- 限制关节：L0 ≤ 0.30 时是 jAG（大腿）下限，L0 ≥ 0.32 时是 jOP（小腿）。余量最大在 0.32 附近（0.81 rad）。
- 静态支撑力矩全程 ≤ 7.55 N·m（限值的 38%），峰值在 0.28。不是约束。
- 腿雅可比 cond 为 3.3–6.2，闭链块 cond 为 12–21。全程远离奇异。旧 VMC 的 |sin(φ3−φ2)| 在 0.30 时为 1.000，在 0.42 时降到 0.767。
- 俯仰 ±20° 时的离地间隙：L0 ≤ 0.18 时机身（base 网格）会触地，0.20 时仅剩 12.3 mm，≥ 0.22 时由小腿 OP 决定（约 19.7 mm，与俯仰无关）。
- 开环极点 √(g/h)：7.62（L0 = 0.14）到 5.19（L0 = 0.42）rad/s。腿越长，发散越慢。
- 静平衡腿角 φ0 约为 −0.05 ~ −0.26 rad（轮心略偏后），原因是机身质心不在髋轴正上方。
- 静态力矩开环保持 0.3 s：0.14–0.42 内腿长漂移 ≤ 3.4 mm，说明静态力矩解正确。闭链残差当时始终是 4.5 mm（即 site 装配误差）；M1 修正 site 后重跑，可达点的闭链残差≤ 0.06 mm，其余各列变化在末位（`results/m0_l0_sweep.md` 已更新）。

是否选定 L0 以及选哪一个，由用户根据上表决定。

### 3.3 L0 = 0.42 m 工作点余量（实测，两腿对称）
| 项 | 值 | 说明 |
|---|---|---|
| φ0（静平衡） | −0.0502 rad | 轮心略偏后 |
| 最小关节余量 | 0.110 rad（jOP，小腿） | 扫描表中各可达点里最小的 |
| 伸腿余量 | L 最大约 0.431 m（jOP 到限），即 **仅 +11 mm** | 缩腿方向可以一直到 0.14 m |
| φ 可达范围（L = 0.42） | [−0.554, +0.544] rad，分别由 jIJ / jIO 限制 | 相对 φ0 约 −0.50 / +0.59 rad |
| 俯仰 ±20° 离地间隙 | 19.7 mm（小腿 OP） | 机身不会触地 |
| 开环极点 √(g/h) | 5.19 rad/s（h = 0.364 m） | 扫描范围内最慢 |
| 静态支撑力矩 | 5.33 N·m（限值的 27%） | |
| 腿雅可比 cond | 4.9；闭链块 12.9 | 远离奇异 |

**对后续的影响（事实，不是建议）**：

- height 任务在伸长方向只有 11 mm 行程。遇到向下的扰动、或需要伸腿来抵抗横滚时，程序会碰到 jOP 的限位。MuJoCo 的关节限位是硬约束，WBC 里没有这条约束。
- 这样一来，roll 任务（两腿差动）实际上只能靠"一侧缩腿"来实现。
- M1 / M2 会把关节限位是否被触发（`efc_type == LIMIT_JOINT`）加入记录和停机判据，如实报告。如果需要更改工作点，由用户决定。

### 3.4 站高重定：L0 = 0.42 → 0.28 m（2026-10-03，用户提出）

用户反馈：默认站高太高、不合理，希望调到合理高度。按实测重新定工作点，不凭感觉：

**机构行程（增量扫，避开 IK 跳分支）**：从各站高出发沿长度方向单调增量扫，伸展上限均为 **0.430 m**
（与起始站高无关）；缩腿方向到 0.14 m。所以伸展余量随站高线性减少：

| L0 m | 可伸到 m | 伸展余量 mm | 缩腿余量 mm | 质心高 m | 6 状态极点 rad/s |
|---|---|---|---|---|---|
| 0.28 | 0.430 | +150 | 140 | 0.264 | 9.48 |
| 0.30 | 0.430 | +130 | 160 | 0.279 | 9.55 |
| 0.32 | 0.430 | +110 | 180 | 0.293 | 9.59 |
| 0.36 | 0.430 | +70 | 220 | 0.321 | 9.61 |
| 0.42 | 0.430 | **+10** | 280 | 0.364 | 9.50 |

**选点依据**：

1. **行程对称**：`0.430 − L0 ≈ L0 − 0.14` → L0 ≈ 0.285。取 **0.28 m**，双向各约 0.15 m。
   原来的 0.42 m 贴在伸展上限上，高度 / 横滚任务基本没有权限（早期 M4 就是因此碰到 jOP 限位）。
2. **关节余量**：0.28 m 时最小关节余量 +0.660 rad（0.42 m 时只有 +0.110 rad）。
3. **腿角范围**：0.28 m 时共模腿角可达 [−0.784, +0.674]，相对 φ0 = −0.1263 即 −0.658 / +0.800 rad，
   所以 `leg_angle_limit` 从 0.50 改为 **0.65 rad**。
4. **极点基本不变**：9.48（0.28） vs 9.50（0.42） rad/s，即降低站高不会让发散变快（扫描带内 9.2–9.6）。
5. **扰动力矩变小**：质心高 0.264 vs 0.364 m，水平推力产生的力矩降约 27%。
6. **代价**：静态腿力矩从 5.33 升到 7.55 N·m（限值 20 N·m 的 38%），仍有余量。
   腿雅可比 cond 3.34（0.42 m 时 4.86），反而更好。

**零输入平衡重跑（L0 = 0.28，30 s + 0.2 m/s 推扰）**：

| 模型 | 后端 | 完成 | max\|θ\| | max\|横滚\| | 末位移 | fallback / clamp / infeasible |
|---|---|---|---|---|---|
| held_leg（6 状态） | nullspace / wls | 是 | 20.4 mrad | 0.01 mrad | 0.67 mm | 0 / 0 / 0 |
| leg_input（8 状态，ZMP 开） | nullspace / wls | 是 | 1.53 mrad | 0.00 mrad | 0.02 mm | 0 / 0 / 0 |

四个组合都完成、都不发散、都没有 fallback。站高可用 viewer 的 `--leg-length` 临时改（机构范围约 0.14–0.43）。

## 4. 复用与重写清单

### 4.1 直接复用（不改代码）

`Algorithm/` 不依赖 `APP/`（已 grep 确认），可以通过 import 复用：

| 模块 | 用途 | 说明 |
|---|---|---|
| `Algorithm/timing.py` | 1 kHz / 250 Hz / 50 Hz 整数步调度 | 不变 |
| `Algorithm/supervision.py` | 模式机、MPC/WBC 编排 | 双轮接触语义相同 |
| `Algorithm/mpc/planner.py`、`trajectory.py`、`qp.py` | 速度规划、凝聚 QP、轨迹采样 | 按 `state_dim` 和 `input_dim` 泛化 |
| `Algorithm/mpc/controller.py` | `WheelMPC` | **所有机器人相关的默认值必须显式覆盖**（第 7 节） |
| `Algorithm/wbc/projection.py` | `NullSpaceHierarchy`（质量加权递推零空间）、`ContactElimination`、`TaskLevel` | **主后端**。闭链 8 行与轮接触 5 行一起作为硬等式；要求行数 < nv：13 < 20，满足 |
| `Algorithm/wbc/sns.py` | `SaturationResolver`（零空间 + 力矩增量限幅）、`ActuatorLimits` | 零空间后端的限幅 |
| `Algorithm/wbc/weighted.py` | `WeightedSaturationResolver`（WLS） | 备选后端，与零空间法同接口 |
| `Algorithm/wbc/task_qp.py` | 单级 QP | 保留为后备，第 4.4 节触发条件满足时再启用 |

不复用：

- `Algorithm/jump.py`、`contact_state.py`：基于曲柄几何的跳跃，GBC486 需要重新设计。
- `Algorithm/wbc/schedule.py`：死点排程，GBC486 没有这个问题。
- `Algorithm/mpc/contact.py`：常量 0.12 / 0.24 是 circleg 专用的。

### 4.2 新写：GBC486 适配层（对应 circleg 的 `APP/mujoco_mpc_wbc.py` 和 `APP/mujoco_balance_model.py`）

**(a) `Gbc486Dynamics`**，对应 `MuJoCoDynamics`：

- `selection`：6×20，由 `actuator_trnid` 和 `gear` 生成（circleg 写法可以直接沿用）。
- `mass()`、`h = qfrc_bias − qfrc_passive`：与 circleg 相同。
- **轮接触行**：每轮切向 + 法向，加共模横向 1 行，共 5 行，排列与 circleg 一致 `[切R, 法R, 切L, 法L, 横]`，所以 `normal_indices = (1, 3)` 不用变。
- **闭链行（新增）**：每个 `connect` 取 `mj_jac(site1) − mj_jac(site2)` 的世界 x、z 两行，共 8 行，放在接触行之后。它们的乘子是内力，不加非负约束。偏置项 `J̇q̇` 沿用 circleg 的 `mj_integratePos` 中心差分。
- `sensed_contact()`：改成识别新的圆柱碰撞体名。

**(b) 腿空间量**，替代 circleg 的 `_hip_common` / `_hip_differential`：

- 每侧以髋轴点 A（右 AG 原点，左 IO 原点）为极点，在机身系 xz 平面内取轮心，得到 `L_i, φ_i`。对应的雅可比由 `mj_jac(轮心) − mj_jac(A)` 转成极坐标得到，是精确值，不经过 VMC 的等效五连杆近似。
- 组合量：
  - `L_c = (L_R + L_L)/2`：高度
  - `L_d = (L_R − L_L)/2`：横滚
  - `φ_c`：共模腿摆角，对应 circleg 的 `q_c`
  - `φ_d`：差模腿摆角，决定两轮前后错开量
- `VMC.py` 保留，只用来做一致性比对和日志，不再参与力矩计算（WBC 直接输出关节力矩）。

**(c) 线性化 Builder**，对应 `HeldHipModelBuilder` / `HipInputModelBuilder`：

- 方法和 circleg 相同：在 MuJoCo 当前位形上用 KKT 求约束加速度，再做中心差分。约束集合 = 轮接触 + 闭链 8 行 + "握住"的腿空间量。
- **阶段一（6 状态，held-leg）**：握住 `L_R, L_L, φ_d, φ_c`。状态 `[s, v, θ, θ̇, ψ, ψ̇]`，输入 `[f_R, f_L]`。
- **阶段二（8 状态，leg-input）**：放开 `φ_c`，作为状态并以共模腿摆力矩作为输入，与 circleg 的 `hip_input` 一一对应。状态追加 `[φ_c, φ̇_c]`，输入追加 `τ_c`。`τ_c` 到 4 个腿电机的分配由腿雅可比的转置给出。
- 输出 `ReducedModel(A, B, wheel_radius, half_track, yaw_inertia, metadata)`，其中 `metadata["equilibrium_pitch"]` 来自配平。

**(d) 任务集 `tasks()`**，对应 circleg `tasks()`。初值照搬 circleg，不调：

| 任务 | 雅可比 | 目标加速度 | 与 circleg 的关系 |
|---|---|---|---|
| pitch | 俯仰欧拉率行 | MPC 前馈 + PD (80, 10) | 相同 |
| roll | 横滚欧拉率行 | −kp·roll − kd·roll_rate | 相同。执行器从"髋差动"变成"腿长差动 `L_d`"，**不需要死点排程** |
| yaw | 偏航欧拉率行 | MPC 前馈 + PD (80, 10) | 相同 |
| height | 腿长共模 `L_c` 行 | kp_h (L* − L_c) − kd_h L̇_c，(100, 12) | circleg 用机身 z，这里改用腿长更直接 |
| forward_velocity | heading 速度行 | `a_mpc + kv (v_mpc − v) + kp (s_mpc − s)`，(14, 49) | 相同（forward_mode = "mpc"） |
| leg_tracking | `φ_c` 行 | 阶段一：保持标称；阶段二：跟踪 MPC 的 `φ_c`（含前馈） | 对应 `hip_tracking` |
| leg_differential | `φ_d`、`L_d` 正则 | 保持名义 | 对应 `hip_differential` |

权重沿用 `_TASK_QP_WEIGHTS`。新增的 `height` 和 `leg_differential` 先用 circleg 中对应项的值，标注为"初值"。

**(e) 主循环**：结构照搬 `MuJoCoMpcWbcSimulation.run()`（扰动注入、viewer 实时同步、保护、记录）。去掉跳跃分支，`data.ctrl` 按本模型的 6 通道顺序写入。

### 4.3 代码放置方式（已定：方案 A）

| 方案 | 做法 | 优点 | 风险 |
|---|---|---|---|
| A（建议） | 在 `wheel_leg_mujoco/` 下新建 `gbc_mpc_wbc/`，通过 `sys.path` 或 包导入复用上层 `Algorithm/` | 不分叉；circleg 后续修复自动生效 | circleg 修改 `Algorithm` 时可能影响本项目，需要两边回归 |
| B | 把 `Algorithm/` 复制进 `wheel_leg_mujoco/` | 两边互不影响 | 两份代码会分叉，以后修复要做两次 |

运行环境建议用 circleg 的 `.venv`（mujoco 3.13、osqp 1.1.3、scipy 已齐全）。readme 中的 mujoco 3.4.0 与 `mj_fullM` 新签名不兼容，circleg 代码里已有版本分支。

### 4.4 WBC 后端：零空间法 / WLS（决策 6）

**维数账（按第 2 节实测，平地双支撑）**：

```
独立自由度   = nv − 闭链平面行 = 20 − 8 = 12
接触约束     = 2 法向 + 2 纵向 + 1 集中横向 = 5
运动自由度   = 12 − 5 = 7  ≈ [s, θ, ψ, L_R, L_L, φ_R, φ_L]
执行器       = 6  →  净欠驱动 1 维（倒立摆方向）
可实现 qacc  = 6 维（由 τ 唯一决定）
```

任务共 7 行：pitch、forward_velocity、yaw、height（L_c）、roll（L_d）、leg_tracking（φ_c）、leg_differential（φ_d）。正好等于运动自由度 7，比可实现维数 6 多 1 行。

这是第 23 次排查里五连杆"前 6 行各有独立通道，被饿死的只是末级"判断在本模型上的具体化。和 circleg 曲柄（7 行对 4 维）相比，好得多。

**严格分层的排序建议（初值，M2 验证；实现时已按 2026-10-03 的照搬方案定稿，见第 5.5 节）**：

> 注：本节的任务表是方案阶段的初值。实际实现按用户 2026-10-03 的要求**直接照搬 circleg**，
> 集合 / 分组 / 增益均取自 circleg，具体顺序见第 5.5 节。下表的物理判断仍然成立。

| 级 | 任务 | 执行通道 | 说明 |
|---|---|---|---|
| 硬 | 闭链 8 行 + 轮接触 5 行 + 欠驱动动力学 | — | `ContactElimination.dynamics_constraint` |
| 1 | pitch | 两轮共模 + 腿摆共模 | 发散最快的通道 |
| 2 | roll | 腿长差模 `L_d` | 有独立执行器，不需要死点排程 |
| 3 | height | 腿长共模 `L_c` | 有独立执行器 |
| 4 | yaw | 两轮差模 | — |
| 5 | leg_tracking `φ_c` | 腿摆共模 | 6 状态阶段为"握住"，与 MPC 模型假设一致 |
| 6 | leg_differential `φ_d` | 腿摆差模 | 防止两轮前后错开引起偏航和 scrub |
| 7 | forward_velocity | — | **结构上分不到独立权限**，由 pitch、φ_c 和欠驱动动力学决定 |

要点：

- forward_velocity 放在最后，与 circleg 原方案 §0 / §4.3 的结论相同：速度跟踪由 MPC 通过俯仰参考来实现。**调速度跟踪要调 MPC，改 WBC 的速度增益基本无效。**
- 8 状态阶段（M5）MPC 同时规划 θ、v、φ_c，三者动力学一致，所以末级被饿死的影响只剩 PD 修正部分。
- 上表的"7 行 / 6 维"是按自由度推算的，M1 要用 `NullSpaceHierarchy` 的 `exhausted` 标志和每级条件数实际确认。

**WLS（`WeightedSaturationResolver`）**：同一组 7 个任务在硬约束零空间内按权重同时分配，不会把某一项清零，但优先级是软的。权重初值取 circleg `_TASK_QP_WEIGHTS` 中的对应项。

**两种后端共有的已知限制（读码确认）**：

| 限制 | 位置 | 影响 | 处理 |
|---|---|---|---|
| 不表达法向非负 `f_n ≥ 0` | 两者都只处理等式和力矩箱 | 可能给出"拉地面"的解 | 沿用 circleg 主循环的 `negative_predicted_normal_force` 停机判据，只记录，不在 WBC 内修正 |
| WLS 越限时直接 `clamp` | `weighted.py` | 裁剪后的 τ 与 qacc 不再动力学一致，`feasible = False` | 记录越限拍数，作为第 4.4 节切换判据之一 |
| 零空间超限时整拍退出任务跟踪 | `sns.py` 的 `fallback` 分支 | 只输出钳位后的力矩 | 记录 `fallback_used` 次数 |
| 无摩擦锥 | 两者都没有 | 与 circleg QP 当前状态相同 | 不处理 |
| WLS 的 `feasible` 判据为 1e-7 | `weighted.py` | 比 QP 的 1e-5 严，可能误报 | M2 观察残差量级，**如需放宽要先报告用户** |

**接入方式**：新链路提供 `--wbc-backend nullspace|wls|qp`，默认 `nullspace`。三个后端共用同一个任务集和同一个 `SupervisedControl`（circleg 的 `wbc_resolver_factory` 注入点）。

**切换到 QP 的触发条件（由用户判断，文档只提供数据）**：

1. M2 零速站立期间，`fallback_used` 或越限 clamp 频繁出现；
2. 出现 `negative_predicted_normal_force` 停机；
3. 末级（forward_velocity）被饿死导致 M4 速度跟踪不达标，并且已确认原因不在 MPC；
4. 计算耗时没有明显优于 QP（M2 同时记录三个后端的耗时，用于对比）。

## 5. 里程碑与验收（只验证可行性）

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | 模型准备（第 3 节） | 见第 3 节的 4 条验收 |
| M1 | `Gbc486Dynamics` + 腿空间量 | 稳态滚动时 `Jc q̇` < 1e-6；闭链行 `J q̇` < 1e-6；QP 回代的接触力与 `mj_contactForce` 误差 < 1%；腿空间 L、φ 与 VMC 相差在毫米级（参照 `PROJECT_ANALYSIS.md` 5.4） |
| M2 | **WBC 单独运行**（`sample = None`，只有 PD 任务，不接 MPC），后端 `nullspace`，同时跑 `wls` 对照 | 零速站立 10 s 不停机；`dynamics_residual` < 1e-5；逐级 `exhausted` 和条件数与第 4.4 节维数账一致；记录 `fallback_used` 和越限拍数、单次耗时 |
| M3 | 6 状态 Builder + WheelMPC 接入 | 开环不稳定极点与 `√(g/h)` 量级一致；零速站立加小扰动后能回位，不触发 `persistent_mpc_fallback` |
| M4 | 速度和偏航指令（键盘） | 阶跃速度能跟踪，不触发 `attitude_guard` / `speed_guard` |
| M5 | 8 状态 leg-input 模型 | 与 M4 同工况对比，指标逐项列出，不做取舍结论 |
| M6（可选） | 腿长变化、粗糙地形、跳跃 | 另行立项 |

M2 必须在 M3 之前完成，原因与 circleg §7.2 相同：WBC 和 MPC 要解耦调试。
每个阶段失败时，按 `AGENTS.md` §1 如实记录现象和证据，不通过改参数让它通过。

### 5.1 M1 实施记录（2026-10-01）

**代码**（都在 `wheel_leg_mujoco/` 下）：

- `gbc_mpc_wbc/dynamics.py::Gbc486Dynamics`
  - 选择矩阵乘上了 `gainprm`，所以 WBC 力矩变量就是 `data.ctrl`，左轮的 −1 已经包含在内。
  - 接触行与 circleg 同一种排列。
  - 闭链 8 行：取机身 x / z 投影。
  - 腿空间 4 行：解析雅可比，机身角速度列为 0。
  - 所有 bias 都用 `mj_integratePos` 中心差分求得。
- `gbc_mpc_wbc/_circleg.py`：只读导入上层 `Algorithm/`，并设置 `sys.dont_write_bytecode`。

**模型修正**：闭链 site 的平面外装配误差（右 2.4 / 4.5 mm，左边相同）。

- 做法：把 AG-D、GH-F、MK-L、KN-N 沿各自关节轴平移，使它们与配对 site 重合。
- 修前：站立时 connect 约束力在平面外 y 方向高达 **7.2×10⁴ N**，这是软约束为了拉回几毫米偏差而产生的内力。
- 修后：降到 2.4 N；接触力回代误差从切向 0.55 / 法向 1.21 N 降到 0.22 / 0.40 N。
- 站立位形下闭链残差为 0。
- site 只用于约束，质量和几何都不变。

**核验结果**（`results/m1_dynamics_check.md`，L0 = 0.42）：

| 项 | 结果 | 评价 |
|---|---|---|
| 选择矩阵 | 与 `qfrc_actuator` 差 0 | ✓ |
| 硬约束 | 13 行（接触 5 + 闭链 8），秩 13，cond 75 | ✓ |
| 运动自由度 | 7 | 与第 4.4 节推算一致 ✓ |
| 欠驱动维数 | 1（`Q_uᵀSᵀ` 7×6，cond 2.8） | 与推算一致 ✓ |
| 计划任务秩 | 7 行全部独立，逐级累计秩 1…7 | 末级 forward_velocity 只剩欠驱动那 1 维，符合预期 |
| roll 与 L_d | 投影后秩 2，但 \|cos\| = 0.98 | **高度相关**：腿长差动几乎就是机身横滚。严格分层时，放在后面的一级权限很弱。第 4.4 节没有把 L_d 单独列为任务，与此一致 |
| 行与 bias 的运动学核验 | 一阶 ≤ 9×10⁻¹⁰，二阶 ≤ 6×10⁻⁶（相对） | ✓ 腿空间行和闭链行的解析雅可比与偏置正确 |
| 接触力回代 vs `mj_contactForce` | 切向 ≤ 0.22 N，法向 ≤ 0.40 N（法向力约 70 N，< 0.6%） | ✓ 验收标准 < 1% |
| 腿长 vs 旧 VMC | 右 −4.1 mm，左 −1.1 mm | 与之前的分析一致 |
| 关节限位 / 非轮接触 | 没有触发 | ✓ |

**未达预定阈值的项（如实记录）**：第 5 节原定的"稳态时 `Jc q̇` < 1e-6"和"闭链 `J q̇` < 1e-6"。

- 在扰动开环仿真中，接触行 max |Jc q̇| = 1.1×10⁻² m/s，闭链行 2.9×10⁻³ m/s。
- 只加静态力矩的稳态下：闭链 J q̇ = 3×10⁻⁶（dt = 1 ms）/ 5×10⁻⁷（dt = 0.1 ms），已经接近阈值；接触行仍有 3.5×10⁻³ m/s。
- 原因：MuJoCo 的接触和闭链都是软约束（穿透约 0.54 mm）。行本身的正确性已由运动学核验独立证明。
- 这是模型软约束与 WBC 硬等式之间的偏差（第 6 节风险 2），不是适配层错误。
- circleg 在 §7.4 里对此也给了同一种阈值。这两条阈值不适合作为本模型的通过标准，是否改用运动学核验作为验收，由用户确认。

**对 M2 的提示**：第 4.4 节的任务集在这个位形下满秩，可以直接进入 M2。

### 5.2 M2 实施记录（2026-10-01）

**代码**：`gbc_mpc_wbc/wbc.py::StandaloneWbc`。

- 7 个任务按第 4.4 节顺序排列。
- 硬约束 = 着地轮接触 + 闭链 8 行。
- 三个后端都直接调用 circleg 的 resolver，没有修改 circleg 代码。没有经过 `SupervisedControl`，因为它的 `ContactBlock` 要求每一块都声明所属的轮子，而闭链行不属于任何一个轮子。M3 接入监督层时再处理。
- 增益和权重都是 circleg 默认值直接搬过来的，**未调参**。
- 没有 MPC 时，forward_velocity 只做速度阻尼 `−kv·v`（kv = 14），与 circleg `sample is None` 分支一致。

**试验**：L0 = 0.42 静平衡位形零速站立 10 s，WBC 250 Hz，物理 1 kHz，没有扰动。

- 停机判据：关节限位、非轮接触、infeasible、预测法向力为负、力矩越限、姿态超过 45°、出现 NaN。
- 完整结果见 `results/m2_standalone_wbc.md`。

| 指标 | nullspace | wls | qp（对照） |
|---|---|---|---|
| 10 s 完成 / 停机 | ✓ / 无 | ✓ / 无 | ✓ / 无 |
| fallback / clamp / infeasible / 求解异常 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| 最大动力学残差 | 7.4e-13 | 7.2e-13 | 1.1e-13 |
| 最小预测法向力 | 70.7 N | 70.7 N | 69.6 N |
| max\|θ\| / max\|roll\| | 0.05 / 0.00 mrad | 0.05 / 0.00 mrad | 0.92 / 0.00 mrad |
| 腿长 / 腿角误差 max | 0.27 mm / 0.20 mrad | 0.27 mm / 0.20 mrad | 4.35 mm / 0.50 mrad |
| x 漂移（10 s 末） | −4.3 mm | −5.8 mm | −653 mm |
| 求解耗时 p50 / p95 / p99 | 1.8 / 2.9 / 3.7 ms | 0.74 / 1.3 / 1.8 ms | 10.0 / 13.8 / 16.3 ms |
| 关节限位 / 非轮接触 | 无 / 无 | 无 / 无 | 无 / 无 |

**验收（第 5 节 M2 一行）**：

1. 零速站立 10 s 不停机：三个后端都 ✓。
2. `dynamics_residual` < 1e-5：✓。
3. 逐级 `exhausted` 与维数账一致：✓。零空间法下前 6 级的 exhausted 都是 0，条件数都是 1.0；只有末级 forward_velocity 在全部 2500 拍都是 exhausted（条件数 inf）。这与 M1 "逐级累计秩 1…7，可实现 6 维"的结论完全一致。
4. 已记录 fallback、越限拍数和耗时：✓。

**观察到的现象（只记录，不是通过 / 不通过判据）**：

1. **nullspace / wls：缓慢向后漂移，姿态保持得很好**。
   - nullspace：延长到 30 s 的诊断运行，x 到 −39.5 mm，v 到 −2.7 mm/s，近似匀加速（约 −0.09 mm/s²）；俯仰始终为 0。
   - wls：同样 30 s，x 到 −119 mm，v 到 −13 mm/s，俯仰慢慢变到 −0.1 mrad，呈指数增长趋势。
   - 原因（已验证的部分）：欠驱动方向实测为 [v, θ] ≈ [0.34, 0.94]，其余任务为 0。这就是倒立摆方向。严格分层把 pitch 放在第 1 级，剩下与它耦合的 forward_velocity 就没有权限了。所以没有 MPC 时，x / v 没有任何反馈。
   - 推动漂移的是 WBC 硬等式模型与 MuJoCo 软接触之间的偏差。实测首拍预测 v̇ = −7.7e-13，但实际为 +0.32 m/s²，并伴随穿透从 0.10 增长到 0.42 mm 的建立过程。这就是第 6 节风险 2。
   - 已排除流体力的影响：去掉空气密度和粘度后重跑（只在内存中诊断），漂移不变。
   - 这个问题应该由 M3 的 MPC 来解决。它规划 θ 参考，把位置和速度误差转成俯仰动作，与 circleg 的结论相同。在这里调 WBC 的速度增益是无效的，零空间法下它已经是 exhausted。
2. **qp：漂移大约是前两者的 150 倍**：10 s 时 x = −653 mm，v = −154 mm/s，θ = −0.92 mrad，全部单调增长。
   - 主要原因：QP 把各任务作为软代价加权竞争，而且有力矩 / 力正则 1e-3。实测稳态的 height 残差为 0.44 m/s²，腿长比 L0 短了 4.35 mm，forward 残差 2.2 m/s²。任务之间的折中在欠驱动方向上留下了持续偏差。
   - 上面是根据残差做的推断，没有做单项剥离实验。
3. **耗时**：
   - wls 0.74 ms、nullspace 1.8 ms、qp 10 ms（p50，纯 Python）。
   - 另外 `rowset()` 每拍约 0.5 ms。
   - qp 慢是因为 circleg 的 resolver 每拍都重新执行 `osqp.setup`（第 23 次排查 W0 已经指出），另外本模型变量有 20 + 13 + 6 = 39 个。
   - 250 Hz 的周期是 4 ms：nullspace 的 p99 为 3.7 ms，接近上限；wls 有余量；qp 达不到实时。
   - 仿真是非实时运行，所以这些耗时不影响结果。

**第 4.4 节切换到 QP 的触发条件检查**：条件 1（fallback / clamp 频繁）、条件 2（负法向力）都未触发；条件 3 要到 M4 才能判断；条件 4 显示零空间法和 WLS 的耗时明显优于 QP。**目前没有切换到 QP 的依据。**

### 5.3 M3 实施记录（2026-10-01）

**circleg 版本记录**（方案 A 的风险，在 M2 时已指出）：

- HEAD `9572a11`。`Algorithm/mpc/controller.py`、`planner.py` 有未提交改动，是别处的工作。
- 第一轮运行时，`controller.py` 在 15:43:38 被修改。因为无法确定当时导入的是哪一版，整轮结果作废并重跑。
- 现在 `tools/m3_mpc_wbc.py` 启动时会记录每个已导入 `Algorithm` 文件的 SHA-256，写入 json 的 `circleg.imported_sha256`。
- 本节数据对应 `controller.py` = `3b43aff9f7c5`，与事后磁盘上的文件一致。
- **上层 `Algorithm/` 改动后，M3 的结果需要重跑确认。**

**代码**：

- `gbc_mpc_wbc/model_builder.py::HeldLegModelBuilder`：对应 circleg `HeldHipModelBuilder`。
  - 约束 = 轮接触 5 + 闭链 8 + 腿空间 4 行“握住”；速度映射满秩 20。
  - 用 KKT 方程加中心差分求导，用 brentq 求配平俯仰角。
  - 自带轮力和偏航的符号校验。
- `gbc_mpc_wbc/simulation.py::Gbc486MpcWbcSimulation`：结构照搬 circleg `run()`（forward_mode = "mpc"、preview_lead = 0）。
  - 复用 `ControlSupervisor`、`WheelMPC`、`ControlTiming`。
  - WBC 的 pitch / yaw / forward 三个任务改为跟踪 MPC 轨迹。
- MPC 设置：权重和规划限值都保持 circleg `MPCSettings()` 的默认值。下面是换模型必须换的物理量（第 7 节），逐项显式覆盖：

| 参数 | circleg | GBC486 | 来源 |
|---|---|---|---|
| `wheel_torque_limit` | 5.0 | 4.0 | 轮 `ctrlrange` |
| `nominal_normal_force` | 46.6 | 71.3 | m·g/2 |
| `yaw_moment_limit` | 7.5 | 25.5 | 2·(τ/R)·d |
| `scrub_moment_coefficient` | 5.59 | 0 | 对称姿态下两轮不前后错开 |
| `compensate_hip_torque` | — | False | held 模型已包含反作用力 |

**降阶模型（MuJoCo 实测）**：

- 特征值 {±9.500, 0, 0, 0, 0}，不稳定极点 **9.500 rad/s**。
- B 的符号：v̇ 对两轮力都为正；θ̈ 对两轮力都为负（非最小相位）；ψ̈ = ±0.543，即 d / I_yaw。
- 与 M0 里的 `√(g/h) = 5.19` 相比约为 1.83 倍。量级一致，但数值不同：
  - 原因：简单单摆公式忽略了轮子可以自由滚动（小车倒立摆的耦合项），整机 96% 的质量都在摆上。
  - 用标准小车倒立摆解析式独立重算：ω² = M_b·g·l / (I_b + M_b·l² − M_b²·l² / M_x)，得到 **9.500 rad/s**，与实测一致到小数点后 3 位。
  - A[1,2] 的解析值 0.3491，实测 0.3492。
  - 模型可信。之前拿来比较的 `√(g/h)` 只是量级参考，M0 扫描表中那一列也要按这个理解。
- 与 circleg 相比：circleg 是 4.70 rad/s，本机发散快 2 倍。每个 MPC 周期的放大倍数为 e^(9.5×0.02) = 1.21。

**闭环结果**（`results/m3_mpc_wbc.md`）：

- 推扰工况沿用 circleg `full_state_eval`：t = 5 s 时给机身 ±0.20 m/s 的前后或横向速度增量。
- 回位判据：|s| < 10 mm，|θ| < 10 mrad。

| 工况 | nullspace | wls |
|---|---|---|
| 30 s 无扰动站立 | 完成；\|s\| ≤ 0.05 mm；M2 的漂移（−40 mm / 30 s）**已消除** | 同左（M2 中为 −119 mm） |
| forward +0.2 m/s | 完成；峰值 s 64.9 mm、θ 63 mrad（3.6°）；s 回位 2.19 s，θ 回位 1.92 s；末 s 0.01 mm | 同左 |
| backward −0.2 m/s | 与 forward 镜像一致 | 同左 |
| lateral +0.2 m/s | 完成；纵向 s 与 θ 几乎不受影响 | 同左 |
| MPC fallback / relaxed / degraded | 0 / 0 / 否 | 0 / 0 / 否 |
| WBC fallback / clip / infeasible | 0 / 0 / 0 | 0 / 0 / 0 |
| 停机 / 关节限位 / 非轮接触 | 无 | 无 |
| 最大轮力矩 | 1.09 N·m（限值 4） | 同左 |
| 推扰后腿长 / 腿角偏差 | ≤ 0.09 mm / ≤ 3.4 mrad | — |
| MPC / WBC 耗时 p50 | 0.9 / 1.8 ms | 0.9 / 0.75 ms |

**验收（第 5 节 M3 一行）**：

1. 开环极点与 √(g/h) 量级一致 ✓，并且已用小车倒立摆解析式精确核对。
2. 零速站立加小扰动后能回位 ✓，没有 `persistent_mpc_fallback`。

**观察**：

- nullspace 和 wls 的轨迹几乎完全重合：位移差 ≤ 0.012 mm，俯仰差 ≤ 0.014 mrad，力矩差 ≤ 0.002 N·m。
  - 原因：前 6 级任务各自有独立权限，都能完全满足。两种解法只在末级 forward_velocity 上有区别：零空间法里它永远是 exhausted，WLS 里它的权重对结果几乎没有影响。
  - 所以在这个任务集下，两个后端的选择只影响耗时（wls 大约快 2 倍）。
- 位移和速度的回位完全靠 MPC 规划 θ 参考来实现，与 circleg 的结论一致。推扰后大约 0.1 s 内俯仰到 −39 mrad，峰值 −63 mrad，然后反向约 +15 mrad，大约 3 s 内回到 1 mm 以内。
- 未做：大扰动和持续推力工况（circleg 的 mid / persist）。验收只要求小扰动；如果需要更大的扰动，另行安排。

### 5.4 M4 实施记录（2026-10-01）

**指令工况**（测试输入，不是控制参数）：

- 用户没有指定具体数值，按 circleg `MotionLimits` 的上限（max_speed = 2.5 m/s，max_yaw_rate = 2.0 rad/s）选了几档：
  - v = 0.5 / 1.0 / 2.0 / −1.0 m/s
  - ψ̇ = 1.0 rad/s
  - v = 1.0 m/s 加 ψ̇ = 0.5 rad/s
- 指令在 t ∈ [2, 8) s 内保持，其余为 0，总时长 14 s。
- 没有接键盘。脚本给的阶跃更容易复现；键盘 / viewer 需要单独的 GUI 入口。
- 增益、权重和规划限值都是 circleg 默认值，**未调参**。
- circleg 版本：`controller.py` = `3b43aff9f7c5`，`planner.py` = `97eefbbda7e9`。

**结果**（`results/m4_command_tracking.md`，nullspace 与 wls 的结果逐项相同）：

| 工况 | 完成 | 稳态误差 | 90% 上升 | 超调 | 停车时间 / 距离 | 刹车反向 | max\|θ\| | max 轮力矩 |
|---|---|---|---|---|---|---|---|---|
| v 0.5 | ✓ | 0.000 m/s | 0.89 s | — | 3.20 s / 0.09 m | 143 mm/s | 7.0° | 1.70 N·m |
| v 1.0 | ✓ | −0.002 | 1.13 s | — | 3.51 s / 0.36 m | 208 mm/s | 10.7° | 1.84 |
| v 2.0 | ✓ | −0.007 | 1.77 s | — | 4.10 s / 1.43 m | 233 mm/s | 12.4° | 1.84 |
| v −1.0 | ✓ | 与 1.0 镜像 | | | | | | |
| v 1.0 + ψ̇ 0.5 | ✓ | v −0.013，ψ̇ −0.026 rad/s | v 1.13 s，ψ̇ 0.18 s | — | 3.51 s / 0.36 m | 204 mm/s | 10.5° | 1.86 |
| **ψ̇ 1.0** | **✗ 3.15 s 停机 `joint_limit:jOP`** | — | — | — | — | — | 0.2° | 1.10 |

- 速度类工况：MPC fallback / relaxed、WBC fallback / clip / infeasible 全部为 0；没有 attitude_guard / speed_guard；轮力矩不超过限值的 47%。
- 起步时有约 60 mm/s 的反向微动。这是 B 矩阵中 θ̈ 对轮力符号为负带来的非最小相位特性，属于预期。
- 刹车时实测 v 落后 planner 参考约 0.2 m/s，刹车结束后反向约 0.2 m/s，约 1.5 s 后回到 0。刹车俯仰最大约 −10°。
- 以上只是记录。是否需要优化，由用户决定；如果要优化，应按 `AGENTS.md` §3 的要求做联合扫描。

**w1.0 停机的根因（已定位，只做了诊断，没有修改模型或参数）**：

1. **现象**：
   - 纯偏航指令给出后，偏航角速度先上升到 0.83 rad/s，然后反而下降到 0.14。
   - 与此同时，差模腿角 φ_d 一直增长到 ±210 mrad：右轮向前、左轮向后，两轮前后错开。
   - 左腿伸长到 0.431 m，小腿 jOP 碰到限位。这正是第 3.3 节指出的"伸腿只剩 11 mm 余量"。
   - 这段时间里，WBC 预测的 φ_d 加速度只有 −0.85，实际却是 +0.49，符号相反。
2. **原因**：
   - WBC（和 circleg 一样）在两轮中点只施加一条集中横向无滑约束。推导依据是两轮无法同时保持横向无滑，否则过约束。
   - 但 MuJoCo 里的圆柱半宽为 11 mm，每轮有 2 个边缘接触点。对每个轮子而言，横向摩擦是很强的实际约束：实测两轮横向力为 ±5.3 N（t = 2.2 s）到 ±26.7 N（t = 2.4 s），远大于 WBC 认为的共模横向力 −0.3 N。
   - 两个轮子各自的横向摩擦力形成一对反向力，绕髋部给出一个 WBC 模型里不存在的力矩，把两条腿前后推开（φ_d）。
   - 转向时这个力矩反过来阻碍偏航，而防止前后错开的 leg_differential 任务认为自己"已满足"（残差 1e-15）。
3. **验证**（只在内存中修改，作为诊断）：
   - 把轮圆柱半宽从 0.011 改成 0.001（近似单点接触，接近 circleg 的 0.01），其余完全不变。
   - 结果 w1.0 **完整跑完 14 s**：稳态 ψ̇ = 1.000 rad/s，max|φ_d| = 0.5 mrad，max|roll| = 0.01 mrad，轮力矩 0.42 N·m。
   - 证实问题在模型与 WBC 接触假设不一致，不在算法。
   - 作为对照，qp 后端在原模型上虽然没有停机（腿长运行点短了 3 mm，恰好躲开了限位），但 φ_d 同样达到 240 mrad，偏航只有 0.08 到 0.47 rad/s，同样跟不上。
   - 这是接触建模的问题，不是后端的问题。第 4.4 节切换 QP 的条件 3 不成立。
4. **可选方案**（用户已选 **A**）：
   - **A（改模型）**：轮碰撞圆柱半宽改为 0.001。已验证有效。代价是碰撞体不再代表实际胎宽 22 mm；实车的宽轮同样会有这个横向摩擦耦合。改动后 M0 到 M4 的接触相关核验需要重跑。
   - **B（改 WBC 接触模型）**：把逐轮横向约束写入 WBC。实测在当前对称位形下约束近乎奇异（最小奇异值 2e-7），执行矩阵 cond 2e10，欠驱动维数从 1 变成 0，数值不可用。需要重新设计，比如按摩擦锥加不等式，这就意味着要用 QP。
   - **C（改工作点）**：缩短 L0，给 jOP 留出余量。只是推迟碰到限位的时间，偏航仍然跟不上（qp 对照已经显示）。**不推荐。**
   - 未验证：半宽介于 0.001 和 0.011 之间时是否可行。

**方案 A 实施与重跑（2026-10-01）**：

- `robot.xml` 两轮圆柱半宽 0.011 → 0.001。半径、中心、摩擦和质量都不变。
- 重跑了所有和接触相关的核验，结果文件已覆盖为新模型的版本：

| 阶段 | 重跑结果 | 与旧模型相比 |
|---|---|---|
| M0 | 每轮仍然是 2 个接触点，但间距只有 2 mm；接触点偏差 0.05 mm；L0 扫描表各列在末位以内不变 | 一致 |
| M1 | 结构：秩 13、运动自由度 7、欠驱动 1、逐级累计秩 1…7 不变；接触力回代误差切向 0.22 → **0.074 N**、法向 0.40 → **0.007 N** | 变好，宽圆柱两点间的力分配误差没有了 |
| M2 | 三个后端跑满 10 s，指标与旧结果相同 | 一致 |
| M3 | 8 个工况全部完成，极点 9.500；推扰峰值和回位时间与旧结果相同 | 一致 |
| M4 | **12 个工况全部完成**，没有停机，fallback / clip / infeasible 全部为 0 | w1.0 从停机变为通过 |

**M4 重跑结果**（nullspace 与 wls 逐项相同）：

| 工况 | 稳态误差 | 90% 上升 | 停车 / 距离 | max\|θ\| | max 轮力矩 | 腿角偏差 max |
|---|---|---|---|---|---|---|
| v 0.5 / 1.0 / 2.0 / −1.0 | ≤ 0.007 m/s | 0.89 / 1.13 / 1.77 / 1.12 s | 3.2–4.1 s / 0.09–1.43 m | 7.0–12.4° | ≤ 1.84 N·m | ≤ 1.8 mrad（旧 ≤ 3.6） |
| **ψ̇ 1.0** | **ψ̇ 稳态 1.000 rad/s，误差 0.000** | 0.22 s | — | 0.0° | 0.42 | 0.5 mrad（旧为 210 mrad 并停机） |
| v 1.0 + ψ̇ 0.5 | v −0.002，ψ̇ 0.000（旧为 −0.013 / −0.026） | v 1.12 s，ψ̇ 0.15 s | 3.51 s / 0.36 m | 10.7° | 2.01 | 1.6 mrad（旧 44.8） |

- w1.0：6 s 内航向变化 5.988 rad，指令为 6.0。原地转向时机身平面漂移最大 70 mm。停止后 |ψ̇| < 2e-4。
- 速度类工况的指标与旧模型几乎相同。上一轮记录的起步反向约 60 mm/s、刹车滞后和反向约 0.2 m/s 仍然存在，属于 MPC 规划特性，只记录。

**验收（第 5 节 M4 一行）**：阶跃速度能跟踪，没有触发 attitude_guard / speed_guard ✓。偏航阶跃也能跟踪 ✓。

**第 4.4 节切换到 QP 的条件检查**：条件 3（末级被饿死导致速度跟踪不达标）不成立，速度稳态误差 ≤ 0.007 m/s。四个条件都不满足，**继续使用零空间法 / WLS**。

**方案 A 的已知代价**：

- 碰撞体宽 2 mm，实际胎宽 22 mm。
- 在仿真里可以认为是"理想单点接触"。实车的宽胎在转向时会出现第 5.4 节所述的逐轮横向摩擦耦合。
- 如果要在仿真里复现这个效应来评估实车风险，可以用 `robot.xml.bak` 中的宽圆柱对照。对应的算法修改属于方案 B。

### 5.5 控制层改为直接照搬 circleg（2026-10-03）

用户要求：控制方式完全照搬轮圈腿的算法，只改建模，不做多余设计；只要求零输入下能正常平衡。
已按此重构，并重跑了零输入工况。

**重构内容**：

| 项 | 重构前（自建） | 重构后（照搬 circleg） |
|---|---|---|
| 控制编排 | 自建 `StandaloneWbc.solve_tasks`，绕过监督层 | circleg `SupervisedControl.solve_mpc` / `solve_wbc`，与 `APP/mujoco_mpc_wbc.py` 同一套 |
| 任务构造 | 自定顺序与分组 | circleg `tasks()` 的集合、平衡 / 运动分组、增益取值 |
| 求解器接入 | 自建工厂 | circleg 的注入方式（零空间默认、`task_qp_resolver_factory`、WLS 同接口） |
| 位姿 / 保护 | 自建 | circleg `ControlSupervisor` 模式机、`WbcFeedbackAccumulator`、同名 guards（attitude / speed / torque / 负法向力 / 非轮接触 / 非有限值） |
| MPC 设置 | 自定 | circleg `MPCSettings()` 默认 + 只换机器人相关物理量（第 7 节） |

**只保留两类“换模型必须改”的差异**：

1. **建模相关**：`height` 任务作用在腿长共模 `L_c`（circleg 用机身高度 `qpos[2]`）；`hip_*` 改名 `leg_*` 并作用在共模 / 差模腿角上；接触块里增加闭链 8 行；权限因子 `permission = 1`（本构型无曲柄死点，与 circleg 默认下限 1.0 等价，激活恒为 1）；
2. **任务顺序**：把 `forward_velocity` 放最后。依据是 circleg 自己在 `docs/MPC-WBC方案.md` §0 / §4.3 的结论（俯仰与 `v_x` 共用同一条“两轮同向滚”通道），本机 M1 实测的欠驱动方向也是 `[v, θ]` 的混合（0.34, 0.94）。本机 7 个任务共用 6 个可实现维度，物理上不可实现的只有 `forward_velocity`，严格分层下必须让它排在最后，否则会把可实现的腿姿态任务饿死。QP / WLS 靠权重竞争，顺序无影响。

**零输入重跑结果**（L0 = 0.42，增益 / 权重未调）：

| 场景 | 后端 | 结果 |
|---|---|---|
| M2 WBC 单独运行 10 s | nullspace / wls | 完成；max\|θ\| = 0.05 mrad，max\|roll\| = 0，位移漂移 ≤ 3.8 mm；fallback / clamp / infeasible 均为 0 |
| M3 MPC + WBC 零速站立 30 s | nullspace / wls | 完成；末 2 s 位移峰峰 0.00 mm；无 fallback / 停机 |
| M3 小推扰 ±0.2 m/s | nullspace / wls | 完成；峰值 s 65 mm、θ 63 mrad，2.2 s 回位 |
| M5 两模型对比（stand30 / push） | nullspace | 两者都完成、都不发散；姿态量级见 `results/m5_model_compare.md` |

两个后端在零输入工况下均无异常，因此按用户要求**没有触发“及时停止”**。

**观察（只记录）**：

- QP 后端在零输入下 0.22 s 就因 `unexpected_body_contact` 停机：`_TASK_QP_WEIGHTS` 里 `leg_differential` 权重是 1.0（circleg 专为曲柄的“保持髋角”设的弱正则），在本机不足以按住两腿的差模腿角，两腿前后错开至 0.66 rad 导致自碰撞。若要切 QP，需要重新考虑这个权重（属于调参，需用户批准）。当前需求（零输入平衡）用零空间法 / WLS 即可，**不切 QP**。
- 8 状态（leg_input）模型在小推扰下用“摆腿”策略：腿角摆到 188 mrad，机体俯仰只有 3.5 mrad；6 状态（held_leg）靠机体前倾，俯仰 63 mrad。两者都能回位。这与本构型“腿长 / 腿角可独立调”的判断一致。
- 离线验证按用户要求不再深入：两个模型的线性化自洽性检查（非线性降阶 ODE 一步 vs 线性预测，扰动缩小 10 倍误差约降 100 倍）保留在 `tools/m5_model_compare.py` 的输出里，作为参考。

### 5.6 同步 circleg 的 ZMP 点控制（2026-10-03）

circleg 在通用 MPC 层新增了 ZMP / 质心偏移参考（`Algorithm/mpc/controller.py` 的
`zmp_reference` / `_apply_zmp_reference`，第 31 次排查），并在 app 的
`_PRESETS["tuned-agile"]` 里开启：`zmp_reference: True, zmp_split: 0.3, reanchor_position_while_braking: True`
（注释写明：默认配置 tuned-fast-accel 未启用，viewer 确认后再决定是否切默认）。

**同步方式**：

- ZMP 的**代码**在算法层，本项目 import 即得，未重写。
- 新增 `preset_settings()`：按 circleg `_build_from_args` 的方式从 `APP._PRESETS` 读 preset
  （weights / hip_angle_limit / use_jerk_limit / tilt_assisted_limits / pitch_feedforward /
  mpc_overrides / tracking_bandwidth），**单一真值来源**，本项目不另存一份参数。
- `SimulationConfig.preset` 默认 `tuned-agile`（含 ZMP）；传 `none` 即回到 circleg viewer 默认
  （tuned-fast-accel，无 ZMP）。viewer 用 `--preset` 切换。
- **本机物理量替换**（ZMP 公式 Δ* = z_com·a/g 里的两个长度）：

| 参数 | circleg | 本机取值 | 来源 |
|---|---|---|---|
| `zmp_com_height` | 0.180 m | **0.3645 m** | 降阶模型 metadata 的实测量心高（接触点以上） |
| `zmp_crank_radius` | 0.06 m（曲柄偏心） | **0.42 m**（标称腿长） | 腿摆 φ_c 使质心相对接触点前后偏移 ≈ L0·sin φ_c |
| `hip_angle_limit` / `drive_hip_angle_limit` | 0.8 / 1.2 rad | **0.50 / 0.50 rad** | M0 实测共模腿角可达范围 |
| `zmp_split` / `zmp_crank_rate_limit` / `reanchor_position_while_braking` | 0.3 / 3.0 / True | 同 circleg | preset 原值 |

- **一个建模限制**：`_apply_zmp_reference` 会写参考的第 6/7 列（腿角），6 状态模型没有这两列，
  所以当 `state_dim < 8` 时自动把 `zmp_reference` 置 False（circleg 也是把 ZMP 与 hip_input
  捆在同一个 preset 里的）。

**零输入验证**（preset = tuned-agile）：

| 模型 | zmp | 30 s + 0.2 m/s 推扰 | max\|θ\| | max\|φ_c−φ0\| | 末位移 | fallback / clamp |
|---|---|---|---|---|---|
| held_leg（6 状态） | 自动关 | 完成 | 21.0 mrad | 3.3 mrad | 0.16 mm | 0 / 0 |
| leg_input（8 状态） | 开 | 完成 | 1.15 mrad | 166.7 mrad | 0.02 mm | 0 / 0 |

注：零输入稳态下 a = 0，Δ* = 0，ZMP 参考不起作用；上表的差别主要来自 8 状态模型本身的摆腿策略
与 preset（S 曲线、倾角上限 20°）差异。ZMP 的实际作用要在有加速度指令的工况下才看得出来，
而那一类工况（M4）按用户要求暂缓。

## 6. 风险

| # | 风险 | 依据 | 对策 |
|---|---|---|---|
| 1 | 闭链 y 向行近冗余导致 QP 秩亏 | 实测 y 行奇异值约 1e-4 | 只取平面内 8 行；M1 逐拍监控条件数 |
| 2 | MuJoCo 闭链是软约束（`solref 0.001`），与 WBC 硬等式不完全一致 | 初始残差 4.5 mm | M0 修正 site；M1 记录残差，评估影响 |
| 3 | 网格轮接触不改就直接上 WBC | 每轮 4 点，轴向不一致 | M0 必须先完成圆柱碰撞体 |
| 3a | 宽轮的逐轮横向摩擦与 WBC 单条集中横向约束不一致（M4 实测） | 半宽 0.011 时，纯偏航两腿前后错开 210 mrad，并触发 jOP 限位 | 仿真：半宽改为 0.001（方案 A，已实施）。实车：风险仍在，对应第 5.4 节的方案 B |
| 4 | 惯量由网格按均匀密度推算 | `PROJECT_ANALYSIS.md` 3.3 | 线性化直接取自 MuJoCo，仿真内部自洽；迁移到实车需要重新标定 |
| 5 | 计算耗时 | nv 从 10 变成 20，QP 变量 20 + 13 + 6 = 39；circleg 每拍新建 OSQP 并做 2 次差分 `mj_forward`（单次约 50 µs） | M2 实测；不达标时先接受非实时运行，不提前优化 |
| 6 | 混用 circleg 机器人专用默认值 | `MPCSettings` 默认 `wheel_torque_limit = 5`、`scrub = 5.59` 等 | 适配层构造 `MPCSettings` 时逐项显式传入（第 7 节），并加断言 |
| 7 | 轴向和符号错误 | 左右腿关节轴相反，左轮执行器 gain = −1 | 对齐 circleg 的 `_hip_axis_sign_value` 做法，用断言守住；M1 做单关节力矩施加测试 |

## 7. 机器人相关参数：来源要重新确定（这是换模型带来的，不是调参）

| 参数 | circleg 值 | GBC486 取值来源 | 当前已知值 |
|---|---|---|---|
| `wheel_radius` | 0.160 | M0 圆柱碰撞体半径 | ≈0.077 |
| `half_track` | 0.120 | `wheel_half_track()` 实测 | 0.25 |
| `wheel_torque_limit` | 5.0 | 轮 `ctrlrange` | 4.0 |
| `hip_torque_limit` | 10.0 | 腿 `ctrlrange` | 20.0 |
| `nominal_normal_force` | 46.6 | m·g / 2 | 71.3 |
| `yaw_moment_limit` | 7.5 | 2 · (τ / R) · half_track | ≈25.9 |
| `scrub_moment_coefficient` | 5.59 | 由 `φ_d` 引起的前后错开推导；对称姿态下为 0 | 待 M3 推导 |
| `pitch_inertia`、`yaw_inertia` | 0.053191 / 实测 | `yaw_inertia_from_model()`（改轮 body 名） | 待 M0 |
| `equilibrium_pitch` | 实测配平 | Builder 配平 | 待 M3 |
| `hip_angle_limit` / `hip_guard` | 0.8 / 1.70 | `φ_c` 的机构可达范围（关节 range 换算） | 待 M0，**需要用户确认** |
| 标称 `L0` | — | 用户指定 | **待用户给出** |

MPC 权重（`QPWeights`）、WBC 增益和任务权重、`MotionLimits` 一律先沿用 circleg 当前默认值作为初值，并在报告中注明。按 `AGENTS.md` §2、§3，调参只在用户明确要求时进行，并且必须按联合扫描的要求执行。

## 8. 待用户决定的事项

1. ~~标称腿长 L0~~：已定 0.42 m。`φ_c` 的安全边界（`leg_guard`）先取机构可达范围，只用于停机保护；正式值待 M3 之后再定。
2. 第 4.4 节严格分层的任务排序是初值，M2 出结果后由用户确认。
3. 零空间法 / WLS 是否切换到 QP：按第 4.4 节的触发条件，由用户判断。

---

## 9. 2026-10-04/05 本轮改动与实测（agent）

本轮目标是“零输入抗扰 + 参数迭代”。范围只在本工作区（`wheel_leg_mujoco/`），
**未改 circleg 的 `Algorithm/`**。以下全部是实际执行过的改动与实测。

### 9.1 算法 / 部署修复（按发现顺序）

| # | 位置 | 问题（证据） | 处理 |
|---|---|---|---|
| 1 | `dynamics.py::sensed_contact` + `simulation.py::run` | 离地判定是“接触力 > 0”的裸阈值。MuJoCo 在几十微米正间隙下就关闭接触，实测轮地间隙仅 **+0.02~0.04 mm** 却被判成离地，连零输入都会误停机 | 接回算法层 `GroundContactEstimator`，`off_duration_idle=0.050`（实测微观卸载低法向力持续 0.008~0.020 s、真实腾空 0.204 s）；**不用** `specific_force_z`/`lift_height`（仿真无加速度计传感器，速度差分噪声过大，实测瞬时卸载时低至 −4.12 m/s²，会把旁证误放行） |
| 2 | `simulation.py::chassis_state` | 腿角反馈用**右腿**角度（`leg_state()[1]`）、腿角速率用**共模**，而降阶模型第 6 状态是共模 `φ_c` | 改为共模 `0.5(φ_R+φ_L) − φ0` |
| 3 | `wbc.py::tasks` | 迁移时丢了 circleg 的 `sample is None` 分支 → 非 GROUND 时腿角目标恒 0、速度目标恒 0（无阻尼）。实测腿角以约 4 rad/s 匀速滑行到限位 | 补回：腿角回标称 + 速度阻尼 |
| 4 | `wbc.py::contact_blocks` | 闭链块声明 `wheels=(0,1)`，单支撑时被 `supervision.py:263` 的块筛选整体剔除 → 单支撑时闭链约束消失 | `wheels` 改为“当前着地的轮子”（腾空时 `(0,1)` 仅占位） |
| 5 | `simulation.py` 预测法向力保护 | 阈值 `-1e-6`（零容差）。实测单拍跳变 +47.3 → −9.1 N 即触发；阶段扫描 41 例不可行里有 38 例是它 | 阈值改 `-max(1e-6, 0.05·名义法向力)`（本机 −3.6 N），且需**连续 3 拍**（`normal_force_duration=0.012`） |
| 6 | `simulation.py` WBC 不可行处理 | `result.feasible=False` 直接停机。实测强扰时动力学残差从 1e-13 跳到 3e-2、轮力矩顶到限幅 | 用户指示：改为状态翻转时警告一次、**继续运行不停机**（发散后的处理方案另行安排） |
| 7 | `wbc.py::tasks` | 7 个任务只有 6 个可实现方向（实算：nv=20、硬约束 13、执行器 6 → 可实现秩 19/20），严格分层下末级被 100% 饿死 | 任务集改为**每条腿独立** `length_R/angle_R/length_L/angle_L` + `pitch/forward_velocity/yaw/roll`；后端默认切 **WLS**（加权竞争，无饿死）；`TASK_WEIGHTS` 按优先级重设 |
| 8 | `MJCF/robot.xml` | 轮执行器 `ctrlrange` ±4 偏小：+0.6 m/s 时两轮顶到 −4.0 N·m 后动力学残差爆掉 | 用户指定改 **±6**；`SimulationConfig` 新增可覆盖的 `wheel_torque_limit`/`leg_torque_limit`（用户要求：限幅纳入优化备选范围） |
| 9 | `simulation.py::_record` | 记录缺 `roll_rate`；腿角/腿长速率用 `np.gradient` 差分（有采样延迟与噪声） | 新增 `roll_rate`/`leg_rate`/`length_rate`；腿角/腿长速率改走雅可比（对应共模行 @ `qvel`） |

### 9.2 评价工具（新增）

- `gbc_mpc_wbc/objective.py`：LQR 型评价（`x'Qx + u'Ru` 的思路，作用在实测闭环轨迹上）。
  每个通道**同时含角度与角速度**：`pitch(+rate)`、`腿角 φ_c(+φ̇_c)`、`位移 s(+v)`、`yaw(+yaw_rate)`、
  `roll(+roll_rate)`、`腿长 L_c(+L̇_c)`。通道分数 `S_i = w_i·(峰值/ref_i + 收敛时间/1s)`，
  收敛口径为“误差进入 `5%·ref` 并持续 1 s”（仅为测量口径，不设硬门槛）。
  tier1（pitch 对）做**硬约束**、不参与权衡；其余通道进 J；另有**弱优先级校验**
  （低优先级改善不得让高优先级恶化超过 10%）。
- `tools/tune_grid.py`：分阶段联合网格扫描（`multiprocessing.Pool`、CSV 逐行落盘可续跑、
  worker 内 BLAS 固定 1 线程、基线通道分全局缓存）。
- 模型变体支持：`_template(variant)` 按变体分别缓存（6 状态模型没有腿角列，ZMP 必须自动关——
  直接传 `mpc_settings` 会绕过 `gbc486_mpc_settings` 的保护，曾因此崩溃）。

### 9.3 实测（关键结论）

- **单线程 BLAS 是复现的前提**：`OMP/MKL/OPENBLAS_NUM_THREADS` 不同时，同一组参数的 T-d
  从 **3.6° 变 44.07°**。本机开环极点 11.6 rad/s，4 秒内足以把 1e-16 放大到宏观量级。
  评价与 viewer 都已在导入 numpy **之前**固定为 1。
- **自查抓到的工具 bug**：`objective.DEFAULTS['pitch_rate']` 曾写 100，而实际 preset 是 **300**，
  导致评估测的不是真正部署的配置。已改正，并在 `_template()` 加了一致性校验（不符即抛错）。
- **模型对比**（同 WLS + 每条腿独立任务，+0.5 m/s 单次推扰）：

  | 模型 | 腿角峰 | 俯仰峰 | 位移峰 | 位移收敛 |
  |---|---|---|---|---|
  | `leg_input`(8 状态) | 388 mrad | 4.5 mrad | 64 mm | 0.76 s |
  | `held_leg`(6 状态) | 11 mrad | 51 mrad | 451 mm | 4.0 s（窗口内未收敛） |

  即：“腿自由”会用摆腿换位移；“腿握住”保住腿角但平衡全压在俯仰上、位移很慢。
- **最终部署**（用户选定）：`leg_input` + `hip_angle_limit=0.15` + `pitch=6e5` + `position=4e4` + `velocity=2667`。
  `hip_angle_limit` 是 MPC 的**硬状态约束**（不是软代价），等于“分给腿的质心偏移预算”，
  超出部分只能由俯仰承担——这正好实现“小角度摆腿 + 小俯仰共同平衡”：

  | `hip_angle_limit` | 腿角峰 | 俯仰峰 | 位移峰 | 位移收敛 | J |
  |---|---|---|---|---|---|
  | 0.10 | 100 mrad | 0.75° | 248 mm | 2.46 s | 6.015 |
  | **0.15** | **150 mrad** | **0.84°** | **167 mm** | **1.78 s** | 4.488 |
  | 0.20 | 200 mrad | 0.77° | 130 mm | 1.24 s | 3.509 |
  | 0.25 | 250 mrad | 0.68° | 107 mm | 1.34 s | 3.277 |

- **ZMP 探针**：受扰（零指令）工况下 `_apply_zmp_reference` 的腿角参考**恒为 0 mrad**。
  原因：规划器 HOLD 进入时执行 `_align(position, yaw, 0.0, 0.0, "hold_capture")`，
  把内部速度与加速度清零且不跟踪实测速度 → `reference.acceleration ≡ 0` → ZMP 没有整形对象。
  按 0.5 m/s 停车、a=1.4 m/s² 计算，ZMP 若被激活会给出：split=0.3 → 腿角 40 mrad + 俯仰 5.73°；
  split=0.9 → 腿角 121 mrad + 俯仰 0.82°。当前部署实测（150 mrad / 0.84°）已相当于 split≈0.9 的分配。

### 9.4 未做 / 待办

1. **F2**：本机两个模型都没有 `accel_capacity` 能力表 → `auto_accel_limits` 退回手填值
   （起步 1.20 / 刹车 1.40 m/s²）；因 tuned-agile 开了 ZMP，俯仰前馈那条分支本来就不走。
2. **F7/F9**：`state.hip_differential` 未设、`WheelMPC.geometry` 用 circleg 默认值——
   已核实当前对控制无影响（scrub 系数为 0；`contact_axis` 只进诊断字段，不进 QP）。
3. `tools/` 下 M0~M5 脚本未重跑（部分硬编码 `model_variant` / `backend`），其 `results/` 可能过期。
4. **ZMP + 规划减速**未验证。建议先用“有指令刹车”工况做 ZMP 开/关对照（规划器在 BRAKE 模式下
   本来就会规划减速、ZMP 自然被激活），确认有效再考虑接到受扰工况；直接改规划器的 HOLD 行为
   属于算法层改动，且有“双参考互相抵消”的风险。
5. 训练集里 **T-e**（40 N×0.2 s + 3 N·m 机身力矩）是最难的一例，会同时顶高俯仰峰与腿角峰。
6. 参数迭代只跑完阶段 1（腿角预算 + pitch + position + velocity）；阶段 2/3 与边界扩展未跑。

---

## 10. 2026-10-05：设计层排查与 MPC-WBC 连接修复（agent）

本轮按用户要求**不跑测试**，只做设计层排查：读 `.md` 参考文档 + 逐环节核对 MPC→WBC→MPC 链路。
结论：找出 7 条问题，其中 2 条是结构性的，**均已修复**；但修复后在当前配置下**不改变任何现有结果**（见 10.3）。

### 10.1 排查发现（按严重度）

**F-1 [严重] WBC→MPC 反馈回路完全断开（两个独立原因）**

1. `simulation.py` 主循环在 `result is not None` 分支只做计数（`wbc_fallback` / `wbc_clip` / `wbc_infeasible` /
   `exhausted` / `scaled`），**从未调用 `self.feedback.observe(result, ...)`**；而
   circleg `APP/mujoco_mpc_wbc.py:593` 在同一位置调用它。`self.feedback` 建了、每拍 `consume` 了，但从不喂数据。
2. 即使补上也不生效：`WeightedSaturationResolver` 返回 `outcomes=()`（`Algorithm/wbc/weighted.py:95`），
   而 `WbcFeedbackAccumulator.observe`（`Algorithm/supervision.py:161-167`）在 `present` 为空时直接 `return`。
   单级 QP 则产生 `outcomes`（`task_qp.py:193`）。

后果链（`Algorithm/mpc/planner.py`）：`fb.saturated` 恒 False、`motion_residual` 恒 0 →
`:379 blocked = fb.saturated and |residual| > residual_limit` 恒 False → `_saturation_time` 恒 0 →
`restricted` 永不因饱和触发 → `capability` 永不降级 → **MPC 永远不知道 WBC 做不做得到**。

**F-2 [严重] WBC 后端选了 `wls`，不是设计路径**

- circleg 默认 `wbc_backend = "qp"`（`APP/mujoco_mpc_wbc.py:87`），命令行只允许 `("nullspace", "qp")`；
  `Algorithm/wbc/weighted.py` 文件头自称"**对照用途**"。设计文档（`docs/算法部署方案与代码地图.md` §3.1）
  的生产路径是"单级 QP 全身控制"。
- 本方案 §4.4 自己列过 WLS 的缺陷与升级触发条件（clamp 频繁 / 负法向力停机 / 末级饿死）；
  三条实测均命中，但当时切成的是 `wls` 而非 `qp`。
- 结构差异：QP 的 `z=[q̈,f,τ]` 全为决策变量、力矩上下限与 `f_n≥0` 是硬约束；
  WLS 只有 `q̈` 是决策变量，`τ`/`f` 由 `q̈` 反解，越限后 `clamp`，**接触力没有分配自由度**。

**F-3 [中] WBC 采样用错时间参数**

circleg：`sample_at(robot.time + config.preview_lead)`（`APP:578-579`，`preview_lead` 是 `SimulationConfig` 字段）；
本项目：`sample_at(d.time + self.mpc.settings.prediction_lead)`。二者含义不同（后者是 MPC 内部前推反馈状态，
`controller.py:585`）。且 `SimulationConfig` 没有 `preview_lead` 字段、`preset_settings()` 也不读 preset 的该键。

**F-4 [中] preset 的 WBC 任务权重没有传播**

circleg 在 `_build_from_args` 把 preset 的 `hip_tracking` 写进 `_TASK_QP_WEIGHTS`（`APP:1044-1045`，值 200）；
本项目的 `TASK_WEIGHTS` 是硬编码。**未修复**，原因见 10.4。

**F-5 [设计] 降阶模型的工作点与 WBC 实际行为不一致**

ZJU 评审（`docs/浙大串联腿文档参考与移植评审.md` §2.E）已点名："`ReducedModel.from_mujoco` 的源码明确注明
**测得的是髋自由工作条件，实际 WBC 下髋行为仍需匹配**"，建议"先定义本机构的静态工作点、接触条件与髋任务，
再在同一条件下建立降阶模型"。该建议至今未执行。实测后果（同一 MuJoCo）：髋自由 `∂v̇/∂f_R = −0.0119`，
髋握死 `+0.1024`，差 9 倍且反号。WBC 实际做的是**位置跟踪腿角**，两种模型都不对应。
**用户 2026-10-05 决定：保留 `leg_input` 8 状态，不做阻尼建模** → 本项不实施。

**F-6 [结论可靠性] ZMP 结论不成立**

本方案 §5.6 自己写明："零输入稳态下 a = 0，Δ* = 0，ZMP 参考不起作用；ZMP 的实际作用要在有加速度指令的
工况下才看得出来"（M4 暂缓）。所以"接 ZMP 没有提升"是在 ZMP 按设计无法起作用的工况下得到的，不能作为结论。

**F-7 [待重新评估] 当初否决 QP 的理由可能已失效**

§5.5 记录 QP 在零输入 0.22 s 因 `unexpected_body_contact` 停机，原因是 `leg_differential` 权重 1.0 不足以
按住两腿差模。但任务集后来已改成**每条腿独立**（`angle_R` / `angle_L`），`leg_differential` 已不在实际任务里。
**用户 2026-10-05 决定：不考虑 QP。**

### 10.2 已修复内容（仅限本工作区，未改 `Algorithm/`）

| 文件 | 改动 |
|---|---|
| `gbc_mpc_wbc/wbc_limited.py`（新建） | 限幅自洽的 WLS 解析器 |
| `gbc_mpc_wbc/wbc.py` | `resolver_factory` 的 `wls` 分支改指向 `LimitedWeightedResolver` |
| `gbc_mpc_wbc/simulation.py` | ① 补 `self.feedback.observe(result, self.dyn.limits)`；② 新增 `SimulationConfig.preview_lead`，WBC 采样改用它 |

**新解析器的原理**：利用 `τ` 对 `q̈` 的仿射性。记硬约束解 `q̈_h`、任务修正 `Δ = N x*`（`Δ ∈ null(A)`）：

```
q̈(s) = q̈_h + s·Δ            对任意 s∈[0,1] 仍满足全部硬约束
τ(s) = τ_h + s·(τ* − τ_h)     仿射
s_max = 一维区间求交（复用 Algorithm.wbc.sns.maximum_scale）
τ = recover_torque(q̈(s_max), h)   ← 由构造保证与 q̈ 严格自洽，永不 clamp
```

同时补齐 `outcomes`（逐任务残差）与 `scaling[name] = s_max`（让反馈的 `curtailed` 生效）。

**已知代价（必须明说）**：`s_max` 是对整个任务修正的**等比例**缩放，不是按优先级逐项牺牲。
方向与配比仍由权重决定，只有幅值被压。这是"保守但自洽"，不承诺最优。

### 10.3 自检与实测（重点：修复是潜伏的）

自检（`AGENTS.md` §4 四项）：

| 检查 | 结果 |
|---|---|
| 导入/语法 | 通过 |
| `τ` 与 `q̈` 自洽（核心不变量） | **max\|τ − recover_torque(q̈)\| = 0.000e+00** |
| 接触残差 | 2.875e-14 |
| `τ` 越限量 / `clip_applied` | 0.000e+00 / **0 次**（旧实现越限必为 True） |
| `outcomes` 条数 | 8（旧实现恒为 **0**） |
| 重复评价逐位一致 | **通过**（解析器内部量与 1750 条记录轨迹完全相同） |

**但限幅在现行配置下从未被激活**（推扰 +1.5 / +2.0 / +2.5 / +3.0 m/s 扫描）：

| 推扰 m/s | min 步长 s | clip | infeasible | 轮占用峰 | 腿占用峰 |
|---|---|---|---|---|---|
| 1.5 | 1.000 | 0 | 0 | 0.763 | 0.535 |
| 2.0 | 1.000 | 0 | 0 | 0.788 | 0.537 |
| 2.5 | 1.000 | 0 | 0 | 0.814 | 0.538 |
| 3.0 | 1.000 | 0 | 0 | 0.840 | 0.636（触速度保护） |

所以 P0-2 在现行配置下**不改变任何结果**（轮限幅还是 ±4 时确实触发过：+1.0 m/s 轮占用 1.00、
`wbc_infeasible` 7 拍）。P0-1 是唯一有可观测变化的一处：`motion_residual` 从恒 0 变成实测峰值 **0.364**，
但触发 planner 降级还需 `blocked = fb.saturated and |residual| > 0.50`——残差 0.364 < 0.50，且 `saturated`
仍为 False（无执行器顶限、无步长缩放）。

**后端语义差异（已查明，未改）**：circleg 零空间后端下 `forward_velocity` 级被饿死 → `scaling=0` →
`curtailed=True` → `saturated` 恒为真；WLS 下 `scaling` 是限幅步长，不因任务饿死而置真。两者最终行为相同
（残差都在阈值下），但语义不同。

### 10.4 未做事项

1. **F-4 的 `hip_tracking` 传播**：`preset_settings` 在 `_settings()` 内部被调用，而调参脚本的
   `_apply_task_weights()` 在更早处执行，在那里写 `TASK_WEIGHTS` 会**静默覆盖调参值**；
   且本项目的映射是 1 个共模任务 → 2 个独立腿任务，一对一映射无定义。需用户先定优先级方案。
2. **`preview_lead` 的 preset 传播**：字段已加、采样已改用，`preset_settings` 的 `entry` 本就带该键；
   所有 circleg preset 该值均为 0.0，故无行为变化。若要让构造点按 preset 自动设置，需改各调用点。
3. **F-5 建模工作点**：用户决定保留 `leg_input` 8 状态、不做阻尼建模 → 不实施。
   **后果**：腿角仍恒被顶到 `hip_angle_limit`（36 组扫描全部 150.1 mrad），该参数是唯一可操作旋钮。
4. **F-7 切 QP**：用户决定不考虑。

### 10.5 对"WBC 只做优先级控制"这一诊断的修正

排查后修正：**该现象不是 F-1 / F-2 造成的**（两者在现行配置下都未激活）。真正原因是：

1. **结构原因（circleg 的设计）**：WBC 的任务目标由 MPC 前馈 `sample.acceleration` 主导，跟踪误差 ~0
   （实测 `kp·误差` / `kd·误差` 只有 0.00~0.31，前馈项是 130 / −58），所以 WBC 自身的控制动作本来就很小。
   这是"MPC 是唯一规划者"的直接后果，circleg 源码注释也把它写成刻意设计。
2. **模型原因（F-5）**：降阶模型在"髋自由"工作点测得，轮子前进权限 `−0.0119`（反号），
   所以 MPC 让腿承担全部纵向任务 → 腿角恒顶到 `hip_angle_limit`。**用户已决定不动模型** → 此路封住。

因此，要"进一步压缩腿摆角"，在不动模型的前提下**只能降 `hip_angle_limit`**，代价是位移变差
（§9.3 的表：0.10 → 100 mrad 但 248 mm / 2.46 s；0.15 → 150 mrad / 167 mm / 1.39 s）。

---

## 11. 2026-10-05：腿角预算 × 位置权重联合扫描与部署更新（agent）

### 11.1 扫描（grid10 阶段 5）

**搜索空间**：`hip_angle_limit` ∈ {0.100, 0.125, 0.150, 0.175, 0.200} × `position` ∈ {2.0e4, 4.0e4, 8.0e4}，共 15 组。
按 `AGENTS.md` §3.2（禁止单项微调、耦合参数必须联合搜索）做成二维：`hip_angle_limit` 是摆幅硬上限，
`position` 决定 MPC 对腿的需求强度，二者共同决定"摆幅 vs 位移"的取舍面。

**评价**：新 J（`W_EFFORT=0`，不用出力作为比较标准）+ pitch 硬约束（峰 ≤1.0°、收敛 ≤0.5 s）+ 可行性闸门。
与 §9.3 的旧表口径不同（旧表含出力项、且当时腿角参与硬约束），所以数字不能逐项对比，**以本节为准**。

产物：`artifacts/tuning/grid10/stage5.csv`、`stage5_top10.md`（15 组，8 可行）。

| `hip_angle_limit` | `position` | 腿角峰 mrad | 腿角收敛 s | 俯仰峰 deg | 俯仰收敛 s | 位移峰 mm | 位移收敛 s |
|---|---|---|---|---|---|---|---|
| 0.100 | 20000 | 100.1 | 1.784 | 0.841 | 0.344 | 243.9 | 3.052 |
| 0.125 | 20000 | 125.2 | 1.544 | 0.795 | 0.252 | 202.0 | 2.532 |
| **0.125 | 40000** | **125.1** | **1.552** | **0.769** | **0.432** | **192.4** | **1.980** |
| 0.150 | 20000 | 150.2 | 1.352 | 0.691 | 0.248 | 174.6 | 2.328 |
| 0.150 | 40000（上一版） | 150.1 | 1.388 | 0.845 | 0.336 | 167.0 | 1.780 |
| 0.175 | 20000 | 175.2 | 1.196 | 0.578 | 0.220 | 149.8 | 2.148 |
| 0.200 | 20000 | 200.1 | 1.112 | 0.654 | 0.228 | 136.5 | 2.044 |
| 0.200 | 40000 | 200.1 | 1.140 | 0.722 | 0.220 | 129.5 | 1.516 |

**7 组不可行的原因**（全部是 tier1 pitch 硬约束或腿力矩占用闸门）：

| 配置 | 失败原因 |
|---|---|
| 0.100 / 40000 | tier1: pitch 收敛 0.636 s > 0.5 |
| 0.100 / 80000 | tier1: pitch 峰 1.042° + 收敛 0.832 s |
| 0.125 / 80000 | tier1: pitch 峰 1.022° + 收敛 0.660 s |
| 0.150 / 80000 | tier1: pitch 峰 1.009° + 收敛 0.508 s |
| 0.175 / 40000 | T-c: 腿力矩占用 0.812 > 0.8 |
| 0.175 / 80000 | tier1: pitch 峰 1.052° |
| 0.200 / 80000 | tier1: pitch 峰 1.111° |

### 11.2 三条结构性结论

1. **腿角峰恒等于 `hip_angle_limit`**（100.1 / 125.1 / 150.1 / 175.2 / 200.1，与 `position` 完全无关）。
   再次证实：摆幅由 MPC 的**硬状态约束**决定，权重调不动；**压摆幅只有降预算一条路**。
2. **`position` 提到 80000 反而全面变差**：5 个预算下全部因俯仰峰/收敛超限而不可行。
   所以 `position` 的有用上界在 40000 与 80000 之间，当前值 40000 已在有效区。
3. **J 的排序与"压摆幅"目标相反**：J 最小的是 0.200/40000（J=3.6846，因为位移最好），
   0.150/40000 排第 4（J=4.4592）。所以本轮**不能用 J 选解**，必须按目标在取舍面上挑。

**边界覆盖**：`position` 的最优在 40000（内部，20000 与 80000 两侧都更差）→ 已覆盖；
`hip_angle_limit` 下界 0.100 已触到可行性边界（0.100/40000 不可行），再往下只会更差；
上界 0.200 仍可行，只有需要"更大摆幅换更小位移"时才需外扩。

### 11.3 部署更新（用户 2026-10-05 选定）

`gbc_mpc_wbc/objective.py::TUNED` 的 `hip_angle_limit` 由 **0.15 改为 0.125**（`position` 仍为 40000）：

```python
TUNED = dict(DEFAULTS, position=40000.0, velocity=2666.7, pitch=600000.0, hip_angle_limit=0.125)
```

选它的理由：摆幅明显下降（150.1 → 125.1 mrad，−17%）而位移只小幅退化（峰 167 → 192 mm，+15%；
收敛 1.78 → 1.98 s，+11%）；俯仰峰反而略好（0.845° → 0.769°），俯仰收敛 0.432 s 仍在 0.5 s 硬约束内。
再往下降到 0.10 时代价不成比例（位移峰 +46%、收敛 +71%）。

**`objective.DEFAULTS` 未动**（仍为 `hip_angle_limit=0.15`）：它是评价函数的基线与优先罚项的参照，
改动会使 `artifacts/tuning/baseline_channels_v2.json` 失效，属另一件事。

### 11.4 viewer 实测（0.100/20000，用户操作，`session_20261005_181646`）

会话 16.8 s、4210 拍。先确认测试性质：**`cmd_v` 全程为 0、`cmd_w` 0 拍**，planner hold 4140 / brake 45 / reset 25
→ 确实是零指令测试，不是键盘驱动。

| 指标 | 实测 | 说明 |
|---|---|---|
| 共模腿角峰 | **120.4 mrad**（6.90°） | 预算 100 mrad；5.6% 采样超 100，超出部分是瞬态过冲 |
| 俯仰峰 | **4.98°** | 发生于 t=11.18，该拍实测速度 0.902 m/s |
| 横滚峰 | 3.74° | t=8.74 |
| 位移峰 | 580 mm | t=14.95 |
| 结束时 | 位移 37.9 mm、速度 −0.077 m/s | **未完全回位** |
| fallback / 动态残差 | **0 拍 / 8.5e-13** | 新解析器健康，无越限裁剪 |

**方法上的限制（必须注明）**：手推幅度与方向不受控（用 MuJoCo 的 ctrl+拖拽直接施加外力），
实测速度达 0.902 m/s、并带出 3.74° 横滚，而训练集最大是 +0.6 m/s 单向推扰、**完全没有横滚激励**。
所以本次成绩**不能与 headless 的 0.841° / 244 mm 逐项对比**，也不能用来判定 0.100/20000 的优劣。
能得到的两条：① 摆幅预算生效且工作正常（腿角被钉在 ±100 mrad）；
② 更强的推扰下俯仰冲到 4.98°，与 headless 预测的"位移退化"方向一致但幅度更大。

### 11.5 `tools/view.py` 新增本次运行有效的覆盖参数

`--tuned` 原本直接读 `objective.TUNED`、无单项覆盖入口，为跑本轮候选新增：

- `--hip-angle-limit <rad>`：覆盖 `TUNED['hip_angle_limit']`；
- `--position-weight <val>`：覆盖 `TUNED['position']`。

二者只作用于本次运行的参数副本，**不改 `objective.TUNED`**（已验：运行后模块值仍为部署值）。

### 11.6 待办

1. viewer 里做**受控扰动**（定量速度脉冲）后，再把 0.125/40000 与 0.150/40000 做同扰动对比；
   当前手推不可控，无法支撑"哪套更好"的结论。
2. 训练集目前**不含横滚/侧向激励**，而 viewer 实测已出现 3.74° 横滚——是否新增横滚工况由用户定。
3. 这轮修改（`wbc_limited.py`、`feedback.observe`、`preview_lead`、view.py 覆盖参数、TUNED 换档）均**未提交 git**。

---

## 12. 2026-10-05：参数档位总表（用户要求保留，以后可能回用）

本次会话反复换档，用户要求**把每一档都留在文档里**。以下每档都给出「改了什么 / 实测指标 / 适用条件」，
便于以后按需求直接取用。**当前部署 = 保持段档位 2 + 驱动段档位 C + 速度上限 1.5。**

### 12.0 一条贯穿全部档位的物理约束

```
质心偏移  Delta = z_com*a/g = r_eff*sin(phi_leg) + z_com*sin(theta_tilt)
⇒ a = g*(r_eff*sin(phi) + z*sin(theta))/z
本机 r_eff = 0.2132 m（实测 d(CoM_x−接触_x)/d(delta_phi)，**不是标称腿长 0.28**）
     z_com = 0.2644 m，r_eff/z = 0.8064
⇒ theta=0 时: phi = asin(z*a/(g*r_eff)) = asin(0.1264*a)
```

**加速度需要多少质心偏移是固定的，档位之间只是在选"这份偏移由腿还是由俯仰承担"。**
想「倾角小 + 摆幅小」就只能压低 a；想「a 大 + 倾角小」就得大摆腿；想「a 大 + 摆幅小」就得前倾。

### 12.1 保持段（零输入抗扰）档位

判据：tier1 = pitch 峰 ≤1.0°、pitch 收敛 ≤0.5 s；训练集 T-a..T-f；位移权重 `position=40000`（grid10 二维扫描确定的最优点）。

| 档位 | `hip_angle_limit` | 位移峰 | 位移收敛 | 腿角峰 | 俯仰峰 | 俯仰收敛 | 可行 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 档位 1 | **0.125** | 192.4 mm | 1.98 s | 125 mrad | 0.769° | 0.432 s | 是 | 2026-10-05 上半天部署；grid10 stage5 选定 |
| — | 0.150 | 167.0 | 1.78 | 150 | 0.845° | 0.336 | 是 | 中间点 |
| — | 0.200 | 129.5 | 1.52 | 200 | 0.722° | 0.220 | 是 | |
| **档位 2（当前）** | **0.250** | **108.5** | **1.36** | 250 | 0.704° | 0.228 | 是 | **位移最小且可行**；用户 2026-10-05 选定 |
| — | 0.300 | — | — | — | 4.27° | — | **否** | 俯仰峰超判据 |
| — | 0.400 | — | — | — | 4.2° | — | **否** | 同上 |

其余参数（两档相同）：`position=40000`、`velocity=2666.7`、`pitch=600000`、`pitch_rate=300`、
`hip_angle=50`、`hip_rate=3`、`yaw=60`、`yaw_rate=8`、`terminal_scale=5`。

### 12.2 驱动段（有速度指令）档位

三档都基于同一个 `zmp_split` 机制，差别在「腿让多少 / 俯仰让多少」以及加速度上限。

| 档位 | `zmp_split` | `tilt_assisted` | 驱动腿角限 | 加速度限值 | 俯仰（巡航/加减速峰） | 腿角峰 | 适用 |
|---|---|---|---|---|---|---|---|
| **A** | 1.0 | False | 0.50 | 3.40 / 3.80 | 0.000° / 0.259°（2.5 m/s） | 500 mrad | 「倾角≈0 + 加速度大」；代价是满速时摆腿到预算顶 |
| **B** | 0.39 | True（15° 能力表） | 0.20 | 3.01 / 3.37 | 0.003° / 12.6° | 200 mrad | 「摆幅小 + 加速度大」；代价是加减速前倾 12.6° |
| **C（当前）** | 1.0 | False | 0.20 | **1.35 / 1.51** | **0.000° / 0.04°** | 200 mrad | 「姿态优先」；代价是加速度上限降到 1.35 |

档位 C 的加速度限值来源：θ=0 能力表在腿角 0.20 rad 处的能力 1.59 m/s² × `accel_fraction_up/down`(0.85/0.95)。
档位 A/B 的限值由 `accel_capacity_tilted`（15°/20° 倾角能力表）反查，同样乘 0.85/0.95。

### 12.3 速度上限档位

| 档位 | `max_speed` | `speed_limit` | `speed_guard` | 键盘步长 | 1.5 m/s 刹车距离 |
|---|---|---|---|---|---|
| 初值（circleg） | 2.5 | 3.0 | 3.0 | 2.5 | — |
| 5.0（已作废） | 5.0 | 6.0 | 6.0 | 5.0 | 4.19 m（实测） |
| **当前** | **1.5** | **3.0** | **3.0** | **1.5** | 约 0.74 m |

**为什么回到 1.5**：1.5 m/s 下制动时间 0.45 s < MPC 视界 0.5 s，**"视界短于刹车时间"的问题自动消失**，
不需要再做稀疏 QP；同时行程与场地要求都降到可接受范围。

### 12.4 未采用但已验证的两条（保留备查）

1. **MPC 加长视界：不可行（已回退）**。凝聚型 QP 在 N≥50 时 OSQP 不收敛（日志 `maximum iterations reached`），
   每拍走兜底通路：N=40 → 俯仰 1.48°；N=50 → 5.45°；N=75 → 44.9° 并被保护停机。
   代价标度：`dt=0.02/N=25` 1.45 ms、`N=75` 7.32 ms、`N=100` 12.66 ms、`dt=0.01/N=200` **146 ms**。
   **代价由 N 主导**，想"长视界 + 高频"必须先换成稀疏 QP 公式。`Algorithm/timing.py` 里的
   "MPC 周期须是 WBC 周期整数倍"约束已按用户批准放开（允许异步）。
2. **去掉规划器平滑：不可行**。只把规划器参考换成阶跃（MPC 约束与代价不变）后：
   俯仰 12.6° → 16~20°，超调变大，5 m/s 时轮力矩顶满；纯阶跃在 **0.19 s** 触发负法向力保护停机。
   结论：规划器仍是必要的"平滑 + 能力预算 + 补视界"层（不再是 LQR 时代的"可行性必需"）。

### 12.5 已确认的物理事实（供以后排查引用）

1. **轮地摩擦真实存在**：接触对 `friction=[1.0, 1.0, 0.005]`、`condim=3`（只有滑动摩擦，**无滚动阻力**）。
   稳态刹车时滑移仅 **1.1%**，轮力矩仅用到 1.8~3.5 N·m（限值 6）→ **减速慢不是摩擦或力矩限制**。
2. **轮毂主动减速本来就在用**：刹车时轮力矩 −1.8 N·m → 47 N → 3.2 m/s²，与同段实测减速完全吻合。
   "质心偏移 vs 轮毂减速"不是两种策略，是力矩平衡的两面。
3. **本机共模腿角符号与 circleg 相反**（正 δφ_c = 质心后移 = 减速）。已按"改定义"方式统一：
   `δφ_c := −(极角 − φ0)`，并在 `build()` 里加了 `∂v̇/∂φ_c > 0` 断言防止再次漂移。
   **注意 `derivative` 的自由坐标行也必须同时取负**，只翻位形会造成模型非镜像（极点 11.614 → 2.309）。
4. **viewer 时间轴被拉长**（用户重要反馈）：实测 headless 0.95 倍实时，viewer **9.16 倍**，
   已通过"`viewer.sync()` 改按 60 Hz 刷新"降到 **3.56 倍**。元凶是 `base_link` 单个网格 91441 顶点。
   **看 viewer 时的时间要除以 3.56 才是真实物理时间**，headless 数据才是真值。

---

## 13. viewer 时间轴与"离线录像"方案（2026-10-05 用户定案）

### 13.1 问题与实测链条

**问题**（用户反馈）：viewer 里看到的时间与真实不符，导致对控制效果产生认知偏差——"看起来收敛慢/效果差"
有相当一部分是显示器上的时间被拉长造成的。

| 配置 | viewer 实时倍率 |
|---|---|
| 初始（每物理步调用 `viewer.sync()`） | **9.16 倍** |
| `sync()` 改按仿真时间 60 Hz 刷新 | **3.56 倍** |
| 再关阴影/反射/skybox/接触力可视化 | **3.45 倍** |
| headless 对照（同物理、同控制、无渲染） | **0.95 倍** |

**已实测排除的原因**：

1. Windows `time.sleep` 粒度 —— 实测放大 1.0~1.1 倍，**不是它**；
2. MuJoCo 物理参数 —— headless 每步 0.95 ms（物理步长 1 ms），**物理本身是实时的**；
3. 控制器开销 —— 每仿真秒仅 0.51 s（MPC 48.75 次 × 3.4 ms + WBC 250 次 × 1.37 ms）。

**剩余 2.5 ms/步的来源**：viewer 的渲染线程与仿真线程**同进程**，抢 CPU/GIL（`sync()` 需要等渲染帧）。
**不是 GPU 负担**——`base_link` 的 9.1 万顶点对 GPU 微不足道（关阴影只从 3.56 降到 3.45 即为旁证）。

### 13.2 定案

**用户决定：不再优化 viewer 时间轴。** 进一步优化只有两条路，都不符合当前约束：

- 换低模视觉网格 —— **属于改模型资产，用户明确不允许**；
- 把 viewer 拆到独立进程 —— 改动大、收益不确定。

**使用约定（重要）**：

```
viewer 里的时间 ÷ 3.45 = 真实物理时间
判断控制效果一律以 headless 数据为准；viewer 只用来看趋势与手感。
```

### 13.3 今后"看效果"的方案：离线录制视频（用户采纳，具体需求后续提出）

- **原理**：仿真时间与渲染耗时**解耦**。每 `1/50 s` 仿真时间取一帧、按 50 fps 编码，
  则**播放速度精确等于物理时间**，与渲染花了多久无关；帧率可任意设（想慢放就按更低帧率编码同样的帧）。
- **禁止用截屏录 viewer 窗口**：那录到的是被拉长 3.45 倍的画面本身。
- **环境现状**：`mujoco.Renderer`（离屏渲染）✓、Pillow 12.3 ✓；
  `ffmpeg` / `imageio` / `cv2` **均缺失**。
  - 不加新依赖：只能出**动画 GIF**（Pillow 直接存），适合 3~5 s、低分辨率短片（体积大、256 色）；
  - 出 **MP4** 需安装 `imageio-ffmpeg`（**新依赖，需用户同意，尚未安装**）。
- **录制口径（两种）**：**实时** = 50 fps 编码（看整体性能）；**慢放 2~4 倍** = 25 fps 编码 50 fps 的帧（看瞬态细节）。
