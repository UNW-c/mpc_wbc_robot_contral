# 轮足 / 足式 MPC + WBC 控制方案参考资料库

本目录为「轮腿（轮足）机器人 MPC + WBC 控制方案」专题资料收集结果，供本项目架构设计与算法排查参考。

采集日期：2026-09-24
采集方式：本地 SOCKS5 代理 `socks5h://127.0.0.1:10808`（v2rayN / xray 监听端口）

---

## 一、结论先行（检索后的判断）

1. **不存在完整开源的「轮足 MPC + WBC」全身控制框架。** 轮足方向的开源生态几乎全部集中在强化学习（Wheel-Legged-Gym、Go2w-RL、Tron1 等）。
2. **轮足的经典 MPC 路线集中在 ETH（Bjelonic 一系）与 KAIST（Tachyon 系列）**，其中 ETH 的 OCS2 仅部分公开轮式变体，Swiss-Mile / ANYmal-on-wheels 的上层控制器未完全开源。
3. **最可落地的借鉴路径是「足式 MPC+WBC 工程基座 + 轮足论文的滚动约束建模」**：
   - 架构照抄 `qiayuanl/legged_control`（OCS2 NMPC 100 Hz + WBC QP 250 Hz + 状态估计 + sim2real）；
   - 把接触模型从「点接触」换成「轮子滚动约束（纵向速度 = 轮径 × 轮速，横向零滑移）」；
   - 建模细节参考 `2010.06322`（ETH 全身 MPC + 在线步态生成）与 `2020_RAL_TO_WheeledLeggedQuadruped_ChallengingTerrain`。
4. **轮足特有的三个坑，对应的论文已在库中**：
   - 滚动约束与腿部动力学的耦合 → `2010.06322`、`2025_TRO_NonsmoothTO_WheeledBalancingRobots`
   - 混合驱动/步行切换（driving-stepping）的状态估计 → `2211.11390`
   - 轮子带来的非最小相位平衡问题 → `2005.11431`（Ascento LQR 辅助 WBC）

---

## 二、目录结构

```
docs/references/mpc-wbc/
├── INDEX.md          本文件
├── papers/           论文 PDF（45 篇，约 217 MB）
│   ├── A/            轮足 / 轮腿 直接对口（23 篇）
│   ├── B/            足式 MPC+WBC 方法论基座（17 篇，可迁移）
│   └── C/            工程实现 / 求解器 / 评测 / RL 对照（5 篇）
├── repos/            开源仓库浅克隆（13 个，约 2.0 GB）
│   ├── A/            轮足直接相关
│   ├── B/            足式 MPC+WBC 工程基座
│   └── C/            WBC / QP 底层库
├── reports/          中文调研资料（含 2 份 PPT 综述）
└── _tmp/             检索与下载脚本（可重复执行以增量补齐）
```

`_tmp/` 下三个脚本均可重复运行，已存在的文件会自动跳过：

| 脚本 | 作用 |
|---|---|
| `search.py` | 按十个主题关键词批量检索 arXiv，输出候选清单 |
| `download_papers.py` | 按策展清单下载 arXiv PDF，校验 `%PDF-` magic |
| `clone_repos.py` | 浅克隆（`--depth 1`）参考仓库 |

---

## 三、A 类：轮足 / 轮腿 直接对口（优先阅读）

### A1. 必读核心（MPC + WBC 架构直接相关）

| 文件 | 出处 | 价值 |
|---|---|---|
| `2010.06322_ETH_WholeBodyMPC_GaitSeq_WheeledLegged.pdf` | ETH, IROS 2021 | **轮足全身 MPC 的标杆**。单一优化同时求解轮子与躯干运动，把轮子建模为移动接触点，在线生成非周期步态，Cost of Transport 最多降 85% |
| `1909.07193_ETH_RollingInTheDeep_HybridLocomotion.pdf` | ETH, RAL 2020 | 步行-驱动混合运动在线轨迹优化，MPC 求解 + **分层全身控制器**跟踪力矩，DARPA 地下挑战赛实战验证 |
| `1809.03557_ETH_KeepRollin_WheeledQuadruped.pdf` | ETH, RAL 2019 | 轮式四足全身运动控制与规划，Bjelonic 系列的起点 |
| `2022_IJRR_Bjelonic_OfflineMotionLibraries_OnlineMPC.pdf` | ETH, IJRR 2022 | **离线运动库 + 在线 MPC 的完整框架综述**。离线轨迹作为 MPC 代价项（而非约束/初值），ANYmal 及其轮式版本验证，含急转 180°、钻桌、0.2 m 台阶 |
| `2312.08668_Tachyon3_FullCentroidalNMPC_TelescopicWheelLeg.pdf` | KAIST, 2023 | 伸缩式轮腿机器人**全质心 NMPC**，非 ETH 路线的另一套完整实现 |

### A2. 轮足轨迹优化与约束建模

| 文件 | 出处 | 价值 |
|---|---|---|
| `2019_RAL_TO_WheeledLeggedQuadruped_LinearizedZMP.pdf` | RAL 2019 | 线性化 ZMP 约束下的轮足轨迹优化 |
| `2020_RAL_TO_WheeledLeggedQuadruped_ChallengingTerrain.pdf` | RAL 2020 | 复杂地形下的轮足轨迹优化，轮地接触建模细节 |
| `2020_RAL_ComputationalFramework_SkilledLeggedWheeled.pdf` | RAL 2020 | 腿轮机器人设计的计算框架 |
| `2025_TRO_NonsmoothTO_WheeledBalancingRobots.pdf` | T-RO 2025 | **非光滑轨迹优化**，显式处理接触切换与碰撞冲击、牵引力极限、执行器约束；Ascento 上实现走楼梯、跳台阶、上坡 |
| `2109.09934_WheelLegged_HighObstacle_BalancingPoseOpt.pdf` | arXiv 2022 | 轮足越障时的平衡控制与位姿优化 |

### A3. 轮式双足 / 轮腿平衡控制

| 文件 | 出处 | 价值 |
|---|---|---|
| `2005.11431_Ascento_LQR_Assisted_WBC_KinematicLoops.pdf` | ETH, RAL 2020 | **轮腿 WBC 的关键参考**：把 LQR 反馈律作为一个运动任务写进 WBC 的 QP 里，解决非最小相位平衡；400 Hz + OSQP。闭链（四杆机构）处理方式值得借鉴 |
| `2005.11435_Ascento_TwoWheeledJumpingRobot.pdf` | ETH, ICRA 2019 | Ascento 平台与倒立摆建模 |
| `2003.03678_WheeledBiped_OnlineDynamicPlanning.pdf` | 2020 | 轮式双足在线动态规划与控制 |
| `2407.21500_DIABLO_6DoF_WheeledBipedal_DirectDrive.pdf` | 中山大学, IROS 2024 | 全直驱 6-DoF 轮式双足；简化二阶倒立摆 → 状态空间，LQR 平衡 + 偏航/劈叉角/高度/横滚控制，未来工作指向 WBC |
| `2511.06397_WheeledBipedal6DoF_WBC_TerrainEstimation.pdf` | 2025 | 6-DoF 轮式双足**全身控制 + 地形估计** |
| `2024_TMech_CompliantMotionControl_WheelLeggedHumanoid.pdf` | TMech 2024 | 轮腿人形机器人在崎岖地面的柔顺运动控制 |
| `2403.10948_WheeledHumanoid_UnknownPayload.pdf` | 2024 | 负载未知的轮式人形控制，平衡点估计 |
| `2012.06156_WheeledBipedal_UnderactuatedJumping.pdf` | 2020 | 欠驱动轮式双足跳跃的运动规划与控制 |
| `2605.03302_WheeledBipedal_JumpHeight_OptimalTorque.pdf` | 2026 | 轮式双足跳跃的高度控制与最优力矩规划 |

### A4. 较新工作（2025–2026）

| 文件 | 出处 | 价值 |
|---|---|---|
| `2602.21612_QuadWheelLegged_Jumping_NMPC_DE.pdf` | 2026, RAL | 四足轮腿机器人跳跃控制：**NMPC + 差分进化** |
| `2606.26313_WheeledQuadruped_Racing_LoadTransferMPC.pdf` | 2026 | Unitree **Go2-W 竞速**：离线时间最优 raceline + 在线 MPC 最小化横向载荷转移率（LTR）+ 底层全身 RL 策略；膝关节作主动悬挂产生反侧倾力矩。LTR 降 44%，圈速快 8.7%。开源代码见 `repos/A/go2w-roll-control-mpc` |
| `2605.13058_MUJICA_WheeledLegged_MultiSkillArch.pdf` | 2026 | 轮足多技能统一控制架构 |
| `2211.11390_StateEstimation_HybridLocomotion_DrivingStepping.pdf` | IRC 2022 | **驱动-步行混合运动的状态估计**（速度/接触/滑移判别），对本项目速度估计环节直接相关 |

---

## 四、B 类：足式 MPC + WBC 方法论基座（架构可迁移）

| 文件 | 出处 | 价值 |
|---|---|---|
| `2208.08373_OCS2_PerceptiveLocomotion_NMPC.pdf` | ETH, T-RO 2023 | OCS2 的感知运动 NMPC 完整论述，本目录 `repos/B/ocs2` 对应论文 |
| `2112.07232_StructureExploitingNewton_SwitchedSystems.pdf` | ETH | OCS2 的**切换系统求解器**原理（SLQ / 结构利用牛顿法），多接触相位的数学基础 |
| `2203.00997_SLEQ_SwitchingTimeOpt_WholeBodyMPC.pdf` | 2022 | 在线切换时间优化的全身 MPC（SLEQ），刚性接触处理 |
| `1712.02889_Winkler_WholeBodyNMPC_ThroughContacts.pdf` | IROS 2018 | Winkler 全套全身 NMPC，腿式 MPC 的奠基工作之一 |
| `1909.06586_MIT_WBIC_Plus_MPC_HighlyDynamic.pdf` | MIT, 2019 | **WBIC + MPC 分工范式**：MPC 算地面反力，WBC 做全身冲量控制 |
| `2103.00946_UnifiedMPC_WholeBody_LocoManipulation.pdf` | 2021 | 运动与操作统一的全身 MPC 框架 |
| `1807.01222_Bellicoso_PrioritizedWholeBodyController.pdf` | IROS 2018 | **分层优先级 WBC** 的经典表述（任务优先级 + 接触约束），WBC 层写法的直接参考 |
| `1810.10270_WholeBodyMPC_ContactForces_CoMHeight.pdf` | 2018 | 含外部接触力与质心高度变化的全身 MPC |
| `1905.06144_FeedbackMPC_TorqueControlledLegged.pdf` | ICRA 2019 | 力矩控制腿式机器人的反馈 MPC |
| `1809.04539_Grandia_FrequencyAwareMPC.pdf` | ETH, RAL 2019 | **频率整形代价函数**：解决 MPC 解与执行器带宽不匹配导致实机抖动，轮足轮电机高速响应下尤为有用 |
| `2407.10789_TailoringSolutionAccuracy_FastWholeBodyMPC.pdf` | 2024 | 按需调整求解精度换取实时性 |
| `2503.04613_WholeBodyMPC_Legged_MuJoCo.pdf` | 2025 | **MuJoCo 全身 MPC**，工程落地视角；代码见 `repos/B/mujoco_mpc_deploy` |
| `2509.17884_LinearModels_WholeBodyMPC_Effectiveness.pdf` | 2025 | 线性模型做全身 MPC 的效果评估，质疑复杂度必要性 |
| `2203.04489_OnlineNMPC_Centroidal_Humanoid.pdf` | 2022 | 在线质心 NMPC（人形），质心动力学写法参考 |
| `2403.03995_CafeMPC_CascadedFidelity_WBC.pdf` | 2024 | 级联保真度 MPC + 免调参 WBC |
| `2109.07236_RecursiveHierarchicalProjection_WBC.pdf` | 2021 | 递归分层投影 WBC，任务优先级平滑切换 |

---

## 五、C 类：工程实现 / 求解器 / RL 对照

| 文件 | 出处 | 价值 |
|---|---|---|
| `2510.21773_RealTime_QPSolvers_Review_LeggedRobots.pdf` | 2025 | **实时 QP 求解器综述与选型指南**（qpOASES / OSQP / ProxQP / HPIPM 等），WBC 层选型直接参考 |
| `2502.01329_Benchmark_QPFormulations_QuadrupedalWalking.pdf` | 2025 | 四足行走的 QP 形式与求解器基准测试 |
| `2505.08574_EndToEnd_MultiTask_Policy_from_NMPC.pdf` | 2025 | 从 NMPC 蒸馏多任务策略（MPC→学习的迁移路线） |
| `2024_ScienceRobotics_LearningNavigation_WheeledLegged.pdf` | Science Robotics 2024 | 轮足自主导航与运动的 RL 方案（对照与混合路线参考） |
| `2024_ICRA_RL_BlindStairClimbing_WheeledLegged.pdf` | ICRA 2024 | 轮足盲爬楼梯 RL |

---

## 六、开源仓库

### A 类：轮足直接相关

| 目录 | 仓库 | 说明 |
|---|---|---|
| `repos/A/go2w-roll-control-mpc` | `meisman-ucb/go2w-roll-control-mpc` | Unitree Go2-W 竞速 MPC 规划 + 全身控制，含实机，对应 `2606.26313` |
| `repos/A/awesome-wheeled-legged` | `XinLang2019/awesome-wheeled-legged` | **轮足机器人开源代码与文献总清单**。内含 `model-base/`（10 篇模型基论文 PDF）、`ppt/`（模型基与 RL 两份中文综述 PPT）、硬件平台参数表 |
| `repos/A/Wheel-Legged-Gym` | `clearlab-sustech/Wheel-Legged-Gym` | 南方科技大学，轮足 RL 训练平台（VMC、崎岖地形、双足轮腿配置），可作动力学建模与仿真环境参考 |
| `repos/A/upkie` | `upkie/upkie` | 开源轮腿平衡机器人，Python/C++ 双栈，**含 MPC / LQR 基线控制器** |
| `repos/A/legged-robots-manipulation` | `aCodeDog/legged-robots-manipulation` | Unitree B2-W + Z1 轮足 loco-manipulation（IROS 2024 Oral） |

### B 类：足式 MPC + WBC 工程基座

| 目录 | 仓库 | 说明 |
|---|---|---|
| `repos/B/legged_control` | `qiayuanl/legged_control` | **最贴近本项目目标架构**：基于 OCS2 + ros-control 的 NMPC + WBC + 状态估计 + sim2real 完整框架（MPC 100 Hz / WBC 250 Hz，Unitree Go1 验证） |
| `repos/B/ocs2` | `leggedrobotics/ocs2` | ETH 切换系统最优控制内核，MPC 求解器底座，含轮式机器人资产 |
| `repos/B/quadruped_ros2_control` | `legubiao/quadruped_ros2_control` | ROS2-Control 版本，`ocs2_quadruped_controller` + `WbcBase` + `SafetyChecker` + 状态机，分层结构清晰 |
| `repos/B/qm_control` | `skywoodsz/qm_control` | 四足+机械臂 MPC+WBC 完整实现（OCS2 系），含力鲁棒、柔顺、实机分支 |
| `repos/B/mujoco_mpc_deploy` | `johnzhang3/mujoco_mpc_deploy` | MuJoCo 全身 MPC 在 Unitree 实机部署，对应 `2503.04613` |

### C 类：WBC / QP 底层库

| 目录 | 仓库 | 说明 |
|---|---|---|
| `repos/C/wbc` | `ARC-OPT/wbc` | DFKI 模块化 WBC 库，模型 / QP 求解器 / 控制器完全解耦，支持闭链与浮动基座 |
| `repos/C/tsid` | `stack-of-tasks/tsid` | LAAS-CNRS，逆动力学分层最小二乘（HLSP），多 QP 后端 |
| `repos/C/OpenSoT` | `ADVRHumanoids/OpenSoT` | IIT，算子式分层 QP，支持速度/加速度/力矩三级控制模式 |

---

## 七、中文资料

| 文件 | 说明 |
|---|---|
| `reports/CN_PPT_轮足_模型基控制综述.pdf` | 轮足机器人**模型基（MPC/WBC/TO）控制方法中文综述 PPT** |
| `reports/CN_PPT_轮足_强化学习综述.pdf` | 轮足机器人强化学习控制中文综述 PPT |
| `reports/CN_Deep_轮足与足式机械臂调研.html` | 轮足与足式机械臂调研报告（含各开源仓库对比、控制架构描述） |

---

## 八、未能获取的条目

| 条目 | 状态 | 原因与来源 |
|---|---|---|
| Di Carlo et al., *Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*, IROS 2018 | **未获取** | MIT DSpace 该条目未开放匿名下载（`/bitstreams/<uuid>/content` 返回 500，`/download` 返回 405）。元数据可访问：<https://dspace.mit.edu/handle/1721.1/138000.2>。内容与库中 `1712.02889`、`1905.06144`、`1909.06586` 有较大重叠 |

已获取但源自非 arXiv 渠道（在 `papers/A/`，文件名以年份开头）：

- `2019_RAL_TO_WheeledLeggedQuadruped_LinearizedZMP.pdf`
- `2020_RAL_TO_WheeledLeggedQuadruped_ChallengingTerrain.pdf`
- `2020_RAL_ComputationalFramework_SkilledLeggedWheeled.pdf`
- `2022_IRC_StateEstimation_HybridLocomotion_DrivingStepping.pdf`
- `2024_TMech_CompliantMotionControl_WheelLeggedHumanoid.pdf`
- `2025_TRO_NonsmoothTO_WheeledBalancingRobots.pdf`
- `2022_IJRR_Bjelonic_OfflineMotionLibraries_OnlineMPC.pdf`（ETH Research Collection REST API 取得）

---

## 九、建议阅读顺序

1. `A/2010.06322` — 轮足全身 MPC 应该长什么样
2. `repos/B/legged_control` 的 `README` 与 WBC 模块 — 分层架构怎么落地
3. `A/2022_IJRR` — 离线运动库 + 在线 MPC 的分工
4. `A/2020_RAL_TO_WheeledLeggedQuadruped_ChallengingTerrain` — 滚动约束怎么进优化
5. `B/1807.01222` + `B/2203.00997` — WBC 层与切换求解器的写法
6. `C/2510.21773` — QP 求解器选型
7. `A/2211.11390` — 混合驱动/步行的状态估计
