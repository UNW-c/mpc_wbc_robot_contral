# 测试agent执行总规程（2026-10-08）

## 一、角色和权限

### 问题排查agent

- 负责阅读代码、建立证据链、提出测试目的、指定数据字段、解释结果。
- 负责判断候选根因是“确认”“排除”还是“证据不足”。
- 未获得用户明确授权时，不修改控制算法、模型参数、阈值、限幅或状态机逻辑。

### 测试agent

- 只负责按照本文件顺序执行测试、保存原始数据、返回日志和结果表。
- 不负责定位根因，不自行改变测试口径，不自行选择参数，不以“让结果更好看”为目的重跑。
- 不得自行回退、合并、删除或覆盖工作区代码和已有结果文件。
- 不得把测试失败改成测试通过；失败、异常和无法执行都必须原样报告。

## 二、所有测试的硬性限制

1. **禁止调参**：不得修改增益、权重、阈值、滞回时间、确认时间、力矩限幅、速度限幅、腿长范围、MPC约束、模型参数、参考轨迹速率、外力大小或持续时间。
2. **禁止改变评价口径**：不得自行增删工况、修改成功标准、改变采样字段、替换单位、改变符号约定或放宽异常判据。
3. **禁止混用外力路径**：`delta_velocity`、脚本 `qfrc_applied`、viewer `xfrc_applied` 必须分组记录，不能合并成“扰动测试”。
4. **禁止跳过基线**：任何隔离对照都必须先完成同一输入、同一初始状态的当前版本基线。
5. **禁止只看最终视频**：必须同时保存事件时间轴、逐拍控制量、接触量、MPC/WBC中间量和最终 `data.ctrl`。
6. **禁止裸打印调试**：测试脚本使用项目日志工具或结构化文件记录，不使用 `print`、`printf`、`console.log`。
7. **禁止静默修工具**：若发现脚本语法、字段、单位或复现性问题，立即停止；先报告问题位置和影响，未经问题排查agent/用户确认不得修复后继续。
8. **禁止用异常结果继续推理**：工具自检失败、字段不一致、输入不一致或运行不可复现时，相关结果全部标记为作废。

## 三、每次测试前必须固定并记录

```text
测试日期和时间
工作区绝对路径
git commit hash
git status --short
Python解释器路径和版本
依赖/仿真环境版本
测试脚本路径和版本
完整执行命令
输入工况和全部参数
输出文件路径
```

工作区存在未提交修改时，必须把 `git status --short` 原样放入报告；不得假设当前代码等于最近一次提交。

## 四、结果文件要求

- 原始逐拍数据、摘要表、日志、异常说明分开保存。
- 文件名必须包含阶段、工况和运行编号，例如 `P0_baseline_run01.json`、`P6_qfrc_run01.json`。
- JSON/CSV字段必须包含单位或在同目录说明文件中统一说明。
- 任何覆盖已有结果的操作都必须先得到明确授权；默认使用新文件名。
- 报告必须同时给出“结果”和“结果是否有效”。

## 五、强制执行顺序

```text
第0阶段 评价工具自检
    ↓ 通过
第1阶段 当前版本基线时序
    ↓ 事件口径和数据链路有效
第2阶段 短暂离地与重接触
    ↓ 排除/解释离地时序问题
第3阶段 释放段腿摆角与pitch链路
    ↓ 明确pitch首次异常来源
第4阶段 MPC参与因果隔离
    ↓ 得到MPC是否必要的对照证据
第5阶段 执行器映射与符号核对
    ↓ 完成控制输出到电机的闭环核对
```

不得因为某个现象“看起来像MPC问题”就跳过第1、2阶段。若第2阶段发现真实离地但尚未解释重接触，必须停在第2阶段。

## 六、第0阶段：评价工具自检

### 操作

1. 运行语法检查和导入检查；
2. 检查脚本读取的字段确实存在于当前对象/记录文件；
3. 运行最小单工况冒烟，不用于评价算法，只确认脚本能完整生成数据；
4. 使用完全相同的输入重复运行两次。

### 必须报告

- 自检命令和完整输出；
- 字段、下标、单位、坐标系、符号映射；
- 冒烟是否完成；
- 两次重复运行的关键字段差异；
- 若失败，明确标记“后续测试未开始”。

## 七、第1阶段：当前版本基线时序

### 目的

只复现当前问题，不做隔离，不改任何参数。确定压缩、蹬伸、释放、飞行、落地及 `pitch` 首次异常的时间顺序。

### 操作

至少完成一组当前默认跳跃工况，并保存完整数据。测试agent不得根据视频自行判断相位，必须以代码状态和逐拍字段为准。

### 必须记录

```text
t, JumpPhase
leg_length_R/L, leg_rate_R/L
right_gap, left_gap, min_gap
轮地contact几何标志
right_normal, left_normal
contact_estimator.contact
feedback.decision.mode, motion_enabled, balance_enabled
pitch, pitch_rate
MPC是否求解, fallback, relaxed
sample.state, sample.acceleration
WBC是否调用
WBC原始torque, tau_jump, 最终data.ctrl
```

### 阶段输出

必须给出事件表：进入 `COMPRESS`、进入 `THRUST`、腿摆角首次偏移、pitch首次偏移、pitch首次发散、首次gap超过阈值、首次估计离地、进入 `FLIGHT`、首轮接触、双轮接触、恢复 `GROUND`。

### 进入下一阶段条件

- 数据字段完整；
- 时间轴可重建；
- `pitch` 首次异常和相位可定位；
- 结果重复运行一致。

否则停止，先修正记录工具并使修复前结果作废。

## 八、第2阶段：短暂离地与重接触

### 目的

确认压缩/蹬伸期间是否发生真实离地，确认离地后是否恢复单轮、双轮、WBC和 `GROUND`。

### 固定的四类输入

必须分开执行和报告：

1. 无外力；
2. `delta_velocity` 速度冲量；
3. 脚本 `Disturbance.force`，写入 `qfrc_applied`；
4. viewer 鼠标扰动，走 `xfrc_applied`。

除输入类型外，其余初始状态、持续时间和控制参数必须一致。不得把四类结果合并。

### 必须记录

```text
qfrc_applied[0:6]
xfrc_applied受力body和wrench
right/left gap
right/left contact几何
right/left法向力
right/left contact_estimator状态
airborne_active
decision.mode
WBC调用和最终data.ctrl
pitch, pitch_rate
```

### 进入下一阶段条件

必须明确以下之一：

- 未发生真实离地，且接触估计没有误触发；
- 发生真实离地，且已经解释进入、退出、单轮/双轮恢复和WBC恢复过程；
- 结果无法区分，明确列出缺失数据，不得继续假定离地链路正常。

## 九、第3阶段：释放段腿摆角与pitch链路

### 目的

区分“腿摆角先异常导致pitch发散”和“pitch目标/力矩先异常导致腿摆角偏移”。

### 必须记录

```text
MPC原始sample.state[2:8]
MPC原始sample.acceleration[1:4]
跳跃改写后的sample.state[2:8]
跳跃改写后的sample.acceleration[1:4]
pitch desired_acceleration
angle_R/L desired_acceleration
length_R/L desired_acceleration
WBC任务residual/scaling/exhausted
WBC原始腿部torque
tau_jump
最终data.ctrl
```

### 进入下一阶段条件

必须确定首次异常来自：

- MPC pitch参考/前馈；
- WBC pitch任务；
- 腿摆角任务；
- `tau_jump`；
- 执行器输出；
- 或数据不足无法判断。

## 十、第4阶段：MPC参与因果隔离

### 前置条件

第1～3阶段数据有效，且真实离地/重接触时序已解释。若不满足，不能进入本阶段。

### 固定对照组

| 组别 | MPC是否计算 | MPC sample是否送WBC | 跳跃前馈 |
|---|---|---|---|
| R0 | 是 | 是 | 保留 |
| R1 | 计算或记录 | 否，使用无sample分支 | 保留 |
| R2 | 是 | 只保留姿态/腿长明确诊断参考 | 保留 |
| R3 | 是 | 是 | 关闭，仅用于区分前馈影响 |

R0必须先跑。R1～R3只允许改变表中指定的控制通道，不得同时修改其它参数或评价标准。

### 必须报告

每组完整报告 `pitch` 峰值、`pitch_rate` 峰值、前向位移、腿摆角峰值、MPC fallback/relaxed、WBC残差/限幅、`tau_jump`峰值、最终各执行器力矩、离地事件时间表和是否恢复 `GROUND`。

不得只报告某一组“效果最好”。必须提供全部组别的完整表格和排序依据；本阶段不是调参搜索，不输出“最优参数”。

## 十一、第5阶段：执行器映射与符号核对

### 目的

排除“控制器计算正确但最终力矩写入错误电机/方向错误”。

### 操作边界

- 只做单执行器、低风险、短时映射核对；
- 不改变算法参数；
- 不把映射测试结果与跳跃性能测试混为一组；
- 每个执行器单独记录名称、绑定joint、`data.ctrl`位置、正负力矩后的关节运动方向、腿长/腿角变化方向。

### 必须核对

```text
Right_front_joint_act -> jAB
Right_rear_joint_act  -> jAG
Left_front_joint_act  -> jIJ
Left_rear_joint_act   -> jIO
Right_Wheel_act       -> jwheel_right
Left_Wheel_act        -> jwheel_left
```

同时核对 `selection` 非零列与 `thrust_torque()` 的写入顺序。

## 十二、阶段报告统一模板

测试agent每个阶段只提交以下内容：

1. 阶段编号和目标；
2. 环境快照与完整命令；
3. 自检状态；
4. 实际执行工况；
5. 原始数据文件和日志文件；
6. 完整结果表；
7. 异常时间轴；
8. 当前阶段结论：`确认`、`排除`、`证据不足`或`工具问题`；
9. 是否满足进入下一阶段条件；
10. 未执行项目及原因。

测试agent不得在报告中直接写“建议调成某个数值”或自行提交代码修改；优化方案由问题排查agent根据证据另行提出。

---

# 问题排查agent：阶段2部分结果审阅与下一步指令（2026-10-08）

## 一、当前结果有效性

测试agent第0阶段自检通过，P0当前默认跳跃基线、P6-E0无外力和P6-E1速度冲量结果有效；P6-E2脚本广义外力在3 s窗口内未完成重接触闭环，不能据此判定“接触恢复失败的根因”，但其“3 s内仍未恢复双轮接触”的事实有效。P6-E3 viewer `xfrc_applied`尚未执行。

有效结果文件：

```text
wheel_leg_mujoco/results/P0_baseline_default_run01.json.gz
wheel_leg_mujoco/results/P6_baseline_default_run01.json.gz
wheel_leg_mujoco/results/P6_delta_velocity_default_run02.json.gz
wheel_leg_mujoco/results/P6_qfrc_default_run02.json.gz
```

视频：

```text
C:\Users\20060\Desktop\circleg-robot\videos\test_default_jump_20261008_50fps.mp4
```

视频已用项目 `.venv` 中的 `imageio + imageio-ffmpeg` 解码：50 FPS、3.0 s、150帧、640×480。视频关键帧与P0基线时序一致：约1.20 s后开始离地，约1.24 s后姿态变化明显；视频本身只作现象佐证，精确事件时间以JSON逐拍记录为准。

## 二、已经确认的时序事实

以下结论来自 `P0_baseline_default_run01.json.gz`，不是由视频猜测：

| 时刻 | 事实 |
|---:|---|
| 1.180 s | `pitch=-0.0218 rad`，双轮仍有约8 N法向力；MPC/WBC pitch任务目标约 `-76.93 rad/s²`；相位为 `COMPRESS` |
| 1.200 s | `pitch=-0.0353 rad`，MPC/WBC pitch任务目标变为约 `+74.36 rad/s²`，同一压缩过程中的姿态加速度参考发生大幅反向变化 |
| 1.206 s | 两轮 signed gap 约 `+1.3 mm`，轮地几何接触为 False，法向力为 `[0,0]`；但接触估计仍为 `[True,True]`，跳跃相位仍为 `COMPRESS`，监督模式仍为 `ground` |
| 1.220 s | gap 已约 `23 mm`，几何仍无接触；估计仍为双轮接地；相位仍为 `COMPRESS`；WBC继续按接地模型求解，pitch目标约 `-77.74 rad/s²` |
| 1.226 s | 跳跃相位才进入 `THRUST`，但此时已经真实离地约20 ms；WBC上一拍力矩包含腿部约 ±40 N·m限幅值 |
| 1.227 s | 进入 `FLIGHT`，但接触估计仍为 `[True,True]` |
| 1.228 s | 因 `any(obs.contact)` 为 True，马上从 `FLIGHT` 进入 `LANDING`；此时 gap 约 `47 mm`、几何无接触、法向力为零，但监督模式仍为 `ground`，WBC再次被调用 |
| 1.235 s | 进入 `ABORT`，`pitch=-0.1056 rad`，`pitch_rate=-6.27 rad/s`；几何仍无接触、估计仍为双轮接地 |
| 1.250 s | 接触估计才变为 `[False,False]`，普通 airborne 路径才开始生效并将输出置零 |

## 三、阶段性判断

### 已确认：存在真实离地，不是单纯离地检查误触发

1.206 s时两轮几何接触消失、两轮gap同时为正、法向力同时为零；因此当前P0中确实发生了真实离地。

### 已确认：真实离地后，状态机和接触估计存在时序错配

- 真实离地后约50 ms，`contact_estimator`仍保持双轮接触；
- 跳跃状态机直到1.226 s才离开 `COMPRESS`；
- `FLIGHT`只保持约1 ms，然后因过期的 `[True,True]` 接触状态立即切入 `LANDING`；
- `LANDING`阶段仍按 `ground` 监督模式调用接地型WBC，而几何上已经完全离地；
- 这段时间输出仍保持或重新求解腿部力矩，足以直接解释pitch继续恶化的控制切换风险。

因此当前最强证据链是：

```text
真实离地
-> 接触估计滞后
-> COMPRESS/THRUST/FLIGHT/LANDING状态不同步
-> 离地后仍按接地WBC约束求解
-> pitch和腿部输出继续发散
```

这条链已比“单纯MuJoCo位置API误判”更具体，但还不能据此断言MPC没有贡献。

### 已确认：MPC姿态参考正在参与pitch控制

1.180 s时，MPC `sample.acceleration[1]` 和WBC `pitch desired_acceleration` 均约 `-76.93 rad/s²`；1.200 s时约 `+74.36 rad/s²`；1.220 s时约 `-77.74 rad/s²`。

当前跳跃实现只清零腿摆角参考，没有清零MPC pitch参考。因此“MPC姿态参考与真实接触/跳跃状态不一致”仍是并行候选原因，必须在阶段2时序问题处理完后进入阶段4隔离验证。

## 四、测试agent下一步指令

### 立即停止的项目

在以下事项完成前，禁止执行：

- 第3阶段P5释放段分析；
- 第4阶段MPC参与因果隔离；
- 第5阶段执行器映射测试；
- 任何控制算法修改、阈值调整、接触确认时间调整、MPC参数调整。

### 允许准备但暂不直接执行的补充测试

只补完第2阶段P6-E2和P6-E3，不改变控制参数：

1. **P6-E2延长观察窗口**：沿用已执行的 `qfrc_applied` 外力、初始状态和控制配置，仅为了观察重接触闭环而延长记录窗口；延长仿真时长属于评价口径变化，测试agent必须先报告拟用时长和依据，得到确认后才能执行。
2. **P6-E2重接触逐拍审计**：不重新解释旧结果，重点记录几何重新出现、首轮接触、双轮接触、`airborne_active`退出、WBC恢复调用、`RECOVERY/GROUND`切换和pitch变化。
3. **P6-E3 viewer路径**：必须真实使用viewer `xfrc_applied`，不得用E1或E2替代；如果当前环境无法自动化复现viewer操作，标记为未执行，不得伪造等价结果。

### P6-E2补测的停止条件

- 若发现延长窗口内真实双轮接触恢复但控制模式/WBC不恢复：停止并报告恢复链路问题；
- 若始终无真实双轮接触：停止并报告“外力工况造成持续失稳/未重接”，不继续进入MPC归因；
- 若几何已接触但法向力重构仍为零或contact估计不恢复：停止并报告接触力/外力扣除口径问题；
- 若出现新的工具字段、复现性或序列化问题：结果作废，回到第0阶段自检。

## 五、下一阶段进入条件

只有同时满足以下条件，才能进入第3阶段或第4阶段：

1. P6-E2重接触路径已解释，或明确证明该工况本身不会重接触；
2. P6-E3已执行或明确记录为环境无法执行；
3. 已确认阶段2的pitch发散是否发生在真实离地前、状态错配期间或离地后；
4. 原始数据可重复读取，所有结果有效性已标记；
5. 问题排查agent重新给出下一阶段书面指令。

当前状态：**阶段2未完成，下一步只做P6补充，不进入MPC隔离。**

---

# 问题排查agent：第1轮静态排查（2026-10-08）

## 当前边界

- 本轮只做代码、模型和控制链路排查，不回退代码、不修改控制参数、不运行测试。
- 历史文档中的测试结论只作为线索，不能替代当前工作区代码和用户对实际表现的描述。
- 当前工作区存在未提交修改和未跟踪的 `wheel_leg_mujoco/gbc_mpc_wbc/jump.py`；本轮不处理工作区清理。

## 已确认的代码事实

1. `Algorithm/contact_state.py` 实现了 `THRUST`、`FLIGHT`、`LANDING` 三种跳跃相位门控；但 `wheel_leg_mujoco/gbc_mpc_wbc/sampling.py::FeedbackSampler.update()` 每次固定传入 `JumpPhaseHint.NONE`。因此这些跳跃专用接触门控当前没有从采样链路生效。
2. `wheel_leg_mujoco/gbc_mpc_wbc/control.py::wbc_stage()` 在 `jump.active` 的全部阶段都会：
   - 从 WBC 任务中移除 `forward_velocity` 和 `yaw`；
   - 将轮执行器行从选择矩阵中清零；
   - 最后把左右轮输出强制设为 `jump.wheel_torque`，默认值为 0。
   这意味着压缩、蹬伸、着地和恢复阶段都不能继续使用 WBC 轮力矩，是否符合当前设计目标需要结合实际观感确认。
3. `wheel_leg_mujoco/gbc_mpc_wbc/loop.py` 在飞行段跳过 WBC，并保持上一拍力矩；非跳跃离地则置零。两条路径的切换依据不同，需检查状态机相位与接触估计是否一致。
4. `wheel_leg_mujoco/gbc_mpc_wbc/jump.py::JumpPlanner.start()` 的类型标注为返回 `JumpCommand`，但函数当前没有显式返回值；现有循环随后立即调用 `update()` 覆盖该变量，所以暂未据此断定为当前主故障，但接口契约不完整。
5. MPC 腿长调度位于 `wheel_leg_mujoco/gbc_mpc_wbc/simulation.py::_schedule_mpc_leg_length()`：当前代码对 `A/B` 做腿长插值，达到换挡条件时整体重建 `WheelMPC`，并迁移 `pitch_bias`。该路径需要和旧提交中的正确版本逐项对照，暂不接受“腿长拟合本身就是根因”的历史结论。

## 给测试agent的首轮测试方案（待用户现象确认后执行）

测试目的仅为区分建模错误、相位/接触切换错误和控制权限互相覆盖，不调参。

### T1：状态机和接触口径对齐

记录每个物理步：`t`、`JumpPhase`、`JumpPhaseHint`、两轮法向力、估计接触、轮地几何间隙、腿长、腿长率、俯仰角/角速度。
重点核对：进入 `THRUST`、几何离地、进入 `FLIGHT`、首次接触、进入 `LANDING` 的先后关系。

### T2：控制权限归属核对

记录 WBC 原始轮/腿力矩、跳跃前馈力矩、最终 `data.ctrl`，以及 `selection` 中轮执行器行；分别标记 `COMPRESS/THRUST/FLIGHT/LANDING/RECOVER`。
重点核对：是否在压缩或着地阶段过早把轮力矩置零，是否存在 WBC 输出与跳跃前馈重复叠加或互相抵消。

### T3：跳跃等效力建模核对

记录当前腿长与压缩终点处的腿雅可比、`tau_jump`、WBC 求解前后的动力学残差、预测法向力、执行器限幅。检查等效竖直力的符号、左右腿映射、`selection.T @ tau_jump` 与最终执行器力矩是否重复计入。

### T4：MPC 腿长调度静态/运行记录

不改参数，记录每次 MPC 求解的实际腿长、最近网格点、是否换挡、插值权重、`A/B` 摘要、能力表文件、MPC 对象身份、求解结果/回退状态。检查换挡前后是否真的使用新对象和新参数，以及参考/偏置状态是否连续。

## 需要用户先确认的实际现象

请以当前仿真观感为准回答，不必按历史文档措辞：

1. 压缩段是否正常：两腿是否同步、机身是否横移/前后窜、俯仰是否明显变化？
2. 蹬伸段具体表现：腿是否伸长、机身是否上抬、轮子是否离地、是否突然停止或回弹？
3. 若已离地，飞行段是姿态发散、腿乱摆、轮子乱转，还是立即重新接地？
4. 着地/恢复段具体表现是什么：单轮先着地、冲击过大、控制器抢回后抖动，还是状态机卡住？
5. MPC 腿长问题的实际感受是什么：换腿长时输出突变、运动方向错误、模型效果变差，还是只在跳跃期间异常？请提供对应结果文件名、日志片段或视频时间点。

## 当前结论

目前只能确认存在“跳跃相位没有传入接触估计器”的代码接线事实，以及多个控制权限覆盖点；在用户补充实际表现和测试agent返回 T1～T4 数据前，不对根因和修改方案下结论。

---

# 问题排查agent：第2轮离地检查专项（2026-10-08）

## 离地判定链路已拆分

当前不是单一的“离地检查”，而是两条不同链路：

### A. 跳跃状态机链路

`wheel_leg_mujoco/gbc_mpc_wbc/loop.py` 在每个物理步先调用 `mj_forward`，随后用
`R.wheel_contact_points(model, data)` 计算两个轮子的几何最低点，并把
`np.min(...[:, 2])` 作为 `JumpObservation.wheel_gap`。

`wheel_leg_mujoco/gbc_mpc_wbc/jump.py::JumpPlanner.update()` 在 `THRUST` 中主要依据：

```text
wheel_gap > 1 mm       -> FLIGHT
leg_length >= release_end -> FLIGHT（兜底）
```

这一条不依赖接触力估计器的最终结果；因此即使普通接触估计器仍认为双轮接地，跳跃状态机也可能先切到 `FLIGHT`，然后 `loop.py` 跳过 WBC。

### B. 普通离地档链路

`sampling.py::FeedbackSampler.update()` 当前流程是：

1. `mj_forward`；
2. 用 `data.contact` 的轮地接触几何确定哪些轮参与反解；
3. 用 `data.qacc`、实际 `data.ctrl` 和 `data.qfrc_applied` 反解接触力；
4. 将法向力交给 `GroundContactEstimator`；
5. `loop.py` 在两轮都判定离地时设置 `airborne_active`。

所以普通离地档当前不是直接用轮子位置，而是“MuJoCo 接触几何筛选 + 动力学接触力反解 + 法向力滞回”。但 `FeedbackSampler` 仍固定传入 `JumpPhaseHint.NONE`，跳跃相位专用门控没有接入。

## 已确认的离地逻辑风险

1. **跳跃几何判据无时间确认和滞回。** `wheel_gap > 1 mm` 单拍即可触发 `FLIGHT`，没有连续确认、回差或独立的双轮事件确认。MuJoCo 的软接触、几何间隙和姿态变化都可能造成瞬时正间隙。
2. **`wheel_gap` 不是接触状态。** `wheel_contact_points()` 只根据轮子的几何位姿估算圆柱最低点，没有读取接触约束是否有效，也没有使用法向冲量/法向力。因此“几何点离地”与“动力学上已经脱离支撑”不是同一个量。
3. **跳跃状态机使用 `np.min`。** 它取两轮最低点的最小值，只有两轮最低点都超过阈值时才进入 `FLIGHT`；这会掩盖单轮离地事件，不能表达单轮先离地/单轮先着地。但它本身不应把只有一轮抬起误判为整车离地。
4. **更严重的兜底条件可能提前切飞行。** `JumpConfig.release_end` 默认返回 `leg_length_nominal`。当前默认 `nominal_length=0.28 m`、`jump_stroke=0.12 m`，压缩终点约为 `0.16 m`；蹬伸腿长回到 `0.28 m` 时，即使轮地仍有接触，也会因 `leg_length >= release_end` 进入 `FLIGHT`。这不是 MuJoCo API 精度问题，而是状态机判据语义问题。
5. **进入 `FLIGHT` 的后果很强。** `loop.py` 的 `jump_disconnect` 会跳过 WBC；`jump.py` 的飞行命令要求保持上一拍力矩。若此前最后一拍仍是蹬伸力矩，这会把蹬伸末端力矩保持到真正离地之后；若此时实际尚未离地，则会直接失去接触平衡闭环。
6. **普通离地档在跳跃 active 期间被显式屏蔽。** `airborne = ... and not jump_active_now`，所以跳跃期间普通接触误判不会触发 `airborne_active`；真正能让跳跃脱离 WBC 的主要是 `JumpPlanner` 的 `wheel_gap/release_end` 判据。这条需要与日志中的实际相位切换时间核对。

## 对当前怀疑的判断

- “离地检查误触发导致未离地却脱离控制器”是**高可信候选**，但首先应查 `release_end` 提前切换和单拍 `wheel_gap`，不能笼统归因于 MuJoCo API。
- “直接调用位置 API 是根因”目前不能直接成立。位置 API 确实只提供几何间隙，不提供支撑状态；但当前跳跃逻辑中更明确的问题是：几何判据没有确认机制，且有一个默认等于标称腿长的提前释放兜底条件。
- 普通离地档的接触力反解是另一条链路，不能把它和跳跃几何 `wheel_gap` 混为一个问题。

## 给测试agent的离地专项方案（不调参）

### L1：状态机切换对照

每个物理步同时记录：

```text
t, JumpPhase, jump_command.disconnect,
leg_length, release_end, leg_length_rate,
right_gap, left_gap, min_gap,
MuJoCo轮地contact几何标志,
两轮反解法向力, contact_estimator.contact,
最终data.ctrl
```

必须标出第一次满足以下条件的时间：

```text
wheel_gap > 1 mm
leg_length >= release_end
两轮接触估计均为 False
jump_command.phase == FLIGHT
```

判定重点：如果 `FLIGHT` 先由 `leg_length >= release_end` 触发，而 `min_gap <= 1 mm` 或轮地仍有接触，则可以确认是状态机兜底条件提前切换；如果先由 `wheel_gap` 触发，则检查该间隙是否仅为单拍/软接触数值波动。

### L2：几何间隙与动力学支撑一致性

在不改变任何阈值的情况下，对齐同一时刻的：

- 两轮几何最低点到地面的 signed gap；
- `data.contact` 中的轮地接触对；
- 接触力反解的逐轮法向力；
- 实际执行器力矩和腿长率。

不要只输出 `min_gap`；必须分别输出右轮和左轮，否则无法识别单轮事件。

### L3：脱离控制器因果核对

在 `FLIGHT` 切换前后逐拍记录：

- WBC 是否在该拍被跳过；
- 切换前最后一拍的腿部/轮部力矩；
- 切换后 `data.ctrl` 是否保持该力矩；
- 切换时轮地接触和法向力是否仍存在。

如果切换后仍接地且 `data.ctrl` 保持蹬伸力矩，应优先判定为“错误切入飞行 + 零阶保持”组合问题，而不是先怀疑 MPC。

### L4：MPC 是否参与异常的隔离记录

同一时刻记录：

```text
MPC是否求解、MPC fallback/relaxed、solution.reference、solution.acceleration、
WBC原始输出、tau_jump、最终data.ctrl
```

判断“大轮力矩”来自哪里：MPC 中间量、WBC 原始输出、跳跃前馈，还是最终执行器输出。只有最终 `data.ctrl` 中轮执行器确实有异常大力矩，才能把它认定为实际驱动原因。

## 当前建议顺序

1. 先验证 `release_end` 是否在未离地时触发 `FLIGHT`。
2. 再验证 `wheel_gap` 单拍触发和左右轮分别的几何/接触一致性。
3. 再隔离 MPC：跳跃期间停止 MPC 求解，但保留其它控制链路不变，只用于因果对照；该方案需测试agent执行，不在本轮修改。
4. 最后做执行器映射/符号测试，避免把状态机误切换造成的异常误判为电机接线错误。

## 对架构问题的暂定建议

跳跃不应继续让接地型 MPC 规划轮力矩。推荐采用阶段权限：

```text
接地行驶：MPC + WBC
压缩：跳跃参考 + 平衡WBC
蹬伸：跳跃专用WBC/VMC任务，MPC不提供运动参考
飞行：无接触姿态/腿部控制，不使用接地型WBC
着地：接触确认后缓冲WBC
恢复：参考重置后恢复MPC
```

但“完全交给 WBC”只有在 WBC 增加跳跃专用任务、无接触模式和接触切换契约后才成立；当前实现仍是“跳跃前馈力 + 接地WBC 剩余任务”，不能直接视为完整跳跃VMC。

---

# 问题排查agent：第3轮 pitch 发散测试清单（2026-10-08）

## 测试目标

当前主问题定义为：压缩段出现前向位移，蹬伸段机身 `pitch` 发散，随后腿摆角发散，沿腿作用力产生倾斜分量并进一步放大姿态误差。

测试只判断因果，不调参、不扩大阈值、不修改控制参数。所有测试结果必须保留完整结果表，不能只报告“能否跳起来”。

## 0. 评价工具自检（测试agent执行任何正式测试前必须先做）

必须报告以下命令和输出：

1. 测试脚本语法与导入通过；
2. 记录字段与实际对象一致，尤其是 `phase`、`wheel_gap`、`contact`、`solution`、`data.ctrl`、`tau_jump`、WBC任务残差；
3. 单工况短时冒烟能运行并产生日志；
4. 同一输入重复运行两次，关键时间点、状态和力矩结果一致。

如果自检失败，后续结果全部作废，先报告工具问题，不得通过放宽判据或扩大扫描规避。

## P0：相位切换与 pitch 发散时刻对齐

### 输入

当前默认跳跃工况，保持现有参数不变。

### 每个物理步记录

```text
t
JumpPhase
jump_command.active / disconnect / hold_torque
leg_length / leg_length_rate
right_gap / left_gap / min_gap
MuJoCo轮地contact几何标志
两轮法向力
contact_estimator.contact
feedback.decision.mode / motion_enabled
pitch / pitch_rate
```

### 必须给出的结果

| 事件 | 时间 | 当时相位 | 右/左 gap | 右/左法向力 | 接触估计 | pitch | pitch_rate |
|---|---:|---|---:|---:|---|---:|---:|
| 进入 COMPRESS | | | | | | | |
| 进入 THRUST | | | | | | | |
| 首次 pitch 明显偏离 | | | | | | | |
| 首次 `wheel_gap > 1 mm` | | | | | | | |
| 首次 `leg_length >= release_end` | | | | | | | |
| 进入 FLIGHT | | | | | | | |
| 首次接触/进入 LANDING | | | | | | | |

### 判定

- `pitch` 在 `THRUST` 内、且 `FLIGHT` 之前已经明显发散：优先查蹬伸力方向、WBC pitch任务和 MPC参考，不先归因于离地误判。
- `pitch` 在 `FLIGHT` 切换后才突然发散：重点查状态机切换、WBC跳过和上一拍力矩保持。
- `FLIGHT` 由 `leg_length >= release_end` 触发，同时 `min_gap <= 1 mm` 或仍有轮地接触：确认释放兜底条件提前切换。
- `FLIGHT` 由 `wheel_gap > 1 mm` 单拍触发，而法向力/接触几何仍显示接地：确认几何判据误触发。

## P1：MPC是否实际把异常参考传给WBC

### 每个WBC周期记录

```text
MPC是否求解
MPC fallback / relaxed
当前腿长与最近网格点
MPC sample.state
MPC sample.acceleration
跳跃代码改写后的 sample.state / acceleration
WBC各任务 desired_acceleration：pitch、angle_R、angle_L、length_R、length_L、forward_velocity、yaw、roll
WBC原始 torque
tau_jump
最终 data.ctrl
```

### 重点核对字段

1. `sample.state[2:6]` 是否在跳跃期间出现异常的 pitch、pitch_rate、yaw 或 yaw_rate参考；
2. `sample.acceleration[1]` 是否产生异常 pitch 加速度要求；
3. `sample.state[6:8]` 和 `sample.acceleration[3]` 是否确实被清零；
4. `forward_velocity` 和 `yaw` 是否确实从跳跃期任务集中移除；
5. 当前代码虽然移除了 `forward_velocity`，但是否仍由 `pitch` 平衡任务间接产生前向轮地反力；
6. `WBC原始 torque + tau_jump` 是否在最终写入前发生符号或执行器位置变化。

### 判定

- MPC的 `sample` 或 `acceleration` 在发散前先出现异常，且WBC任务跟随该异常：支持“MPC仍参与跳跃并超出有效预测范围”。
- MPC数值正常，但WBC的 `pitch` desired acceleration 或原始腿力矩异常：优先查WBC任务构造、行符号或接触约束。
- WBC原始输出正常，但 `tau_jump` 加入后 pitch加速度反号或急增：优先查沿腿力雅可比、左右腿映射和力矩叠加。
- 所有控制器中间量正常，但最终 `data.ctrl` 的关节顺序/符号不一致：优先查执行器映射。

## P2：MPC参与/不参与的因果隔离

这不是调参，而是控制权限隔离。必须使用相同初始状态、相同输入、相同物理时长，并输出完整对照表。

| 工况 | MPC求解 | WBC接收MPC sample | 跳跃前馈 | 目的 |
|---|---|---|---|---|
| A 当前基线 | 是 | 是 | 是 | 复现当前问题 |
| B 跳跃期不执行MPC | 否 | 否，使用无MPC参考分支 | 是 | 判断MPC是否是必要原因 |
| C MPC继续计算但sample不送WBC | 是 | 否 | 是 | 区分“求解发散”和“参考实际影响执行” |
| D 保留MPC sample但关闭跳跃前馈 | 是 | 是 | 否，仅作诊断 | 判断蹬伸力/雅可比是否主因 |

### 每组必须报告

```text
是否进入THRUST / FLIGHT
首次pitch偏离时间
pitch峰值、pitch_rate峰值
前向位移、前向速度峰值
腿长峰值、腿角峰值
MPC fallback/relaxed次数
WBC infeasible/clip次数
tau_jump峰值
最终轮/腿data.ctrl峰值
是否出现joint_limit、attitude_guard或其它告警
```

### 判定

- 只有 A 发散，B/C 稳定：MPC参与跳跃控制是高可信根因。
- A/B 都发散，D稳定：跳跃前馈或沿腿力建模是高可信根因。
- A/B/C/D 都在同一时刻发散：优先查状态切换、执行器符号或基础动力学映射。
- A与C都发散但B稳定：MPC虽然计算异常，但其结果确实通过WBC路径影响了执行器。

## P3：状态机/离地误触发因果隔离

### 必须检查

1. `FLIGHT` 的真实触发源是 `wheel_gap` 还是 `release_end`；
2. 触发前后是否仍存在轮地接触；
3. 触发后WBC是否停止；
4. 触发后最终 `data.ctrl` 是否保持上一拍蹬伸力矩；
5. `pitch` 发散是否紧跟在“跳过WBC/保持力矩”之后。

### 必须报告

```text
FLIGHT触发源
触发时 right_gap / left_gap / min_gap
触发时 right_normal / left_normal
触发时 contact_geometry
触发前一拍最终data.ctrl
触发后连续若干拍最终data.ctrl
WBC调用是否停止
pitch与pitch_rate触发前后变化
```

### 判定

- 未离地但WBC停止且保持蹬伸力矩：确认“误切FLIGHT + 零阶保持”是姿态发散直接原因。
- 真实离地后才停止WBC，但pitch在此之前已发散：离地检查不是首要根因。
- 仅几何gap触发而接触力和接触几何不同步：确认几何API只适合作为影子诊断量，不能单独控制切换。

## P4：执行器映射与力矩方向核对

该项用于排除“控制器算对、输出到错误电机”的可能性。测试agent只做小幅、短时、单执行器映射验证，不修改控制参数。

### 必须输出

```text
执行器名称
绑定joint名称
data.ctrl下标
selection非零列
施加正/负测试力矩后的对应joint位置变化方向
对leg_length和leg_angle的瞬时变化方向
```

重点核对：

- `Right_front_joint_act -> jAB`；
- `Right_rear_joint_act -> jAG`；
- `Left_front_joint_act -> jIJ`；
- `Left_rear_joint_act -> jIO`；
- `Right_Wheel_act -> jwheel_right`；
- `Left_Wheel_act -> jwheel_left`。

同时核对 `thrust_torque()` 的 `tau` 写入顺序是否与上述执行器顺序一致，以及左右腿正力是否都使腿长增加。

## 测试agent最终需要返回的最小结果包

1. 评价工具自检命令和输出；
2. P0事件时间表；
3. P1一拍控制链路表；
4. P2四组隔离对照表；
5. P3离地触发源及触发前后控制输出；
6. P4执行器映射表；
7. 对每个候选根因标记：`确认`、`排除`或`证据不足`；
8. 原始日志/JSON文件名，以及关键数据字段说明。

## 当前不允许据此直接下的结论

- 不能仅凭pitch发散就认定MPC预测越界；必须看到MPC输出、WBC任务和最终执行器输出的时间先后关系。
- 不能仅凭轮子位置变化就认定电机接线错误；必须完成单执行器映射表。
- 不能仅凭进入FLIGHT就认定已经离地；必须同时核对几何gap、接触几何和法向力。

---

# 问题排查agent：第4轮释放段姿态/腿摆角耦合专项（2026-10-08）

## 已确认的代码事实

当前跳跃期对 MPC 的处理是不对称的：

- `control.py::wbc_stage()` 会把 `sample.state[6:8]` 和 `sample.acceleration[3]` 清零，用于断开 MPC 腿摆角参考；
- 但 `wbc.py::Gbc486Tasks.tasks()` 的 `pitch_target` 仍然使用：

```text
pitch_target = sample.acceleration[1]
              + kp * (sample.state[2] - measured_pitch)
              + kd * (sample.state[3] - measured_pitch_rate)
```

只要 `solution` 不为 `None`，跳跃期间 pitch 目标就不是强制 0，而是继续来自 MPC 轨迹。当前代码没有实现“跳跃期间 pitch 目标固定为 0”的契约。

此外，跳跃期的腿摆角任务虽然把 MPC 腿摆角参考清零，但仍会执行腿摆角回零/阻尼任务；腿长任务又被改写为跳跃腿长目标。释放段同时存在：

```text
腿长快速变化
腿摆角回零/阻尼
MPC pitch参考与pitch加速度
WBC pitch高优先级任务
```

这些任务是否在当前执行器权限和接触约束下同时可实现，必须用逐拍任务输出核对，不能从最终姿态单独推断。

## 释放段专项假设

待验证的因果链为：

```text
释放开始
-> 腿摆角先出现偏移
-> MPC/WBC腿摆角任务产生回正动作或腿部反力不对称
-> 接触点反力产生pitch力矩
-> WBC pitch任务试图纠正，但其目标仍受MPC sample影响
-> pitch误差/角速度继续增大
-> 沿腿作用力方向偏离竖直，进一步放大pitch
```

这条链目前是候选根因，不是已确认根因。需要先确认第一处异常是腿摆角、pitch目标、WBC任务残差还是最终执行器输出。

## P5：释放段逐拍因果表

测试agent必须在释放开始前后至少覆盖以下字段：

```text
t, JumpPhase
leg_length_R/L, leg_rate_R/L
leg_angle_R/L, leg_angle_rate_R/L
MPC sample.state[2:8]
MPC sample.acceleration[1:4]
改写后的 sample.state[2:8]
改写后的 sample.acceleration[1:4]
measured_pitch, measured_pitch_rate
pitch desired_acceleration
angle_R/L desired_acceleration
length_R/L desired_acceleration
WBC原始腿部torque
tau_jump右/左腿四个关节分量
最终data.ctrl右/左腿四个关节分量
WBC每个任务的residual、scaling、exhausted
pitch、pitch_rate
```

### 必须标出四个首次异常时刻

| 事件 | 需要回答 |
|---|---|
| 腿摆角首次偏离 | 哪一侧、偏离方向、发生时是否仍双轮接触 |
| pitch目标首次非零/突变 | 来自 `sample.state` 还是 `sample.acceleration[1]` |
| pitch实际首次发散 | 是否早于腿摆角偏移，还是滞后于腿摆角偏移 |
| 最终执行器力矩首次异常 | 来自WBC原始力矩、`tau_jump`，还是合并后的`data.ctrl` |

## P5-A：pitch参考来源核对

必须分别给出以下值，而不能只给最终 `pitch`：

```text
MPC pitch reference = sample.state[2]
MPC pitch-rate reference = sample.state[3]
MPC pitch acceleration feedforward = sample.acceleration[1]
WBC pitch desired acceleration
measured pitch error
measured pitch-rate error
```

### 判定

- 若释放前后 `sample.state[2]` 或 `sample.acceleration[1]` 出现明显变化，且 `pitch_target` 同步变化：MPC仍在直接定义跳跃期姿态目标。
- 若理论姿态目标应为0，但 `sample.state[2]` 非0或 `sample.acceleration[1]` 非0：确认当前实现不满足跳跃姿态契约；这还不能单独证明该值导致发散，但必须隔离。
- 若 pitch目标保持合理，实际pitch仍发散：查WBC可实现性、接触力方向、执行器饱和和力矩映射。

## P5-B：腿摆角先导还是pitch先导

根据首次异常时刻分支：

### 分支1：腿摆角先异常

检查：

- 两侧 `angle_R/L desired_acceleration` 是否符号相反或大小不对称；
- 两侧腿长任务是否同时、同值跟踪；
- `tau_jump` 是否左右腿对称；
- 左右实际关节角速度是否同向/反向符合机构定义；
- 释放前后是否有一侧先失去接触。

若腿摆角任务残差先增加，而pitch任务仍正常，优先查腿角行符号、腿角回零逻辑和左右执行器映射。

### 分支2：pitch先异常

检查：

- MPC pitch参考/前馈是否先突变；
- WBC pitch任务是否被限幅或降级；
- `tau_jump` 的腿部力矩是否本身带有非零pitch广义力矩；
- 两腿几何姿态是否使相同竖直腿力映射成不同的pitch力矩。

若pitch任务从第一拍就不可实现，不能把后续腿摆角偏移视为根因，它可能只是姿态发散后的结果。

## P5-C：MPC姿态参考隔离

在不调参的前提下做三个因果对照：

| 对照 | 释放段MPC计算 | 送入WBC的pitch参考 | 送入WBC的腿摆角参考 | 目的 |
|---|---|---|---|---|
| R0 当前基线 | 保留 | 当前sample值 | 当前代码清零后的值 | 复现问题 |
| R1 姿态参考断开 | 可保留并记录 | 固定为0的诊断参考 | 保持当前清零逻辑 | 判断MPC pitch参考是否直接致因 |
| R2 完全不送MPC sample | 可只计算不执行 | 由WBC无sample分支生成 | 由WBC本地零目标生成 | 判断MPC参考是否整体致因 |

每组必须使用相同初始状态和相同跳跃输入，并报告：

```text
释放前后腿摆角峰值
pitch峰值 / pitch_rate峰值
前向位移
WBC任务残差与exhausted/scaling
腿部tau_jump峰值
最终各执行器data.ctrl峰值
FLIGHT切换时间及触发源
```

### 判定

- R0异常、R1/R2明显改善：MPC姿态/整体sample耦合是高可信原因。
- R0/R1异常、R2改善：不只是pitch参考，MPC sample中的其它量或sample存在性分支也在影响WBC。
- R0/R1/R2均异常且第一异常为腿角任务：优先查WBC腿角任务、腿部雅可比和执行器方向。
- 三组都在FLIGHT误切后发散：先处理状态机/离地切换因果，不能把MPC作为首要原因。

## P5-D：压缩段前向位移专项

当前代码在跳跃 active 时会移除 `forward_velocity` 和 `yaw` 任务，因此压缩段前向位移不是“前向速度任务仍直接存在”这一简单问题。但 MPC pitch任务仍在，且腿长压缩改变了接触几何；需要区分三条路径：

1. MPC `sample.state[2:6]` / `sample.acceleration[1]` 通过 pitch任务间接生成前向接触反力；
2. 腿长任务和闭链机构反力在左右不完全对称时产生水平力或pitch力矩；
3. 腿长调度改变MPC模型/参考后造成姿态参考不连续。

压缩段必须单独报告：

```text
前向位移与前向速度
forward_velocity任务是否确实不存在
pitch desired acceleration
MPC pitch reference/前馈
右左腿长/腿角任务输出
左右腿WBC torque差值
左右tau_jump差值（压缩段应为零前馈，但仍需记录）
轮执行器最终data.ctrl
轮地切向力、法向力
MPC换挡点、腿长、A/B摘要和fallback状态
```

若 `forward_velocity` 任务不存在但前向位移仍出现，不能直接说MPC在规划前进；应进一步判断是 pitch任务通过接触耦合产生，还是腿长/机构反力产生。

---

# 问题排查agent：第5轮短暂离地与重接触专项（2026-10-08）

## 当前需要区分的三种情况

不能把以下现象统称为“离地误判”：

1. **几何/力估计误判离地**：轮子实际上仍支撑，但估计接触变为 False；
2. **测试外力确实造成短暂真实离地**：接触断开后又恢复；
3. **真实离地后接触已恢复，但控制链路没有恢复到双轮稳定状态**：例如只恢复单轮、仍处于 `RECOVERY`，或者接触力重构没有正确恢复。

## 已确认的时序事实

### 1. 普通离地档的进入条件

`loop.py` 使用：

```text
airborne = feedback存在
           and 两轮contact_estimator均为False
           and jump_active_now为False
```

因此在跳跃状态机 active 时，普通 `airborne_active` 被显式屏蔽；跳跃期间的控制脱离主要由 `JumpPlanner` 的 `FLIGHT` 命令触发，而不是普通离地档触发。

### 2. 普通离地档的退出条件

只要 `feedback.decision.contact` 中任意一轮恢复为 True，`airborne` 就变为 False：

```text
not any(contact) -> airborne
any(contact)     -> 退出 airborne，恢复 WBC调用
```

这意味着它不是“双轮重新接地后恢复”，而是“任一轮接触后退出全车 airborne”。退出后控制监督层可能处于 `SINGLE_SUPPORT` 或 `RECOVERY`，并不代表已经恢复到双轮 `GROUND`。

### 3. 接触确认时间

当前 `FeedbackSampler` 固定以 `JumpPhaseHint.NONE` 调用估计器：

- 离地：`off_duration_idle=0.050 s`；
- 重接地：`on_duration=0.012 s`；
- 只有一轮法向力达到接地门限时，该轮即可先恢复 True。

因此真实离地少于 50 ms，按当前普通离地链路理论上不应进入 `airborne_active`；真实离地超过 50 ms 才会进入。这个判断必须用逐拍数据核实，不能只看视频。

## 外力注入方式存在不同的数据路径

### 脚本 `Disturbance`

`loop.py` 将持续外力写入 `data.qfrc_applied[:6]`，采样级接触力反解调用：

```text
recover_contact_force(..., external=data.qfrc_applied)
```

这条路径显式扣除了脚本外力。

### viewer 鼠标扰动

viewer 路径调用 `mjv_applyPerturbForce()`，并通过 `data.xfrc_applied` 表示外力；但当前采样级反解仍只传入 `data.qfrc_applied`，没有显式把 `data.xfrc_applied` 转换/扣除到同一广义力口径。

因此 viewer 外力测试和脚本 `Disturbance(force=...)` 测试不一定具有相同的接触力重构结果。这是已确认的代码路径差异，是否造成当前误判需要通过同一输入的双路径对照确认。

另外，`Disturbance.delta_velocity` 不是外力，而是在物理推进前直接修改 `qvel`；它不会通过 `qfrc_applied` 出现在接触力反解的外力扣除项中。它适合做速度冲量测试，不应和持续外力测试混为一类。

## P6：短暂离地/重接触逐拍时序测试

### 每个物理步必须记录

```text
t
外力注入方式：none / delta_velocity / qfrc_applied / viewer_xfrc_applied
qfrc_applied[0:6]
xfrc_applied中受力body的完整wrench
右/左轮MuJoCo contact几何标志
右/左signed gap
右/左反解法向力
right/left contact_estimator状态
feedback.decision.mode
feedback.decision.reset_reference
airborne_active
WBC是否调用
WBC输出是否写入data.ctrl
最终data.ctrl
pitch / pitch_rate
```

### 必须生成的事件时间表

| 事件 | 时间 | 右接触 | 左接触 | 右法向力 | 左法向力 | 右/左gap | 监督模式 | airborne_active | WBC调用 |
|---|---:|---|---|---:|---:|---:|---|---|---|
| 外力开始 | | | | | | | | | |
| 首次真实几何脱离 | | | | | | | | | |
| 首次估计单轮离地 | | | | | | | | | |
| 两轮估计离地 | | | | | | | | | |
| 首轮重新接触 | | | | | | | | | |
| 双轮重新接触 | | | | | | | | | |
| 退出 airborne | | | | | | | | | |
| 恢复 GROUND | | | | | | | | | |
| pitch开始发散 | | | | | | | | | |

## P6-A：判断是否真实离地

不能只看 `contact_estimator.contact`。对同一时刻同时核对：

1. 轮地几何 contact pair 是否存在；
2. 右/左 signed gap 是否真正为正并持续；
3. 右/左法向力是否下降到离地门限以下并持续 50 ms；
4. 机身竖直速度/质心高度是否与离地方向一致；
5. 外力结束后是否出现明确的着地冲击或法向力恢复。

### 判定

- 几何间隙持续为正、轮地 contact pair消失、法向力归零：真实离地。
- 几何仍接触、法向力正常，但估计 contact变 False：接触力重构/外力扣除问题。
- 只有 viewer 外力路径异常，脚本 `qfrc_applied` 路径正常：优先查 `xfrc_applied` 未纳入反解。
- `delta_velocity` 路径才出现离地：说明测试输入本身改变了速度状态，不能作为“持续外力接触估计”的等价验证。

## P6-B：判断是否“接地后没有接回控制器”

接触恢复后必须逐拍检查：

```text
airborne_active是否由True变False
WBC是否在下一个due周期重新调用
control mode是否按 AIRBORNE -> SINGLE_SUPPORT/RECOVERY -> GROUND变化
decision.reset_reference是否出现
WBC输出是否重新写入data.ctrl
两轮是否都恢复，而不是只有一轮恢复
```

### 判定

- `airborne_active` 已退出、WBC已调用但仅单轮接触：不是“未接回”，而是进入单支撑/恢复阶段，需单独分析姿态能力。
- `airborne_active` 仍为 True，但几何和法向力均已恢复：接触估计器或外力扣除链路有问题。
- `airborne_active` 已退出但 `WBC` 没有重新调用：检查 `wbc_due` 时序或跳跃 `disconnect` 是否仍为 True。
- WBC已调用但 `data.ctrl`仍保持旧值：检查输出级零阶保持和写回路径。
- 已回到 `GROUND` 但 pitch仍发散：离地档不是唯一原因，应转查MPC pitch参考、腿角任务和tau_jump。

## P6-C：同一物理输入的外力路径对照

不调参数，只做路径对照：

| 工况 | 输入形式 | 目的 |
|---|---|---|
| E0 | 无外力 | 跳跃自身基线 |
| E1 | `delta_velocity` | 验证速度冲量造成的真实运动响应 |
| E2 | `Disturbance.force` -> `qfrc_applied` | 验证脚本广义外力与反解扣除 |
| E3 | viewer鼠标 -> `xfrc_applied` | 验证viewer外力路径是否污染接触力重构 |

每组必须报告相同的事件时间表，特别是：首次真实间隙、首次估计离地、退出 airborne、恢复 GROUND 和 pitch开始发散的时间差。

## 本轮对pitch根因的判定优先级

1. 先确认 `pitch` 发散前是否发生真实短暂离地；
2. 若发生，确认离地是跳跃控制本身、外力测试输入还是接触模型导致；
3. 再确认接触恢复后是否只恢复单轮、是否恢复 `GROUND`、WBC是否重新写回；
4. 只有排除上述时序问题后，才把MPC pitch参考/腿长调度视为主要根因。

---

# 问题排查agent：测试执行顺序（2026-10-08）

测试agent必须按以下顺序执行，前一阶段没有形成可核对证据时，不得进入后一阶段：

## 第0阶段：评价工具自检

先完成语法/导入、字段口径、单工况冒烟、重复运行一致性四项自检。任何一项失败，后续结果作废，先报告工具问题。

## 第1阶段：当前基线时序

执行第1轮中的 T1～T4，优先得到一份当前版本的完整时间轴和控制链路记录。重点是确认 `COMPRESS -> THRUST -> FLIGHT`、pitch首次发散、MPC/WBC输出和最终 `data.ctrl` 的先后关系。

## 第2阶段：离地与重接触核对

执行第2轮 L1～L4，以及第5轮 P6。由于用户最新怀疑涉及真实短暂离地，这一阶段优先级高于MPC深度隔离。必须先回答：是否真实离地、是否进入普通 airborne、是否只恢复单轮、WBC是否重启、外力注入路径是否污染接触力反解。

## 第3阶段：pitch/腿摆角释放段链路

只有第2阶段确认不存在足以解释pitch发散的离地/重接触问题后，执行第4轮 P5，核对腿摆角、MPC pitch参考、WBC pitch任务和 `tau_jump` 的首次异常时刻。

## 第4阶段：MPC参与因果隔离

执行第3轮 P1～P3中的 P2 对照：当前基线、跳跃期停止MPC并不给WBC sample、MPC计算但不送sample、关闭跳跃前馈。每组使用相同初始状态和相同输入，不调参数。

## 第5阶段：执行器映射与符号核对

执行第3轮 P4，验证单执行器名称、joint绑定、正负方向、腿长变化方向和 `thrust_torque()` 写入顺序。该阶段用于排除“控制器中间量正确、最终电机执行错误”。

## 阶段间停止规则

- 评价工具自检失败：立即停止；
- 发现结果不可复现：立即停止，先查环境或记录工具；
- 发现测试脚本字段错误：结果作废，不得继续扩大工况；
- 发现真实离地且尚未解释重接触：先完成 P6，不得直接把问题归因于MPC；
- 发现MPC隔离组与基线行为不同：保留完整对照表，不得直接修改MPC参数；
- 任何阶段均不得为了通过测试调整增益、阈值、限幅或模型参数。


---

# 测试agent：第0阶段诊断工具修复审计（2026-10-08）

## 状态

本轮正式算法测试结果全部标记为无效，未进入第1～5阶段。原因是诊断工具自检和工况口径尚未完全通过，不能据此确认或排除任何候选根因。

## 已获授权的观测性改动

用户已授权只修改测试算法和日志/数据记录，不修改核心算法模块和控制参数。当前工作区增加的观测字段包括：

- 每个物理步的跳跃相位、相位命令和输出来源；
- 左右轮 signed gap、轮地几何接触、法向力、切向力；
- 接触估计器、监督模式、airborne_active、reset_reference；
- MPC 调用、fallback、relaxed、sample 原始值和跳跃改写值；
- WBC 调用、任务目标、原始力矩、tau_jump、最终 ctrl、求解残差和限幅状态；
- MPC 对象身份、腿长网格点和 A/B 模型摘要。

未改变 MPC/WBC 数学计算、跳跃状态机判据、模型参数、控制增益、权重、阈值、限幅和参考输入。

## 现场保全

现场审计目录：

```text
artifacts/test_agent/diagnostic_audit_20261008_181716/
```

审计结果显示，当前三个带 trace 的工作文件在删除本轮观测代码后，均能重建出与观测改动前的 SHA-256 一致的文件：

```text
wheel_leg_mujoco/gbc_mpc_wbc/control.py       match=True
wheel_leg_mujoco/gbc_mpc_wbc/loop.py          match=True
wheel_leg_mujoco/gbc_mpc_wbc/simulation.py    match=True
```

审计目录保存了：

- 原始工作文件快照；
- 观测改动前文件的验证副本；
- 仅观测改动的 unified diff；
- 当前 `git status --short`；
- 当前环境和依赖版本；
- 已生成结果的有效性标记。

## 已执行但作废的结果

```text
wheel_leg_mujoco/results/P0_baseline_smoke01.json.gz
wheel_leg_mujoco/results/P0_baseline_smoke02.json.gz
wheel_leg_mujoco/results/P0_baseline_run01.json.gz
wheel_leg_mujoco/results/P0_baseline_run02.json.gz
wheel_leg_mujoco/results/P6_baseline_run01.json.gz
wheel_leg_mujoco/results/P6_delta_velocity_run01.json.gz
wheel_leg_mujoco/results/P6_qfrc_run01.json.gz
```

作废原因：

1. P0 运行使用了 `preset="none"`，不能直接代表文档要求的当前默认 preset；
2. P6 为区分普通离地链路使用了 `--no-jump`，与P6-C表述的跳跃基线不是同一工况；
3. 诊断工具临时使用了 `|pitch| > 0.01 rad` 事件标记，但规划没有给出该阈值，因此该字段只能作为原始事件辅助，不能作为评价判据；
4. P6 的 `delta_velocity` 和 `qfrc` 首次运行在结果JSON序列化阶段失败，只有部分压缩文件，不能作为结果使用。

上述文件和日志均保留，未删除、未覆盖、未用于根因判断。无外力 P6 结果虽然完整保存，也一并标记为无效，避免与未完成的E1/E2/E3对照混用。

## 当前自检证据

- AST语法检查：通过；
- 导入检查：通过；
- 短时冒烟：通过；
- 重复短时运行：除 `mpc_object_id` 运行实例身份字段外逐字段一致；
- 完整规划口径自检：未通过，因此后续正式结果作废；
- 结果 JSON 读取：P0完整结果可解析，P6首次失败文件不可解析，均已记录在审计目录 `INVALID_RESULTS.json`。

## 下一步边界

在重新执行前，需要先把诊断工具的工况固定为交互文档规定的当前默认配置，明确P6四类输入的统一初始状态、跳跃启用状态和扰动时间；同时把事件字段与评价判据分离，未规定的阈值只保留原始数据，不参与结论。确认后再重新执行第0阶段自检，修复前结果不复用。


---

# 测试agent：第0阶段修复后及第1～2阶段报告（2026-10-08）

## 第0阶段：修复后的自检

### 修复内容

1. 诊断工具恢复使用当前 `SimulationConfig` 默认配置：
   - backend：`wls`；
   - model_variant：`held_leg`；
   - preset：`tuned-agile`。
2. 跳跃目标高度使用方案文档已定的首次 100 mm 输入，不作为调参。
3. 删除未授权的 `|pitch| > 0.01 rad` 评价判据；保留原始 pitch、pitch_rate 和相位时间轴。
4. 修复 `Disturbance` 数据类的JSON序列化。
5. 明确P6普通接触链路使用 `jump_height=None`，与P0跳跃基线分开记录。
6. viewer `xfrc_applied` GUI路径没有用其它路径替代，结果中明确记录为未执行。

### 自检结果

- 语法检查：通过，4个文件；
- 导入检查：通过；
- 环境：Python 3.13.2、MuJoCo 3.13.0、NumPy 2.5.3、SciPy 1.18.1、OSQP 1.1.3；
- 冒烟：完成；
- 同一输入重复运行：完成；
- 重复结果：摘要相同、事件相同、逐拍字段除进程身份字段 `mpc_object_id` 外全部相同，差异字段数为0。

第0阶段通过。

## 第1阶段：当前默认100 mm跳跃基线

### 工况

```text
结果：wheel_leg_mujoco/results/P0_baseline_default_run01.json.gz
日志：wheel_leg_mujoco/results/P0_baseline_default_run01.log
配置：当前默认 wls + held_leg + tuned-agile
jump_height=0.10 m
jump_stroke=0.12 m
jump_time=1.0 s
duration=3.0 s
```

### 摘要数据

```text
frame_count=3000
simulated=3.000 s
MPC calls=150
WBC calls=434
max_abs_pitch=0.529612 rad
max_abs_pitch_rate=7.257747 rad/s
max_forward_displacement=0.726006 m
max_forward_velocity=1.994605 m/s
max_positive_wheel_gap=0.092220 m
airborne_frames=1267
max_abs_ctrl=[40.0, 40.0, 40.0, 40.0, 6.0, 6.0]
phase_counts=IDLE 1000, COMPRESS 226, THRUST 1, FLIGHT 1, LANDING 7, ABORT 1765
```

### 关键事件

```text
COMPRESS       t=1.000 s，双轮几何接触，双轮估计接触
首次几何双轮脱离 t=1.204 s，gap=[+0.000301,+0.000293] m，估计仍为 [True,True]
THRUST         t=1.226 s，gap=[+0.040534,+0.040276] m，法向力=[0,0]
FLIGHT         t=1.227 s，gap=[+0.043893,+0.043628] m，估计仍为 [True,True]
LANDING        t=1.228 s，估计仍为 [True,True]，但几何仍无轮地接触
ABORT          t=1.235 s
估计双轮离地   t=1.250 s，mode=airborne，开始 airborne_zero
```

`FLIGHT` 前一拍已经出现正gap和几何接触消失；当前数据支持“发生真实几何离地”，但不能仅凭该组数据判断姿态发散的唯一根因。`FLIGHT` 后WBC不再调用，最终输出来源转为零阶保持或airborne路径；该控制切换时序已完整保留。

## 第2阶段：P6普通接触链路

P6使用当前默认控制配置、无跳跃输入，扰动输入采用工作区已有文档工况：

- E0：无外力；
- E1：`delta_velocity=(0,0,+0.8)` m/s；
- E2：`qfrc_applied`，`force=(40,0,0)` N、`torque=(0,3,0)` N*m、持续0.2 s；
- E3：viewer `xfrc_applied` 未执行，未用E1/E2替代。

### E0无外力

```text
结果：wheel_leg_mujoco/results/P6_baseline_default_run01.json.gz
日志：wheel_leg_mujoco/results/P6_baseline_default_run01.log
max_positive_wheel_gap=-0.000100 m
airborne_frames=0
max_abs_pitch=0.000212 rad
mode始终未进入airborne
```

E0未出现几何离地或估计离地。

### E1速度冲量

```text
结果：wheel_leg_mujoco/results/P6_delta_velocity_default_run02.json.gz
日志：wheel_leg_mujoco/results/P6_delta_velocity_default_run02.log
冲量时刻：t=1.0 s
首次几何双轮脱离：t=1.001 s
首次估计双轮离地：t=1.049 s
首次 airborne：t=1.049 s
首次几何重新出现：t=1.093 s，先单轮后双轮
估计接触恢复：约t=1.104 s，进入 recovery/single_support路径
airborne_frames=55
max_positive_wheel_gap=0.010836 m
max_abs_pitch=0.247461 rad
```

该组几何接触、法向力和估计接触的时间顺序可区分，说明确实发生了短暂真实离地；接触恢复后WBC重新调用，未表现为“airborne一直未退出”。

### E2脚本广义外力

```text
结果：wheel_leg_mujoco/results/P6_qfrc_default_run02.json.gz
日志：wheel_leg_mujoco/results/P6_qfrc_default_run02.log
外力开始：t=1.000 s
外力结束：t=1.201 s
首次几何双轮脱离：t=2.028 s
首次估计双轮离地/airborne：t=2.519 s
首次几何重新出现：t=2.773 s，仍为单轮几何接触
仿真结束：t=3.000 s，仍处于airborne，未完成双轮接触恢复
airborne_frames=278
max_positive_wheel_gap=0.281849 m
max_abs_pitch=0.870282 rad
```

E2在当前3 s窗口内没有形成完整的“离地-重接触-恢复GROUND”闭环，因此该组离地后恢复过程尚未解释。

## 阶段停止结论

- 第0阶段：通过；
- 第1阶段：数据链路有效，确认存在真实几何离地及后续控制切换；
- 第2阶段：E0和E1时序可重建，E2尚未完成重接触解释，E3未执行；
- 按停止规则，暂不进入第3阶段P5、第4阶段MPC因果隔离和第5阶段执行器映射；
- 当前不对MPC、WBC、跳跃状态机或外力扣除链路下根因结论。

原始逐拍数据字段、单位和运行环境均记录在对应 `.json.gz` 文件的 `metadata` 中；P6关键状态变化另存于：

```text
wheel_leg_mujoco/results/P6_key_events.log
wheel_leg_mujoco/results/P6_transition_report_compact.log
```


---

# 测试agent：P6-E2延长观察窗口报告（2026-10-08）

## 工况与授权

依据阶段2补测指令，经用户授权，将原P6-E2工况从3.0 s延长到6.0 s。除观察时长外保持完全一致：

```text
配置：wls + held_leg + tuned-agile
jump_height=None
外力开始：t=1.0 s
force=(40,0,0) N
 torque=(0,3,0) N*m
duration=0.2 s
```

原3.0 s结果未覆盖：

```text
wheel_leg_mujoco/results/P6_qfrc_default_run02.json.gz
```

本次新结果：

```text
wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.json.gz
wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.log
```

## 工具和结果复核

- 诊断工具语法/导入：通过；
- Python：3.13.2；
- MuJoCo：3.13.0；
- JSON重新读取：通过；
- 逐拍帧数：6000；
- 仿真时长：6.000 s；
- 序列化无错误；
- 未修改控制参数、模型参数、阈值、限幅和外力输入。

## 关键时序

```text
外力开始             t=1.000 s
外力结束             t=1.201 s
首次几何双轮脱离     t=2.028 s
首次估计双轮离地     t=2.519 s
airborne开始         t=2.519 s
首次几何单轮重新接触 t=2.773 s
首次估计单轮接触     t=2.797 s
首次几何双轮接触     t=4.387 s
仿真结束             t=6.000 s
```

## 结果摘要

```text
frame_count=6000
airborne_frames=2136
max_abs_pitch=0.870282 rad
max_abs_pitch_rate=9.063302 rad/s
max_positive_wheel_gap=0.281849 m
max_abs_ctrl=[40,40,40,40,6,6]
WBC calls=966
MPC calls=300
```

关键末状态：

```text
几何接触：两轮均为 True
法向力：两轮均为 0
contact_estimator：两轮均为 False
监督模式：airborne
airborne_active：True
WBC：未调用
输出来源：airborne_zero
```

## 阶段2停止判定

本次延长窗口触发以下停止条件：

```text
几何上已恢复双轮接触；
但法向力重构仍为零；
contact_estimator未恢复；
airborne_active未退出；
WBC未恢复调用；
未回到RECOVERY或GROUND。
```

因此当前证据支持：

```text
真实离地
-> 几何接触后法向力重构未恢复
-> 接触估计未恢复
-> airborne状态未退出
-> WBC未重新接管
```

这属于接触力重构/接触估计恢复链路问题的有效证据，但不据此判断MPC为根因。

## 下一步边界

- P6-E2已完成并解释到停止条件；
- P6-E3 viewer `xfrc_applied`仍未执行；
- 由于P6-E2暴露出“几何已接触但法向力与估计未恢复”，暂不执行P5、MPC隔离或执行器映射；
- 等待问题排查agent给出新的书面指令。

---

# 问题排查agent：阶段2延长补测审批与下一步安排（2026-10-08）

## 一、审批结论

### 已批准：P6-E2延长观察窗口结果有效

批准文件：

```text
wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.json.gz
wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.log
```

批准理由：

- 延长时长已获得用户授权；
- 外力、初始状态、控制配置和全部控制参数保持不变；
- 工具语法/导入、JSON重读、逐拍帧数和序列化均通过；
- 结果与原3 s工况的关键事件一致，并完成了更长时间的恢复观察。

### 阶段性确认

1. P6-E0无外力：无离地，无 airborne，基线有效。
2. P6-E1速度冲量：发生真实短暂离地，接触估计恢复，WBC可以重新调用；不存在普遍性的“airborne永不退出”。
3. P6-E2脚本广义外力：真实离地后，先单轮几何接触，随后双轮几何接触；但法向力一直为零，接触估计没有恢复，`airborne_active`保持 True，WBC不再接管。

### 已确认的恢复死锁

P6-E2在约 `4.387 s` 已出现双轮几何接触，但到 `6.000 s` 仍为：

```text
几何接触：双轮 True
法向力：两轮 0
contact_estimator：两轮 False
监督模式：airborne
airborne_active：True
WBC：未调用
最终输出：airborne_zero
```

因此当前可以确认：

```text
几何接触恢复
-> 法向力反解没有恢复支撑力
-> contact_estimator无法恢复
-> airborne_active无法退出
-> WBC无法重新接管
-> 输出持续为零
```

这不是“接触恢复逻辑普遍失效”，而是当前控制输出策略与接触力恢复条件之间存在工况相关的闭环死锁。根因仍需区分：

- 几何接触只是擦边/未产生有效支撑；
- 零力矩输出导致没有恢复法向力的动力学条件；
- 接触力反解口径在该姿态下失效；
- 单轮恢复后没有有效的恢复控制；
- 或上述因素共同作用。

## 二、当前不批准的测试

以下测试继续冻结：

- 第3阶段 P5 释放段腿摆角与pitch链路；
- 第4阶段 MPC参与因果隔离；
- 第5阶段执行器映射与符号核对；
- 任何核心算法、状态机、接触阈值、确认时间、力矩限幅或控制参数修改。

原因：P6-E2已经暴露出重接触恢复死锁，必须先确认 viewer 外力路径是否产生不同的接触力重构结果，避免把外力路径问题误归因于MPC。

## 三、下一步唯一批准项目：P6-E3 viewer外力路径

### 测试目的

验证 viewer `xfrc_applied` 路径与脚本 `qfrc_applied` 路径的接触力反解是否一致，重点检查外力是否被正确扣除，以及离地/重接触状态是否发生不同步。

### 固定条件

- 使用当前默认控制配置：`wls + held_leg + tuned-agile`；
- `jump_height=None`，不启用跳跃状态机；
- 初始状态与P6-E2一致；
- 外力施加时间：`t=1.0 s`；
- 外力方向、大小、力矩和持续时间必须与P6-E2一致：
  `force=(40,0,0) N`、`torque=(0,3,0) N*m`、持续 `0.2 s`；
- 只改变外力注入通道为真实 viewer `xfrc_applied`；
- 不得用 `qfrc_applied`、`delta_velocity` 或预设结果替代 viewer路径；
- 不得修改任何控制参数、模型参数、接触阈值、确认时间、限幅和评价判据；
- 观察时长使用与P6-E2相同的 `6.0 s`，除非测试工具无法执行并先报告原因。

### 必须记录

```text
qfrc_applied[0:6]
xfrc_applied中实际受力body和wrench
right/left wheel gap
right/left contact_geometry
right/left normal_forces
right/left tangent_forces
contact_estimator.contact
decision.mode
airborne_active
WBC调用、输出写回、output_source
pitch、pitch_rate
```

### 必须报告的事件时间

```text
外力开始
外力结束
首次几何双轮离地
首次估计双轮离地
airborne开始
首次几何单轮接触
首次估计单轮接触
首次几何双轮接触
首次估计双轮接触
airborne退出
WBC恢复调用
恢复RECOVERY
恢复GROUND
```

### P6-E3停止条件

- viewer路径无法自动化复现：标记“未执行”，不得用其它路径替代；
- 工具字段、序列化或复现性异常：结果作废，停止并报告；
- 发生几何接触但法向力仍为零：停止，不延长或改变其它条件；
- 结果与P6-E2有差异：保留完整对照表，不直接判定viewer或qfrc哪条正确；
- P6-E3完成后，测试agent暂停，等待问题排查agent审批下一阶段。

## 四、P6-E3完成后的进入条件

只有P6-E3结果有效并完成对照后，问题排查agent才决定是否进入：

1. 接触恢复链路专项分析；或
2. 第3阶段P5释放段pitch/腿摆角链路。

MPC因果隔离暂不自动批准，必须在接触恢复问题完成归因后重新书面审批。

当前指令：**批准P6-E2延长结果；只执行P6-E3 viewer路径；其余测试和代码修改全部暂停。**

---

# 问题排查agent：跳跃失败快速根因排查方案（2026-10-08）

## 一、重新定义目标

本轮不以“把所有安全边界逐项验证完”为目标，也不以优化算法、增加保护或提高成功率为目标。

唯一目标：用最少的因果对照，确认导致当前跳跃失败的首要原因，并决定应该修复：

1. MPC姿态参考/预测链路；
2. 跳跃状态机与接触估计时序；
3. 两者叠加；
4. 其它控制映射问题。

## 二、已有证据排序

### 证据1：MPC在真实离地前已经给出异常pitch任务

P0基线逐拍数据：

- `1.180 s`：仍接地，MPC/WBC pitch目标约 `-76.93 rad/s²`；
- `1.200 s`：pitch目标突然反向到约 `+74.36 rad/s²`；
- `1.220 s`：已经几何离地，pitch目标再次约 `-77.74 rad/s²`。

这说明pitch发散不是完全由离地之后引起，MPC/WBC姿态参考在释放前已经强烈参与。

### 证据2：真实离地后状态机和接触估计不同步

- 约 `1.206 s` 两轮几何已经离地，法向力为零；
- 接触估计仍为双轮接地；
- 相位仍为 `COMPRESS`；
- `1.227 s` 才进入 `FLIGHT`；
- `1.228 s` 因过期接触状态立即进入 `LANDING`；
- 离地后接地型WBC再次被调用。

这是第二个高可信原因，但它发生在MPC异常参考已经出现之后。

### 证据3：P6-E2恢复死锁不是当前跳跃首要原因

P6-E2是在 `jump_height=None` 的普通接触链路下、脚本持续广义外力造成大姿态后观察到的。它证明了普通 airborne 恢复存在死锁，但不能解释P0中 `1.180～1.228 s` 的首次pitch发散。因此P6-E3和普通恢复问题暂不作为当前跳跃失败的第一优先级。

## 三、快速排查顺序

```text
已有P0基线，不重跑
    ↓
Q1：跳跃期断开MPC sample/求解
    ↓
Q2：保持MPC，诊断性修正离地/FLIGHT-LANDING时序
    ↓
Q3：仅当Q1/Q2均有改善时做组合确认
    ↓
再决定是否处理P6-E3普通恢复链路
```

不再要求在Q1/Q2之前执行viewer外力测试、执行器映射测试或全面边界测试。

## 四、Q1：MPC参与因果隔离，最高优先级

### 目的

判断MPC是否是释放前pitch发散的必要原因。

### 唯一变化

从跳跃开始到跳跃结束，诊断性地不把MPC `solution/sample` 送入WBC；可以继续记录MPC，但WBC使用无sample分支。保持以下内容完全不变：

- 跳跃状态机；
- 接触估计器；
- WBC增益、权重和限幅；
- 腿长目标；
- `tau_jump`；
- 外力和初始状态；
- 仿真时长和评价字段。

这不是调参，也不是最终修复，只是隔离MPC控制权限的诊断对照。

### 必须比较

与已有P0基线逐项比较：

```text
1.180～1.230 s内的pitch、pitch_rate
首次pitch偏移时间
首次几何离地时间
WBC pitch desired_acceleration
angle_R/L desired_acceleration
length_R/L desired_acceleration
右左腿最终torque
前向位移和速度
FLIGHT/LANDING/ABORT时间
```

### 判定

- Q1明显消除释放前pitch发散，且几何离地前姿态保持明显改善：确认MPC姿态/sample参与是首要原因；
- Q1只减少发散幅度但仍在真实离地后崩溃：MPC是前置原因，状态机/接触时序是后置原因；
- Q1与P0在1.206 s前几乎一致：MPC不是首要原因，进入Q2；
- Q1结果工具不一致或改变了其它控制通道：结果作废，不得继续推理。

## 五、Q2：接触/状态机时序因果隔离

### 前置条件

Q1完成并且结果有效。Q2不与Q1同时改变，避免无法区分原因。

### 唯一变化

保持P0的MPC sample和WBC任务不变，只做诊断性状态门控：

1. 一旦两轮几何均无轮地接触且两轮signed gap为正，立即记录并进入飞行控制路径；
2. 几何仍显示双轮离地时，禁止使用滞后的 `[True,True]` 接触估计触发 `FLIGHT -> LANDING`；
3. 几何离地期间不调用接地型WBC；
4. 不修改接触阈值、确认时间和力矩参数；
5. 该开关只用于因果隔离，不作为最终修复提交。

### 必须比较

```text
真实几何离地时间
FLIGHT进入时间
FLIGHT持续时间
LANDING进入时间
离地期间WBC调用次数
pitch、pitch_rate
腿部torque峰值
ABORT时间
```

### 判定

- Q2明显消除 `FLIGHT 1 ms -> LANDING` 和离地后pitch继续恶化：确认状态机/接触时序是首要后置原因；
- Q2仍在1.206 s之前出现同样pitch发散：MPC问题独立存在；
- Q2只改善离地后的姿态，不能改善1.180～1.206 s：状态机是次要原因，MPC是前置原因。

## 六、Q3：必要时做组合确认

只有以下情况才执行Q3：

- Q1和Q2单独都改善，但无法判断两者是否互相放大；
- 或Q1只改善释放前、Q2只改善释放后，需要确认组合后是否消除完整失败链。

Q3同时采用Q1和Q2的诊断隔离，仍不修改参数。若Q3能消除“释放前pitch发散 + 离地后状态错配”，才可以进入修复设计。

## 七、暂缓项目

以下项目全部降级，不作为当前第一轮测试：

1. P6-E3 viewer `xfrc_applied`：只解释普通外力恢复链路，不解释P0首次跳跃失败；
2. P4执行器映射：当前P0左右腿输出和几何变化仍需保留为后续排除项，但不是最高信息增益测试；
3. 腿长网格/MPC模型全面扫描：在确认MPC是否直接致因前，不做参数拟合优劣分析；
4. 任何调参、阈值修改、力矩限幅修改或保护逻辑增强。

## 八、测试agent执行要求

- 不重跑已有P0基线；
- 先执行Q1，只允许一个隔离变量；
- Q1完成后暂停，返回完整对照表；
- 未收到问题排查agent根据Q1结果给出的书面指令，不得执行Q2；
- Q2完成后暂停，不得自行执行Q3或进入修复；
- 所有诊断性门控必须标记为“仅测试隔离”，不得混入正式算法版本；
- 任何工具问题、字段缺失或结果不一致，立即停止并报告。

当前批准状态：**批准Q1；Q2待Q1结果审批；Q3条件触发；P6-E3及其它项目暂缓。**

---

# 问题排查agent：修正Q1方案——MPC输出审计与显式跳跃参考隔离（2026-10-08）

## 一、撤销此前不完整的Q1

此前“跳跃期间不把MPC `solution/sample` 送入WBC、让WBC走 `sample=None` 分支”的方案不再执行，原因如下：

- `sample=None` 不是一个规划好的跳跃参考；
- WBC会进入默认分支，自动把pitch/yaw参考置为零，并使用内部默认阻尼/任务逻辑；
- 这会同时改变参考存在性、任务计算分支和控制输入语义；
- 即使结果变好，也不能证明“仅仅断开MPC”是原因。

因此旧Q1结果若有执行，不能作为正式因果结论。新的Q1分为“只读审计”和“显式参考隔离”两步。

## 二、Q1-A：先只读审计MPC实际下发值

### 目的

不改变控制链路，直接确认MPC在压缩/释放过程中给WBC的参考是否脱离正常平衡姿态，以及异常值首次出现的时间。

### 使用已有P0结果优先

不必先重跑。优先读取：

```text
wheel_leg_mujoco/results/P0_baseline_default_run01.json.gz
```

已有逐拍字段足以审计：

```text
sample_raw.state
sample_raw.acceleration
sample.state
sample.acceleration
wbc_task_trace
pitch
pitch_rate
leg_state
leg_rate
wheel_gap
normal_forces
phase
```

### 必须区分三类值

1. **MPC原始输出**：`sample_raw.state`、`sample_raw.acceleration`；
2. **跳跃接线改写后的值**：`sample.state`、`sample.acceleration`；
3. **WBC实际任务目标**：`wbc_task_trace` 中的 `pitch`、`angle_R/L`、`length_R/L`。

不能只看最终 `data.ctrl` 推断MPC值是否异常。

### 审计重点

- `sample_raw.state[2]`：MPC pitch参考；
- `sample_raw.state[3]`：MPC pitch-rate参考；
- `sample_raw.acceleration[1]`：MPC pitch加速度前馈；
- `sample_raw.state[6:8]`、`sample_raw.acceleration[3]`：腿摆角参考和前馈；
- 改写后这些值是否与原始值不同；
- WBC pitch任务是否直接继承了异常 `sample.acceleration[1]`；
- 异常值出现时是否仍双轮接触、是否尚未真实离地。

### 已有证据的初步结论

P0中至少存在以下现象：

```text
1.180 s：sample.acceleration[1] ≈ -76.93 rad/s²
1.200 s：sample.acceleration[1] ≈ +74.36 rad/s²
1.220 s：sample.acceleration[1] ≈ -77.74 rad/s²
```

这些值已经在真实离地之前进入WBC pitch任务。该结论来自已有日志，不需要先做断开实验。

## 三、Q1-B：用显式跳跃参考替代MPC sample

### 目的

在不让WBC退回 `sample=None` 默认分支的前提下，隔离“当前MPC参考是否导致失败”。

### 唯一变化

跳跃 active 期间，不使用MPC的运动参考，但向WBC传入一个明确构造的跳跃参考对象。所有WBC任务、增益、权重、限幅、腿长目标、接触逻辑和`tau_jump`保持不变。

### 诊断参考的定义

参考必须是显式、可记录、可复现的跳跃参考，而不是空值。第一版诊断参考按以下语义构造：

#### 6状态模型

```text
position      = 起跳时的当前位置保持值
velocity      = 0
pitch         = 0
pitch_rate    = 0
yaw           = 起跳时的当前yaw保持值
yaw_rate      = 0

acceleration  = [0, 0, 0, 0]
```

#### 8状态模型

在上述6个状态后追加：

```text
leg_angle     = 0
leg_angle_rate= 0
```

即：

```text
state[6] = 0
state[7] = 0
acceleration[3] = 0
```

该参考表示“跳跃期间保持水平、保持当前水平位置/偏航，不接受MPC的姿态和腿摆角规划”，不表示最终修复方案。

### 为什么不能直接用零向量

- `pitch=0`、`pitch_rate=0`是明确姿态目标；
- `position=当前值`、`velocity=0`避免引入新的水平位移目标；
- `yaw=当前值`避免把当前偏航误差混入诊断；
- `leg_angle=0`保持现有跳跃代码的腿摆角解耦契约；
- `acceleration`全零，排除MPC加速度前馈的影响。

### Q1-B必须记录

```text
显式诊断参考state/acceleration
WBC实际任务目标
pitch/pitch_rate
leg_angle_R/L及速率
leg_length_R/L及速率
几何gap和法向力
JumpPhase及contact状态
WBC原始torque
最终data.ctrl
FLIGHT/LANDING/ABORT时间
```

### Q1-B判定

- Q1-A显示MPC pitch参考异常，Q1-B明显消除释放前pitch发散：MPC参考是高可信首要原因；
- Q1-B只消除释放前发散，但真实离地后仍因状态错配发散：MPC是前置原因，接触/状态机是后置原因；
- Q1-B与P0在释放前基本一致：MPC参考不是首要原因，转入状态机隔离；
- Q1-B出现新的任务缺失、维度错误或参考语义变化：结果作废，不得解释为控制效果。

## 四、修正后的执行顺序

```text
1. Q1-A：读取已有P0，完成MPC原始/改写/WBC任务三层对齐
2. 问题排查agent审阅Q1-A结果
3. 批准后执行Q1-B：显式跳跃参考隔离
4. 问题排查agent审阅Q1-B结果
5. 只有必要时执行Q2：接触/状态机时序隔离
```

测试agent不得：

- 执行 `sample=None` 版本作为正式Q1；
- 用全零/空对象代替显式跳跃参考；
- 同时改变MPC参考、状态机、接触估计和WBC任务；
- 在Q1-A未审阅前执行Q1-B；
- 在Q1-B未审阅前执行Q2。

当前批准状态：**撤销旧Q1；批准Q1-A只读审计；Q1-B需Q1-A审阅后再批准。**


---

# 测试agent：Q1-A只读MPC输出审计报告（2026-10-08）

## 执行边界

- 使用已有 `P0_baseline_default_run01.json.gz`；
- 未重跑P0；
- 未修改控制算法、状态机、接触估计、参数或限幅；
- 未执行旧版Q1 `sample=None`隔离；
- 未执行Q1-B、Q2或其它测试。

审计结果文件：

```text
wheel_leg_mujoco/results/Q1A_mpc_audit_default_run01.json
```

## 审计窗口

```text
t = 1.180 s ～ 1.230 s
```

窗口内逐拍字段可读取，包含：

- `sample_raw.state/acceleration`；
- 跳跃改写后的 `sample.state/acceleration`；
- WBC `pitch/angle_R/angle_L/length_R/length_L`任务目标；
- pitch、pitch_rate；
- wheel gap、几何接触、法向力和估计接触；
- WBC原始力矩、`tau_jump`和最终ctrl。

## 关键数值

| 时刻 | 几何/估计接触 | pitch | pitch_rate | MPC原始pitch前馈 | WBC pitch目标 |
|---:|---|---:|---:|---:|---:|
| 1.180 s | 几何双轮、估计双轮 | -0.021784 | -0.304346 | -76.934760 | -76.934760 |
| 1.200 s | 几何双轮、估计双轮 | -0.035259 | -1.025442 | +74.360024 | +74.360024 |
| 1.220 s | 几何无接触、估计双轮 | -0.057634 | -1.403151 | -77.737232 | -77.737232 |
| 1.226 s | 几何无接触、估计双轮 | -0.069241 | -2.361823 | WBC未调用 | WBC未调用 |
| 1.227 s | 几何无接触、估计双轮 | -0.071795 | -2.554238 | WBC未调用 | WBC未调用 |
| 1.228 s | 几何无接触、估计双轮 | -0.074543 | -2.748452 | -78.099369 | -70.624368 |

窗口内：

```text
MPC原始pitch acceleration范围：[-78.099369, +74.360024] rad/s^2
WBC pitch desired acceleration范围：[-83.651597, +88.530899] rad/s^2
MPC原始pitch前馈符号切换：约1.196 -> 1.200 s、约1.216 -> 1.220 s
WBC pitch目标符号切换：与上述两次切换同步
```

## 跳跃sample改写审计

窗口内共13个WBC sample发生跳跃改写：

- `state[6:8]`被清零；
- `acceleration[3]`被清零；
- `state[0:6]`和`acceleration[0:3]`未被跳跃接线改写。

因此当前实现中，跳跃期间腿摆角参考被断开，但MPC pitch参考、pitch-rate参考和pitch加速度前馈仍然原样进入WBC pitch任务。

同时观察到：

```text
1.180 s：angle_R/L约 +8.48，length_R/L约 -0.69/-0.78
1.200 s：angle_R/L约 +5.34，length_R/L约 -5.07/-5.09
1.220 s：angle_R/L约 +39.57/+39.25，length_R/L约 +44.67/+44.26
```

1.220 s时已几何离地，但监督模式仍为`ground`，WBC仍按接地模型求解。

## Q1-A结论

只读证据确认：

1. MPC原始pitch加速度前馈在真实几何离地前已发生大幅反向切换；
2. WBC pitch任务直接继承该前馈值；
3. 跳跃接线只清零腿摆角参考，没有隔离MPC pitch参考；
4. 该现象与几何离地和接触估计滞后同时存在，但Q1-A不改变控制链路，因此不能单凭审计判断MPC是否为必要原因。

当前结论：**Q1-A有效，MPC姿态参考参与得到确认；等待问题排查agent审批Q1-B显式跳跃参考隔离。**

---

# 问题排查agent：Q1-A审阅结论与Q1-B正式审批（2026-10-08）

## 一、Q1-A审阅结论

Q1-A只读审计有效，确认：

1. 跳跃接线只清零了腿摆角相关的MPC参考；
2. MPC pitch参考、pitch-rate参考和pitch加速度前馈仍然进入WBC；
3. 在真实几何离地前，MPC pitch前馈已经在约 `-78` 到 `+74 rad/s²` 之间快速反向切换；
4. WBC pitch任务直接继承了这些值；
5. 异常与主动腿长变化和腿长调度同时发生。

### 证据边界

当前可以确认：

```text
主动腿长变化
-> MPC仍按接地运动模型继续预测
-> MPC输出异常pitch参考/前馈
-> WBC直接执行异常pitch任务
```

当前还不能单独确认：

```text
异常究竟来自：
A. 不应在跳跃期间继续使用接地型MPC；
B. 腿长变化后的A/B拟合或调度错误；
C. A+B共同作用。
```

因此暂不回退或删除腿长拟合基础设施，也不把“腿长拟合错误”直接定为最终根因。

## 二、正式批准Q1-B：跳跃专用显式WBC参考

### 目的

验证“跳跃期间不让MPC控制输出进入WBC，由跳跃模块给WBC显式目标”是否能消除释放前的pitch发散。

### 控制权限定义

Q1-B期间：

- MPC可以继续运行并记录原始输出，作为影子诊断；
- MPC输出不得进入WBC的运动/姿态参考；
- WBC必须继续运行，不能传 `sample=None`；
- 跳跃模块必须提供显式、完整、可记录的参考对象；
- 跳跃状态机、接触估计、WBC参数、腿长目标、`tau_jump`、执行器限幅和模型参数全部保持当前值；
- 不修复状态机、不改变离地判据，确保本轮只隔离MPC参考权限。

### 显式参考内容

#### 6状态模型

```text
state[0] = 起跳时当前位置保持值
state[1] = 0
state[2] = 0                  # pitch目标
state[3] = 0                  # pitch_rate目标
state[4] = 起跳时当前yaw保持值
state[5] = 0                  # yaw_rate目标
acceleration[:] = 0
```

#### 8状态模型

```text
state[6] = 0                  # leg_angle目标
state[7] = 0                  # leg_angle_rate目标
acceleration[3] = 0
```

该参考不是最终控制方案，而是“跳跃姿态由WBC控制、MPC只保留影子记录”的因果隔离输入。

### Q1-B必须输出

与P0基线逐项对照：

```text
MPC shadow原始sample.state/acceleration
显式跳跃参考state/acceleration
WBC实际pitch/angle/length任务目标
pitch、pitch_rate
首次pitch偏移时间
首次几何离地时间
前向位移和速度
leg_angle_R/L及速率
FLIGHT/LANDING/ABORT时间
WBC原始torque
最终data.ctrl
contact_geometry、normal_forces、contact_estimator
```

### Q1-B判定标准

- **结果A**：释放前pitch明显稳定，说明MPC参考进入WBC是首要前置原因；若离地后仍因状态错配发散，则确认“前置MPC问题 + 后置接触状态问题”双重链路。
- **结果B**：释放前和离地后均明显改善，说明MPC参考是当前失败的主导原因；状态机问题仍需修复，但不是本次发散的首要触发点。
- **结果C**：与P0释放前基本一致，说明MPC输出虽然异常，但不是必要原因，转入状态机/接触时序隔离。
- **结果D**：显式参考维度、任务分支或输出语义不一致，结果作废，不得作为控制结论。

## 三、对“跳跃独立解耦状态”的初步评价

你的架构建议是合理的，推荐目标架构为：

```text
GROUND
  MPC + WBC正常控制

JUMP_PREPARE
  触发跳跃后冻结MPC对WBC的输出权限
  WBC接收跳跃准备参考

THRUST
  跳跃专用显式参考 + WBC/跳跃前馈
  MPC只做影子记录，不参与执行

FLIGHT
  无接触控制，不使用接地型MPC/WBC约束

LANDING
  使用跳跃着地控制和接触恢复逻辑

RECOVERY
  双轮接触稳定后重新锚定位置、姿态和参考状态
  再恢复MPC输出权限
```

### 关于“跳跃时不需要腿长拟合”的边界

如果Q1-B证明跳跃期间隔离MPC即可恢复主要姿态行为，那么：

- 跳跃期间确实不需要让MPC使用不同腿长的拟合参数；
- 可以冻结或旁路腿长调度，避免主动腿长变化污染接地MPC；
- 接地行驶期间是否仍需要腿长拟合，不能由跳跃结果推出；
- 普通接地状态下的不同腿长模型仍需单独验证，不能直接全部删除。

因此合理的工程方向是：

```text
接地运动：保留并单独验证腿长模型调度
跳跃运动：MPC旁路，跳跃WBC使用显式目标
着地恢复：先重锚定，再恢复MPC和腿长调度
```

## 四、Q1-B执行约束

- 测试agent只执行Q1-B，不执行Q2、Q3、P6-E3或任何调参；
- 必须保留MPC shadow输出，不能停止记录，否则无法确认跳跃期间MPC实际产生了什么；
- 必须确认WBC收到的是显式参考对象，不是 `None`；
- 不得同时修改状态机离地条件，否则结果无法区分MPC解耦和状态机修复；
- 结果必须与P0基线使用相同输入、初始状态、持续时间和评价字段；
- 测试完成后暂停，等待问题排查agent审阅。

当前批准状态：**Q1-A有效；正式批准Q1-B显式跳跃参考隔离；暂不批准腿长拟合回退、状态机修复或最终架构代码修改。**

---

# 问题排查agent：跳跃功能具体修复方案（待用户批准，2026-10-08）

## 一、修复目标

修复目标不是让当前参数“更容易跳起来”，也不是增加保护阈值，而是消除以下已确认的控制权限错误：

```text
主动腿长变化
-> 接地型MPC仍继续预测并输出姿态/加速度参考
-> WBC直接执行异常pitch任务
-> 真实离地后接触估计仍滞后
-> 跳跃状态机错误切换或离地后继续使用接地WBC
-> pitch和腿部控制发散
```

## 二、推荐总体架构

将跳跃作为独立的控制模式，而不是只在普通WBC中叠加一组跳跃力矩：

```text
GROUND
  MPC + 接地WBC

JUMP_PREPARE
  MPC可影子运行，但输出不再进入WBC
  WBC接收显式跳跃准备参考

JUMP_THRUST
  显式跳跃参考 + 跳跃腿长任务 + WBC平衡
  MPC不参与执行，不进行腿长模型换挡

JUMP_FLIGHT
  无接触飞行控制
  不使用接地型MPC或接地约束WBC

JUMP_LANDING
  几何接触触发着地控制
  使用着地/阻抗/单支撑控制恢复接触力

JUMP_RECOVERY
  双轮接触和姿态稳定后重锚定MPC
  恢复MPC和腿长模型调度
```

## 三、具体修改边界

### F1：建立跳跃控制权限边界

增加明确的跳跃控制上下文，至少包含：

```text
普通MPC输出是否允许进入WBC
跳跃显式参考
跳跃是否允许腿长模型调度
跳跃是否允许接地WBC
当前输出所有权：MPC / WBC / Jump / Flight
```

不要用 `jump.active` 同时表达“跳跃正在进行”“WBC是否接管”“是否有接触”“是否允许MPC输出”多个语义。

### F2：跳跃期间旁路MPC执行输出

跳跃触发后：

- MPC可以继续求解并记录，作为shadow诊断；
- `sample_raw`继续记录；
- MPC的`sample.state`和`sample.acceleration`不得进入跳跃WBC任务；
- 不调用或不执行`_schedule_mpc_leg_length()`对跳跃控制产生作用；
- 不删除腿长网格、能力表和MPC模型基础设施；
- 跳跃结束并完成MPC重锚定后，才恢复MPC输出和腿长调度。

### F3：建立显式JumpReference，不允许用`sample=None`

跳跃WBC必须收到完整参考对象。

#### 6状态参考

```text
position      = 起跳时位置保持值
velocity      = 0
pitch         = 0
pitch_rate    = 0
yaw           = 起跳时yaw保持值
yaw_rate      = 0
acceleration  = [0, 0, 0, 0]
```

#### 8状态参考

```text
state[6] = 0
state[7] = 0
acceleration[3] = 0
```

腿长目标不塞入MPC状态参考，而由跳跃任务单独覆盖`length_R/L`目标。这样可以明确区分：

```text
姿态/水平保持：JumpReference
腿长伸缩：JumpPlanner
接触约束和力矩分配：WBC
```

### F4：修正跳跃状态机的接触证据链

当前问题是用滞后的接触估计直接决定飞行/着地，而几何已经显示真实离地。建议改成：

```text
THRUST
  几何双轮离地事件 -> FLIGHT_PENDING

FLIGHT_PENDING
  确认几何双轮离地 -> FLIGHT
  期间禁止接地WBC

FLIGHT
  几何任一轮重新接触 -> LANDING_PENDING
  不使用旧的contact_estimator状态直接进入LANDING

LANDING_PENDING
  启用着地/单支撑恢复控制
  法向力只用于确认支撑是否恢复，不用于决定是否开始着地控制

LANDING/RECOVERY
  双轮几何接触 + 法向力恢复 + 姿态稳定 -> RECOVERY
```

关键原则：

- 几何量负责快速识别“离开/重新接触”；
- 法向力负责确认“已经形成有效支撑”；
- 两者不能都作为唯一状态切换依据；
- 不允许在等待法向力期间把所有输出置零，否则会复现P6-E2的恢复死锁。

### F5：修正 airborne 输出策略

当前P6-E2显示：几何接触恢复后仍因法向力为零保持`airborne_zero`，系统没有机会主动恢复支撑力。

建议输出所有权改为：

```text
真正飞行且无几何接触：FlightController
几何重新接触但法向力未恢复：LandingController / 单支撑WBC
双轮支撑稳定：RecoveryWBC
```

不能使用“法向力恢复前持续零力矩”作为唯一策略。

### F6：MPC重锚定和恢复

重新接触后，不立即恢复旧MPC参考。恢复顺序应为：

1. 等待双轮几何接触；
2. 等待有效法向力和姿态稳定；
3. 以当前测量位置、速度、yaw和腿长建立新的MPC初始状态；
4. 清理跳跃期间的旧参考反馈、旧饱和反馈和旧预测轨迹；
5. 重置`pitch_bias`/参考累积量；
6. 重新选择当前腿长对应的MPC模型；
7. 恢复MPC输出权限。

## 四、腿长拟合的处理决定

当前不回退、不删除腿长网格和能力表，理由是证据只证明：

```text
跳跃期间使用接地型MPC输出会产生异常姿态参考
```

尚未证明：

```text
接地状态下的腿长拟合模型本身错误
```

修复后的职责划分为：

```text
接地行驶：MPC可以使用腿长调度，后续单独验证
跳跃阶段：旁路MPC执行输出，不使用腿长调度影响WBC
着地恢复：稳定后重新选择腿长模型并恢复MPC
```

这样即使腿长拟合仍有问题，也不会继续污染跳跃控制；同时不会因为一次跳跃故障误删接地控制所需的模型基础设施。

## 五、推荐实施顺序

### R1：先做诊断隔离，不改最终架构

执行已批准的Q1-B：

- MPC继续shadow记录；
- 显式JumpReference进入WBC；
- 不改接触状态机；
- 判断释放前pitch是否改善。

### R2：实现控制权限解耦

如果Q1-B确认改善，正式实现F1～F3：

- 跳跃控制上下文；
- MPC旁路；
- 显式JumpReference；
- 跳跃期间停止MPC腿长调度对执行的影响。

### R3：修正离地/着地状态交接

再实现F4～F5：

- 几何离地进入飞行；
- 几何重接触进入着地控制；
- 法向力只做有效支撑确认；
- 不再用零力矩等待法向力。

### R4：恢复MPC

最后实现F6，完成着地后的MPC重锚定和模型调度恢复。

## 六、验收顺序

每一步只验证一个新增控制权限变化：

1. P0当前基线；
2. Q1-B显式JumpReference；
3. F1～F3正式跳跃MPC旁路；
4. F4～F5离地/着地交接；
5. F6着地后MPC重锚定；
6. 最后才重新验证腿长拟合在普通接地运动中的必要性。

在R1结果出来前，不修改核心算法。当前正式修复方案状态：**方案已提出，Q1-B先验证，尚未授权代码实现。**

---

# 问题排查agent：Q1-B增加WBC实际作用审计（2026-10-08）

## 一、补充原则

Q1-B不能只比较最终pitch和跳跃视频，必须先确认WBC在每个关键阶段是否真正计算、输出并写入执行器。

否则“显式JumpReference效果不好”可能有以下完全不同的原因：

```text
WBC根本没有调用
WBC调用但监督模式禁止balance
WBC调用但求解不可行/任务被缩放为0
WBC输出被airborne_zero覆盖
WBC输出被零阶保持覆盖
WBC输出与tau_jump合并错误
WBC输出写入data.ctrl时发生顺序/符号变化
WBC确实发力，但当前力矩能力不足或任务不可实现
```

## 二、Q1-B强制新增字段

每个WBC周期必须记录以下字段，不能只记录最终`data.ctrl`：

```text
WBC是否到期调用：wbc_due
WBC函数是否实际调用：wbc_called
调用时JumpPhase
调用时decision.mode
balance_enabled / motion_enabled
显式JumpReference state/acceleration
WBC每个任务的desired_acceleration
WBC每个任务的bias_acceleration
WBC每个任务的activation
WBC原始输出wbc_raw_torque
WBC result是否存在
result.feasible
result.fallback_used
result.clip_applied
result.dynamics_residual_norm
result.qacc
result.scaling
hierarchy.outcomes[].name
hierarchy.outcomes[].residual_norm
hierarchy.outcomes[].exhausted
predicted_normal
跳跃前馈tau_jump
合并后torque
最终data.ctrl
最终输出来源output_source
最终输出是否写回wbc_output_written
```

## 三、WBC作用链必须逐拍核对

在以下关键时刻分别核对完整链路：

```text
跳跃触发前最后一个GROUND周期
COMPRESS中段
pitch第一次异常前一拍
pitch第一次异常时刻
真实几何离地前一拍
真实几何离地后一拍
FLIGHT进入时刻
LANDING进入时刻
ABORT时刻
```

每个时刻必须能回答：

```text
WBC有没有调用？
WBC为什么调用/没有调用？
WBC是否允许balance？
WBC求解是否可行？
WBC任务是否被缩放或耗尽？
WBC算出的原始力矩是多少？
原始力矩是否被tau_jump、airborne_zero或zero_order_hold覆盖？
最终data.ctrl是否等于预期合并结果？
```

## 四、WBC异常分类

### A类：WBC没有真正起作用，属于控制逻辑/接线问题

满足任一条件即归入A类候选：

- `wbc_called=False`，但当前相位和监督模式理论上允许WBC；
- `balance_enabled=False`导致`solve_wbc`直接返回零力矩；
- `result=None`但最终输出被误认为是WBC结果；
- `wbc_raw_torque`非零，但`output_source=airborne_zero`或`zero_order_hold`覆盖；
- WBC输出与最终`data.ctrl`不满足明确的合并关系；
- `tau_jump`重复叠加或覆盖了WBC本应控制的执行器；
- WBC任务中缺少显式JumpReference，实际仍使用MPC sample或`sample=None`默认分支；
- 任务要求存在，但`selection`把对应执行器行清零导致任务没有执行权限；
- WBC认为接地而几何已离地，或WBC认为飞行而仍在接地，导致接触模型与实际状态不一致。

### B类：WBC确实起作用，但求解能力/参数不足

只有在A类全部排除后，才允许考虑B类：

- `wbc_called=True`；
- `balance_enabled=True`；
- `result`存在且求解流程有效；
- 输出确实写入最终`data.ctrl`；
- 显式参考和任务目标口径正确；
- 但任务残差持续较大；
- `scaling<1`、`exhausted=True`、`clip_applied=True`或力矩达到限幅；
- 动力学残差、预测法向力或接触约束显示任务不可实现。

此时才能把问题归为WBC任务能力、执行器余量或参数/权重问题。当前仍禁止调参，只能报告证据。

### C类：参考或建模问题

如果WBC调用和执行链路都正常，但：

- 显式参考本身与跳跃目标不一致；
- 任务目标方向/符号错误；
- 腿长/腿角雅可比映射使期望竖直力产生异常pitch力矩；
- `tau_jump`方向或左右腿映射异常；
- 接触模型与几何真实接触不一致；

则归入参考/建模问题，不得归为“WBC参数没调好”。

## 五、Q1-B新增判定顺序

测试agent必须按以下顺序判断，不能直接看最终效果：

```text
1. 显式JumpReference是否正确进入WBC
2. WBC在关键时刻是否真正调用
3. WBC是否允许balance和对应运动任务
4. result是否有效、是否可行
5. WBC原始力矩是否非零且方向合理
6. 原始力矩是否真正写入最终data.ctrl
7. 任务残差/缩放/限幅是否显示能力不足
8. 最后才比较pitch和跳跃效果
```

## 六、Q1-B报告必须增加的表格

| 时刻 | phase | mode | WBC调用 | balance | feasible | fallback/clip | 最大残差 | 最大缩放 | WBC原始腿矩 | tau_jump | 最终ctrl | 输出来源 |
|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 起跳前 | | | | | | | | | | | | |
| 压缩中段 | | | | | | | | | | | | |
| pitch首次异常 | | | | | | | | | | | | |
| 几何离地前 | | | | | | | | | | | | |
| 几何离地后 | | | | | | | | | | | | |
| FLIGHT | | | | | | | | | | | | |
| LANDING/ABORT | | | | | | | | | | | | |

## 七、当前执行要求修正

Q1-B仍然只改变MPC参考来源：MPC shadow记录保留，显式JumpReference送入WBC。除此之外，不允许同时修复接触状态机、airborne输出、腿长模型或任何参数。

如果Q1-B效果不好，测试agent必须先返回上述WBC作用链表，再判断属于A类逻辑问题、B类能力/参数问题或C类参考/建模问题；不得直接报告“WBC参数需要调整”。

当前批准状态：**Q1-B可以执行，但必须包含WBC实际输出审计；未完成审计前，禁止进入参数讨论或最终修复实现。**




---

# 测试agent：Q1-B显式JumpReference正式测试报告（2026-10-08）

## 一、执行结论

**测试结论：Q1-B结果有效，显式JumpReference已实际进入跳跃期间WBC；MPC sample_raw shadow保留；WBC实际作用链审计完成。按当前批准边界暂停，不执行Q2、Q3、P6-E3或任何参数修改。**

本次只改变诊断测试开关 `explicit_jump_reference=True` 对跳跃期间WBC实际sample的来源，未改变输入、初始状态、仿真时长、控制参数、接触估计器、状态机、airborne输出策略或MPC求解本身。正式采用 `run02`；`run01`控制结果一致，但在补充逐拍审计字段前生成，因此不作为最终审计版。

## 二、固定环境和命令

- 测试日期：2026-10-08
- 工作区：`C:\Users\20060\Desktop\circleg-robot`
- Git commit：`35e914786755c7f2b55cafae0382f0d390f4d724`
- Python解释器：`C:\Users\20060\Desktop\circleg-robot\.venv\Scripts\python.exe`
- Python版本：3.13.2
- NumPy：2.5.3
- MuJoCo：3.13.0
- 工况：`wls + held_leg + tuned-agile`，`jump_height=0.10 m`，`jump_stroke=0.12 m`，`jump_time=1.0 s`，`duration=3.0 s`
- 外力：无；`delta_velocity/qfrc_applied/viewer_xfrc_applied`均未使用
- 正式命令：

```text
.\.venv\Scripts\python.exe wheel_leg_mujoco\tools\jump_diagnostic.py --stage Q1B --case baseline --run-id run02 --duration 3.0 --jump-height 0.10 --jump-stroke 0.12 --jump-time 1.0 --explicit-jump-reference
```

系统默认MSYS2 Python曾因缺少NumPy导致导入失败；已按用户授权使用工作区`.venv`并通过镜像源执行安装/校验，确认`.venv`中的NumPy已满足依赖。后续所有结果均固定使用上述`.venv`解释器。

## 三、第0阶段工具自检

- 语法检查：通过，`control.py`、`loop.py`、`simulation.py`、`jump_diagnostic.py`、`video.py`均可编译。
- 导入检查：通过。
- CLI字段检查：`--explicit-jump-reference`可见并可执行。
- 记录字段检查：`wbc_due`、`wbc_called`、`explicit_jump_reference_active`、显式参考、`sample_raw`、实际`sample`、WBC原始力矩、`tau_jump`、WBC结果、最终`ctrl`、输出来源均存在。
- 冒烟：`Q1B_baseline_smoke03/04`均完成，输入完全相同。
- 重复性：两次冒烟的全部审计字段逐帧一致；均为1300帧、325个WBC到期周期、313个WBC实际调用周期、249个显式参考active帧，未发现stale显式参考记录。
- 显式参考维度：6状态；状态为`[起跳位置, 0, 0, 0, 起跳yaw, 0]`；加速度全零。
- 自检结果：通过，正式测试开始。

## 四、正式结果文件

- 原始逐物理步数据：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\Q1B_baseline_run02.json.gz`
- 原始运行日志：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\Q1B_baseline_run02.log`
- WBC作用链审计JSON：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\Q1B_wbc_audit_run02.json`
- WBC作用链审计日志：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\Q1B_wbc_audit_run02.log`
- 同工况录像：`C:\Users\20060\Desktop\circleg-robot\videos\Q1B_explicit_jump_20261008_50fps.mp4`

录像参数与正式工况一致，50 fps、640x480、150帧；录像仅用于查看现象，评价以逐拍数据为准。

## 五、正式摘要与P0对照

| 指标 | P0当前基线 | Q1-B显式参考 | 备注 |
|---|---:|---:|---|
| 最大绝对pitch | 0.529612 rad | 0.208047 rad | Q1-B较小 |
| 最大绝对pitch_rate | 7.25775 rad/s | 6.15256 rad/s | Q1-B较小 |
| 最大前向位移 | 0.726006 m | 0.116010 m | Q1-B较小 |
| 最大正轮隙 | 0.092220 m | 0.039282 m | 仍发生几何离地 |
| WBC实际调用次数 | 434 | 313 | Q1-B在后续airborne段不再调用 |
| airborne帧数 | 1267 | 1746 | Q1-B于1.254 s进入airborne |
| 仿真停止原因 | None | None | 3.0 s完成积分 |
| JumpPhase ABORT帧数 | 1765 | 1751 | Q1-B未完成稳定着地恢复 |

## 六、WBC作用链关键时刻

下表中的“最大残差”为该拍`hierarchy.outcomes[].residual_norm`最大值；力矩单位为N*m。

| 时刻 | phase/mode | wbc_due | WBC调用 | balance | feasible | fallback/clip | 最大残差 | 最大缩放 | WBC原始腿矩 | tau_jump | 最终ctrl摘要 | 输出来源 |
|---:|---|---:|---:|---:|---|---|---:|---:|---|---|---|---|
| 0.999 s，起跳前最后GROUND | IDLE/ground | 否 | 否 | 是 | - | - | - | - | - | - | `[-7.521,7.553,7.511,-7.545,0.000,0.000]` | zero_order_hold |
| 1.116 s，COMPRESS中段WBC周期 | COMPRESS/ground | 是 | 是 | 是 | True | 否/否 | 8.686 | 1.000 | `[7.485,-7.494]` | 全零 | `[-7.677,7.485,7.681,-7.494,0,0]` | wbc_update |
| 1.196 s，pitch首次异常前一拍 | COMPRESS/ground | 是 | 是 | 是 | True | 否/否 | 24.017 | 1.000 | `[-0.903,0.870]` | 全零 | `[0.333,-0.903,-0.307,0.870,0,0]` | wbc_update |
| 1.200 s，MPC pitch前馈反向 | COMPRESS/ground | 是 | 是 | 是 | True | 否/否 | 24.984 | 1.000 | `[-1.786,1.754]` | 全零 | `[1.199,-1.786,-1.174,1.754,0,0]` | wbc_update |
| 1.221 s，几何离地前一拍 | COMPRESS/ground | 否 | 否 | 是 | - | - | - | - | - | - | `[6.368,-7.245,-6.369,7.224,0,0]` | zero_order_hold |
| 1.222 s，几何轮隙首次超过1 mm | COMPRESS/ground | 否 | 否 | 是 | - | - | - | - | - | - | 保持上一拍 | zero_order_hold |
| 1.236 s，FLIGHT | FLIGHT/ground | 是 | 否 | 是 | - | - | - | - | - | - | `[15.018,-16.953,-15.048,16.911,0,0]` | zero_order_hold |
| 1.237 s，LANDING | LANDING/ground | 否 | 否 | 是 | - | - | - | - | - | - | 保持上一拍 | zero_order_hold |
| 1.252 s，ABORT前后WBC周期 | ABORT/ground | 是 | 是 | 是 | False | 否/否 | 447.026 | 0.000 | `[24.939,-24.506]` | 全零 | `[-40,24.939,40,-24.506,1.592,1.955]` | wbc_update |
| 1.254 s，监督层进入airborne | ABORT/airborne | 否 | 否 | 否 | - | - | - | - | - | - | `[0,0,0,0,0,0]` | airborne_zero |

补充逐拍事实：

1. 显式参考active期间，实际WBC sample为固定参考；例如起跳参考为`[0.000911,0,0,0,-0.0000547,0]`，加速度全零。
2. 1.196 s的MPC shadow pitch加速度为`+76.551`，1.200 s变为`-76.215`；同期实际送入WBC的显式sample pitch为`0`，说明MPC shadow与WBC实际参考已分离。
3. active且WBC实际调用的62个周期中，`sample_raw != sample`均成立；`feasible=True`为62次，`fallback_used=False`、`clip_applied=False`；`ctrl-(wbc_raw_torque+tau_jump)`最大绝对误差为0。
4. 1.248 s时WBC仍可行但最大任务缩放约0.7908、最大残差约655.257；1.252 s时求解结果`feasible=False`、任务缩放为0、残差约447.026。
5. 几何离地后，估计接触仍为`[True,True]`，直到1.254 s监督层才进入airborne；FLIGHT/LANDING期间WBC没有实际调用，输出保持上一拍，1.254 s后输出置零。

## 七、结果有效性和停止边界

- 评价工具：有效。
- Q1-B正式数据：有效。
- Q1-B效果：相对P0，释放前pitch和前向位移明显减小；但仍出现几何离地、接触估计滞后、ABORT以及后续airborne输出，不能据此宣布跳跃成功或完成着地恢复。
- 本报告只提交Q1-B证据，不把后续状态机/接触交接现象混入Q1-B修复，不执行Q2或其它隔离。
- `sample_override`虽是测试接口参数，但当前主运行路径仅由`explicit_jump_reference`开关控制；本次未发现绕过主路径的实际调用。

## 八、工作区现场（git status --short原样）

```text
 M Algorithm/supervision.py
 M wheel_leg_mujoco/MJCF/robot.xml
 M wheel_leg_mujoco/gbc_mpc_wbc/control.py
 M wheel_leg_mujoco/gbc_mpc_wbc/dynamics.py
 M wheel_leg_mujoco/gbc_mpc_wbc/loop.py
 M wheel_leg_mujoco/gbc_mpc_wbc/simulation.py
 M wheel_leg_mujoco/tools/video.py
 M wheel_leg_mujoco/tools/view.py
?? MUJOCO_LOG.TXT
?? URDF_Model/mjcf/URDF_Model_field.xml
?? URDF_Model/mjcf/field_meshes/
?? "docs/agent\344\272\244\344\272\222\346\226\207\346\241\243.md"
?? field_assets/
?? rmuc2026-mujoco/
?? videos/
?? wheel_leg_mujoco/gbc_mpc_wbc/jump.py
?? wheel_leg_mujoco/results/P0_baseline_default_run01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_run01.log
?? wheel_leg_mujoco/results/P0_baseline_default_smoke01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_smoke01.log
?? wheel_leg_mujoco/results/P0_baseline_default_smoke02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_smoke02.log
?? wheel_leg_mujoco/results/P0_baseline_run01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_run01.log
?? wheel_leg_mujoco/results/P0_baseline_run01_inspect.log
?? wheel_leg_mujoco/results/P0_baseline_run02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_run02.log
?? wheel_leg_mujoco/results/P0_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_smoke01.log
?? wheel_leg_mujoco/results/P0_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_smoke02.log
?? wheel_leg_mujoco/results/P6_baseline_default_run01.json.gz
?? wheel_leg_mujoco/results/P6_baseline_default_run01.log
?? wheel_leg_mujoco/results/P6_baseline_run01.json.gz
?? wheel_leg_mujoco/results/P6_baseline_run01.log
?? wheel_leg_mujoco/results/P6_delta_velocity_default_run02.json.gz
?? wheel_leg_mujoco/results/P6_delta_velocity_default_run02.log
?? wheel_leg_mujoco/results/P6_delta_velocity_run01.json.gz
?? wheel_leg_mujoco/results/P6_delta_velocity_run01.log
?? wheel_leg_mujoco/results/P6_key_events.log
?? wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.log
?? wheel_leg_mujoco/results/P6_qfrc_default_run02.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_default_run02.log
?? wheel_leg_mujoco/results/P6_qfrc_run01.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_run01.log
?? wheel_leg_mujoco/results/P6_transition_report.log
?? wheel_leg_mujoco/results/P6_transition_report_compact.log
?? wheel_leg_mujoco/results/Q1A_mpc_audit_default_run01.json
?? wheel_leg_mujoco/results/Q1B_baseline_run01.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_run01.log
?? wheel_leg_mujoco/results/Q1B_baseline_run02.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_run02.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke01.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke02.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke03.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke04.log
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run01.json
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run01.log
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run02.json
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run02.log
?? wheel_leg_mujoco/results/_backup_before_field/
?? wheel_leg_mujoco/results/debug_zero_input/
?? wheel_leg_mujoco/results/jump_test_airsign.json
?? wheel_leg_mujoco/results/jump_test_baseline.json
?? wheel_leg_mujoco/results/jump_test_cg.json
?? wheel_leg_mujoco/results/jump_test_cg2.json
?? wheel_leg_mujoco/results/jump_test_clean.json
?? wheel_leg_mujoco/results/jump_test_comp0.json
?? wheel_leg_mujoco/results/jump_test_comp1.json
?? wheel_leg_mujoco/results/jump_test_crouch.json
?? wheel_leg_mujoco/results/jump_test_crouch014.json
?? wheel_leg_mujoco/results/jump_test_crouch014b.json
?? wheel_leg_mujoco/results/jump_test_crouch020.json
?? wheel_leg_mujoco/results/jump_test_cutforce.json
?? wheel_leg_mujoco/results/jump_test_dbl.json
?? wheel_leg_mujoco/results/jump_test_dbl0.json
?? wheel_leg_mujoco/results/jump_test_dbl1.json
?? wheel_leg_mujoco/results/jump_test_dblfix.json
?? wheel_leg_mujoco/results/jump_test_decoup.json
?? wheel_leg_mujoco/results/jump_test_deployed.json
?? wheel_leg_mujoco/results/jump_test_diag.json
?? wheel_leg_mujoco/results/jump_test_diag2.json
?? wheel_leg_mujoco/results/jump_test_dualcut.json
?? wheel_leg_mujoco/results/jump_test_event1.json
?? wheel_leg_mujoco/results/jump_test_event2.json
?? wheel_leg_mujoco/results/jump_test_event3.json
?? wheel_leg_mujoco/results/jump_test_fix1.json
?? wheel_leg_mujoco/results/jump_test_fix3.json
?? wheel_leg_mujoco/results/jump_test_flight_angle.json
?? wheel_leg_mujoco/results/jump_test_flightdiag.json
?? wheel_leg_mujoco/results/jump_test_follow.json
?? wheel_leg_mujoco/results/jump_test_gcheck.json
?? wheel_leg_mujoco/results/jump_test_gcheck2.json
?? wheel_leg_mujoco/results/jump_test_isolate.json
?? wheel_leg_mujoco/results/jump_test_legchk.json
?? wheel_leg_mujoco/results/jump_test_legrate.json
?? wheel_leg_mujoco/results/jump_test_limit70.json
?? wheel_leg_mujoco/results/jump_test_link.json
?? wheel_leg_mujoco/results/jump_test_m0.json
?? wheel_leg_mujoco/results/jump_test_m1.json
?? wheel_leg_mujoco/results/jump_test_mergefix.json
?? wheel_leg_mujoco/results/jump_test_mergefix2.json
?? wheel_leg_mujoco/results/jump_test_mpcchk.json
?? wheel_leg_mujoco/results/jump_test_mpcdata.json
?? wheel_leg_mujoco/results/jump_test_mpcdata2.json
?? wheel_leg_mujoco/results/jump_test_normal1.json
?? wheel_leg_mujoco/results/jump_test_planA5.json
?? wheel_leg_mujoco/results/jump_test_planA6.json
?? wheel_leg_mujoco/results/jump_test_planAB.json
?? wheel_leg_mujoco/results/jump_test_planA_final.json
?? wheel_leg_mujoco/results/jump_test_planB.json
?? wheel_leg_mujoco/results/jump_test_planB2.json
?? wheel_leg_mujoco/results/jump_test_planB3.json
?? wheel_leg_mujoco/results/jump_test_planD2.json
?? wheel_leg_mujoco/results/jump_test_planDA.json
?? wheel_leg_mujoco/results/jump_test_planDA2.json
?? wheel_leg_mujoco/results/jump_test_plan_a.json
?? wheel_leg_mujoco/results/jump_test_plan_a2.json
?? wheel_leg_mujoco/results/jump_test_policy.json
?? wheel_leg_mujoco/results/jump_test_policy2.json
?? wheel_leg_mujoco/results/jump_test_policy3.json
?? wheel_leg_mujoco/results/jump_test_pre5.json
?? wheel_leg_mujoco/results/jump_test_prechk.json
?? wheel_leg_mujoco/results/jump_test_prehold.json
?? wheel_leg_mujoco/results/jump_test_preload.json
?? wheel_leg_mujoco/results/jump_test_probe_thrust.json
?? wheel_leg_mujoco/results/jump_test_pscan.json
?? wheel_leg_mujoco/results/jump_test_rate.json
?? wheel_leg_mujoco/results/jump_test_rate0.json
?? wheel_leg_mujoco/results/jump_test_rate1.json
?? wheel_leg_mujoco/results/jump_test_ratechk.json
?? wheel_leg_mujoco/results/jump_test_rchk.json
?? wheel_leg_mujoco/results/jump_test_refonly.json
?? wheel_leg_mujoco/results/jump_test_reverted.json
?? wheel_leg_mujoco/results/jump_test_ronly2.json
?? wheel_leg_mujoco/results/jump_test_scan.json
?? wheel_leg_mujoco/results/jump_test_schedchk.json
?? wheel_leg_mujoco/results/jump_test_signfix.json
?? wheel_leg_mujoco/results/jump_test_signflip.json
?? wheel_leg_mujoco/results/jump_test_signflip2.json
?? wheel_leg_mujoco/results/jump_test_taskshare.json
?? wheel_leg_mujoco/results/jump_test_vertical.json
?? wheel_leg_mujoco/results/jump_test_vtor.json
?? wheel_leg_mujoco/results/jump_test_wbc_angle.json
?? wheel_leg_mujoco/results/jump_test_wbcmodel3.json
?? wheel_leg_mujoco/results/m2_nullspace.json.gz
?? wheel_leg_mujoco/results/m2_qp.json.gz
?? wheel_leg_mujoco/results/m2_wls.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_backward.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_forward.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_lateral.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_stand30.json.gz
?? wheel_leg_mujoco/results/m3_wls_backward.json.gz
?? wheel_leg_mujoco/results/m3_wls_forward.json.gz
?? wheel_leg_mujoco/results/m3_wls_lateral.json.gz
?? wheel_leg_mujoco/results/m3_wls_stand30.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v-1.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v0.5.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v1.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v2.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_w1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v-1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v0.5.json.gz
?? wheel_leg_mujoco/results/m4_wls_v1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m4_wls_v2.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_w1.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_push_fwd.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_push_lat.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_stand30.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v1.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v2.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_w1.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_push_fwd.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_push_lat.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_stand30.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v1.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v2.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_w1.0.json.gz
?? wheel_leg_mujoco/results/view_test/
?? wheel_leg_mujoco/tools/jump_diagnostic.py
?? "wheel_leg_mujoco/\350\267\263\350\267\203\345\212\237\350\203\275\346\226\271\346\241\210.md"
```

当前结果和现场均保留，未执行回退、清理、覆盖或删除。

测试agent：Q1-B正式测试完成，结果有效；等待问题排查agent审阅，暂停后续测试。

# 问题排查agent：Q1-B后对压缩段部署逻辑的复核（2026-10-08）

## 一、当前结论先行

当前异常不能先归结为“压缩阶段被全局离地档误触发”。现有代码的真实执行链是：

```text
COMPRESS
  -> JumpCommand.leg_force = 0
  -> JumpCommand.leg_length_target = L0 - compression_stroke
  -> control.py 用 WBC 腿长任务的 PD 加速度追踪该目标
  -> 到达压缩终点后才切换 THRUST
  -> THRUST 的 tau_jump 只能在后续 WBC 节拍计算并写出
```

因此，用户提出的“压缩阶段应由雅可比力矩施加压缩力，而不是直接改腿长目标”与当前实现不一致；当前实现确实是“直接改变目标腿长，让 WBC 完成压缩”。

但现有证据还不能证明力的方向已经错误，必须将“压缩执行方式错误”和“离地/状态切换错误”分开验证。

## 二、可直接核对的代码证据

1. `wheel_leg_mujoco/gbc_mpc_wbc/jump.py:242-251`：
   - `COMPRESS` 明确输出 `leg_force=0.0`、`leg_length_target=compression_end`；
   - `THRUST` 才输出 `leg_force=config.leg_force()`、`leg_length_target=release_end`。
2. `wheel_leg_mujoco/gbc_mpc_wbc/control.py:102-115`：跳跃期只要存在 `leg_length_target`，就用 `kp_h*(target-current)-kd_h*rate`重写 WBC 的 `length_R/length_L` 任务；这就是压缩的实际驱动源。
3. `wheel_leg_mujoco/gbc_mpc_wbc/control.py:129-160`：`tau_jump`只在当前 `jump.leg_force`非零时产生实际蹬伸力矩，并在 WBC 求解后叠加到最终力矩；压缩阶段不会产生该前馈力矩。
4. `wheel_leg_mujoco/gbc_mpc_wbc/loop.py:120-126`：跳跃每个物理步都用轮最低点几何间隙和算法接触估计更新状态机。
5. `wheel_leg_mujoco/gbc_mpc_wbc/loop.py:153-182`：全局 `airborne_active` 只有在 `not jump_active_now` 时才生效；跳跃处于 `COMPRESS/THRUST/FLIGHT` 时，全局离地档不会直接把 `airborne_active` 置真。但 `FLIGHT` 的 `jump_disconnect` 会使 WBC 跳过，并由输出级保持上一拍力矩。
6. `wheel_leg_mujoco/gbc_mpc_wbc/dynamics.py:247-275`：当前力映射为 `F_L=F_z*cos(phi)`、`F_phi=-F_z*L*sin(phi)`、`tau=J.T@[F_L,F_phi]`，并按 `leg_actuators` 写入腿部执行器。该公式在几何功率对偶意义下是自洽的，但仍需用实际电机映射和正负方向试验确认“正力是否造成预期伸缩/质心加速度”。

## 三、现有Q1-B证据如何解释

Q1-B报告中的时序是：几何轮隙在约 `1.221 s` 已出现，但 JumpPhase 仍为 `COMPRESS`；约 `1.226 s` 才进入 `THRUST`，约 `1.227 s` 进入 `FLIGHT`。这说明至少在该正式结果中：

- 压缩阶段没有因为 `JumpPlanner` 的 THRUST 离地判据提前进入 `FLIGHT`；
- 但 WBC 腿长目标驱动已经造成了真实几何离地/接触状态错配；
- 进入 `THRUST` 后，状态机可能在下一物理步先看到已有轮隙，再等不到下一次有效 WBC 蹬伸输出，就进入 `FLIGHT`；
- 这会表现为“压缩后没有释放弹射，而是保持旧力矩/脱离控制”。

所以当前最可疑的不是简单的“MuJoCo位置API误触发”，而是以下时序竞争：

```text
WBC腿长目标造成主动腿部运动
  -> 轮隙在真正蹬伸命令落地前已经出现
  -> COMPRESS到THRUST的切换发生在WBC节拍之外
  -> 下一物理步先进入FLIGHT
  -> jump_disconnect跳过WBC，未实际执行有效THRUST
```

该推理仍需由逐物理步记录确认，不能把它当作最终根因。

## 四、必须优先核查的测试点

测试agent下一轮只做时序与符号核查，不改参数、不改状态机、不改接触阈值：

1. 在 `1.20-1.24 s` 逐物理步记录 `JumpPhase`、两轮各自几何间隙、`contact_geometry`、`contact_estimator`、腿长及腿长率，确认是否存在 `COMPRESS + wheel_gap>1 mm`。
2. 同一窗口记录 `wbc_due`、`wbc_called`、`jump.leg_force`、`jump.leg_length_target`、腿长任务期望加速度、`tau_jump`、最终 `data.ctrl` 和 `output_source`。
3. 重点确认从 `COMPRESS -> THRUST` 的那一拍到 `THRUST -> FLIGHT` 的那一拍之间，是否至少有一拍实际写入非零 `tau_jump`；若没有，则是释放命令部署时序问题，不是WBC能力不足。
4. 将轮隙判据拆成左右轮，不再只看 `np.min(wheel_contact_points[:,2])` 的汇总值，确认是否单轮先离地、另一轮仍接触；同时对比 MuJoCo 接触几何标志，排除“最低点汇总掩盖双轮状态”的问题。
5. 对 `thrust_torque()` 做不改变控制参数的符号审计：记录每条腿的 `phi/L/J`、`wrench`、`tau_jump`，并用虚功关系核对 `tau_jump.T @ qdot_act` 与 `F_z * z_dot_wheel` 的符号；再做正负力的单步方向验证，确认正 `leg_force` 是否产生预期的伸腿/质心向上趋势。
6. 单独核对执行器顺序：`leg_actuators`、`selection` 的行、MJCF actuator/joint 绑定，以及正力矩导致的实际关节运动方向。若左右腿或前后髋映射错位，优先归类为连接/符号问题，不讨论增益。

## 五、修复方向暂不落地

若上述证据确认“压缩由腿长目标造成主动失稳，且THRUST未实际落地”，候选架构应改为：

- COMPRESS由独立的压缩力/压缩力矩命令驱动，使用当前位形实时雅可比；不得同时保留会反向抵消该命令的 `length_R/length_L` 目标；
- THRUST由独立的正向竖直等效力驱动，必须有明确的“命令已写入执行器”交接状态，不能仅依赖状态机相位改变；
- FLIGHT只能在有效蹬伸命令已执行后，依据双轮几何离地事件进入；
- `tau_jump`的正负号和四个腿部执行器映射先通过虚功及单步方向证据确认，再决定代码修复；
- 不回退或删除MPC腿长拟合，不把Q1-B现象直接归因于MPC参数。

在测试agent返回上述逐拍结果前，问题排查agent不修改 `jump.py`、`control.py` 或接触状态机。

# 问题排查agent：状态机与离地检测链路澄清及方案优化（2026-10-08）

## 一、Q1-B状态机实际切换

`Q1B_baseline_run02.log`给出的实际切换顺序为：

```text
1.000 s  起跳，IDLE -> COMPRESS
1.235 s  COMPRESS -> THRUST
         原因：腿长达到 0.1600 m，腿长率 -2.208 m/s
1.236 s  THRUST -> FLIGHT
         原因：轮地间隙 7.8 mm，被 JumpPlanner 几何判据确认离地
1.237 s  FLIGHT -> LANDING
         原因：contact_estimator 仍报告至少一轮接触
1.249 s  跳跃 ABORT
         原因：腿长 0.1111 m 越出机械范围 [0.114, 0.430]
1.254 s  ControlSupervisor ground -> airborne
         两轮法向力估计连续低于离地条件，loop 记录“离地检测生效”
```

因此视频中只看到压缩、没有看到有效释放是有数据支持的：`THRUST`只存在1个物理帧，随后立即进入`FLIGHT`，没有形成可观察的释放阶段。

必须区分两个“离地”事件：

1. `1.236 s`是跳跃状态机的几何离地事件，输入是两轮最低点的轮地间隙，不是质心偏移。
2. `1.254 s`才是普通监督层的力接触离地事件，输入是接触几何筛选后由动力学方程反解的法向力，再经过滞回和持续时间判定。

所以当前证据不支持“普通离地检查在压缩阶段先触发，直接导致第一次失控”。普通离地档在`jump_active`期间被loop显式屏蔽；真正先切断WBC的是`JumpPlanner`在`1.236 s`给出的`FLIGHT.disconnect=True`。但是，普通离地档在跳跃ABORT后于`1.254 s`触发，随后输出变成`airborne_zero`，属于后置放大因素。

## 二、当前离地检测并非质心位置判断

- 跳跃状态机使用 `wheel_contact_points(... )[:, 2]` 的轮最低点世界坐标，计算轮隙；没有使用质心偏移。
- 普通离地档使用 `FeedbackSampler` 的法向力反解结果；没有使用 `state.position` 或质心位置直接判断。
- 当前普通接触估计器本身已经有双阈值滞回：`off_force=5 N`、`on_force=15 N`、离地持续 `8 ms`、接地持续 `12 ms`。
- 仿真配置又将非跳跃阶段的离地持续时间设为 `50 ms`，用于过滤短暂卸载。
- 但 `sampling.py:115-116` 每拍固定传入 `JumpPhaseHint.NONE`，因此`THRUST/FLIGHT/LANDING`专用门控没有接入实际采样链路。这是已确认的接线缺陷，不能把现有接触检测器描述为完整的跳跃双证据检测。

## 三、对“压缩和释放都用外部力”的修正理解

用户的方向是合理的，但“外部力”必须明确为**由腿部雅可比换算出的执行器力矩命令**，不能直接给WBC一个腿长目标后再声称WBC在建模跳跃力。

当前实现的问题是：

```text
COMPRESS: leg_force = 0，leg_length_target = L0 - stroke
          WBC腿长任务主动把腿压短

THRUST:   leg_force > 0，tau_jump = J.T @ wrench
          但仍同时保留 leg_length_target = release_ceiling
```

这会产生两个风险：

1. 压缩阶段跳跃力没有进入WBC动力学外力通道，WBC只是在追踪几何目标；
2. 蹬伸阶段同时存在`tau_jump`和腿长目标，WBC任务与跳跃力之间存在潜在的重复控制或抵消关系。

WBC的正确角色应是：保留姿态、腿摆角和接触约束的残差控制；跳跃模块拥有压缩/蹬伸的腿部力矩，并把这份已知力矩通过`external_force=selection.T @ tau_jump`告知WBC，避免WBC重复抵消。腿长测量只作为相位退出和安全边界，不再作为压缩/释放的主要控制输入。

## 四、建议的V4控制与检测架构

### 4.1 压缩段

```text
COMPRESS_FORCE
  -> 跳跃模块给定压缩方向的腿部等效力 F_compress
  -> 每拍按当前 L/phi/J 计算 tau_compress
  -> WBC接收该已知力矩的external项，只求姿态/腿角/接触残差
  -> 腿长达到压缩终点时准备切换，不用腿长目标驱动压缩
```

这里的压缩力方向必须由单步方向测试确认，不能直接复用正向蹬伸力的符号。压缩力应使腿长减小、轮子保持接地，不应使轮隙增大。

### 4.2 蹬伸段

```text
THRUST_FORCE
  -> 切换时立即生成正向 F_thrust 和 tau_thrust
  -> 先完成一次实际执行器写入
  -> 记录 thrust_applied=True
  -> 只有之后才允许评估离地确认
```

当前代码的主要时序缺陷是相位切换发生在每物理步，而WBC只有`4 ms`节拍；`COMPRESS -> THRUST`后可能还没有一次实际非零`tau_jump`写入，下一物理步就因已有轮隙进入`FLIGHT`。修复时应增加“蹬伸命令已执行”的交接语义，而不是简单把离地阈值改大。

### 4.3 离地确认

离地确认建议使用双证据，但两个证据的职责要分开：

```text
几何证据：两轮都没有有效轮地几何接触，或两轮最低点均形成正间隙
力证据：两轮法向力均低于 off_force，并持续 off_duration
附加条件：thrust_applied=True，且仍处于THRUST/LIFTOFF_PENDING
```

只有在“蹬伸已实际执行 + 几何离地 + 法向力滞回确认”同时成立时，才进入`FLIGHT`并断开接地WBC。若只有法向力暂时降为零但几何仍接地，继续保持接地控制；若几何真实离地但力反解暂时滞后，进入`LIFTOFF_PENDING`而不是立即置零。

### 4.4 着地与恢复

`FLIGHT -> LANDING`不能继续使用当前的`any(contact)`单拍判定，因为Q1-B已经证明估计器滞后会在真实飞行的下一毫秒误触发。建议要求：几何接触恢复，并且法向力通过`on_force`持续确认；单轮接触先进入`LANDING_PENDING`，双轮稳定后再恢复完整WBC和MPC。

## 五、下一轮验证顺序

在允许修改架构前，测试agent按以下顺序取证：

1. 对`1.235-1.237 s`逐物理步核对`THRUST`期间是否实际写入非零`tau_jump`；这是当前最关键的证据。
2. 对正、负`F_z`分别做单步符号验证，记录腿长率、轮隙变化、质心竖直加速度和四个执行器关节运动方向。
3. 核对`leg_actuators`、`selection`、MJCF actuator gear/gain与`J`列顺序，排除电机接线/符号错误。
4. 检查跳跃状态提示是否传入`GroundContactEstimator.update()`；当前固定`JumpPhaseHint.NONE`必须作为接线问题单独报告。
5. 在不改变控制参数的前提下，离线重放同一记录，分别标出几何离地、法向力离地、`airborne_active`、`JumpPhase`和WBC写入事件。

本轮只允许确认部署链、力方向和状态机交接，不允许先调大离地持续时间、放宽腿长范围或修改WBC增益来掩盖问题。

# 问题排查agent：用质心、轮毂、轮最低点和腿长联合验证压缩离地猜想（2026-10-08）

## 一、结论

可以通过对比测试时的以下四类实际量，直接判断压缩阶段是否真的发生了离地：

1. 机体质心的世界坐标和竖直速度；
2. 左右轮毂中心的世界坐标和竖直速度；
3. 左右轮最低点的世界坐标/轮隙；
4. 左右腿解析腿长、腿长率以及当前 `JumpPhase`、几何接触、法向力接触估计和 `airborne_active`。

现有Q1-B只能确认“轮最低点间隙已变成正值”，还不能确认这是不是机体整体真实上升、轮毂脱离地面，还是轮最低点计算/轮姿态变化造成的几何假象。因此目前不能仅凭Q1-B断言压缩阶段已经真实起跳。

## 二、必须区分的三个位置

- **质心位置**：建议记录 `robot_com(model, data)` 的 `com_x/com_z`，不要用机体基座原点代替质心。
- **轮毂位置**：记录每个轮几何体的 `data.geom_xpos[wheel_geom_id]`，这是轮毂/轮几何中心位置。
- **轮最低点位置**：记录现有 `wheel_contact_points(model, data)[:, 2]`。它不是轮毂位置，会随轮姿态和圆柱轴方向变化。

当前JumpPlanner使用的是轮最低点世界坐标的最小值；普通离地档使用的是接触几何筛选后反解的法向力。两者都不是质心位置判断。

## 三、测试窗口与字段

测试agent只需对现有同工况重放增加记录，不改变控制参数或状态机：

### 重点窗口

```text
1.200-1.240 s：压缩末端、COMPRESS -> THRUST -> FLIGHT
1.240-1.260 s：FLIGHT -> LANDING -> ABORT以及普通airborne触发
```

### 每个物理步记录

```text
t
JumpPhase
com_x, com_z, com_vx, com_vz
right_hub_x, right_hub_z, left_hub_x, left_hub_z
right_lowest_z, left_lowest_z
right_gap, left_gap
L_R, L_L, Ldot_R, Ldot_L
pitch, pitch_rate
contact_geometry_R, contact_geometry_L
normal_force_R, normal_force_L
contact_estimator_R, contact_estimator_L
airborne_active
wbc_due, wbc_called
jump.leg_force, jump.leg_length_target, tau_jump
data.ctrl
output_source
```

不能只记录`min(wheel_contact_points[:,2])`，因为最小值会掩盖左右轮不同步，也不能区分轮毂运动和轮缘最低点运动。

## 四、判定矩阵

### 情况A：压缩阶段真实轮毂离地

若在`COMPRESS`期间同时出现：

```text
两侧轮毂 z 均上升并持续离开地面
两侧轮最低点 gap 均为正
法向力下降并满足现有 off_force/off_duration
质心 z 开始上升或出现明显向上速度
```

则说明不是单纯离地检测误报，而是腿长目标控制已经让轮子真实离地。根因应优先归为“压缩控制方式和释放交接错误”，而不是先调离地阈值。

### 情况B：轮毂保持接地，只有力估计短暂下降

若：

```text
轮毂 z 和轮最低点仍保持地面几何关系
contact_geometry 仍为 True
质心没有进入向上运动
但 normal_force 短暂低于 off_force
```

则是接触力反解或力滞回链路的短时卸载问题。此时双阈值滞回和持续时间可以过滤，但应先确认现有`GroundContactEstimator`输入可信，不能直接扩大阈值。

### 情况C：轮毂接地，但轮最低点出现正间隙

若轮毂高度未变化、几何接触仍在，但`wheel_contact_points`的最低点计算出现正间隙，则应排查轮圆柱轴、几何旋转矩阵和最低点算法，而不是改状态机。这属于轮隙观测量错误。

### 情况D：腿长缩短但轮毂接地、质心下降

若：

```text
L_R/L_L 下降
轮毂和轮最低点仍接地
质心 z 下降或基本不变
```

则压缩确实发生，但没有提前离地；随后若状态机仍在`THRUST`后立即`FLIGHT`，问题就是相位交接时序和蹬伸命令没有落地，而不是压缩阶段离地误判。

### 情况E：轮毂接地但质心发生前向/俯仰运动

若轮毂保持接地、腿长变短，但`com_x`明显变化、`pitch`和`pitch_rate`先发散，则压缩腿长目标通过腿部几何和接触约束产生了非期望水平力/俯仰力矩。这与视频中压缩段前向位移和释放前pitch发散的现象一致，应优先停用腿长目标驱动，改为显式力矩链路验证。

## 五、对离地检测方案的优化边界

“法向力解算 + 双阈值滞回”方向可以保留，但不能单独替代几何和状态机：

1. **普通行驶/扰动恢复**：使用法向力反解、`off_force/on_force`双阈值和持续时间，过滤短暂卸载。
2. **跳跃压缩和蹬伸**：使用几何接触作为观测，法向力作为确认；几何短暂离开但力未确认时进入`LIFTOFF_PENDING`，不立即置零。
3. **进入FLIGHT**：必须同时满足“蹬伸命令已实际写入”“双轮几何离地”“法向力达到离地滞回条件”。
4. **进入LANDING**：必须用几何重新接触和法向力通过`on_force`持续确认，不能使用当前`any(contact)`单拍切换。
5. **恢复WBC/MPC**：至少双轮稳定接触后再恢复，避免Q1-B中真实飞行1 ms后因陈旧估计器立即进入`LANDING`。

这组质心、轮毂、轮最低点和腿长对齐数据是下一轮的最小充分证据。拿到它之前，不修改离地阈值，也不把“轮最低点为正”直接等同于“质心已经起跳”。

# 问题排查agent：增加基于前态的状态流保护（2026-10-09）

## 一、当前代码的准确问题

当前`JumpPlanner.update()`已经使用`self.phase`分支，因此不是“单次调用中完全不看前一状态”。现有合法路径表面上是：

```text
IDLE -> COMPRESS -> THRUST -> FLIGHT -> LANDING -> RECOVER -> IDLE
```

但每个转移只检查当前观测条件，缺少以下保护：

- 没有要求前一状态已经完成必要的执行动作；
- 没有要求当前状态的进入条件持续确认；
- 没有区分候选事件和确认事件；
- 没有记录“蹬伸命令已经真正写入执行器”；
- `FLIGHT -> LANDING`只用`any(obs.contact)`单拍判定；
- `LANDING -> RECOVER`只用腿长邻域，不要求接触已经稳定；
- `RECOVER -> IDLE`不要求双轮接触和控制器交接条件同时成立。

因此Q1-B虽然没有代码层面的直接跳跃，但出现了：

```text
COMPRESS 约235 ms
THRUST   约1 ms
FLIGHT   约1 ms
LANDING  约12 ms
随后ABORT
```

这属于状态流保护不足和事件确认不足，不是简单的`if`分支漏写。

## 二、建议的前态保护状态流

不建议仅在原有判断前增加一个普通布尔条件；应明确合法前态和事件确认：

```text
IDLE
  -> COMPRESS_FORCE
     仅允许由新的起跳事件进入

COMPRESS_FORCE
  -> THRUST_ARMED
     仅允许由压缩终点达到且仍确认双轮支撑进入

THRUST_ARMED
  -> THRUST_FORCE
     仅允许由蹬伸命令完成装配并实际写入执行器进入

THRUST_FORCE
  -> LIFTOFF_PENDING
     仅允许由双轮几何离地候选进入

LIFTOFF_PENDING
  -> FLIGHT
     仅允许由双轮几何离地、法向力通过off滞回、且thrust_applied成立进入

FLIGHT
  -> LANDING_PENDING
     仅允许由几何接触候选进入

LANDING_PENDING
  -> LANDING/RECOVER
     仅允许由法向力通过on滞回并确认双轮接触进入

RECOVER
  -> IDLE
     仅允许由姿态、腿长、速度和双轮接触均恢复进入
```

`ABORT`仍可作为任意活动状态的异常出口，但进入`ABORT`后不得因为普通接触恢复自动回到`LANDING`或`IDLE`，必须由新的跳跃周期或明确复位动作重新初始化。

## 三、最小实现保险

若暂时不引入完整的`PENDING`状态，至少应增加以下前态锁存量：

```text
compress_confirmed
thrust_command_built
thrust_applied
liftoff_candidate
liftoff_confirmed
touchdown_candidate
touchdown_confirmed
```

每次转移必须同时满足：

```text
previous_phase == 允许的前态
and 当前观测条件满足
and 必要的前置锁存量已成立
```

状态机不得仅因为腿长、轮隙或接触布尔值单独变化就完成整条状态流。

## 四、对Q1-B的直接解释

Q1-B中`COMPRESS -> THRUST`发生在`1.235 s`，但没有证据表明该拍已经实际写入非零`tau_jump`；下一物理步`1.236 s`就因为轮隙进入`FLIGHT`。因此新的前态保护至少必须阻止：

```text
THRUST未确认执行
  -> 直接进入FLIGHT
```

同样，`FLIGHT -> LANDING`必须阻止：

```text
FLIGHT刚进入1 ms
  -> 使用滞后的contact_estimator=True
  -> 立即进入LANDING
```

这两个保护是为恢复正确状态流，不是为了延长状态停留时间，也不是调大阈值。

## 五、测试agent必须验证的状态流断言

下一轮不改控制参数，只对状态事件做断言和记录：

1. 任何`COMPRESS -> THRUST`必须有明确的前态为`COMPRESS`。
2. 任何`THRUST -> FLIGHT`必须证明`thrust_applied=True`。
3. `THRUST`若没有完成一次实际非零蹬伸输出，不得进入`FLIGHT`。
4. `FLIGHT -> LANDING`必须同时有几何接触候选和法向力接地确认，不能只用`any(contact)`。
5. `LANDING -> RECOVER`必须确认双轮接触，而不是只确认腿长回到邻域。
6. `RECOVER -> IDLE`必须确认状态机仍处于`RECOVER`、姿态和腿长有效，且控制权交接没有被`airborne_zero`覆盖。
7. 所有非法转移都要记录`previous_phase`、`requested_phase`、阻断原因，不得静默改写状态。

当前建议先验证这些断言和事件序列，再决定是否实现状态机重构；不直接修改状态机或阈值。


---

# 测试agent：Q1-B联合几何与状态流观测重放报告（2026-10-09）

## 一、执行结论

**测试结论：联合观测工具自检通过，正式同工况重放结果有效。已确认状态流中存在两个未满足的前置保护：`THRUST -> FLIGHT`发生时没有实际非零蹬伸输出，`FLIGHT -> LANDING`发生时没有几何接触和正法向力确认。当前只完成证据采集，不修改状态机、阈值或压缩控制。**

本轮使用与Q1-B相同的输入和显式JumpReference，仅增加测试侧观测与断言字段：质心、轮毂、轮最低点、有限差分质心速度、腿长、接触、蹬伸实际写入和状态转移原因。控制决策未使用这些新增字段。

## 二、工具自检

- 语法检查和导入检查通过。
- 新增字段维度检查通过：`com_xz=(2,)`、`com_vxz=(2,)`、`wheel_hub_xz=(2,2)`。
- 初始读取`subtree_linvel`并调用`mj_comVel`后仍全零，与运动状态不一致；该中间结果作废。
- 已修正为相邻物理步`com_xz`有限差分，修正后最大质心速度量级为`|vx|=0.5775 m/s`、`|vz|=1.0733 m/s`，符合当前运动量级。
- `Q1B_GEOM_baseline_smoke05/06`完全相同输入重复运行，所有联合观测、状态转移、输出字段逐帧一致。
- 冒烟结果：5次状态转移，2次断言失败；`thrust_command_built=1`，`thrust_applied=0`。
- 自检结论：通过，正式结果有效。

## 三、正式结果文件

- 原始数据：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\Q1B_GEOM_baseline_run02.json.gz`
- 原始日志：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\Q1B_GEOM_baseline_run02.log`
- 联合观测审计JSON：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\Q1B_GEOM_audit_run02.json`
- 联合观测审计日志：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\Q1B_GEOM_audit_run02.log`

固定工况仍为：`wls + held_leg + tuned-agile`，跳跃目标`0.10 m`，压缩行程`0.12 m`，起跳时刻`1.0 s`，总时长`3.0 s`，无外力，显式JumpReference开启。

## 四、联合观测关键数据

| 时刻 | phase | COM z / vz | 右左轮毂 z | 右左gap | 右左腿长 | 几何接触 | 法向力 | 估计接触 |
|---:|---|---:|---|---|---|---|---|---|
| 1.200 s | COMPRESS | 0.207523 / -0.517519 | 0.076744 / 0.076747 | -0.000254 / -0.000256 | 0.201619 / 0.201841 | True/True | 11.995 / 12.197 | True/True |
| 1.222 s | COMPRESS | 0.193722 / -0.730030 | 0.078038 / 0.078027 | 0.001041 / 0.001023 | 0.181519 / 0.181858 | False/False | 0 / 0 | True/True |
| 1.235 s | THRUST | 0.183341 / -0.857206 | 0.083939 / 0.083907 | 0.006943 / 0.006902 | 0.159766 / 0.160203 | False/False | 0 / 0 | True/True |
| 1.236 s | FLIGHT | 0.182474 / -0.866971 | 0.084814 / 0.084779 | 0.007818 / 0.007775 | 0.157440 / 0.157885 | False/False | 0 / 0 | True/True |
| 1.237 s | LANDING | 0.181597 / -0.876735 | 0.085767 / 0.085730 | 0.008771 / 0.008726 | 0.154997 / 0.155451 | False/False | 0 / 0 | True/True |
| 1.252 s | ABORT | 0.167270 / -1.024679 | 0.113178 / 0.113204 | 0.036183 / 0.036200 | 0.099047 / 0.099491 | False/False | 0 / 0 | True/True |
| 1.254 s | ABORT/airborne | 0.165193 / -1.043372 | 0.115266 / 0.115334 | 0.038271 / 0.038329 | 0.093293 / 0.093701 | False/False | 0 / 0 | False/False |

联合窗口`1.200–1.260 s`的范围：

- COM z：`0.159085–0.206997 m`，始终下降；
- COM vz：`-1.07325–-0.526476 m/s`，窗口内没有向上速度；
- 左右轮毂 z：分别约`0.076753–0.116156 m`、`0.076756–0.116287 m`，均持续上升；
- 左右最低点gap最大约`0.039160 m`、`0.039282 m`；
- 左右腿长从约`0.201 m`继续缩短至约`0.083 m`。

这些数据表明：不能把该窗口直接归为“质心已向上起跳”；同时也不能归为“轮毂保持接地且只有法向力短时下降”，因为轮毂上升、几何接触变为False且最低点gap为正。最终分类留给问题排查agent，不在测试报告中猜测根因。

## 五、状态流断言结果

| 转移 | 时刻 | 断言 | 证据 |
|---|---:|---|---|
| IDLE -> COMPRESS | 1.000 s | 通过 | 前态合法，双轮几何接触、估计接触和正法向力均成立 |
| COMPRESS -> THRUST | 1.235 s | 通过前态检查 | 压缩终点到达，但此时`wbc_due=False`、`wbc_called=False`；只构建了蹬伸命令 |
| THRUST -> FLIGHT | 1.236 s | 失败 | `thrust_command_built=True`，但`thrust_applied=False`；阻断原因：`thrust_not_applied` |
| FLIGHT -> LANDING | 1.237 s | 失败 | `contact_estimator=True`为陈旧估计，但几何接触False、法向力为0；阻断原因：`geometry_contact_not_confirmed`、`normal_force_not_positive` |
| LANDING -> ABORT | 1.249 s | 记录为异常出口 | ABORT允许作为异常出口，之后没有自动恢复到LANDING/IDLE |

正式摘要：

```text
phase_transition_count = 5
phase_transition_assertion_failures = 2
thrust_command_built_frames = 1
thrust_applied_frames = 0
first airborne = 1.254 s
```

## 六、对压缩段判定矩阵的证据边界

- 情况A未满足：COM z没有开始上升，COM vz在窗口内为负。
- 情况B未满足：法向力为0的同时，轮毂上升、几何接触为False、最低点gap为正。
- 情况C未直接满足：轮毂z并非保持不变，且几何接触同时为False。
- 情况D不完全满足：腿长缩短且COM下降，但轮毂没有保持固定接地关系。
- 情况E不完全满足：有明显前向/俯仰变化，但几何接触已丢失，不能直接套用“轮毂保持接地”的前提。

当前可确认的是：压缩末端在蹬伸命令真正写入前，轮毂和最低点已经出现离地迹象；随后状态机在没有实际蹬伸输出的情况下进入FLIGHT，并依据陈旧接触估计立即进入LANDING。是否将其作为“压缩控制与释放交接错误”的修复依据，由问题排查agent审阅后决定。

## 七、后续修改边界

本轮没有实现状态机保险，也没有重建压缩控制逻辑。原因是交互文档最后明确要求先验证状态事件序列，再决定是否实现状态机重构，并禁止直接修改状态机或阈值。

用户提出的“状态机保险”和“压缩控制逻辑重建1”将作为后续分离任务处理：

1. 先由问题排查agent书面明确两项修改各自的唯一变化、不得触碰的参数和验收断言；
2. 状态机保险与压缩控制重建不能在同一结果文件中混合验证；
3. 每项修改均须先保留当前`Q1B_GEOM_baseline_run02`作为基线，再做语法/导入/冒烟/重复性检查和正式重放；
4. 当前不修改离地阈值、不改法向力确认时间、不调WBC/MPC参数。

## 八、工作区现场（git status --short原样）

```text
 M Algorithm/supervision.py
 M wheel_leg_mujoco/MJCF/robot.xml
 M wheel_leg_mujoco/gbc_mpc_wbc/control.py
 M wheel_leg_mujoco/gbc_mpc_wbc/dynamics.py
 M wheel_leg_mujoco/gbc_mpc_wbc/loop.py
 M wheel_leg_mujoco/gbc_mpc_wbc/simulation.py
 M wheel_leg_mujoco/tools/video.py
 M wheel_leg_mujoco/tools/view.py
?? MUJOCO_LOG.TXT
?? URDF_Model/mjcf/URDF_Model_field.xml
?? URDF_Model/mjcf/field_meshes/
?? "docs/agent\344\272\244\344\272\222\346\226\207\346\241\243.md"
?? field_assets/
?? rmuc2026-mujoco/
?? videos/
?? wheel_leg_mujoco/gbc_mpc_wbc/jump.py
?? wheel_leg_mujoco/results/P0_baseline_default_run01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_run01.log
?? wheel_leg_mujoco/results/P0_baseline_default_smoke01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_smoke01.log
?? wheel_leg_mujoco/results/P0_baseline_default_smoke02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_smoke02.log
?? wheel_leg_mujoco/results/P0_baseline_run01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_run01.log
?? wheel_leg_mujoco/results/P0_baseline_run01_inspect.log
?? wheel_leg_mujoco/results/P0_baseline_run02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_run02.log
?? wheel_leg_mujoco/results/P0_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_smoke01.log
?? wheel_leg_mujoco/results/P0_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_smoke02.log
?? wheel_leg_mujoco/results/P6_baseline_default_run01.json.gz
?? wheel_leg_mujoco/results/P6_baseline_default_run01.log
?? wheel_leg_mujoco/results/P6_baseline_run01.json.gz
?? wheel_leg_mujoco/results/P6_baseline_run01.log
?? wheel_leg_mujoco/results/P6_delta_velocity_default_run02.json.gz
?? wheel_leg_mujoco/results/P6_delta_velocity_default_run02.log
?? wheel_leg_mujoco/results/P6_delta_velocity_run01.json.gz
?? wheel_leg_mujoco/results/P6_delta_velocity_run01.log
?? wheel_leg_mujoco/results/P6_key_events.log
?? wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.log
?? wheel_leg_mujoco/results/P6_qfrc_default_run02.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_default_run02.log
?? wheel_leg_mujoco/results/P6_qfrc_run01.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_run01.log
?? wheel_leg_mujoco/results/P6_transition_report.log
?? wheel_leg_mujoco/results/P6_transition_report_compact.log
?? wheel_leg_mujoco/results/Q1A_mpc_audit_default_run01.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run01.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run02.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run01.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run02.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke03.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke04.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke05.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke05.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke06.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke06.log
?? wheel_leg_mujoco/results/Q1B_baseline_run01.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_run01.log
?? wheel_leg_mujoco/results/Q1B_baseline_run02.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_run02.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke01.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke02.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke03.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke04.log
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run01.json
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run01.log
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run02.json
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run02.log
?? wheel_leg_mujoco/results/_backup_before_field/
?? wheel_leg_mujoco/results/debug_zero_input/
?? wheel_leg_mujoco/results/jump_test_airsign.json
?? wheel_leg_mujoco/results/jump_test_baseline.json
?? wheel_leg_mujoco/results/jump_test_cg.json
?? wheel_leg_mujoco/results/jump_test_cg2.json
?? wheel_leg_mujoco/results/jump_test_clean.json
?? wheel_leg_mujoco/results/jump_test_comp0.json
?? wheel_leg_mujoco/results/jump_test_comp1.json
?? wheel_leg_mujoco/results/jump_test_crouch.json
?? wheel_leg_mujoco/results/jump_test_crouch014.json
?? wheel_leg_mujoco/results/jump_test_crouch014b.json
?? wheel_leg_mujoco/results/jump_test_crouch020.json
?? wheel_leg_mujoco/results/jump_test_cutforce.json
?? wheel_leg_mujoco/results/jump_test_dbl.json
?? wheel_leg_mujoco/results/jump_test_dbl0.json
?? wheel_leg_mujoco/results/jump_test_dbl1.json
?? wheel_leg_mujoco/results/jump_test_dblfix.json
?? wheel_leg_mujoco/results/jump_test_decoup.json
?? wheel_leg_mujoco/results/jump_test_deployed.json
?? wheel_leg_mujoco/results/jump_test_diag.json
?? wheel_leg_mujoco/results/jump_test_diag2.json
?? wheel_leg_mujoco/results/jump_test_dualcut.json
?? wheel_leg_mujoco/results/jump_test_event1.json
?? wheel_leg_mujoco/results/jump_test_event2.json
?? wheel_leg_mujoco/results/jump_test_event3.json
?? wheel_leg_mujoco/results/jump_test_fix1.json
?? wheel_leg_mujoco/results/jump_test_fix3.json
?? wheel_leg_mujoco/results/jump_test_flight_angle.json
?? wheel_leg_mujoco/results/jump_test_flightdiag.json
?? wheel_leg_mujoco/results/jump_test_follow.json
?? wheel_leg_mujoco/results/jump_test_gcheck.json
?? wheel_leg_mujoco/results/jump_test_gcheck2.json
?? wheel_leg_mujoco/results/jump_test_isolate.json
?? wheel_leg_mujoco/results/jump_test_legchk.json
?? wheel_leg_mujoco/results/jump_test_legrate.json
?? wheel_leg_mujoco/results/jump_test_limit70.json
?? wheel_leg_mujoco/results/jump_test_link.json
?? wheel_leg_mujoco/results/jump_test_m0.json
?? wheel_leg_mujoco/results/jump_test_m1.json
?? wheel_leg_mujoco/results/jump_test_mergefix.json
?? wheel_leg_mujoco/results/jump_test_mergefix2.json
?? wheel_leg_mujoco/results/jump_test_mpcchk.json
?? wheel_leg_mujoco/results/jump_test_mpcdata.json
?? wheel_leg_mujoco/results/jump_test_mpcdata2.json
?? wheel_leg_mujoco/results/jump_test_normal1.json
?? wheel_leg_mujoco/results/jump_test_planA5.json
?? wheel_leg_mujoco/results/jump_test_planA6.json
?? wheel_leg_mujoco/results/jump_test_planAB.json
?? wheel_leg_mujoco/results/jump_test_planA_final.json
?? wheel_leg_mujoco/results/jump_test_planB.json
?? wheel_leg_mujoco/results/jump_test_planB2.json
?? wheel_leg_mujoco/results/jump_test_planB3.json
?? wheel_leg_mujoco/results/jump_test_planD2.json
?? wheel_leg_mujoco/results/jump_test_planDA.json
?? wheel_leg_mujoco/results/jump_test_planDA2.json
?? wheel_leg_mujoco/results/jump_test_plan_a.json
?? wheel_leg_mujoco/results/jump_test_plan_a2.json
?? wheel_leg_mujoco/results/jump_test_policy.json
?? wheel_leg_mujoco/results/jump_test_policy2.json
?? wheel_leg_mujoco/results/jump_test_policy3.json
?? wheel_leg_mujoco/results/jump_test_pre5.json
?? wheel_leg_mujoco/results/jump_test_prechk.json
?? wheel_leg_mujoco/results/jump_test_prehold.json
?? wheel_leg_mujoco/results/jump_test_preload.json
?? wheel_leg_mujoco/results/jump_test_probe_thrust.json
?? wheel_leg_mujoco/results/jump_test_pscan.json
?? wheel_leg_mujoco/results/jump_test_rate.json
?? wheel_leg_mujoco/results/jump_test_rate0.json
?? wheel_leg_mujoco/results/jump_test_rate1.json
?? wheel_leg_mujoco/results/jump_test_ratechk.json
?? wheel_leg_mujoco/results/jump_test_rchk.json
?? wheel_leg_mujoco/results/jump_test_refonly.json
?? wheel_leg_mujoco/results/jump_test_reverted.json
?? wheel_leg_mujoco/results/jump_test_ronly2.json
?? wheel_leg_mujoco/results/jump_test_scan.json
?? wheel_leg_mujoco/results/jump_test_schedchk.json
?? wheel_leg_mujoco/results/jump_test_signfix.json
?? wheel_leg_mujoco/results/jump_test_signflip.json
?? wheel_leg_mujoco/results/jump_test_signflip2.json
?? wheel_leg_mujoco/results/jump_test_taskshare.json
?? wheel_leg_mujoco/results/jump_test_vertical.json
?? wheel_leg_mujoco/results/jump_test_vtor.json
?? wheel_leg_mujoco/results/jump_test_wbc_angle.json
?? wheel_leg_mujoco/results/jump_test_wbcmodel3.json
?? wheel_leg_mujoco/results/m2_nullspace.json.gz
?? wheel_leg_mujoco/results/m2_qp.json.gz
?? wheel_leg_mujoco/results/m2_wls.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_backward.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_forward.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_lateral.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_stand30.json.gz
?? wheel_leg_mujoco/results/m3_wls_backward.json.gz
?? wheel_leg_mujoco/results/m3_wls_forward.json.gz
?? wheel_leg_mujoco/results/m3_wls_lateral.json.gz
?? wheel_leg_mujoco/results/m3_wls_stand30.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v-1.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v0.5.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v1.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v2.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_w1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v-1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v0.5.json.gz
?? wheel_leg_mujoco/results/m4_wls_v1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m4_wls_v2.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_w1.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_push_fwd.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_push_lat.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_stand30.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v1.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v2.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_w1.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_push_fwd.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_push_lat.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_stand30.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v1.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v2.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_w1.0.json.gz
?? wheel_leg_mujoco/results/view_test/
?? wheel_leg_mujoco/tools/jump_diagnostic.py
?? "wheel_leg_mujoco/\350\267\263\350\267\203\345\212\237\350\203\275\346\226\271\346\241\210.md"
```

测试agent：联合几何与状态流观测完成，正式结果有效；状态机保险和压缩控制重建等待书面修改边界确认，当前暂停实现。


---

# 测试agent：正负Fz、执行器映射与JumpPhaseHint核查报告

## 一、执行结论

**测试结论：三项核查结果有效且重复运行一致。当前执行器/Jacobian映射数值一致，正负`F_z`方向产生相反的腿部运动；但接触估计器运行时实际收到的跳跃相位提示全部为`JumpPhaseHint.NONE`，确认存在相位提示接线缺失。**

本轮只使用独立诊断工具和包装式观测器，没有修改控制参数、状态机、接触阈值、WBC/MPC逻辑或压缩控制。

## 二、工具与结果文件

- 诊断工具：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\tools\sign_mapping_diagnostic.py`
- 正式结果1：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\SIGN_MAPPING_run01.json`
- 正式日志1：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\SIGN_MAPPING_run01.log`
- 正式结果2：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\SIGN_MAPPING_run02.json`
- 正式日志2：`C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\SIGN_MAPPING_run02.log`

固定使用工作区虚拟环境：

```text
C:\Users\20060\Desktop\circleg-robot\.venv\Scripts\python.exe
```

测试力来自当前跳跃配置的`JumpConfig.leg_force()`，对应`h=0.10 m`、压缩行程`0.12 m`，没有另外调节力值。

## 三、第0阶段自检

- `sign_mapping_diagnostic.py`语法检查通过。
- CLI导入和帮助检查通过。
- 正常运行完成。
- `run01`与`run02`的映射、正力、负力和相位提示字段完全一致。
- 自检结论：通过。

## 四、正负Fz单步符号结果

| 项目 | 正`F_z` | 负`F_z` |
|---|---:|---:|
| 每腿力 | `+157.172 N` | `-157.172 N` |
| 四腿执行器力矩 | 正向结果 | 严格反向 |
| 右腿长度变化 | `+0.0000795 m` | `-0.0003837 m` |
| 左腿长度变化 | `+0.0000796 m` | `-0.0003841 m` |
| 右左轮隙变化 | `-4.21e-6 / -4.21e-6 m` | `+7.81e-5 / +7.80e-5 m` |
| COM竖直加速度有限差分 | `+13.199 m/s²` | `-9.805 m/s²` |

观测事实：

- 正`F_z`使两腿长度增加，轮隙略减小，质心竖直加速度为正。
- 负`F_z`使两腿长度减小，但轮隙增大，质心竖直加速度为负。
- 因此“压缩力”不能未经确认直接复用正向蹬伸力的符号；负力确实产生压缩方向腿长运动，但同时出现轮隙增加。该结果提交问题排查agent判断，不直接下根因结论。

## 五、Jacobian和执行器映射核查

- Jacobian到执行器力矩的最大残差：`0.000e+00 N*m`。
- 正负力矩反对称最大误差：`0.000e+00 N*m`。
- 执行器顺序：

```text
Right_front_joint_act -> jAB -> dof 9 -> gear 1 -> gain 1
Right_rear_joint_act  -> jAG -> dof 6 -> gear 1 -> gain 1
Left_front_joint_act  -> jIJ -> dof 13 -> gear 1 -> gain 1
Left_rear_joint_act   -> jIO -> dof 17 -> gear 1 -> gain 1
```

- 四个腿执行器控制范围均为`[-40, 40] N*m`。
- 右腿Jacobian驱动关节顺序：`[jAB, jAG]`；被动关节顺序：`[jGH, jBE, jEC, jCF]`。
- 左腿Jacobian驱动关节顺序：`[jIJ, jIO]`；被动关节顺序：`[jOP, jJM, jMK, jKN]`。
- `selection`每个腿执行器行均只映射到对应单一关节dof，且非零值为`1.0`。

结论边界：当前静态映射和单步力矩生成链一致；该核查不等价于已证明真实跳跃压缩方向满足功能目标。

## 六、JumpPhaseHint运行时接线核查

使用包装式观测器记录`GroundContactEstimator.update()`的真实调用参数，1.3 s同工况共记录1300次：

```text
phase_counts = {"none": 1300}
all_phase_none = True
```

跳跃活动期间也没有收到：

```text
JumpPhaseHint.THRUST
JumpPhaseHint.FLIGHT
JumpPhaseHint.LANDING
```

源码与运行时证据一致：

```text
sampling.py:
contact_estimator.update(tuple(forces[:, 2]), self.dt,
                         phase=JumpPhaseHint.NONE).contact
```

因此当前相位门控逻辑虽然存在于`Algorithm.contact_state.GroundContactEstimator`，但跳跃状态机的相位提示没有接入实际采样链路。该项是已确认的接线缺陷，不能继续把当前接触估计结果描述为完整的跳跃双证据检测。

## 七、后续修改边界

本轮核查完成后，仍未实施状态机保险或压缩控制重建。后续修改必须分开进行：

1. 状态机保险：增加前态锁存、`thrust_applied`、离地候选/确认和着地候选/确认，不先改阈值；
2. 压缩控制重建：先依据正负`F_z`结果明确压缩方向，再决定压缩力矩接入方式；不能直接把正向蹬伸力符号复制到压缩段；
3. 接触相位提示接线修复必须单独记录，不能与状态机保险和压缩控制重建混在同一次结果中；
4. 每项修改都必须先保留当前`Q1B_GEOM_baseline_run02`或本报告对应的独立核查结果作为基线，再做自检和正式重放。

测试agent：三项核查完成，证据有效；等待问题排查agent确定状态机保险、压缩控制重建和相位提示接线修复的分步实施顺序。

## 八、工作区现场（git status --short原样）

```text
 M Algorithm/supervision.py
 M wheel_leg_mujoco/MJCF/robot.xml
 M wheel_leg_mujoco/gbc_mpc_wbc/control.py
 M wheel_leg_mujoco/gbc_mpc_wbc/dynamics.py
 M wheel_leg_mujoco/gbc_mpc_wbc/loop.py
 M wheel_leg_mujoco/gbc_mpc_wbc/simulation.py
 M wheel_leg_mujoco/tools/video.py
 M wheel_leg_mujoco/tools/view.py
?? MUJOCO_LOG.TXT
?? URDF_Model/mjcf/URDF_Model_field.xml
?? URDF_Model/mjcf/field_meshes/
?? "docs/agent\344\272\244\344\272\222\346\226\207\346\241\243.md"
?? field_assets/
?? rmuc2026-mujoco/
?? videos/
?? wheel_leg_mujoco/gbc_mpc_wbc/jump.py
?? wheel_leg_mujoco/results/P0_baseline_default_run01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_run01.log
?? wheel_leg_mujoco/results/P0_baseline_default_smoke01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_smoke01.log
?? wheel_leg_mujoco/results/P0_baseline_default_smoke02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_smoke02.log
?? wheel_leg_mujoco/results/P0_baseline_run01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_run01.log
?? wheel_leg_mujoco/results/P0_baseline_run01_inspect.log
?? wheel_leg_mujoco/results/P0_baseline_run02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_run02.log
?? wheel_leg_mujoco/results/P0_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_smoke01.log
?? wheel_leg_mujoco/results/P0_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_smoke02.log
?? wheel_leg_mujoco/results/P6_baseline_default_run01.json.gz
?? wheel_leg_mujoco/results/P6_baseline_default_run01.log
?? wheel_leg_mujoco/results/P6_baseline_run01.json.gz
?? wheel_leg_mujoco/results/P6_baseline_run01.log
?? wheel_leg_mujoco/results/P6_delta_velocity_default_run02.json.gz
?? wheel_leg_mujoco/results/P6_delta_velocity_default_run02.log
?? wheel_leg_mujoco/results/P6_delta_velocity_run01.json.gz
?? wheel_leg_mujoco/results/P6_delta_velocity_run01.log
?? wheel_leg_mujoco/results/P6_key_events.log
?? wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.log
?? wheel_leg_mujoco/results/P6_qfrc_default_run02.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_default_run02.log
?? wheel_leg_mujoco/results/P6_qfrc_run01.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_run01.log
?? wheel_leg_mujoco/results/P6_transition_report.log
?? wheel_leg_mujoco/results/P6_transition_report_compact.log
?? wheel_leg_mujoco/results/Q1A_mpc_audit_default_run01.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run01.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run02.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run01.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run02.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke03.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke04.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke05.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke05.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke06.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke06.log
?? wheel_leg_mujoco/results/Q1B_baseline_run01.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_run01.log
?? wheel_leg_mujoco/results/Q1B_baseline_run02.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_run02.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke01.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke02.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke03.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke04.log
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run01.json
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run01.log
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run02.json
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run02.log
?? wheel_leg_mujoco/results/SIGN_MAPPING_run01.json
?? wheel_leg_mujoco/results/SIGN_MAPPING_run01.log
?? wheel_leg_mujoco/results/SIGN_MAPPING_run02.json
?? wheel_leg_mujoco/results/SIGN_MAPPING_run02.log
?? wheel_leg_mujoco/results/_backup_before_field/
?? wheel_leg_mujoco/results/debug_zero_input/
?? wheel_leg_mujoco/results/jump_test_airsign.json
?? wheel_leg_mujoco/results/jump_test_baseline.json
?? wheel_leg_mujoco/results/jump_test_cg.json
?? wheel_leg_mujoco/results/jump_test_cg2.json
?? wheel_leg_mujoco/results/jump_test_clean.json
?? wheel_leg_mujoco/results/jump_test_comp0.json
?? wheel_leg_mujoco/results/jump_test_comp1.json
?? wheel_leg_mujoco/results/jump_test_crouch.json
?? wheel_leg_mujoco/results/jump_test_crouch014.json
?? wheel_leg_mujoco/results/jump_test_crouch014b.json
?? wheel_leg_mujoco/results/jump_test_crouch020.json
?? wheel_leg_mujoco/results/jump_test_cutforce.json
?? wheel_leg_mujoco/results/jump_test_dbl.json
?? wheel_leg_mujoco/results/jump_test_dbl0.json
?? wheel_leg_mujoco/results/jump_test_dbl1.json
?? wheel_leg_mujoco/results/jump_test_dblfix.json
?? wheel_leg_mujoco/results/jump_test_decoup.json
?? wheel_leg_mujoco/results/jump_test_deployed.json
?? wheel_leg_mujoco/results/jump_test_diag.json
?? wheel_leg_mujoco/results/jump_test_diag2.json
?? wheel_leg_mujoco/results/jump_test_dualcut.json
?? wheel_leg_mujoco/results/jump_test_event1.json
?? wheel_leg_mujoco/results/jump_test_event2.json
?? wheel_leg_mujoco/results/jump_test_event3.json
?? wheel_leg_mujoco/results/jump_test_fix1.json
?? wheel_leg_mujoco/results/jump_test_fix3.json
?? wheel_leg_mujoco/results/jump_test_flight_angle.json
?? wheel_leg_mujoco/results/jump_test_flightdiag.json
?? wheel_leg_mujoco/results/jump_test_follow.json
?? wheel_leg_mujoco/results/jump_test_gcheck.json
?? wheel_leg_mujoco/results/jump_test_gcheck2.json
?? wheel_leg_mujoco/results/jump_test_isolate.json
?? wheel_leg_mujoco/results/jump_test_legchk.json
?? wheel_leg_mujoco/results/jump_test_legrate.json
?? wheel_leg_mujoco/results/jump_test_limit70.json
?? wheel_leg_mujoco/results/jump_test_link.json
?? wheel_leg_mujoco/results/jump_test_m0.json
?? wheel_leg_mujoco/results/jump_test_m1.json
?? wheel_leg_mujoco/results/jump_test_mergefix.json
?? wheel_leg_mujoco/results/jump_test_mergefix2.json
?? wheel_leg_mujoco/results/jump_test_mpcchk.json
?? wheel_leg_mujoco/results/jump_test_mpcdata.json
?? wheel_leg_mujoco/results/jump_test_mpcdata2.json
?? wheel_leg_mujoco/results/jump_test_normal1.json
?? wheel_leg_mujoco/results/jump_test_planA5.json
?? wheel_leg_mujoco/results/jump_test_planA6.json
?? wheel_leg_mujoco/results/jump_test_planAB.json
?? wheel_leg_mujoco/results/jump_test_planA_final.json
?? wheel_leg_mujoco/results/jump_test_planB.json
?? wheel_leg_mujoco/results/jump_test_planB2.json
?? wheel_leg_mujoco/results/jump_test_planB3.json
?? wheel_leg_mujoco/results/jump_test_planD2.json
?? wheel_leg_mujoco/results/jump_test_planDA.json
?? wheel_leg_mujoco/results/jump_test_planDA2.json
?? wheel_leg_mujoco/results/jump_test_plan_a.json
?? wheel_leg_mujoco/results/jump_test_plan_a2.json
?? wheel_leg_mujoco/results/jump_test_policy.json
?? wheel_leg_mujoco/results/jump_test_policy2.json
?? wheel_leg_mujoco/results/jump_test_policy3.json
?? wheel_leg_mujoco/results/jump_test_pre5.json
?? wheel_leg_mujoco/results/jump_test_prechk.json
?? wheel_leg_mujoco/results/jump_test_prehold.json
?? wheel_leg_mujoco/results/jump_test_preload.json
?? wheel_leg_mujoco/results/jump_test_probe_thrust.json
?? wheel_leg_mujoco/results/jump_test_pscan.json
?? wheel_leg_mujoco/results/jump_test_rate.json
?? wheel_leg_mujoco/results/jump_test_rate0.json
?? wheel_leg_mujoco/results/jump_test_rate1.json
?? wheel_leg_mujoco/results/jump_test_ratechk.json
?? wheel_leg_mujoco/results/jump_test_rchk.json
?? wheel_leg_mujoco/results/jump_test_refonly.json
?? wheel_leg_mujoco/results/jump_test_reverted.json
?? wheel_leg_mujoco/results/jump_test_ronly2.json
?? wheel_leg_mujoco/results/jump_test_scan.json
?? wheel_leg_mujoco/results/jump_test_schedchk.json
?? wheel_leg_mujoco/results/jump_test_signfix.json
?? wheel_leg_mujoco/results/jump_test_signflip.json
?? wheel_leg_mujoco/results/jump_test_signflip2.json
?? wheel_leg_mujoco/results/jump_test_taskshare.json
?? wheel_leg_mujoco/results/jump_test_vertical.json
?? wheel_leg_mujoco/results/jump_test_vtor.json
?? wheel_leg_mujoco/results/jump_test_wbc_angle.json
?? wheel_leg_mujoco/results/jump_test_wbcmodel3.json
?? wheel_leg_mujoco/results/m2_nullspace.json.gz
?? wheel_leg_mujoco/results/m2_qp.json.gz
?? wheel_leg_mujoco/results/m2_wls.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_backward.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_forward.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_lateral.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_stand30.json.gz
?? wheel_leg_mujoco/results/m3_wls_backward.json.gz
?? wheel_leg_mujoco/results/m3_wls_forward.json.gz
?? wheel_leg_mujoco/results/m3_wls_lateral.json.gz
?? wheel_leg_mujoco/results/m3_wls_stand30.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v-1.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v0.5.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v1.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v2.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_w1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v-1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v0.5.json.gz
?? wheel_leg_mujoco/results/m4_wls_v1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m4_wls_v2.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_w1.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_push_fwd.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_push_lat.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_stand30.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v1.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v2.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_w1.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_push_fwd.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_push_lat.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_stand30.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v1.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v2.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_w1.0.json.gz
?? wheel_leg_mujoco/results/view_test/
?? wheel_leg_mujoco/tools/jump_diagnostic.py
?? wheel_leg_mujoco/tools/sign_mapping_diagnostic.py
?? "wheel_leg_mujoco/\350\267\263\350\267\203\345\212\237\350\203\275\346\226\271\346\241\210.md"
```


---

# 测试agent：状态保险、相位提示接线与压缩控制重建测试报告

## 一、执行顺序与边界

本轮按用户指定顺序执行：

```text
1. 状态机前态保险
2. JumpPhaseHint接线修复
3. 压缩控制逻辑重建
4. 自检、冒烟和正式测试
```

未修改增益、权重、阈值、确认时间、力矩限幅、MPC参数或WBC参数。

## 二、状态保险与相位提示阶段

修改内容：

- `THRUST -> FLIGHT`必须先确认上一拍实际写入非零蹬伸力矩；
- `FLIGHT -> LANDING`必须同时确认双轮几何接触和估计接触；
- `LANDING -> RECOVER`、`RECOVER -> IDLE`增加双轮几何/估计接触前置条件；
- `FeedbackSampler.update()`接收当前`JumpPhaseHint`；
- 运行循环按`JumpCommand`把`THRUST/FLIGHT/LANDING`传给接触估计器；
- `JumpObservation`增加几何接触和`thrust_applied`观测。

阶段结果文件：

```text
C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\STATE_HINT_baseline_smoke01.json.gz
C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\STATE_HINT_baseline_smoke02.json.gz
```

自检结果：

```text
两次冒烟逐帧一致
phase_transition_count = 4
phase_transition_assertion_failures = 0
thrust_applied_frames = 1
phase_frame_counts = IDLE 1000 / COMPRESS 235 / THRUST 2 / FLIGHT 12 / ABORT 51
```

阶段结论：**状态保险逻辑生效，原先的`THRUST -> FLIGHT`和`FLIGHT -> LANDING`断言失败在该阶段冒烟中消失；接触提示接线代码通过编译和运行。**

## 三、压缩控制重建

修改内容：

- `COMPRESS`不再设置`leg_length_target`；
- `THRUST`不再设置`leg_length_target`；
- `COMPRESS`使用已完成符号核查的负向等效力矩：`leg_force=-config.leg_force()`；
- `THRUST`保留正向等效力矩；
- 跳跃活动期间从WBC运动任务中移除`length_R/length_L`，避免WBC腿长PD抵消已知跳跃力矩；
- WBC只保留姿态、腿角、滚转和接触残差任务。

正式结果文件：

```text
C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\V4_JUMP_baseline_run01.json.gz
C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\V4_JUMP_baseline_run01.log
```

冒烟重复结果：

```text
V4_JUMP_baseline_smoke03/04
两次结果逐帧一致
```

## 四、V4正式结果

```text
simulated_seconds = 3.0 s
phase_frame_counts = IDLE 1000 / COMPRESS 2000
THRUST = 0
FLIGHT = 0
ABORT = 0
airborne_frames = 0
thrust_applied_frames = 0
wbc_calls = 750
```

压缩结果：

- 最大腿长：约`0.280464 m`；
- 最小腿长：约`0.279708 m`；
- 压缩行程目标：`0.120 m`；
- 实际腿长变化约`0.0003 m`，未达到压缩终点`0.160 m`；
- 3.0 s内始终停留在`COMPRESS`，未进入THRUST。

逐拍证据显示：

- `tau_jump`在COMPRESS期间确实非零；
- WBC任务中已无`length_R/length_L`；
- WBC求解可行，输出正常写入；
- 但外部压缩力矩没有将腿长推进到压缩终点。

因此本次压缩重建结果判定为：**部署链生效，但功能目标未达成，测试失败。** 不能据此继续调大力值、修改符号或调整WBC参数。

## 五、结果有效性

- 工具语法和导入：通过。
- 冒烟重复性：通过。
- 正式V4结果：有效，失败现象真实可复现。
- 失败不是评价工具字段问题：`tau_jump`、WBC任务列表、最终ctrl和输出来源均已记录。
- 当前压缩重建不能宣称完成，也不能宣称跳跃功能恢复。

## 六、当前后续边界

状态保险和相位提示接线可以保留；压缩控制重建需要问题排查agent继续判定：

1. 外部压缩力矩在接触约束下没有产生足够腿长变化的具体动力学原因；
2. 是否需要保留某个非压缩PD残差任务或改变控制权限边界；
3. 在没有新的书面方案前，不调压缩力、不改WBC增益、不改接触阈值。

测试agent：状态保险与相位提示接线测试通过；压缩控制重建部署成功但功能验证失败，现场和结果已保留，等待下一步书面规划。

## 七、工作区现场（git status --short原样）

```text
 M Algorithm/supervision.py
 M wheel_leg_mujoco/MJCF/robot.xml
 M wheel_leg_mujoco/gbc_mpc_wbc/control.py
 M wheel_leg_mujoco/gbc_mpc_wbc/dynamics.py
 M wheel_leg_mujoco/gbc_mpc_wbc/loop.py
 M wheel_leg_mujoco/gbc_mpc_wbc/sampling.py
 M wheel_leg_mujoco/gbc_mpc_wbc/simulation.py
 M wheel_leg_mujoco/tools/video.py
 M wheel_leg_mujoco/tools/view.py
?? MUJOCO_LOG.TXT
?? URDF_Model/mjcf/URDF_Model_field.xml
?? URDF_Model/mjcf/field_meshes/
?? "docs/agent\344\272\244\344\272\222\346\226\207\346\241\243.md"
?? field_assets/
?? rmuc2026-mujoco/
?? videos/
?? wheel_leg_mujoco/gbc_mpc_wbc/jump.py
?? wheel_leg_mujoco/results/P0_baseline_default_run01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_run01.log
?? wheel_leg_mujoco/results/P0_baseline_default_smoke01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_smoke01.log
?? wheel_leg_mujoco/results/P0_baseline_default_smoke02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_smoke02.log
?? wheel_leg_mujoco/results/P0_baseline_run01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_run01.log
?? wheel_leg_mujoco/results/P0_baseline_run01_inspect.log
?? wheel_leg_mujoco/results/P0_baseline_run02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_run02.log
?? wheel_leg_mujoco/results/P0_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_smoke01.log
?? wheel_leg_mujoco/results/P0_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_smoke02.log
?? wheel_leg_mujoco/results/P6_baseline_default_run01.json.gz
?? wheel_leg_mujoco/results/P6_baseline_default_run01.log
?? wheel_leg_mujoco/results/P6_baseline_run01.json.gz
?? wheel_leg_mujoco/results/P6_baseline_run01.log
?? wheel_leg_mujoco/results/P6_delta_velocity_default_run02.json.gz
?? wheel_leg_mujoco/results/P6_delta_velocity_default_run02.log
?? wheel_leg_mujoco/results/P6_delta_velocity_run01.json.gz
?? wheel_leg_mujoco/results/P6_delta_velocity_run01.log
?? wheel_leg_mujoco/results/P6_key_events.log
?? wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.log
?? wheel_leg_mujoco/results/P6_qfrc_default_run02.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_default_run02.log
?? wheel_leg_mujoco/results/P6_qfrc_run01.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_run01.log
?? wheel_leg_mujoco/results/P6_transition_report.log
?? wheel_leg_mujoco/results/P6_transition_report_compact.log
?? wheel_leg_mujoco/results/Q1A_mpc_audit_default_run01.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run01.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run02.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run01.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run02.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke03.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke04.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke05.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke05.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke06.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke06.log
?? wheel_leg_mujoco/results/Q1B_baseline_run01.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_run01.log
?? wheel_leg_mujoco/results/Q1B_baseline_run02.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_run02.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke01.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke02.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke03.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke04.log
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run01.json
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run01.log
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run02.json
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run02.log
?? wheel_leg_mujoco/results/SIGN_MAPPING_run01.json
?? wheel_leg_mujoco/results/SIGN_MAPPING_run01.log
?? wheel_leg_mujoco/results/SIGN_MAPPING_run02.json
?? wheel_leg_mujoco/results/SIGN_MAPPING_run02.log
?? wheel_leg_mujoco/results/STATE_HINT_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/STATE_HINT_baseline_smoke01.log
?? wheel_leg_mujoco/results/STATE_HINT_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/STATE_HINT_baseline_smoke02.log
?? wheel_leg_mujoco/results/V4_JUMP_baseline_run01.json.gz
?? wheel_leg_mujoco/results/V4_JUMP_baseline_run01.log
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke01.log
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke02.log
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke03.log
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke04.log
?? wheel_leg_mujoco/results/_backup_before_field/
?? wheel_leg_mujoco/results/debug_zero_input/
?? wheel_leg_mujoco/results/jump_test_airsign.json
?? wheel_leg_mujoco/results/jump_test_baseline.json
?? wheel_leg_mujoco/results/jump_test_cg.json
?? wheel_leg_mujoco/results/jump_test_cg2.json
?? wheel_leg_mujoco/results/jump_test_clean.json
?? wheel_leg_mujoco/results/jump_test_comp0.json
?? wheel_leg_mujoco/results/jump_test_comp1.json
?? wheel_leg_mujoco/results/jump_test_crouch.json
?? wheel_leg_mujoco/results/jump_test_crouch014.json
?? wheel_leg_mujoco/results/jump_test_crouch014b.json
?? wheel_leg_mujoco/results/jump_test_crouch020.json
?? wheel_leg_mujoco/results/jump_test_cutforce.json
?? wheel_leg_mujoco/results/jump_test_dbl.json
?? wheel_leg_mujoco/results/jump_test_dbl0.json
?? wheel_leg_mujoco/results/jump_test_dbl1.json
?? wheel_leg_mujoco/results/jump_test_dblfix.json
?? wheel_leg_mujoco/results/jump_test_decoup.json
?? wheel_leg_mujoco/results/jump_test_deployed.json
?? wheel_leg_mujoco/results/jump_test_diag.json
?? wheel_leg_mujoco/results/jump_test_diag2.json
?? wheel_leg_mujoco/results/jump_test_dualcut.json
?? wheel_leg_mujoco/results/jump_test_event1.json
?? wheel_leg_mujoco/results/jump_test_event2.json
?? wheel_leg_mujoco/results/jump_test_event3.json
?? wheel_leg_mujoco/results/jump_test_fix1.json
?? wheel_leg_mujoco/results/jump_test_fix3.json
?? wheel_leg_mujoco/results/jump_test_flight_angle.json
?? wheel_leg_mujoco/results/jump_test_flightdiag.json
?? wheel_leg_mujoco/results/jump_test_follow.json
?? wheel_leg_mujoco/results/jump_test_gcheck.json
?? wheel_leg_mujoco/results/jump_test_gcheck2.json
?? wheel_leg_mujoco/results/jump_test_isolate.json
?? wheel_leg_mujoco/results/jump_test_legchk.json
?? wheel_leg_mujoco/results/jump_test_legrate.json
?? wheel_leg_mujoco/results/jump_test_limit70.json
?? wheel_leg_mujoco/results/jump_test_link.json
?? wheel_leg_mujoco/results/jump_test_m0.json
?? wheel_leg_mujoco/results/jump_test_m1.json
?? wheel_leg_mujoco/results/jump_test_mergefix.json
?? wheel_leg_mujoco/results/jump_test_mergefix2.json
?? wheel_leg_mujoco/results/jump_test_mpcchk.json
?? wheel_leg_mujoco/results/jump_test_mpcdata.json
?? wheel_leg_mujoco/results/jump_test_mpcdata2.json
?? wheel_leg_mujoco/results/jump_test_normal1.json
?? wheel_leg_mujoco/results/jump_test_planA5.json
?? wheel_leg_mujoco/results/jump_test_planA6.json
?? wheel_leg_mujoco/results/jump_test_planAB.json
?? wheel_leg_mujoco/results/jump_test_planA_final.json
?? wheel_leg_mujoco/results/jump_test_planB.json
?? wheel_leg_mujoco/results/jump_test_planB2.json
?? wheel_leg_mujoco/results/jump_test_planB3.json
?? wheel_leg_mujoco/results/jump_test_planD2.json
?? wheel_leg_mujoco/results/jump_test_planDA.json
?? wheel_leg_mujoco/results/jump_test_planDA2.json
?? wheel_leg_mujoco/results/jump_test_plan_a.json
?? wheel_leg_mujoco/results/jump_test_plan_a2.json
?? wheel_leg_mujoco/results/jump_test_policy.json
?? wheel_leg_mujoco/results/jump_test_policy2.json
?? wheel_leg_mujoco/results/jump_test_policy3.json
?? wheel_leg_mujoco/results/jump_test_pre5.json
?? wheel_leg_mujoco/results/jump_test_prechk.json
?? wheel_leg_mujoco/results/jump_test_prehold.json
?? wheel_leg_mujoco/results/jump_test_preload.json
?? wheel_leg_mujoco/results/jump_test_probe_thrust.json
?? wheel_leg_mujoco/results/jump_test_pscan.json
?? wheel_leg_mujoco/results/jump_test_rate.json
?? wheel_leg_mujoco/results/jump_test_rate0.json
?? wheel_leg_mujoco/results/jump_test_rate1.json
?? wheel_leg_mujoco/results/jump_test_ratechk.json
?? wheel_leg_mujoco/results/jump_test_rchk.json
?? wheel_leg_mujoco/results/jump_test_refonly.json
?? wheel_leg_mujoco/results/jump_test_reverted.json
?? wheel_leg_mujoco/results/jump_test_ronly2.json
?? wheel_leg_mujoco/results/jump_test_scan.json
?? wheel_leg_mujoco/results/jump_test_schedchk.json
?? wheel_leg_mujoco/results/jump_test_signfix.json
?? wheel_leg_mujoco/results/jump_test_signflip.json
?? wheel_leg_mujoco/results/jump_test_signflip2.json
?? wheel_leg_mujoco/results/jump_test_taskshare.json
?? wheel_leg_mujoco/results/jump_test_vertical.json
?? wheel_leg_mujoco/results/jump_test_vtor.json
?? wheel_leg_mujoco/results/jump_test_wbc_angle.json
?? wheel_leg_mujoco/results/jump_test_wbcmodel3.json
?? wheel_leg_mujoco/results/m2_nullspace.json.gz
?? wheel_leg_mujoco/results/m2_qp.json.gz
?? wheel_leg_mujoco/results/m2_wls.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_backward.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_forward.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_lateral.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_stand30.json.gz
?? wheel_leg_mujoco/results/m3_wls_backward.json.gz
?? wheel_leg_mujoco/results/m3_wls_forward.json.gz
?? wheel_leg_mujoco/results/m3_wls_lateral.json.gz
?? wheel_leg_mujoco/results/m3_wls_stand30.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v-1.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v0.5.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v1.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v2.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_w1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v-1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v0.5.json.gz
?? wheel_leg_mujoco/results/m4_wls_v1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m4_wls_v2.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_w1.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_push_fwd.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_push_lat.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_stand30.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v1.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v2.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_w1.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_push_fwd.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_push_lat.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_stand30.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v1.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v2.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_w1.0.json.gz
?? wheel_leg_mujoco/results/view_test/
?? wheel_leg_mujoco/tools/jump_diagnostic.py
?? wheel_leg_mujoco/tools/sign_mapping_diagnostic.py
?? "wheel_leg_mujoco/\350\267\263\350\267\203\345\212\237\350\203\275\346\226\271\346\241\210.md"
```


---

# 测试agent：最终电机输出与髋关节力矩审计报告

## 一、结论

**当前输出链路已经符合“解耦后合并总力矩，再直接传给电机”的设计，不需要修改核心输出逻辑。**

实际链路为：

```text
WBC求解结果 wbc_raw_torque
        +
Jacobian等效力矩 tau_jump
        ↓
合并总力矩 torque
        ↓
write_actuator_output(data, torque)
        ↓
data.ctrl
        ↓
MuJoCo actuator
```

输出级没有再次调用Jacobian，也没有对合并后的总力矩重复映射。

## 二、代码证据

- `control.py`中：

```python
torque, result = control.solve_wbc(...)
raw_torque = np.asarray(torque, dtype=float).copy()
torque = torque + tau_jump
```

- `loop.py`中：

```python
torque = outcome.torque
write_actuator_output(d, torque)
```

- `output.py`中：

```python
def write_actuator_output(data, torque):
    data.ctrl[:] = torque
```

当前模型执行器顺序为：

```text
0 Right_front_joint_act
1 Right_rear_joint_act
2 Left_front_joint_act
3 Left_rear_joint_act
4 Right_Wheel_act
5 Left_Wheel_act
```

四个腿/髋执行器的gear和gain均为1，没有额外比例变换。

## 三、正式V4逐拍审计

数据源：

```text
C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\V4_JUMP_baseline_run01.json.gz
```

审计文件：

```text
C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\V4_FINAL_MOTOR_AUDIT_run01.json
C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\V4_FINAL_MOTOR_AUDIT_run01.log
```

750个WBC周期中：

```text
max |data.ctrl - (wbc_raw_torque + tau_jump)| = 0.000e+00 N*m
```

说明逐拍最终`data.ctrl`与“WBC剩余力矩 + Jacobian跳跃力矩”完全一致。

## 四、髋/腿关节电机具体力矩

### t=1.000 s，COMPRESS

| 电机 | WBC原始力矩 | Jacobian tau_jump | 合并总力矩 | 最终data.ctrl |
|---|---:|---:|---:|---:|
| Right_front_joint_act | -28.067105 | +20.564397 | -7.502707 | -7.502707 |
| Right_rear_joint_act | +33.643953 | -26.109490 | +7.534463 | +7.534463 |
| Left_front_joint_act | +28.057160 | -20.564607 | +7.492553 | +7.492553 |
| Left_rear_joint_act | -33.635651 | +26.109687 | -7.525964 | -7.525964 |

### t=1.100 s，COMPRESS

| 电机 | WBC原始力矩 | Jacobian tau_jump | 合并总力矩 | 最终data.ctrl |
|---|---:|---:|---:|---:|
| Right_front_joint_act | -28.087907 | +20.565911 | -7.521996 | -7.521996 |
| Right_rear_joint_act | +33.662870 | -26.108854 | +7.554016 | +7.554016 |
| Left_front_joint_act | +28.077935 | -20.566120 | +7.511815 | +7.511815 |
| Left_rear_joint_act | -33.654549 | +26.109052 | -7.545497 | -7.545497 |

### t=1.200 s，COMPRESS

| 电机 | WBC原始力矩 | Jacobian tau_jump | 合并总力矩 | 最终data.ctrl |
|---|---:|---:|---:|---:|
| Right_front_joint_act | -28.107824 | +20.567515 | -7.540310 | -7.540310 |
| Right_rear_joint_act | +33.680623 | -26.108142 | +7.572481 | +7.572481 |
| Left_front_joint_act | +28.097824 | -20.567722 | +7.530102 | +7.530102 |
| Left_rear_joint_act | -33.672284 | +26.108342 | -7.563942 | -7.563942 |

## 五、整个正式V4范围

| 电机 | WBC原始范围 | Jacobian tau_jump范围 | 最终data.ctrl范围 |
|---|---:|---:|---:|
| Right_front_joint_act | [-28.123042, -7.489908] | [0, 20.569243] | [-7.553799, -7.489908] |
| Right_rear_joint_act | [7.522072, 33.693377] | [-26.110677, 0] | [7.521791, 7.585993] |
| Left_front_joint_act | [7.479860, 28.113019] | [-20.569449, 0] | [7.479860, 7.543570] |
| Left_rear_joint_act | [-33.685024, -7.513539] | [0, 26.110876] | [-7.577439, -7.513310] |

轮电机的`tau_jump`全程为0；轮电机最终输出范围为：

```text
Right_Wheel_act: [-0.000680, 0.002737] N*m
Left_Wheel_act : [-0.000666, 0.002783] N*m
```

测试agent：最终总力矩已经直接传给电机，当前没有发现重复Jacobian、重复映射或中间丢失；本轮不修改输出链路。

## 六、工作区现场（git status --short原样）

```text
 M Algorithm/supervision.py
 M wheel_leg_mujoco/MJCF/robot.xml
 M wheel_leg_mujoco/gbc_mpc_wbc/control.py
 M wheel_leg_mujoco/gbc_mpc_wbc/dynamics.py
 M wheel_leg_mujoco/gbc_mpc_wbc/loop.py
 M wheel_leg_mujoco/gbc_mpc_wbc/sampling.py
 M wheel_leg_mujoco/gbc_mpc_wbc/simulation.py
 M wheel_leg_mujoco/tools/video.py
 M wheel_leg_mujoco/tools/view.py
?? MUJOCO_LOG.TXT
?? URDF_Model/mjcf/URDF_Model_field.xml
?? URDF_Model/mjcf/field_meshes/
?? "docs/agent\344\272\244\344\272\222\346\226\207\346\241\243.md"
?? field_assets/
?? rmuc2026-mujoco/
?? videos/
?? wheel_leg_mujoco/gbc_mpc_wbc/jump.py
?? wheel_leg_mujoco/results/P0_baseline_default_run01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_run01.log
?? wheel_leg_mujoco/results/P0_baseline_default_smoke01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_smoke01.log
?? wheel_leg_mujoco/results/P0_baseline_default_smoke02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_default_smoke02.log
?? wheel_leg_mujoco/results/P0_baseline_run01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_run01.log
?? wheel_leg_mujoco/results/P0_baseline_run01_inspect.log
?? wheel_leg_mujoco/results/P0_baseline_run02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_run02.log
?? wheel_leg_mujoco/results/P0_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/P0_baseline_smoke01.log
?? wheel_leg_mujoco/results/P0_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/P0_baseline_smoke02.log
?? wheel_leg_mujoco/results/P6_baseline_default_run01.json.gz
?? wheel_leg_mujoco/results/P6_baseline_default_run01.log
?? wheel_leg_mujoco/results/P6_baseline_run01.json.gz
?? wheel_leg_mujoco/results/P6_baseline_run01.log
?? wheel_leg_mujoco/results/P6_delta_velocity_default_run02.json.gz
?? wheel_leg_mujoco/results/P6_delta_velocity_default_run02.log
?? wheel_leg_mujoco/results/P6_delta_velocity_run01.json.gz
?? wheel_leg_mujoco/results/P6_delta_velocity_run01.log
?? wheel_leg_mujoco/results/P6_key_events.log
?? wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_default_extended_run01.log
?? wheel_leg_mujoco/results/P6_qfrc_default_run02.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_default_run02.log
?? wheel_leg_mujoco/results/P6_qfrc_run01.json.gz
?? wheel_leg_mujoco/results/P6_qfrc_run01.log
?? wheel_leg_mujoco/results/P6_transition_report.log
?? wheel_leg_mujoco/results/P6_transition_report_compact.log
?? wheel_leg_mujoco/results/Q1A_mpc_audit_default_run01.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run01.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run02.json
?? wheel_leg_mujoco/results/Q1B_GEOM_audit_run02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run01.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run02.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_run02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke01.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke02.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke03.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke04.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke05.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke05.log
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke06.json.gz
?? wheel_leg_mujoco/results/Q1B_GEOM_baseline_smoke06.log
?? wheel_leg_mujoco/results/Q1B_baseline_run01.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_run01.log
?? wheel_leg_mujoco/results/Q1B_baseline_run02.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_run02.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke01.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke02.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke03.log
?? wheel_leg_mujoco/results/Q1B_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/Q1B_baseline_smoke04.log
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run01.json
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run01.log
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run02.json
?? wheel_leg_mujoco/results/Q1B_wbc_audit_run02.log
?? wheel_leg_mujoco/results/SIGN_MAPPING_run01.json
?? wheel_leg_mujoco/results/SIGN_MAPPING_run01.log
?? wheel_leg_mujoco/results/SIGN_MAPPING_run02.json
?? wheel_leg_mujoco/results/SIGN_MAPPING_run02.log
?? wheel_leg_mujoco/results/STATE_HINT_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/STATE_HINT_baseline_smoke01.log
?? wheel_leg_mujoco/results/STATE_HINT_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/STATE_HINT_baseline_smoke02.log
?? wheel_leg_mujoco/results/V4_FINAL_MOTOR_AUDIT_run01.json
?? wheel_leg_mujoco/results/V4_FINAL_MOTOR_AUDIT_run01.log
?? wheel_leg_mujoco/results/V4_JUMP_baseline_run01.json.gz
?? wheel_leg_mujoco/results/V4_JUMP_baseline_run01.log
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke01.json.gz
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke01.log
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke02.json.gz
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke02.log
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke03.json.gz
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke03.log
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke04.json.gz
?? wheel_leg_mujoco/results/V4_JUMP_baseline_smoke04.log
?? wheel_leg_mujoco/results/_backup_before_field/
?? wheel_leg_mujoco/results/debug_zero_input/
?? wheel_leg_mujoco/results/jump_test_airsign.json
?? wheel_leg_mujoco/results/jump_test_baseline.json
?? wheel_leg_mujoco/results/jump_test_cg.json
?? wheel_leg_mujoco/results/jump_test_cg2.json
?? wheel_leg_mujoco/results/jump_test_clean.json
?? wheel_leg_mujoco/results/jump_test_comp0.json
?? wheel_leg_mujoco/results/jump_test_comp1.json
?? wheel_leg_mujoco/results/jump_test_crouch.json
?? wheel_leg_mujoco/results/jump_test_crouch014.json
?? wheel_leg_mujoco/results/jump_test_crouch014b.json
?? wheel_leg_mujoco/results/jump_test_crouch020.json
?? wheel_leg_mujoco/results/jump_test_cutforce.json
?? wheel_leg_mujoco/results/jump_test_dbl.json
?? wheel_leg_mujoco/results/jump_test_dbl0.json
?? wheel_leg_mujoco/results/jump_test_dbl1.json
?? wheel_leg_mujoco/results/jump_test_dblfix.json
?? wheel_leg_mujoco/results/jump_test_decoup.json
?? wheel_leg_mujoco/results/jump_test_deployed.json
?? wheel_leg_mujoco/results/jump_test_diag.json
?? wheel_leg_mujoco/results/jump_test_diag2.json
?? wheel_leg_mujoco/results/jump_test_dualcut.json
?? wheel_leg_mujoco/results/jump_test_event1.json
?? wheel_leg_mujoco/results/jump_test_event2.json
?? wheel_leg_mujoco/results/jump_test_event3.json
?? wheel_leg_mujoco/results/jump_test_fix1.json
?? wheel_leg_mujoco/results/jump_test_fix3.json
?? wheel_leg_mujoco/results/jump_test_flight_angle.json
?? wheel_leg_mujoco/results/jump_test_flightdiag.json
?? wheel_leg_mujoco/results/jump_test_follow.json
?? wheel_leg_mujoco/results/jump_test_gcheck.json
?? wheel_leg_mujoco/results/jump_test_gcheck2.json
?? wheel_leg_mujoco/results/jump_test_isolate.json
?? wheel_leg_mujoco/results/jump_test_legchk.json
?? wheel_leg_mujoco/results/jump_test_legrate.json
?? wheel_leg_mujoco/results/jump_test_limit70.json
?? wheel_leg_mujoco/results/jump_test_link.json
?? wheel_leg_mujoco/results/jump_test_m0.json
?? wheel_leg_mujoco/results/jump_test_m1.json
?? wheel_leg_mujoco/results/jump_test_mergefix.json
?? wheel_leg_mujoco/results/jump_test_mergefix2.json
?? wheel_leg_mujoco/results/jump_test_mpcchk.json
?? wheel_leg_mujoco/results/jump_test_mpcdata.json
?? wheel_leg_mujoco/results/jump_test_mpcdata2.json
?? wheel_leg_mujoco/results/jump_test_normal1.json
?? wheel_leg_mujoco/results/jump_test_planA5.json
?? wheel_leg_mujoco/results/jump_test_planA6.json
?? wheel_leg_mujoco/results/jump_test_planAB.json
?? wheel_leg_mujoco/results/jump_test_planA_final.json
?? wheel_leg_mujoco/results/jump_test_planB.json
?? wheel_leg_mujoco/results/jump_test_planB2.json
?? wheel_leg_mujoco/results/jump_test_planB3.json
?? wheel_leg_mujoco/results/jump_test_planD2.json
?? wheel_leg_mujoco/results/jump_test_planDA.json
?? wheel_leg_mujoco/results/jump_test_planDA2.json
?? wheel_leg_mujoco/results/jump_test_plan_a.json
?? wheel_leg_mujoco/results/jump_test_plan_a2.json
?? wheel_leg_mujoco/results/jump_test_policy.json
?? wheel_leg_mujoco/results/jump_test_policy2.json
?? wheel_leg_mujoco/results/jump_test_policy3.json
?? wheel_leg_mujoco/results/jump_test_pre5.json
?? wheel_leg_mujoco/results/jump_test_prechk.json
?? wheel_leg_mujoco/results/jump_test_prehold.json
?? wheel_leg_mujoco/results/jump_test_preload.json
?? wheel_leg_mujoco/results/jump_test_probe_thrust.json
?? wheel_leg_mujoco/results/jump_test_pscan.json
?? wheel_leg_mujoco/results/jump_test_rate.json
?? wheel_leg_mujoco/results/jump_test_rate0.json
?? wheel_leg_mujoco/results/jump_test_rate1.json
?? wheel_leg_mujoco/results/jump_test_ratechk.json
?? wheel_leg_mujoco/results/jump_test_rchk.json
?? wheel_leg_mujoco/results/jump_test_refonly.json
?? wheel_leg_mujoco/results/jump_test_reverted.json
?? wheel_leg_mujoco/results/jump_test_ronly2.json
?? wheel_leg_mujoco/results/jump_test_scan.json
?? wheel_leg_mujoco/results/jump_test_schedchk.json
?? wheel_leg_mujoco/results/jump_test_signfix.json
?? wheel_leg_mujoco/results/jump_test_signflip.json
?? wheel_leg_mujoco/results/jump_test_signflip2.json
?? wheel_leg_mujoco/results/jump_test_taskshare.json
?? wheel_leg_mujoco/results/jump_test_vertical.json
?? wheel_leg_mujoco/results/jump_test_vtor.json
?? wheel_leg_mujoco/results/jump_test_wbc_angle.json
?? wheel_leg_mujoco/results/jump_test_wbcmodel3.json
?? wheel_leg_mujoco/results/m2_nullspace.json.gz
?? wheel_leg_mujoco/results/m2_qp.json.gz
?? wheel_leg_mujoco/results/m2_wls.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_backward.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_forward.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_lateral.json.gz
?? wheel_leg_mujoco/results/m3_nullspace_stand30.json.gz
?? wheel_leg_mujoco/results/m3_wls_backward.json.gz
?? wheel_leg_mujoco/results/m3_wls_forward.json.gz
?? wheel_leg_mujoco/results/m3_wls_lateral.json.gz
?? wheel_leg_mujoco/results/m3_wls_stand30.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v-1.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v0.5.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v1.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_v2.0.json.gz
?? wheel_leg_mujoco/results/m4_nullspace_w1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v-1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v0.5.json.gz
?? wheel_leg_mujoco/results/m4_wls_v1.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m4_wls_v2.0.json.gz
?? wheel_leg_mujoco/results/m4_wls_w1.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_push_fwd.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_push_lat.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_stand30.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v1.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_v2.0.json.gz
?? wheel_leg_mujoco/results/m5_held_leg_w1.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_push_fwd.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_push_lat.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_stand30.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v1.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v1.0w0.5.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_v2.0.json.gz
?? wheel_leg_mujoco/results/m5_leg_input_w1.0.json.gz
?? wheel_leg_mujoco/results/view_test/
?? wheel_leg_mujoco/tools/jump_diagnostic.py
?? wheel_leg_mujoco/tools/sign_mapping_diagnostic.py
?? "wheel_leg_mujoco/\350\267\263\350\267\203\345\212\237\350\203\275\346\226\271\346\241\210.md"
```
