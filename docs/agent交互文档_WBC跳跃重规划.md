# 问题排查agent：WBC跳跃控制与离地检测重规划

更新时间：2026-10-09

## 0. 文档与工作边界

- 本文件是后续问题排查agent与测试agent的唯一交互文档。
- 历史文档 `docs/agent交互文档.md` 仅作为历史资料，不再追加。
- 本文件中的结论只基于当前对话、当前源码和已经生成的测试结果。
- 当前角色只负责问题排查、方案设计、测试要求和结果解释，不代替测试agent执行正式测试。
- 未经用户明确批准，不修改控制算法、模型、参数、增益、权重、阈值、限幅或MPC腿长拟合。
- 测试目的首先是确认根因，不是通过调参让结果看起来更好。
- 所有测试脚本必须先完成语法/导入、字段口径、单工况冒烟和重复性自检。

---

## 1. 当前核心结论

### 1.1 跳跃力不应再通过腿长终点直接交给WBC

此前方案是：

```text
COMPRESS:
    给WBC一个压缩终点腿长，由腿长PD任务驱动压缩

THRUST:
    通过腿部雅可比生成tau_jump，同时仍给WBC腿长目标
```

该方案会让主动腿长变化、姿态控制、MPC参考和接触约束相互影响，不能保证跳跃力被正确建模，也不能保证压缩期间轮子保持接地。

### 1.2 external_force不是“请WBC执行这份跳跃力”接口

`external_force`表示已经存在的已知外部广义力。它改变WBC动力学右端，但不会自动要求WBC实现该力造成的跳跃运动。

V4已经证明：

```text
tau_jump进入WBC external通道
WBC继续满足原有姿态/接触/闭链任务
WBC原始力矩反向抵消tau_jump的大部分作用
最终总力矩仍接近站立控制输出
```

因此撤回此前“只要把tau_jump放进external_force，WBC就不会抵消它”的判断。

### 1.3 推荐方向：WBC统一求解跳跃运动和总执行器力矩

```text
JumpPlanner
    -> 相位管理
    -> 生成压缩/制动/释放阶段的质心竖直目标

WBC
    -> 接触、闭链、欠驱动约束
    -> pitch、roll、腿角、质心竖直任务
    -> 一次求解全部执行器总力矩

Output
    -> 只写WBC求出的总力矩
```

跳跃模块不再单独生成并叠加`tau_jump`。WBC成为唯一执行器力矩求解者，但JumpPlanner仍负责相位、参考生成和状态事件管理。

---

## 2. 已有证据

### 2.1 V4压缩控制重建失败

结果文件：

```text
C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\V4_JUMP_baseline_run01.json.gz
C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\V4_JUMP_baseline_run01.log
C:\Users\20060\Desktop\circleg-robot\wheel_leg_mujoco\results\V4_FINAL_MOTOR_AUDIT_run01.json
```

测试配置：

```text
model_variant = held_leg
preset = tuned-agile
jump_height = 0.10 m
jump_stroke = 0.12 m
jump_time = 1.0 s
duration = 3.0 s
```

实际结果：

```text
IDLE = 1000 frames
COMPRESS = 2000 frames
THRUST = 0
FLIGHT = 0
ABORT = 0
airborne = 0 frames
WBC calls = 750
```

腿长只在约`0.279708～0.280464 m`范围内变化，实际变化约`0.0003 m`，没有达到`0.160 m`压缩终点。

这说明：

- 压缩力矩没有丢失；
- 最终输出链没有丢失；
- WBC没有被离地检测切断；
- 但当前WBC任务没有形成有效的竖直压缩运动目标。

### 2.2 输出链路没有发现重复映射或丢失

V4电机审计显示：

```text
data.ctrl = wbc_raw_torque + tau_jump
```

逐拍最大误差为`0.000e+00 N*m`。这证明输出级没有丢数据，但不代表跳跃行为正确。

### 2.3 V4中WBC抵消了大部分压缩前馈

典型t=1.000 s数据：

| 执行器 | WBC原始力矩 | tau_jump | 最终ctrl |
|---|---:|---:|---:|
| Right_front | -28.067105 | +20.564397 | -7.502707 |
| Right_rear | +33.643953 | -26.109490 | +7.534463 |
| Left_front | +28.057160 | -20.564607 | +7.492553 |
| Left_rear | -33.635651 | +26.109687 | -7.525964 |

起跳前站立力矩约为：

```text
[-7.521191, +7.553089, +7.511047, -7.544554] N*m
```

最终总力矩仍接近站立力矩。这是“WBC没有跳跃运动任务，只是在维持原有接触/姿态任务”的直接证据。

### 2.4 旧Q1-B存在真实异常卸载，但不是普通airborne档先触发

旧Q1-B关键时序：

```text
1.222 s：仍为COMPRESS，双轮几何接触False，gap约+1 mm，法向力为0，估计接触仍True
1.235 s：进入THRUST，腿长约0.160 m，thrust_built=True，但thrust_applied=False
1.236 s：进入FLIGHT，尚未完成有效蹬伸输出
1.237 s：因陈旧估计接触进入LANDING
1.249 s：腿长越出机械下限，进入ABORT
1.254 s：普通监督层才进入airborne，输出置为airborne_zero
```

轮子可以先失去地面支撑，而质心仍继续下降，因此不能要求`com_vz > 0`作为离地必要条件。

---

## 3. 当前WBC模型的实际能力

### 3.1 当前求解链

当前WBC主要处理：

1. 接触约束；
2. 闭链约束；
3. 欠驱动动力学约束；
4. pitch、roll、腿角和其他运动任务；
5. 执行器限幅。

V4移除了`length_R/length_L`，但没有增加质心竖直任务。因此WBC没有被告知“需要下压多少”或“需要产生怎样的质心竖直加速度”。

### 3.2 当前WBC接触力不是独立传感器

当前流程是：

```text
contact_estimator
    -> feedback.decision.contact
    -> contact_blocks
    -> WBC求解
    -> predicted_normal
```

所以WBC接触力不能独立决定接触拓扑，否则会形成循环：

```text
假设接触存在
    -> 解出接触力
    -> 用该接触力证明接触存在
```

WBC接触力适合做接触卸载预警、支撑可行性监控和任务冲突监控，不适合单独做最终离地真值。

### 3.3 external_force的正确使用边界

现有求解器使用：

```text
h_effective = h - external_force
```

这适用于已知外部推力、已知扰动力或不由当前执行器产生的外部广义力。不适用于表达“我希望电机产生一份跳跃力矩，请WBC不要抵消它”。同一组电机产生的跳跃作用，应作为WBC任务目标统一求解。

---

## 4. V5：WBC统一跳跃控制方案

### 4.1 新的控制权分配

```text
JumpPlanner
    -> 相位管理
    -> 生成质心竖直目标
    -> 提供pitch、腿角和安全状态参考

WBC
    -> 接触/闭链/欠驱动约束
    -> 姿态任务
    -> 腿角任务
    -> 质心竖直任务
    -> 统一求解所有执行器总力矩

MPC
    -> 跳跃期间只保留shadow记录
    -> 不向WBC提供跳跃期间的腿长、pitch或加速度参考
```

跳跃期间禁止再次执行：

```text
tau_wbc + tau_jump
leg_length_target -> WBC length PD
```

### 4.2 新增真实质心竖直任务

当前WBC任务接口是加速度任务，建议增加：

```text
J_com_z @ qacc + bias_com_z = desired_com_acceleration_z
```

要求：

- `J_com_z`由整机质量加权质心Jacobian构造；
- `bias_com_z`是对应的`Jdot*qvel`；
- 质心定义与当前`robot_com()`使用的机器人子树一致；
- 不得用base原点、腿长Jacobian或MPC降阶模型替代整机质心行。

如果JumpPlanner以总竖直支撑力为接口：

```text
desired_com_acceleration_z = F_normal_total / robot_mass - g
```

重力只计入一次。WBC根据该任务、接触约束和执行器约束，统一决定左右腿和髋关节总力矩。

### 4.3 压缩阶段不能使用负接触力

压缩的正确物理含义不是给地面一个负法向力，而是降低正向支撑力，让质心在接触约束下向下运动：

```text
下压加速：0 < F_normal_total < robot_mass * g
下压制动：F_normal_total > robot_mass * g
```

不能继续使用`leg_force = -config.leg_force()`作为压缩主控制输入，因为轮地接触不能稳定提供拉力。已有符号测试也显示负力会导致轮隙增加、质心向下加速。

### 4.4 压缩、制动、释放阶段

建议将原COMPRESS细分为：

```text
COMPRESS_ACCEL
    目标：质心向下运动，轮子保持支撑

COMPRESS_BRAKE
    目标：压缩终点前降低腿长下降速度

THRUST_ARMED
    目标：确认仍可接触、参考已写入、WBC可执行

THRUST
    目标：生成正向质心竖直加速度，达到目标离地速度
```

压缩终点只能作为阶段事件，不能直接作为腿长PD目标。

### 4.5 V5输出契约

```text
WBC qacc/task/contact solution
    -> total actuator torque
    -> data.ctrl
```

禁止：

- 再次调用`thrust_torque()`叠加到WBC输出；
- 同一跳跃力同时放进`external_force`和执行器前馈；
- WBC输出后由跳跃模块覆盖髋关节力矩；
- 用非零`tau_jump`作为V5执行确认条件。

---

## 5. WBC参与离地检测的正确方式

### 5.1 三类证据

#### A. WBC预测支撑证据

每个WBC周期记录：

```text
predicted_normal_R/L
wbc_result.feasible
wbc_result.scaling
wbc_result.dynamics_residual_norm
当前接触模型
```

作用是发现支撑力下降、任务与接触约束冲突、WBC无法维持当前接触模型。预测异常只能生成`UNLOAD_CANDIDATE`，不能单独进入FLIGHT。

#### B. 物理步后实际接触力

使用`data.qacc`、`data.ctrl`、实际接触几何块和动力学反解，再经过`off_force/on_force`双阈值和持续时间确认实际支撑力消失或恢复。

#### C. 几何接触证据

左右轮必须独立记录：

```text
contact_geometry_R/L
right_gap/left_gap
right_hub_z/left_hub_z
right_lowest_z/left_lowest_z
```

不能只记录两个gap的最小值。

### 5.2 推荐状态流

```text
GROUND_SUPPORT
    -> UNLOAD_CANDIDATE
    -> LIFTOFF_PENDING
    -> FLIGHT

FLIGHT
    -> TOUCHDOWN_CANDIDATE
    -> LANDING
    -> RECOVER
    -> GROUND_SUPPORT
```

每次状态转移必须检查：

```text
previous_phase合法
and当前条件满足
and必要控制动作已经实际执行
```

### 5.3 离地确认

`UNLOAD_CANDIDATE`可以由WBC预测法向力下降、任务缩放下降或动力学残差增加触发，但此时不能立即关闭WBC。

进入`LIFTOFF_PENDING`后，要求：

```text
当前相位允许离地
WBC已经求解并写入当前跳跃参考
至少完成一个物理步
两轮几何接触持续消失或两轮gap持续为正
实际法向力通过off_force/off_duration
```

`com_vz > 0`不能作为离地必要条件。

进入`FLIGHT`后：

- 不再使用旧双轮接触块建立接地硬约束；
- 不使用陈旧`contact_estimator=True`立即切换LANDING；
- 飞行段输出策略必须独立审批；
- 记录进入FLIGHT的全部原因和证据。

### 5.4 着地确认

`FLIGHT -> TOUCHDOWN_CANDIDATE`可以使用任一轮几何接触恢复；随后要求：

```text
几何接触持续
actual_normal通过on_force/on_duration
```

不能用陈旧接触估计的单帧`any(contact)`直接进入LANDING。

---

## 6. 当前接触检测代码的重点排查项

### 6.1 预测力与实际力分开记录

必须同时给出：

```text
predicted_normal_R/L
normal_forces_R/L
contact_geometry_R/L
contact_estimator_R/L
```

`control.py`中的`predicted_normal`是WBC预测结果；`sampling.py`中的`normal_forces`是物理推进后的实际反解结果，不能混用。

### 6.2 相位提示接线要验证输入契约

当前`loop.py`已经把`THRUST/FLIGHT/LANDING`映射为`JumpPhaseHint`并传入采样器，但接触估计器的THRUST门控还依赖髋角输入。测试agent必须确认：

- `hip_angle`是否实际传入；
- 没有传入时是否导致THRUST期间接触状态被强制保持；
- 该髋角是否代表当前模型真正的释放事件；
- 不得通过硬编码髋角或随意调阈值掩盖接线问题。

### 6.3 释放上限不能直接判飞行

`leg_length >= release_end`只能表示腿部运动达到几何阶段，不能单独证明轮子离地。它最多生成`RELEASE_LIMIT_REACHED`事件，仍需等待几何和实际法向力确认。

---

## 7. 测试agent执行计划

测试agent不得直接部署完整V5跳跃，必须按顺序执行。

### E0：已有证据复核

不重复完整3秒仿真，只核对：

- V4结果身份和工作树；
- `tau_jump`是否进入最终`data.ctrl`；
- WBC任务是否缺少质心竖直目标；
- `predicted_normal`和`normal_forces`是否分开记录。

### E1：质心Jacobian接口自检

仅验证新增质心行的计算，不改变控制参数：

1. 质心位置与`robot_com()`一致；
2. `J_com_z @ qvel`与质心竖直速度一致；
3. `J_com_z @ qacc + bias_com_z`与有限差分加速度一致；
4. 字段、单位、符号和关节维度正确；
5. 同输入重复结果一致。

### E2：静态WBC目标响应

在固定接触、固定初始位形下比较“关闭质心竖直任务”和“开启质心竖直任务”。记录：

- 期望/实际质心加速度；
- qacc；
- 总执行器力矩；
- 逐轮预测法向力；
- 接触残差和动力学残差；
- WBC任务残差、scaling和限幅。

判定重点是新增任务是否真正改变闭环运动目标，而不是某个中间力矩是否变大。

### E3：只测试压缩与制动

先不测试完整飞行和落地，确认：

- 质心是否按目标下降；
- 轮毂是否保持支撑；
- gap是否保持接触范围；
- 腿长是否按目标变化；
- WBC预测支撑力与实际反解法向力是否一致；
- 是否出现前向位移和pitch发散。

### E4：释放交接

仅在E2、E3通过后执行，核对：

- WBC是否生成统一释放总力矩；
- 释放参考是否实际写入`data.ctrl`；
- 释放后质心竖直速度是否符合目标；
- 几何离地、实际法向力和WBC预测支撑状态的先后顺序；
- 是否出现THRUST尚未执行就进入FLIGHT；
- 是否出现FLIGHT 1 ms后因陈旧接触状态进入LANDING。

### E5：着地和MPC恢复

只在释放和飞行证据明确后执行：

- 几何接触恢复；
- 实际法向力通过接地滞回；
- 单轮和双轮接触时序；
- WBC恢复调用；
- MPC重新锚定；
- 是否恢复GROUND而不是卡在airborne_zero或ABORT。

---

## 8. 禁止事项

- 不重新加入`tau_wbc + tau_jump`来弥补WBC任务不足。
- 不用负法向力实现压缩。
- 不用腿长终点PD替代跳跃运动任务。
- 不只用WBC预测接触力判定离地。
- 不只用质心上升判定离地。
- 不用`leg_length >= release_end`直接判飞行。
- 不同时切换WLS/QP、修改接触阈值和修改跳跃目标。
- 不调大力、增益、权重或放宽机械限位来掩盖任务不可行。
- 不修改或回退MPC腿长拟合参数。

---

## 9. 测试结果反馈格式

测试agent每次反馈必须以以下格式追加：

```markdown
# 测试agent：E阶段名称（日期）

## 修改边界

## 自检结果

## 测试配置

## 关键时序

## WBC预测接触力

## 物理步后实际接触力

## 几何接触与质心/轮毂轨迹

## 判定

## 失败证据或通过依据

## 下一步请求
```

结果必须区分：

```text
接口接线正确
任务确实生效
动力学约束可行
控制输出正确写入
压缩功能成功
释放功能成功
完整跳跃成功
```

不能把其中任意一项直接等同于完整跳跃成功。

---

## 10. 当前状态

- 新Markdown文档已重建。
- 本轮未修改控制代码、模型、参数、旧文档或测试结果。
- 当前推荐架构是“JumpPlanner生成质心竖直目标，WBC统一求解总力矩，WBC接触力只作为预测监控，几何和物理步后实际力共同确认离地”。
- 在用户批准V5方案、明确新增质心任务的参考口径和测试边界前，测试agent暂停部署完整跳跃控制重构。
