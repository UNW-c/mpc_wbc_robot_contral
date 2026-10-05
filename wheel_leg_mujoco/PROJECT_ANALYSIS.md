# wheel_leg_mujoco 项目解析

本文档只记录已读到的代码事实和已实际运行的数值核验结果。未运行或未验证的内容会单独标注。
编写过程中没有修改任何代码或参数（遵守工作区 `AGENTS.md` 第 1、2 条）。

## 1. 概览

| 项 | 内容 |
|---|---|
| 定位 | 双轮腿（GBC486）MuJoCo 仿真"空白框架"：模型、传感器读取、VMC 正运动学与雅可比已就绪，控制律未实现 |
| readme 声明环境 | Ubuntu 22.04，Python 3.10，`mujoco==3.4.0`，`pynput` |
| 入口 | `python3 Simulation.py` |
| 代码规模 | Python 6 个文件约 500 行；MJCF 3 个 xml |
| 版本管理 | 处于上层仓库 `circleg-robot` 中，当前为未跟踪（`??`）状态 |
| 副本 | 上层目录的 `wheel_leg_mujoco copy` 与本目录内容完全一致（`diff -rq` 无差异，已排除 `__pycache__`） |

## 2. 目录结构

```
wheel_leg_mujoco/
├── Simulation.py     主循环：分频调度 传感器 → VMC → 力矩下发
├── environment.py    LegWheelRobot：加载模型、启动 viewer、读传感器、写执行器
├── VMC.py            leg_VMC：五连杆正运动学 + 雅可比力矩映射
├── Controller.py     PID 类（当前未被调用）
├── keyboard.py       pynput 键盘指令（主循环读取但未使用）
├── caculation.py     四元数 → ZYX 欧拉角
├── readme.txt
└── MJCF/
    ├── env.xml       场景：仿真选项、地面、天空盒、世界坐标轴，include robot.xml
    ├── robot.xml     机器人本体：刚体树、闭链约束、执行器、传感器
    ├── axis.xml      独立坐标轴模型（未被引用）
    ├── *.STL         15 个连杆网格
    ├── GBC486.SLDASM.urdf / .csv   SolidWorks 导出原始文件（MJCF 未直接使用）
    └── desert.png    天空盒贴图
```

## 3. 机构与 MuJoCo 模型（robot.xml）

### 3.1 刚体树

MuJoCo 只支持树结构，闭链通过"树 + equality connect 约束"表达。每条腿在髋部有两个同轴（仅 y 向错开）的主动关节。

右腿（左腿镜像：IJ↔AB，JM↔BE，MK↔EC，KN↔CF，IO↔AG，OP↔GH）：

```
base (free joint, 初始 z=0.2 m)
├── AG  大腿     jAG  主动  range [-1.57, 0.3]
│   └── GH  小腿     jGH  被动  range [-0.85, 1]
│       └── wheel_right  jwheel_right 主动
└── AB  曲柄     jAB  主动  range [-1.57, 0.5]
    └── BE  连杆     jBE  被动
        └── EC  三角连杆  jEC  被动
            └── CF  连杆   jCF  被动

闭链约束 (equality/connect):
  right_D : EC-D ↔ AG-D     (EC 与大腿在 D 点铰接)
  right_F : CF-F ↔ GH-F     (CF 与小腿在 F 点铰接)
```

### 3.2 实测几何（初始姿态，xz 平面，已运行 `mj_forward` 核验）

| 杆 | 长度 m | 杆 | 长度 m |
|---|---|---|---|
| AB | 0.0980 | DC | 0.0664 |
| BE | 0.1176 | CF | 0.1168 |
| ED | 0.1172 | FG | 0.0663 |
| DA | 0.0982 | GD | 0.1168 |
| AG 大腿 | 0.2150 | GH 小腿（到轮心） | 0.2580 |

ABED 和 DCFG 都是平行四边形（对边长度差都在 1 mm 以内）。E、D、C 在同一刚体 EC 上，所以小腿 GF 的方向和曲柄 AB 刚性联动。
结论：从机构上看，这是"大腿 + 小腿"的串联腿，膝关节由髋部同轴电机经双平行四边形远程驱动。readme 里写的也是"串联腿"。
"异构五连杆"这个说法对应的是 VMC.py 里的等效解算模型（见第 5 节），并不是 MJCF 里的物理拓扑。

### 3.3 质量

| 部件 | 质量 kg |
|---|---|
| base | 8.4 |
| AG / IO 大腿 | 0.667 ×2 |
| GH / OP 小腿 | 1.64 ×2 |
| AB / IJ | 0.202 ×2 |
| BE、CF / JM、KN | 0.057 ×4 |
| EC / MK | 0.128 ×2 |
| 轮 | 0.322 ×2 |
| 整机（运行核验） | 14.546 |

惯量没有显式给出，由 MuJoCo 按网格体积与 `mass` 推算（均匀密度假设）。
`GBC486.SLDASM.csv` 中每个 link 的质量、质心和惯量都与 base_link 完全相同（0.806 kg），说明导出的惯性数据无效，不能作为参考。

### 3.4 碰撞

- `class="touch"` 参与碰撞：base、小腿、BE/EC/CF、轮。
- `class="untouch"` 不参与碰撞：大腿 AG/IO、曲柄 AB/IJ。
- 轮子网格包围盒半径为 0.077 m（运行核验）。

### 3.5 执行器（`<general>`，力矩直驱，`ctrl` 即力矩）

| ctrl 索引 | 名称 | 关节 | gain | 限幅 N·m |
|---|---|---|---|---|
| 0 | Right_front_joint_act | jAB | 1 | ±20 |
| 1 | Right_rear_joint_act | jAG | 1 | ±20 |
| 2 | Left_front_joint_act | jIJ | 1 | ±20 |
| 3 | Left_rear_joint_act | jIO | 1 | ±20 |
| 4 | Right_Wheel_act | jwheel_right | 1 | ±4 |
| 5 | Left_Wheel_act | jwheel_left | **-1** | ±4 |

左轮 gain 为 -1，用来抵消左轮关节轴（+z）和右轮（-z）的镜像差异。

### 3.6 传感器

| 名称 | 类型 | 说明 |
|---|---|---|
| orientation | framequat @ site imu | 世界系四元数 wxyz |
| gyro | gyro @ site imu | 体坐标系角速度 |
| Right/Left_Wheel_pos | jointpos | 轮转角，代码差分得到轮速 |
| Right/Left_front/rear_joint_pos | jointpos | jAB、jAG、jIJ、jIO 关节角 |

### 3.7 仿真选项（env.xml）

`timestep=0.001`，`implicitfast` 积分器，有空气密度与粘度。闭链约束 `solref="0.001 1"`。
运行核验：初始姿态下 connect 约束残差最大 4.5 mm（右 F 点、左 N 点），零力矩落地 3 s 后残差依旧是 4.5 mm，说明 site 装配位置本身就不完全重合。

## 4. 软件架构与运行时序

### 4.1 模块关系

```
Simulation.main()
  ├── LegWheelRobot(env.xml)   environment.py  ── caculation.orientation2euler
  ├── leg_VMC() ×2             VMC.py
  ├── KeyboardController()     keyboard.py (pynput 后台线程)
  └── PID                      Controller.py（已 import，未实例化）
```

### 4.2 主循环分频（计数 i 每步 +1，仿真时间 1 ms/步）

| 周期 | 频率（仿真时间） | 动作 |
|---|---|---|
| t1 = 1 | 1 kHz | `sensor_read_data()` |
| t2 = 4 | 250 Hz | 两腿 `vmc_calc_pos` → 设置 F0/Tp → `vmc_calc_torque` → `actuator_set_torque` |
| t3 = 20 | 50 Hz | 读取键盘指令（目前未使用） |

当前 `F0 = Tp = 0`，轮力矩也是 0。所以运行效果是：机器人在无力矩状态下受重力塌落。运行核验：base 高度 3 s 内从 0.2 m 降到 0.056 m。

### 4.3 传感器预处理（environment.py）

- 欧拉角：四元数 → ZYX（roll, pitch, yaw），奇异时 pitch 钳到 ±π/2。
- 轮速：`(pos - last_pos) × 1000`，单位 rad/s；左轮取负号。
- 关节零位补偿：

| 关节 | 补偿 rad |
|---|---|
| jAB 右前 | +0.027 |
| jAG 右后 | +1.3 |
| jIJ 左前 | +0.003 |
| jIO 左后 | -1.3 |

### 4.4 关节 → VMC 输入映射（Simulation.py）

| 腿 | phi1 | phi4 | pitch / gyro |
|---|---|---|---|
| 右 | jAB + π | jAG | +euler[1] / +gyro[1] |
| 左 | jIO + π | jIJ | -euler[1] / -gyro[1] |

输出力矩回写：`[R.torque_set[1], R.torque_set[0], L.torque_set[0], L.torque_set[1]]` → ctrl[0..3]。
`torque_set[1]` 对应 phi1 关节，`torque_set[0]` 对应 phi4 关节。核对结果是左右两腿的索引对应一致。

## 5. VMC 解算（VMC.py）

### 5.1 等效模型

对称五连杆，两个电机同轴（l5 = 0）：

```
l1 = l4 = 0.215  (等于实测大腿 AG)
l2 = l3 = 0.258  (等于实测小腿 GH)
l5 = 0
```

### 5.2 正运动学 `vmc_calc_pos`

1. `B = l1∠phi1`，`D = l5 + l4∠phi4`
2. 由 `A0 = 2·l2·(XD-XB)`，`B0 = 2·l2·(YD-YB)`，`C0 = l2² + lBD² - l3²` 求
   `phi2 = 2·atan2(B0 + √(A0²+B0²-C0²), A0 + C0)`
3. `phi3 = atan2(YB - YD + l2·sinφ2, XB - XD + l2·cosφ2)`
4. `C = B + l2∠phi2`，得到 `L0 = |C - (l5/2, 0)|` 和 `phi0 = atan2(YC, XC - l5/2)`
5. 状态量：`theta = π/2 - PitchR - phi0`，`d_theta = -GyroR - d_phi0`，其中 `PitchR = -pitch`，`GyroR = -gyro`
6. 用后向差分得到 `d_phi0`、`d_L0`、`dd_L0`、`dd_theta`，dt 默认 0.004，对应 250 Hz。差分没有滤波。

### 5.3 力映射 `vmc_calc_torque`

```
T_phi1 = j11·F0 + j12·Tp      (torque_set[1])
T_phi4 = j21·F0 + j22·Tp      (torque_set[0])

j11 = l1·sin(φ0-φ3)·sin(φ1-φ2) / sin(φ3-φ2)
j12 = l1·cos(φ0-φ3)·sin(φ1-φ2) / (L0·sin(φ3-φ2))
j21 = l4·sin(φ0-φ2)·sin(φ3-φ4) / sin(φ3-φ2)
j22 = l4·cos(φ0-φ2)·sin(φ3-φ4) / (L0·sin(φ3-φ2))
```

F0 是沿腿方向的支撑力，Tp 是绕髋部的摆杆力矩。分母 `sin(φ3-φ2)` 没有做奇异保护。

### 5.4 数值核验：VMC 与真实机构是否一致

方法：在内存中去掉 free joint（车体悬空，不改文件），在两腿主动关节上施加随机力矩，每条腿采样 30 个稳态姿态。用 Simulation.py 的映射计算 VMC 的 L0 和 phi0，再与 MuJoCo 中"髋关节 → 轮心"的真实距离和角度比较。

| 腿 | L0 误差最大值 | phi0 与真实角度的关系 | 关系残差 std |
|---|---|---|---|
| 右 | 4.9 mm | phi0 ≈ π + 0.034 − 真实角 | 0.006 rad |
| 左 | 2.0 mm | phi0 ≈ 真实角 − 0.020 | 0.005 rad |

（真实角的定义：从车体 +x 方向向下量到"髋→轮心"连线的角度。）

结论：在采样到的工作空间内（L0 约 0.29–0.43 m），用串联腿的长度代入 l5 = 0 的五连杆模型，再加上 Simulation.py 中的 +π 和零位映射，可以复现腿长和腿角，误差在毫米级。左右腿的 phi0 互为镜像，Simulation.py 对左腿的 pitch 和 gyro 取负，与此对应。这里只做了数值验证，没有推导等效关系的解析证明。

悬空力映射测试（F0、Tp 各施加 ±20 / ±2，持续 1.2 s）：
- F0 = −20 时两腿同步收腿（L0 从 0.43 降到约 0.14），说明左右两腿 F0 的符号一致，负值表示收腿。
- F0 = +20 时腿已在伸展极限，看不出变化。
- Tp = ±2 两种符号下 phi0 的变化几乎相同，受关节限位影响，Tp 的方向性未能在这个测试中确认。

## 6. 已发现的问题（只列事实，均未修改）

| # | 位置 | 现象 |
|---|---|---|
| 1 | environment.py `d_x` | `wheel_vel` 单位已经是 rad/s，公式却按 rpm 计算：`×2π·r/60`。同时 `wheel_r = 0.77`，而模型轮半径约 0.077 m。两处误差恰好部分抵消：等效系数 0.0806，正确值是 0.077，偏大约 4.7%。 |
| 2 | environment.py `set_joint_positions` | 把关节 id 直接当作 qpos/qvel 索引。free joint 占用了 qpos[0:7]，所以会写到车体位姿上。正确做法是使用 `model.jnt_qposadr` / `jnt_dofadr`。主循环中未调用这个函数。 |
| 3 | keyboard.py `__main__` | `while viewer.is_running:` 少了括号，条件恒为真。另外示例把"前向力/转向"写到了 ctrl[0]/ctrl[1]，实际对应的是右腿两个关节电机，不是轮子。 |
| 4 | Simulation.py | `while True` 不检查 viewer 是否关闭，也没有实时同步（sleep），仿真速度取决于算力。`keyboard` 变量名覆盖了同名模块。 |
| 5 | VMC.py | 雅可比分母 `sin(φ3-φ2)` 和 `L0` 没有奇异保护。各阶导数是无滤波的差分，`dd_*` 噪声会很大。 |
| 6 | Controller.py | PID 积分不乘 dt，没有积分限幅，微分不除 dt。目前未被使用。 |
| 7 | robot.xml | connect 约束 site 初始不重合，残差最大 4.5 mm（见 3.7）。 |
| 8 | 零位补偿 | 右前 +0.027 和左前 +0.003 不对称。右腿 L0 误差（4.9 mm）比左腿（2.0 mm）大，两者是否相关尚未验证。 |
| 9 | env.xml | floor geom 中 `material="plane"condim="3"` 缺少空格（MuJoCo 能解析）。模型名为 `inverted_pendulum`，属于沿用模板。 |
| 10 | GBC486.SLDASM.csv | 所有 link 的惯性数据都是 base_link 的复制（见 3.3）。 |

## 7. 待实现部分

框架已经预留了以下接口，但内容为空：

- 腿长控制：给 F0 赋值（如 PID(L0) + 重力前馈）
- 平衡控制：用 `theta / d_theta / x / d_x / pitch / gyro` 构造 LQR，输出轮力矩和 Tp
- 键盘指令接入：`cmd` 已经在 50 Hz 读取，尚未用于速度或转向参考
- 偏航和横滚控制：需要 `euler[0]/[2]` 和左右腿差动

## 8. 验证范围说明

已验证（本机 Windows，Python 3.13.2，mujoco 3.11.0，仅在内存中运行）：
- 模型可编译：nq=21，nv=20，nu=6，neq=4
- 杆长、质量、约束残差
- VMC 与真实机构的一致性
- F0 的符号

未验证：
- 未运行 `Simulation.py` 本体（需要 GUI viewer；本机未安装 `pynput`）
- 未在 readme 声明的环境（Ubuntu 22.04，Python 3.10，mujoco 3.4.0）下测试
- Tp 的方向性
- 落地状态下的 VMC 力控效果
