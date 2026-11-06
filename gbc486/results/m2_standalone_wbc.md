# M2：WBC 单独运行（不接 MPC）零速站立

脚本：`tools/m2_standalone_wbc.py`（mujoco 3.13.0）。L0 = 0.42 m，WBC 250 Hz，物理 1 kHz。
增益 / 权重为 circleg 默认值直接搬用（attitude 80/10、height 100/12、regularize 60/6、速度阻尼 14；权重见 `gbc_mpc_wbc/wbc.py`），**未调参**。

| 后端 | 完成 | 停机原因 | 仿真时长 s | max\|θ\| mrad | max\|roll\| mrad | max\|x\| mm | max\|v\| mm/s | 腿长误差 max mm | 腿角误差 max mrad | 末 2 s θ 峰峰 mrad | 末 2 s x 峰峰 mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nullspace | 是 | — | 3.000 | 0.04 | 0.00 | 42.06 | 29.57 | 0.46 | 2.78 | 0.001 | 38.521 |
| wls | 是 | — | 3.000 | 0.04 | 0.00 | 42.22 | 29.68 | 0.46 | 2.78 | 0.002 | 38.666 |

| 后端 | WBC 次数 | fallback | 力矩越限 clamp | infeasible | 求解异常 | 最小预测法向力 N | max 动力学残差 | max\|τ\| [jAB, jAG, jIJ, jIO, 右轮, 左轮] N·m | 关节限位 |
|---|---|---|---|---|---|---|---|---|---|
| nullspace | 750 | 0 | 0 | 0 | 0 | 71.1 | 7.0e-13 | [7.53, 7.56, 7.51, 7.55, 0.02, 0.01] | 无 |
| wls | 750 | 0 | 0 | 0 | 0 | 71.1 | 6.9e-13 | [7.53, 7.56, 7.51, 7.55, 0.02, 0.01] | 无 |

## 任务统计

exhausted = 零空间法中该级已无剩余权限的拍数；scaled = 该级增量被力矩限幅缩小的拍数（零空间法，不含 exhausted 级）。末 2 s 残差 = |J q̈ + bias − 目标| 的最大值 [m/s² 或 rad/s²]。

| 后端 | pitch | roll | yaw | height | forward_velocity | leg_tracking | leg_differential |
|---|---|---|---|---|---|---|---|
| nullspace exhausted | 0 | 0 | 0 | 0 | 726 | 0 | 0 |
| nullspace scaled | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| nullspace 末 2 s 残差 | 6.81e-17 | 1.40e-17 | 3.12e-17 | 3.47e-18 | 1.11e-02 | 5.55e-17 | 4.86e-17 |
| wls exhausted | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| wls scaled | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| wls 末 2 s 残差 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

零空间法首拍各级条件数：pitch 1.0，leg_differential 1.0，roll 1.0
