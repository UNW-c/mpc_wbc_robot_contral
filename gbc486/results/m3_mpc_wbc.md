# M3：6 状态 held-leg MPC + WBC 闭环

脚本：`tools/m3_mpc_wbc.py`（mujoco 3.13.0；circleg HEAD 516380d，`Algorithm/` 未提交改动：无；导入文件 SHA-256 前 12 位见 json 的 `circleg.imported_sha256`；`Algorithm/mpc/controller.py` = 88aae4e059f4）。
L0 = 0.42 m。MPC 50 Hz（circleg `WheelMPC` 原样复用），WBC 250 Hz，物理 1 kHz。增益 / 权重为 circleg 默认值直接搬用，**未调参**。

## 1. 降阶模型（MuJoCo 实测线性化）

状态 [s, v, θ, θ̇, ψ, ψ̇]，输入 [f_R, f_L] (N)。

```
A =
[[ 0.      1.      0.      0.      0.      0.    ]
 [ 0.     -0.     -3.5797  0.     -0.     -0.    ]
 [ 0.      0.      0.      1.      0.      0.    ]
 [-0.      0.     89.9061 -0.      0.     -0.    ]
 [ 0.      0.      0.      0.      0.      1.    ]
 [-0.      0.     -0.0226 -0.      0.     -0.    ]]
B =
[[ 0.      0.    ]
 [ 0.1024  0.102 ]
 [ 0.      0.    ]
 [-0.8763 -0.8759]
 [ 0.      0.    ]
 [ 0.5016 -0.5015]]
```

- 特征值：[9.4819, -9.4819, 0.0001, -0.0001, 0.0, -0.0]；不稳定极点 **9.482 rad/s**
- 质心高（接触点以上）h = 0.2644 m，√(g/h) = 6.091 rad/s（比值 1.56）
- 轮半径 0.077，半轮距 0.2450 m，偏航惯量 0.4885 kg·m²，配平俯仰 2.46e-06 rad

## 2. 闭环结果

推扰：t = 5.0 s 机身世界系速度增量；回位判据 |s| < 10 mm、|θ| < 10 mrad（最后一次超出后的时刻；inf = 到结束仍超出）。

| 后端 | 工况 | 完成 | 停机 | 时长 s | MPC 次数 / fallback / relaxed | WBC fallback / clip / infeasible | 峰值 s mm | 峰值 θ mrad | 峰值 v mm/s | s 回位 s | θ 回位 s | 末 s mm | 末 2 s s 峰峰 mm | max\|τ\| 轮 N·m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nullspace | stand30 | 是 | — | 30.00 | 1495 / 0 / 0 | 0 / 0 / 0 | 23.0 | 2.3 | 9 | inf | 0.00 | -16.22 | 0.03 | 0.01 |
| nullspace | forward | 是 | — | 12.00 | 595 / 0 / 0 | 0 / 0 / 0 | 164.2 | 19.0 | 224 | inf | 2.26 | -53.96 | 36.57 | 0.36 |
| nullspace | backward | 是 | — | 12.00 | 595 / 0 / 0 | 0 / 0 / 0 | 195.5 | 21.9 | 218 | inf | 2.48 | 19.49 | 37.91 | 0.36 |
| nullspace | lateral | 是 | — | 12.00 | 595 / 0 / 0 | 0 / 0 / 0 | 21.4 | 1.9 | 9 | inf | 0.00 | -19.48 | 3.20 | 0.02 |
| wls | stand30 | 是 | — | 30.00 | 1495 / 0 / 0 | 0 / 0 / 0 | 22.9 | 2.3 | 9 | inf | 0.00 | -16.20 | 0.03 | 0.01 |
| wls | forward | 是 | — | 12.00 | 595 / 0 / 0 | 0 / 0 / 0 | 164.2 | 19.0 | 224 | inf | 2.26 | -53.92 | 36.58 | 0.36 |
| wls | backward | 是 | — | 12.00 | 595 / 0 / 0 | 0 / 0 / 0 | 195.5 | 21.9 | 218 | inf | 2.48 | 19.50 | 37.91 | 0.36 |
| wls | lateral | 是 | — | 12.00 | 595 / 0 / 0 | 0 / 0 / 0 | 21.4 | 1.9 | 9 | inf | 0.00 | -19.46 | 3.19 | 0.02 |

| 后端 | 工况 | MPC 耗时 p50/p95 ms | WBC 耗时 p50/p95 ms | 模式拍数 | forward exhausted 拍数 |
|---|---|---|---|---|---|
| nullspace | stand30 | 1.13/1.75 | 1.69/2.84 | {'recovery': 24, 'ground': 7476} | 7476 |
| nullspace | forward | 1.11/1.70 | 1.69/2.85 | {'recovery': 24, 'ground': 2976} | 2976 |
| nullspace | backward | 1.12/1.67 | 1.69/2.90 | {'recovery': 24, 'ground': 2976} | 2976 |
| nullspace | lateral | 1.12/1.66 | 1.68/2.87 | {'recovery': 24, 'ground': 2976} | 2976 |
| wls | stand30 | 0.98/1.48 | 0.75/1.28 | {'recovery': 24, 'ground': 7476} | 0 |
| wls | forward | 1.05/1.60 | 0.75/1.29 | {'recovery': 24, 'ground': 2976} | 0 |
| wls | backward | 1.07/1.72 | 0.75/1.24 | {'recovery': 24, 'ground': 2976} | 0 |
| wls | lateral | 1.01/1.59 | 0.76/1.30 | {'recovery': 24, 'ground': 2976} | 0 |

MPC 机器人相关量（显式覆盖）：{'wheel_torque_limit': 4.0, 'nominal_normal_force': 71.34813000000001, 'yaw_moment_limit': 25.454545454596886, 'scrub_moment_coefficient': 0.0, 'limits': {'commanded_acceleration': 1.2, 'max_acceleration': 1.4, 'max_braking': 1.4}}
