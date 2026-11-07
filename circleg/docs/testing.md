# 测试与复现

本文记录轮圈腿四个标准工况的条件、指标、视频和数据文件。所有数据来自本机 MuJoCo、Python 3.13，物理步长 1 ms，WBC 250 Hz，MPC 50 Hz，默认 `hip_input` + `tuned-fast-accel` + QP 后端。

## 1. 工况

| 工况 | 条件 | 目的 |
|---|---|---|
| 前行 | 0→1.5 m/s，保持 3 s，随后停车，总时长 6 s | 速度跟踪、加减速、姿态保持 |
| 原地转圈 | 偏航 1.5 rad/s，保持 4.2 s，总时长 6 s | 偏航跟踪、原地转向 |
| 原地跳跃 | 1.0 s 触发，总时长 5 s | 起跳、飞行、落地、恢复 |
| 前进+跳跃 | 1.0 m/s 指令，1.5 s 起跳，总时长 6 s | 带速度的跳跃 |

## 2. 指标

### 2.1 前行

| 指标 | 值 |
|---|---|
| 最大速度 | 1.513 m/s |
| 最终速度 | 0.000 m/s |
| 里程 | 4.332 m |
| 全程 max\|pitch\| | 0.2627 rad（15.1°） |
| 底座高度范围 | 0.189–0.205 m |

### 2.2 原地转圈

| 指标 | 值 |
|---|---|
| 最大偏航角速度 | 1.519 rad/s |
| 最终偏航角 | −0.118 rad（连续转角约 353°） |
| 纵向位移 | −0.00007 m |
| 全程 max\|pitch\| | 0.0002 rad（0.01°） |
| 底座高度范围 | 0.1927–0.1933 m |

### 2.3 原地跳跃

| 指标 | 值 |
|---|---|
| 底座高度范围 | 0.175–0.585 m |
| 飞行时间 | 1.204→1.732 s，约 0.528 s |
| 相位 | thrust 1.116 / flight 1.204 / landing 1.732 / recover 1.828 / idle 1.836 |
| 全程 max\|pitch\| | 0.6273 rad（35.9°） |
| 触地时 pitch | −0.219 rad |
| 纵向漂移 | 0.198 m |

### 2.4 前进+跳跃

| 指标 | 值 |
|---|---|
| 最大速度 | 1.403 m/s |
| 里程 | 4.605 m |
| 底座高度范围 | 0.176–0.584 m |
| 飞行时间 | 1.724→2.252 s，约 0.528 s |
| 全程 max\|pitch\| | 0.6251 rad（35.8°） |
| 触地时 pitch | −0.218 rad |

## 3. 复现

### 3.1 指标

```powershell
python tools/scenario.py forward --json results/forward.json
python tools/scenario.py spin --json results/spin.json
python tools/scenario.py jump --json results/jump.json
python tools/scenario.py forward_jump --json results/forward_jump.json
```

每个 JSON 包含 Summary 指标和逐拍曲线字段：`time_s`、`pitch_rad`、`speed_m_s`、`yaw_rad`、`yaw_rate_rad_s`、`base_height_m`，以及跳跃相位日志 `jump_log`。

### 3.2 视频

```powershell
python tools/video.py --tag serial_forward --duration 6 --fps 50 --no-jump `
  --command-velocity 1.5 --command-hold 3

python tools/video.py --tag serial_spin --duration 6 --fps 50 --no-jump `
  --command-yaw-rate 1.5 --command-yaw-hold 4.2

python tools/video.py --tag serial_jump --duration 5 --fps 50 --jump-time 1.0

python tools/video.py --tag serial_forward_jump --duration 6 --fps 50 --jump-time 1.5 `
  --command-velocity 1.0 --command-hold 4.5
```

视频参数：640×480、50 fps、跟踪 `base_link`、方位角 140°、距离 1.6 m、俯仰 −12°。输出在 `videos/`。

视频工具通过 `MuJoCoMpcWbcSimulation.run(frame_callback=..., frame_interval=...)` 按物理步取帧，时间轴与仿真时间一致。

## 4. 数据文件

| 文件 | 内容 |
|---|---|
| `results/forward.json` | 前行 6 s |
| `results/spin.json` | 原地转圈 6 s |
| `results/jump.json` | 原地跳跃 5 s |
| `results/forward_jump.json` | 前进+跳跃 6 s |

`results/` 只保留这四个标准工况的指标与曲线，不存整段仿真记录。

## 5. 测试套件

```powershell
python -m pytest -q
```

当前有 4 个既有失败：

| 测试 | 现象 |
|---|---|
| `test_control_regressions::test_multirate_loop_recomputes_wbc_five_times_per_mpc_tick` | 非法 `wbc_dt/physics_dt` 不再抛 `ValueError` |
| `test_jerk_ramp::test_soft_reanchor_keeps_velocity_acceleration_and_command_flag` | `soft_reanchor` 不再把位置改到 5.0 |
| `test_velocity_joint::test_velocity_release_and_reverse[False/True]` | 误差单调性断言 `≤1e-9` 过严，实测回升 `2.3e-5` |

这 4 项在迁移前后一致，不是目录调整引入的。修改断言等于改变评价口径，未自行处理。

## 6. 读数说明

- `max_abs_pitch` 取机身俯仰角绝对值的全程最大值，包含起跳前的预倾和落地后的恢复。
- 转圈的连续转角由最终偏航角换算：偏航角在 ±180° 回绕，`2π − 0.118 rad ≈ 353°`。
- 跳跃的飞行时间取相位日志 `flight` 到 `landing` 的时间差。
- 底座高度是 `qpos[2]`，不是轮心高度；两者相差轮半径。
- 轮圈腿的跳跃仍是早期单质量/腿长方案，姿态指标（约 36°）明显弱于串腿的双质量块方案（约 12°）。
