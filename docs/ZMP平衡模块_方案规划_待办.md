# ZMP 平衡模块 方案规划（待办）

状态：**待办，未实施**。本文只规划，未改任何代码。
日期：2026-10-16
发起人：用户（2026-10-16）。原话要点：ZMP 目前只用于前进移动上，或许可以解耦成单独模块，
凡是和平衡相关的，都可以接上 ZMP 点的规划。

---

## 0. 摘要

现在 ZMP 不是模块，是 `Algorithm/mpc/controller.py` 里的一个**参考整形函数**
（`_apply_zmp_reference`），输入是 MPC 参考加速度，输出写俯仰参考与曲柄参考。
它挂在 MPC 时域上，所以只在有加速度指令的前进行驶里生效，跳跃路径（FLIGHT/LANDING
没有 MPC 轨迹）根本调不到。

规划方向：把**物理内核**抽成纯函数模块，产物是"ZMP 参考 + 可行域 + 残差 + 所需角动量率"，
不 import MPC、不认模式，让 MPC / WBC / 落地控制器都能调。

**不能借的是"拆成俯仰+曲柄"那一段**：它在轮腿上已被写死成 `zmp_split = 1.0`
（俯仰参考严格为 0），而落地恰恰要用俯仰通道。

**它不增加权限**：可行域仍由摩擦（`mu*z_com`）和力矩上限决定。它的价值是
"提前知道不可行"和"给腿摆与轮一个共同目标"，不是把 44 度落地变成可行。

---

## 1. 现状（证据）

| 项 | 位置 | 事实 |
|---|---|---|
| 配置字段 | `Algorithm/mpc/controller.py:88-99` | `zmp_reference / zmp_split / zmp_lead / zmp_com_height / zmp_crank_radius / zmp_crank_rate_limit / zmp_crank_limit`，全部默认关闭 |
| 唯一实现 | `Algorithm/mpc/controller.py:379` `_apply_zmp_reference` | `delta = z_com * a / 9.81`；按 `split` 拆成 `tilt` 与 `crank`；写 `reference_states[:,2]` 与 `[:,6]` |
| 调用点 | `Algorithm/mpc/controller.py:689` | `if self.settings.zmp_reference:` |
| 轮腿侧覆盖 | `wheel_leg_mujoco/gbc_mpc_wbc/simulation.py:165-195` | `zmp_com_height` 用实测质心高；`zmp_crank_radius` 用实测 `leg_offset_radius`（注释强调不能用 L0=0.28）；**`zmp_split` 强制 1.0，"质心偏移全部由摆腿承担，俯仰参考严格为 0"** |
| 6 状态自动关 | `simulation.py` 末段、`objective.py:175` | 写 `[:,6]/[:,7]` 需要 8 状态模型的腿角列，6 状态模型自动关闭 |
| 生效条件 | `objective.py:453` | 只在有加速度指令时起作用；现有受扰工况全是零指令，所以看不到 |

结论：它**不是**平衡判据，也**不是**独立模块，是一个绑在 MPC 上、只服务于行驶加速度的
参考生成器。

---

## 2. 物理口径（2026-10-16 更正）

根据上一轮讨论中更正过的推导。2D 矢状面，质心在 `(x_c, z)`，接触点在 `(x_p, 0)`，
地面反力 `(F_x, F_z)`。

绕质心的力矩（绕 y 轴）：

```text
M_y = r_z*F_x - r_x*F_z = (-z)*F_x - (x_p - x_c)*F_z
```

静止（`theta_dd = 0`）令 `M_y = 0`：

```text
x_zmp = x_p + z*(F_x/F_z)  =  x_c
```

含角动量项的一般式：

```text
x_zmp = x_c - (z/g)*a_x - Hdot_c/(m*g)
```

要点：

1. **ZMP 不等于接触点。** 它相对接触点偏移 `z*(F_x/F_z)`；静止时落在质心竖直线上。
   （把角动量项也塞进同一定义时 ZMP 才会恒等于接触点，那种口径下判据退化——上一轮我用错了口径。）
2. **带角动量项时 ZMP 会离开质心竖直线**，偏移量由 `Hdot_c` 决定。这套机构里能修 `Hdot_c`
   的执行器只有两个：腿摆（phi 通道）和轮子。
3. **摩擦锥给出可行域**：`|F_x| <= mu*F_z`，即 `|x_zmp - x_p| <= mu*z`。
   点接触的"支撑多边形"退化成一个点，但**摩擦把可行域重新撑成接触点两侧 `±mu*z_com` 的线段**。
   这是本构型下 ZMP 有意义、而且能当判据用的真正来源。
4. 角动量通道能额外挪的量级是 `tau/(m*g)`，**不是** `m*g*delta`（上一轮我写错过一次，已更正）。

### 本机实参下的数字

`mu = 1.0`（`robot.xml` 未写 friction，取 MuJoCo 默认）、`z_com = 0.2643 m`（`standing_pose(L=0.28)` 实算）、
`m = 14.546 kg`、`m*g = 142.70 N`、`r = 0.077 m`、轮力矩合计 `tau_max = 12 N*m`。

| 落地倾角 | ZMP 相对接触点偏移 `z*tan(theta)` | 占 `mu*z_com`(=0.2643 m) | 需要的轮力矩 `m*g*tan(theta)*r` |
|---|---|---|---|
| 0.30 rad (17 deg) | 0.082 m | 31% | 3.40 N*m |
| 0.50 rad (29 deg) | 0.144 m | 55% | 6.00 N*m |
| 0.764 rad (44 deg) | 0.253 m | **96%** | 10.53 N*m |

轮力矩那列到 44 度也才 10.53 N*m（未顶 12），**所以卡住的是摩擦（96%），不是电机**。
角动量通道能额外挪 `12/142.7 = 0.084 m`，相对 44 度需要的 0.253 m 只够三成。

对照轮圈腿（`URDF_Model/mjcf/URDF_Model_plane.xml`）：`R = 0.16 m`、`tau` 合计 `10 N*m`、
`m*g = 93.2 N`，`tau/R = 62.5 N`，电机先卡住，倾角上限 33.9 度。

---

## 3. 模块接口（待评审）

纯函数，不 import MPC，不认跳跃/行走模式。

```text
输入:
  com_state      (x, z, dx, dz)                 质心位置与速度
  contacts       接触点序列（本构型只有轮子，退化成一个点）
  model_params   (mu, z_com, r, m, tau_max)     由调用方注入，不硬编码

输出:
  1) x_zmp_ref(t)                               参考 ZMP 轨迹（可选，None 表示只做检查）
  2) feasible      |x_zmp - x_contact| <= mu*z_com  且  |tau| <= tau_max
  3) residual      x_zmp_required - x_zmp_feasible        [m]
  4) Hdot_need     m*g*residual                           [N*m]
```

第 4 项是关键产物：它把"还差多少"直接翻译成 phi 通道与轮子各自要出多少角动量。
现在这两者靠一条经验交换线手工权衡，`Hdot_need` 是它的解析形式。

---

## 4. 实参来源（全部复用已标定值，不新造）

| 参数 | 取值 | 来源 |
|---|---|---|
| `mu` | 1.0 | `robot.xml`/`env.xml` 均未写 `friction`，MuJoCo 默认滑动摩擦 |
| `z_com` | 0.2643 m（L=0.28） | `robot.standing_pose` 实算，随腿长变化 |
| `m` | 14.546 kg | MJCF 汇总 |
| `r` | 0.077 m | 轮碰撞体 geom size |
| `tau_max` | 6 N*m（每轮） | MJCF `ctrlrange` |
| 曲柄/腿角等效半径 | 0.2133 m | 已标定 `leg_offset_radius`（**不能用 L0=0.28**） |

---

## 5. 接入点

| 接入点 | 现状 | 接上 ZMP 后 |
|---|---|---|
| MPC 参考整形 | `_apply_zmp_reference`，内部自己算 `delta = z*a/g` | 换成模块调用，`delta` 由模块给；`split` 拆法保留在调用方（构型相关） |
| WBC 的 pitch / angle_R / angle_L 任务 | 各自追自己的参考（pitch 追 0，腿角追世界系角），互相抢执行器（FL39 就是这么 ABORT 的） | 三者服务于同一个 `x_zmp_ref`，分工由调用方显式给 |
| LANDING / RECOVER | 现在完全没有 ZMP，落地压缩段由 `landing_buffer` 覆盖 | 用 `x_zmp_ref` 与 `residual` 做落地可行性闸门与参考 |

第三处是本次讨论的直接动机：FLIGHT/LANDING 没有 MPC 轨迹，所以现有实现根本够不到。

---

## 6. 与 phi/pitch 交换线的关系

实测拟合的交换线 `pitch_landing ~= -0.83 + 1.21*phi_world`（FL44/48/53/54/55 五点，残差 < 0.09 rad）
本质是"`Hdot_c` 被离地瞬间定死，phi 与 pitch 只能在这份角动量内分配"。
接上 ZMP 后这条线就是 `Hdot_need` 的展开形式，不再是经验常数。

---

## 7. 不做什么

- 不增加权限。可行域仍由 `mu*z_com` 与 `tau_max` 决定。
- 不照搬 `zmp_split` 拆法当通用接口。
- 不动现有 `Algorithm/mpc/controller.py` 的默认行为（默认关闭）。
- 不在模块里硬编码任何本机常数。

---

## 8. 前置条件与排期

前置条件：

1. 先完成 LAND0 空白对照（`docs/agent交互文档_WBC跳跃_测试方案部署.txt` 的 LAND0 段），
   把"落地收敛是谁带来的"定下来。
2. 飞行段轮间隙 0.1886 m 低于 200 mm 硬下限的处理（同上文档的增补段，三个杠杆待指定）。

排期：按用户 2026-10-14 定的顺序（先恢复完整跳跃 -> 再 phi -> 再 pitch），
本模块属于第 2/3 步的工具，不打断当前收尾。

未决问题（需要用户拍板）：

1. 模块放哪个包（`Algorithm/` 共享，还是 `wheel_leg_mujoco/gbc_mpc_wbc/` 内）。
2. `landing_buffer` 与 WBC 的接管顺序要不要一起改（现在 `landing_buffer` 整条覆盖，
   LAND1A 已改成只覆写腿通道）。
3. 是否先用离线数据（现有 `results/*.json.gz`）把 `residual` 曲线算一遍验证口径，
   再谈接入。
