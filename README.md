# CirclegRobot

CirclegRobot 是一个基于 MuJoCo 的轮圈腿式平衡机器人仿真与控制项目。
项目使用统一的轮毂 LQR 控制器，同时支持固定腿长、腿长调节、线速度、偏航角速度以及 Xbox/SDL 兼容手柄输入。

## 环境安装

项目推荐使用 Conda 环境。以下命令可在 Windows PowerShell 或 Linux Shell 中执行：

```powershell
conda activate env_mujoco
python -m pip install -r requirements.txt
```

如果环境尚未创建，可以先创建 Python 3.13.11 环境：

```powershell
conda create -n env_mujoco python=3.13.11
conda activate env_mujoco
python -m pip install -r requirements.txt
```

键盘输入使用 `pynput`。手柄在 Linux 使用 SDL2/Pygame 标准控制器映射，在 Windows 优先使用 SDL2，未安装 SDL2 时自动回退到系统 XInput。Linux 的交互控制需要图形桌面会话；在 Wayland 下需启用兼容的 X11/XWayland 输入后端。

## 运行

运行 rough 地形交互仿真：

```powershell
python main.py --terrain rough
```

运行平地仿真：

```powershell
python main.py --terrain plane
```

无图形界面的控制器测试：

```powershell
python main.py --terrain rough --headless --duration 10
```

## 控制方式

### 键盘

| 按键 | 功能 |
| --- | --- |
| `↑` / `↓` | 前进 / 后退，最大速度为 `2.5 m/s` |
| `←` / `→` | 偏航角速度控制 |
| `Q` / `E` | 缩短 / 加长腿部 |

### Xbox/SDL 兼容手柄

| 输入 | 功能 |
| --- | --- |
| 左摇杆上下 | 连续控制前进/后退速度，最大速度为 `2.5 m/s` |
| 右摇杆左右 | 连续控制偏航角速度 |
| 左扳机 / 右扳机 | 缩短 / 加长腿部 |
| `B` 键 | 恢复配置中的默认腿部姿态 |

连接手柄时优先使用手柄；手柄未连接时自动回退到键盘控制。

## 项目结构

```text
Algorithm/                  # LQR、状态估计、几何和控制算法
APP/                        # MuJoCo 仿真、输入和控制器应用层
URDF_Model/mjcf/            # MuJoCo 模型文件
main.py                     # 命令行入口
requirements.txt            # Python 运行依赖
```

控制器状态采用 `[x, pitch, v, pitch_rate]`，左右轮使用同一套 LQR 设计和权重。
默认轮毂扭矩上限为 `5 N·m`，速度参考变化率等控制参数见 `APP/config.py`。

## 模型说明

`URDF_Model/mjcf/URDF_Model_rough.xml` 中包含 rough 高度场和后方测试台阶。
台阶测试时，机器人初始位置约为 `x=-1.0 m`，台阶位于机器人后方，应使用后退指令进行测试。

## 开源说明

本仓库用于 CirclegRobot 项目的开源协作。提交代码或修改模型时，请保持左右轮轴方向、腿长机构定义和模型坐标约定一致，并在提交前完成至少一次平地和 rough 地形仿真测试。
