"""方案 §5.3 的统一增益排程表——单一真值来源，禁止各任务里再硬编码。

本机器人几乎所有奇异性和耦合都挂在髋角 ``q`` 上，核心变量是 ``p = sin q``。
同一条表里 ``p`` 和 ``1/p`` 会同时出现，这不是矛盾，而是 §1.4 那个"互为倒数的
双重奇异"的直接体现：高度权限 ``dz/dq = 0.06p → 0``，而髋到法向力的增益
``∝ 1/p`` 在死点发散。

| 对象 | 排程 | 依据 |
|---|---|---|
| 高度任务阻尼 | ``1/(p² + ε)``，ε = 0.01 | ``dz/dq → 0``（§1.4） |
| 高度任务 activation | ``smoothstep · p²`` | 死点处该级失效且 ``N_i`` 不再收缩（§4.4） |
| 横滚回路增益 | ``p`` | 髋→法向力增益 ``∝ 1/p``，行程上变化 11.5 倍（§1.4） |
| 偏航权限上限 | ``7.5 − 5.59p`` | scrub 库仑力矩吃掉偏航权限（§1.2） |
| 偏航前馈 | ``0.5·cos q·q̇`` | 升降腿的寄生偏航（§1.3） |
| 俯仰/横滚解耦阻尼 | 按 ``β = atan(0.5p)`` | 中行程俯仰与横滚不再独立（§1.1） |

本模块只依赖 numpy，不依赖 MuJoCo。

关于 §5.3 表里"俯仰/横滚解耦阻尼：按 ``β``"这行：文档只给了排程量、没给映射。
这里取 ``damping = nominal · sin β``，理由是耦合强度随 ``β`` 单调增大（死点处
``β = 0``，不可控方向正好是俯仰轴本身，两者反而"解耦"），而且 ``sin β`` 与
:mod:`Algorithm.mpc.contact` 暴露的 ``pitch_controllability`` 是同一个量，
两边不会各算一套。若标定时取别的映射，只改这里一处。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 偏航权限三件套（方案 §1.2）：满权限 7.5 N·m，中行程 scrub 吃掉 5.59 N·m
DEFAULT_YAW_MOMENT_LIMIT = 7.5
DEFAULT_SCRUB_MOMENT_COEFFICIENT = 5.59
DEFAULT_YAW_MOMENT_FLOOR = 0.5
# 高度任务的奇异阻尼
DEFAULT_HEIGHT_DAMPING_EPSILON = 0.01
# 高度任务淡入淡出的髋角窗口（度）
DEFAULT_HEIGHT_FADE_IN = 3.0
DEFAULT_HEIGHT_FADE_OUT = 8.0


def smoothstep(value: float, edge_low: float, edge_high: float) -> float:
    """标准 smoothstep：区间外为 0/1，区间内 C¹ 连续。

    用在任务淡入淡出上（§4.4）：硬开关会让层级结构突变，力矩跳变。
    """
    low = float(edge_low)
    high = float(edge_high)
    if high <= low:
        raise ValueError("smoothstep 的上边界必须大于下边界")
    x = (float(value) - low) / (high - low)
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class ScheduleParameters:
    """排程表里的全部常数。默认值来自 docs/MPC-WBC方案.md §1.2 / §1.4 / §5.3。"""

    height_damping_epsilon: float = DEFAULT_HEIGHT_DAMPING_EPSILON
    height_fade_in_degrees: float = DEFAULT_HEIGHT_FADE_IN
    height_fade_out_degrees: float = DEFAULT_HEIGHT_FADE_OUT
    yaw_moment_limit: float = DEFAULT_YAW_MOMENT_LIMIT
    scrub_moment_coefficient: float = DEFAULT_SCRUB_MOMENT_COEFFICIENT
    yaw_moment_floor: float = DEFAULT_YAW_MOMENT_FLOOR
    decoupling_angle_gain: float = 0.5  # atan(0.5·sin q)


def dead_point_factor(hip_angle: float) -> float:
    """``p = |sin q|``：0 表示在死点，1 表示在全行程中段。

    ``hip_angle`` 是**单腿曲柄角**。两腿共模/差模混合时，权限因子应取
    ``mean(|sin q1|, |sin q3|)`` 并用 :func:`roll_gain_scale_from_factor` /
    :func:`height_task_activation_from_factor`（#13）。
    """
    return abs(math.sin(float(hip_angle)))


def roll_gain_scale_from_factor(permission: float) -> float:
    """按单腿权限因子 ``permission = mean(|sin q_i|)`` 给出横滚回路增益（#13）。"""
    return float(min(max(float(permission), 0.0), 1.0))


def height_task_damping(
    hip_angle: float,
    parameters: ScheduleParameters | None = None,
) -> float:
    """高度任务的奇异阻尼因子 ``1/(p² + ε)``。

    死点处 ``dz/dq → 0``，逆解要求 ``q̈ → ∞``，所以要按这个因子加阻尼。返回值是
    **相对因子**（p=1 时约 1，p=0 时到 100），调用方乘自己的标称阻尼即可。
    """
    params = parameters or ScheduleParameters()
    p = dead_point_factor(hip_angle)
    return 1.0 / (p * p + params.height_damping_epsilon)


def height_task_activation(
    hip_angle: float,
    parameters: ScheduleParameters | None = None,
) -> float:
    """高度任务的激活因子，取值 ``[0, 1]``（单腿曲柄角口径）。

    两个因素相乘：``p²`` 是权重量级（死点处该级本就无效），``smoothstep`` 负责在
    ``[3°, 8°]`` 上淡入淡出，避免层级结构突变（§4.4）。
    两腿混合摆动时用 :func:`height_task_activation_from_factor`（#13）。
    """
    return height_task_activation_from_factor(dead_point_factor(hip_angle), parameters)


def height_task_activation_from_factor(
    permission: float,
    parameters: ScheduleParameters | None = None,
) -> float:
    """按单腿权限因子 ``permission = mean(|sin q_i|)`` 计算高度任务激活因子（#13）。"""
    params = parameters or ScheduleParameters()
    p = min(max(float(permission), 0.0), 1.0)
    fade = smoothstep(
        math.degrees(math.asin(min(p, 1.0))),
        params.height_fade_in_degrees,
        params.height_fade_out_degrees,
    )
    return fade * p * p


def roll_gain_scale(hip_angle: float) -> float:
    """横滚回路增益的排程因子 ``p = |sin q|``（单腿曲柄角口径）。

    髋到法向力的增益 ``∝ 1/p``，在 ``[5°, 90°]`` 上变化 11.5 倍（§1.4），所以
    横滚回路的增益要反着按 ``p`` 排程，否则死点附近噪声被放大 11 倍。
    两腿混合摆动时用 :func:`roll_gain_scale_from_factor`（#13）。
    """
    return roll_gain_scale_from_factor(dead_point_factor(hip_angle))


def scrub_moment(hip_angle: float, parameters: ScheduleParameters | None = None) -> float:
    """两接触点前后错开造成的库仑 scrub 阻力矩 ``5.59·sin q`` N·m（§1.2）。"""
    params = parameters or ScheduleParameters()
    return params.scrub_moment_coefficient * dead_point_factor(hip_angle)


def yaw_moment_budget(
    hip_angle: float,
    parameters: ScheduleParameters | None = None,
) -> float:
    """扣掉 scrub 之后可用的偏航力矩上限 ``max(7.5 − 5.59p, floor)`` N·m。

    中行程时 scrub 吃掉 74.5%，只剩 1.91 N·m；死点处回到满值（§1.2）。
    """
    params = parameters or ScheduleParameters()
    budget = params.yaw_moment_limit - scrub_moment(hip_angle, params)
    return max(budget, params.yaw_moment_floor)


def yaw_rate_feedforward(
    hip_angle: float,
    hip_rate: float,
    parameters: ScheduleParameters | None = None,
) -> float:
    """升降腿引起的寄生偏航角速度前馈 ``0.5·cos q·q̇``（§1.3）。

    髋转动时两轮心在机身系里以 ``∓0.06cos q·q̇`` 反向平移，纵向不滑移就要求机身
    转动。半行程升降会留下 ``0.5 rad = 28.6°`` 的净偏航，**必须前馈**，只做反馈
    一定跟不上。
    """
    del parameters  # 该式没有可调参数
    return 0.5 * math.cos(float(hip_angle)) * float(hip_rate)


def decoupling_angle(hip_angle: float, parameters: ScheduleParameters | None = None) -> float:
    """``β = atan(0.5·sin q)``：接触连线与横向轴的夹角（§1.1）。

    与 :class:`Algorithm.mpc.contact.ContactAxis` 的 ``tilt`` 是同一个量，平地上
    两处必须一致（有测试守着）。
    """
    params = parameters or ScheduleParameters()
    return math.atan(params.decoupling_angle_gain * dead_point_factor(hip_angle))


def decoupling_damping(
    hip_angle: float,
    nominal: float,
    parameters: ScheduleParameters | None = None,
) -> float:
    """俯仰/横滚解耦阻尼 ``nominal · sin β``。

    中行程 ``β`` 最大、两轴耦合最强，阻尼也最大；死点处 ``β = 0``，不可控方向
    正好就是俯仰轴本身，反而无需解耦（§1.1 / §5.3）。
    """
    return float(nominal) * math.sin(decoupling_angle(hip_angle, parameters))
