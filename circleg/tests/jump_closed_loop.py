"""T-J3～T-J5（第 17 次排查）：J-M 离线模型上的完整跳跃闭环。

控制栈（每 4 ms 调用一次，与 WBC 同频）：
- 接触：``GroundContactEstimator``，输入 J-M 的法向力（合并腿按两轮平分）、机身竖直比力；
- 跳跃：``JumpController``；
- 上层平衡的替身（J-M 没有 MPC / WBC）：轮子 PD 把机身俯仰稳定到参考值（IDLE / PREPARE / RECOVER 使用），
  髋在 IDLE 时 PD 保持站姿。它只用于提供 PREPARE 的后仰姿态和落地后的恢复，不代表实际平衡控制器的性能。

输出一次完整跳跃的关键指标。用法：
    .venv/Scripts/python.exe tests/jump_closed_loop.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Algorithm.contact_state import ContactEvent, GroundContactEstimator, JumpPhaseHint  # noqa: E402
from Algorithm.jump import JumpController, JumpObservation, JumpParams, JumpPhase  # noqa: E402
from tests.jump_model import JumpModelParams, PlanarJumpModel  # noqa: E402

_CONTROL_DT = 0.004
_SIM_DT = 2e-4


@dataclass
class BalanceStandIn:
    """上层平衡替身：轮子（两轮合计）力矩 = −(kp·(θ_ref − θ) + kd·(−θ̇))·sign，髋 PD 保持。

    J-M 中轮子力矩为正时反作用使机身后仰（θ 减小），因此"要让 θ 增大"需要负的轮力矩。
    这组增益只为让 PREPARE 能把机身稳定在后仰姿态，不参与性能结论。
    """

    kp: float = 60.0
    kd: float = 6.0
    kx: float = 4.0            # 位置 / 速度回拉（防止为保持后仰而一直加速）
    kv: float = 6.0
    hip_kp: float = 60.0
    hip_kd: float = 3.0

    def wheel(self, theta_ref, theta, theta_rate, x, v) -> float:
        # 倒立摆式：要维持后仰，轮子需要把接触点移到质心下方 → 由 θ 误差与位置/速度共同决定
        return float(-(self.kp * (theta_ref - theta) - self.kd * theta_rate) + self.kx * x + self.kv * v)

    def hip(self, target, q, qd) -> float:
        return float(self.hip_kp * (target - q) - self.hip_kd * qd)


def run_jump(jump_params: JumpParams | None = None, model_params: JumpModelParams | None = None,
             request_time: float = 0.3, duration: float = 3.0, verbose: bool = False) -> dict:
    model = PlanarJumpModel(model_params)
    jump = JumpController(jump_params)
    contact = GroundContactEstimator()
    balance = BalanceStandIn()
    state = model.standing_state(0.0)
    p = model.p
    total_mass = p.body_mass + p.crank_mass + p.wheel_mass
    ground_com = None
    t = 0.0
    next_control = 0.0
    u = np.zeros(2)
    normal = total_mass * 9.81
    last_vz = 0.0
    log = []
    liftoff = touchdown = None
    apex_com = -np.inf
    apex_clear = -np.inf
    requested = False
    wheel_speed_td = None
    pitch_td = None
    hip_peak_landing = 0.0
    normal_peak_landing = 0.0
    x_request = None
    while t < duration:
        if t + 1e-12 >= next_control:
            g, gd = state.g, state.gd
            vz = gd[1]
            specific = (vz - last_vz) / _CONTROL_DT + 9.81 if t > 0 else 9.81
            last_vz = vz
            hint = {JumpPhase.THRUST: JumpPhaseHint.THRUST, JumpPhase.FLIGHT: JumpPhaseHint.FLIGHT,
                    JumpPhase.LANDING: JumpPhaseHint.LANDING}.get(jump.phase, JumpPhaseHint.NONE)
            out = contact.update((0.5 * normal, 0.5 * normal), _CONTROL_DT, phase=hint,
                                 hip_angle=g[3], specific_force_z=specific)
            wheel_omega = gd[2] + gd[3] + gd[4]
            obs = JumpObservation(time=t, hip_angle=g[3], hip_rate=gd[3], wheel_speed=wheel_omega,
                                  body_forward_speed=gd[0], body_vertical_speed=vz, pitch=g[2],
                                  pitch_rate=gd[2], contact=out.contact, events=tuple(out.events),
                                  wheel_radius=p.wheel_radius)
            if not requested and t >= request_time:
                jump.request()
                requested = True
                ground_com = _com(model, g)[1]
            phase_before = jump.phase
            cmd = jump.step(obs, _CONTROL_DT)
            if jump.phase == JumpPhase.FLIGHT and phase_before != JumpPhase.FLIGHT:
                contact.set_predicted_flight(jump.predicted_flight)
                liftoff = dict(t=t, vz=vz, pitch=np.degrees(g[2]), pitch_rate=np.degrees(gd[2]), hip=g[3])
            if jump.phase == JumpPhase.LANDING and phase_before == JumpPhase.FLIGHT:
                touchdown = dict(t=t, pitch=np.degrees(g[2]), pitch_rate=np.degrees(gd[2]), hip=g[3],
                                 wheel_mismatch=abs(wheel_omega * p.wheel_radius + 0.0 - gd[0]),
                                 vz=vz)
            # 组合输出（合并腿：髋力矩 ×2，轮力矩 ×2）
            if cmd.hip_torque is not None:
                hip_total = 2.0 * cmd.hip_torque
            else:
                # 替身在 IDLE 时保持当前站姿：跳跃前为 0（低位），着地后为 land_hip（高位）
                target = 0.0
                hip_total = 2.0 * float(np.clip(balance.hip(target, g[3], gd[3]), -10, 10))
            if cmd.wheel_torque is not None:
                wheel_total = 2.0 * cmd.wheel_torque
            else:
                ref = cmd.pitch_reference if cmd.pitch_reference is not None else 0.0
                wheel_total = float(np.clip(balance.wheel(ref, g[2], gd[2], g[0], gd[0]), -10, 10))
            u = np.array([hip_total, wheel_total])
            if jump.phase in (JumpPhase.LANDING,):
                hip_peak_landing = max(hip_peak_landing, abs(g[3]))
            if requested and x_request is None:
                x_request = g[0]
            if verbose and int(round(t / _CONTROL_DT)) % 10 == 0:
                print(f"t={t:.3f} {jump.phase.value:8s} q={g[3]:.2f} θ={np.degrees(g[2]):+6.1f} "
                      f"z={g[1]:.3f} vz={vz:+.2f} N={normal:6.1f} contact={out.contact}")
            log.append((t, jump.phase.value, g.copy(), gd.copy(), normal))
            next_control += _CONTROL_DT
        state, normal, _ = model.step(state, u, _SIM_DT)
        t = state.t
        if touchdown is not None and jump.phase in (JumpPhase.LANDING, JumpPhase.RECOVER):
            normal_peak_landing = max(normal_peak_landing, normal)
        if liftoff is not None and touchdown is None:
            apex_com = max(apex_com, _com(model, state.g)[1])
            apex_clear = max(apex_clear, model.wheel_bottom(state))
        if abs(state.g[2]) > 1.6 or not np.all(np.isfinite(state.g)):
            break
        if jump.phase == JumpPhase.IDLE and requested and touchdown is not None:
            settle_until = settle_until if 'settle_until' in dir() else t + 1.0
            if t >= settle_until:
                break
    final = state
    return dict(
        phase_log=jump.log,
        liftoff=liftoff, touchdown=touchdown,
        com_rise=(apex_com - ground_com) if (ground_com is not None and liftoff) else float("nan"),
        wheel_clearance=apex_clear if liftoff else float("nan"),
        final_phase=jump.phase.value, final_pitch=np.degrees(final.g[2]), final_hip=final.g[3],
        final_speed=final.gd[0], end_time=final.t, hip_peak_landing=hip_peak_landing,
        max_abs_pitch=max(np.degrees(abs(entry[2][2])) for entry in log),
        normal_peak_landing=normal_peak_landing,
        drift=(final.g[0] - x_request) if x_request is not None else float("nan"),
    )


def _com(model, g):
    b, _, c, w = model.points(g)
    p = model.p
    return (p.body_mass * b + p.crank_mass * c + p.wheel_mass * w) / (p.body_mass + p.crank_mass + p.wheel_mass)


def _report(name, r):
    lo, td = r["liftoff"], r["touchdown"]
    print(f"[{name}] 阶段: {' → '.join(p for _, p in r['phase_log'])}")
    if lo:
        print(f"  离地 t={lo['t']:.3f}s v_z={lo['vz']:.2f} θ={lo['pitch']:+.1f}° dθ={lo['pitch_rate']:+.0f}°/s q={lo['hip']:.2f}")
    if td:
        print(f"  着地 t={td['t']:.3f}s θ={td['pitch']:+.1f}° dθ={td['pitch_rate']:+.0f}°/s q={td['hip']:.2f} 轮速失配={td['wheel_mismatch']:.2f} m/s")
    print(f"  质心上升={r['com_rise']*1000:.0f} mm 轮底离地={r['wheel_clearance']*1000:.0f} mm 着地段髋峰={r['hip_peak_landing']:.2f} "
          f"|θ|max={r['max_abs_pitch']:.1f}° 着地法向力峰={r['normal_peak_landing']:.0f} N 漂移={r['drift']:+.2f} m 结束: {r['final_phase']} θ={r['final_pitch']:+.1f}° q={r['final_hip']:.2f} v={r['final_speed']:+.2f} t={r['end_time']:.2f}")


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    _report("T-J3 τ=10", run_jump(verbose=verbose))
