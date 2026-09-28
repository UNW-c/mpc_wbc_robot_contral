"""按整数物理步调度多频率控制器，避免浮点累计误差。"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class ControlTiming:
    physics_dt: float = 0.001
    wbc_dt: float = 0.004
    mpc_dt: float = 0.02
    wbc_steps: int = field(init=False)
    mpc_steps: int = field(init=False)

    def __post_init__(self) -> None:
        if any(not math.isfinite(dt) or dt <= 0.0 for dt in (self.physics_dt, self.wbc_dt, self.mpc_dt)):
            raise ValueError("所有控制周期必须有限且为正")
        for name in ("wbc", "mpc"):
            ratio = getattr(self, f"{name}_dt") / self.physics_dt
            count = round(ratio)
            if count < 1 or not math.isclose(ratio, count, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("控制周期必须是物理步长的整数倍")
            object.__setattr__(self, f"{name}_steps", count)
        if self.mpc_steps % self.wbc_steps:
            raise ValueError("MPC 周期必须是 WBC 周期的整数倍")

    def wbc_due(self, physics_step: int) -> bool:
        return physics_step % self.wbc_steps == 0

    def mpc_due(self, physics_step: int) -> bool:
        return physics_step % self.mpc_steps == 0
