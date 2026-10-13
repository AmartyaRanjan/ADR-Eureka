"""Evaluation metrics for the 2-DOF reacher, fixed to match the stage-1 report.

Every number this module produces is defined once here so that results stay
comparable across vanilla/Eureka rewards, DR variants and future PINN runs.

Definitions (frozen — do not change without bumping METRICS_VERSION):
  terminal error d_T   Euclidean distance ||p_target - p_ee|| at the LAST step
                       of the episode, in metres.
  P(d < eps)           fraction of episodes whose terminal error is below eps.
  mean joint velocity  mean over (episode steps, envs, joints) of |qd|, rad/s.
  mean control effort  mean over (episode steps, envs) of ||a||_2^2, where a is
                       the normalised action in [-1, 1]^2 (so effort in [0, 2]).
  settling time        first time (s) after which d stays below SETTLE_EPS for
                       the remainder of the episode; episode length if never.
  overshoot            max(0, max_t d(t) - d(0)) over the episode, in metres —
                       how much further away the tip got than it started.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

import torch

METRICS_VERSION = "1.0.0"

# Report thresholds (5 cm / 2 cm / 1 cm) plus the finer ones the roadmap asks for.
PRECISION_THRESHOLDS_M: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
SETTLE_EPS_M = 0.05


@dataclass
class EvalResult:
    """Frozen metric set for one evaluated policy."""

    run_name: str
    num_episodes: int
    metrics_version: str = METRICS_VERSION

    success_rate: dict[str, float] = field(default_factory=dict)
    mean_terminal_error_m: float = float("nan")
    median_terminal_error_m: float = float("nan")
    p90_terminal_error_m: float = float("nan")
    worst_decile_error_m: float = float("nan")

    mean_joint_velocity_rad_s: float = float("nan")
    max_joint_velocity_rad_s: float = float("nan")
    mean_control_effort: float = float("nan")
    mean_abs_action: float = float("nan")

    mean_settling_time_s: float = float("nan")
    mean_overshoot_m: float = float("nan")

    extra: dict = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)

    def summary_line(self) -> str:
        s = self.success_rate
        return (
            f"{self.run_name}: "
            f"5cm={s.get('0.05', float('nan')):.2%} "
            f"2cm={s.get('0.02', float('nan')):.2%} "
            f"1cm={s.get('0.01', float('nan')):.2%} | "
            f"mean={self.mean_terminal_error_m * 1000:.1f}mm "
            f"med={self.median_terminal_error_m * 1000:.1f}mm | "
            f"vel={self.mean_joint_velocity_rad_s:.3f}rad/s "
            f"eff={self.mean_control_effort:.3f}"
        )


class RolloutAccumulator:
    """Collects per-step tensors over a fixed-horizon vectorised rollout.

    Call `record` once per control step with per-env quantities, then `finalize`.
    Kept on-device; only the final reduction moves to CPU.
    """

    def __init__(self, num_envs: int, dt: float, device: torch.device | str = "cuda"):
        self.num_envs = num_envs
        self.dt = dt
        self.device = torch.device(device)
        self._dist: list[torch.Tensor] = []
        self._vel_abs_sum: list[torch.Tensor] = []
        self._vel_max: list[torch.Tensor] = []
        self._effort: list[torch.Tensor] = []
        self._abs_action: list[torch.Tensor] = []

    def record(self, dist: torch.Tensor, joint_vel: torch.Tensor, action: torch.Tensor) -> None:
        """dist: (N,) metres; joint_vel: (N, J) rad/s; action: (N, A) normalised."""
        self._dist.append(dist.detach().flatten())
        self._vel_abs_sum.append(joint_vel.detach().abs().mean(dim=-1))
        self._vel_max.append(joint_vel.detach().abs().amax(dim=-1))
        self._effort.append(action.detach().pow(2).sum(dim=-1))
        self._abs_action.append(action.detach().abs().mean(dim=-1))

    def finalize(self, run_name: str, extra: dict | None = None) -> EvalResult:
        dist = torch.stack(self._dist)  # (T, N)
        vel = torch.stack(self._vel_abs_sum)
        vel_max = torch.stack(self._vel_max)
        effort = torch.stack(self._effort)
        abs_action = torch.stack(self._abs_action)

        terminal = dist[-1]
        sorted_terminal = terminal.sort().values
        n = sorted_terminal.numel()
        worst_decile = sorted_terminal[int(math.floor(0.9 * n)) :].mean()

        success = {
            f"{eps:g}": (terminal < eps).float().mean().item()
            for eps in PRECISION_THRESHOLDS_M
        }

        return EvalResult(
            run_name=run_name,
            num_episodes=int(n),
            success_rate=success,
            mean_terminal_error_m=terminal.mean().item(),
            median_terminal_error_m=terminal.median().item(),
            p90_terminal_error_m=torch.quantile(terminal.float(), 0.9).item(),
            worst_decile_error_m=worst_decile.item(),
            mean_joint_velocity_rad_s=vel.mean().item(),
            max_joint_velocity_rad_s=vel_max.max().item(),
            mean_control_effort=effort.mean().item(),
            mean_abs_action=abs_action.mean().item(),
            mean_settling_time_s=settling_time(dist, self.dt).mean().item(),
            mean_overshoot_m=overshoot(dist).mean().item(),
            extra=extra or {},
        )

    @property
    def terminal_distance(self) -> torch.Tensor:
        return self._dist[-1]

    @property
    def per_env_mean_velocity(self) -> torch.Tensor:
        """(N,) mean |qdot| over the episode, per environment."""
        return torch.stack(self._vel_abs_sum).mean(dim=0)

    @property
    def per_env_mean_effort(self) -> torch.Tensor:
        """(N,) mean ||a||^2 over the episode, per environment."""
        return torch.stack(self._effort).mean(dim=0)

    @property
    def per_env_settling_time(self) -> torch.Tensor:
        """(N,) settling time in seconds, per environment."""
        return settling_time(torch.stack(self._dist), self.dt)


def settling_time(dist: torch.Tensor, dt: float, eps: float = SETTLE_EPS_M) -> torch.Tensor:
    """(T, N) distances -> (N,) seconds until d stays < eps for the rest of the episode."""
    inside = dist < eps
    # Suffix-AND: index t is True iff every step from t onward is inside eps.
    stays = torch.flip(torch.cumprod(torch.flip(inside.int(), dims=[0]), dim=0), dims=[0]).bool()
    horizon = dist.shape[0]
    first = torch.where(
        stays.any(dim=0),
        stays.float().argmax(dim=0),
        torch.full((dist.shape[1],), horizon, device=dist.device, dtype=torch.long),
    )
    return first.float() * dt


def overshoot(dist: torch.Tensor) -> torch.Tensor:
    """(T, N) distances -> (N,) metres the tip receded beyond its starting error."""
    return (dist.amax(dim=0) - dist[0]).clamp_min(0.0)


def precision_curve(terminal: torch.Tensor, thresholds: Sequence[float] = PRECISION_THRESHOLDS_M) -> dict[str, float]:
    return {f"{eps:g}": (terminal < eps).float().mean().item() for eps in thresholds}


def physics_binned_success(
    terminal: torch.Tensor,
    physics_value: torch.Tensor,
    num_bins: int = 8,
    threshold: float = 0.01,
) -> dict:
    """P(d < threshold | physics parameter in bin) — the physics-performance map."""
    lo, hi = physics_value.min().item(), physics_value.max().item()
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    edges = torch.linspace(lo, hi, num_bins + 1, device=physics_value.device)
    idx = torch.bucketize(physics_value, edges[1:-1].contiguous())
    centers, rates, counts = [], [], []
    for b in range(num_bins):
        mask = idx == b
        count = int(mask.sum().item())
        centers.append(0.5 * (edges[b] + edges[b + 1]).item())
        rates.append((terminal[mask] < threshold).float().mean().item() if count else float("nan"))
        counts.append(count)
    return {"threshold": threshold, "bin_centers": centers, "success_rate": rates, "counts": counts}


def physics_binned_metrics(
    accum: "RolloutAccumulator",
    physics_value: torch.Tensor,
    num_bins: int = 8,
) -> dict:
    """Full M(xi) = [P_5cm, P_2cm, P_1cm, E_d, E_qdot, E_tau, T_settle] per bin.

    Extends physics_binned_success (which reports P(d<1cm) only) to the full
    metric vector the roadmap defines, from quantities the accumulator already
    computes per rollout -- no new simulation required, this is purely a richer
    reduction of data already collected by eval.py's rollout() call.
    """
    terminal = accum.terminal_distance
    velocity = accum.per_env_mean_velocity
    effort = accum.per_env_mean_effort
    settle = accum.per_env_settling_time

    lo, hi = physics_value.min().item(), physics_value.max().item()
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    edges = torch.linspace(lo, hi, num_bins + 1, device=physics_value.device)
    idx = torch.bucketize(physics_value, edges[1:-1].contiguous())

    centers, counts = [], []
    p5cm, p2cm, p1cm, mean_err, mean_vel, mean_eff, mean_settle = [], [], [], [], [], [], []
    for b in range(num_bins):
        mask = idx == b
        count = int(mask.sum().item())
        centers.append(0.5 * (edges[b] + edges[b + 1]).item())
        counts.append(count)
        if count:
            d = terminal[mask]
            p5cm.append((d < 0.05).float().mean().item())
            p2cm.append((d < 0.02).float().mean().item())
            p1cm.append((d < 0.01).float().mean().item())
            mean_err.append(d.mean().item())
            mean_vel.append(velocity[mask].mean().item())
            mean_eff.append(effort[mask].mean().item())
            mean_settle.append(settle[mask].mean().item())
        else:
            for lst in (p5cm, p2cm, p1cm, mean_err, mean_vel, mean_eff, mean_settle):
                lst.append(float("nan"))

    return {
        "bin_centers": centers,
        "counts": counts,
        "P_5cm": p5cm,
        "P_2cm": p2cm,
        "P_1cm": p1cm,
        "mean_terminal_error_m": mean_err,
        "mean_joint_velocity_rad_s": mean_vel,
        "mean_control_effort": mean_eff,
        "mean_settling_time_s": mean_settle,
    }
