"""Adaptive domain-randomisation controller — rung F (RAPP + Adaptive DR, no PINN).

Why not the full w(xi) = a_u*U_PINN + a_f*(1-S) + a_s*||grad_xi J|| priority score
from the roadmap: our own measurement (train_pinn.py, results/pinn_observer.json)
found the ensemble's uncertainty only weakly tracks its actual error (Spearman
rank correlation +0.11 interpolation, +0.10 extrapolation — see EXPERIMENT_LOG.md,
"PINN uncertainty negative result"). Weighting U_PINN into the controller would
inject near-noise into the one decision this experiment is meant to validate.
This controller therefore uses only the policy-weakness term (1-S), which is
exactly OpenAI's ADR (Akkaya et al. 2019) boundary-widening rule applied to range
WIDTH rather than per-sample boundary probing — the generic adaptive-DR baseline
the roadmap's own rung ladder calls "rung E/F" (adaptive DR, no PINN).

Rule, per parameter dimension d independently:
  S_d = mean success P(d_T < 5cm) measured under the CURRENT training range
  if S_d >= success_target:  widen range_d toward its C_RAPP bound (step_frac
                              of the remaining gap)
  else:                      hold range_d fixed (do not widen into a region the
                              policy cannot yet handle); after `patience` stalled
                              outer iterations in a row, narrow back toward the
                              starting range instead of stalling forever.

Mass is asymmetric in the measured RAPP envelope (feasible [1.0, 4.0], infeasible
below 1.0 — see run_rapp.py / rapp_eureka500_envelope.json), so only its upper
bound moves; lower bound is pinned at the nominal 1.0.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


# Parameter order matches env.physics_xi minus mass's lower bound special-case.
DIMENSIONS = ("friction", "damping_scale", "effectiveness", "mass_scale")

# Which bound each dimension widens (the other bound is held fixed at its
# starting value for the whole run). Determined directly from the measured
# RAPP envelope: friction/damping/mass get harder going UP, effectiveness gets
# harder going DOWN.
WIDEN_DIRECTION = {
    "friction": "upper",
    "damping_scale": "upper",
    "effectiveness": "lower",
    "mass_scale": "upper",
}


@dataclass
class AdaptiveDRState:
    """ψ_t: the current per-dimension training range, plus controller bookkeeping."""

    ranges: dict[str, tuple[float, float]]
    stalled_count: dict[str, int] = field(default_factory=lambda: {d: 0 for d in DIMENSIONS})
    outer_iteration: int = 0

    def to_json(self) -> dict:
        return {"ranges": {k: list(v) for k, v in self.ranges.items()}, "stalled_count": self.stalled_count,
                "outer_iteration": self.outer_iteration}


class AdaptiveDRController:
    """Slow outer-loop controller: observe S_d, update ψ_t -> ψ_{t+1}, clipped to C_RAPP."""

    def __init__(
        self,
        initial_ranges: dict[str, tuple[float, float]],
        c_rapp: dict[str, tuple[float, float]],
        success_target: float = 0.90,
        step_frac: float = 0.25,
        narrow_frac: float = 0.15,
        patience: int = 2,
    ):
        self.state = AdaptiveDRState(ranges=dict(initial_ranges))
        self.initial_ranges = dict(initial_ranges)
        self.c_rapp = dict(c_rapp)
        self.success_target = success_target
        self.step_frac = step_frac
        self.narrow_frac = narrow_frac
        self.patience = patience
        self.history: list[dict] = []

    def update(self, success_per_dim: dict[str, float],
               step_multiplier: dict[str, float] | None = None) -> dict[str, tuple[float, float]]:
        """success_per_dim[d] = measured P(d_T<5cm) under the current range for dim d.

        Each dimension is evaluated and updated independently (the roadmap's
        one-parameter-at-a-time discipline, carried over from RAPP) — this
        avoids conflating which dimension caused a drop in success.
        """
        mult = step_multiplier or {}
        record = {"outer_iteration": self.state.outer_iteration, "success": dict(success_per_dim), "action": {},
                  "step_multiplier": {d: mult.get(d, 1.0) for d in DIMENSIONS}}

        for d in DIMENSIONS:
            lo, hi = self.state.ranges[d]
            rapp_lo, rapp_hi = self.c_rapp[d]
            s = success_per_dim[d]
            direction = WIDEN_DIRECTION[d]
            step = min(1.0, self.step_frac * mult.get(d, 1.0))

            if s >= self.success_target:
                self.state.stalled_count[d] = 0
                if direction == "upper":
                    gap = rapp_hi - hi
                    new_hi = hi + step * gap
                    self.state.ranges[d] = (lo, min(new_hi, rapp_hi))
                    record["action"][d] = f"widen upper {hi:.4f} -> {self.state.ranges[d][1]:.4f}"
                else:  # lower
                    gap = lo - rapp_lo
                    new_lo = lo - step * gap
                    self.state.ranges[d] = (max(new_lo, rapp_lo), hi)
                    record["action"][d] = f"widen lower {lo:.4f} -> {self.state.ranges[d][0]:.4f}"
            else:
                self.state.stalled_count[d] += 1
                if self.state.stalled_count[d] > self.patience:
                    init_lo, init_hi = self.initial_ranges[d]
                    if direction == "upper":
                        new_hi = hi - self.narrow_frac * (hi - init_hi) if hi > init_hi else hi
                        self.state.ranges[d] = (lo, max(new_hi, init_hi))
                    else:
                        new_lo = lo + self.narrow_frac * (init_lo - lo) if lo < init_lo else lo
                        self.state.ranges[d] = (min(new_lo, init_lo), hi)
                    record["action"][d] = f"stalled {self.state.stalled_count[d]}x -> narrow to {self.state.ranges[d]}"
                else:
                    record["action"][d] = f"stalled {self.state.stalled_count[d]}x (<= patience, hold)"

        self.state.outer_iteration += 1
        self.history.append(record)
        return dict(self.state.ranges)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({
                "initial_ranges": {k: list(v) for k, v in self.initial_ranges.items()},
                "c_rapp": {k: list(v) for k, v in self.c_rapp.items()},
                "success_target": self.success_target,
                "step_frac": self.step_frac,
                "narrow_frac": self.narrow_frac,
                "patience": self.patience,
                "final_state": self.state.to_json(),
                "history": self.history,
            }, fh, indent=2)
