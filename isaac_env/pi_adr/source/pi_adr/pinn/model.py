"""Physics observer: infer the physics vector xi from observed transitions.

Roadmap role (section 8): a *frozen* observer that reads per-environment
trajectories and reports both a physics estimate and how uncertain it is. The
uncertainty becomes U_PINN(xi) in the adaptive-DR priority score, so a point
estimate alone is not enough — a poorly excited trajectory carries little
information about friction and the controller has to know that.

Design choices worth stating:

* **Set encoder, not a sequence model.** The EOM is instantaneous: every
  transition (q, qd, qdd, tau) independently constrains xi. Pooling over a
  permutation-invariant set matches that structure and is robust to the window
  length changing between training and deployment.
* **Bounded outputs.** xi is squashed into physically admissible ranges by
  construction rather than penalised for leaving them, so the observer can
  never hand the DR controller a negative mass. This replaces the roadmap's
  soft L_reg with a hard constraint.
* **Deep ensemble** for uncertainty — simple, inspectable, and the roadmap's
  recommended first choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dynamics import XI_NAMES, forward_dynamics, physics_residual

# Physically admissible ranges for xi = [m1, m2, f1, f2, b1, b2, a1, a2].
XI_BOUNDS = torch.tensor(
    [
        [0.25, 4.0],   # m1 kg
        [0.10, 2.0],   # m2 kg
        [0.0, 1.0],    # f1 Coulomb friction (N m)
        [0.0, 1.0],    # f2
        [0.0, 0.5],    # b1 viscous damping
        [0.0, 0.5],    # b2
        [0.2, 1.2],    # a1 actuator effectiveness
        [0.2, 1.2],    # a2
    ]
)


class PhysicsObserver(nn.Module):
    """(B, T, 8) transitions -> xi estimate (B, 6) with per-parameter log-variance."""

    def __init__(self, hidden: int = 256, embed: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, embed),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * embed, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2 * len(XI_NAMES)),
        )
        self.register_buffer("bounds", XI_BOUNDS.clone())

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(window)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        out = self.head(pooled)
        raw, log_var = out.chunk(2, dim=-1)

        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        xi = lo + (hi - lo) * torch.sigmoid(raw)
        return xi, log_var.clamp(-10.0, 5.0)


class ObserverEnsemble(nn.Module):
    """Deep ensemble; predictive variance across members is U_PINN."""

    def __init__(self, num_members: int = 5, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([PhysicsObserver(**kwargs) for _ in range(num_members)])

    def forward(self, window: torch.Tensor):
        xis, log_vars = zip(*(m(window) for m in self.members))
        xi_stack = torch.stack(xis)
        mean = xi_stack.mean(dim=0)
        # Total predictive variance = spread between members (epistemic) plus
        # the mean of each member's own predicted variance (aleatoric).
        epistemic = xi_stack.var(dim=0, unbiased=False)
        aleatoric = torch.stack(log_vars).exp().mean(dim=0)
        return mean, epistemic + aleatoric, epistemic


def normalize_xi(xi: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (xi - lo) / (hi - lo)


def observer_losses(
    xi_pred: torch.Tensor,
    log_var: torch.Tensor,
    xi_true: torch.Tensor,
    window: torch.Tensor,
    bounds: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """L_data (acceleration), L_phys (EOM residual), L_sup (recovered parameters).

    window packs (q, qd, tau, qdd) so the physics terms can be evaluated at every
    transition against the single xi inferred for that environment.
    """
    q, qd, tau, qdd = window.split(2, dim=-1)
    xi_b = xi_pred.unsqueeze(1).expand(-1, window.shape[1], -1)

    qdd_pred = forward_dynamics(q, qd, tau, xi_b)
    l_data = (qdd_pred - qdd).pow(2).mean()

    l_phys = physics_residual(q, qd, qdd, tau, xi_b).pow(2).mean()

    err = normalize_xi(xi_pred, bounds) - normalize_xi(xi_true, bounds)
    l_sup = (err.pow(2) * torch.exp(-log_var) + log_var).mean()

    return {"data": l_data, "phys": l_phys, "sup": l_sup}
