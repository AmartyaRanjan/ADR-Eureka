"""Closed-form dynamics of the 2-link planar arm, batched and differentiable.

The arm lies in the horizontal plane (joints about Z), so gravity produces no
joint torque and the equations of motion reduce to

    M(q; xi) qdd + C(q, qd; xi) + F(qd; xi) = alpha * tau_cmd

where tau_cmd is the torque the policy *asked for* and alpha is per-joint
actuator effectiveness. Keeping alpha on the right-hand side (rather than
feeding the observer the already-scaled torque) is what makes actuator
degradation identifiable — otherwise it is absorbed into the applied torque and
the physics observer can never report it.

Having this in closed form is the main reason the 2-DOF testbed is worth using:
the PINN's physics residual is exact rather than itself approximated, and the
same functions serve as forward dynamics, inverse dynamics and the residual.
Verified against PhysX by scripts/validate_dynamics.py (2.3% mean relative
error on qdd; 0.04 Nm on recovered torque).

Parameter vector, ordered as in the roadmap:
    xi = [m1, m2, f1, f2, b1, b2, a1, a2]
    m  link masses (kg)                 f  Coulomb joint friction (N m)
    b  viscous damping (N m s/rad)      a  actuator effectiveness (1.0 = healthy)

Geometry (link lengths, COM offsets) is treated as known: the roadmap
explicitly excludes randomising link lengths, since they change the kinematics
and workspace and would confound the task distribution.
"""

from __future__ import annotations

import torch

L1 = 0.5
L2 = 0.5
LC1 = 0.25
LC2 = 0.25
# Link inertia about its own COM per kg, thin rod of radius 0.02 m (matches the URDF).
I1_UNIT = (1.0 / 12.0) * L1**2 + 0.25 * 0.02**2
I2_UNIT = (1.0 / 12.0) * L2**2 + 0.25 * 0.02**2

XI_NAMES = ("m1", "m2", "f1", "f2", "b1", "b2", "a1", "a2")
XI_DIM = len(XI_NAMES)
NOMINAL_XI = (1.0, 0.5, 0.0, 0.0, 0.05, 0.05, 1.0, 1.0)

# Smoothing width for the Coulomb term; sign(qd) is not differentiable and
# would make the residual useless as a training signal near zero velocity.
_COULOMB_EPS = 1e-2


def _split(xi: torch.Tensor):
    return xi[..., 0], xi[..., 1], xi[..., 2:4], xi[..., 4:6], xi[..., 6:8]


def mass_matrix(q: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """(..., 2) joint angles -> (..., 2, 2) inertia matrix."""
    m1, m2, _, _, _ = _split(xi)
    i1 = m1 * I1_UNIT
    i2 = m2 * I2_UNIT
    c2 = torch.cos(q[..., 1])

    m11 = i1 + i2 + m1 * LC1**2 + m2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * c2)
    m12 = i2 + m2 * (LC2**2 + L1 * LC2 * c2)
    m22 = i2 + m2 * LC2**2

    row0 = torch.stack([m11, m12], dim=-1)
    row1 = torch.stack([m12, m22.expand_as(m12)], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def coriolis_vector(q: torch.Tensor, qd: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Velocity-product term C(q, qd) qd, shape (..., 2)."""
    _, m2, _, _, _ = _split(xi)
    h = -m2 * L1 * LC2 * torch.sin(q[..., 1])
    qd1, qd2 = qd[..., 0], qd[..., 1]
    return torch.stack([h * (2.0 * qd1 * qd2 + qd2**2), -h * qd1**2], dim=-1)


def friction_vector(qd: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Viscous + smoothed Coulomb joint friction, shape (..., 2)."""
    _, _, f, b, _ = _split(xi)
    return b * qd + f * torch.tanh(qd / _COULOMB_EPS)


def passive_torque(q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """M(q) qdd + C(q, qd) + F(qd) — the mechanical side of the EOM."""
    m = mass_matrix(q, xi)
    return (m @ qdd.unsqueeze(-1)).squeeze(-1) + coriolis_vector(q, qd, xi) + friction_vector(qd, xi)


def inverse_dynamics(q, qd, qdd, xi) -> torch.Tensor:
    """Commanded torque needed to produce qdd, accounting for actuator effectiveness."""
    _, _, _, _, alpha = _split(xi)
    return passive_torque(q, qd, qdd, xi) / alpha.clamp_min(1e-3)


def forward_dynamics(q, qd, tau_cmd, xi) -> torch.Tensor:
    """Joint acceleration produced by a commanded torque under parameters xi."""
    _, _, _, _, alpha = _split(xi)
    m = mass_matrix(q, xi)
    rhs = alpha * tau_cmd - coriolis_vector(q, qd, xi) - friction_vector(qd, xi)
    return torch.linalg.solve(m, rhs.unsqueeze(-1)).squeeze(-1)


def physics_residual(q, qd, qdd, tau_cmd, xi) -> torch.Tensor:
    """r = M qdd + C + F - alpha * tau_cmd. Zero when xi explains the transition."""
    _, _, _, _, alpha = _split(xi)
    return passive_torque(q, qd, qdd, xi) - alpha * tau_cmd
