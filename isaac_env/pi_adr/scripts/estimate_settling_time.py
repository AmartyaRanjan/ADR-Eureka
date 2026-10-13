"""Analytic estimate of the damping settling time constant M/b across the PINN
dataset's physics ranges -- sizes --long_hold_steps in collect_pinn_data.py.

For a single damped joint M*qdd + b*qd = tau (Coriolis/friction-sign terms
neglected for this estimate -- they are smaller than viscous damping near the
velocities this arm operates at), qd(t) = (tau/b)*(1 - exp(-b*t/M)), so the
time constant is tau_c = M/b. Reaching fraction f of steady state takes
t = -tau_c * ln(1-f).

No simulator needed -- pure analysis of source/pi_adr/pinn/dynamics.py's
closed-form M(q;xi), evaluated at the PINN dataset's own declared ranges.

    env_isaaclab/bin/python pi_adr/scripts/estimate_settling_time.py
"""

import math

import torch

from pi_adr.pinn.dynamics import mass_matrix

PHYSICS_DT = 1.0 / 120.0

# Must match collect_pinn_data.py's RANGES["interp"] -- nominal damping is 0.05,
# so joint_damping_scale_range (0.5, 4.0) -> b in [0.025, 0.2].
DAMPING_B_RANGE = (0.05 * 0.5, 0.05 * 4.0)
MASS_SCALE_RANGE = (0.7, 1.4)


def main() -> None:
    q_worst = torch.tensor([[0.0, 0.0]])  # cos(q2)=1 maximizes M11 (joint-1 inertia)

    print("=== settling time constant tau_c = M/b across the PINN training range ===\n")
    print(f"{'mass_scale':>10} {'b (damping)':>12} {'M11 (kg m^2)':>14} {'tau_c (s)':>10} {'steps@120Hz':>12} "
          f"{'steps for 90%':>14} {'steps for 95%':>14}")
    for m_scale in (MASS_SCALE_RANGE[0], 1.0, MASS_SCALE_RANGE[1]):
        xi = torch.tensor([[m_scale, 0.5 * m_scale, 0.0, 0.0, 0.05, 0.05, 1.0, 1.0]])
        M = mass_matrix(q_worst, xi)
        m11 = M[0, 0, 0].item()
        for b in DAMPING_B_RANGE:
            tau_c = m11 / b
            steps_1tc = tau_c / PHYSICS_DT
            steps_90 = -tau_c * math.log(1 - 0.90) / PHYSICS_DT
            steps_95 = -tau_c * math.log(1 - 0.95) / PHYSICS_DT
            print(f"{m_scale:>10.2f} {b:>12.4f} {m11:>14.5f} {tau_c:>10.3f} {steps_1tc:>12.0f} "
                  f"{steps_90:>14.0f} {steps_95:>14.0f}")

    print(
        "\nConclusion: full (90-95%) settling across the WHOLE training range would need thousands "
        "of steps at the low-damping end -- impractical for a dataset covering many environments. "
        "A long_hold_steps in the few-hundred range gives near-complete settling for the "
        "higher-damping half of the range and partial (but still velocity-term-dominated, hence "
        "informative) settling for the rest -- an honest middle ground, not full convergence."
    )


if __name__ == "__main__":
    main()
