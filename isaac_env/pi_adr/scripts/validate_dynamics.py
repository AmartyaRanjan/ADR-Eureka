"""Check the closed-form arm dynamics against PhysX.

If these disagree, the PINN's physics residual is measuring the wrong thing, so
this must pass before any PINN result is meaningful. Also reports what joint
properties the simulator actually holds (damping/friction/armature/mass), which
is the ground truth the physics observer is supposed to recover.

    env_isaaclab/bin/python pi_adr/scripts/validate_dynamics.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import pi_adr.tasks  # noqa: F401, E402
from pi_adr.pinn.dynamics import XI_NAMES, forward_dynamics, inverse_dynamics  # noqa: E402
from pi_adr.tasks.two_dof_reach.two_dof_reach_env_cfg import (  # noqa: E402
    TwoDofReachEnvCfg,
    VELOCITY_LIMIT,
)


def main() -> int:
    cfg = TwoDofReachEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.domain_rand.enabled = False

    env = gym.make("PIADR-TwoDofReach-Vanilla-v0", cfg=cfg).unwrapped
    robot = env._robot
    jid = env._joint_ids
    device = env.device
    env.reset()

    print("\n=== simulator joint properties (nominal) ===")
    props = {
        "joint_damping": robot.data.joint_damping[:, jid],
        "joint_stiffness": robot.data.joint_stiffness[:, jid],
        "joint_friction_coeff": robot.data.joint_friction_coeff[:, jid],
        "joint_armature": robot.data.joint_armature[:, jid],
    }
    for name, val in props.items():
        print(f"  {name:<22} {val[0].tolist()}")
    print(f"  link masses            {robot.data.default_mass[0].tolist()}")
    print(f"  body names             {robot.body_names}")

    b_sim = robot.data.joint_damping[0, jid].mean().item()
    f_sim = robot.data.joint_friction_coeff[0, jid].mean().item()
    masses = robot.data.default_mass[0]
    body_names = robot.body_names
    m1 = masses[body_names.index("link_1")].item()
    m2 = masses[body_names.index("link_2")].item()

    # xi = [m1, m2, f1, f2, b1, b2, a1, a2]; actuators are healthy at nominal.
    xi = torch.tensor([m1, m2, f_sim, f_sim, b_sim, b_sim, 1.0, 1.0], device=device)
    xi = xi.unsqueeze(0).expand(args_cli.num_envs, -1)
    print(f"\n  xi used for the analytic model: {dict(zip(XI_NAMES, xi[0].tolist()))}")

    # Random states and torques; compare one physics step of PhysX against the
    # closed-form forward dynamics evaluated at that state.
    # Stay clear of the +/-pi joint limits: PhysX applies large constraint
    # forces there, which the unconstrained closed-form model does not model.
    torch.manual_seed(0)
    # Torques are kept moderate as well: link 2 has M22 ~ 0.042 kg m^2, so a
    # full 20 Nm gives ~480 rad/s^2 and saturates the +/-5 rad/s velocity limit
    # within one 1/120 s step, which would corrupt the finite-difference
    # acceleration rather than reveal a modelling error.
    q = torch.empty(args_cli.num_envs, 2, device=device).uniform_(-2.5, 2.5)
    qd = torch.empty(args_cli.num_envs, 2, device=device).uniform_(-1.0, 1.0)
    tau = torch.empty(args_cli.num_envs, 2, device=device).uniform_(-5.0, 5.0)

    robot.write_joint_state_to_sim(q, qd, joint_ids=jid)
    robot.set_joint_effort_target(tau, joint_ids=jid)
    robot.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)

    qd_next = robot.data.joint_vel[:, jid]
    qdd_sim = (qd_next - qd) / env.physics_dt
    qdd_analytic = forward_dynamics(q, qd, tau, xi)

    saturated = (qd_next.abs() >= 0.99 * VELOCITY_LIMIT).float().mean()
    print(f"\n  fraction of samples at the velocity limit: {saturated.item() * 100:.2f} %")

    err = (qdd_sim - qdd_analytic).abs()
    rel = err / qdd_analytic.abs().clamp_min(1.0)
    print("\n=== forward dynamics: PhysX vs closed form ===")
    print(f"  |qdd| range        {qdd_sim.abs().min().item():.3f} .. {qdd_sim.abs().max().item():.3f} rad/s^2")
    print(f"  mean abs error     {err.mean().item():.4f} rad/s^2")
    print(f"  max abs error      {err.max().item():.4f} rad/s^2")
    print(f"  mean rel error     {rel.mean().item() * 100:.2f} %")

    tau_recovered = inverse_dynamics(q, qd, qdd_sim, xi)
    tau_err = (tau_recovered - tau).abs()
    print("\n=== inverse dynamics: recovered torque vs commanded ===")
    print(f"  mean abs error     {tau_err.mean().item():.4f} Nm  (commanded up to 20 Nm)")
    print(f"  max abs error      {tau_err.max().item():.4f} Nm")

    passed = rel.mean().item() < 0.05 and tau_err.mean().item() < 1.0
    print(f"\n==== dynamics check {'PASSED' if passed else 'FAILED'} ====\n")

    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
