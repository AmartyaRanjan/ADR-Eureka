"""Generate the dynamics dataset for the physics observer.

Follows the roadmap's dataset rules (section 17): sample physics from a
controlled distribution, excite with scripted/random torques rather than one
policy's narrow state distribution, record q/qd/tau/qdd, and split by *physics
configuration* so a model cannot pass validation by memorising a configuration
it already saw.

Each environment is one physics configuration and contributes one window of
transitions, which is exactly the input the observer consumes at deployment.

Saturation matters here: when a joint is pinned at +/-5 rad/s the measured
acceleration is produced by the velocity limit, not by xi, and no value of xi
explains it. Those samples are masked out rather than silently taught to the
model as physics.

    env_isaaclab/bin/python pi_adr/scripts/collect_pinn_data.py --headless \
        --split train --num_envs 4096 --steps 128
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=128, help="transitions recorded per environment")
parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "extrap"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--torque_scale", type=float, nargs="+", default=[0.25],
    help="fraction(s) of the 20 Nm limit. One value = single regime (original behaviour). "
         "Two+ values = each hold-cycle samples uniformly from the given discrete set, so the "
         "SAME environment (same xi) is excited at multiple amplitudes within one window -- this "
         "is the fix for the confirmed m/alpha identifiability degeneracy (EXPERIMENT_LOG.md, "
         "2026-10-10, PINN covariance diagnosis): a single torque regime cannot separate mass and "
         "actuator effectiveness since alpha*tau/M enters multiplicatively; varying tau changes "
         "the relative size of the Coriolis/friction terms and should break the degeneracy.",
)
parser.add_argument(
    "--max_speed",
    type=float,
    default=3.5,
    help="mask transitions above this |qd|; PhysX ramps the 5 rad/s limit in from ~4 rad/s",
)
parser.add_argument("--hold", type=int, default=6, help="physics steps each random torque is held")
parser.add_argument("--resample", type=int, default=32, help="physics steps between state re-seeds")
parser.add_argument(
    "--long_hold_steps", type=int, default=0,
    help="if >0, append this many EXTRA physics steps per environment holding one constant torque "
         "each, after the normal transient phase -- a second, confirmed-necessary fix for the m/alpha "
         "degeneracy (EXPERIMENT_LOG.md, 2026-10-10, 'dual-torque-regime fix... REFUTED'): varying "
         "torque AMPLITUDE cannot separate mass from actuator effectiveness since alpha*tau/M scales "
         "linearly with tau regardless of the alpha/M split; only velocity-dependent terms (damping, "
         "Coriolis) can, and those only matter relative to the inertial term near steady state "
         "(q_ddot -> 0). At this environment's own low-damping end the settling time constant M/b is "
         "~21s (2521 physics steps) -- full convergence across the whole training range is not "
         "practical in one dataset, so this is a PARTIAL long hold: higher-damping environments settle "
         "much more completely than low-damping ones within the same window. Analytic time-constant "
         "estimate: scripts/estimate_settling_time.py.",
)
parser.add_argument("--tag", type=str, default="", help="suffix for the output file: pinn_<split>_<tag>.pt")
parser.add_argument(
    "--long_hold_mode", type=str, default="simultaneous", choices=["simultaneous", "sequential"],
    help="simultaneous: both joints driven together for the whole long hold (the 11 Oct 'longhold' "
         "dataset). sequential: first half drives ONLY joint 1 (joint 2 torque 0), second half ONLY "
         "joint 2 -- tests the hypothesis (EXPERIMENT_LOG.md, long-hold PINN entry) that joint 2's "
         "mass/effectiveness degeneracy persisted because simultaneous holds couple the joints "
         "through M(q) and Coriolis, contaminating joint 2's steady-state reading.",
)
parser.add_argument(
    "--long_hold_target_velocity", type=float, default=2.0,
    help="target steady-state |qdot| (rad/s) during the long-hold phase, comfortably under the "
         "3.5 rad/s mask. Hold torque is sized PER-ENVIRONMENT from its own ground-truth (b, alpha) "
         "to reach this velocity, not a single fixed torque for every environment -- a fixed torque "
         "was tried first and failed (see --long_hold_steps help text): low-damping/high-"
         "effectiveness environments hit the arm's 5 rad/s hardware limit, a different and "
         "uninformative regime, well before any natural settling.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import pi_adr.tasks  # noqa: F401, E402
from pi_adr.tasks.two_dof_reach.two_dof_reach_env_cfg import (  # noqa: E402
    TwoDofReachEnvCfg,
    VELOCITY_LIMIT,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Training/val/test share one physics distribution and differ only by seed, so
# the split is across configurations. "extrap" deliberately lies outside it, to
# separate interpolation from extrapolation as the roadmap requires.
RANGES = {
    "interp": {
        "joint_friction_range": (0.0, 0.25),
        "joint_damping_scale_range": (0.5, 4.0),
        "actuator_effectiveness_range": (0.5, 1.0),
        "link_mass_scale_range": (0.7, 1.4),
    },
    "extrap": {
        "joint_friction_range": (0.25, 0.45),
        "joint_damping_scale_range": (4.0, 7.0),
        "actuator_effectiveness_range": (0.3, 0.5),
        "link_mass_scale_range": (1.4, 1.9),
    },
}


def main() -> None:
    torch.manual_seed(args_cli.seed)

    cfg = TwoDofReachEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    cfg.domain_rand.enabled = True
    for key, value in RANGES["extrap" if args_cli.split == "extrap" else "interp"].items():
        setattr(cfg.domain_rand, key, value)

    env = gym.make("PIADR-TwoDofReach-Vanilla-v0", cfg=cfg).unwrapped
    device = env.device
    env.reset()

    xi = env.physics_xi.clone()
    n = env.num_envs
    t_transient = args_cli.steps
    t_long_hold = args_cli.long_hold_steps
    t = t_transient + t_long_hold

    q_buf = torch.zeros(n, t, 2, device=device)
    qd_buf = torch.zeros(n, t, 2, device=device)
    tau_buf = torch.zeros(n, t, 2, device=device)
    qdd_buf = torch.zeros(n, t, 2, device=device)
    valid_buf = torch.zeros(n, t, dtype=torch.bool, device=device)
    is_long_hold_buf = torch.zeros(n, t, dtype=torch.bool, device=device)

    # Step the simulator one PHYSICS step at a time rather than through
    # env.step(). A control step holds the torque for 4 physics substeps, over
    # which the state changes a lot, so (qd_after - qd_before)/step_dt is an
    # average acceleration that does not satisfy the EOM at the recorded state:
    # measured against the true xi it gives a 78.8 rad/s^2 residual, and an
    # observer trained on it fits label noise instead of physics. Differencing
    # over a single 1/120 s step matches the closed form to ~2%.
    robot = env._robot
    jid = env._joint_ids
    action = torch.zeros(n, 2, device=device)
    scales = torch.tensor(args_cli.torque_scale, device=device)

    with torch.inference_mode():
        for step in range(t_transient):
            if step % args_cli.hold == 0:
                # Per-env, per-hold-cycle regime choice: with >1 scale given, the
                # SAME environment (same xi) gets excited at different amplitudes
                # across the window, not just different environments at different
                # amplitudes -- see the --torque_scale help text for why this
                # matters for identifiability.
                regime = scales[torch.randint(0, len(scales), (n, 1), device=device)]
                action = torch.empty(n, 2, device=device).uniform_(-1.0, 1.0) * regime
            if step % args_cli.resample == 0:
                # Re-seed the state so the window covers the workspace instead of
                # wherever the random walk drifted.
                q_new = torch.empty(n, 2, device=device).uniform_(-torch.pi, torch.pi)
                qd_new = torch.empty(n, 2, device=device).uniform_(-1.5, 1.5)
                robot.write_joint_state_to_sim(q_new, qd_new, joint_ids=jid)

            q_before = env._wrapped_joint_pos().clone()
            qd_before = robot.data.joint_vel[:, jid].clone()
            tau_cmd = env.cfg.torque_limit * action

            robot.set_joint_effort_target(tau_cmd * env._actuator_effectiveness, joint_ids=jid)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

            qd_after = robot.data.joint_vel[:, jid]
            qdd = (qd_after - qd_before) / env.physics_dt

            q_buf[:, step] = q_before
            qd_buf[:, step] = qd_before
            tau_buf[:, step] = tau_cmd
            qdd_buf[:, step] = qdd
            # Drop transitions near the velocity limit: PhysX ramps the
            # constraint in from roughly 4 rad/s, and there the limit rather
            # than xi sets the acceleration. Measured median model error jumps
            # from 4.2 rad/s^2 below 4 rad/s to 68 rad/s^2 in [4, 4.9).
            fast = (qd_after.abs() >= args_cli.max_speed) | (qd_before.abs() >= args_cli.max_speed)
            valid_buf[:, step] = ~fast.any(dim=-1)

        # Long-hold phase: ONE random-sign torque per environment, held constant
        # (no resampling, no regime switching) for t_long_hold steps, so the arm
        # approaches (partially, for low-damping envs -- see
        # scripts/estimate_settling_time.py) its steady-state velocity. This is
        # what can actually separate mass from actuator effectiveness; varying
        # torque AMPLITUDE in the transient phase above cannot (see
        # --long_hold_steps help text / EXPERIMENT_LOG.md).
        if t_long_hold > 0:
            q_new = torch.empty(n, 2, device=device).uniform_(-torch.pi, torch.pi)
            qd_new = torch.zeros(n, 2, device=device)
            robot.write_joint_state_to_sim(q_new, qd_new, joint_ids=jid)

            # A FIXED torque for every environment was tried first and failed: a
            # smoke test showed only 0.2% of the final (most-settled) long-hold
            # steps were still below the velocity mask -- low-damping/high-
            # effectiveness environments have a steady-state velocity (v_ss =
            # alpha*tau/b) that exceeds this arm's own 5 rad/s limit well before
            # they'd naturally settle, so "steady state" for them is actually
            # "pinned at the velocity constraint", a different, uninformative
            # regime the mask correctly (if unhelpfully) excludes.
            # Fix: size each environment's hold torque from its OWN ground-truth
            # (b, alpha) -- legitimate here since this is a simulation dataset,
            # not a real-hardware protocol the frozen PINN must later replicate --
            # to target one safe steady-state velocity across the whole range.
            b_env = env._joint_damping.clamp_min(1e-3)
            alpha_env = env._actuator_effectiveness.clamp_min(0.2)
            target_v = args_cli.long_hold_target_velocity
            calibrated_scale = (target_v * b_env / (alpha_env * env.cfg.torque_limit)).clamp(0.01, 1.0)
            hold_action = torch.sign(torch.empty(n, 2, device=device).uniform_(-1.0, 1.0)) * calibrated_scale

            half = t_long_hold // 2
            for k in range(t_long_hold):
                step = t_transient + k
                if args_cli.long_hold_mode == "sequential":
                    drive = torch.zeros(1, 2, device=device)
                    drive[0, 0 if k < half else 1] = 1.0
                    hold_action_k = hold_action * drive
                else:
                    hold_action_k = hold_action
                q_before = env._wrapped_joint_pos().clone()
                qd_before = robot.data.joint_vel[:, jid].clone()
                tau_cmd = env.cfg.torque_limit * hold_action_k

                robot.set_joint_effort_target(tau_cmd * env._actuator_effectiveness, joint_ids=jid)
                robot.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)

                qd_after = robot.data.joint_vel[:, jid]
                qdd = (qd_after - qd_before) / env.physics_dt

                q_buf[:, step] = q_before
                qd_buf[:, step] = qd_before
                tau_buf[:, step] = tau_cmd
                qdd_buf[:, step] = qdd
                is_long_hold_buf[:, step] = True
                fast = (qd_after.abs() >= args_cli.max_speed) | (qd_before.abs() >= args_cli.max_speed)
                valid_buf[:, step] = ~fast.any(dim=-1)

    window = torch.cat([q_buf, qd_buf, tau_buf, qdd_buf], dim=-1).cpu()
    payload = {
        "window": window,
        "valid": valid_buf.cpu(),
        "is_long_hold": is_long_hold_buf.cpu(),
        "xi": xi.cpu(),
        "split": args_cli.split,
        "seed": args_cli.seed,
        "ranges": RANGES["extrap" if args_cli.split == "extrap" else "interp"],
        "torque_scale": args_cli.torque_scale,
        "long_hold_steps": t_long_hold,
        "long_hold_mode": args_cli.long_hold_mode,
        "long_hold_target_velocity": args_cli.long_hold_target_velocity,
        "step_dt": env.step_dt,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    suffix = f"_{args_cli.tag}" if args_cli.tag else ""
    path = os.path.join(DATA_DIR, f"pinn_{args_cli.split}{suffix}.pt")
    torch.save(payload, path)

    frac = valid_buf.float().mean().item()
    print(f"\n[pinn-data] split={args_cli.split}  envs={n}  steps={t}")
    print(f"  usable transitions: {frac:.1%} (rest rode the velocity limit)")
    print(f"  xi ranges observed:")
    for i, name in enumerate(("m1", "m2", "f1", "f2", "b1", "b2", "a1", "a2")):
        print(f"    {name}: [{xi[:, i].min().item():.4f}, {xi[:, i].max().item():.4f}]")
    print(f"  saved -> {path}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
