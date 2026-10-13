"""Scripted IK + PD controller — the precision ceiling of this benchmark.

Neither reward gets below ~32 mm median error, so before blaming PPO we need to
know what the environment itself permits: with 30 Hz control, +/-20 Nm and
+/-5 rad/s on a near-frictionless arm, is millimetre accuracy reachable at all?

This controller solves the 2-link inverse kinematics for the goal marker and
drives the joints with a PD law inside the same action interface the policy
uses (normalised actions -> 20 Nm), measured with the same frozen metric suite.
It is an upper bound on achievable precision, not a learned policy.

    env_isaaclab/bin/python pi_adr/scripts/pd_baseline.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--kp", type=float, default=12.0)
parser.add_argument("--kd", type=float, default=1.2)
parser.add_argument("--run_name", type=str, default="PD_reference")
parser.add_argument(
    "--wide_limits",
    action="store_true",
    help="widen joint position limits to +/-10 rad, to test whether the +/-pi limits trap the arm",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import pi_adr.tasks  # noqa: F401, E402
from pi_adr.evaluation.metrics import RolloutAccumulator  # noqa: E402
from pi_adr.tasks.two_dof_reach.two_dof_reach_env_cfg import (  # noqa: E402
    LINK_LENGTH_1,
    LINK_LENGTH_2,
    TORQUE_LIMIT,
    TwoDofReachEnvCfg,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def ik_two_link(target: torch.Tensor, elbow_sign: torch.Tensor) -> torch.Tensor:
    """Planar 2-link inverse kinematics; returns (N, 2) joint angles."""
    x, y = target[:, 0], target[:, 1]
    r2 = (x**2 + y**2).clamp(1e-6, (LINK_LENGTH_1 + LINK_LENGTH_2) ** 2 - 1e-6)
    cos_q2 = ((r2 - LINK_LENGTH_1**2 - LINK_LENGTH_2**2) / (2 * LINK_LENGTH_1 * LINK_LENGTH_2)).clamp(-1.0, 1.0)
    q2 = elbow_sign * torch.acos(cos_q2)
    q1 = torch.atan2(y, x) - torch.atan2(
        LINK_LENGTH_2 * torch.sin(q2), LINK_LENGTH_1 + LINK_LENGTH_2 * torch.cos(q2)
    )
    return torch.stack([torch.atan2(torch.sin(q1), torch.cos(q1)), q2], dim=-1)


def main() -> None:
    cfg = TwoDofReachEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    cfg.domain_rand.enabled = False

    env = gym.make("PIADR-TwoDofReach-Vanilla-v0", cfg=cfg).unwrapped
    device = env.device
    env.reset()

    if args_cli.wide_limits:
        wide = torch.tensor([[-10.0, 10.0], [-10.0, 10.0]], device=device)
        wide = wide.unsqueeze(0).expand(env.num_envs, -1, -1).clone()
        env._robot.write_joint_position_limit_to_sim(wide, joint_ids=env._joint_ids)
        print("[pd] joint position limits widened to +/-10 rad")

    accum = RolloutAccumulator(env.num_envs, env.step_dt, device)
    horizon = int(env.max_episode_length)
    stalled_at_limit = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    with torch.inference_mode():
        for _ in range(horizon - 2):
            q = env._robot.data.joint_pos[:, env._joint_ids]
            qd = env._robot.data.joint_vel[:, env._joint_ids]
            target_local = env._target_pos_w[:, :2] - env.scene.env_origins[:, :2]

            # Pick the elbow solution closest to the current configuration.
            cand = torch.stack([ik_two_link(target_local, torch.ones_like(q[:, 0])),
                                ik_two_link(target_local, -torch.ones_like(q[:, 0]))], dim=0)
            err = torch.atan2(torch.sin(cand - q), torch.cos(cand - q)).abs().sum(-1)
            q_star = torch.where((err[0] <= err[1]).unsqueeze(-1), cand[0], cand[1])

            angle_err = torch.atan2(torch.sin(q_star - q), torch.cos(q_star - q))
            tau = args_cli.kp * angle_err - args_cli.kd * qd
            action = (tau / TORQUE_LIMIT).clamp(-1.0, 1.0)

            env.step(action)
            accum.record(env.target_distance, env._robot.data.joint_vel[:, env._joint_ids], action)

    result = accum.finalize(
        args_cli.run_name,
        extra={
            "controller": f"IK + PD (kp={args_cli.kp}, kd={args_cli.kd})",
            "num_envs": args_cli.num_envs,
            "eval_seed": args_cli.seed,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "note": "scripted reference: precision ceiling of the environment, not a learned policy",
        },
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result.to_json(os.path.join(RESULTS_DIR, f"{args_cli.run_name}.json"))
    with open(os.path.join(RESULTS_DIR, "index.jsonl"), "a") as fh:
        fh.write(json.dumps({"run_name": result.run_name, "success_rate": result.success_rate,
                             "mean_terminal_error_m": result.mean_terminal_error_m,
                             "median_terminal_error_m": result.median_terminal_error_m,
                             "mean_joint_velocity_rad_s": result.mean_joint_velocity_rad_s,
                             "mean_control_effort": result.mean_control_effort,
                             **result.extra}) + "\n")

    final_q = env._robot.data.joint_pos[:, env._joint_ids]
    at_limit = (final_q.abs() > 3.10).any(dim=-1)
    failed = accum.terminal_distance > 0.05
    print("\n================ PD REFERENCE ================")
    print(
        f"  failures={failed.float().mean().item():.2%}   "
        f"at joint limit={at_limit.float().mean().item():.2%}   "
        f"failed AND at limit={(failed & at_limit).float().mean().item():.2%}"
    )
    print(result.summary_line())
    print(f"  precision curve: {result.success_rate}")
    print(f"  settling={result.mean_settling_time_s:.2f}s  max|qd|={result.max_joint_velocity_rad_s:.2f} rad/s")
    print("==============================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
