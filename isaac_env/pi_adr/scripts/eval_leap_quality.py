"""Evaluate a LEAP Hand reorientation policy.

Track B's counterpart to scripts/eval.py. The task is in-hand cube rotation, so
terminal Cartesian error is meaningless here; the comparable quantities are the
task's own success measure (DrEureka reports rotations per 20 s) plus the same
motion/effort quantities we record for the 2-DOF arm, so both tracks are
described by one vocabulary: how well it does the task, how fast it moves, how
much control it spends.

    env_isaaclab/bin/python pi_adr/scripts/eval_leap.py --headless \
        --checkpoint <path.pth> --run_name LEAP_500
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-Reorient-Cube-Leap")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--steps", type=int, default=600, help="control steps to roll out")
parser.add_argument("--adr_increments", type=int, default=0,
                    help="FROZEN ADR difficulty used for evaluation (0 = nominal, 25 = full native DR range); ADR never advances during eval")
parser.add_argument("--note", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import LEAP_Isaaclab.tasks  # noqa: F401, E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # Identical, frozen difficulty for every policy regardless of how it was trained.
    env_cfg.enable_adr = True
    env_cfg.starting_adr_increments = args_cli.adr_increments
    env_cfg.min_rot_adr_coeff = 1.0e9

    agent_cfg["params"]["config"]["num_actors"] = args_cli.num_envs
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args_cli.checkpoint)
    agent.reset()

    z_steps = base_env.cfg.z_rotation_steps
    dt = base_env.cfg.sim.dt * base_env.cfg.decimation

    vel_sum = torch.zeros((), device=base_env.device)
    effort_sum = torch.zeros((), device=base_env.device)
    abs_action_sum = torch.zeros((), device=base_env.device)
    success_total = torch.zeros((), device=base_env.device)
    n_rec = 0

    with torch.inference_mode():
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

        prev_successes = base_env.successes.clone()
        falls_total = torch.zeros((), device=base_env.device); timeouts_total = torch.zeros((), device=base_env.device)
        for _ in range(args_cli.steps):
            actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            falls_total += base_env.reset_terminated.float().sum(); timeouts_total += base_env.reset_time_outs.float().sum()

            # successes resets to 0 on episode reset, so only count increases.
            delta = (base_env.successes - prev_successes).clamp_min(0.0)
            success_total += delta.sum()
            prev_successes = base_env.successes.clone()

            vel_sum += base_env.hand.data.joint_vel.abs().mean()
            effort_sum += actions.pow(2).mean()
            abs_action_sum += actions.abs().mean()
            n_rec += 1

            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for s in agent.states:
                    s[:, dones, :] = 0.0

    elapsed_s = args_cli.steps * dt
    rotations = success_total.item() / z_steps
    per_env_rot = rotations / base_env.num_envs

    result = {
        "run_name": args_cli.run_name,
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "num_envs": base_env.num_envs,
        "eval_seed": args_cli.seed,
        "eval_adr_increments_frozen": args_cli.adr_increments,
        "control_steps": args_cli.steps,
        "sim_seconds_per_env": round(elapsed_s, 2),
        "z_rotation_steps": z_steps,
        "total_goal_reaches": success_total.item(),
        "rotations_per_env": per_env_rot,
        "falls_per_20s_per_env": falls_total.item() / base_env.num_envs * (20.0 / elapsed_s),
        "timeouts_per_20s_per_env": timeouts_total.item() / base_env.num_envs * (20.0 / elapsed_s),
        "rotations_per_20s_per_env": per_env_rot * (20.0 / elapsed_s),
        "mean_joint_velocity_rad_s": (vel_sum / n_rec).item(),
        "mean_control_effort_per_joint": (effort_sum / n_rec).item(),
        "mean_abs_action": (abs_action_sum / n_rec).item(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": args_cli.note,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{args_cli.run_name}.json"), "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    with open(os.path.join(RESULTS_DIR, "index.jsonl"), "a") as fh:
        fh.write(json.dumps(result) + "\n")

    print("\n================ LEAP EVAL ================")
    print(f"  rotations per 20 s per hand : {result['rotations_per_20s_per_env']:.3f}")
    print(f"  goal reaches (all envs)     : {result['total_goal_reaches']:.0f} over {elapsed_s:.1f}s each")
    print(f"  mean joint velocity         : {result['mean_joint_velocity_rad_s']:.3f} rad/s")
    print(f"  mean control effort/joint   : {result['mean_control_effort_per_joint']:.3f}")
    print("==========================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
