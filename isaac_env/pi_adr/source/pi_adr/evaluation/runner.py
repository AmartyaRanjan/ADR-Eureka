"""Shared env + policy construction so every evaluation path is identical.

eval.py, run_rapp.py and the adaptive-DR loop all go through here — if the
rollout protocol differed between them the numbers would not be comparable.
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

from .metrics import RolloutAccumulator

NOMINAL_PHYSICS = {
    "friction": 0.0,
    "damping_scale": 1.0,
    "effectiveness": 1.0,
    "mass_scale": 1.0,
}


def make_env_and_agent(
    task: str,
    checkpoint: str,
    num_envs: int,
    seed: int,
    device: str | None = None,
):
    """Build the env and restore an rl_games player from a checkpoint."""
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")

    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    if device is not None:
        env_cfg.sim.device = device

    agent_cfg["params"]["config"]["num_actors"] = num_envs
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()

    return env, base_env, agent, env_cfg


def set_domain_rand(base_env, **overrides) -> None:
    """Set the DR distribution in place; unspecified parameters go to nominal.

    Each override is either a scalar (degenerate distribution — every env gets
    exactly that value, which is what a RAPP one-parameter sweep needs) or a
    (low, high) pair.
    """
    dr = base_env.cfg.domain_rand
    values = {**NOMINAL_PHYSICS, **overrides}
    ranges = {k: (v, v) if isinstance(v, (int, float)) else tuple(v) for k, v in values.items()}

    dr.joint_friction_range = ranges["friction"]
    dr.joint_damping_scale_range = ranges["damping_scale"]
    dr.actuator_effectiveness_range = ranges["effectiveness"]
    dr.link_mass_scale_range = ranges["mass_scale"]
    dr.enabled = True


def rollout(env, base_env, agent, run_name: str, extra: dict | None = None):
    """One full deterministic episode per environment; returns an EvalResult.

    Steps horizon-2 times on purpose. DirectRLEnv fires time_out when
    episode_length_buf >= max_episode_length-1 and resets *inside* that same
    step, so a rollout of horizon-1 steps would record the freshly reset pose
    as the terminal error. horizon-2 leaves the last sample at t=9.93 s of the
    10 s episode, with the policy long settled.
    """
    horizon = int(base_env.max_episode_length)
    accum = RolloutAccumulator(base_env.num_envs, base_env.step_dt, base_env.device)

    # reset() must sit inside inference_mode too. Stepping under inference_mode
    # marks the simulator's buffers as inference tensors, and a later reset from
    # outside that context then fails trying to write joint_acc in place — which
    # only shows up on the *second* rollout in a process, e.g. a RAPP sweep.
    with torch.inference_mode():
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

        for _ in range(horizon - 2):
            actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, _, _, _ = env.step(actions)
            accum.record(
                base_env.target_distance,
                base_env._robot.data.joint_vel[:, base_env._joint_ids],
                actions,
            )

    return accum.finalize(run_name, extra=extra), accum
