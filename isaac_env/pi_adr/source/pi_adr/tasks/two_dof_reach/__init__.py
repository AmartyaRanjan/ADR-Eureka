"""2-DOF planar reacher task registration."""

import gymnasium as gym

from . import agents
from .two_dof_reach_env import TwoDofReachEnv
from .two_dof_reach_env_cfg import TwoDofReachEnvCfg, TwoDofReachEurekaEnvCfg

gym.register(
    id="PIADR-TwoDofReach-Vanilla-v0",
    entry_point=f"{__name__}.two_dof_reach_env:TwoDofReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.two_dof_reach_env_cfg:TwoDofReachEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="PIADR-TwoDofReach-Eureka-v0",
    entry_point=f"{__name__}.two_dof_reach_env:TwoDofReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.two_dof_reach_env_cfg:TwoDofReachEurekaEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

__all__ = ["TwoDofReachEnv", "TwoDofReachEnvCfg", "TwoDofReachEurekaEnvCfg"]
