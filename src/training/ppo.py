"""Clipped PPO for the vectorized LEAP cube-rotation environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.config.models import PPOConfig
from src.training.leap_hand_env import LeapHandCubeRotEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(obs).squeeze(-1)
        return dist, value


@dataclass
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float


class PPOTrainer:
    def __init__(self, env: LeapHandCubeRotEnv, config: PPOConfig, device: str = "cpu"):
        self.env = env
        self.config = config
        self.device = torch.device(device)
        self.policy = ActorCritic(env.num_obs, env.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.obs = env.reset().to(self.device)

    def collect(self, horizon: int):
        n = self.env.num_envs
        obs_buf = torch.zeros((horizon, n, self.env.num_obs), device=self.device)
        act_buf = torch.zeros((horizon, n, self.env.num_actions), device=self.device)
        logp_buf = torch.zeros((horizon, n), device=self.device)
        rew_buf = torch.zeros((horizon, n), device=self.device)
        done_buf = torch.zeros((horizon, n), device=self.device)
        val_buf = torch.zeros((horizon, n), device=self.device)
        extras_acc = []
        with torch.no_grad():
            for t in range(horizon):
                dist, value = self.policy(self.obs)
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
                next_obs, reward, terminated, truncated, extras = self.env.step(action.cpu())
                done = torch.clamp(terminated + truncated, 0, 1).to(self.device)
                obs_buf[t] = self.obs
                act_buf[t] = action
                logp_buf[t] = logp
                rew_buf[t] = reward.to(self.device)
                done_buf[t] = done
                val_buf[t] = value
                extras_acc.append(extras)
                self.obs = next_obs.to(self.device)
        with torch.no_grad():
            _, last_value = self.policy(self.obs)
        advantages, returns = self._gae(rew_buf, val_buf, done_buf, last_value)
        batch = {
            "obs": obs_buf.reshape(-1, self.env.num_obs),
            "actions": act_buf.reshape(-1, self.env.num_actions),
            "logp": logp_buf.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": val_buf.reshape(-1),
            "rewards": rew_buf,
            "dones": done_buf,
            "extras": extras_acc,
        }
        return batch

    def _gae(self, rewards, values, dones, last_value):
        horizon = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(rewards.shape[1], device=rewards.device)
        next_value = last_value
        for t in reversed(range(horizon)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * mask - values[t]
            last_adv = delta + self.config.gamma * self.config.gae_lambda * mask * last_adv
            advantages[t] = last_adv
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self, batch) -> PPOUpdateStats:
        obs = batch["obs"]
        actions = batch["actions"]
        old_logp = batch["logp"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_size = obs.shape[0]
        minibatch = min(self.config.batch_size, batch_size)
        policy_losses, value_losses, entropies, kls, clips = [], [], [], [], []
        for _ in range(self.config.num_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch):
                idx = perm[start : start + minibatch]
                dist, values = self.policy(obs[idx])
                logp = dist.log_prob(actions[idx]).sum(-1)
                ratio = torch.exp(logp - old_logp[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (returns[idx] - values).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                approx_kl = (old_logp[idx] - logp).mean().detach()
                clip_frac = (torch.abs(ratio - 1.0) > self.config.clip_epsilon).float().mean().detach()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                kls.append(float(approx_kl.item()))
                clips.append(float(clip_frac.item()))
        var_y = batch["returns"].var()
        explained = 1.0 - (batch["returns"] - batch["values"]).var() / (var_y + 1e-8)
        return PPOUpdateStats(
            policy_loss=sum(policy_losses) / len(policy_losses),
            value_loss=sum(value_losses) / len(value_losses),
            entropy=sum(entropies) / len(entropies),
            approx_kl=sum(kls) / len(kls),
            clip_fraction=sum(clips) / len(clips),
            explained_variance=float(explained.item()),
        )

    def train_iteration(self) -> Dict[str, float]:
        import time

        t0 = time.perf_counter()
        batch = self.collect(self.config.rollout_horizon)
        stats = self.update(batch)
        extras = batch["extras"][-1]
        rewards = batch["rewards"]
        return {
            "episode_reward_mean": float(rewards.mean().item()),
            "episode_reward_std": float(rewards.std().item()),
            "oracle_reward_mean": extras.get("oracle_reward_mean", 0.0),
            "success_rate": extras.get("success_frac", 0.0),
            "drop_frac": extras.get("drop_frac", 0.0),
            "yaw_angvel_mean": extras.get("yaw_angvel_mean", 0.0),
            "episode_yaw_mean": extras.get("episode_yaw_mean", 0.0),
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy": stats.entropy,
            "approx_kl": stats.approx_kl,
            "clip_fraction": stats.clip_fraction,
            "explained_variance": stats.explained_variance,
            "learning_rate": self.config.learning_rate,
            "fps": (self.config.rollout_horizon * self.env.num_envs) / max(time.perf_counter() - t0, 1e-6),
            "step_time_s": time.perf_counter() - t0,
        }

    def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        env = self.env
        returns = []
        oracle_returns = []
        lengths = []
        yaws = []
        successes = 0
        drops = 0
        crashes = 0
        episodes = 0
        obs = env.reset()
        ep_ret = torch.zeros(env.num_envs)
        ep_oracle = torch.zeros(env.num_envs)
        ep_len = torch.zeros(env.num_envs)
        while episodes < num_episodes:
            with torch.no_grad():
                dist, _ = self.policy(obs.to(self.device))
                action = dist.mean
            try:
                obs, reward, terminated, truncated, extras = env.step(action.cpu())
            except Exception:
                crashes += 1
                obs = env.reset()
                continue
            ep_ret += reward.cpu()
            ep_oracle += extras.get("oracle_reward_mean", 0.0)
            ep_len += 1
            done = (terminated + truncated) > 0
            for i in range(env.num_envs):
                if done[i] and episodes < num_episodes:
                    returns.append(float(ep_ret[i]))
                    oracle_returns.append(float(ep_oracle[i]))
                    lengths.append(float(ep_len[i]))
                    yaws.append(float(env.episode_yaw[i].item()) if env.episode_yaw[i].numel() else 0.0)
                    successes += 1 if extras.get("success_frac", 0) > 0 and terminated[i] else 0
                    drops += 1 if extras.get("drop_frac", 0) > 0 and terminated[i] else 0
                    ep_ret[i] = 0
                    ep_oracle[i] = 0
                    ep_len[i] = 0
                    episodes += 1
        import math

        def _mean(xs):
            return sum(xs) / len(xs) if xs else 0.0

        def _std(xs):
            if len(xs) < 2:
                return 0.0
            m = _mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

        return {
            "num_episodes": episodes,
            "mean_return": _mean(returns),
            "std_return": _std(returns),
            "mean_oracle_return": _mean(oracle_returns),
            "success_rate": successes / max(episodes, 1),
            "drop_rate": drops / max(episodes, 1),
            "mean_episode_length": _mean(lengths),
            "mean_episode_yaw": _mean(yaws),
            "mean_yaw_angvel": extras.get("yaw_angvel_mean", 0.0),
            "crashes": crashes,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.optimizer.load_state_dict(data["optimizer"])
