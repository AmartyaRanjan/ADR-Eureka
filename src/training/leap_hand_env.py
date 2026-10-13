"""Vectorized LEAP-hand in-palm cube rotation environment.

Isaac Gym / Isaac Lab are not required. Dynamics follow the LEAP Hand Rot task
from https://github.com/leap-hand/LEAP_Hand_Sim : 16-DoF hand, cube in the palm,
rotate about -z, drop if the cube falls below the height threshold.

Joint limits are taken from ``assets/leap_hand/robot.urdf``. Canonical grasp pose
is the one published in ``leapsim/cfg/task/LeapHandRot.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for the LEAP Eureka environment.") from exc


# Four fingers (index/middle/ring/thumb) x 4 joints. Order matches LEAP URDF.
LEAP_JOINT_LOWER = np.array(
    [
        -1.047, -0.314, -0.506, -0.366,
        -1.047, -0.314, -0.506, -0.366,
        -1.047, -0.314, -0.506, -0.366,
        -0.349, -0.47, -1.20, -1.34,
    ],
    dtype=np.float32,
)
LEAP_JOINT_UPPER = np.array(
    [
        1.047, 2.23, 1.885, 2.042,
        1.047, 2.23, 1.885, 2.042,
        1.047, 2.23, 1.885, 2.042,
        2.094, 2.443, 1.90, 1.88,
    ],
    dtype=np.float32,
)
CANONICAL_POSE = np.array(
    [
        0.1458, -1.0470, 1.7247, -0.2187,
        1.2985, 1.6139, 0.9757, 1.0728,
        -0.1054, 0.0290, 1.6039, 0.1227,
        -0.1084, 0.9652, 1.7317, 0.1071,
    ],
    dtype=np.float32,
)


def _clamp(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(x, lo), hi)


@dataclass
class StepInfo:
    extras: Dict[str, float]
    terminated: torch.Tensor
    truncated: torch.Tensor


class LeapHandCubeRotEnv:
    """Direct-workflow-style env whose generated Eureka reward is ``_get_rewards_eureka``."""

    num_actions = 16
    num_obs = 44  # q(16) + qd(16) + contacts(5) + cube_quat(4) + cube_pos(3)

    def __init__(
        self,
        num_envs: int = 32,
        max_episode_steps: int = 500,
        device: str = "cpu",
        seed: int = 42,
        dt: float = 0.05,
        drop_threshold: float = 0.4,
        rotation_tolerance_deg: float = 5.0,
    ):
        self.num_envs = num_envs
        self.max_episode_length = max_episode_steps
        self.max_episode_length_s = max_episode_steps * dt
        self.device = torch.device(device)
        self.dt = dt
        self.reset_z_threshold = drop_threshold
        self.rotation_tolerance_rad = np.deg2rad(rotation_tolerance_deg)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)

        self.leap_hand_dof_lower_limits = torch.as_tensor(LEAP_JOINT_LOWER, device=self.device)
        self.leap_hand_dof_upper_limits = torch.as_tensor(LEAP_JOINT_UPPER, device=self.device)
        self.canonical_pose = torch.as_tensor(CANONICAL_POSE, device=self.device)

        n = num_envs
        self.leap_hand_dof_pos = torch.zeros((n, 16), device=self.device)
        self.leap_hand_dof_vel = torch.zeros((n, 16), device=self.device)
        self.init_pose_buf = torch.zeros((n, 16), device=self.device)
        self.actions = torch.zeros((n, 16), device=self.device)
        self.torques = torch.zeros((n, 16), device=self.device)
        self.contact_forces = torch.zeros((n, 5), device=self.device)
        self.object_pos = torch.zeros((n, 3), device=self.device)
        self.object_linvel = torch.zeros((n, 3), device=self.device)
        self.object_angvel = torch.zeros((n, 3), device=self.device)
        self.object_angvel_finite_diff = torch.zeros((n, 3), device=self.device)
        self.object_rot = torch.zeros((n, 4), device=self.device)
        self.object_yaw = torch.zeros((n,), device=self.device)
        self.episode_yaw = torch.zeros((n,), device=self.device)
        self.progress_buf = torch.zeros((n,), device=self.device, dtype=torch.long)
        self.reset_buf = torch.zeros((n,), device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((n,), device=self.device, dtype=torch.long)
        self._eureka_episode_sums: Dict[str, torch.Tensor] = {}
        self.extras: Dict[str, object] = {"log": {}}
        self._reward_fn: Optional[Callable] = None
        self.reset()

    def set_eureka_reward(self, reward_fn: Optional[Callable]) -> None:
        self._reward_fn = reward_fn

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Observation used by Eureka prompts and the policy.

        Returns a policy observation concatenating:
        - LEAP joint positions (16)
        - LEAP joint velocities (16)
        - fingertip + palm contact magnitudes (5)
        - cube orientation quaternion wxyz (4)
        - cube position xyz (3)
        """
        obs = torch.cat(
            [
                self.leap_hand_dof_pos,
                self.leap_hand_dof_vel,
                self.contact_forces,
                self.object_rot,
                self.object_pos,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards_oracle(self) -> torch.Tensor:
        """Oracle reward aligned with LeapHandRot (yaw rotation, penalties, drop)."""
        yaw_angvel = self.object_angvel[:, 2]
        rotate = torch.clamp(-yaw_angvel, -0.25, 0.25)
        linvel_pen = -0.3 * (self.object_linvel ** 2).sum(dim=-1)
        pose_pen = -0.1 * ((self.leap_hand_dof_pos - self.init_pose_buf) ** 2).sum(dim=-1)
        torque_pen = -0.1 * (self.torques ** 2).sum(dim=-1)
        work_pen = -1.0 * ((self.torques * self.leap_hand_dof_vel).sum(dim=-1) ** 2)
        fallen = -10.0 * (self.object_pos[:, 2] < self.reset_z_threshold).float()
        return rotate + linvel_pen + pose_pen + torque_pen + work_pen + fallen

    def _get_rewards_eureka(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._reward_fn is None:
            reward = self._get_rewards_oracle()
            return reward, {"oracle": reward}
        return self._reward_fn(self)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = env_ids.numel()
        noise = (torch.rand((n, 16), generator=self._rng) - 0.5) * 0.05
        pose = _clamp(self.canonical_pose.unsqueeze(0) + noise.to(self.device), self.leap_hand_dof_lower_limits, self.leap_hand_dof_upper_limits)
        self.leap_hand_dof_pos[env_ids] = pose
        self.leap_hand_dof_vel[env_ids] = 0
        self.init_pose_buf[env_ids] = pose
        self.actions[env_ids] = 0
        self.torques[env_ids] = 0
        self.contact_forces[env_ids] = 0.4
        self.object_pos[env_ids] = torch.tensor([-0.03, 0.04, 0.57], device=self.device)
        self.object_pos[env_ids, 0] += 0.002 * (torch.rand((n,), generator=self._rng).to(self.device) - 0.5)
        self.object_pos[env_ids, 1] += 0.002 * (torch.rand((n,), generator=self._rng).to(self.device) - 0.5)
        self.object_linvel[env_ids] = 0
        self.object_angvel[env_ids] = 0
        self.object_angvel_finite_diff[env_ids] = 0
        self.object_rot[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.object_yaw[env_ids] = 0
        self.episode_yaw[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        return self._get_observations()["policy"]

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions
        mid = 0.5 * (self.leap_hand_dof_upper_limits + self.leap_hand_dof_lower_limits)
        half = 0.5 * (self.leap_hand_dof_upper_limits - self.leap_hand_dof_lower_limits)
        targets = mid + actions * half
        kp, kd = 3.0, 0.1
        acc = kp * (targets - self.leap_hand_dof_pos) - kd * self.leap_hand_dof_vel
        self.leap_hand_dof_vel = self.leap_hand_dof_vel + self.dt * acc
        self.leap_hand_dof_pos = _clamp(
            self.leap_hand_dof_pos + self.dt * self.leap_hand_dof_vel,
            self.leap_hand_dof_lower_limits,
            self.leap_hand_dof_upper_limits,
        )
        self.torques = kp * (targets - self.leap_hand_dof_pos) - kd * self.leap_hand_dof_vel

        # Coordinated finger motion produces negative yaw (LEAP rot_axis z = -1).
        finger_coupling = self.leap_hand_dof_vel.view(self.num_envs, 4, 4).mean(dim=-1)
        yaw_cmd = -0.35 * (finger_coupling[:, 0] - finger_coupling[:, 2] + 0.5 * finger_coupling[:, 3])
        grasp = torch.exp(-((self.leap_hand_dof_pos - self.init_pose_buf) ** 2).mean(dim=-1) * 4.0)
        self.object_angvel[:, 2] = yaw_cmd * grasp
        self.object_angvel[:, 0] = 0.02 * self.leap_hand_dof_vel[:, 0]
        self.object_angvel[:, 1] = 0.02 * self.leap_hand_dof_vel[:, 4]
        self.object_angvel_finite_diff = self.object_angvel.clone()
        self.object_yaw = self.object_yaw + self.object_angvel[:, 2] * self.dt
        self.episode_yaw = self.episode_yaw + (-self.object_angvel[:, 2]) * self.dt

        slip = (1.0 - grasp) * 0.01
        self.object_linvel[:, 0] = 0.05 * (self.leap_hand_dof_pos[:, 0] - self.init_pose_buf[:, 0])
        self.object_linvel[:, 1] = 0.05 * (self.leap_hand_dof_pos[:, 4] - self.init_pose_buf[:, 4])
        self.object_linvel[:, 2] = -slip
        self.object_pos = self.object_pos + self.dt * self.object_linvel
        yaw = self.object_yaw
        self.object_rot = torch.stack(
            [torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)],
            dim=-1,
        )
        self.contact_forces = torch.clamp(grasp.unsqueeze(-1) * 0.8 + 0.05 * torch.rand((self.num_envs, 5), generator=self._rng).to(self.device), 0, 2)

        rewards, reward_parts = self._get_rewards_eureka()
        oracle = self._get_rewards_oracle()
        self.progress_buf += 1
        self.episode_length_buf += 1
        dropped = self.object_pos[:, 2] < self.reset_z_threshold
        success = self.episode_yaw >= (np.pi / 2 - self.rotation_tolerance_rad)
        truncated = self.progress_buf >= self.max_episode_length
        terminated = dropped | success
        done = terminated | truncated

        extras = {
            "oracle_reward_mean": float(oracle.mean().item()),
            "eureka_reward_mean": float(rewards.mean().item()),
            "yaw_angvel_mean": float(self.object_angvel[:, 2].mean().item()),
            "episode_yaw_mean": float(self.episode_yaw.mean().item()),
            "drop_frac": float(dropped.float().mean().item()),
            "success_frac": float(success.float().mean().item()),
            "grasp_quality_mean": float(grasp.mean().item()),
            "torque_l2_mean": float((self.torques ** 2).sum(dim=-1).mean().item()),
            "reward_parts": {k: float(v.mean().item()) for k, v in reward_parts.items()},
        }
        if done.any():
            finished = torch.nonzero(done, as_tuple=False).squeeze(-1)
            self.reset(finished)
        obs = self._get_observations()["policy"]
        return obs, rewards, terminated.float(), truncated.float(), extras
