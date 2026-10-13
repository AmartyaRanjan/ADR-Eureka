"""2-DOF planar reacher: move the tip of link 2 onto a randomly placed goal marker."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers

from .two_dof_reach_env_cfg import TwoDofReachEnvCfg, arm_articulation_cfg, build_arm_usd


class TwoDofReachEnv(DirectRLEnv):
    cfg: TwoDofReachEnvCfg

    def __init__(self, cfg: TwoDofReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, _ = self._robot.find_joints(["joint_1", "joint_2"])
        self._ee_body_id, _ = self._robot.find_bodies(["ee_link"])

        self._actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_pos_w[:, 2] = self.cfg.init_height

        # Per-env physics; identity until domain randomisation is switched on.
        self._actuator_effectiveness = torch.ones(self.num_envs, 2, device=self.device)
        self._joint_friction = torch.zeros(self.num_envs, 2, device=self.device)
        self._joint_damping = torch.full(
            (self.num_envs, 2), self.cfg.nominal_joint_damping, device=self.device
        )

        self._nominal_masses = self._robot.data.default_mass.clone()
        self._link_body_ids = [self._robot.body_names.index(n) for n in ("link_1", "link_2")]
        self._link_masses = self._nominal_masses[:, self._link_body_ids].clone().to(self.device)

    def _setup_scene(self):
        usd_path = build_arm_usd()
        self._robot = Articulation(arm_articulation_cfg(usd_path))

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        self._goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        torque = self.cfg.torque_limit * self._actions * self._actuator_effectiveness
        self._robot.set_joint_effort_target(torque, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        # Joints are continuous, so raw angles drift unbounded once the arm
        # winds past +/-pi. Wrapping keeps the observation distribution bounded
        # and identical in layout to the report's 8-D vector.
        q = self._wrapped_joint_pos()
        qd = self._robot.data.joint_vel[:, self._joint_ids]
        target_local = self._target_pos_w[:, :2] - self.scene.env_origins[:, :2]
        ee_local = self._ee_pos_local()
        obs = torch.cat([q, qd, target_local, target_local - ee_local], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        d = self._target_distance()
        if self.cfg.reward_variant == "vanilla":
            # Coefficients are cfg fields (default = the report's exact equation 7:
            # -d + 5*I(d<0.05) - 0.005*||a||^2) so a reward-incentive-strength variant
            # can be run via a hydra override, without a new reward_variant string.
            c = self.cfg.vanilla_reward
            reward = (
                -c.distance_weight * d
                + c.success_bonus * (d < self.cfg.success_radius).float()
                - c.action_penalty_coef * self._actions.pow(2).sum(-1)
            )
        elif self.cfg.reward_variant == "eureka":
            qd = self._robot.data.joint_vel[:, self._joint_ids]
            reward = (
                torch.exp(-10.0 * d)
                + 100.0 * (d < self.cfg.success_radius).float()
                - 0.01 * qd.pow(2).sum(-1)
                - 0.1
            )
        else:
            raise ValueError(f"unknown reward_variant: {self.cfg.reward_variant}")
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        joint_pos = torch.empty(num_reset, 2, device=self.device).uniform_(-torch.pi, torch.pi)
        joint_vel = torch.zeros(num_reset, 2, device=self.device)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self._joint_ids, env_ids=env_ids)

        r = torch.empty(num_reset, device=self.device).uniform_(*self.cfg.target_radius_range)
        theta = torch.empty(num_reset, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        self._target_pos_w[env_ids, 0] = self.scene.env_origins[env_ids, 0] + r * torch.cos(theta)
        self._target_pos_w[env_ids, 1] = self.scene.env_origins[env_ids, 1] + r * torch.sin(theta)
        self._target_pos_w[env_ids, 2] = self.cfg.init_height

        if self.cfg.domain_rand.enabled:
            self._randomize_physics(env_ids)

        self._actions[env_ids] = 0.0
        self._goal_markers.visualize(translations=self._target_pos_w)

    def _randomize_physics(self, env_ids: Sequence[int]):
        dr = self.cfg.domain_rand
        n = len(env_ids)

        friction = torch.empty(n, 2, device=self.device).uniform_(*dr.joint_friction_range)
        damping = self.cfg.nominal_joint_damping * torch.empty(n, 2, device=self.device).uniform_(
            *dr.joint_damping_scale_range
        )
        effectiveness = torch.empty(n, 2, device=self.device).uniform_(*dr.actuator_effectiveness_range)

        self._joint_friction[env_ids] = friction
        self._joint_damping[env_ids] = damping
        self._actuator_effectiveness[env_ids] = effectiveness

        self._robot.write_joint_friction_coefficient_to_sim(friction, joint_ids=self._joint_ids, env_ids=env_ids)
        self._robot.write_joint_damping_to_sim(damping, joint_ids=self._joint_ids, env_ids=env_ids)

        # Always written, including at scale 1.0. Skipping the write when the
        # range is (1.0, 1.0) leaves whatever masses a previous episode or sweep
        # set, which silently corrupted the RAPP mass sweep: "scale 1.0" was
        # still carrying the preceding candidate's masses.
        cpu_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
        scale = torch.empty(n, self._nominal_masses.shape[1]).uniform_(*dr.link_mass_scale_range)
        masses = self._nominal_masses[cpu_ids] * scale
        self._robot.root_physx_view.set_masses(masses, cpu_ids)
        self._link_masses[env_ids] = masses[:, self._link_body_ids].to(self.device)

    def _wrapped_joint_pos(self) -> torch.Tensor:
        q = self._robot.data.joint_pos[:, self._joint_ids]
        return torch.atan2(torch.sin(q), torch.cos(q))

    def _ee_pos_local(self) -> torch.Tensor:
        return self._robot.data.body_pos_w[:, self._ee_body_id[0], :2] - self.scene.env_origins[:, :2]

    def _target_distance(self) -> torch.Tensor:
        target_local = self._target_pos_w[:, :2] - self.scene.env_origins[:, :2]
        return torch.norm(target_local - self._ee_pos_local(), dim=-1)

    @property
    def target_distance(self) -> torch.Tensor:
        return self._target_distance()

    @property
    def physics_params(self) -> torch.Tensor:
        """(N, 6) per-env physics vector: friction x2, damping x2, effectiveness x2."""
        return torch.cat([self._joint_friction, self._joint_damping, self._actuator_effectiveness], dim=-1)

    @property
    def physics_xi(self) -> torch.Tensor:
        """(N, 8) ground-truth xi = [m1, m2, f1, f2, b1, b2, a1, a2] for the PINN."""
        return torch.cat(
            [self._link_masses, self._joint_friction, self._joint_damping, self._actuator_effectiveness],
            dim=-1,
        )

    @property
    def commanded_torque(self) -> torch.Tensor:
        """Torque the policy asked for, before actuator effectiveness is applied."""
        return self.cfg.torque_limit * self._actions
