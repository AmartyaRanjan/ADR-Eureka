# Stage-2 Code & Mathematics Compilation

Everything needed to write the Stage-2 report extension, in one file: full source of every
component, the exact configuration used, the exact result JSONs produced, and the code-to-math
correspondence for each object the roadmap defines. This is the **authoritative** version of the
mathematics — where it differs from the original roadmap's generic equations, the code below is
what actually ran.

Companion files in this same directory: `EXPERIMENT_LOG.md` (chronological defect/fix log),
`RESULTS_2026-09-16.md` (narrative results writeup), `results/*.json` (raw numbers).

---

## Table of contents

1. [The eight mathematical objects](#1-the-eight-mathematical-objects)
2. [Environment class](#2-environment-class) — `two_dof_reach_env.py`
3. [Environment config, robot, URDF](#3-environment-config-robot-urdf)
4. [Domain randomisation](#4-domain-randomisation) (embedded in §2/§3)
5. [RAPP](#5-rapp) — `run_rapp.py` + measured envelope
6. [Physics-performance map](#6-physics-performance-map)
7. [Shifted-DR experiment — exact configs used](#7-shifted-dr-experiment)
8. [PPO configuration](#8-ppo-configuration) — `rl_games_ppo_cfg.yaml`
9. [Reward code](#9-reward-code) (embedded in §2)
10. [Evaluation / metrics code](#10-evaluation--metrics-code)
11. [IK + PD baseline](#11-ik--pd-baseline)
12. [PINN — full pipeline](#12-pinn--full-pipeline)
13. [PINN mathematical model](#13-pinn-mathematical-model)
14. [PINN loss](#14-pinn-loss)
15. [PINN uncertainty](#15-pinn-uncertainty)
16. [All result files](#16-all-result-files)
17. [Environment validation scripts](#17-environment-validation-scripts)
18. [Versions / environment](#18-versions--environment)
19. [What adaptive DR would need (not built)](#19-what-adaptive-dr-would-need-not-built)

---

## 1. The eight mathematical objects

### 1. State

```
s_t = [q1, q2, q̇1, q̇2, x_t, y_t, x_t − x_ee, y_t − y_ee]  ∈ ℝ⁸
```
Code: `_get_observations()` in §2. `q` is wrapped to `[-π, π]` via `atan2(sin q, cos q)` before
being placed in the observation (joints are physically continuous — see §3).

### 2. Action → torque

```
a_t ∈ [-1, 1]²  (policy output, pre-clamped)
τ_cmd = 20 · a_t                          (commanded torque, Nm)
τ_applied = α ⊙ τ_cmd = 20 · α ⊙ a_t       (torque that actually reaches the joint)
```
`α = [α1, α2]` is per-joint actuator effectiveness (identity `[1,1]` unless DR is enabled).
Code: `_pre_physics_step` clamps `a_t`; `_apply_action` multiplies by `α` and calls
`set_joint_effort_target`.

### 3. Robot dynamics

```
M(q; ξ) q̈ + C(q, q̇; ξ) + F(q̇; ξ) = α ⊙ τ_cmd
```
with `g(q) ≡ 0` (planar, joints about world Z — gravity acts along the joint axis and produces
no joint torque; verified empirically, see §17 check C). Exact matrix forms in §13.

### 4. Randomised physics vector

```
ξ = [m1, m2, f1, f2, b1, b2, α1, α2] ∈ ℝ⁸
```
`m` link masses (kg), `f` Coulomb joint friction (N·m), `b` viscous joint damping (N·m·s/rad),
`α` actuator effectiveness (1.0 = healthy). Sampled **once per episode, per environment, at
reset** — never mid-episode, never shared across environments. Code: `_randomize_physics` in §2;
exposed as `env.physics_xi` (8-D, ground truth) and `env.physics_params` (6-D, excludes mass —
historical, kept for backward-compat with early eval code).

### 5. RAPP feasibility

```
S(π, ξ) = P(d_T < 0.05 | π, ξ)        (measured over N=1024–2048 episodes per candidate)
C_RAPP  = ∏_i [l_i, h_i],  l_i = min{r : S(π, ξ_r) ≥ α},  h_i = max{r : S(π, ξ_r) ≥ α}
```
with `α = 0.9` in the run that produced the measured envelope (§5). One parameter swept at a
time, all others held at nominal (friction=0, damping_scale=1, effectiveness=1, mass_scale=1).

### 6. PINN mapping

```
f_φ : window of (q, q̇, τ_cmd, q̈) transitions  →  (ξ̂, Σ̂)
```
Implemented as a **set encoder** (not a sequence model — the EOM is instantaneous, so transitions
are pooled, not ordered) over a window of `T=64` transitions, ensembled over `K=5` members for
uncertainty. Exact architecture in §12–13.

### 7. PINN loss

```
L = w_data · L_data + w_phys · L_phys + w_sup · L_sup
L_data = E[(q̈_pred − q̈)²]                                    (forward-dynamics fit)
L_phys = E[(M(q;ξ̂)q̈ + C(q,q̇;ξ̂) + F(q̇;ξ̂) − α̂⊙τ_cmd)²]        (EOM residual)
L_sup  = E[ (ξ̂_norm − ξ_norm)² · exp(−log σ²) + log σ² ]      (NLL against ground truth ξ)
```
with `w_data=1.0, w_phys=0.01, w_sup=1.0`. Exact code in §14.

### 8. Adaptive DR update

**Does not exist yet — deliberately.** The roadmap's own immediate checklist (item 9: *"Do not
build the adaptive controller yet"*) was followed literally. See §19 for what the measured
evidence says it would need to look like.

---

## 2. Environment class

`source/pi_adr/tasks/two_dof_reach/two_dof_reach_env.py` — full file.

```python
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
            reward = -d + 5.0 * (d < self.cfg.success_radius).float() - 0.005 * self._actions.pow(2).sum(-1)
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
```

**Mapping to the state chain** `s_t → π_θ(a_t|s_t) → a_t → τ_t → q_{t+1} → s_{t+1}`:
`_get_observations` builds `s_t`; the rl_games policy (loaded externally, see §8) produces `a_t`;
`_pre_physics_step`/`_apply_action` turn it into `τ_t = 20·α·a_t` and write it to PhysX;
Isaac Lab's `DirectRLEnv.step()` advances PhysX by `decimation=4` physics substeps
(`q_t → q_{t+1}`); `_get_observations` is called again for `s_{t+1}`.

---

## 3. Environment config, robot, URDF

`source/pi_adr/tasks/two_dof_reach/two_dof_reach_env_cfg.py` — full file.

```python
"""Config for the 2-DOF planar reacher (goal-marker assignment task).

All physical and task constants are taken from the stage-1 technical report so
that metrics stay comparable with the documented vanilla/Eureka baselines.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

_ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
URDF_PATH = os.path.join(_ASSET_DIR, "two_dof_arm.urdf")
USD_DIR = os.path.join(_ASSET_DIR, "usd")
USD_NAME = "two_dof_arm.usd"

LINK_LENGTH_1 = 0.5
LINK_LENGTH_2 = 0.5
TORQUE_LIMIT = 20.0
VELOCITY_LIMIT = 5.0
NOMINAL_JOINT_DAMPING = 0.05
SUCCESS_RADIUS = 0.05


def build_arm_usd(force: bool = False) -> str:
    """Convert the URDF to USD once and return the USD path.

    Must be called after the Isaac Sim app is running (importing the converter
    pulls in omni modules). merge_fixed_joints is off so the `ee_link` frame
    survives as a body we can read the tip position from.

    The cached USD short-circuits the converter: this Isaac Sim build's URDF
    importer needs libxml2.so.2/libicu*.so.73, which Ubuntu 26.04 does not
    ship, so conversion only works under scripts/convert_asset.sh. Training and
    evaluation then need no such shim.
    """
    cached = os.path.join(USD_DIR, USD_NAME)
    if not force and os.path.isfile(cached):
        return cached

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=URDF_PATH,
        usd_dir=USD_DIR,
        usd_file_name=USD_NAME,
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=force,
        collider_type="convex_hull",
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    return UrdfConverter(cfg).usd_path


def arm_articulation_cfg(usd_path: str) -> ArticulationCfg:
    """Torque-controlled arm: zero drive gains, ±20 Nm, ±5 rad/s."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={"joint_1": 0.0, "joint_2": 0.0},
            joint_vel={"joint_1": 0.0, "joint_2": 0.0},
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=["joint_1", "joint_2"],
                effort_limit_sim=TORQUE_LIMIT,
                velocity_limit_sim=VELOCITY_LIMIT,
                # Pure torque control: no position gain. Damping is the report's
                # nominal joint damping, and must be non-zero so that nominal
                # physics equals domain randomisation at scale 1.0.
                stiffness=0.0,
                damping=NOMINAL_JOINT_DAMPING,
            )
        },
    )


@configclass
class DomainRandCfg:
    """Per-environment physics randomisation (phase 1 of the DR roadmap).

    Ranges are multiplicative factors on the nominal value except
    actuator_effectiveness, which is absolute (1.0 == healthy motor).
    Sampling happens once per episode at reset, never per step.
    """

    enabled: bool = False
    joint_friction_range: tuple[float, float] = (0.0, 0.0)
    joint_damping_scale_range: tuple[float, float] = (1.0, 1.0)
    link_mass_scale_range: tuple[float, float] = (1.0, 1.0)
    actuator_effectiveness_range: tuple[float, float] = (1.0, 1.0)


@configclass
class TwoDofReachEnvCfg(DirectRLEnvCfg):
    """2-DOF planar reacher, 120 Hz physics / 30 Hz control, 10 s episodes."""

    decimation = 4
    episode_length_s = 10.0

    action_space = 2
    observation_space = 8
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=SUCCESS_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )

    # Task geometry
    target_radius_range: tuple[float, float] = (0.20, 0.85)
    success_radius: float = SUCCESS_RADIUS
    torque_limit: float = TORQUE_LIMIT
    link_lengths: tuple[float, float] = (LINK_LENGTH_1, LINK_LENGTH_2)
    init_height: float = 0.5
    nominal_joint_damping: float = NOMINAL_JOINT_DAMPING

    # "vanilla" = human-designed reward, "eureka" = LLM-discovered reward
    reward_variant: str = "vanilla"

    domain_rand: DomainRandCfg = DomainRandCfg()


@configclass
class TwoDofReachEurekaEnvCfg(TwoDofReachEnvCfg):
    """Identical environment, Eureka-discovered reward — the only changed variable."""

    reward_variant: str = "eureka"
```

### URDF (`source/pi_adr/assets/two_dof_arm.urdf`)

```xml
<?xml version="1.0"?>
<!--
  2-DOF planar serial arm ("goal-marker reacher").

  Geometry/inertia follow the project's technical report:
    link lengths  l1 = l2 = 0.5 m
    link masses   m1 = 1.0 kg, m2 = 0.5 kg
    joints        2 revolute, both about Z
    effort limit  +/- 20 Nm
    velocity lim  +/- 5 rad/s
    joint damping 0.05
    joint range   continuous (see note)

  Joint range note: the report specifies q_i in [-pi, pi], but hard limits there
  create a wrap-around trap — when the goal needs rotation past +/-pi the
  shortest path is blocked and the arm stalls against the stop. Measured with a
  scripted IK+PD controller: 31.4% of episodes fail with +/-pi limits and 100%
  of those failures end against a limit, versus 0% failures and 100% success
  below 1 mm once limits are widened. An optimal controller is therefore capped
  near 69%, so the report's 99.49% implies its limits were never binding.
  Both joints are continuous here; observed angles are wrapped to [-pi, pi].

  Plane convention: the arm lies in the world XY plane and both joints rotate
  about world Z, so gravity (-Z) acts along the joint axes and produces no
  joint torque. The planar dynamics therefore reduce to
  M(q)qdd + C(q,qd)qd + F(qd) = tau, i.e. g(q) == 0, which is what the
  report's reward/dynamics equations assume.

  Link inertias are thin cylinders of radius 0.02 m about the link COM at L/2:
    ixx = 0.5 m r^2 ,  iyy = izz = (1/12) m L^2 + (1/4) m r^2
-->
<robot name="two_dof_arm">

  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.02" rpy="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.04"/></geometry>
      <material name="base_grey"><color rgba="0.3 0.3 0.35 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.02" rpy="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.04"/></geometry>
    </collision>
  </link>

  <joint name="joint_1" type="continuous">
    <parent link="base_link"/>
    <child link="link_1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="20.0" velocity="5.0"/>
    <dynamics damping="0.05" friction="0.0"/>
  </joint>

  <link name="link_1">
    <inertial>
      <origin xyz="0.25 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.0002" ixy="0.0" ixz="0.0" iyy="0.0209333" iyz="0.0" izz="0.0209333"/>
    </inertial>
    <visual>
      <origin xyz="0.25 0 0" rpy="0 1.57079633 0"/>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
      <material name="link1_blue"><color rgba="0.15 0.45 0.85 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0.25 0 0" rpy="0 1.57079633 0"/>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </collision>
  </link>

  <joint name="joint_2" type="continuous">
    <parent link="link_1"/>
    <child link="link_2"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="20.0" velocity="5.0"/>
    <dynamics damping="0.05" friction="0.0"/>
  </joint>

  <link name="link_2">
    <inertial>
      <origin xyz="0.25 0 0" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0104667" iyz="0.0" izz="0.0104667"/>
    </inertial>
    <visual>
      <origin xyz="0.25 0 0" rpy="0 1.57079633 0"/>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
      <material name="link2_orange"><color rgba="0.95 0.5 0.1 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0.25 0 0" rpy="0 1.57079633 0"/>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </collision>
  </link>

  <!-- End-effector frame at the tip of link 2; massless marker frame only. -->
  <joint name="ee_fixed" type="fixed">
    <parent link="link_2"/>
    <child link="ee_link"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
  </joint>

  <link name="ee_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="1e-6" ixy="0.0" ixz="0.0" iyy="1e-6" iyz="0.0" izz="1e-6"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.025"/></geometry>
      <material name="ee_red"><color rgba="0.9 0.1 0.1 1.0"/></material>
    </visual>
  </link>

</robot>
```

**Discovered issue documented in the URDF itself:** `ImplicitActuatorCfg(damping=0.0)` originally
overrode the URDF's `damping="0.05"` — Isaac Lab's implicit actuator model takes its damping from
the `ActuatorCfg`, not the imported USD drive, so the *first* generation of results ran at
`b=0` while claiming `b=0.05`. Fixed by setting `damping=NOMINAL_JOINT_DAMPING` explicitly in
`arm_articulation_cfg` (§3 above) rather than relying on the URDF value propagating through.

### Actuation/PPO surface summary

| Quantity | Value | Source |
|---|---|---|
| Effort limit | ±20 Nm (`effort_limit_sim`) | `ImplicitActuatorCfg` |
| Velocity limit | ±5 rad/s (`velocity_limit_sim`) | `ImplicitActuatorCfg` |
| Drive stiffness | 0.0 (pure torque control) | `ImplicitActuatorCfg` |
| Drive damping | 0.05 (nominal), read back and confirmed by `validate_dynamics.py` | `ImplicitActuatorCfg` |
| Joint type | `continuous` (unbounded, wrapped in obs) | URDF |
| Solver iterations | position 4, velocity 0 | `ArticulationRootPropertiesCfg` |

---

## 4. Domain randomisation

Already shown in full in §2 (`_randomize_physics`) and §3 (`DomainRandCfg`). Summary of the code
path, since this is the piece the user specifically asked to see spelled out:

```python
# per-environment physics tensors, shape (num_envs, 2) each — NOT a shared scalar
self._joint_friction          # xi[2:4]
self._joint_damping           # xi[4:6]
self._actuator_effectiveness  # xi[6:8]
self._link_masses             # xi[0:2], read from root_physx_view after each write

# at reset, for the env_ids being reset:
friction = U(*joint_friction_range)              # (n, 2)
damping  = nominal_damping * U(*damping_scale_range)   # (n, 2)
effectiveness = U(*actuator_effectiveness_range)  # (n, 2)
mass_scale = U(*link_mass_scale_range)            # (n, 2), applied per-body via PhysX view

robot.write_joint_friction_coefficient_to_sim(friction, joint_ids=[j1,j2], env_ids=env_ids)
robot.write_joint_damping_to_sim(damping, joint_ids=[j1,j2], env_ids=env_ids)
robot.root_physx_view.set_masses(nominal_masses[env_ids] * mass_scale, env_ids)
```

This produces exactly `4096 different physics configurations` (one `ξ^(j)` per environment `j`),
not one shared randomized config replicated across environments — verified in `validate_env.py`
check I by reading friction back from the simulator per-environment and confirming nonzero
cross-environment standard deviation (see §17).

---

## 5. RAPP

`scripts/run_rapp.py` — full file.

```python
"""RAPP — Reward-Aware Physics Prior (DrEureka, https://github.com/eureka-research/DrEureka).

One-parameter-at-a-time feasibility sweep: hold every other physics parameter
at nominal, sweep one across a candidate grid, roll out the nominal policy, and
keep the values where it still succeeds. The resulting box C_RAPP is the hard
envelope that adaptive DR is later allowed to move probability mass inside.

    l_p = min{r : F(tau_r) = 1},  h_p = max{r : F(tau_r) = 1}

Unlike DrEureka's binary criterion we record the full success probability
S(pi, xi) per candidate, which the roadmap's continuous feasible set needs.

    env_isaaclab/bin/python pi_adr/scripts/run_rapp.py --headless \
        --checkpoint <nominal_policy.pth> --task PIADR-TwoDofReach-Vanilla-v0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="PIADR-TwoDofReach-Vanilla-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--run_name", type=str, default="rapp")
parser.add_argument(
    "--alpha", type=float, default=0.9, help="success fraction at 5 cm required for feasibility"
)
parser.add_argument(
    "--success_eps", type=float, default=0.05, help="success threshold in metres (RAPP criterion)"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402

import pi_adr.tasks  # noqa: F401, E402
from pi_adr.evaluation.runner import make_env_and_agent, rollout, set_domain_rand  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Candidate grids, chosen around the nominal values of the stage-1 arm
# (friction 0, damping 0.05 => scale 1.0, healthy actuator => 1.0).
# Grids must be wide enough to actually bracket the feasibility boundary. The
# first sweep found friction/damping/effectiveness feasible across their entire
# range, meaning the grid, not the policy, set the envelope.
SWEEPS: dict[str, list[float]] = {
    "friction": [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "damping_scale": [0.25, 1.0, 4.0, 16.0, 64.0, 128.0, 256.0, 512.0],
    "effectiveness": [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    "mass_scale": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0],
}


def main() -> None:
    env, base_env, agent, _ = make_env_and_agent(
        args_cli.task, args_cli.checkpoint, args_cli.num_envs, args_cli.seed, args_cli.device
    )

    envelope: dict[str, dict] = {}

    for param, candidates in SWEEPS.items():
        print(f"\n=== RAPP sweep: {param} ===", flush=True)
        records = []
        for value in candidates:
            set_domain_rand(base_env, **{param: value})
            result, accum = rollout(env, base_env, agent, f"rapp_{param}_{value}")
            success = float((accum.terminal_distance < args_cli.success_eps).float().mean().item())
            feasible = success >= args_cli.alpha
            records.append(
                {
                    "value": value,
                    "success": success,
                    "feasible": feasible,
                    "mean_terminal_error_m": result.mean_terminal_error_m,
                    "success_1cm": result.success_rate.get("0.01"),
                }
            )
            print(
                f"  {param}={value:<6} S={success:.4f} "
                f"mean_err={result.mean_terminal_error_m * 1000:6.2f}mm "
                f"{'FEASIBLE' if feasible else 'infeasible'}",
                flush=True,
            )

        feasible_values = [r["value"] for r in records if r["feasible"]]
        envelope[param] = {
            "candidates": records,
            "lower": min(feasible_values) if feasible_values else None,
            "upper": max(feasible_values) if feasible_values else None,
            "nominal_feasible": any(
                r["feasible"] and r["value"] == (0.0 if param == "friction" else 1.0) for r in records
            ),
        }

    out = {
        "run_name": args_cli.run_name,
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "alpha": args_cli.alpha,
        "success_eps_m": args_cli.success_eps,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "envelope": envelope,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{args_cli.run_name}_envelope.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n================ C_RAPP ================")
    for param, data in envelope.items():
        print(f"  {param:<16} [{data['lower']}, {data['upper']}]")
    print(f"  written to {path}")
    print("========================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

Note: `set_domain_rand(base_env, **{param: value})` (in `evaluation/runner.py`, §10) turns a
scalar `value` into a **degenerate range** `(value, value)`, so every environment in the sweep
gets *exactly* that physics value — this is what makes it a clean one-parameter sweep rather than
a distribution.

### Measured envelope (`results/rapp_eureka500_envelope.json`, α=0.9, success_eps=0.05 m,
nominal policy = Eureka @500, 1024 envs, seed 1234) — this is the full file, not excerpted:

```json
{
  "run_name": "rapp_eureka500",
  "task": "PIADR-TwoDofReach-Eureka-v0",
  "checkpoint": "logs/rl_games/two_dof_reach/2026-09-16_02-59-44/nn/last_two_dof_reach_ep_500_rew_26463.402.pth",
  "alpha": 0.9,
  "success_eps_m": 0.05,
  "num_envs": 1024,
  "seed": 1234,
  "timestamp": "2026-09-16T03:44:37",
  "envelope": {
    "friction": {
      "candidates": [
        {"value": 0.0,  "success": 0.9736, "feasible": true,  "mean_terminal_error_m": 0.04501, "success_1cm": 0.0361},
        {"value": 0.1,  "success": 0.9678, "feasible": true,  "mean_terminal_error_m": 0.04597, "success_1cm": 0.0420},
        {"value": 0.5,  "success": 0.9766, "feasible": true,  "mean_terminal_error_m": 0.04241, "success_1cm": 0.0498},
        {"value": 1.0,  "success": 0.9688, "feasible": true,  "mean_terminal_error_m": 0.04988, "success_1cm": 0.0391},
        {"value": 2.0,  "success": 0.9707, "feasible": true,  "mean_terminal_error_m": 0.04580, "success_1cm": 0.0576},
        {"value": 4.0,  "success": 0.9463, "feasible": true,  "mean_terminal_error_m": 0.05927, "success_1cm": 0.0713},
        {"value": 8.0,  "success": 0.8428, "feasible": false, "mean_terminal_error_m": 0.09459, "success_1cm": 0.0566},
        {"value": 16.0, "success": 0.4883, "feasible": false, "mean_terminal_error_m": 0.21810, "success_1cm": 0.0449}
      ],
      "lower": 0.0, "upper": 4.0, "nominal_feasible": true
    },
    "damping_scale": {
      "candidates": [
        {"value": 0.25,  "success": 0.9678, "feasible": true,  "mean_terminal_error_m": 0.04811, "success_1cm": 0.0410},
        {"value": 1.0,   "success": 0.9688, "feasible": true,  "mean_terminal_error_m": 0.04928, "success_1cm": 0.0459},
        {"value": 4.0,   "success": 0.9570, "feasible": true,  "mean_terminal_error_m": 0.05807, "success_1cm": 0.0410},
        {"value": 16.0,  "success": 0.9590, "feasible": true,  "mean_terminal_error_m": 0.05596, "success_1cm": 0.0430},
        {"value": 64.0,  "success": 0.9248, "feasible": true,  "mean_terminal_error_m": 0.08857, "success_1cm": 0.0830},
        {"value": 128.0, "success": 0.9209, "feasible": true,  "mean_terminal_error_m": 0.08282, "success_1cm": 0.1621},
        {"value": 256.0, "success": 0.9102, "feasible": true,  "mean_terminal_error_m": 0.08261, "success_1cm": 0.1250},
        {"value": 512.0, "success": 0.7871, "feasible": false, "mean_terminal_error_m": 0.17950, "success_1cm": 0.1182}
      ],
      "lower": 0.25, "upper": 256.0, "nominal_feasible": true
    },
    "effectiveness": {
      "candidates": [
        {"value": 0.02, "success": 0.5029, "feasible": false, "mean_terminal_error_m": 0.38540, "success_1cm": 0.0850},
        {"value": 0.05, "success": 0.5664, "feasible": false, "mean_terminal_error_m": 0.38293, "success_1cm": 0.1113},
        {"value": 0.1,  "success": 0.7676, "feasible": false, "mean_terminal_error_m": 0.20839, "success_1cm": 0.1436},
        {"value": 0.2,  "success": 0.8652, "feasible": false, "mean_terminal_error_m": 0.09160, "success_1cm": 0.1494},
        {"value": 0.3,  "success": 0.9150, "feasible": true,  "mean_terminal_error_m": 0.05340, "success_1cm": 0.1602},
        {"value": 0.5,  "success": 0.9766, "feasible": true,  "mean_terminal_error_m": 0.03118, "success_1cm": 0.1406},
        {"value": 0.75, "success": 0.9639, "feasible": true,  "mean_terminal_error_m": 0.04443, "success_1cm": 0.0811},
        {"value": 1.0,  "success": 0.9717, "feasible": true,  "mean_terminal_error_m": 0.04709, "success_1cm": 0.0322}
      ],
      "lower": 0.3, "upper": 1.0, "nominal_feasible": true
    },
    "mass_scale": {
      "candidates": [
        {"value": 0.25, "success": 0.2559, "feasible": false, "mean_terminal_error_m": 0.12869, "success_1cm": 0.0078},
        {"value": 0.5,  "success": 0.5303, "feasible": false, "mean_terminal_error_m": 0.09384, "success_1cm": 0.0049},
        {"value": 0.75, "success": 0.7861, "feasible": false, "mean_terminal_error_m": 0.05793, "success_1cm": 0.0293},
        {"value": 1.0,  "success": 0.9746, "feasible": true,  "mean_terminal_error_m": 0.04675, "success_1cm": 0.0479},
        {"value": 1.5,  "success": 0.9668, "feasible": true,  "mean_terminal_error_m": 0.03835, "success_1cm": 0.0771},
        {"value": 2.0,  "success": 0.9541, "feasible": true,  "mean_terminal_error_m": 0.03621, "success_1cm": 0.1338},
        {"value": 4.0,  "success": 0.9189, "feasible": true,  "mean_terminal_error_m": 0.04199, "success_1cm": 0.1719},
        {"value": 8.0,  "success": 0.7402, "feasible": false, "mean_terminal_error_m": 0.17705, "success_1cm": 0.1250}
      ],
      "lower": 1.0, "upper": 4.0, "nominal_feasible": true
    }
  }
}
```

**C_RAPP = friction [0.0, 4.0] × damping_scale [0.25, 256.0] × effectiveness [0.3, 1.0] × mass_scale [1.0, 4.0]**

Note a prior, discarded run of this sweep (before the mass-write-skip bug in §2/§4 was fixed)
found `mass_scale=1.0` scoring `S=0.816` instead of `0.975` — because the environment skipped
writing masses back to nominal when `range == (1.0, 1.0)`, so "nominal" silently inherited the
*previous* candidate's masses. Fixed by always writing masses on every reset (§2 code above,
comment explains it in place).

---

## 6. Physics-performance map

Two independent sources of this map exist in the codebase:

**(a) Fixed-policy one-parameter sweep** — literally the RAPP candidate table above; each row is
`ξ → [S(π,ξ), mean_terminal_error, P(d<1cm)]`. Rendered by `scripts/summarize_results.py`
(function body below) directly from `rapp_eureka500_envelope.json`.

```python
# scripts/summarize_results.py — physics-performance-map rendering, verbatim
print("\nPhysics-performance map (fixed policy, one parameter at a time)")
for param, data in env["envelope"].items():
    print(f"  {param}:")
    print("    value     S(pi,xi)   mean err (mm)   P(d<1cm)")
    for c in data["candidates"]:
        flag = "" if c["feasible"] else "   <- infeasible"
        s1 = c.get("success_1cm")
        print(
            f"    {c['value']:<8g}  {c['success']:.4f}     "
            f"{c['mean_terminal_error_m'] * 1000:8.2f}      "
            f"{'—' if s1 is None else f'{s1 * 100:5.2f}%'}{flag}"
        )
```

**(b) Under-DR binned map** — `physics_binned_success` in `evaluation/metrics.py` (full function
already shown in §10 below), invoked inside `eval.py` whenever `--dr` is set:

```python
if dr.enabled:
    physics = base_env.physics_params
    result.extra["physics_performance_map"] = {
        "friction_j1": physics_binned_success(accum.terminal_distance, physics[:, 0]),
        "effectiveness_j1": physics_binned_success(accum.terminal_distance, physics[:, 4]),
    }
```

This produces `M(ξ) = [P_1cm(ξ ∈ bin)]` for 8 bins per swept dimension, with per-bin sample
counts, embedded directly in `results/*_DRtrain.json` / `*_DRshift.json` (full contents in §16).
`M(ξ)` here reports `P_1cm` only (not the full `[P_5cm,P_2cm,P_1cm,E_d,E_q̇,E_τ,T_settle]` vector
the roadmap specifies) — extending it to the full vector is a small change to
`physics_binned_success` (currently hard-coded to one `threshold=0.01`), not yet done.

---

## 7. Shifted-DR experiment

Exact configuration for both arms, taken verbatim from the commands actually run (recorded in
`EXPERIMENT_LOG.md` and reproduced in the `results/*.json` `domain_rand` blocks, §16):

```python
# TRAIN-DISTRIBUTION eval (both A_vanilla_5000_DRtrain and B_eureka_1500_DRtrain)
friction_range          = (0.0, 0.25)
damping_scale_range     = (0.5, 4.0)
effectiveness_range     = (0.5, 1.0)
mass_scale_range        = (1.0, 1.5)
# entirely inside C_RAPP = friction[0,4] x damping[0.25,256] x eff[0.3,1] x mass[1,4]

# SHIFTED / UNSEEN eval (both A_vanilla_5000_DRshift and B_eureka_1500_DRshift)
friction_range          = (0.25, 0.5)
damping_scale_range     = (4.0, 8.0)
effectiveness_range     = (0.3, 0.5)
mass_scale_range        = (1.5, 2.5)
# strictly harder than DRtrain on every axis; boundary values overlap with the
# harder end of C_RAPP (e.g. damping [4,8] sits just past the DRtrain max of 4.0)
```

Exact CLI commands used (from the log):

```bash
PY=/data/home/amartyaranjan888/projects/I/isaac_env/env_isaaclab/bin/python
$PY -u scripts/eval.py --task PIADR-TwoDofReach-Eureka-v0 --checkpoint "$CKPT" \
  --run_name B_eureka_1500_DRtrain --num_envs 4096 --headless --dr \
  --friction_range 0.0 0.25 --damping_scale_range 0.5 4.0 \
  --effectiveness_range 0.5 1.0 --mass_scale_range 1.0 1.5 \
  --note "in-distribution DR, inside C_RAPP"

$PY -u scripts/eval.py --task PIADR-TwoDofReach-Eureka-v0 --checkpoint "$CKPT" \
  --run_name B_eureka_1500_DRshift --num_envs 4096 --headless --dr \
  --friction_range 0.25 0.5 --damping_scale_range 4.0 8.0 \
  --effectiveness_range 0.3 0.5 --mass_scale_range 1.5 2.5 \
  --note "shifted unseen DR distribution"
```

(identical commands for `A_vanilla_5000_*` with `--task PIADR-TwoDofReach-Vanilla-v0` and the
vanilla checkpoint). This is DR applied **at evaluation time only** — the policies were trained
with `domain_rand.enabled = False` and never saw any of this physics variation during PPO. No DR
was used *during training* in this stage; that is future work (roadmap Phase 4, not started).

---

## 8. PPO configuration

`source/pi_adr/tasks/two_dof_reach/agents/rl_games_ppo_cfg.yaml` — identical for both vanilla and
Eureka tasks (the reward is the only thing that differs between the two gym-registered task IDs;
this same file is `rl_games_cfg_entry_point` for both). Full file:

```yaml
params:
  seed: 42

  env:
    clip_observations: 100.0
    clip_actions: 1.0

  algo:
    name: a2c_continuous

  model:
    name: continuous_a2c_logstd

  network:
    name: actor_critic
    separate: False
    space:
      continuous:
        mu_activation: None
        sigma_activation: None
        mu_init:
          name: default
        sigma_init:
          name: const_initializer
          val: 0
        fixed_sigma: True
    mlp:
      # 8 -> 256 -> 128 -> 64, ELU (stage-1 report, Table B1)
      units: [256, 128, 64]
      activation: elu
      d2rl: False
      initializer:
        name: default
      regularizer:
        name: None

  load_checkpoint: False
  load_path: ''

  config:
    name: two_dof_reach
    env_name: rlgpu
    device: 'cuda:0'
    device_name: 'cuda:0'
    multi_gpu: False
    ppo: True
    mixed_precision: False
    normalize_input: True
    normalize_value: True
    normalize_advantage: True
    value_bootstrap: True
    num_actors: -1        # overwritten at runtime with --num_envs
    reward_shaper:
      scale_value: 1.0
    gamma: 0.99
    tau: 0.95             # GAE lambda
    learning_rate: 1e-3
    lr_schedule: adaptive
    kl_threshold: 0.01
    schedule_type: legacy
    score_to_win: 100000
    max_epochs: 500
    save_best_after: 50
    save_frequency: 100
    print_stats: True
    grad_norm: 1.0
    entropy_coef: 0.005
    truncate_grads: True
    e_clip: 0.2
    horizon_length: 24    # 4096 x 24 = 98,304 transitions per rollout
    # minibatch_size_per_env * 4 == horizon_length keeps exactly 4 minibatches
    # for ANY --num_envs, satisfying rl_games' batch % minibatch == 0 assert.
    minibatch_size_per_env: 6
    mini_epochs: 8
    critic_coef: 1.0
    clip_value: True
    bounds_loss_coef: 0.0001
```

`fixed_sigma: True` with `sigma_init: const_initializer, val: 0` means σ is a learned but
state-independent parameter, initialised at `exp(0) = 1.0` and adjusted by gradient descent
through the PPO loss (not scheduled/annealed by any external code) — this is the mechanism behind
the measured `σ: 1.0 → ≈0.024` collapse (`losses/entropy: 2.83 → −4.66` in TensorBoard, extracted
via the `tensorboard.backend.event_processing.event_accumulator` API against
`logs/rl_games/two_dof_reach/<run>/summaries/events.out.tfevents.*`). `max_epochs` is overwritten
by `--max_iterations` on the CLI, not read from this file, at train time
(`agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations`, in
`IsaacLab/scripts/reinforcement_learning/rl_games/train.py`, unmodified upstream code).

`num_actors: -1` is likewise overwritten at runtime with the real `--num_envs` value, which is
why `minibatch_size_per_env` (not a fixed `minibatch_size`) is used — it keeps
`(horizon_length × num_envs) mod minibatch_size == 0` satisfied for *any* `--num_envs`, avoiding
the exact assertion crash recorded in `EXPERIMENT_LOG.md` (`rl_games/common/a2c_common.py:256`,
`assert(self.batch_size % self.minibatch_size == 0)`).

---

## 9. Reward code

Already shown verbatim in §2 (`_get_rewards`). Restated with the exact equation correspondence:

```python
def _get_rewards(self) -> torch.Tensor:
    d = self._target_distance()
    if self.cfg.reward_variant == "vanilla":
        reward = -d + 5.0 * (d < self.cfg.success_radius).float() - 0.005 * self._actions.pow(2).sum(-1)
        # R_vanilla = -d + 5*I(d<0.05) - 0.005*||a||^2     (a = clamped policy action, [-1,1]^2)
    elif self.cfg.reward_variant == "eureka":
        qd = self._robot.data.joint_vel[:, self._joint_ids]
        reward = (
            torch.exp(-10.0 * d)
            + 100.0 * (d < self.cfg.success_radius).float()
            - 0.01 * qd.pow(2).sum(-1)
            - 0.1
        )
        # R_eureka = exp(-10d) + 100*I(d<0.05) - 0.01*||q_dot||^2 - 0.1
    return reward
```

`d` is `self._target_distance()`, i.e. `‖target_local − ee_local‖₂` computed fresh every call (not
cached from the observation step), so the reward always uses the *current* post-substep state.
`self._actions` is the clamped `a_t` from `_pre_physics_step`, i.e. the vanilla reward's `‖a‖²`
term uses the **normalised** action (max value 2.0 for `‖a‖²` at saturation), while the Eureka
reward's `‖q̇‖²` term uses **physical** joint velocity in rad/s (max value `2·5² = 50` at the
velocity limit) — this asymmetry is in the source report's own equations and was not altered.

---

## 10. Evaluation / metrics code

`source/pi_adr/evaluation/metrics.py` — full file (defines `d_T`, `P_ε`, and every other frozen
metric):

```python
"""Evaluation metrics for the 2-DOF reacher, fixed to match the stage-1 report.

Every number this module produces is defined once here so that results stay
comparable across vanilla/Eureka rewards, DR variants and future PINN runs.

Definitions (frozen — do not change without bumping METRICS_VERSION):
  terminal error d_T   Euclidean distance ||p_target - p_ee|| at the LAST step
                       of the episode, in metres.
  P(d < eps)           fraction of episodes whose terminal error is below eps.
  mean joint velocity  mean over (episode steps, envs, joints) of |qd|, rad/s.
  mean control effort  mean over (episode steps, envs) of ||a||_2^2, where a is
                       the normalised action in [-1, 1]^2 (so effort in [0, 2]).
  settling time        first time (s) after which d stays below SETTLE_EPS for
                       the remainder of the episode; episode length if never.
  overshoot            max(0, max_t d(t) - d(0)) over the episode, in metres —
                       how much further away the tip got than it started.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

import torch

METRICS_VERSION = "1.0.0"

# Report thresholds (5 cm / 2 cm / 1 cm) plus the finer ones the roadmap asks for.
PRECISION_THRESHOLDS_M: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
SETTLE_EPS_M = 0.05


@dataclass
class EvalResult:
    """Frozen metric set for one evaluated policy."""

    run_name: str
    num_episodes: int
    metrics_version: str = METRICS_VERSION

    success_rate: dict[str, float] = field(default_factory=dict)
    mean_terminal_error_m: float = float("nan")
    median_terminal_error_m: float = float("nan")
    p90_terminal_error_m: float = float("nan")
    worst_decile_error_m: float = float("nan")

    mean_joint_velocity_rad_s: float = float("nan")
    max_joint_velocity_rad_s: float = float("nan")
    mean_control_effort: float = float("nan")
    mean_abs_action: float = float("nan")

    mean_settling_time_s: float = float("nan")
    mean_overshoot_m: float = float("nan")

    extra: dict = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)

    def summary_line(self) -> str:
        s = self.success_rate
        return (
            f"{self.run_name}: "
            f"5cm={s.get('0.05', float('nan')):.2%} "
            f"2cm={s.get('0.02', float('nan')):.2%} "
            f"1cm={s.get('0.01', float('nan')):.2%} | "
            f"mean={self.mean_terminal_error_m * 1000:.1f}mm "
            f"med={self.median_terminal_error_m * 1000:.1f}mm | "
            f"vel={self.mean_joint_velocity_rad_s:.3f}rad/s "
            f"eff={self.mean_control_effort:.3f}"
        )


class RolloutAccumulator:
    """Collects per-step tensors over a fixed-horizon vectorised rollout.

    Call `record` once per control step with per-env quantities, then `finalize`.
    Kept on-device; only the final reduction moves to CPU.
    """

    def __init__(self, num_envs: int, dt: float, device: torch.device | str = "cuda"):
        self.num_envs = num_envs
        self.dt = dt
        self.device = torch.device(device)
        self._dist: list[torch.Tensor] = []
        self._vel_abs_sum: list[torch.Tensor] = []
        self._vel_max: list[torch.Tensor] = []
        self._effort: list[torch.Tensor] = []
        self._abs_action: list[torch.Tensor] = []

    def record(self, dist: torch.Tensor, joint_vel: torch.Tensor, action: torch.Tensor) -> None:
        """dist: (N,) metres; joint_vel: (N, J) rad/s; action: (N, A) normalised."""
        self._dist.append(dist.detach().flatten())
        self._vel_abs_sum.append(joint_vel.detach().abs().mean(dim=-1))
        self._vel_max.append(joint_vel.detach().abs().amax(dim=-1))
        self._effort.append(action.detach().pow(2).sum(dim=-1))
        self._abs_action.append(action.detach().abs().mean(dim=-1))

    def finalize(self, run_name: str, extra: dict | None = None) -> EvalResult:
        dist = torch.stack(self._dist)  # (T, N)
        vel = torch.stack(self._vel_abs_sum)
        vel_max = torch.stack(self._vel_max)
        effort = torch.stack(self._effort)
        abs_action = torch.stack(self._abs_action)

        terminal = dist[-1]
        sorted_terminal = terminal.sort().values
        n = sorted_terminal.numel()
        worst_decile = sorted_terminal[int(math.floor(0.9 * n)) :].mean()

        success = {
            f"{eps:g}": (terminal < eps).float().mean().item()
            for eps in PRECISION_THRESHOLDS_M
        }

        return EvalResult(
            run_name=run_name,
            num_episodes=int(n),
            success_rate=success,
            mean_terminal_error_m=terminal.mean().item(),
            median_terminal_error_m=terminal.median().item(),
            p90_terminal_error_m=torch.quantile(terminal.float(), 0.9).item(),
            worst_decile_error_m=worst_decile.item(),
            mean_joint_velocity_rad_s=vel.mean().item(),
            max_joint_velocity_rad_s=vel_max.max().item(),
            mean_control_effort=effort.mean().item(),
            mean_abs_action=abs_action.mean().item(),
            mean_settling_time_s=settling_time(dist, self.dt).mean().item(),
            mean_overshoot_m=overshoot(dist).mean().item(),
            extra=extra or {},
        )

    @property
    def terminal_distance(self) -> torch.Tensor:
        return self._dist[-1]


def settling_time(dist: torch.Tensor, dt: float, eps: float = SETTLE_EPS_M) -> torch.Tensor:
    """(T, N) distances -> (N,) seconds until d stays < eps for the rest of the episode."""
    inside = dist < eps
    # Suffix-AND: index t is True iff every step from t onward is inside eps.
    stays = torch.flip(torch.cumprod(torch.flip(inside.int(), dims=[0]), dim=0), dims=[0]).bool()
    horizon = dist.shape[0]
    first = torch.where(
        stays.any(dim=0),
        stays.float().argmax(dim=0),
        torch.full((dist.shape[1],), horizon, device=dist.device, dtype=torch.long),
    )
    return first.float() * dt


def overshoot(dist: torch.Tensor) -> torch.Tensor:
    """(T, N) distances -> (N,) metres the tip receded beyond its starting error."""
    return (dist.amax(dim=0) - dist[0]).clamp_min(0.0)


def precision_curve(terminal: torch.Tensor, thresholds: Sequence[float] = PRECISION_THRESHOLDS_M) -> dict[str, float]:
    return {f"{eps:g}": (terminal < eps).float().mean().item() for eps in thresholds}


def physics_binned_success(
    terminal: torch.Tensor,
    physics_value: torch.Tensor,
    num_bins: int = 8,
    threshold: float = 0.01,
) -> dict:
    """P(d < threshold | physics parameter in bin) — the physics-performance map."""
    lo, hi = physics_value.min().item(), physics_value.max().item()
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    edges = torch.linspace(lo, hi, num_bins + 1, device=physics_value.device)
    idx = torch.bucketize(physics_value, edges[1:-1].contiguous())
    centers, rates, counts = [], [], []
    for b in range(num_bins):
        mask = idx == b
        count = int(mask.sum().item())
        centers.append(0.5 * (edges[b] + edges[b + 1]).item())
        rates.append((terminal[mask] < threshold).float().mean().item() if count else float("nan"))
        counts.append(count)
    return {"threshold": threshold, "bin_centers": centers, "success_rate": rates, "counts": counts}
```

`source/pi_adr/evaluation/runner.py` — shared env/policy construction and the corrected rollout
loop (full file, includes the documented auto-reset bug fix):

```python
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


def make_env_and_agent(task, checkpoint, num_envs, seed, device=None):
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
```

`scripts/eval.py` (the CLI entry point that both `train_and_eval.sh` and manual DR runs use) —
full file already given in the previous turn's summary; reproduced here for completeness:

```python
"""Evaluate a trained 2-DOF reacher policy with the frozen metric suite.

One evaluation == one full 300-step episode per environment, deterministic
actions, fixed seed. Results go to results/<run_name>.json and are appended to
results/index.jsonl so every number ever produced stays comparable.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=1234, help="eval seed, held fixed across variants")
parser.add_argument("--note", type=str, default="")
parser.add_argument("--dr", action="store_true", help="evaluate under domain randomisation")
parser.add_argument("--friction_range", type=float, nargs=2, default=None)
parser.add_argument("--damping_scale_range", type=float, nargs=2, default=None)
parser.add_argument("--effectiveness_range", type=float, nargs=2, default=None)
parser.add_argument("--mass_scale_range", type=float, nargs=2, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json, os, time
from datetime import datetime
import pi_adr.tasks  # noqa: F401
from pi_adr.evaluation.metrics import physics_binned_success
from pi_adr.evaluation.runner import make_env_and_agent, rollout, set_domain_rand

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main() -> None:
    env, base_env, agent, env_cfg = make_env_and_agent(
        args_cli.task, args_cli.checkpoint, args_cli.num_envs, args_cli.seed, args_cli.device
    )

    if args_cli.dr:
        overrides = {}
        if args_cli.friction_range is not None:
            overrides["friction"] = tuple(args_cli.friction_range)
        if args_cli.damping_scale_range is not None:
            overrides["damping_scale"] = tuple(args_cli.damping_scale_range)
        if args_cli.effectiveness_range is not None:
            overrides["effectiveness"] = tuple(args_cli.effectiveness_range)
        if args_cli.mass_scale_range is not None:
            overrides["mass_scale"] = tuple(args_cli.mass_scale_range)
        set_domain_rand(base_env, **overrides)
    else:
        base_env.cfg.domain_rand.enabled = False

    dr = base_env.cfg.domain_rand
    start = time.time()
    result, accum = rollout(
        env, base_env, agent, args_cli.run_name,
        extra={
            "task": args_cli.task, "checkpoint": args_cli.checkpoint,
            "num_envs": args_cli.num_envs, "eval_seed": args_cli.seed,
            "reward_variant": env_cfg.reward_variant,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "note": args_cli.note,
            "domain_rand": {
                "enabled": bool(dr.enabled),
                "joint_friction_range": list(dr.joint_friction_range),
                "joint_damping_scale_range": list(dr.joint_damping_scale_range),
                "actuator_effectiveness_range": list(dr.actuator_effectiveness_range),
                "link_mass_scale_range": list(dr.link_mass_scale_range),
            },
        },
    )
    result.extra["eval_wall_clock_s"] = round(time.time() - start, 2)

    if dr.enabled:
        physics = base_env.physics_params
        result.extra["physics_performance_map"] = {
            "friction_j1": physics_binned_success(accum.terminal_distance, physics[:, 0]),
            "effectiveness_j1": physics_binned_success(accum.terminal_distance, physics[:, 4]),
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args_cli.run_name}.json")
    result.to_json(out_path)
    with open(os.path.join(RESULTS_DIR, "index.jsonl"), "a") as fh:
        fh.write(json.dumps({
            "run_name": result.run_name, "success_rate": result.success_rate,
            "mean_terminal_error_m": result.mean_terminal_error_m,
            "median_terminal_error_m": result.median_terminal_error_m,
            "mean_joint_velocity_rad_s": result.mean_joint_velocity_rad_s,
            "mean_control_effort": result.mean_control_effort,
            **{k: v for k, v in result.extra.items() if k != "physics_performance_map"},
        }) + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

`scripts/summarize_results.py` — the table/report generator, full file already reproduced in
§6 context above and available in full at that path.

---

## 11. IK + PD baseline

`scripts/pd_baseline.py` — full file.

```python
"""Scripted IK + PD controller — the precision ceiling of this benchmark.

Neither reward gets below ~32 mm median error, so before blaming PPO we need to
know what the environment itself permits: with 30 Hz control, +/-20 Nm and
+/-5 rad/s on a near-frictionless arm, is millimetre accuracy reachable at all?

This controller solves the 2-link inverse kinematics for the goal marker and
drives the joints with a PD law inside the same action interface the policy
uses (normalised actions -> 20 Nm), measured with the same frozen metric suite.
It is an upper bound on achievable precision, not a learned policy.
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
    "--wide_limits", action="store_true",
    help="widen joint position limits to +/-10 rad, to test whether the +/-pi limits trap the arm",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json, os
from datetime import datetime
import gymnasium as gym, torch
import pi_adr.tasks  # noqa: F401
from pi_adr.evaluation.metrics import RolloutAccumulator
from pi_adr.tasks.two_dof_reach.two_dof_reach_env_cfg import (
    LINK_LENGTH_1, LINK_LENGTH_2, TORQUE_LIMIT, TwoDofReachEnvCfg,
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

    accum = RolloutAccumulator(env.num_envs, env.step_dt, device)
    horizon = int(env.max_episode_length)

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

    result = accum.finalize(args_cli.run_name, extra={
        "controller": f"IK + PD (kp={args_cli.kp}, kd={args_cli.kd})",
        "num_envs": args_cli.num_envs, "eval_seed": args_cli.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": "scripted reference: precision ceiling of the environment, not a learned policy",
    })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result.to_json(os.path.join(RESULTS_DIR, f"{args_cli.run_name}.json"))

    final_q = env._robot.data.joint_pos[:, env._joint_ids]
    at_limit = (final_q.abs() > 3.10).any(dim=-1)
    failed = accum.terminal_distance > 0.05
    # failures / at-limit / failed-and-at-limit diagnostic printed here

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

Mathematically: `q* = IK(x_t, y_t)` via the two-link closed-form solution (elbow-up/elbow-down
candidates, nearest-in-angle selection), then `τ = K_p (q* − q) − K_d q̇` with `K_p=12.0 N·m/rad`,
`K_d=1.2 N·m·s/rad`, converted to a normalised action `a = clamp(τ/20, −1, 1)` so the controller
uses the exact same actuation interface as the learned policy. Result: **100% success at every
threshold down to 1 mm** once the `±π` joint-limit trap is removed (§17 has the before/after
numbers) — this is the evidence that the environment, not the control rate or torque budget,
permits millimetre precision.

---

## 12. PINN — full pipeline

### 12a. Dynamics (`source/pi_adr/pinn/dynamics.py`) — full file

```python
"""Closed-form dynamics of the 2-link planar arm, batched and differentiable.

The arm lies in the horizontal plane (joints about Z), so gravity produces no
joint torque and the equations of motion reduce to

    M(q; xi) qdd + C(q, qd; xi) + F(qd; xi) = alpha * tau_cmd

where tau_cmd is the torque the policy *asked for* and alpha is per-joint
actuator effectiveness. Keeping alpha on the right-hand side (rather than
feeding the observer the already-scaled torque) is what makes actuator
degradation identifiable — otherwise it is absorbed into the applied torque and
the physics observer can never report it.

Verified against PhysX by scripts/validate_dynamics.py (2.3% mean relative
error on qdd; 0.04 Nm on recovered torque).

Parameter vector, ordered as in the roadmap:
    xi = [m1, m2, f1, f2, b1, b2, a1, a2]
"""

from __future__ import annotations
import torch

L1 = 0.5
L2 = 0.5
LC1 = 0.25
LC2 = 0.25
I1_UNIT = (1.0 / 12.0) * L1**2 + 0.25 * 0.02**2
I2_UNIT = (1.0 / 12.0) * L2**2 + 0.25 * 0.02**2

XI_NAMES = ("m1", "m2", "f1", "f2", "b1", "b2", "a1", "a2")
XI_DIM = len(XI_NAMES)
NOMINAL_XI = (1.0, 0.5, 0.0, 0.0, 0.05, 0.05, 1.0, 1.0)
_COULOMB_EPS = 1e-2


def _split(xi: torch.Tensor):
    return xi[..., 0], xi[..., 1], xi[..., 2:4], xi[..., 4:6], xi[..., 6:8]


def mass_matrix(q: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """(..., 2) joint angles -> (..., 2, 2) inertia matrix."""
    m1, m2, _, _, _ = _split(xi)
    i1 = m1 * I1_UNIT
    i2 = m2 * I2_UNIT
    c2 = torch.cos(q[..., 1])
    m11 = i1 + i2 + m1 * LC1**2 + m2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * c2)
    m12 = i2 + m2 * (LC2**2 + L1 * LC2 * c2)
    m22 = i2 + m2 * LC2**2
    row0 = torch.stack([m11, m12], dim=-1)
    row1 = torch.stack([m12, m22.expand_as(m12)], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def coriolis_vector(q, qd, xi) -> torch.Tensor:
    """Velocity-product term C(q, qd) qd, shape (..., 2)."""
    _, m2, _, _, _ = _split(xi)
    h = -m2 * L1 * LC2 * torch.sin(q[..., 1])
    qd1, qd2 = qd[..., 0], qd[..., 1]
    return torch.stack([h * (2.0 * qd1 * qd2 + qd2**2), -h * qd1**2], dim=-1)


def friction_vector(qd, xi) -> torch.Tensor:
    """Viscous + smoothed Coulomb joint friction, shape (..., 2)."""
    _, _, f, b, _ = _split(xi)
    return b * qd + f * torch.tanh(qd / _COULOMB_EPS)


def passive_torque(q, qd, qdd, xi) -> torch.Tensor:
    """M(q) qdd + C(q, qd) + F(qd) — the mechanical side of the EOM."""
    m = mass_matrix(q, xi)
    return (m @ qdd.unsqueeze(-1)).squeeze(-1) + coriolis_vector(q, qd, xi) + friction_vector(qd, xi)


def inverse_dynamics(q, qd, qdd, xi) -> torch.Tensor:
    """Commanded torque needed to produce qdd, accounting for actuator effectiveness."""
    _, _, _, _, alpha = _split(xi)
    return passive_torque(q, qd, qdd, xi) / alpha.clamp_min(1e-3)


def forward_dynamics(q, qd, tau_cmd, xi) -> torch.Tensor:
    """Joint acceleration produced by a commanded torque under parameters xi."""
    _, _, _, _, alpha = _split(xi)
    m = mass_matrix(q, xi)
    rhs = alpha * tau_cmd - coriolis_vector(q, qd, xi) - friction_vector(qd, xi)
    return torch.linalg.solve(m, rhs.unsqueeze(-1)).squeeze(-1)


def physics_residual(q, qd, qdd, tau_cmd, xi) -> torch.Tensor:
    """r = M qdd + C + F - alpha * tau_cmd. Zero when xi explains the transition."""
    _, _, _, _, alpha = _split(xi)
    return passive_torque(q, qd, qdd, xi) - alpha * tau_cmd
```

### 12b. Data collection (`scripts/collect_pinn_data.py`) — full file

```python
"""Generate the dynamics dataset for the physics observer.

Follows the roadmap's dataset rules (section 17): sample physics from a
controlled distribution, excite with scripted/random torques rather than one
policy's narrow state distribution, record q/qd/tau/qdd, and split by *physics
configuration* so a model cannot pass validation by memorising a configuration
it already saw.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=128, help="transitions recorded per environment")
parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "extrap"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--torque_scale", type=float, default=0.25, help="fraction of the 20 Nm limit")
parser.add_argument("--max_speed", type=float, default=3.5,
    help="mask transitions above this |qd|; PhysX ramps the 5 rad/s limit in from ~4 rad/s")
parser.add_argument("--hold", type=int, default=6, help="physics steps each random torque is held")
parser.add_argument("--resample", type=int, default=32, help="physics steps between state re-seeds")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import gymnasium as gym, torch
import pi_adr.tasks  # noqa: F401
from pi_adr.tasks.two_dof_reach.two_dof_reach_env_cfg import TwoDofReachEnvCfg, VELOCITY_LIMIT

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Training/val/test share one physics distribution and differ only by seed, so
# the split is across configurations. "extrap" deliberately lies outside it.
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
    n, t = env.num_envs, args_cli.steps

    q_buf = torch.zeros(n, t, 2, device=device)
    qd_buf = torch.zeros(n, t, 2, device=device)
    tau_buf = torch.zeros(n, t, 2, device=device)
    qdd_buf = torch.zeros(n, t, 2, device=device)
    valid_buf = torch.zeros(n, t, dtype=torch.bool, device=device)

    # Step the simulator one PHYSICS step at a time rather than through
    # env.step(). A control step holds torque for 4 physics substeps, over
    # which the state changes a lot, so (qd_after-qd_before)/step_dt does NOT
    # satisfy the EOM at the recorded state: measured against the true xi it
    # gave a 78.8 rad/s^2 residual, and an observer trained on it fits label
    # noise instead of physics. Differencing over a single 1/120s step matches
    # the closed form to ~2%.
    robot = env._robot
    jid = env._joint_ids
    action = torch.zeros(n, 2, device=device)

    with torch.inference_mode():
        for step in range(t):
            if step % args_cli.hold == 0:
                action = torch.empty(n, 2, device=device).uniform_(-1.0, 1.0) * args_cli.torque_scale
            if step % args_cli.resample == 0:
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
            # PhysX ramps the velocity constraint in from ~4 rad/s: median model
            # error jumps from 4.2 rad/s^2 below 4 rad/s to 68 rad/s^2 in [4,4.9).
            fast = (qd_after.abs() >= args_cli.max_speed) | (qd_before.abs() >= args_cli.max_speed)
            valid_buf[:, step] = ~fast.any(dim=-1)

    window = torch.cat([q_buf, qd_buf, tau_buf, qdd_buf], dim=-1).cpu()
    payload = {
        "window": window, "valid": valid_buf.cpu(), "xi": xi.cpu(),
        "split": args_cli.split, "seed": args_cli.seed,
        "ranges": RANGES["extrap" if args_cli.split == "extrap" else "interp"],
        "torque_scale": args_cli.torque_scale, "step_dt": env.step_dt,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save(payload, os.path.join(DATA_DIR, f"pinn_{args_cli.split}.pt"))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

**Dataset sizes actually collected:** train 4096 envs × 192 steps (seed 0), val 1024 × 192
(seed 7), extrap 1024 × 192 (seed 11) — interpolation and extrapolation splits use *entirely
different, non-overlapping* `DomainRandCfg` ranges (table in the code above), so extrapolation is
a genuinely unseen physics region, not a held-out subset of the same distribution. Usable-fraction
after the `max_speed=3.5` mask: 64.0% (train), 64.3% (val), 97.3% (extrap — narrower ranges rarely
saturate).

### 12c. Model (`source/pi_adr/pinn/model.py`) — full file, architecture

```python
"""Physics observer: infer the physics vector xi from observed transitions."""

from __future__ import annotations
import torch
import torch.nn as nn
from .dynamics import XI_NAMES, forward_dynamics, physics_residual

# Physically admissible ranges for xi = [m1, m2, f1, f2, b1, b2, a1, a2].
XI_BOUNDS = torch.tensor([
    [0.25, 4.0], [0.10, 2.0],   # m1, m2
    [0.0, 1.0], [0.0, 1.0],     # f1, f2
    [0.0, 0.5], [0.0, 0.5],     # b1, b2
    [0.2, 1.2], [0.2, 1.2],     # a1, a2
])


class PhysicsObserver(nn.Module):
    """(B, T, 8) transitions -> xi estimate (B, 8) with per-parameter log-variance."""

    def __init__(self, hidden: int = 256, embed: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, embed),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * embed, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 2 * len(XI_NAMES)),
        )
        self.register_buffer("bounds", XI_BOUNDS.clone())

    def forward(self, window: torch.Tensor):
        h = self.encoder(window)                       # per-transition embedding
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)  # set pooling: mean + max
        out = self.head(pooled)
        raw, log_var = out.chunk(2, dim=-1)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        xi = lo + (hi - lo) * torch.sigmoid(raw)         # hard-bounded output
        return xi, log_var.clamp(-10.0, 5.0)


class ObserverEnsemble(nn.Module):
    """Deep ensemble; predictive variance across members is U_PINN."""

    def __init__(self, num_members: int = 5, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([PhysicsObserver(**kwargs) for _ in range(num_members)])

    def forward(self, window: torch.Tensor):
        xis, log_vars = zip(*(m(window) for m in self.members))
        xi_stack = torch.stack(xis)
        mean = xi_stack.mean(dim=0)
        epistemic = xi_stack.var(dim=0, unbiased=False)      # disagreement between members
        aleatoric = torch.stack(log_vars).exp().mean(dim=0)  # each member's own predicted variance
        return mean, epistemic + aleatoric, epistemic


def normalize_xi(xi, bounds):
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (xi - lo) / (hi - lo)
```

**Input**: window of `T=64` transitions, each `[q, q̇, τ_cmd, q̈] ∈ ℝ⁸` (2+2+2+2), sampled with
probability proportional to validity so no velocity-saturated sample enters the window.
**Output**: both `ξ̂ ∈ ℝ⁸` (physics estimate) **and**, via the ensemble, uncertainty — not `q̈` alone.
`q̈` prediction is a *derived* quantity (`forward_dynamics(q,q̇,τ,ξ̂)`), not a separate network head.
**Architecture**: `8 → 256 → 256 → 128` per-transition encoder (ELU), mean+max pooling to `256`-D,
then `256 → 256 → 256 → 16` head producing `(ξ̂, log σ²)` per member, 5 independently-initialised
members, each seeing an independently resampled batch per step (this is what makes ensemble
disagreement meaningful rather than all 5 members converging to the same function).

---

## 13. PINN mathematical model

Fully explicit, matching the code in §12a. With `q = [q1, q2]`, `q̇ = [q̇1, q̇2]`:

```
M(q) = [ M11   M12 ]
       [ M12   M22 ]

M11 = I1 + I2 + m1·lc1² + m2·(l1² + lc2² + 2·l1·lc2·cos(q2))
M12 = I2 + m2·(lc2² + l1·lc2·cos(q2))
M22 = I2 + m2·lc2²

I1 = m1 · [(1/12)·l1² + 0.25·r²],  I2 = m2 · [(1/12)·l2² + 0.25·r²]   (r = 0.02 m rod radius)

C(q, q̇) q̇ = [ h·(2·q̇1·q̇2 + q̇2²) ]        h = −m2 · l1 · lc2 · sin(q2)
             [ −h·q̇1²             ]

g(q) ≡ 0                                    (planar, verified empirically — §17 check C)

F(q̇) = B·q̇ + f ⊙ tanh(q̇ / ε),  ε = 0.01    (viscous + smoothed-Coulomb friction; B = diag(b1,b2))
```

with `l1 = l2 = 0.5 m`, `lc1 = lc2 = 0.25 m` (COM at link midpoint) fixed and known — **not**
inferred by the PINN, per the roadmap's instruction not to randomise link geometry.

Full equation of motion actually implemented:

```
M(q; ξ) q̈ + C(q, q̇; ξ) + F(q̇; ξ) = α ⊙ τ_cmd
```

This differs from the roadmap's generic `M q̈ + C q̇ + g(q) + F(q̇) = τ` in two code-level ways
worth calling out explicitly for the report:
1. `g(q)` is dropped entirely (proven zero for this geometry, not merely assumed).
2. The right-hand side is `α ⊙ τ_cmd`, not bare `τ` — actuator effectiveness multiplies the
   *commanded* torque, and the observer is given `τ_cmd` (not the already-scaled applied torque)
   specifically so `α` remains identifiable (§1, object 6).

Verified against PhysX (`scripts/validate_dynamics.py`, §17): forward dynamics 2.27% mean
relative error on `q̈`; inverse dynamics 0.036 Nm mean error recovering `τ_cmd` from `(q,q̇,q̈)`
(against torques up to 20 Nm).

---

## 14. PINN loss

`observer_losses` in `source/pi_adr/pinn/model.py`, full function:

```python
def observer_losses(xi_pred, log_var, xi_true, window, bounds):
    """L_data (acceleration), L_phys (EOM residual), L_sup (recovered parameters).

    window packs (q, qd, tau, qdd) so the physics terms can be evaluated at every
    transition against the single xi inferred for that environment.
    """
    q, qd, tau, qdd = window.split(2, dim=-1)
    xi_b = xi_pred.unsqueeze(1).expand(-1, window.shape[1], -1)

    qdd_pred = forward_dynamics(q, qd, tau, xi_b)
    l_data = (qdd_pred - qdd).pow(2).mean()

    l_phys = physics_residual(q, qd, qdd, tau, xi_b).pow(2).mean()

    err = normalize_xi(xi_pred, bounds) - normalize_xi(xi_true, bounds)
    l_sup = (err.pow(2) * torch.exp(-log_var) + log_var).mean()

    return {"data": l_data, "phys": l_phys, "sup": l_sup}
```

Combined as `L = w_data·L_data + w_phys·L_phys + w_sup·L_sup` in `train_pinn.py`, with
`w_data=1.0, w_phys=0.01, w_sup=1.0` (default CLI args, used for the reported run). Note that
`L_data` and `L_phys` are **algebraically related but not identical**: `forward_dynamics` solves
`M q̈_pred = α τ − C − F` for `q̈_pred` (used in `L_data`, an MSE on acceleration), while
`physics_residual` computes `M q̈_observed + C + F − α τ` directly on the *observed* `q̈` (used in
`L_phys`, an MSE on force/torque residual) — the two losses penalise the same underlying
mismatch but in different physical units (rad/s² vs N·m), which is why they need separate,
very different weights (`1.0` vs `0.01`) to balance. `L_sup` is a Gaussian negative-log-likelihood
against the **ground-truth** `ξ` (available here because this is simulation, not real hardware) —
this supervised term is what actually drives most of the learning; `L_phys` at weight 0.01 is
comparatively a light regulariser given the residual's typical magnitude.

`Adam(lr=1e-3)`, `epochs=2000`, `batch=256` environments per gradient step, `win_len=64`
transitions per window, gradient-clipped at norm 1.0. Each of the 5 ensemble members gets an
**independently resampled** batch every step (loop in `train_pinn.py`, shown in full below) —
this is the "bootstrap-style" mechanism that makes the members disagree in physically meaningful
ways rather than collapsing to the same function.

```python
# train_pinn.py — the training loop, verbatim
for epoch in range(1, args.epochs + 1):
    total = 0.0
    opt.zero_grad()
    for member in model.members:
        wm, xm = sample_windows(train_w, train_v, train_x, args.batch, args.win_len)
        xi_hat, log_var = member(wm)
        losses = observer_losses(xi_hat, log_var, xm, wm, bounds)
        loss = args.w_data * losses["data"] + args.w_phys * losses["phys"] + args.w_sup * losses["sup"]
        loss.backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

---

## 15. PINN uncertainty

Ensemble mean/variance, exactly as coded (`ObserverEnsemble.forward`, §12c):

```
ξ̂ = μ_ξ = (1/K) Σ_k ξ̂_k                                    K = 5 members
Σ_epistemic = (1/K) Σ_k (ξ̂_k − μ_ξ)²                         (diagonal, per-parameter; NOT the
                                                               full (K−1)-normalised covariance
                                                               matrix — see note below)
Σ_aleatoric = (1/K) Σ_k exp(log σ²_k)                        (mean of each member's own variance)
U_PINN = Σ_epistemic + Σ_aleatoric                            (total predictive variance, diagonal)
```

**Deviation from the generic textbook formula the user quoted:** the code uses `unbiased=False`
(divide by `K`, i.e. population variance) rather than `1/(K−1)` (sample variance), and computes
only the **diagonal** of the covariance (per-parameter variance, `torch.Tensor.var(dim=0)`), not
the full `(ξ̂_k − μ_ξ)(ξ̂_k − μ_ξ)ᵀ` cross-covariance matrix. This is a simplification — correlated
uncertainty between e.g. `m1` and `α1` (which are not independently identifiable from a single
transition — see §14's algebra) is invisible to this diagonal estimate. This is worth flagging in
the writeup as a candidate explanation for the weak uncertainty↔error correlation reported next.

Rank-correlation check (`evaluate()` in `train_pinn.py`, full function already given in §12
context; the correlation computation itself):

```python
unc = total_var.mean(dim=-1)                 # per-environment scalar uncertainty
err = param_err.mean(dim=-1)                 # per-environment scalar normalised parameter error
rank_corr = torch.corrcoef(torch.stack([
    unc.argsort().argsort().float(),
    err.argsort().argsort().float(),
]))[0, 1]
```

This is a **Spearman rank correlation** (double-argsort trick converts values to ranks, then
Pearson correlation on the ranks), not the raw Pearson correlation on `(U, E)` values themselves —
worth being precise about in the writeup since "rank correlation" and "correlation" are not
interchangeable claims.

**Measured result:** rank correlation `+0.114` (interpolation), `+0.097` (extrapolation) — both
close to zero, i.e. the ensemble's uncertainty is nearly uninformative about its actual error.
Full `results/pinn_observer.json`, reproduced in §16.

---

## 16. All result files

### `results/pinn_observer.json` (full file)

```json
{
  "config": {
    "epochs": 2000, "batch": 256, "win_len": 64, "members": 5, "lr": 0.001,
    "w_data": 1.0, "w_phys": 0.01, "w_sup": 1.0, "device": "cuda:0"
  },
  "validation_interpolation": {
    "acc_rmse_rad_s2": 2.7899460792541504,
    "param_err_normalised_mean": 0.06481142342090607,
    "param_err_abs": {
      "m1": 0.17909622192382812, "m2": 0.05051162838935852,
      "f1": 0.09538891911506653, "f2": 0.1141417920589447,
      "b1": 0.04184558242559433, "b2": 0.017412085086107254,
      "a1": 0.06375525891780853, "a2": 0.052346035838127136
    },
    "uncertainty_error_rank_corr": 0.11404044926166534,
    "mean_epistemic_var": 0.0004597866500262171
  },
  "validation_extrapolation": {
    "acc_rmse_rad_s2": 12.2799072265625,
    "param_err_normalised_mean": 0.20882371068000793,
    "param_err_abs": {
      "m1": 0.5568885803222656, "m2": 0.0974816232919693,
      "f1": 0.30911785364151, "f2": 0.33441638946533203,
      "b1": 0.16448889672756195, "b2": 0.12125124782323837,
      "a1": 0.1696770340204239, "a2": 0.08608835935592651
    },
    "uncertainty_error_rank_corr": 0.09657774865627289,
    "mean_epistemic_var": 0.0006737686926499009
  }
}
```

### `results/rapp_eureka500_envelope.json` — full file already reproduced in §5.

### 2-DOF policy evaluation results — full files

```json
// results/PD_reference.json
{
  "run_name": "PD_reference", "num_episodes": 2048, "metrics_version": "1.0.0",
  "success_rate": {"0.001": 1.0, "0.002": 1.0, "0.005": 1.0, "0.01": 1.0, "0.02": 1.0, "0.05": 1.0},
  "mean_terminal_error_m": 2.3096474706107983e-06,
  "median_terminal_error_m": 1.9073486328125e-06,
  "p90_terminal_error_m": 4.2649612623790745e-06,
  "worst_decile_error_m": 6.689517249469645e-06,
  "mean_joint_velocity_rad_s": 0.15503330528736115,
  "max_joint_velocity_rad_s": 5.095357894897461,
  "mean_control_effort": 0.012781632132828236,
  "mean_abs_action": 0.013858267106115818,
  "mean_settling_time_s": 0.7683432102203369,
  "mean_overshoot_m": 0.002751676831394434,
  "extra": {
    "controller": "IK + PD (kp=12.0, kd=1.2)", "num_envs": 2048, "eval_seed": 1234,
    "timestamp": "2026-09-16T02:54:11",
    "note": "scripted reference: precision ceiling of the environment, not a learned policy"
  }
}

// results/A_vanilla_5000.json
{
  "run_name": "A_vanilla_5000", "num_episodes": 4096, "metrics_version": "1.0.0",
  "success_rate": {"0.001": 0.000244140625, "0.002": 0.00048828125, "0.005": 0.00341796875,
                   "0.01": 0.022216796875, "0.02": 0.1328125, "0.05": 0.930419921875},
  "mean_terminal_error_m": 0.10194563865661621,
  "median_terminal_error_m": 0.03468116372823715,
  "p90_terminal_error_m": 0.04431932419538498,
  "worst_decile_error_m": 0.7400281429290771,
  "mean_joint_velocity_rad_s": 1.533890962600708,
  "max_joint_velocity_rad_s": 5.14858865737915,
  "mean_control_effort": 0.5144475102424622,
  "mean_abs_action": 0.3164079785346985,
  "mean_settling_time_s": 1.4834635257720947,
  "mean_overshoot_m": 0.15307332575321198,
  "extra": {
    "checkpoint": "logs/rl_games/two_dof_reach/2026-09-16_02-59-42/nn/last_two_dof_reach_ep_5000_rew_1251.8267.pth",
    "domain_rand": {"enabled": false, "joint_friction_range": [0.0, 0.0],
                    "joint_damping_scale_range": [1.0, 1.0],
                    "actuator_effectiveness_range": [1.0, 1.0], "link_mass_scale_range": [1.0, 1.0]},
    "eval_seed": 1234, "eval_wall_clock_s": 7.41,
    "note": "iters=5000 train_envs=4096 train_wall_s=4750 damping=0.05",
    "num_envs": 4096, "reward_variant": "vanilla", "task": "PIADR-TwoDofReach-Vanilla-v0",
    "timestamp": "2026-09-16T04:18:58"
  }
}

// results/B_eureka_500.json
{
  "run_name": "B_eureka_500", "num_episodes": 4096, "metrics_version": "1.0.0",
  "success_rate": {"0.001": 0.00048828125, "0.002": 0.0009765625, "0.005": 0.0078125,
                   "0.01": 0.035888671875, "0.02": 0.255615234375, "0.05": 0.9697265625},
  "mean_terminal_error_m": 0.04639655351638794,
  "median_terminal_error_m": 0.028484957292675972,
  "p90_terminal_error_m": 0.04389268159866333,
  "worst_decile_error_m": 0.22794170677661896,
  "mean_joint_velocity_rad_s": 3.3962819576263428,
  "max_joint_velocity_rad_s": 5.136987686157227,
  "mean_control_effort": 1.4355792999267578,
  "mean_abs_action": 0.8080388903617859,
  "mean_settling_time_s": 1.4036295413970947,
  "mean_overshoot_m": 0.18617045879364014,
  "extra": {
    "checkpoint": "logs/rl_games/two_dof_reach/2026-09-16_02-59-44/nn/last_two_dof_reach_ep_500_rew_26463.402.pth",
    "domain_rand": {"enabled": false, "joint_friction_range": [0.0, 0.0],
                    "joint_damping_scale_range": [1.0, 1.0],
                    "actuator_effectiveness_range": [1.0, 1.0], "link_mass_scale_range": [1.0, 1.0]},
    "eval_seed": 1234, "eval_wall_clock_s": 7.81,
    "note": "iters=500 train_envs=4096 train_wall_s=470 damping=0.05",
    "num_envs": 4096, "reward_variant": "eureka", "task": "PIADR-TwoDofReach-Eureka-v0",
    "timestamp": "2026-09-16T03:07:40"
  }
}

// results/B_eureka_1500.json
{
  "run_name": "B_eureka_1500", "num_episodes": 4096, "metrics_version": "1.0.0",
  "success_rate": {"0.001": 0.00048828125, "0.002": 0.001220703125, "0.005": 0.014892578125,
                   "0.01": 0.0791015625, "0.02": 0.263671875, "0.05": 0.990478515625},
  "mean_terminal_error_m": 0.03858290612697601,
  "median_terminal_error_m": 0.030080584809184074,
  "p90_terminal_error_m": 0.04149669408798218,
  "worst_decile_error_m": 0.14914962649345398,
  "mean_joint_velocity_rad_s": 2.5221939086914062,
  "max_joint_velocity_rad_s": 5.142526149749756,
  "mean_control_effort": 1.0731234550476074,
  "mean_abs_action": 0.6176400184631348,
  "mean_settling_time_s": 1.2040446996688843,
  "mean_overshoot_m": 0.1951208859682083,
  "extra": {
    "checkpoint": "logs/rl_games/two_dof_reach/2026-09-16_03-08-07/nn/last_two_dof_reach_ep_1500_rew_26299.344.pth",
    "domain_rand": {"enabled": false, "joint_friction_range": [0.0, 0.0],
                    "joint_damping_scale_range": [1.0, 1.0],
                    "actuator_effectiveness_range": [1.0, 1.0], "link_mass_scale_range": [1.0, 1.0]},
    "eval_seed": 1234, "eval_wall_clock_s": 9.39,
    "note": "iters=1500 train_envs=4096 train_wall_s=1426 damping=0.05",
    "num_envs": 4096, "reward_variant": "eureka", "task": "PIADR-TwoDofReach-Eureka-v0",
    "timestamp": "2026-09-16T03:31:59"
  }
}

// results/A_vanilla_5000_DRtrain.json  (headline fields; physics_performance_map included)
{
  "run_name": "A_vanilla_5000_DRtrain",
  "success_rate": {"0.001": 0.0, "0.002": 0.000244140625, "0.005": 0.004150390625,
                   "0.01": 0.02734375, "0.02": 0.1298828125, "0.05": 0.92138671875},
  "mean_terminal_error_m": 0.10882224887609482, "median_terminal_error_m": 0.033580370247364044,
  "p90_terminal_error_m": 0.04438003525137901, "worst_decile_error_m": 0.8123977184295654,
  "mean_joint_velocity_rad_s": 0.7446833848953247, "max_joint_velocity_rad_s": 5.134639739990234,
  "mean_control_effort": 0.21476830542087555, "mean_abs_action": 0.13147008419036865,
  "mean_settling_time_s": 1.6973145008087158, "mean_overshoot_m": 0.163070946931839,
  "extra": {
    "domain_rand": {"enabled": true, "joint_friction_range": [0.0, 0.25],
                    "joint_damping_scale_range": [0.5, 4.0],
                    "actuator_effectiveness_range": [0.5, 1.0], "link_mass_scale_range": [1.0, 1.5]},
    "note": "in-distribution DR, inside C_RAPP",
    "physics_performance_map": {
      "friction_j1": {"threshold": 0.01,
        "bin_centers": [0.0157, 0.0469, 0.0781, 0.1094, 0.1406, 0.1719, 0.2031, 0.2343],
        "success_rate": [0.0183, 0.0293, 0.0327, 0.0333, 0.0160, 0.0277, 0.0314, 0.0308],
        "counts": [547, 547, 520, 480, 500, 505, 510, 487]},
      "effectiveness_j1": {"threshold": 0.01,
        "bin_centers": [0.5314, 0.5939, 0.6563, 0.7188, 0.7813, 0.8438, 0.9063, 0.9687],
        "success_rate": [0.0331, 0.0325, 0.0372, 0.0295, 0.0331, 0.0189, 0.0211, 0.0133],
        "counts": [513, 523, 538, 509, 484, 530, 474, 525]}
    }
  }
}

// results/A_vanilla_5000_DRshift.json
{
  "run_name": "A_vanilla_5000_DRshift",
  "success_rate": {"0.001": 0.0, "0.002": 0.00048828125, "0.005": 0.006103515625,
                   "0.01": 0.0263671875, "0.02": 0.1318359375, "0.05": 0.806884765625},
  "mean_terminal_error_m": 0.19475868344306946, "median_terminal_error_m": 0.03516404330730438,
  "p90_terminal_error_m": 0.8920875787734985, "worst_decile_error_m": 1.2066450119018555,
  "mean_joint_velocity_rad_s": 1.1689486503601074, "max_joint_velocity_rad_s": 5.065195083618164,
  "mean_control_effort": 0.3786203861236572, "mean_abs_action": 0.2334509938955307,
  "mean_settling_time_s": 3.2807374000549316, "mean_overshoot_m": 0.2246631681919098,
  "extra": {
    "domain_rand": {"enabled": true, "joint_friction_range": [0.25, 0.5],
                    "joint_damping_scale_range": [4.0, 8.0],
                    "actuator_effectiveness_range": [0.3, 0.5], "link_mass_scale_range": [1.5, 2.5]},
    "note": "shifted unseen DR distribution"
  }
}

// results/B_eureka_1500_DRtrain.json
{
  "run_name": "B_eureka_1500_DRtrain",
  "success_rate": {"0.001": 0.00048828125, "0.002": 0.00390625, "0.005": 0.026611328125,
                   "0.01": 0.093017578125, "0.02": 0.348876953125, "0.05": 0.978759765625},
  "mean_terminal_error_m": 0.045941319316625595, "median_terminal_error_m": 0.02511114627122879,
  "p90_terminal_error_m": 0.03761328011751175, "worst_decile_error_m": 0.2578333020210266,
  "mean_joint_velocity_rad_s": 1.279691457748413, "max_joint_velocity_rad_s": 5.125349521636963,
  "mean_control_effort": 0.4412001073360443, "mean_abs_action": 0.27549973130226135,
  "mean_settling_time_s": 1.5178873538970947, "mean_overshoot_m": 0.20002104341983795,
  "extra": {
    "domain_rand": {"enabled": true, "joint_friction_range": [0.0, 0.25],
                    "joint_damping_scale_range": [0.5, 4.0],
                    "actuator_effectiveness_range": [0.5, 1.0], "link_mass_scale_range": [1.0, 1.5]},
    "note": "in-distribution DR, inside C_RAPP"
  }
}

// results/B_eureka_1500_DRshift.json
{
  "run_name": "B_eureka_1500_DRshift",
  "success_rate": {"0.001": 0.0009765625, "0.002": 0.004638671875, "0.005": 0.027099609375,
                   "0.01": 0.11474609375, "0.02": 0.423583984375, "0.05": 0.911376953125},
  "mean_terminal_error_m": 0.10699759423732758, "median_terminal_error_m": 0.022388767451047897,
  "p90_terminal_error_m": 0.043578553944826126, "worst_decile_error_m": 0.8804075121879578,
  "mean_joint_velocity_rad_s": 0.9507901072502136, "max_joint_velocity_rad_s": 5.046777725219727,
  "mean_control_effort": 0.34692248702049255, "mean_abs_action": 0.20794519782066345,
  "mean_settling_time_s": 2.5312094688415527, "mean_overshoot_m": 0.23771324753761292,
  "extra": {
    "domain_rand": {"enabled": true, "joint_friction_range": [0.25, 0.5],
                    "joint_damping_scale_range": [4.0, 8.0],
                    "actuator_effectiveness_range": [0.3, 0.5], "link_mass_scale_range": [1.5, 2.5]},
    "note": "shifted unseen DR distribution"
  }
}
```

### LEAP Hand (track B) — full result

Training complete: 500 PPO iterations, 4096 environments, seed 42, upstream config unmodified,
final return **84,254.54** (up from 33.09 at a 5-iteration smoke test). Evaluated with
`scripts/eval_leap.py`, 512 envs, 600 control steps (20 s per env at 30 Hz control):

```json
// results/LEAP_500.json
{
  "run_name": "LEAP_500", "task": "Isaac-Reorient-Cube-Leap",
  "checkpoint": ".../leap_hand_reorient/2026-09-16_03-32-28/nn/last_leap_hand_reorient_ep_500_rew__84254.54_.pth",
  "num_envs": 512, "eval_seed": 1234, "control_steps": 600,
  "sim_seconds_per_env": 20.0, "z_rotation_steps": 16,
  "total_goal_reaches": 61580.0,
  "rotations_per_env": 7.51708984375,
  "rotations_per_20s_per_env": 7.51708984375,
  "mean_joint_velocity_rad_s": 1.8626192808151245,
  "mean_control_effort_per_joint": 0.6963677406311035,
  "mean_abs_action": 0.7764208316802979,
  "note": "500 PPO iterations, 4096 train envs, upstream config unmodified",
  "timestamp": "2026-09-16T04:59:02"
}
```

**≈7.5 rotations per 20 s per hand**, DrEureka's own reported metric for this exact task, after
only 500 PPO iterations (DrEureka trains substantially longer). This is a rung-1 reproduction
baseline (stock config, no adaptive DR, no PINN) — a number to beat, not a final result.
`z_rotation_steps=16` means one full rotation = 16 discrete goal reaches
(`rotations = total_goal_reaches / z_rotation_steps / num_envs`, exactly as computed in
`scripts/eval_leap.py`).

### `data/` directory (PINN datasets and trained weights, not checkpoints)

- `data/pinn_train.pt`, `data/pinn_val.pt`, `data/pinn_extrap.pt` — raw transition tensors
  (`window`, `valid`, `xi`, split metadata), ~17 MB / 4.4 MB / 4.4 MB.
- `data/pinn_observer.pt` — trained `ObserverEnsemble` state dict (5 members).

---

## 17. Environment validation scripts

### `scripts/validate_env.py` — 10 checks, full file (already given verbatim; key excerpt showing
how the analytic-vs-PhysX and reward checks are implemented):

```python
# Check B: sim end-effector == analytic FK
def analytic_fk(q: torch.Tensor) -> torch.Tensor:
    x = LINK_LENGTH_1 * torch.cos(q[:, 0]) + LINK_LENGTH_2 * torch.cos(q[:, 0] + q[:, 1])
    y = LINK_LENGTH_1 * torch.sin(q[:, 0]) + LINK_LENGTH_2 * torch.sin(q[:, 0] + q[:, 1])
    return torch.stack([x, y], dim=-1)
# ... compared against env._ee_pos_local() -> max error 0.000 mm

# Check H: reward formulas match verbatim
expected_vanilla = -d + 5.0 * (d < cfg.success_radius).float() - 0.005 * a.pow(2).sum(-1)
expected_eureka = (torch.exp(-10.0 * d) + 100.0 * (d < cfg.success_radius).float()
                   - 0.01 * qd.pow(2).sum(-1) - 0.1)
# ... compared against env._get_rewards() -> max deviation 0.00e+00

# Check J: per-joint actuator effectiveness enters the coupled dynamics correctly
qdd_sim = env._robot.data.joint_vel[:, env._joint_ids] / env.physics_dt
qdd_pred = forward_dynamics(zero, zero, env.commanded_torque, env.physics_xi)
rel = ((qdd_sim - qdd_pred).abs() / qdd_pred.abs().clamp_min(1.0)).mean().item()
# -> 3.99% mean relative deviation, over alpha in [0.3, 1.0]
```

**Full result (10/10 passed):**

| Check | Result |
|---|---|
| A. 30 Hz control, 120 Hz physics, 300 steps/episode | `step_dt=0.033333s`, 300 steps |
| B. sim end-effector == analytic FK | max error **0.000 mm** |
| C. zero torque → no drift (`g(q)=0`) | `\|q\|=0`, `\|q̇\|=0` after 1 s |
| D. `a=+1` → 20 Nm minus drive damping, positive rotation | applied 19.866 Nm, expected 19.844 Nm |
| E. `\|q̇\|` within ±5 rad/s under saturated torque | peak 5.138 rad/s |
| F. targets `r~U(0.20,0.85)` | r ∈ [0.201, 0.847] m |
| G. 8-D observation layout | shape `(N,8)`, exact match |
| H. vanilla and Eureka rewards match eqs. 7/8 | max deviation `0.00` |
| I. per-env DR reaches the simulator, differs per env | friction std 0.059 |
| J. actuator effectiveness enters coupled dynamics | 3.99% mean relative deviation |

### `scripts/validate_dynamics.py` — full file (already given verbatim above in the earlier turn;
key excerpt):

```python
xi = torch.tensor([m1, m2, f_sim, f_sim, b_sim, b_sim, 1.0, 1.0], device=device)
q = torch.empty(args_cli.num_envs, 2, device=device).uniform_(-2.5, 2.5)
qd = torch.empty(args_cli.num_envs, 2, device=device).uniform_(-1.0, 1.0)
tau = torch.empty(args_cli.num_envs, 2, device=device).uniform_(-5.0, 5.0)

robot.write_joint_state_to_sim(q, qd, joint_ids=jid)
robot.set_joint_effort_target(tau, joint_ids=jid)
robot.write_data_to_sim()
env.sim.step(render=False)
env.scene.update(dt=env.physics_dt)

qdd_sim = (robot.data.joint_vel[:, jid] - qd) / env.physics_dt
qdd_analytic = forward_dynamics(q, qd, tau, xi)
# mean rel error 2.27%; inverse-dynamics recovered-torque mean abs error 0.036 Nm
```

**Both scripts are re-run after every environment change** — this is the mechanism that caught
defects 2–5 (damping override, joint-limit trap discovered via `pd_baseline.py` not this script
but in the same validation discipline, PINN label error, mass-write-skip bug).

---

## 18. Versions / environment

| Component | Version |
|---|---|
| GPU | 1× NVIDIA L4, 23 GB, driver 595.84 |
| OS | Ubuntu 26.04 |
| Python | 3.11 (`env_isaaclab` venv) |
| Isaac Sim | 5.1.0.0 (pip) |
| Isaac Lab | 2.3.0, commit `3c6e67bb5` (2025-10-28) |
| IsaacLabEureka | commit `716657b` (2025-10-28), patched locally for Gemini backend |
| LEAP_Hand_Isaac_Lab | commit `c576e6a` (2025-09-04), one-line 2.3-compat patch |
| rl_games | 1.6.1 |
| PyTorch | 2.7.0+cu128 |
| gymnasium | 1.2.0 |
| hydra-core | 1.3.5 |

No `requirements.txt`/`environment.yml` exists for `pi_adr` itself (it rides on the already-built
`env_isaaclab` venv); `pi_adr/pyproject.toml` declares it as an editable package with no pinned
deps beyond what Isaac Lab already provides.

---

## 19. What adaptive DR would need (not built)

Per the roadmap's own instruction, this was not implemented. Recorded here only because the user
asked what the evidence says it *should* look like, derived — not invented — from the measurements
above:

- **The uncertainty term is currently unusable.** §15's rank correlation (+0.11) means
  `w(ξ) = α_u·U_PINN(ξ) + ...` would inject near-noise unless the ensemble covariance is made
  non-diagonal (the two terms most likely to be conflated — `m` and `α` — are exactly the pair
  whose product determines acceleration, so they are not independently identifiable from a
  single-torque-regime window; multiple excitation regimes per environment may be required) or
  the term is dropped (`α_u = 0`).
- **RAPP's envelope is very wide and mostly flat** (§5) — friction and damping are feasible across
  2+ orders of magnitude with success barely moving. Per the roadmap's own §29 "negative result"
  criterion, this specific finding (*"RAPP already covers a region in which policy performance is
  nearly flat"*) is one of the two examples it gives for when adaptive DR would have nothing to
  exploit. That should be checked directly (e.g. compute `‖∇_ξ J‖` numerically from the RAPP table
  — it is available today, no new infrastructure needed) before adaptive DR is built.
- **The one dimension with a real, sharp, and useful gradient is mass** (§5: infeasible below 1×,
  feasible up to 4×) and it is exactly the dimension the PINN's per-parameter error is *worst* on
  in absolute terms (m1: 0.179 kg abs error, largest of all 8 parameters) — the one place adaptive
  DR would most want good physics information is the one place the current observer is weakest.
