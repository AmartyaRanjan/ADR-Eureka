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
class VanillaRewardCfg:
    """Coefficients for the vanilla (human-designed) reward.

    Defaults reproduce the stage-1 report's equation 7 exactly:
    R = -d + 5*I(d<0.05) - 0.005*||a||^2. Exposed as cfg fields (rather than
    hardcoded) so Session-2's reward-incentive-strength diagnostic
    (PROJECT_STATUS_AND_ROADMAP.md Priority 3) can run as a hydra override --
    e.g. `env.vanilla_reward.action_penalty_coef=0.001` -- without adding a new
    reward_variant branch. Changing these changes the published equation; any
    run using non-default values must say so explicitly in its logged note.
    """

    distance_weight: float = 1.0
    success_bonus: float = 5.0
    action_penalty_coef: float = 0.005


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
    vanilla_reward: VanillaRewardCfg = VanillaRewardCfg()

    domain_rand: DomainRandCfg = DomainRandCfg()


@configclass
class TwoDofReachEurekaEnvCfg(TwoDofReachEnvCfg):
    """Identical environment, Eureka-discovered reward — the only changed variable."""

    reward_variant: str = "eureka"
