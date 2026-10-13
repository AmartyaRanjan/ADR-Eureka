"""Physics/task validation for the 2-DOF reacher.

Checks the environment actually does what the report's equations say before any
result produced with it is trusted. Run after every change to the env or asset:

    env_isaaclab/bin/python pi_adr/scripts/validate_env.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import pi_adr.tasks  # noqa: F401, E402
from pi_adr.pinn.dynamics import forward_dynamics  # noqa: E402
from pi_adr.tasks.two_dof_reach.two_dof_reach_env_cfg import (  # noqa: E402
    TwoDofReachEnvCfg,
    LINK_LENGTH_1,
    LINK_LENGTH_2,
    TORQUE_LIMIT,
    VELOCITY_LIMIT,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")


def analytic_fk(q: torch.Tensor) -> torch.Tensor:
    x = LINK_LENGTH_1 * torch.cos(q[:, 0]) + LINK_LENGTH_2 * torch.cos(q[:, 0] + q[:, 1])
    y = LINK_LENGTH_1 * torch.sin(q[:, 0]) + LINK_LENGTH_2 * torch.sin(q[:, 0] + q[:, 1])
    return torch.stack([x, y], dim=-1)


def main() -> int:
    cfg = TwoDofReachEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.domain_rand.enabled = False

    env = gym.make("PIADR-TwoDofReach-Vanilla-v0", cfg=cfg).unwrapped
    device = env.device
    n = env.num_envs
    zero = torch.zeros(n, 2, device=device)

    env.reset()

    # --- A. control/sim rates -------------------------------------------------
    check(
        "A. control rate 30 Hz, physics 120 Hz, 300 steps/episode",
        abs(1.0 / env.step_dt - 30.0) < 1e-6 and int(env.max_episode_length) == 300,
        f"step_dt={env.step_dt:.6f}s ({1.0 / env.step_dt:.1f} Hz), "
        f"max_episode_length={int(env.max_episode_length)}",
    )

    # --- B. forward kinematics matches the report's equations -----------------
    q_test = torch.tensor(
        [[0.0, 0.0], [math.pi / 2, 0.0], [0.0, math.pi / 2], [math.pi / 4, -math.pi / 2]],
        device=device,
    ).repeat(max(1, n // 4), 1)[:n]
    env._robot.write_joint_state_to_sim(q_test, torch.zeros_like(q_test), joint_ids=env._joint_ids)
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)

    fk_err = (env._ee_pos_local() - analytic_fk(q_test)).norm(dim=-1)
    check(
        "B. sim end-effector == analytic FK",
        fk_err.max().item() < 2e-3,
        f"max error {fk_err.max().item() * 1000:.3f} mm",
    )

    # --- C. no gravity torque in the plane (arm holds still at zero torque) ---
    env._robot.write_joint_state_to_sim(zero, zero, joint_ids=env._joint_ids)
    for _ in range(30):
        env.step(zero)
    q_drift = env._robot.data.joint_pos[:, env._joint_ids].abs().max().item()
    qd_drift = env._robot.data.joint_vel[:, env._joint_ids].abs().max().item()
    check(
        "C. zero torque -> no drift (planar, g(q)=0)",
        q_drift < 1e-2 and qd_drift < 1e-2,
        f"max |q|={q_drift:.2e} rad, max |qd|={qd_drift:.2e} rad/s after 1 s",
    )

    # --- D. torque sign, magnitude and effort limit ---------------------------
    # applied_torque is the net joint torque, so the implicit drive's damping
    # term (-b*qd) is already subtracted from the commanded effort.
    env._robot.write_joint_state_to_sim(zero, zero, joint_ids=env._joint_ids)
    env.step(torch.cat([torch.ones(n, 1, device=device), torch.zeros(n, 1, device=device)], dim=-1))
    applied = env._robot.data.applied_torque[:, env._joint_ids]
    q_after = env._robot.data.joint_pos[:, env._joint_ids]
    damping = env._robot.data.joint_damping[:, env._joint_ids]
    qd_after = env._robot.data.joint_vel[:, env._joint_ids]
    expected_d = TORQUE_LIMIT - damping[:, 0] * qd_after[:, 0]
    check(
        "D. a=+1 -> 20 Nm minus drive damping on joint 1, positive rotation",
        (applied[:, 0] - expected_d).abs().max().item() < 0.05 and q_after[:, 0].mean().item() > 0,
        f"applied={applied[:, 0].mean().item():.3f} Nm, expected={expected_d.mean().item():.3f} Nm, "
        f"dq1={q_after[:, 0].mean().item():.4f} rad",
    )

    # --- E. velocity limit respected under saturated torque -------------------
    env._robot.write_joint_state_to_sim(zero, zero, joint_ids=env._joint_ids)
    peak = 0.0
    for _ in range(120):
        env.step(torch.ones(n, 2, device=device))
        peak = max(peak, env._robot.data.joint_vel[:, env._joint_ids].abs().max().item())
    check(
        "E. joint velocity stays within +/- 5 rad/s",
        peak <= VELOCITY_LIMIT * 1.05,
        f"peak |qd|={peak:.3f} rad/s (limit {VELOCITY_LIMIT})",
    )

    # --- F. target distribution ----------------------------------------------
    env.reset()
    radii = []
    for _ in range(20):
        env._reset_idx(torch.arange(n, device=device))
        local = env._target_pos_w[:, :2] - env.scene.env_origins[:, :2]
        radii.append(local.norm(dim=-1))
    radii = torch.cat(radii)
    lo, hi = cfg.target_radius_range
    check(
        "F. targets sampled in r ~ U(0.20, 0.85), inside 1.0 m reach",
        radii.min().item() >= lo - 1e-3 and radii.max().item() <= hi + 1e-3,
        f"r in [{radii.min().item():.3f}, {radii.max().item():.3f}] m",
    )

    # --- G. observation layout ------------------------------------------------
    obs = env._get_observations()["policy"]
    q = env._robot.data.joint_pos[:, env._joint_ids]
    target_local = env._target_pos_w[:, :2] - env.scene.env_origins[:, :2]
    ok_obs = (
        obs.shape == (n, 8)
        and torch.allclose(obs[:, 0:2], q, atol=1e-5)
        and torch.allclose(obs[:, 4:6], target_local, atol=1e-5)
        and torch.allclose(obs[:, 6:8], target_local - env._ee_pos_local(), atol=1e-5)
    )
    check("G. 8-D observation layout [q, qd, target, target-ee]", ok_obs, f"shape={tuple(obs.shape)}")

    # --- H. reward formulas ---------------------------------------------------
    d = env.target_distance
    a = torch.full((n, 2), 0.5, device=device)
    env._pre_physics_step(a)
    expected_vanilla = -d + 5.0 * (d < cfg.success_radius).float() - 0.005 * a.pow(2).sum(-1)
    got_vanilla = env._get_rewards()
    env.cfg.reward_variant = "eureka"
    qd = env._robot.data.joint_vel[:, env._joint_ids]
    expected_eureka = (
        torch.exp(-10.0 * d) + 100.0 * (d < cfg.success_radius).float() - 0.01 * qd.pow(2).sum(-1) - 0.1
    )
    got_eureka = env._get_rewards()
    env.cfg.reward_variant = "vanilla"
    check(
        "H. vanilla and Eureka rewards match report equations 7 and 8",
        torch.allclose(got_vanilla, expected_vanilla, atol=1e-5)
        and torch.allclose(got_eureka, expected_eureka, atol=1e-5),
        f"max dev {max((got_vanilla - expected_vanilla).abs().max().item(), (got_eureka - expected_eureka).abs().max().item()):.2e}",
    )

    # --- I. domain randomisation is not silent --------------------------------
    env.cfg.domain_rand.enabled = True
    env.cfg.domain_rand.joint_friction_range = (0.0, 0.2)
    env.cfg.domain_rand.joint_damping_scale_range = (0.5, 2.0)
    env.cfg.domain_rand.actuator_effectiveness_range = (0.7, 1.0)
    env._reset_idx(torch.arange(n, device=device))
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)

    friction_sim = env._robot.data.joint_friction_coeff[:, env._joint_ids]
    spread_ok = (
        env.physics_params.std(dim=0).min().item() > 1e-4
        and friction_sim.std().item() > 1e-4
        and torch.allclose(friction_sim, env._joint_friction, atol=1e-5)
    )
    check(
        "I. per-env DR reaches the simulator and differs across envs",
        spread_ok,
        f"friction read-back std={friction_sim.std().item():.4f}, "
        f"range=[{friction_sim.min().item():.3f}, {friction_sim.max().item():.3f}]",
    )

    # --- J. actuator effectiveness scales the applied torque ------------------
    # data.applied_torque is only an approximate reconstruction for implicit
    # actuators, so test the dynamics instead: with per-env, per-joint alpha the
    # joints stay coupled through M(q), so qd is NOT componentwise proportional
    # to alpha. The right check is against the closed-form model.
    env.cfg.domain_rand.joint_friction_range = (0.0, 0.0)
    env.cfg.domain_rand.joint_damping_scale_range = (1.0, 1.0)
    env.cfg.domain_rand.actuator_effectiveness_range = (0.3, 1.0)
    env._reset_idx(torch.arange(n, device=device))

    # Small action so the +/-5 rad/s limit never binds within the step.
    action = torch.full((n, 2), 0.05, device=device)
    env._robot.write_joint_state_to_sim(zero, zero, joint_ids=env._joint_ids)
    env._pre_physics_step(action)
    env._apply_action()
    env._robot.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)

    qdd_sim = env._robot.data.joint_vel[:, env._joint_ids] / env.physics_dt
    qdd_pred = forward_dynamics(zero, zero, env.commanded_torque, env.physics_xi)
    # Same criterion as validate_dynamics.py: a one-step finite difference of an
    # implicit integrator carries a few percent of discretisation error.
    rel = ((qdd_sim - qdd_pred).abs() / qdd_pred.abs().clamp_min(1.0)).mean().item()
    check(
        "J. per-joint actuator effectiveness enters the coupled dynamics",
        rel < 0.05,
        f"mean relative deviation from the closed-form model {rel * 100:.2f}% over alpha in [0.3, 1.0]",
    )

    env.close()

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n==== {n_pass}/{len(RESULTS)} checks passed ====")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} :: {detail}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
