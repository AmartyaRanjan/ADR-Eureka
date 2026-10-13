"""Frozen-observer probe of the CURRENT adaptive-DR training distribution (rung G's observation step).

For each DR dimension, with only that dimension randomised over the controller's current range
(others nominal -- RAPP's one-parameter discipline), run the validated excitation protocol
(transient bursts + sequential single-joint long holds, physics-step differencing, velocity mask),
feed windows to the frozen PINN ensemble, and report how uncertain the observer is about the
parameters belonging to that dimension.

Reported per dimension (all in units normalised by the observer's own XI_BOUNDS so dimensions are
comparable): mean epistemic variance, mean total predictive variance, and -- for diagnostics only,
never used by the controller -- the actual mean absolute estimation error against ground truth.

    env_isaaclab/bin/python pi_adr/scripts/pinn_probe.py --headless --observer_tag seq \
        --ranges_json '{"friction":[0,0.25],...}' --out probe.json
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ranges_json", type=str, required=True)
parser.add_argument("--observer_tag", type=str, default="seq")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=96, help="transient transitions per env")
parser.add_argument("--long_hold_steps", type=int, default=200, help="sequential long-hold steps per env (half per joint)")
parser.add_argument("--torque_scale", type=float, nargs="+", default=[0.15, 0.45])
parser.add_argument("--target_velocity", type=float, default=2.0)
parser.add_argument("--max_speed", type=float, default=3.5)
parser.add_argument("--win_len", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import pi_adr.tasks  # noqa: F401, E402
from pi_adr.evaluation.runner import set_domain_rand  # noqa: E402
from pi_adr.pinn.model import XI_BOUNDS, ObserverEnsemble, normalize_xi  # noqa: E402
from pi_adr.tasks.two_dof_reach.two_dof_reach_env_cfg import TwoDofReachEnvCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# xi = [m1, m2, f1, f2, b1, b2, a1, a2] -> which indices belong to each controller dimension
DIM_PARAMS = {"mass_scale": [0, 1], "friction": [2, 3], "damping_scale": [4, 5], "effectiveness": [6, 7]}


@torch.no_grad()
def collect(env, device):
    """Same protocol as collect_pinn_data.py (transient bursts, then sequential single-joint holds)."""
    robot, jid = env._robot, env._joint_ids
    n = env.num_envs
    t = args_cli.steps + args_cli.long_hold_steps
    buf = torch.zeros(n, t, 8, device=device)
    valid = torch.zeros(n, t, dtype=torch.bool, device=device)
    scales = torch.tensor(args_cli.torque_scale, device=device)

    def step_once(k, tau_cmd):
        q0 = env._wrapped_joint_pos().clone()
        qd0 = robot.data.joint_vel[:, jid].clone()
        robot.set_joint_effort_target(tau_cmd * env._actuator_effectiveness, joint_ids=jid)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        qd1 = robot.data.joint_vel[:, jid]
        buf[:, k] = torch.cat([q0, qd0, tau_cmd, (qd1 - qd0) / env.physics_dt], dim=-1)
        fast = (qd1.abs() >= args_cli.max_speed) | (qd0.abs() >= args_cli.max_speed)
        valid[:, k] = ~fast.any(dim=-1)

    action = torch.zeros(n, 2, device=device)
    for k in range(args_cli.steps):
        if k % 6 == 0:
            action = torch.empty(n, 2, device=device).uniform_(-1, 1) * scales[torch.randint(0, len(scales), (n, 1), device=device)]
        if k % 32 == 0:
            robot.write_joint_state_to_sim(torch.empty(n, 2, device=device).uniform_(-torch.pi, torch.pi),
                                           torch.empty(n, 2, device=device).uniform_(-1.5, 1.5), joint_ids=jid)
        step_once(k, env.cfg.torque_limit * action)

    if args_cli.long_hold_steps > 0:
        robot.write_joint_state_to_sim(torch.empty(n, 2, device=device).uniform_(-torch.pi, torch.pi),
                                       torch.zeros(n, 2, device=device), joint_ids=jid)
        b = env._joint_damping.clamp_min(1e-3)
        a = env._actuator_effectiveness.clamp_min(0.2)
        scale = (args_cli.target_velocity * b / (a * env.cfg.torque_limit)).clamp(0.01, 1.0)
        hold = torch.sign(torch.empty(n, 2, device=device).uniform_(-1, 1)) * scale
        half = args_cli.long_hold_steps // 2
        for k in range(args_cli.long_hold_steps):
            drive = torch.zeros(1, 2, device=device)
            drive[0, 0 if k < half else 1] = 1.0
            step_once(args_cli.steps + k, env.cfg.torque_limit * hold * drive)
    return buf, valid


def main():
    torch.manual_seed(args_cli.seed)
    device = torch.device("cuda:0")
    ranges = json.loads(args_cli.ranges_json)

    model = ObserverEnsemble(num_members=5).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, "data", f"pinn_observer_{args_cli.observer_tag}.pt"), map_location=device))
    model.eval()
    bounds = XI_BOUNDS.to(device)
    width2 = (bounds[:, 1] - bounds[:, 0]).pow(2)

    cfg = TwoDofReachEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    cfg.domain_rand.enabled = True
    env = gym.make("PIADR-TwoDofReach-Vanilla-v0", cfg=cfg).unwrapped

    result = {}
    for dim, (lo, hi) in ranges.items():
        set_domain_rand(env, **{dim: (lo, hi)})
        with torch.inference_mode():
            env.reset()
            xi = env.physics_xi.clone()
            buf, valid = collect(env, device)
        buf, valid, xi = buf.clone(), valid.clone(), xi.clone()

        # sample win_len valid transitions per env, exactly as in training
        picks = torch.multinomial(valid.float() + 1e-6, args_cli.win_len, replacement=True)
        w = torch.gather(buf, 1, picks.unsqueeze(-1).expand(-1, -1, 8))
        with torch.no_grad():
            mean, total_var, epistemic = model(w)

        idx = DIM_PARAMS[dim]
        result[dim] = {
            "range": [lo, hi],
            "epistemic_var_norm": float((epistemic / width2)[:, idx].mean()),
            "total_var_norm": float((total_var / width2)[:, idx].mean()),
            "abs_err_norm_diagnostic": float((normalize_xi(mean, bounds) - normalize_xi(xi, bounds)).abs()[:, idx].mean()),
            "usable_frac": float(valid.float().mean()),
        }
        print(f"  {dim:<14} range=[{lo:.3f},{hi:.3f}]  epi_var={result[dim]['epistemic_var_norm']:.5f} "
              f"abs_err={result[dim]['abs_err_norm_diagnostic']:.4f} usable={result[dim]['usable_frac']:.2f}", flush=True)

    with open(args_cli.out, "w") as fh:
        json.dump(result, fh, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
