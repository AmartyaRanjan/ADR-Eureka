"""Per-dimension success S_d under the CURRENT training range — the adaptive-DR
controller's observation step.

One parameter randomised at a time (others held at nominal), exactly RAPP's
discipline, except the swept "value" is the controller's current continuous
range [lo, hi] rather than a fixed candidate grid. Writes a small JSON the
orchestrator (run_adaptive_dr.py) reads back; no training happens here.

    env_isaaclab/bin/python pi_adr/scripts/eval_dr_dims.py --headless \
        --checkpoint <ckpt> --task PIADR-TwoDofReach-Eureka-v0 \
        --ranges_json '{"friction":[0.0,0.25], ...}' --out results/adaptive_dr/step_0.json
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--ranges_json", type=str, required=True, help="JSON dict: {dim: [lo, hi]}")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--success_eps", type=float, default=0.05)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import pi_adr.tasks  # noqa: F401, E402
from pi_adr.evaluation.runner import make_env_and_agent, rollout, set_domain_rand  # noqa: E402


def main() -> None:
    ranges = json.loads(args_cli.ranges_json)
    env, base_env, agent, _ = make_env_and_agent(
        args_cli.task, args_cli.checkpoint, args_cli.num_envs, args_cli.seed, args_cli.device
    )

    success_per_dim = {}
    mean_err_per_dim = {}
    for dim, (lo, hi) in ranges.items():
        set_domain_rand(base_env, **{dim: (lo, hi)})
        _, accum = rollout(env, base_env, agent, f"adim_{dim}")
        success_per_dim[dim] = float((accum.terminal_distance < args_cli.success_eps).float().mean().item())
        mean_err_per_dim[dim] = float(accum.terminal_distance.mean().item())
        print(f"  {dim:<16} range=[{lo:.4f},{hi:.4f}]  S={success_per_dim[dim]:.4f}  "
              f"mean_err={mean_err_per_dim[dim]*1000:.2f}mm", flush=True)

    with open(args_cli.out, "w") as fh:
        json.dump({"ranges": ranges, "success_per_dim": success_per_dim, "mean_err_per_dim": mean_err_per_dim}, fh, indent=2)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
