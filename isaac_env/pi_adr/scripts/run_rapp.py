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
