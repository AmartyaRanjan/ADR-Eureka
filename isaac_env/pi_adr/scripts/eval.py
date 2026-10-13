"""Evaluate a trained 2-DOF reacher policy with the frozen metric suite.

One evaluation == one full 300-step episode per environment, deterministic
actions, fixed seed. Results go to results/<run_name>.json and are appended to
results/index.jsonl so every number ever produced stays comparable.

    env_isaaclab/bin/python pi_adr/scripts/eval.py \
        --task PIADR-TwoDofReach-Vanilla-v0 --checkpoint <path.pth> \
        --run_name A_vanilla_nominal --headless
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

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import pi_adr.tasks  # noqa: F401, E402
from pi_adr.evaluation.metrics import physics_binned_metrics, physics_binned_success  # noqa: E402
from pi_adr.evaluation.runner import make_env_and_agent, rollout, set_domain_rand  # noqa: E402

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
        env,
        base_env,
        agent,
        args_cli.run_name,
        extra={
            "task": args_cli.task,
            "checkpoint": args_cli.checkpoint,
            "num_envs": args_cli.num_envs,
            "eval_seed": args_cli.seed,
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
        # Full M(xi) = [P_5cm,P_2cm,P_1cm,E_d,E_qdot,E_tau,T_settle] vector per bin
        # (roadmap-complete version of the line above, kept alongside it rather
        # than replacing it so existing readers of physics_performance_map are
        # unaffected).
        result.extra["physics_performance_map_full"] = {
            "friction_j1": physics_binned_metrics(accum, physics[:, 0]),
            "effectiveness_j1": physics_binned_metrics(accum, physics[:, 4]),
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args_cli.run_name}.json")
    result.to_json(out_path)
    with open(os.path.join(RESULTS_DIR, "index.jsonl"), "a") as fh:
        fh.write(
            json.dumps(
                {
                    "run_name": result.run_name,
                    "success_rate": result.success_rate,
                    "mean_terminal_error_m": result.mean_terminal_error_m,
                    "median_terminal_error_m": result.median_terminal_error_m,
                    "mean_joint_velocity_rad_s": result.mean_joint_velocity_rad_s,
                    "mean_control_effort": result.mean_control_effort,
                    **{k: v for k, v in result.extra.items() if k != "physics_performance_map"},
                }
            )
            + "\n"
        )

    print("\n================ EVAL RESULT ================")
    print(result.summary_line())
    print(f"  precision curve: {result.success_rate}")
    print(f"  p90={result.p90_terminal_error_m * 1000:.1f}mm worst-decile={result.worst_decile_error_m * 1000:.1f}mm")
    print(f"  settling={result.mean_settling_time_s:.2f}s overshoot={result.mean_overshoot_m * 1000:.1f}mm")
    print(f"  max|qd|={result.max_joint_velocity_rad_s:.2f} rad/s")
    print(f"  written to {out_path}")
    print("=============================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
