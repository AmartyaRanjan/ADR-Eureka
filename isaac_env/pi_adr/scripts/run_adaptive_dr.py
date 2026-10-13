#!/usr/bin/env python3
"""Orchestrate rung F: RAPP-initialised Adaptive DR, no PINN (see dr/adaptive.py
for why PINN uncertainty is deliberately excluded from this controller).

Pure orchestration — no Isaac Sim app in THIS process. Each training chunk and
each per-dimension evaluation runs as its own subprocess (train.py / eval_dr_dims.py),
exactly like every other script in this project; this script just shells out,
reads results back as JSON, and drives source/pi_adr/dr/adaptive.py.

    env_isaaclab/bin/python pi_adr/scripts/run_adaptive_dr.py \
        --task PIADR-TwoDofReach-Eureka-v0 --num_envs 2048 \
        --chunk_iters 100 --outer_steps 8 --run_name F_adaptive_dr
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = "/data/home/amartyaranjan888/projects/I/isaac_env"
PI_ADR = os.path.join(ROOT, "pi_adr")
PY = os.path.join(ROOT, "env_isaaclab", "bin", "python")

sys.path.insert(0, os.path.join(PI_ADR, "source"))
from pi_adr.dr.adaptive import AdaptiveDRController, DIMENSIONS  # noqa: E402


def sh(cmd: list[str], log_path: str, timeout: int = 1800) -> int:
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, cwd=PI_ADR, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode


def find_checkpoint(train_log: str, iters: int) -> str | None:
    m = re.search(r"Exact experiment name requested from command line: ([0-9_-]+)", open(train_log).read())
    if not m:
        return None
    run_dir = os.path.join(PI_ADR, "logs", "rl_games", "two_dof_reach", m.group(1), "nn")
    if not os.path.isdir(run_dir):
        return None
    cands = [f for f in os.listdir(run_dir) if f"_ep_{iters}_" in f]
    return os.path.join(run_dir, cands[0]) if cands else None


def hydra_dr_overrides(ranges: dict[str, tuple[float, float]]) -> list[str]:
    name_map = {
        "friction": "joint_friction_range",
        "damping_scale": "joint_damping_scale_range",
        "effectiveness": "actuator_effectiveness_range",
        "mass_scale": "link_mass_scale_range",
    }
    out = ["env.domain_rand.enabled=true"]
    for dim, (lo, hi) in ranges.items():
        out.append(f"env.domain_rand.{name_map[dim]}=[{lo},{hi}]")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PIADR-TwoDofReach-Eureka-v0")
    ap.add_argument("--num_envs", type=int, default=2048)
    ap.add_argument("--chunk_iters", type=int, default=100)
    ap.add_argument("--outer_steps", type=int, default=8)
    ap.add_argument("--eval_envs", type=int, default=1024)
    ap.add_argument("--run_name", default="F_adaptive_dr")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--envelope", default=os.path.join(PI_ADR, "results", "rapp_eureka500_envelope.json")
    )
    ap.add_argument(
        "--step_frac", type=float, default=0.25,
        help="fraction of remaining C_RAPP gap to widen per outer step when S_d >= target "
             "(Session-2 D-vs-F run used 0.25, found too aggressive given an 800-iter budget — "
             "see PROJECT_STATUS_AND_ROADMAP.md Priority 1)",
    )
    ap.add_argument("--success_target", type=float, default=0.90)
    ap.add_argument(
        "--gate_threshold", type=float, default=0.05,
        help="precision threshold (metres) the controller's widen/hold decision is gated on. "
             "F1/F2 used 0.05 (coarse success) and found it lets the controller widen through "
             "physics bands where fine precision has already degraded -- see EXPERIMENT_LOG.md, "
             "'F2' entry. 0.02 gates on 2cm precision instead.",
    )
    ap.add_argument("--narrow_frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument(
        "--no_rapp", action="store_true",
        help="rung E (roadmap Table 7): adaptive DR WITHOUT RAPP's feasibility-derived envelope -- "
             "isolates whether RAPP's constraint itself adds value, separate from the adaptive "
             "mechanism. Replaces the measured C_RAPP bounds with wide, arbitrary ones instead of "
             "loading --envelope.",
    )
    ap.add_argument("--pinn_observer", default=None,
                    help="rung G: tag of the frozen PINN observer (data/pinn_observer_<tag>.pt). Enables a per-dimension "
                         "widening-step multiplier from the observer's epistemic uncertainty.")
    ap.add_argument("--pinn_mode", choices=["pinn", "shuffle"], default="pinn",
                    help="'shuffle' = placebo: same multiplier values, randomly permuted across dimensions")
    ap.add_argument("--pinn_strength", type=float, default=1.0, help="multiplier = clip(1+strength*(u/mean(u)-1), 0.5, 1.5)")
    args = ap.parse_args()

    out_dir = os.path.join(PI_ADR, "results", "adaptive_dr")
    log_dir = os.path.join(PI_ADR, "logs", "adaptive_dr")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.no_rapp:
        # Arbitrary wide bounds, NOT derived from any feasibility measurement --
        # that absence is the point of this condition.
        c_rapp = {
            "friction": (0.0, 50.0),
            "damping_scale": (0.1, 1000.0),
            "effectiveness": (0.01, 2.0),
            "mass_scale": (0.1, 20.0),
        }
    else:
        envelope = json.load(open(args.envelope))["envelope"]
        c_rapp = {d: (envelope[d]["lower"], envelope[d]["upper"]) for d in DIMENSIONS}

    # Starting point: the same "in-distribution DR" range already used for the
    # DRtrain/DRshift evaluations, so rung F starts from a point already
    # measured in this project rather than an arbitrary new one.
    initial_ranges = {
        "friction": (0.0, 0.25),
        "damping_scale": (0.5, 4.0),
        "effectiveness": (0.5, 1.0),
        "mass_scale": (1.0, 1.5),
    }

    controller = AdaptiveDRController(
        initial_ranges=initial_ranges, c_rapp=c_rapp,
        success_target=args.success_target, step_frac=args.step_frac,
        narrow_frac=args.narrow_frac, patience=args.patience,
    )

    checkpoint = None
    t_start = time.time()
    for outer in range(args.outer_steps):
        ranges = controller.state.ranges
        print(f"\n=== outer {outer}  ranges={ {k: [round(v[0],4), round(v[1],4)] for k,v in ranges.items()} } ===", flush=True)

        # rl_games restores epoch_num from the checkpoint on --checkpoint
        # resume (a2c_common.py:639) and max_epochs is an absolute target
        # (training loop is `while epoch_num < max_epochs`), so --max_iterations
        # must be the CUMULATIVE target across all chunks so far, not the size
        # of this chunk alone.
        cumulative_iters = args.chunk_iters * (outer + 1)

        train_log = os.path.join(log_dir, f"{args.run_name}_train_{outer}.log")
        cmd = [
            PY, "-u", "scripts/train.py", "--task", args.task, "--headless",
            "--num_envs", str(args.num_envs), "--max_iterations", str(cumulative_iters),
            "--seed", str(args.seed),
        ]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        cmd += hydra_dr_overrides(ranges)

        rc = sh(cmd, train_log)
        new_ckpt = find_checkpoint(train_log, cumulative_iters)
        if rc != 0 or new_ckpt is None:
            print(f"  TRAIN FAILED at outer={outer} (rc={rc}, ckpt_found={new_ckpt}) -- see {train_log}")
            break
        checkpoint = new_ckpt
        print(f"  trained -> {checkpoint}", flush=True)

        eval_out = os.path.join(out_dir, f"{args.run_name}_eval_{outer}.json")
        eval_log = os.path.join(log_dir, f"{args.run_name}_eval_{outer}.log")
        rc = sh([
            PY, "-u", "scripts/eval_dr_dims.py", "--headless",
            "--task", args.task, "--checkpoint", checkpoint,
            "--ranges_json", json.dumps({k: list(v) for k, v in ranges.items()}),
            "--num_envs", str(args.eval_envs), "--out", eval_out,
            "--success_eps", str(args.gate_threshold),
        ], eval_log)
        if rc != 0 or not os.path.isfile(eval_out):
            print(f"  EVAL FAILED at outer={outer} (rc={rc}) -- see {eval_log}")
            break

        measured = json.load(open(eval_out))
        print(f"  measured S_d = {measured['success_per_dim']}", flush=True)
        mult = None
        if args.pinn_observer:
            probe_out = os.path.join(out_dir, f"{args.run_name}_probe_{outer}.json")
            rc = sh([PY, "-u", "scripts/pinn_probe.py", "--headless", "--observer_tag", args.pinn_observer,
                     "--ranges_json", json.dumps({k: list(v) for k, v in ranges.items()}),
                     "--seed", str(args.seed + outer), "--out", probe_out],
                    os.path.join(log_dir, f"{args.run_name}_probe_{outer}.log"))
            if rc != 0 or not os.path.isfile(probe_out):
                print(f"  PROBE FAILED at outer={outer} (rc={rc})")
                break
            u = {d: json.load(open(probe_out))[d]["epistemic_var_norm"] for d in DIMENSIONS}
            mean_u = sum(u.values()) / len(u)
            mult = {d: min(1.5, max(0.5, 1 + args.pinn_strength * (u[d] / mean_u - 1))) for d in DIMENSIONS}
            if args.pinn_mode == "shuffle":
                import random
                vals = list(mult.values()); random.Random(args.seed * 1000 + outer).shuffle(vals)
                mult = dict(zip(DIMENSIONS, vals))
            print(f"  uncertainty={u}  step_multiplier={mult}", flush=True)
        controller.update(measured["success_per_dim"], step_multiplier=mult)

        controller.save(os.path.join(out_dir, f"{args.run_name}_controller.json"))

    wall = time.time() - t_start
    print(f"\n=== adaptive DR run done: {controller.state.outer_iteration} outer steps, {wall:.0f}s wall clock ===")
    print(f"  final checkpoint: {checkpoint}")
    print(f"  final ranges: {controller.state.ranges}")
    controller.save(os.path.join(out_dir, f"{args.run_name}_controller.json"))

    with open(os.path.join(out_dir, f"{args.run_name}_final_checkpoint.txt"), "w") as fh:
        fh.write((checkpoint or "") + "\n")


if __name__ == "__main__":
    main()
