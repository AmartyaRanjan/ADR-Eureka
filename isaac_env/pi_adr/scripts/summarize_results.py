"""Collect results/*.json into one comparison table.

Reads the per-run JSON files rather than the append-only index, so re-running an
evaluation updates the table instead of duplicating a row.

    env_isaaclab/bin/python pi_adr/scripts/summarize_results.py [--markdown]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")

ORDER = [
    "PD_reference",
    "A_vanilla_5000",
    "B_eureka_500",
    "B_eureka_1500",
    "B_eureka_1500_DRtrain",
    "B_eureka_1500_DRshift",
]


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:.2f}%"


def mm(x) -> str:
    return "—" if x is None else f"{x * 1000:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    runs = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.endswith("_envelope") or name.startswith("pinn"):
            continue
        with open(path) as fh:
            runs[name] = json.load(fh)

    twod = {k: v for k, v in runs.items() if "success_rate" in v}
    leap = {k: v for k, v in runs.items() if "rotations_per_20s_per_env" in v}

    ranked = sorted(twod, key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))

    header = ["run", "5cm", "2cm", "1cm", "5mm", "1mm", "mean(mm)", "med(mm)", "p90(mm)", "vel", "effort"]
    rows = []
    for name in ranked:
        r = twod[name]
        s = r.get("success_rate", {})
        rows.append([
            name,
            pct(s.get("0.05")), pct(s.get("0.02")), pct(s.get("0.01")),
            pct(s.get("0.005")), pct(s.get("0.001")),
            mm(r.get("mean_terminal_error_m")), mm(r.get("median_terminal_error_m")),
            mm(r.get("p90_terminal_error_m")),
            f"{r.get('mean_joint_velocity_rad_s', float('nan')):.3f}",
            f"{r.get('mean_control_effort', float('nan')):.3f}",
        ])

    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")
    else:
        widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
                  for i in range(len(header))]
        print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        print("  ".join("-" * w for w in widths))
        for row in rows:
            print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    if leap:
        print("\nLEAP Hand (Isaac-Reorient-Cube-Leap)")
        for name, r in sorted(leap.items()):
            print(
                f"  {name:<28} rotations/20s={r['rotations_per_20s_per_env']:.3f}  "
                f"vel={r['mean_joint_velocity_rad_s']:.3f} rad/s  "
                f"effort/joint={r['mean_control_effort_per_joint']:.3f}"
            )

    env_path = os.path.join(RESULTS_DIR, "rapp_eureka500_envelope.json")
    if os.path.exists(env_path):
        with open(env_path) as fh:
            env = json.load(fh)
        print(f"\nC_RAPP (alpha={env['alpha']}, success<{env['success_eps_m'] * 100:.0f}cm)")
        for param, data in env["envelope"].items():
            print(f"  {param:<16} [{data['lower']}, {data['upper']}]")

        # Fixed-policy physics sweep: terminal error and success against each
        # parameter. This is the physics-performance map the roadmap calls its
        # most valuable diagnostic artifact.
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


if __name__ == "__main__":
    main()
