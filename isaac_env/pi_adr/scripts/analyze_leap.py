"""LEAP matrix summary: rotations/20 s (frozen-ADR eval) per condition, mean±std over seeds. Writes results/MATRIX_LEAP.md."""
import json, os, statistics as st
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS = (42, 123, 7)
C = {"L0": "native ADR off", "L1": "native ADR (upstream)", "L2": "ADR warm-started at 8/25 increments (not RAPP)"}
rows = ["# LEAP Hand matrix (500 PPO iters, 4096 envs, ADR frozen at eval; metric = rotations / 20 s / hand)\n",
        "| cond | description | n | nominal (ADR=0) | full DR (ADR=25) | per-seed nominal / fullDR |", "|---|---|---|---|---|---|"]
for c, desc in C.items():
    nom, full = [], []
    for s in SEEDS:
        a, b = (os.path.join(R, f"LEAP_{c}_s{s}_{v}.json") for v in ("nominal", "fullDR"))
        if os.path.exists(a) and os.path.exists(b):
            nom.append(json.load(open(a))["rotations_per_20s_per_env"]); full.append(json.load(open(b))["rotations_per_20s_per_env"])
    f = lambda x: f"{st.mean(x):.2f} ± {st.stdev(x):.2f}" if len(x) > 1 else (f"{x[0]:.2f}" if x else "—")
    rows.append(f"| {c} | {desc} | {len(nom)} | {f(nom)} | {f(full)} | " + ", ".join(f"{a:.1f}/{b:.1f}" for a, b in zip(nom, full)) + " |")
open(os.path.join(R, "MATRIX_LEAP.md"), "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
