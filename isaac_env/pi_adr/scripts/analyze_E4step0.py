"""E4 step-0 analysis exactly as pre-registered (project_docs/18_E4_step0_preregistration.md)."""
import json, os, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "results")
CONDS = ("L0", "L1", "L2"); SEEDS = (42, 123, 7); ADRS = (0, 5, 10, 15, 20, 25)
def g(c, s, a, k):
    p = os.path.join(R, f"LEAPQ_{c}_s{s}_adr{a}.json"); return json.load(open(p))[k] if os.path.exists(p) else None
out = ["# E4 step-0 diagnostic (pre-registered, project_docs/18_E4_step0_preregistration.md)\n"]
# consistency check vs earlier rotations
bad = []
for c in CONDS:
    for s in SEEDS:
        for a in ADRS:
            q = g(c, s, a, "rotations_per_20s_per_env")
            old = {0: f"LEAP_{c}_s{s}_nominal", 25: f"LEAP_{c}_s{s}_fullDR"}.get(a, f"LEAP_{c}_s{s}_adr{a}")
            p = os.path.join(R, old + ".json")
            if q is not None and os.path.exists(p):
                o = json.load(open(p))["rotations_per_20s_per_env"]
                if abs(q - o) > 0.01 * max(o, 1e-9) + 1e-6: bad.append((c, s, a, round(q, 3), round(o, 3)))
out.append(f"Consistency check (rotations reproduce earlier values within 1 %): {'PASSED' if not bad else 'FAILED ' + str(bad[:6])}\n")
for k, lab in (("rotations_per_20s_per_env", "rotations / 20 s"), ("falls_per_20s_per_env", "falls / 20 s")):
    out.append(f"\n## {lab} (mean ± sd over seeds; per-seed in brackets)\n\n| ADR | " + " | ".join(CONDS) + " |\n|---|" + "---|" * 3)
    for a in ADRS:
        cells = []
        for c in CONDS:
            x = [g(c, s, a, k) for s in SEEDS]; x = [v for v in x if v is not None]
            cells.append(f"{st.mean(x):.2f} ± {st.stdev(x):.2f} [{', '.join(f'{v:.2f}' for v in x)}]" if len(x) > 1 else "—")
        out.append(f"| {a} | " + " | ".join(cells) + " |")
out.append("\n## Pre-declared criterion: quality degrades while the quantity gate has not fired\n")
cond_gap = {}
for c in CONDS:
    seeds_gap = []
    for s in SEEDS:
        r0, f0 = g(c, s, 0, "rotations_per_20s_per_env"), g(c, s, 0, "falls_per_20s_per_env")
        if r0 is None or f0 is None: seeds_gap.append(None); continue
        hit = []
        for a in ADRS[1:]:
            r, f = g(c, s, a, "rotations_per_20s_per_env"), g(c, s, a, "falls_per_20s_per_env")
            if r is None: continue
            if r >= 0.8 * r0 and f >= 2 * f0 and (f - f0) >= 0.5: hit.append(a)
        seeds_gap.append(hit)
    n_gap = sum(1 for h in seeds_gap if h)
    cond_gap[c] = n_gap >= 2
    out.append(f"- {c}: per-seed ADR levels meeting the gap rule (rot ≥ 80 % of ADR-0, falls ≥ 2× and +0.5): {seeds_gap} → seeds with a gap: {n_gap}/3 → condition gap: {cond_gap[c]}")
support = sum(cond_gap.values()) >= 2
out.append(f"\n**Conditions with a quality-before-quantity gap: {sum(cond_gap.values())}/3 → diagnostic {'SUPPORTS' if support else 'DOES NOT SUPPORT'} building E4.**")
open(os.path.join(R, "E4step0_analysis.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
