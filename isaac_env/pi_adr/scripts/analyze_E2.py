"""E2 analysis as pre-registered (project_docs/17_E2_preregistration.md): rotations-vs-ADR curve, AUC, ADR 10/15."""
import itertools, json, os, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "results")
CONDS = ("L0", "L1", "L2"); SEEDS = (42, 123, 7); PTS = (0, 5, 10, 15, 20, 25)
def f(c, s, a):
    n = f"LEAP_{c}_s{s}_nominal" if a == 0 else (f"LEAP_{c}_s{s}_fullDR" if a == 25 else f"LEAP_{c}_s{s}_adr{a}")
    p = os.path.join(R, n + ".json"); return json.load(open(p))["rotations_per_20s_per_env"] if os.path.exists(p) else None
def auc(c, s):
    y = [f(c, s, a) for a in PTS]
    if any(v is None for v in y): return None
    return sum((y[i] + y[i + 1]) / 2 * (PTS[i + 1] - PTS[i]) for i in range(len(PTS) - 1)) / 25
def perm(a, b):
    pool = a + b; n = len(a); obs = abs(st.mean(a) - st.mean(b)); c = t = 0
    for idx in itertools.combinations(range(len(pool)), n):
        sidx = set(idx); ga = [pool[i] for i in idx]; gb = [pool[i] for i in range(len(pool)) if i not in sidx]
        t += 1; c += abs(st.mean(ga) - st.mean(gb)) >= obs - 1e-12
    return c / t
out = ["# E2 analysis (pre-registered in project_docs/17_E2_preregistration.md)\n", "## Rotations / 20 s vs frozen ADR increments (mean ± sd over 3 seeds; per-seed in brackets)\n",
       "| ADR | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
for a in PTS:
    cells = []
    for c in CONDS:
        x = [f(c, s, a) for s in SEEDS]; x = [v for v in x if v is not None]
        cells.append(f"{st.mean(x):.2f} ± {st.stdev(x):.2f} [{', '.join(f'{v:.1f}' for v in x)}]" if len(x) > 1 else "—")
    out.append(f"| {a} | " + " | ".join(cells) + " |")
A = {c: [auc(c, s) for s in SEEDS] for c in CONDS}
out.append("\n## Primary: normalised AUC (0–25)\n")
for c in CONDS: out.append(f"- {c}: {[None if v is None else round(v, 3) for v in A[c]]}" + (f"  mean {st.mean(A[c]):.3f} ± {st.stdev(A[c]):.3f}" if all(v is not None for v in A[c]) else "  (incomplete)"))
if all(all(v is not None for v in A[c]) for c in CONDS):
    m = {c: st.mean(A[c]) for c in CONDS}
    out.append(f"\nOrdering predicted AUC(L2) ≥ AUC(L1) ≥ AUC(L0): L2 {m['L2']:.3f}, L1 {m['L1']:.3f}, L0 {m['L0']:.3f} → hypothesis {'consistent' if m['L2'] >= m['L1'] >= m['L0'] else 'NOT consistent'}")
    for a, b in (("L1", "L0"), ("L2", "L1"), ("L2", "L0")):
        out.append(f"- AUC {a} − {b}: {m[a]-m[b]:+.3f}, exact perm p = {perm(A[a], A[b]):.3f} (min attainable 0.10)")
    hits = sum(m[a] > m[b] for a, b in (("L1", "L0"), ("L2", "L1"), ("L2", "L0")))
    out.append(f"\nPairings in the hypothesised direction: {hits}/3")
out.append("\n## Secondary: ADR 10 and 15, per-seed consistency\n")
for a in (10, 15):
    for x, y in (("L1", "L0"), ("L2", "L1"), ("L2", "L0")):
        vx = [f(x, s, a) for s in SEEDS]; vy = [f(y, s, a) for s in SEEDS]
        if None in vx or None in vy: continue
        sep = min(vx) > max(vy); out.append(f"- ADR {a}: {x} {[round(v,2) for v in vx]} vs {y} {[round(v,2) for v in vy]}; complete separation ({x} above {y}): {sep}; perm p = {perm(vx, vy):.3f}")
open(os.path.join(R, "E2_analysis.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
