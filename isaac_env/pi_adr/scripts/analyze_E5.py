"""E5 analysis exactly as pre-registered (project_docs/19_E5_sample_efficiency_preregistration.md)."""
import itertools, json, os, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "results")
SEEDS = (42, 123, 7, 1, 2); ITS = (100, 200, 300, 400, 600, 800)
def g(r, s, it):
    p = os.path.join(R, f"E5_{r}_seed{s}_it{it}_nominal.json"); return json.load(open(p)) if os.path.exists(p) else None
def perm(a, b):
    pool = a + b; n = len(a); obs = abs(st.mean(a) - st.mean(b)); c = t = 0
    for idx in itertools.combinations(range(len(pool)), n):
        sidx = set(idx); ga = [pool[i] for i in idx]; gb = [pool[i] for i in range(len(pool)) if i not in sidx]
        t += 1; c += abs(st.mean(ga) - st.mean(gb)) >= obs - 1e-12
    return c / t
M = {"mean err (mm)": lambda d: d["mean_terminal_error_m"] * 1000, "worst-10% (mm)": lambda d: d["worst_decile_error_m"] * 1000,
     "2cm %": lambda d: d["success_rate"]["0.02"] * 100, "5cm %": lambda d: d["success_rate"]["0.05"] * 100}
def area(r, s, f):
    y = []
    for it in ITS:
        d = g(r, s, it)
        if d is None: return None
        y.append(f(d))
    return sum((y[i] + y[i + 1]) / 2 * (ITS[i + 1] - ITS[i]) for i in range(len(ITS) - 1)) / (ITS[-1] - ITS[0])
out = ["# E5 analysis (pre-registered, project_docs/19_E5_sample_efficiency_preregistration.md)\n", "A = human (vanilla) reward, B = Eureka reward; seeds 42,123,7,1,2; nominal physics\n"]
for name, f in M.items():
    out.append(f"\n**{name} vs training iteration (mean ± sd, n = 5)**\n\n| iteration | A human | B Eureka |\n|---|---|---|")
    for it in ITS:
        cells = []
        for r in "AB":
            x = [f(d) for d in (g(r, s, it) for s in SEEDS) if d]
            cells.append(f"{st.mean(x):.2f} ± {st.stdev(x):.2f}" if len(x) > 1 else "—")
        out.append(f"| {it} | {cells[0]} | {cells[1]} |")
out.append("\n## Primary: learning-curve area of mean error (lower = better; hypothesis: B < A)\n")
a = [area("A", s, M["mean err (mm)"]) for s in SEEDS]; b = [area("B", s, M["mean err (mm)"]) for s in SEEDS]
if None not in a + b:
    p = perm(a, b)
    out.append(f"- A: {[round(x,1) for x in a]} mean {st.mean(a):.2f} ± {st.stdev(a):.2f}\n- B: {[round(x,1) for x in b]} mean {st.mean(b):.2f} ± {st.stdev(b):.2f}\n- diff (B − A) = {st.mean(b)-st.mean(a):+.2f} mm; exact permutation p = {p:.4f}; direction as hypothesised (B smaller): {st.mean(b) < st.mean(a)}")
    out.append(f"\n**PRIMARY CRITERION (B smaller AND p < 0.05): {st.mean(b) < st.mean(a) and p < 0.05}**")
out.append("\n## Secondary\n")
for it in (100, 200):
    x = [M["mean err (mm)"](g("A", s, it)) for s in SEEDS]; y = [M["mean err (mm)"](g("B", s, it)) for s in SEEDS]
    out.append(f"- S1 mean error at {it} iterations: A {st.mean(x):.1f} ± {st.stdev(x):.1f} vs B {st.mean(y):.1f} ± {st.stdev(y):.1f}; p = {perm(x, y):.4f}")
def first(r, s, cond):
    for it in ITS:
        d = g(r, s, it)
        if d and cond(d): return it
    return None
for lab, cond in (("mean error ≤ 45 mm", lambda d: d["mean_terminal_error_m"] * 1000 <= 45), ("2 cm success ≥ 20 %", lambda d: d["success_rate"]["0.02"] >= 0.20)):
    fa = [first("A", s, cond) for s in SEEDS]; fb = [first("B", s, cond) for s in SEEDS]
    out.append(f"- S2 first checkpoint with {lab}: A {fa} vs B {fb} (None = never reached by 800)")
for name in ("worst-10% (mm)", "2cm %"):
    a2 = [area("A", s, M[name]) for s in SEEDS]; b2 = [area("B", s, M[name]) for s in SEEDS]
    if None not in a2 + b2: out.append(f"- S3/S4 curve area of {name}: A {st.mean(a2):.2f} ± {st.stdev(a2):.2f} vs B {st.mean(b2):.2f} ± {st.stdev(b2):.2f}; p = {perm(a2, b2):.4f}")
open(os.path.join(R, "E5_analysis.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
