"""Pre-declared G vs shuffled-placebo retest (12 Oct): 10 seeds/arm, exact permutation test C(20,10)."""
import itertools, json, os, statistics as st
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS = (42, 123, 7, 1, 2, 3, 4, 5, 6, 8)
M = {"worst10% mm (PRIMARY)": lambda d: d["worst_decile_error_m"] * 1000, "mean err mm": lambda d: d["mean_terminal_error_m"] * 1000,
     "5cm %": lambda d: d["success_rate"]["0.05"] * 100, "2cm %": lambda d: d["success_rate"]["0.02"] * 100}
def get(prefix, v, f):
    out = []
    for s in SEEDS:
        p = os.path.join(R, f"{prefix}_seed{s}_{v}.json")
        if os.path.exists(p): out.append(f(json.load(open(p))))
    return out
def perm(a, b):
    pool = a + b; n = len(a); obs = abs(st.mean(a) - st.mean(b)); c = t = 0
    for idx in itertools.combinations(range(len(pool)), n):
        s = set(idx); ga = [pool[i] for i in idx]; gb = [pool[i] for i in range(len(pool)) if i not in s]
        t += 1; c += abs(st.mean(ga) - st.mean(gb)) >= obs - 1e-12
    return c / t
for v in ("DRshift", "nominal", "DRtrain"):
    print(f"\n## {v}")
    for m, f in M.items():
        a, b = get("G", v, f), get("Gplacebo", v, f)
        if len(a) < 3 or len(b) < 3: print(m, "insufficient"); continue
        print(f"{m:<24} G n={len(a)} {st.mean(a):8.2f}±{st.stdev(a):7.2f} | placebo n={len(b)} {st.mean(b):8.2f}±{st.stdev(b):7.2f} | diff {st.mean(a)-st.mean(b):+8.2f} p={perm(a,b) if len(a)+len(b)<=20 else float('nan'):.3f}")
