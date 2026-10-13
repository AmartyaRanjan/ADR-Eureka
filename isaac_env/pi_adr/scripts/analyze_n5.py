"""n=5 analysis: D vs E (adaptive DR, no RAPP) and vanilla default-entropy vs entropy=0.02.

Exact two-sided permutation test on the difference of means (C(10,5)=252 relabelings) -- no
distributional assumption, appropriate at n=5. Leads with tail metrics (worst-decile, mean error)
as the 11 Oct log recommended; 2cm/5cm success reported alongside.
"""
import itertools, json, os, statistics as st
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def load(n):
    p = os.path.join(R, n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None

METRICS = {
    "5cm %":        lambda d: d["success_rate"]["0.05"] * 100,
    "2cm %":        lambda d: d["success_rate"]["0.02"] * 100,
    "1cm %":        lambda d: d["success_rate"]["0.01"] * 100,
    "mean err mm":  lambda d: d["mean_terminal_error_m"] * 1000,
    "worst10% mm":  lambda d: d["worst_decile_error_m"] * 1000,
}

def perm_p(a, b):
    obs = abs(st.mean(a) - st.mean(b)); pool = a + b; n = len(a); cnt = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        ga = [pool[i] for i in idx]; gb = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1; cnt += abs(st.mean(ga) - st.mean(gb)) >= obs - 1e-12
    return cnt / tot

def compare(title, names_a, names_b, la, lb):
    A = [load(n) for n in names_a]; B = [load(n) for n in names_b]
    A = [d for d in A if d]; B = [d for d in B if d]
    print(f"\n### {title}   ({la}: n={len(A)}, {lb}: n={len(B)})")
    if len(A) < 3 or len(B) < 3:
        print("  insufficient runs yet"); return
    print(f"  {'metric':<13}{la:>18}{lb:>18}{'perm p':>9}")
    for m, f in METRICS.items():
        a = [f(d) for d in A]; b = [f(d) for d in B]
        print(f"  {m:<13}{st.mean(a):>11.2f}±{st.stdev(a):>5.2f}{st.mean(b):>11.2f}±{st.stdev(b):>5.2f}{perm_p(a, b):>9.3f}")

D  = lambda v: [f"D{s}_{v}" for s in ("", "_seed123", "_seed7", "_seed1", "_seed2")]
E  = lambda v: ["E_adaptive_no_rapp_" + v] + [f"E_seed{s}_no_rapp_{v}" for s in (123, 7, 1, 2)]
for v in ("nominal", "DRtrain", "DRshift"):
    compare(f"D vs E, {v}", D(v), E(v), "D", "E")

van_default = [f"vanilla_seed{s}_800iter_nominal" for s in (42, 123, 7)] + \
              [f"vanilla_ent0.005_seed{s}_800iter_nominal" for s in (1, 2)]
van_ent     = [f"vanilla_ent0.02_seed{s}_800iter_nominal" for s in (42, 123, 7, 1, 2)]
compare("vanilla: entropy 0.005 vs 0.02 (nominal, 800 iters)", van_default, van_ent, "ent=0.005", "ent=0.02")
