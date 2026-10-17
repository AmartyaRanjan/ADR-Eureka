"""E1 analysis, exactly as pre-registered in project_docs/16_E1_preregistration.md.
Writes results/E1_analysis.md. Run any time; reports only what exists (flags incomplete seeds)."""
import itertools, json, os, glob, collections, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
SEEDS = (42, 123, 7, 1, 2); ENTS = ("0.005", "0.02"); ITS = (200, 400, 800, 1500, 3000, 5000)
M = {"mean err (mm)": lambda d: d["mean_terminal_error_m"] * 1000, "worst-10% (mm)": lambda d: d["worst_decile_error_m"] * 1000,
     "5cm %": lambda d: d["success_rate"]["0.05"] * 100, "2cm %": lambda d: d["success_rate"]["0.02"] * 100, "1cm %": lambda d: d["success_rate"]["0.01"] * 100}

def load(ent, s, it, v="nominal"):
    p = os.path.join(R, f"E1_ent{ent}_seed{s}_it{it}_{v}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def perm(a, b):
    pool = a + b; n = len(a); obs = abs(st.mean(a) - st.mean(b)); c = t = 0
    for idx in itertools.combinations(range(len(pool)), n):
        sidx = set(idx); ga = [pool[i] for i in idx]; gb = [pool[i] for i in range(len(pool)) if i not in sidx]
        t += 1; c += abs(st.mean(ga) - st.mean(gb)) >= obs - 1e-12
    return c / t

def vals(ent, it, f, v="nominal"):
    return [f(d) for d in (load(ent, s, it, v) for s in SEEDS) if d]

def entropy_final(ent, s):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        exp = open(os.path.join(R, "E1", f"E1_ent{ent}_seed{s}_experiment_dir.txt")).read().strip()
        f = glob.glob(os.path.join(ROOT, "logs/rl_games/two_dof_reach", exp, "summaries", "events*"))[0]
        ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
        sc = ea.Scalars("losses/entropy"); return st.mean(x.value for x in sc[-20:]), len(sc)
    except Exception:
        return None, 0

out = ["# E1 analysis (pre-registered; see project_docs/16_E1_preregistration.md)\n"]
status = {(e, s): (open(os.path.join(R, "E1", f"E1_ent{e}_seed{s}_STATUS.txt")).read().strip() if os.path.exists(os.path.join(R, "E1", f"E1_ent{e}_seed{s}_STATUS.txt")) else "not finished") for e in ENTS for s in SEEDS}
out.append("Run status: " + "; ".join(f"ent{e}/s{s}: {v}" for (e, s), v in status.items()) + "\n")
dirs = collections.defaultdict(list)
for e in ENTS:
    for s in SEEDS:
        p = os.path.join(R, "E1", f"E1_ent{e}_seed{s}_experiment_dir.txt")
        if os.path.exists(p): dirs[open(p).read().strip()].append(f"ent{e}/s{s}")
dup = {k: v for k, v in dirs.items() if len(v) > 1}
out.append(f"Run-directory collision audit: {'**COLLISIONS ' + str(dup) + '**' if dup else 'none'}\n")

out.append("## S3 dose-response (nominal; mean ± sd over seeds; n shown)\n")
for name, f in M.items():
    out.append(f"\n**{name}**\n\n| iteration | ent 0.005 | ent 0.02 |\n|---|---|---|")
    for it in ITS:
        cells = []
        for e in ENTS:
            x = vals(e, it, f); cells.append(f"{st.mean(x):.2f} ± {st.stdev(x):.2f} (n={len(x)})" if len(x) > 1 else (f"{x[0]:.2f} (n=1)" if x else "—"))
        out.append(f"| {it} | {cells[0]} | {cells[1]} |")

out.append("\n## Primary criterion (iteration 5000, nominal; exact permutation, 5 vs 5)\n")
ok = {}
for name in ("mean err (mm)", "worst-10% (mm)"):
    a, b = vals("0.005", 5000, M[name]), vals("0.02", 5000, M[name])
    if len(a) == 5 and len(b) == 5:
        p = perm(a, b); ok[name] = (st.mean(b) < st.mean(a)) and p < 0.05
        out.append(f"- {name}: ent0.005 {st.mean(a):.2f} ± {st.stdev(a):.2f} vs ent0.02 {st.mean(b):.2f} ± {st.stdev(b):.2f}; diff (0.02 − 0.005) {st.mean(b)-st.mean(a):+.2f}; p = {p:.4f}; per-seed 0.005: {[round(x,1) for x in a]}, 0.02: {[round(x,1) for x in b]}")
    else:
        out.append(f"- {name}: incomplete (ent0.005 n={len(a)}, ent0.02 n={len(b)})")
if len(ok) == 2: out.append(f"\n**PRIMARY CRITERION MET: {all(ok.values())}**")

out.append("\n## S1 (collapse with training): ent 0.005, 2 cm success at 5000 vs 800, per seed\n")
pairs = [(load("0.005", s, 800), load("0.005", s, 5000)) for s in SEEDS]; pairs = [(a, b) for a, b in pairs if a and b]
if pairs:
    d = [M["2cm %"](b) - M["2cm %"](a) for a, b in pairs]
    import math
    k = sum(x < 0 for x in d)
    ptxt = f"; exact one-sided sign-test p = {sum(math.comb(len(d), j) for j in range(k, len(d)+1)) / 2**len(d):.4f}" if len(d) == 5 else ""
    out.append(f"- per-seed change (5000 − 800): {[round(x,1) for x in d]}; declines in {k} of {len(d)} seeds" + ptxt)
d = [M["mean err (mm)"](b) - M["mean err (mm)"](a) for a, b in pairs]
if pairs: out.append(f"- per-seed change in mean error (5000 − 800), mm: {[round(x,1) for x in d]}")

out.append("\n## S2 (mechanism): final action entropy (mean of last 20 epochs of losses/entropy)\n")
for e in ENTS:
    xs = [entropy_final(e, s)[0] for s in SEEDS]; out.append(f"- ent {e}: {[None if x is None else round(x,2) for x in xs]}")

out.append("\n## S4 (exploratory): DRshift at 5000 and 800\n")
for it in (800, 5000):
    for name in ("mean err (mm)", "worst-10% (mm)"):
        a, b = vals("0.005", it, M[name], "DRshift"), vals("0.02", it, M[name], "DRshift")
        if len(a) == 5 and len(b) == 5: out.append(f"- it{it} {name}: 0.005 {st.mean(a):.1f} ± {st.stdev(a):.1f} vs 0.02 {st.mean(b):.1f} ± {st.stdev(b):.1f}; p = {perm(a,b):.4f}")
open(os.path.join(R, "E1_analysis.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
