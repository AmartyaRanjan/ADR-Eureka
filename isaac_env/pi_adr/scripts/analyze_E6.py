"""E6 analysis exactly as pre-registered (project_docs/22_E6_preregistration.md).
Eureka arms = results/E6_ent*_seed*_it*_*.json ; human-reward arms = the existing E1 results. Use env python (tensorboard)."""
import itertools, json, os, glob, math, collections, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "results")
SEEDS = (42, 123, 7, 1, 2); ENTS = ("0.005", "0.02"); ITS = (200, 400, 800, 1500, 3000, 5000)
def load(prefix, e, s, it, v="nominal"):
    p = os.path.join(R, f"{prefix}_ent{e}_seed{s}_it{it}_{v}.json"); return json.load(open(p)) if os.path.exists(p) else None
M = {"mean err (mm)": lambda d: d["mean_terminal_error_m"] * 1000, "worst-10% (mm)": lambda d: d["worst_decile_error_m"] * 1000,
     "2cm %": lambda d: d["success_rate"]["0.02"] * 100, "1cm %": lambda d: d["success_rate"]["0.01"] * 100, "5cm %": lambda d: d["success_rate"]["0.05"] * 100}
def perm(a, b):
    pool = a + b; n = len(a); obs = abs(st.mean(a) - st.mean(b)); c = t = 0
    for idx in itertools.combinations(range(len(pool)), n):
        sidx = set(idx); ga = [pool[i] for i in idx]; gb = [pool[i] for i in range(len(pool)) if i not in sidx]
        t += 1; c += abs(st.mean(ga) - st.mean(gb)) >= obs - 1e-12
    return c / t
def vals(prefix, e, it, f, v="nominal"): return [f(d) for d in (load(prefix, e, s, it, v) for s in SEEDS) if d]
def ms(x): return f"{st.mean(x):.2f} ± {st.stdev(x):.2f} (n={len(x)})" if len(x) > 1 else (f"{x[0]:.2f} (n=1)" if x else "—")
out = ["# E6 analysis (pre-registered, project_docs/22_E6_preregistration.md)\n"]
status = {(e, s): (open(os.path.join(R, "E6", f"E6_ent{e}_seed{s}_STATUS.txt")).read().strip()[:40] if os.path.exists(os.path.join(R, "E6", f"E6_ent{e}_seed{s}_STATUS.txt")) else "not finished") for e in ENTS for s in SEEDS}
out.append("Run status: " + "; ".join(f"E6 ent{e}/s{s}: {v}" for (e, s), v in status.items()) + "\n")
dirs = collections.defaultdict(list)
for pre in ("E1", "E6"):
    for f in glob.glob(os.path.join(R, pre, "*_experiment_dir.txt")): dirs[open(f).read().strip()].append(pre + ":" + os.path.basename(f))
dup = {k: v for k, v in dirs.items() if len(v) > 1}
out.append(f"Run-directory collision audit (E1+E6 final directories): {'**COLLISIONS ' + str(dup) + '**' if dup else 'none'}\n")
out.append("## Dose-response, nominal (mean ± sd): human (E1) vs Eureka (E6), by entropy coefficient\n")
for name in ("mean err (mm)", "worst-10% (mm)", "2cm %"):
    out.append(f"\n**{name}**\n\n| iteration | human 0.005 | human 0.02 | Eureka 0.005 | Eureka 0.02 |\n|---|---|---|---|---|")
    for it in ITS:
        out.append(f"| {it} | " + " | ".join(ms(vals(pre, e, it, M[name])) for pre in ("E1", "E6") for e in ENTS) + " |")
out.append("\n## Primary (P): iteration 5000, nominal, Eureka-0.02 vs human-0.02 (exact permutation, 5 vs 5)\n")
P = {}
for name in ("mean err (mm)", "worst-10% (mm)"):
    a, b = vals("E1", "0.02", 5000, M[name]), vals("E6", "0.02", 5000, M[name])
    if len(a) == 5 and len(b) == 5:
        p = perm(a, b); P[name] = p
        out.append(f"- {name}: human-0.02 {st.mean(a):.2f} ± {st.stdev(a):.2f} vs Eureka-0.02 {st.mean(b):.2f} ± {st.stdev(b):.2f}; diff (Eureka − human) {st.mean(b)-st.mean(a):+.2f}; p = {p:.4f}; per-seed human {[round(x,1) for x in a]}, Eureka {[round(x,1) for x in b]}")
    else: out.append(f"- {name}: incomplete (human n={len(a)}, Eureka n={len(b)})")
out.append("\n## S2 (Stage-1-style, default entropy 0.005 for both rewards, iteration 5000)\n")
S2 = {}
for name in ("mean err (mm)", "worst-10% (mm)"):
    a, b = vals("E1", "0.005", 5000, M[name]), vals("E6", "0.005", 5000, M[name])
    if len(a) == 5 and len(b) == 5:
        p = perm(a, b); S2[name] = p
        out.append(f"- {name}: human-0.005 {st.mean(a):.2f} ± {st.stdev(a):.2f} vs Eureka-0.005 {st.mean(b):.2f} ± {st.stdev(b):.2f}; diff (Eureka − human) {st.mean(b)-st.mean(a):+.2f}; p = {p:.4f}; per-seed human {[round(x,1) for x in a]}, Eureka {[round(x,1) for x in b]}")
    else: out.append(f"- {name}: incomplete (human n={len(a)}, Eureka n={len(b)})")
out.append("\n## S1: does Eureka also collapse? (entropy 0.005, 5000 vs 800)\n")
pairs = [(load("E6", "0.005", s, 800), load("E6", "0.005", s, 5000)) for s in SEEDS]; pairs = [(a, b) for a, b in pairs if a and b]
if pairs:
    d = [M["2cm %"](b) - M["2cm %"](a) for a, b in pairs]; k = sum(x < 0 for x in d)
    ptxt = f"; exact one-sided sign-test p = {sum(math.comb(len(d), j) for j in range(k, len(d)+1)) / 2**len(d):.4f}" if len(d) == 5 else ""
    out.append(f"- 2 cm change (5000 − 800) per seed: {[round(x,1) for x in d]}; declines in {k} of {len(d)}" + ptxt)
    dm = [M["mean err (mm)"](b) - M["mean err (mm)"](a) for a, b in pairs]; out.append(f"- mean error change (mm) per seed: {[round(x,1) for x in dm]}")
out.append("\n## S3: final action entropy (mean of last 20 epochs of losses/entropy), Eureka arms\n")
def ent_final(e, s):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        exp = open(os.path.join(R, "E6", f"E6_ent{e}_seed{s}_experiment_dir.txt")).read().strip()
        f = glob.glob(os.path.join(ROOT, "logs/rl_games/two_dof_reach", exp, "summaries", "events*"))[0]
        ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload(); sc = ea.Scalars("losses/entropy"); return st.mean(x.value for x in sc[-20:])
    except Exception: return None
for e in ENTS: out.append(f"- Eureka ent {e}: {[None if v is None else round(v,2) for v in (ent_final(e, s) for s in SEEDS)]}")
out.append("\n## S4 (exploratory): precision and DRshift at 5000, all four arms\n")
for name in ("2cm %", "1cm %"):
    out.append(f"- {name} at 5000: " + "; ".join(f"{pre}-{e}: {ms(vals(pre, e, 5000, M[name]))}" for pre in ("E1", "E6") for e in ENTS))
for name in ("mean err (mm)", "worst-10% (mm)"):
    out.append(f"- DRshift {name} at 5000: " + "; ".join(f"{pre}-{e}: {ms(vals(pre, e, 5000, M[name], 'DRshift'))}" for pre in ("E1", "E6") for e in ENTS))
if len(P) == 2 and len(S2) == 2:
    out.append(f"\n**Pre-declared reading:** P significant (p<0.05) on {[n for n,p in P.items() if p<0.05] or 'neither metric'}; S2 significant on {[n for n,p in S2.items() if p<0.05] or 'neither metric'}. Apply the interpretation table in doc 22 using the direction (sign of 'diff') printed above.")
open(os.path.join(R, "E6_analysis.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
