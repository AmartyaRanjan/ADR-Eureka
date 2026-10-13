# 10 — The A–G ablation matrix (5 seeds), rung G, the placebo, and the contaminated retest

**Purpose (12–13 Oct 2026).** Fill the roadmap's full matrix under one protocol so every rung is compared on equal footing, then decide, with statistics, which of the roadmap's claims survive. Everything here: Eureka/vanilla reward as specified per rung, 800 PPO iterations (adaptive rungs: 8 × 100), 2048 envs, seeds **42, 123, 7, 1, 2**, frozen metrics v1.0.0, 4096 eval envs, eval seed 1234. Source tables: `results/MATRIX_A_to_G.md` (regenerate: `python scripts/analyze_matrix.py`).

## 1. Final matrix (mean ± std over 5 seeds)

### Nominal physics
| rung | mean err (mm) | worst-10 % (mm) | 5 cm % | 2 cm % | 1 cm % |
|---|---|---|---|---|---|
| A human, no DR | 37.75 ± 3.95 | 149.38 ± 31.63 | 98.48 ± 0.44 | 28.99 ± 4.20 | 8.39 ± 3.05 |
| B Eureka, no DR | 37.69 ± 9.39 | 138.71 ± 74.10 | 98.98 ± 0.76 | 26.87 ± 6.26 | 7.92 ± 2.07 |
| C human + static DR | 36.44 ± 7.16 | 136.14 ± 69.28 | 96.30 ± 0.98 | 29.28 ± 8.94 | 8.23 ± 3.02 |
| D Eureka + RAPP static DR | 37.78 ± 8.08 | 147.42 ± 70.44 | 97.52 ± 0.77 | 26.83 ± 4.50 | 7.00 ± 2.57 |
| E adaptive, no RAPP | 39.37 ± 5.54 | 109.29 ± 39.02 | 80.41 ± 12.70 | 20.87 ± 5.02 | 5.15 ± 1.45 |
| F4 adaptive + RAPP | 44.74 ± 14.56 | 202.33 ± 145.10 | 94.07 ± 4.57 | 23.13 ± 4.75 | 6.41 ± 1.56 |
| G F4 + PINN-scaled steps | 34.12 ± 3.91 | 81.84 ± 32.86 | 94.08 ± 3.67 | 19.14 ± 3.88 | 4.16 ± 1.45 |
| Gp placebo (shuffled) | 33.74 ± 3.99 | 87.26 ± 45.05 | 94.43 ± 3.94 | 22.18 ± 5.92 | 5.49 ± 2.27 |

### DRtrain (in-distribution randomization)
| rung | mean err | worst-10 % | 5 cm % | 2 cm % | 1 cm % |
|---|---|---|---|---|---|
| A | 41.70 ± 12.02 | 213.21 ± 128.69 | 97.49 ± 1.91 | 34.19 ± 7.13 | 9.52 ± 3.72 |
| B | 39.84 ± 12.90 | 201.22 ± 126.41 | 97.81 ± 1.52 | 40.77 ± 11.27 | 11.79 ± 3.95 |
| C | 37.28 ± 5.31 | 145.20 ± 35.35 | 98.34 ± 0.64 | 29.02 ± 8.85 | 8.29 ± 1.89 |
| D | 41.08 ± 12.74 | 164.22 ± 112.53 | 98.68 ± 0.96 | 21.44 ± 7.94 | 5.90 ± 3.84 |
| E | 30.91 ± 6.01 | 94.34 ± 55.26 | 98.11 ± 1.08 | 32.47 ± 13.78 | 8.97 ± 4.98 |
| F4 | 40.66 ± 8.20 | 139.95 ± 84.12 | 98.69 ± 0.97 | 17.79 ± 9.20 | 3.72 ± 2.16 |
| G | 33.59 ± 1.76 | 68.82 ± 19.49 | 99.50 ± 0.37 | 16.04 ± 2.72 | 3.36 ± 1.15 |
| Gp | 33.16 ± 3.33 | 75.73 ± 34.60 | 99.51 ± 0.33 | 20.84 ± 14.62 | 5.59 ± 3.89 |

### DRshift (unseen, strictly harder)
| rung | mean err | worst-10 % | 5 cm % | 2 cm % | 1 cm % |
|---|---|---|---|---|---|
| A | 87.13 ± 15.17 | 666.00 ± 153.48 | 91.98 ± 3.06 | 37.57 ± 10.30 | 10.57 ± 4.76 |
| B | 97.28 ± 92.48 | 581.14 ± 502.57 | 91.45 ± 9.59 | 43.24 ± 11.81 | 14.30 ± 5.10 |
| C | 65.58 ± 26.42 | 415.93 ± 244.25 | 94.61 ± 3.40 | 26.32 ± 8.54 | 7.36 ± 1.60 |
| D | 61.67 ± 20.96 | 362.91 ± 180.93 | 96.05 ± 1.83 | 19.81 ± 7.96 | 5.59 ± 3.30 |
| E | 51.62 ± 37.51 | 273.20 ± 370.97 | 97.13 ± 4.00 | 23.48 ± 17.03 | 5.87 ± 5.09 |
| F4 | 51.74 ± 9.86 | 236.79 ± 108.58 | 97.43 ± 1.44 | 14.38 ± 9.16 | 2.65 ± 2.05 |
| G | 45.35 ± 5.96 | 173.81 ± 55.32 | 97.92 ± 0.96 | 12.27 ± 4.07 | 2.12 ± 1.22 |
| Gp | 68.31 ± 28.55 | 385.19 ± 200.87 | 93.90 ± 4.95 | 15.81 ± 15.08 | 3.36 ± 2.68 |

(Gp numbers are the 5 clean seeds 42, 123, 7, 1, 2.)

## 2. Pre-declared contrasts and what they tell us (exact permutation p; raw, uncorrected)

| contrast | nominal mean err / worst-10 % / 5 cm | DRtrain | DRshift | Reading |
|---|---|---|---|---|
| B − A (reward effect) | −0.06 (0.98) / −10.7 (0.78) / +0.50 (0.24) | 2 cm +6.6 (0.29) | 2 cm +5.7 (0.43) | **No reward effect** at matched budget |
| C − A (static DR, human) | −1.3 (0.74) / −13.2 (0.70) / **−2.18 (0.008)** | −4.4 (0.66) mean err | −21.6 (0.18) mean err | Static DR costs a little coarse success at nominal; otherwise no effect |
| D − B (static DR, Eureka) | +0.09 (0.97) / +8.7 (0.83) / **−1.46 (0.016)** | **2 cm −19.3 (0.016)**, 1 cm −5.9 (0.056) | **2 cm −23.4 (0.016)**, 1 cm −8.7 (0.016) | Static DR **lowers 2 cm/1 cm success** vs no-DR when evaluated under physics randomization |
| E − D | +1.6 (0.75) / −38.1 (0.31) / **−17.1 (0.008)** | mean err −10.2 (0.15) | −10.1 (0.65) | Nominal coarse-success cost is the only solid effect |
| F4 − D | +7.0 (0.37) / +54.9 (0.52) / −3.5 (0.063) | −0.4 (0.92) | −9.9 (0.36) | No adaptive-DR benefit |
| F4 − E | +5.4 (0.44) / +93.1 (0.21) / +13.7 (0.040) | +9.8 (0.071) | +0.1 (1.0) | RAPP's ceiling recovers nominal coarse success |
| G − F4 | −10.6 (0.16) / −120.5 (0.12) / +0.01 (1.0) | −7.1 (0.13) | −6.4 (0.29) | Not significant |
| G − Gp (placebo) | +0.4 (0.85) / −5.4 (0.82) / −0.35 (0.89) | +0.4 (0.79) / −6.9 (0.74) | −23.0 (0.071) / worst-10 % **−211 (0.024)** / 5 cm +4.0 (0.048) | See §4: signal did not hold up |
| G − D | −3.7 (0.35) / −65.6 (0.12) / −3.4 (0.087); 2 cm **−7.7 (0.032)** | −7.5 (0.30) | −16.3 (0.16); worst-10 % −189 (0.063); 1 cm −3.5 (0.032) | Post-hoc; G has the lowest tail but also the lowest 2 cm/1 cm |

**Multiple-comparisons caution** (doc 03): 150 tests, 12 with raw p < 0.05 (7.5 expected by chance); with n = 5 the smallest attainable p is 0.0079, so no result survives a Bonferroni correction. The robust (complete-separation, mechanistically explained) results are: **E's nominal 5 cm cost** and **no reward effect (A vs B)**. Everything else is suggestive or null.

### Per-seed raw values behind the key means (seed order 42, 123, 7, 1, 2)

**Nominal 5 cm:** A 98.6, 98.6, 98.9, 98.6, 97.7 · B 98.6, 99.1, 99.3, 97.9, 99.9 · D 97.8, 97.6, 96.9, 96.7, 98.6 · E 65.1, 88.2, 92.9, 87.5, 68.4 · F4 95.4, 86.5, 98.9, 95.3, 94.3 · G 99.7, 93.7, 90.3, 91.5, 95.2.
**Nominal 2 cm:** A 27.9, 29.4, 30.5, 22.8, 34.4 · B 17.3, 28.0, 31.4, 24.6, 33.0 · D 32.5, 22.3, 29.5, 22.3, 27.5 · E 12.8, 20.6, 26.1, 23.8, 21.1 · F4 25.6, 24.4, 27.4, 15.1, 23.1.
**Nominal worst-decile (mm):** A 186.9, 112.9, 124.6, 174.8, 147.7 · B 188.5, 134.3, 87.3, 233.4, 50.1 · D 89.9, 208.4, 130.2, 233.1, 75.5 · E 89.1, 59.2, 108.4, 162.8, 126.9 · F4 400.5, 265.0, 51.8, 225.5, 68.8 · G 45.3, 128.0, 102.0, 67.1, 66.8.
**DRtrain 2 cm:** B 25.6, 48.3, 41.1, 34.6, 54.3 · D 21.8, 16.2, 34.3, 13.7, 21.1 (B exceeds D in 5 of 5 seeds paired by seed; B's minimum 25.6 exceeds D's median 21.1 (not complete separation: D's maximum is 34.3)).
**DRshift 2 cm:** B 29.5, 58.1, 49.6, 33.1, 45.8 · D 21.1, 14.2, 30.7, 10.2, 22.9.
(Full per-seed grid for all metrics is in the generated tables; the 10-seed layout is in `project_docs/appendix_per_seed.txt`.)

### Within-policy paired effect: harder physics raises 2 cm success for no-DR policies
A: 27.9, 29.4, 30.5, 22.8, 34.4 → DRshift 51.9, 34.0, 31.0, 26.7, 44.3 (5/5 up). B: 17.3, 28.0, 31.4, 24.6, 33.0 → 29.5, 58.1, 49.6, 33.1, 45.8 (5/5 up). Exact sign test one-sided p = 0.031 each (doc 05).

## 3. What the matrix says

1. **Reward choice does not matter at matched budget** (A ≈ B on every metric in every evaluation).
2. **DR during training did not buy robustness under evaluation-time DR** for this reward/budget: C (vs A) and D (vs B) do not improve mean or tail error; D actually lowers 2 cm/1 cm success relative to B. Plausible mechanism: DR-trained policies are conservative; the no-DR policies are *accidentally* forced slower by harsh physics at eval (doc 05) and gain fine precision. (Mechanism untested directly.)
3. **Adaptive DR (F4, E) does not beat static DR** on tail/mean error. E pays a robust nominal coarse-success cost; its tail benefit is directional only.
4. **RAPP's ceiling does what it was designed for** — it limits that nominal cost (F4 − E 5 cm +13.7 points, p = 0.04) — but is not shown to improve robustness.
5. **Rung G has the best-looking tail and the least seed variance but the lowest fine precision; the placebo matches it** (next section).

## 4. Rung G and the placebo

**Design (`scripts/pinn_probe.py`, `AdaptiveDRController.update(step_multiplier=)`, `run_adaptive_dr.py --pinn_observer seq --pinn_mode {pinn,shuffle} --pinn_strength 1.0`):** identical to F4 (gate 2 cm, success_target 0.30, step 0.10) except each dimension's widening step is multiplied by

m_d = clip(1 + s·(u_d / mean u − 1), 0.5, 1.5),

where u_d is the frozen observer's epistemic variance for that dimension's parameters, probed with only that dimension randomised over its current range. Higher uncertainty ⇒ larger step ("explore where the model knows least"). **Placebo:** identical values, randomly permuted across dimensions each outer step — so any benefit from G that survives shuffling is not from physics information.

**What happened in practice.** The observer's epistemic variance for damping scale often exceeded the other dimensions by up to ≈ 10×, so the clipped multipliers frequently sat at (damping 1.5; the rest 0.5). G therefore behaved like a **fixed re-weighting** of step sizes — a weak test of a *feedback* signal. Example, G seed 42 final ranges: friction [0, 0.57], damping [0.5, 21.4], effectiveness [0.47, 1], mass [1, 1.63]; placebo seed 42: friction 0.62, damping 33.1, effectiveness 0.47, mass 1.96.

**Results.** G − Gp (5 clean seeds): nominal and DRtrain indistinguishable (mean err +0.4 mm p = 0.85 / 0.79; worst-10 % −5 and −7 mm, p ≥ 0.74). DRshift: mean −23.0 mm (p = 0.071), worst-decile −211 mm (**p = 0.024**), 5 cm +4.0 points (p = 0.048): the only hint of a PINN-specific effect — one of ≈ 15 DRshift contrasts, with the placebo's variance large (68.3 ± 28.6 mm vs G's 45.4 ± 6.0).

## 5. The pre-declared retest, its contamination, and the repair

**Pre-registration (12 Oct, before new data):** primary metric = DRshift worst-decile error (G vs Gp); secondary = DRshift mean error, 5 cm. Add seeds 3, 4, 5, 6, 8 for both arms.

**First result (10 seeds/arm):** worst-decile G 256.8 ± 197.9 vs Gp 381.9 ± 289.2 mm, diff −125, p = 0.281; mean err p = 0.280; 5 cm p = 0.181 — **the 5-seed signal did not replicate.** (That text is now superseded.)

**The red flag.** While writing this documentation, per-seed values showed **G seed 4 and Gp seed 4 had identical numbers** to the decimal across all three evaluations. Investigation:

- chain13 (G) and chain14 (Gp) ran **concurrently**. Run directories are timestamp-named to the second. Whenever both arms started a chunk in the same second, they wrote to the same directory; `find_checkpoint()` globs that directory and could return the other arm's checkpoint, which the next chunk then resumed from.
- Audit of all `*_final_checkpoint.txt` and every adaptive chunk's experiment name: **8 shared-directory chunks, all in seeds 3, 4, 5, 6** (seed 3 chunks 0–2; seed 4 chunks 5 and 7; seed 5 chunk 0; seed 6 chunks 0–1). Seed 4's final checkpoint path is the same file for both arms (`…/2026-10-12_17-15-03/nn/last_two_dof_reach_ep_800_rew__27182.79_.pth`). Seeds 42/123/7/1/2 and 8 have no collisions, and no static run collided.

**Repair.** Contaminated seeds 3–6 moved to `results/_quarantine_contaminated_retest_20261012/`. Recomputed on the **6 clean seeds** (42, 123, 7, 1, 2, 8), `results/G_vs_placebo_retest_clean_n6.txt`:

| DRshift (n = 6 vs 6) | G | Gp | diff | p |
|---|---|---|---|---|
| worst-decile (mm) — PRIMARY | 250.7 ± 194.7 | 487.1 ± 307.5 | −236.4 | **0.147** |
| mean err (mm) | 53.0 ± 19.4 | 88.3 ± 55.1 | −35.3 | 0.169 |
| 5 cm % | 96.97 ± 2.46 | 91.16 ± 8.03 | +5.8 | 0.104 |
| 2 cm % | 13.39 ± 4.56 | 14.45 ± 13.89 | −1.1 | 0.885 |

Nominal (n = 6): worst-decile 88.2 vs 80.9 (p = 0.76), mean err 34.9 vs 33.4 (p = 0.50). A clean rerun of seeds 3–6 with the two arms **run sequentially** is queued behind the LEAP jobs (`logs/chain15_retest_rerun.log`, writes the same file names; the contaminated copies are already in quarantine). Whatever it shows will be appended to `EXPERIMENT_LOG.md`.

**Interpretation.** The direction favours G on DRshift tail metrics throughout (−125 to −236 mm worst-decile), but the effect is not significant, shrinks as clean seeds are added, depends on the placebo's high variance (a few terrible placebo seeds, e.g. seed 8: worst-decile 996.5 mm; seed 1: 715.3), and is absent at nominal/DRtrain. **Verdict: no demonstrated benefit of PINN information; rung G is closed as inconclusive-negative on this task**, pending the rerun, which can change power but not the ceiling imposed by the weak observer signal.

## 6. Why the placebo design was worth it

Without Gp, G's lower tail error and lower seed variance (DRtrain mean err 33.6 ± 1.8 mm vs F4's 40.7 ± 8.2) would have looked like support for the PINN. The placebo shows the *same* values appear when the information is scrambled (nominal 33.7 vs 34.1 mm; DRtrain 33.2 vs 33.6 mm). Whatever G does, it is **not** attributable to the observer's information.

## 7. What was verified and what was not

| Item | Status |
|---|---|
| Matrix seeds 42/123/7/1/2 are independent per rung | Verified (no shared directories; per-seed distinct values) |
| Metrics identical protocol across rungs | Verified (same scripts, same eval seed) |
| G/Gp seeds 3–6 | **Contaminated**, quarantined, rerun queued |
| G's multipliers carry physics information | Weak: dominated by one dimension; placebo equals G |
| Observer reliability outside its training range | Poor (rank corr −0.16) |
