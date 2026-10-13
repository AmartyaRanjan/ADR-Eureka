# 09 — Adaptive DR: controller design, F1 → F4, rung E, and the multi-seed corrections

**Purpose.** Build the adaptive-DR outer loop (rung F, without the PINN) and test whether it beats static DR (rung D). This document is the story of how three apparent wins were each overturned by replication — which is itself the main methodological result — and which parts survived.

## 1. Scope decision: build F before G (10 Oct)

The roadmap's rung G feeds the PINN's uncertainty into the controller. The Session-1 observer's uncertainty had rank correlation ≈ +0.11 with its error, so a G built on it could not distinguish "adaptive DR does not help" from "the PINN signal is noise". Building **F first** isolates the adaptive-DR question from the PINN-calibration question. (G was built later once the observer improved — doc 08/10.)

## 2. Controller design (`source/pi_adr/dr/adaptive.py`)

Slow outer loop, one dimension at a time (RAPP's discipline, so a change in success is attributable to one cause):

```
S_d  = P(d_T < gate)  measured under the CURRENT range for dimension d (others nominal)
if S_d >= success_target:  widen the moving bound by step_frac of the remaining gap to its C_RAPP limit
else:                      hold; after `patience` stalled steps, narrow narrow_frac back toward the start
```

- Each dimension widens in one direction, taken from the measured RAPP envelope: friction/damping/mass upward, effectiveness downward.
- Starting ranges = the "in-distribution" box (friction [0,0.25], damping ×[0.5,4], effectiveness [0.5,1], mass ×[1,1.5]), so rung F starts from a point already measured.
- Orchestrated by `scripts/run_adaptive_dr.py`: 8 outer steps × 100 PPO iterations = 800 total, 2048 envs, evaluation of S_d between chunks (1024 envs). Total ≈ 30 min/run.
- Rung G adds a per-dimension `step_multiplier` multiplying `step_frac` (clipped to ≤ 1 of the gap) — doc 10.

## 3. F1 (coarse gate, step 0.25) vs D — single seed (10 Oct)

| Run | physics | <5 cm | <2 cm | <1 cm | mean err | settling |
|---|---|---|---|---|---|---|
| D | nominal | 97.78 % | 32.50 % | 8.76 % | 30.5 mm | 1.17 s |
| D | in-distribution | 99.73 % | 21.80 % | 6.23 % | 28.7 mm | 0.92 s |
| D | shifted | 96.39 % | 21.07 % | 4.96 % | 57.0 mm | 1.80 s |
| F1 | nominal | **72.88 %** | 20.14 % | 5.42 % | 38.3 mm | **7.18 s** |
| F1 | in-distribution | 94.19 % | 25.37 % | 8.15 % | 30.7 mm | 2.23 s |
| F1 | shifted | 96.44 % | 21.66 % | 5.22 % | 54.1 mm | 1.63 s |

Controller trajectory (confirms the mechanism): mass and effectiveness widened fastest and furthest (mass ×1 → 3.75 of 4; effectiveness → 0.32 of 0.3), friction repeatedly stalled (success dipped to 0.79, 0.75) — behaving exactly as the RAPP asymmetries suggest. Final friction [0, 2.42], damping [0.5, 211].
**Reading:** shifted — indistinguishable; in-distribution — mixed; **nominal — F1 clearly worse** (settling 7.18 vs 1.17 s). Hypothesis: widening 25 % of the remaining gap per step, gated only by 90 % coarse success, left F1 training on a far broader distribution than D's fixed box — *not* an equal-distribution comparison.

## 4. F2 (step 0.10): slower widening

| nominal | 5 cm | 2 cm | mean err | settling |
|---|---|---|---|---|
| D | 97.78 % | 32.50 % | 30.5 | 1.17 s |
| F1 | 72.88 % | 20.14 % | 38.3 | 7.18 s |
| F2 | **90.94 %** | 20.00 % | 32.4 | 4.04 s |

Budget-mismatch hypothesis **confirmed directionally** (nominal 5 cm: D > F2 > F1). But F2 became the **worst** of all three at 2 cm in every condition (20.00 / 15.43 / 14.55 %). **Mechanism (a new, controller-level instance of the project's central theme):** the widening gate was P(d<5 cm) — the same coarse threshold that hides precision collapse. A slower schedule spends *more* iterations inside physics bands that clear 5 cm but have already degraded precision. Next step: gate on a precision-aware signal.

## 5. F3 (mis-specified) and F4 (precision gate recalibrated)

F3 = gate 2 cm, success_target left at 0.90. After 3 outer steps S_d sat at 0.28–0.37 with ranges frozen: **no policy in the project has ever exceeded ≈ 42 % at 2 cm**, so the controller could never widen. Killed (no waste beyond 3 diagnostic steps). Relaunched as **F4**: gate 2 cm, **success_target 0.30** (matched to observed 2 cm rates), step 0.10, 800 iterations. F4 oscillated (S_d 0.26–0.37 around 0.30), ending conservative: mass only [1.0, 1.61] (vs F1's 3.75). *Lesson: a success threshold must be recalibrated to the precision epsilon it gates on.*

Seed 42 four-way comparison led to: "F4 beats F1 and F2 at 2 cm in every condition; competitive with D (in-distribution 2 cm 29.42 vs 21.80 %); weak point: nominal worst-decile 400.5 mm." It was labelled "the cleanest adaptive-DR result today… still single-seed".

## 6. The multi-seed test: **the F4 finding did not replicate** (seeds 42, 123, 7)

| (n=3) | 5 cm | 2 cm | 1 cm | mean err |
|---|---|---|---|---|
| D nominal | 97.44 ± 0.47 | 28.12 ± 5.24 | 8.05 ± 3.02 | 37.1 ± 7.2 |
| F4 nominal | 93.59 ± 6.38 | 25.81 ± 1.49 | 6.83 ± 1.47 | 47.4 ± 17.9 |
| D in-dist | 98.89 ± 0.73 | 24.10 ± 9.26 | 7.50 ± 4.46 | 37.8 ± 9.4 |
| F4 in-dist | 98.58 ± 1.08 | 23.79 ± 5.02 | 4.99 ± 1.65 | 40.0 ± 10.1 |
| D shifted | 96.28 ± 2.24 | 21.98 ± 8.27 | 6.61 ± 3.94 | 58.2 ± 20.2 |
| F4 shifted | 96.88 ± 1.61 | 20.82 ± 2.95 | 3.90 ± 1.46 | 53.6 ± 13.1 |

Raw 2 cm in-distribution (seeds 42,123,7): D 21.80, 16.21, 34.30; F4 29.42, 19.80, 22.14. The seed-42 pair that drove the "F4 beats D" claim lies well inside D's *own* cross-seed spread (16.2–34.3). Conclusion: **three controller designs (coarse-fast, coarse-slow, precision-gated) were tried and none beat static DR with multiple seeds.** What this does *not* rule out: a small real effect (underpowered), the *mechanism* of coarse-gate blindness (separate from net benefit), and untested rungs E/G.

## 6b. F4 at n=5 (final matrix, nominal / DRtrain / DRshift)

| metric | D | F4 |
|---|---|---|
| mean err nominal (mm) | 37.78 ± 8.08 | 44.74 ± 14.56 (p = 0.37) |
| worst-decile nominal | 147.4 ± 70.4 | 202.3 ± 145.1 (p = 0.52) |
| 2 cm DRtrain | 21.44 ± 7.94 | 17.79 ± 9.20 (p = 0.51) |
| mean err DRshift | 61.67 ± 20.96 | 51.74 ± 9.86 (p = 0.36) |

No F4−D contrast reaches p < 0.06 on mean or tail error. Per-seed nominal worst-decile for F4: 400.5, 265.0, 51.8, 225.5, 68.8 mm — one very weak seed (42) dominates the mean, a pattern also seen in D (233.1 vs 75.5).

## 7. Rung E (adaptive DR without RAPP): the largest apparent effect — and its correction

**Design.** Identical controller to F4 but C_RAPP replaced by arbitrary wide bounds (friction [0,50], damping ×[0.1,1000], effectiveness [0.01,2], mass ×[0.1,20]). Isolates whether RAPP's *constraint* adds value separately from the adaptive mechanism.

**Seed 42 (10 Oct):** mass widened to [1, 8.22] (> 2× past C_RAPP's 4.0), effectiveness [0.27, 1] (past 0.3). Results: nominal 5 cm **65.11 %** (collapse), in-distribution 2 cm **43.09 %** / 1 cm 13.31 % (nearly double D's 21.80 / 6.23 %), shifted 33.67 / 9.96 %. Written up then as "the largest effect this session" and a "clean trade-off".

**Multi-seed (n=3, 11 Oct) — what held and what did not:**
- **Nominal cost: robust.** E's nominal 5 cm was below D's *minimum* seed in 3/3 seeds (65.1, 88.2, 92.9 vs D 96.9–97.8).
- **2 cm gain under DR: not established.** E in-dist 2 cm: 43.1, 20.4, 51.2 — seed 42's "2× win" was the *second-best* of three; seed 123 equals D's level. Welch t ≈ 1.3.
- **Mean (tail) error under DR: the most consistent signal** (E lower than D in 3/3 seeds in both DR evals; 27.1 ± 2.4 vs 37.8 ± 9.4 in-distribution) — suggestive, t ≈ 1.9–2.1.
So E's benefit looked like *fewer bad episodes*, not better typical precision.

**n=5 (final matrix):**

| E vs D | nominal | DRtrain | DRshift |
|---|---|---|---|
| 5 cm % | 80.41 ± 12.70 vs 97.52 ± 0.77, **p = 0.008** | 98.11 vs 98.68 (0.42) | 97.13 vs 96.05 (0.64) |
| 2 cm % | 20.87 vs 26.83 (0.087) | 32.47 vs 21.44 (0.18) | 23.48 vs 19.81 (0.68) |
| mean err (mm) | 39.37 vs 37.78 (0.75) | **30.91 vs 41.08 (0.15)** | 51.62 vs 61.67 (0.65) |
| worst-decile (mm) | 109.3 vs 147.4 (0.31) | 94.3 vs 164.2 (0.27) | 273.2 vs 362.9 (0.67) |

Per-seed nominal 5 cm (42,123,7,1,2): E 65.1, 88.2, 92.9, 87.5, 68.4; D 97.8, 97.6, 96.9, 96.7, 98.6 — **complete separation** (the minimum attainable p, 0.0079). E's tail advantage under DR is directional (worst-decile about 43 % lower in DRtrain and 25 % lower in DRshift, neither significant at n=5).

**Verdict:** the *nominal cost of removing the ceiling* is real; the *DR benefit* is not established. The mechanistic story (no ceiling → mass toward 8× → harder distribution, less time near nominal) explains the nominal collapse; it has not been shown to explain a robustness gain. RAPP's role is thus best described as **limiting the nominal-performance cost**, which is a real but modest and unspectacular function.

## 8. How these claims were tested

| Claim | Test | Outcome |
|---|---|---|
| Budget mismatch explains F1's nominal weakness | F2 with slower widening | Confirmed directionally (90.9 vs 72.9 %) |
| Coarse gate blinds the controller | F2 worse at 2 cm; F4 with 2 cm gate | Mechanism plausible; net benefit not replicated |
| F4 > D | seeds 123, 7 (n=3), later n=5 | **Not supported** |
| E ≫ D under DR | seeds 123, 7, later n=5 | **Not supported** (only tail direction); nominal cost **confirmed** |
| Controller behaves as designed | Full per-step trajectory logs (`results/adaptive_dr/*_controller.json`) | Confirmed |
