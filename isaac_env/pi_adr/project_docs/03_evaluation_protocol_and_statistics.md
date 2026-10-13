# 03 — Evaluation protocol and statistics

This document defines how every number in the project is produced, and — just as important — the toolkit used to decide whether a number reflects a real effect or an anomaly.

## 1. The frozen evaluation protocol (METRICS_VERSION 1.0.0)

Implemented once in `source/pi_adr/evaluation/metrics.py` and `runner.py`; used unchanged for every 2-DOF run. Changing any definition requires a version bump and re-running everything.

| Item | Definition | Why |
|---|---|---|
| Environments | 4096 per evaluation, deterministic actions (policy mean) | Remove sampling noise from the metric |
| Evaluation seed | 1234, fixed | Same goals/physics draws for every policy, so differences come from the policy |
| Terminal error | Distance from end effector to goal at t = 9.93 s (rollout horizon − 2) | See "post-reset bug" in doc 04 |
| Precision curve | P(d < 5 cm, 2 cm, 1 cm, 5 mm, 2 mm, 1 mm) | A single threshold hides the coarse-vs-fine mismatch that is the project's central phenomenon |
| Mean / median / p90 / worst-decile error | worst-decile = mean of the worst 10 % of episodes | Tail behaviour (rare big failures) matters for robustness and was repeatedly the most informative metric |
| Motion/effort | mean |q̇|, mean control effort, settling time, overshoot | Distinguishes "precise" from "slow" |
| Three physics distributions | *nominal*, *DRtrain*, *DRshift* (ranges in doc 00 glossary) | Nominal measures fit to the easy case; DRtrain in-distribution robustness; DRshift generalisation to an unseen harder distribution |

LEAP uses an analogous frozen protocol: rotations per 20 s per hand, 1024 envs, 600 control steps, eval seed 1234, **ADR frozen** at a fixed difficulty (0 or 25 increments) so every policy faces identical conditions (doc 11).

## 2. Why the protocol looks the way it does (design reasoning)

1. **Fixed eval seed + fixed environments.** Between-policy differences must not come from different test goals.
2. **Precision curve, not a single number.** The Stage-1 claim is about the *shape* of the curve (high at 5 cm, low at 1 cm). A scalar success rate would hide it.
3. **Tail metric.** Early in the project, mean error differences between rungs were dominated by a few catastrophic episodes. Worst-decile error separates "typically worse" from "occasionally terrible", and was later pre-declared as the primary metric for the G-vs-placebo test.
4. **Three physics distributions.** A change that helps under DR but hurts at nominal (rung E) would be invisible with one distribution.
5. **Terminal sample inside the episode.** Because the environment auto-resets on timeout (doc 04).

## 3. Statistics: why exact permutation tests

- Sample sizes are small (5 seeds per rung; 6–10 for the G-vs-placebo question). With n = 5 per group, normality assumptions (t-test) are not credible, and PPO seed-to-seed variance is large (std of 5–17 points on 2 cm success).
- **Exact two-sided permutation test on the difference of means** (`scripts/analyze_matrix.py`, `analyze_n5.py`, `analyze_g_retest.py`): enumerate all C(10,5) = 252 relabelings of the pooled values and count the fraction with a difference at least as large as observed. No distribution assumed.
- **Resolution limit (important for interpretation).** With 5 vs 5 samples the *smallest possible* two-sided p is 2/252 = 0.0079. So p = 0.008 means *complete separation* of the two groups (every value of one group beyond every value of the other). It also means a Bonferroni correction across many contrasts makes significance mathematically unreachable at n = 5; results here are therefore reported with raw p-values, ranked by plausibility, with the multiple-comparisons caveat stated explicitly.
- **Multiple comparisons.** The matrix has 10 pre-listed contrasts × 5 metrics × 3 evaluations = 150 tests. At α = 0.05 one expects ≈ 7.5 nominally significant results by chance alone; 12 were observed, many of them correlated (e.g. 5 cm success across several contrasts). Conclusion drawn: only effects that are *complete-separation* (p = 0.008), mechanistically explained, **and** consistent across evaluations are treated as findings; everything else is a hypothesis.
- **Paired designs** where the same checkpoint is evaluated under different physics (e.g. A and B nominal vs DR) allow an exact sign test over seeds (5/5 improved ⇒ two-sided p = 0.0625, one-sided 0.031).

## 4. The anti-anomaly toolkit — how we tested that a finding is not a fluke

Every claim was subjected to as many of these as practical; doc 12 shows which applied to which claim.

| Tool | What it protects against | Example in this project |
|---|---|---|
| **Multi-seed replication (n ≥ 3, then ≥ 5)** | Seed luck | F4's single-seed "win" (29.4 vs 21.8 % at 2 cm) vanished: D's own seeds ranged 16.2–34.3 % |
| **Raw per-seed values reported next to means** | Means hiding outliers | Rung E's seed-42 value (43.1 %) was the second-best of its three seeds |
| **Matched budget & protocol** | Confounds from compute | Eureka vs vanilla compared at 800 iterations; Stage-1's comparison (5000 vs 1500) was confounded |
| **Independent oracle** | Trusting the pipeline | IK+PD controller reaches 100 % at 1 mm ⇒ the environment is not the limit; analytic FK equals simulator to 0.000 mm; closed-form dynamics vs PhysX |
| **Read-back verification** | Silent configuration failures | DR values read back from the simulator; Hydra overrides verified in dumped `env.yaml` |
| **Controlled ablation with a placebo** | Attributing an effect to the wrong cause | Rung G vs shuffled-multiplier Gp |
| **Pre-declared primary metric** | Post-hoc cherry-picking | Worst-decile error on DRshift fixed before the retest data existed |
| **Retest of a surprising result** | Chance findings | G's p = 0.024 (5 seeds) failed to reappear in the retest |
| **Direct mechanism check** | A correlation with no explanation | Controller trajectory logs confirm *why* ranges widened; TensorBoard confirms LEAP's ADR increments |
| **Isolating a single variable** | Entangled causes | V1 (entropy only) vs V2 (reward only) |
| **Red-flag audits** | Pipeline bugs | Identical metrics for G and Gp at seed 4 led to discovering shared run directories (doc 10) |
| **Refutation tests of one's own hypothesis** | Confirmation bias | The "two torque regimes" PINN fix was tested and refuted; its physics rationale was wrong |
| **Logging corrections in place** | Silent revision of history | Append-only `EXPERIMENT_LOG.md`, superseded entries marked, not deleted |

## 5. Standing rules adopted after the multi-seed corrections

1. No finding is reported from a single seed.
2. Comparisons use n ≥ 5 and exact permutation tests; effect sizes with raw p-values, no claims from p ≈ 0.05–0.1 alone.
3. Lead with tail metrics (worst-decile, mean error); treat mean 2 cm success cautiously (high variance).
4. Pre-declare the primary metric before looking at new data.
5. Identical results across supposedly different arms are a defect until proven otherwise.
6. Concurrent runs must not share timestamp-keyed output directories.
