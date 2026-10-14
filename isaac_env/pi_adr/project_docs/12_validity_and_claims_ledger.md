# 12 — Validity audit and claims ledger

**Purpose.** For every claim that was ever made in this project: what the original evidence was, **how we tested whether it was ground truth or an anomaly**, and whether it survived. This is the document a reviewer should read first. Status key: **CONFIRMED** (independent oracle or replicated ≥ 5 seeds), **SUPPORTED** (≥ 3 seeds, directional), **INCONCLUSIVE**, **RETRACTED**, **PENDING**.

## 1. Environment and measurement claims

| # | Claim | Original evidence | How it was tested | Outcome | Status |
|---|---|---|---|---|---|
| 1 | The simulated arm matches analytic kinematics and dynamics | Built from the report's constants | `validate_env.py` 10/10 (FK error 0.000 mm, torque law 0.0000 Nm, DR read-back), `validate_dynamics.py` (2.27 % q̈ error vs PhysX) | Passed after correcting three *validator* checks that had encoded an idealised law | CONFIRMED |
| 2 | Evaluation reports true terminal error | Policy scored 0.39 % at 5 cm vs 93 % implied by return | Inspect auto-reset order; roll out `horizon − 2`; re-evaluate same checkpoint | 93.75 % / 20.70 % / 3.91 % | CONFIRMED (bug fixed) |
| 3 | The environment permits millimetre precision | Learned policies plateau at ≈ 30 mm | Scripted IK+PD through the same interface and metrics | 100 % at every threshold to 1 mm; failures with ±π limits were 100 % joint-stop stalls | CONFIRMED |
| 4 | Joint limits at ±π cap success at ≈ 69 % | PD failed 31.4 % of episodes | Failure analysis (all failures at a stop); widening limits → 100 % | Wrap-around trap | CONFIRMED |
| 5 | DR is actually applied and per-environment | Config exists | Read values back from the simulator; std 0.054 across envs | Applied | CONFIRMED |
| 6 | Mass randomization works at scale 1.0 | RAPP nominal read 0.816 | Compare to independent nominal eval (0.975) | Write was skipped at 1.0 | CONFIRMED (bug fixed) |

## 2. Reward and baseline claims

| # | Claim | Original evidence | How it was tested | Outcome | Status |
|---|---|---|---|---|---|
| 7 | Eureka reward is coarse-success dominated (high 5 cm, low 2/1 cm) | Stage-1 report; seed 42 reproduction (96.97 / 25.56 / 3.59 %) | n = 5 matched-budget runs (B): 5 cm 98.98 ± 0.76, 2 cm 26.87 ± 6.26, 1 cm 7.92 ± 2.07 % | Pattern holds in every seed | CONFIRMED |
| 8 | Eureka beats the human reward at every threshold | Single seed; vanilla 5000 vs Eureka 1500 iterations | Matched 800 iterations, n = 5 per reward, exact permutation | Mean err 37.75 vs 37.69 mm (p = 0.98); 2 cm 28.99 vs 26.87 % (p = 0.56) | **RETRACTED** |
| 9 | Vanilla fails to reproduce Stage 1's 98 % @ 2 cm | 13.28 % @ 2 cm at 5000 iters | Cross-check at several budgets (200–5000) and with entropy probes; matched n = 5 | Vanilla ≈ 29 % @ 2 cm at 800 iterations; never near 98 % | CONFIRMED (our setup cannot reproduce); cause of discrepancy **open** |
| 10 | The vanilla gap is due to entropy collapse | V1 (entropy 0.02): 97.83 / 17.36 / 3.91 %, 3.3× fewer iterations; V2 (reward change) no help | Single-variable isolation; then n = 5 at matched 800 iterations | At matched budget entropy has no significant effect on precision (p ≥ 0.18; worst-decile 149 → 93 mm, p = 0.071) | Collapse mechanism SUPPORTED; "entropy fixes precision" **INCONCLUSIVE/overstated** |
| 11 | Harder physics raises fine precision of no-DR policies | Eureka @1500 2 cm 26.4 → 42.4 % under shifted DR (single seed; "vanilla shows no such effect") | n = 5 paired by seed for both rewards | 5/5 seeds improve for A **and** B (sign test one-sided p = 0.031 each); the original statement that vanilla does not show it is **wrong** at 800 iterations | CONFIRMED (direction, both rewards); mechanism (slow-down) untested |

## 3. RAPP claims

| # | Claim | Evidence | Test | Outcome | Status |
|---|---|---|---|---|---|
| 12 | Feasible envelope is wide and nearly flat | friction ≤ 4, damping ≤ 256×, effectiveness ≥ 0.3, mass 1–4× at S ≥ 0.90 | Compare to adaptive rung E (no ceiling) behaviour; consistency of nominal (0.975) | Qualitatively corroborated: removing the ceiling lost nominal coarse success | SUPPORTED (single policy seed; edges uncertain) |
| 13 | Binding constraint is asymmetric in mass (light fails, heavy tolerated) | mass ×0.75 infeasible (S = 0.786), ×4 feasible (0.919) | Physical reasoning (lower inertia → overshoot); controller used the asymmetry | Consistent | SUPPORTED |
| 14 | 5 cm feasibility ≠ precision | 1 cm success rises in "hard" corners | F1/F2 coarse-gate behaviour; F4 precision gate | Same phenomenon at the controller level | SUPPORTED |

## 4. PINN claims

| # | Claim | Evidence | Test | Outcome | Status |
|---|---|---|---|---|---|
| 15 | Dataset labels were inconsistent with physics | True-ξ RMSE 78.8 > model 23.6 | Oracle (closed-form at ground truth); fix → 6.69 | Fixed | CONFIRMED |
| 16 | Mass–effectiveness degeneracy explains weak uncertainty | corr(m1,a1) +0.535, corr(m2,a2) +0.466 | Pre-set threshold, eigen-analysis, trace check | Present | CONFIRMED |
| 17 | Dual torque regimes break the degeneracy | Hypothesis | Same diagnostic | Made it worse (corr(m2,a2) +0.861, variance ×3.3); physics reasoning flawed | **REFUTED** |
| 18 | Steady-state holds break the degeneracy | Equations (mass drops out at q̈ = 0) | Same diagnostic | Joint 1 fixed, joint 2 not | PARTLY CONFIRMED |
| 19 | Sequential single-joint holds fix joint 2 (coupling) | Hypothesis | Same diagnostic | corr(m2,a2) 0.731 → 0.561; extrap worse; uncertainty rank corr +0.27 / −0.16 | PARTLY CONFIRMED |
| 20 | Uncertainty is a usable control signal | Roadmap assumption | Rank correlation with error; G vs shuffled placebo | Weak, negative out-of-range; placebo = G | **NOT SUPPORTED** |

## 5. Adaptive-DR and matrix claims

| # | Claim | Original evidence | Test | Outcome | Status |
|---|---|---|---|---|---|
| 21 | F1's nominal weakness is a budget/width mismatch | F1 nominal 5 cm 72.88 % | F2 (slower widening) | 90.94 % — directionally confirmed | SUPPORTED |
| 22 | A coarse gate blinds the controller to precision loss | F2 worst at 2 cm | F4 (2 cm gate) | Mechanism plausible; net benefit not replicated | SUPPORTED (mechanism only) |
| 23 | F4 beats static DR (cleanest result) | Seed 42: 29.42 vs 21.80 % @ 2 cm | n = 3 then n = 5 | D's own seeds span 16.2–34.3 %; F4−D p ≥ 0.2 everywhere | **RETRACTED** |
| 24 | Rung E has ~2× the precision of D under DR | Seed 42: 43.09 vs 21.80 % | n = 3 then n = 5 | Seed 42 was second-best of three; E−D 2 cm p = 0.18 | **RETRACTED** |
| 25 | Removing RAPP's ceiling costs nominal coarse success | E seed 42: 65.11 % | n = 5, per-seed values | E below D in 5/5 seeds, complete separation, p = 0.008 | CONFIRMED |
| 26 | RAPP's ceiling limits that cost | F4 vs E | n = 5 | F4−E nominal 5 cm +13.7 points, p = 0.040 | SUPPORTED |
| 27 | Static DR hurts fine precision for the Eureka reward when evaluated under DR | D vs B | n = 5, exact permutation, per-seed | 2 cm DRtrain 40.8 → 21.4 %, DRshift 43.2 → 19.8 % (p = 0.016 both); B > D in 5/5 seeds | SUPPORTED (p near minimum attainable; not mechanism-tested) |
| 28 | PINN-informed adaptive DR (G) beats F4 / static | G lowest mean/tail error and variance | Placebo, pre-declared primary metric, retest | G ≈ placebo at nominal/DRtrain (p ≥ 0.74); DRshift p 0.024 (5 seeds) → 0.147 (6 clean) | **NOT SUPPORTED** |
| 29 | A reward/DR variant improves tail error robustly | Various single-seed hints | 150-test matrix with exact permutation | Nothing survives multiplicity correction; n = 5 cannot reach Bonferroni thresholds | INCONCLUSIVE |

## 6. LEAP claims

| # | Claim | Evidence | Test | Outcome | Status |
|---|---|---|---|---|---|
| 30 | `LEAP_500` is a no-DR baseline | Stock config assumed | Read config; TensorBoard `num_adr_increases` 0 → 8 of 25, `adr_criteria` peak 0.315 | Native ADR was active | **RETRACTED** (relabelled) |
| 31 | LEAP already ships adaptive DR | Code reading | Verified in TensorBoard of the actual run | Confirmed | CONFIRMED |
| 32 | "Queue failures = host-RAM OOM" | Free RAM ≈ 0 | Per-attempt logs: rc = 0, "Network won!" | Wrong; upstream `score_to_win` early stop | **RETRACTED** |
| 33 | L0 vs L1 vs L2 differ | Seed 42: nominal 10.88 vs 7.53 (L0 ≫ L1) | 3 seeds each, frozen-ADR evaluation, exact permutation | Nominal 8.42 / 8.68 / 8.00 (p ≥ 0.70); full DR 0.56 / 0.75 / 1.53 (p ≥ 0.20); seed spread > condition gaps; L2's full-DR mean is one seed | **NOT SUPPORTED** (seed-42 impression retracted); min attainable p is 0.10 |
| 33b | ADR-trained LEAP policies use more control effort | L1 0.72 vs L0 0.60 per joint | 3/3 seeds separate (complete separation) | Directional (p = 0.10 is the minimum for n=3) | SUPPORTED (weak) |

## 7. Pipeline-integrity claims

| # | Claim | Test | Outcome |
|---|---|---|---|
| 34 | Matrix seeds are independent between rungs | Audit of all final-checkpoint paths and adaptive chunk experiment names | Seeds 42/123/7/1/2 clean; **G/Gp seeds 3–6 contaminated** (shared timestamped run directories); quarantined; rerun queued |
| 35 | The 10-seed G-vs-placebo retest (p = 0.281) | Red flag: identical G/Gp numbers at seed 4 | Retest used contaminated seeds; clean n = 6 gives p = 0.147; verdict unchanged | **SUPERSEDED** |

## 8. How to read the whole ledger

- **What is firm:** the environment's correctness, the precision ceiling, the Eureka coarse-precision signature, no reward effect at matched budget, E's nominal cost, the PINN identifiability findings, LEAP's native ADR.
- **What is *suggestive* but unproven:** static DR lowering 2 cm precision (p = 0.016, n = 5), RAPP's ceiling effect, G's tail advantage.
- **What was claimed and withdrawn:** reward superiority, F4 and E wins, entropy "fix" magnitude, G's DRshift signal, the LEAP baseline label.
- **Meta-lesson:** all three *positive* adaptive-DR findings came from one seed and each disappeared on replication; each replication was triggered by an explicit "do not claim until multi-seeded" rule. The final results are therefore stronger *because* they are mostly null.
