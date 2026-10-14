# 15 — Plan: how to earn a genuine "success" for the paper

**Written for:** the project owner and co-authors deciding what to run next and how to frame the paper. **Date:** 13 Oct 2026, after the 2-DOF matrix (5 seeds), the G-vs-placebo analysis, and the 3-seed LEAP matrix.
**Status:** a proposal. Nothing in this document has been run; all probabilities are my judgment.

## 0. What "success" can honestly mean here

A paper needs a result a reviewer can *use*. There are three kinds, in increasing difficulty:

| Kind | Example | Evidence needed | Can we earn it this week? |
|---|---|---|---|
| **Explanatory success** | "The Stage-1 reward gap is explained by training-length-dependent exploration collapse; here is the mechanism, the dose-response curve and the fix" | One clean, replicated mechanism with a dose-response | **Yes — most likely** |
| **Methodological success** | "A validation and evaluation protocol that exposes spurious gains, demonstrated on real cases (including our own retractions)" | What we already have, plus one more case | Yes (largely done) |
| **Method success** | "Method X improves robustness on LEAP by Y %" | Improvement over baselines, ≥ 5–10 seeds, long budgets | Possible but uncertain; ≈ 25–35 % chance for the best candidate |

**Rules for earning it (non-negotiable):** (1) the success criterion — metric, seeds, test — is written down *before* running; (2) the same protocol is applied whichever way the result falls; (3) a failed criterion is reported as a failure; (4) no claim from a single seed or from p ≈ 0.05–0.1 alone. These rules are what make a success worth reporting, and what protect the paper from the failure mode we hit three times already (single-seed wins that vanished on replication).

## 1. What our own data tell us about where to aim

1. **Seed variance dominates small effects.** LEAP nominal: mean ≈ 8.4, std 2.1–2.8 rotations (CV ≈ 30 %); 2-DOF worst-decile: std 40–150 % of the mean. With 3–5 seeds only effects larger than ≈ 1.5–2 standard deviations can be detected.
2. **Floor effects hide differences.** At full DR (ADR = 25) every LEAP policy sits at 0.6–1.5 rotations; an evaluation at the extreme cannot discriminate. An intermediate-difficulty curve could.
3. **One mechanism has real support:** exploration collapse. Entropy fell 2.83 → −4.66 in the long vanilla run (Session 1); vanilla at 5000 iterations scored 13.28 % @ 2 cm while the same reward at 800 iterations scores 28.99 ± 4.20 % (n = 5). That is a training-length effect we have *observed but never isolated or replicated*.
4. **Our PINN-informed test was weak,** not decisive: the observer's uncertainty was dominated by one dimension and anti-correlated with error out of range. The right question — *would any informative signal help?* — was never asked.
5. **LEAP's ADR rule is coarse-threshold** (rotations/s ≥ 0.15), structurally like our F1/F2 controllers whose coarse gate hid precision loss. Whether LEAP has an analogous quality/quantity gap is unmeasured.
6. **Budgets are short** (500 iterations LEAP; 800 iterations 2-DOF), so ADR had little time to matter.

## 2. Candidate experiments, ranked

### E1 — Long-run exploration collapse: the dose-response experiment (**recommended first**)
- **Claim to earn:** *Stage-1's vanilla-reward precision failure is a training-length-dependent exploration-collapse artifact, preventable by an entropy bonus.* This is an explanatory success and directly rebuts/reframes Stage 1, with a mechanism.
- **Design:** vanilla reward, 2048 envs, seeds {42, 123, 7, 1, 2}, entropy_coef ∈ {0.005, 0.02}, **train 5000 iterations**, saving checkpoints every 200 iterations (`agent.params.config.save_frequency=200`). Evaluate the checkpoints at iterations {200, 400, 800, 1500, 3000, 5000} with the frozen nominal protocol. Log action σ / entropy from TensorBoard at the same points.
- **Pre-declared primary criterion:** at iteration 5000, entropy 0.02 has lower *mean terminal error* **and** lower *worst-decile error* than 0.005, exact permutation p < 0.05 for each (n = 5 vs 5 allows p down to 0.0079). Secondary: for entropy 0.005, 2 cm success at 5000 is lower than at 800 (paired over seeds, 5/5 declining, sign-test one-sided p = 0.031) — the "collapse degrades with training" half of the story. Tertiary: entropy trace correlates with precision loss across iterations.
- **Why it should work:** Session-1 evidence (single seed): 13.28 % @ 2 cm at 5000 (entropy −4.66) vs ≈ 29 % at 800; V1 (entropy 0.02, 1500 iters) 17.36 % and σ ≈ 0.29. The 800-iteration null (p ≥ 0.18) is *consistent* with the story: collapse has not yet happened at 800.
- **If it fails:** the 5000-iteration degradation is real but entropy does not fix it → report "degradation with training under PPO; entropy bonus insufficient", still a useful dose-response figure; or no degradation → the Stage-1 gap is *not* explained by this and stays an open discrepancy (also reportable).
- **Cost:** 10 runs × ≈ 60–80 min ≈ 10–13 h sequential (≈ 7 h with two concurrent if memory allows, using **distinct start times** to avoid the shared-directory bug) + ≈ 1 h of evaluations. **My estimated chance the primary criterion is met: 55–65 %.**
- **Why this is the best use of GPU time:** cheap, mechanistically motivated, the largest expected effect size (≈ 2× on 2 cm success in single-seed data), and it converts our weakest result (entropy at 800 iterations) into a dose-response curve.

### E2 — LEAP robustness curve on the existing checkpoints (analysis only, ≈ 2 h)
- **Problem it solves:** the ADR = 0 / 25 endpoints are floor-limited. Evaluate each of the 9 checkpoints at ADR ∈ {5, 10, 15, 20} in addition to {0, 25}: 36 extra evaluations (≈ 3 min each).
- **Pre-declared primary metric:** area under the rotations-vs-ADR curve (trapezoid over 0–25, normalised by 25). Secondary: rotations at ADR = 10 (the difficulty L2 starts near) and 15.
- **Hypothesis:** policies trained with ADR (L1) or from a warm start (L2) degrade more slowly with difficulty than L0: AUC(L2) ≥ AUC(L1) ≥ AUC(L0).
- **Statistics caveat:** n = 3 per condition ⇒ smallest p = 0.10 ⇒ this is **exploratory**. Its job is to (i) tell us whether any condition separates at an intermediate difficulty, and (ii) choose the operating point for a confirmatory run (E3) with more seeds. It cannot itself support a claim.
- **Chance it shows a clear separation:** ≈ 30 %; either outcome informs E3.

### E3 — Oracle-information ablation for rung G (≈ 6 h; a *ceiling* test)
- **Question:** was G's null due to a poor observer, or would *any* informative signal fail to help the adaptive controller?
- **Design:** controller identical to G, but the step multiplier for each dimension is computed from the observer's **true absolute error** in that dimension's parameters (available in the probe as a diagnostic) instead of its epistemic variance — a perfectly calibrated "uncertainty". Arms: oracle, placebo (shuffled oracle values), F4. Seeds ≥ 6, run **sequentially, arm-by-arm**.
- **Pre-declared primary:** DRshift worst-decile error, oracle vs placebo, exact permutation; secondary DRshift mean error, nominal 5 cm.
- **Decision logic:** if oracle ≈ placebo ⇒ the PINN-informed adaptive-DR idea is dead *independent of observer quality* — a strong, clean negative result for the roadmap's central hypothesis. If oracle beats placebo ⇒ calibration is the bottleneck, which justifies investing in a better observer (an actionable positive finding).
- **Chance oracle beats placebo:** ≈ 20–30 %. **Value regardless of outcome: high**, because it converts "the observer was weak" (an excuse) into a conclusion.

### E4 — Precision-aware ADR criterion for LEAP (method contribution; conditional)
- **Idea:** LEAP widens when rotations/s ≥ 0.15. If policies reach the rotation target but with sloppy control (drops, high effort, poor orientation error at goal), a *quantity-only* gate lets ADR advance through regimes where quality has degraded — exactly the F1/F2 coarse-gate failure we measured in 2-DOF.
- **Step 0 (diagnostic, ≈ 1 h, analysis only): is there a quantity/quality gap on LEAP?** Using the existing checkpoints, measure per-episode rotation count, drop rate, orientation error at goal reach, and effort vs difficulty (ADR increments). Proceed to E4 only if quality degrades earlier than rotation count (cheap test that the method is motivated).
- **If motivated:** modify `adr_criteria` to require both rotations/s ≥ 0.15 **and** a quality constraint (e.g. drop-rate below a threshold calibrated on L1 at ADR = 0); train 1500 iterations (≈ 2.5–3 h/run) for new vs native ADR, **≥ 7 seeds each** (power: with SD ≈ 2.5 rotations, detecting Δ = 4 at 80 % power needs ≈ 7 per group; Δ = 3 needs ≈ 11), evaluate the E2 robustness curve.
- **Cost:** 4–6 h coding/debug + ≈ 35–40 h GPU (14 runs). **Chance of a significant improvement: 25–35 %** (higher if the diagnostic shows a clear gap, ≈ 10 % if it does not — in which case stop).
- **Why include it:** it is the only candidate that yields a *method* success, and the mechanism (coarse-gate blindness) is already evidenced on the 2-DOF track, giving the paper a coherent story across both platforms.

### E5 — LEAP RAPP sweep and RAPP-initialised ADR (not this week)
Needed for the roadmap's literal claim; ≈ 1–2 days of coding + a day of runs. Defer unless E2/E4 show an ADR-initialisation effect worth testing.

## 3. Proposed order and schedule (Wed 14 – Sun 18 Oct; GPU shared, one run at a time)

| Step | What | GPU h | Decision gate |
|---|---|---|---|
| 0 (running) | clean G/placebo rerun, seeds 3–6 (chain15) | ≈ 4 (done ≈ 02:00) | none |
| 1 | **E1** 5000-iteration entropy dose-response | ≈ 10–13 | success → headline of the paper |
| 2 | **E2** LEAP robustness curve + E4 step-0 diagnostic | ≈ 3 | decides E4 |
| 3 | **E3** oracle ablation | ≈ 6 | settles the PINN question |
| 4 | **E4** (only if step-2 diagnostic motivates it) | ≈ 35–40 | method success or clean null |
Total ≈ 25 h without E4, ≈ 65 h with E4. E1–E3 fit this week; E4 fits only if the GPU stays available most of the time (shared-machine risk). I would write the paper sections for E1–E3 as results arrive, so a draft exists even if E4 slips.

## 4. How each outcome becomes a paper

**Best case (E1 succeeds, E3 negative, E4 positive):** title along the lines *"Spurious gains in learned reaching and adaptive randomization: a validation protocol, a mechanism, and a fix."* Contributions: validation protocol + audit trail; exploration-collapse dose-response; oracle-information ablation closing the PINN question; a precision-aware ADR criterion transferring from 2-DOF to LEAP. → a credible workshop paper (≈ 65–75 %) and a reasonable journal/RA-L candidate (≈ 40–55 %); a main-track conference becomes plausible (≈ 25–35 %).

**Middle case (E1 succeeds only):** an explanatory-success paper: Stage-1's gap rebuilt as a training-length artifact, with the protocol. Workshop ≈ 65–70 %; simulation journal ≈ 40–50 %.

**Worst case (E1 fails too):** the paper becomes a methodology paper whose message is "five attempts to establish gains failed under controlled replication, and here is why that matters"; workshop ≈ 55–60 %, journal ≈ 30–40 %. Still publishable because the audit trail and statistics are the contribution.

## 5. Risks and safeguards

| Risk | Safeguard |
|---|---|
| Concurrent-run directory collision (it contaminated seeds 3–6) | Launch jobs **sequentially**, or staggered; run the checkpoint-collision audit after every batch (script to be added: list all final-checkpoint paths and every chunk's experiment name; fail on duplicates) |
| Optional stopping / p-hacking | Success criteria above are fixed *now*; seeds fixed in advance; no peeking-and-adding |
| Shared GPU unavailable | Queue with the existing wait-for-resources logic; E1 first so the most valuable result lands early |
| Floor/ceiling effects | E2 intermediate-difficulty curve; report the primary metric chosen *before* looking |
| Single-run resume artifacts | Prefer uninterrupted runs; flag any resumed run in the tables |
| Over-claiming | Ledger (doc 12) updated for every result; retractions stay visible |

## 6. What I need from you

1. Approval to start **E1** after the G rerun (or ahead of it — the rerun cannot change the PINN verdict).
2. Whether to include **E3** (cheap, high informational value) — I recommend yes.
3. Whether to authorise **E4's** coding after the diagnostic (it is the only path to a method success; the cost is ≈ 40 GPU hours).
4. The target venue, so the paper can be framed (workshop vs simulation-methodology journal vs main track).
