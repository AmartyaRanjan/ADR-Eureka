# 17 — Pre-registration: E2, LEAP robustness curve on the existing checkpoints

**Written before any E2 evaluation was run.** Analysis-only: no new training. Purpose: the ADR = 0 and ADR = 25 endpoints were floor-limited (every policy ≈ 0.6–1.5 rotations at 25). An intermediate-difficulty curve may show whether the conditions separate somewhere, and decides whether the precision-aware ADR experiment (E4 in doc 15) is worth its ≈ 40 GPU hours.

## Design (fixed)
- Checkpoints: the 9 existing policies (L0, L1, L2 × seeds 42, 123, 7), paths in `results/LEAP_<cond>_s<seed>_checkpoint.txt`.
- New evaluations at frozen ADR ∈ {5, 10, 15, 20} (existing: 0 and 25), same protocol as before (1024 envs, 600 steps, eval seed 1234, ADR frozen).
- **Primary metric:** area under the rotations-vs-ADR curve (trapezoid over ADR 0–25, normalised by 25) per policy. **Secondary:** rotations at ADR = 10 and at ADR = 15.
- **Hypothesis (directional, set in advance):** AUC(L2) ≥ AUC(L1) ≥ AUC(L0).
- Statistics: exact permutation tests on the 3 vs 3 seeds. With n = 3 per group the smallest attainable p is 0.10, so **this experiment is exploratory by construction**; it cannot support a claim by itself.

## Decision rule for E4 (precision-aware ADR criterion), decided now
- Proceed to E4's step-0 diagnostic only if the primary-metric ordering matches the hypothesis in at least 2 of 3 pairings **or** if any condition shows a per-seed-consistent advantage (all three seeds of one condition above all three of another) at ADR 10 or 15.
- Otherwise: do **not** invest in E4 on this evidence; report that no difference in robustness could be detected on the existing checkpoints.

## Rules
1. All 36 evaluations are completed before interpretation.
2. A failed evaluation (simulator start-up error etc.) is retried with the same checkpoint; the incident is logged.
3. Results are reported whichever way they fall.
PREREG_TIME: 2026-10-14 07:20:34 UTC
