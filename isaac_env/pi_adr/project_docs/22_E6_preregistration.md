# 22 — Pre-registration: E6, the Eureka reward in the long run — was Eureka's apparent advantage an exploration-collapse artifact?

**Written before any E6 run was started.** Nothing below may change after the first E6 training run starts; later changes are logged as deviations.

## Why this question
E1 (doc 21) showed that with the default entropy coefficient (0.005) the **human** reward's policy degrades over 5000 iterations while entropy 0.02 prevents it. Stage 1 reported the Eureka reward beating the human reward — but in a long run at the default entropy. If the Eureka reward is *less* sensitive to exploration collapse, a long-run comparison at the default entropy would favour Eureka for reasons unrelated to reward quality. E6 tests this directly with the Eureka reward at the same budgets, seeds and protocol as E1.

## Design (fixed)
- Task `PIADR-TwoDofReach-Eureka-v0` (Eureka reward), no DR, 2048 envs, **5000 iterations**, `entropy_coef` ∈ {0.005, 0.02}, seeds **42, 123, 7, 1, 2** (the same seeds as E1). Identical evaluation protocol and checkpoints as E1 (nominal at iterations 200, 400, 800, 1500, 3000, 5000; DRshift at 800 and 5000; frozen metrics v1.0.0, 4096 eval envs, eval seed 1234). The human-reward arms are the existing E1 results (same seeds, same protocol, same budgets).
- Lanes run in parallel, started ≥ 100 s apart; start-up failures retried; every evaluation validated (arm moved, error < 1.5 m); post-batch audit of run-directory collisions and final epochs.

## Pre-declared questions and criteria
- **Primary (P):** with exploration protected in both rewards (entropy 0.02), at 5000 iterations on nominal physics, compare **Eureka-0.02 vs human-0.02** on mean terminal error and worst-decile error (exact two-sided permutation, 5 vs 5, p < 0.05 per metric).
- **S1 (does Eureka also collapse?):** for Eureka at entropy 0.005, 2 cm success at 5000 < at 800 in 5 of 5 seeds (exact sign test, one-sided p = 0.031); and mean-error change reported.
- **S2 (Stage-1-style comparison):** both rewards at the default entropy 0.005, 5000 iterations — Eureka-0.005 vs human-0.005 on mean and worst-decile error (exact permutation). This reproduces how Stage 1 likely compared them.
- **S3:** final action entropy (TensorBoard `losses/entropy`, last 20 epochs) for the Eureka arms.
- **S4 (exploratory):** 2 cm and 1 cm success at 5000 for all four arms; DRshift at 5000.

## Interpretation table (decided now)
| Outcome | What we write |
|---|---|
| S2 significant in Eureka's favour **and** P not significant | *Eureka's long-run advantage at default settings is an exploration artifact*: it disappears when both rewards are given adequate exploration |
| S2 significant in Eureka's favour **and** P significant in Eureka's favour | Eureka retains a genuine long-run advantage even with exploration protected |
| P significant in the human reward's favour | The human reward is better in the long run once exploration is protected |
| S2 and P both not significant | No long-run reward advantage in either direction at n = 5; the reward comparison is a null at 800 and 5000 iterations |
| S1 met | Eureka also degrades with long training under the default entropy (a PPO-level, not reward-specific, effect) |

## Caveats stated in advance
1. The Eureka reward was taken from the Stage-1 report; reward-search cost is not counted.
2. Two tasks' worth of generality is not tested: this is still the 2-DOF arm.
3. n = 5 per arm; minimum attainable two-sided p = 0.0079.

## Rules
All 10 runs complete before interpretation; a failed run is repeated from scratch with the same seed; interrupted runs are resumed and flagged; results reported either way; the ledger (doc 12) is updated whichever way it falls.
PREREG_TIME: 2026-10-15 16:35:43 UTC
