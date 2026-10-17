# 16 — Pre-registration: E1, long-run exploration-collapse dose-response

**Written before any E1 run was started.** Timestamp of this file's creation: see the line `PREREG_TIME` at the bottom (UTC) and the git/file modification time. Nothing below may be changed after the first E1 training run starts; any later change is a *deviation* and must be logged in `EXPERIMENT_LOG.md` with its reason.

## Question
Is the Stage-1 vanilla-reward precision failure (13.28 % at 2 cm after 5000 iterations) a **training-length-dependent exploration-collapse artifact** that an entropy bonus prevents? The 800-iteration test (n = 5) found no significant entropy effect (2 cm: 28.99 vs 27.21 %, p = 0.49; worst-decile 149 vs 93 mm, p = 0.071), which is *consistent* with collapse not having happened yet at 800 iterations. E1 tests the long run directly.

## Design (fixed)
- Task `PIADR-TwoDofReach-Vanilla-v0`, human reward, **no DR**, 2048 envs, **5000 PPO iterations**, hyperparameters as in the 800-iteration runs except the one manipulated variable.
- Manipulated variable: `agent.params.config.entropy_coef` ∈ {0.005 (default), 0.02}.
- Seeds (both arms, fixed in advance): **42, 123, 7, 1, 2**. No seeds will be added or dropped whatever the outcome.
- Checkpoints every 100 iterations (already the config default). Evaluated checkpoints: iterations **{200, 400, 800, 1500, 3000, 5000}**, nominal physics, frozen metrics v1.0.0 (4096 eval envs, eval seed 1234, terminal error at t = 9.93 s). Secondary robustness evaluation (DRshift) at iterations {800, 5000}.
- Action-distribution entropy / σ read from the run's TensorBoard scalars at the same iterations.
- Runs are executed in two lanes (one per arm) started ≥ 90 s apart, with a post-batch audit that no two runs share a run directory or a final-checkpoint path.

## Pre-declared primary criterion (success)
At iteration 5000, nominal physics, **entropy 0.02 has lower mean terminal error AND lower worst-decile terminal error than entropy 0.005**, each by an exact two-sided permutation test over the 5 + 5 seeds with p < 0.05. (With n = 5 vs 5 the smallest attainable p is 0.0079.)

## Pre-declared secondary criteria (reported regardless)
- **S1 (collapse with training):** for entropy 0.005, 2 cm success at iteration 5000 is lower than at iteration 800 in **5 of 5 seeds** (exact sign test, one-sided p = 0.031).
- **S2 (mechanism):** final action entropy for 0.005 is lower than for 0.02 in 5 of 5 seeds; report σ-trajectory.
- **S3 (dose-response):** report mean ± sd of mean error, worst-decile, and 2 cm success versus iteration for both arms (figure/table); no test beyond S1.
- **S4 (robustness):** DRshift mean and worst-decile at 5000, entropy 0.02 vs 0.005 (exact permutation). Exploratory.

## Interpretation table (decided now)
| Outcome | What we will write |
|---|---|
| Primary met and S1 met | Explanatory success: the Stage-1 gap is a training-length exploration-collapse artifact, prevented by entropy; present the dose-response and mechanism |
| S1 met, primary not met | Degradation with training is real but entropy 0.02 does not fix it; report as a PPO long-run degradation finding |
| Neither met, no degradation | The Stage-1 vanilla gap is **not** explained by this mechanism and remains an unexplained discrepancy; report as such |
| Primary met, S1 not met | Entropy helps at 5000 iterations without a measurable decline of the default — report both facts, no collapse-mechanism claim |

## Rules
1. No peeking-and-adding: all 10 runs are completed before any decision.
2. A failed run (crash, truncation, collision) is **repeated from scratch with the same seed**, and the incident logged; it is never replaced by a different seed.
3. Resumed/interrupted runs are flagged in the tables.
4. The result is reported whichever way it falls; the ledger (doc 12) is updated either way.

PREREG_TIME: 2026-10-14 07:12:30 UTC
