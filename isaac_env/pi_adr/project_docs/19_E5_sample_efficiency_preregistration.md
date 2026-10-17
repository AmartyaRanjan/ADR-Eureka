# 19 — Pre-registration: E5, does the Eureka reward reach a given quality in fewer iterations than the human reward?

**Written before any E5 evaluation was run.** Analysis-only on existing checkpoints (no new training).

## Why this question
The five-seed matrix compared A (human reward) and B (Eureka reward) **at a fixed final budget of 800 iterations** and found no difference (mean error 37.75 vs 37.69 mm, p = 0.98). That does not test *speed*: a reward can leave final quality unchanged yet reach it earlier. The Stage-1 report's motivation for Eureka is partly efficiency. This experiment tests learning speed with the identical frozen metrics.

## Design (fixed)
- Runs: the 10 existing 800-iteration runs — A (vanilla, default entropy 0.005) and B (Eureka), seeds **42, 123, 7, 1, 2** (A seeds 1, 2 are the `vanilla_ent0.005_seed{1,2}` runs). Same protocol as the matrix (2048 envs, no DR).
- Checkpoints evaluated: iterations **{100, 200, 300, 400, 600, 800}**, nominal physics, frozen metrics v1.0.0 (4096 eval envs, eval seed 1234, terminal error at t = 9.93 s). 60 evaluations.
- **Primary metric:** the learning-curve area of *mean terminal error* (mean over the six checkpoints, trapezoid weights by iteration, lower is better). **Hypothesis (directional, set in advance):** Eureka (B) has the smaller area. Exact two-sided permutation test, 5 vs 5, p < 0.05 required.
- **Secondary:** (S1) mean error at 100 and at 200 iterations separately; (S2) iterations-to-threshold — the first checkpoint with mean error ≤ 45 mm — per seed, and with 2 cm success ≥ 20 %; seeds that never reach it are recorded as "> 800"; (S3) the same area for worst-decile error; (S4) 2 cm success area.

## Interpretation table (decided now)
| Outcome | What we write |
|---|---|
| Primary met in favour of Eureka | "Eureka reaches comparable quality earlier (sample-efficiency advantage) although the final budget shows no difference" — with the caveat below about search cost |
| Primary met in favour of the human reward | The human reward is both faster and equal at 800 iterations |
| Primary not met | No sample-efficiency advantage detectable at n = 5; the reward comparison is a null on both final quality and speed |

## Caveats stated in advance
1. **Reward-search cost is not counted.** The Eureka reward used here was taken from the Stage-1 report; its discovery required an LLM search with many candidate-reward training runs. Any "earlier" advantage must be weighed against that cost, which this experiment does not measure.
2. The two rewards have different scales, so training-return curves are not comparable; only frozen-metric evaluations of checkpoints are used.
3. n = 5 per arm; the minimum attainable p is 0.0079.

## Rules
All 60 evaluations complete before interpretation; transient simulator start-up errors are retried; results are reported either way.
PREREG_TIME: 2026-10-14 12:29:12 UTC
