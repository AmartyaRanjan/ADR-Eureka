# 07 — The vanilla-reward underperformance: exploration collapse and the entropy probes

**Why this investigation happened.** Stage 1 reported that the human reward reaches 98.36 % at 2 cm and 79.74 % at 1 cm; our rebuilt benchmark gave 13.28 % and 2.22 %. Two anomalies in our own data pointed at the optimiser, not the reward equation: vanilla *got worse with training* (reward peaked at epoch 200 and decayed), and the action standard deviation collapsed.

## 1. Evidence of collapse (Session 1 log)

From TensorBoard of `A_vanilla_5000`: `losses/entropy` fell from **2.83 to −4.66**; the implied action σ fell from 1.0 to ≈ 0.024 (a naive exp(entropy) gives ≈ 0.009 — either way two orders below the report's stated late-training σ of 0.10–0.11). Adaptive learning rate dropped to 1e-4 with KL held near 0.01. Interpretation: exploration dies long before fine precision can be discovered; the policy settles into a coarse solution and the small precision gradient (≈ 0.6 % of return) is never found.

## 2. Two isolating probes (10 Oct 2026; vanilla reward, seed 42, 2048 envs, 1500 iterations)

Design principle: **change one variable per probe** so the cause is attributable.

| Probe | Change | Reward | entropy_coef |
|---|---|---|---|
| V1 | optimiser only | unchanged | 0.005 → **0.02** |
| V2 | reward only | action-penalty coefficient 0.005 → **0.001** | unchanged |

`VanillaRewardCfg(distance_weight, success_bonus, action_penalty_coef)` was added so V2 is a Hydra override, not a new reward branch; `validate_env.py` was re-run (10/10) to confirm the defaults still reproduce the report's equation exactly.

| | <5 cm | <2 cm | <1 cm | mean err | final entropy |
|---|---|---|---|---|---|
| Session-1 baseline (5000 iters, 4096 envs, entropy 0.005) | 93.04 % | 13.28 % | 2.22 % | 101.9 mm | −4.66 (σ ≈ 0.024) |
| **V1** (1500 iters, 2048 envs, entropy 0.02) | **97.83 %** | **17.36 %** | **3.91 %** | 30.6 mm | **−1.23 (σ ≈ 0.29)** |
| **V2** (action penalty 0.001) | 99.24 % | 12.23 % | 1.51 % | 38.0 mm | −4.12 |

**Reading (as written on 10 Oct):** V1 beat the baseline on every metric despite 3.3× fewer iterations and half the environments, and kept ≈ 30× more action σ. V2 left the collapse intact and did not improve precision. Conclusion at the time: *the dominant cause of the vanilla gap is the PPO entropy coefficient, not the reward's action penalty.*

## 3. The n = 5 recheck — what survives

V1 was a single seed and compared across different budgets (1500 vs 5000 iterations), which is a confound. A proper test was run at the **matched 800-iteration / 2048-env protocol** with 5 seeds per setting (nominal physics):

| metric | entropy 0.005 (n=5) | entropy 0.02 (n=5) | permutation p |
|---|---|---|---|
| 5 cm % | 98.48 ± 0.44 | 97.28 ± 1.82 | 0.262 |
| 2 cm % | 28.99 ± 4.20 | 27.21 ± 3.65 | 0.492 |
| 1 cm % | 8.39 ± 3.05 | 8.85 ± 5.80 | 0.889 |
| mean err (mm) | 37.75 ± 3.95 | 33.71 ± 4.81 | 0.183 |
| worst-decile (mm) | 149.38 ± 31.63 | 92.99 ± 45.67 | 0.071 |

Per-seed 2 cm, entropy 0.005 (seeds 42,123,7,1,2): 27.9, 29.4, 30.5, 22.8, 34.4. Entropy 0.02: 27.0, 33.4, 24.8, 26.7, 24.2.

**What this means.** At 800 iterations — when the default-entropy policy has not yet collapsed — raising entropy makes **no significant difference** to precision. The only suggestive effect is on the tail (worst-decile 149 → 93 mm, p = 0.071), consistent with entropy preserving robustness against rare failures. So:

1. The collapse is real (entropy −4.66 in the long run) and the long Session-1 run's degraded results were driven by **over-training with collapsed exploration** together with the mismatched budgets, not by an intrinsic weakness of the human reward.
2. The claim "entropy 0.02 fixes vanilla's precision" is **not** supported at matched short budgets. It is untested at 5000 iterations multi-seed; that is the experiment that would confirm the collapse mechanism directly (doc 13).
3. The residual gap to Stage 1's 98 %/80 % remains **unexplained**: no variant here exceeds ≈ 30 % at 2 cm. Either Stage 1's setup contained something not described in the report, or its numbers used a different success definition. This stays an open discrepancy; it is not claimed as resolved.

## 4. Why the exposure matters

The original Stage-1 narrative ("the discovered reward fixes the precision failure of the human reward") rested on the vanilla policy failing. The probes show the failure was largely an optimisation artifact; at matched budgets the rewards tie (doc 05). This is why the Eureka-vs-human comparison was retracted.

## 5. How we tested that the V1 effect was not an anomaly

| Test | Result |
|---|---|
| Single-variable isolation (V1 vs V2) | Only the optimiser change moved entropy and precision |
| Mechanism check (entropy trace) | σ retained ≈ 30× more with entropy 0.02 |
| Multi-seed matched-budget replication | Effect on precision **not** replicated; tail effect suggestive only (p = 0.071) |
| Verdict | The *mechanism* (collapse under long training) is credible; the *magnitude* claimed in V1 was inflated by comparing unequal budgets |


## 6. UPDATE (14 Oct 2026): E1 resolved the open question

The long-run test recommended above was run as the pre-registered experiment E1 (doc 16; results doc 21): vanilla reward, 5000 iterations, entropy 0.005 vs 0.02, 5 seeds each. **At 5000 iterations entropy 0.02 beats the default on mean error (30.4 ± 3.1 vs 61.8 ± 22.4 mm) and worst-decile error (66 vs 346 mm), p = 0.0079 for both** (complete separation); the default arm's 2 cm success falls from 29.0 % at 800 iterations to 15.2 % at 5000 in 5 of 5 seeds; final action entropy is −3.5…−4.5 (default) vs −0.6…+2.2 (0.02). So the section-3 conclusion is now settled in the direction anticipated there: *the collapse is real and is a training-length effect; entropy 0.02 prevents it; at 800 iterations it has not yet occurred.* The residual gap to Stage 1's reported 98 %/80 % precision remains unexplained (best E1 arm: ≈ 25 % at 2 cm).
