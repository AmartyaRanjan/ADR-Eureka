# 05 — Stage-1 baselines and reproduction

**Goal at this point (15 Sep 2026):** establish trustworthy baselines on the rebuilt benchmark and test whether Stage 1's central claims reproduce, using one frozen metric suite so every later result is comparable.

## 1. Two run generations

**Generation 1 (superseded).** First full set, trained before two environment defects (damping override, ±π trap — doc 04) were found:

| Run | 5 cm | 2 cm | 1 cm | mean err | median | |q̇| | effort |
|---|---|---|---|---|---|---|---|
| vanilla @5000 | 79.98 % | 3.88 % | 0.81 % | 142.3 mm | 43.1 mm | 1.64 | 0.571 |
| vanilla @1000 | 87.11 % | 23.14 % | 5.96 % | 92.6 mm | 33.4 mm | 2.22 | 0.773 |
| vanilla @200 | 92.43 % | 14.75 % | 2.29 % | 55.0 mm | 32.8 mm | 3.07 | 1.290 |
| Eureka @500 | 95.56 % | 26.15 % | 7.74 % | 50.3 mm | 32.2 mm | 3.09 | 1.372 |
| Eureka @1500 | 93.02 % | 10.96 % | 2.64 % | 63.9 mm | 37.5 mm | 2.78 | 1.075 |

Two anomalies drove the investigation: vanilla got *worse* with training (reward peaked at epoch 200 = 1281 and fell to 1098 by epoch 5000) and every policy plateaued near a **32 mm median error** regardless of reward or training length. A plateau independent of both reward and training length points at the environment or optimiser, not the reward — which is what the IK+PD control then tested.

**Generation 2 (current baselines, corrected benchmark).** `A_vanilla_5000`, `B_eureka_500`, `B_eureka_1500`, `PD_reference` (all seed 42; vanilla/Eureka 4096 envs).

## 2. Headline results (generation 2, nominal physics, seed 42)

| Run | <5 cm | <2 cm | <1 cm | <5 mm | <1 mm | mean (mm) | median (mm) | p90 (mm) | mean |q̇| | effort |
|---|---|---|---|---|---|---|---|---|---|---|
| **IK+PD reference** (not learned) | 100 % | 100 % | 100 % | 100 % | 100 % | 0.0 | 0.0 | 0.0 | 0.155 | 0.013 |
| Vanilla PPO @5000 | 93.04 % | 13.28 % | 2.22 % | 0.34 % | 0.02 % | 101.9 | 34.7 | 44.3 | 1.534 | 0.514 |
| Eureka PPO @500 | 96.97 % | 25.56 % | 3.59 % | 0.78 % | 0.05 % | 46.4 | 28.5 | 43.9 | 3.396 | 1.436 |
| Eureka PPO @1500 | 99.05 % | 26.37 % | 7.91 % | 1.49 % | 0.05 % | 38.6 | 30.1 | 41.5 | 2.522 | 1.073 |

### Comparison with the Stage-1 report

| Metric | Report (Eureka 500) | **Ours (Eureka 500)** | Report (Eureka ~1500) | **Ours (Eureka 1500)** |
|---|---|---|---|---|
| <5 cm | 97.90 % | 96.97 % | 95.3 % | 99.05 % |
| <2 cm | 19.02 % | 25.56 % | 18.1 % | 26.37 % |
| <1 cm | 3.56 % | 3.59 % | 5.3 % | 7.91 % |
| mean error | 33.5 mm | 46.4 mm | 45.5 mm | 38.6 mm |
| median error | 34.6 mm | 28.5 mm | 33.3 mm | 30.1 mm |

**What this means.** The Eureka reward's signature — high success at 5 cm, then a sharp collapse at 2 cm and 1 cm — **reproduced on an independently rebuilt benchmark** (no Stage-1 code, logs or checkpoints exist on this machine, so this is a re-implementation from the report's equations, not a continuation). The proposed mechanism was measured: the +100 success bonus dwarfs the residual e^(−10d) shaping once inside 5 cm, so there is no economic reason to refine centimetres to millimetres.

## 3. The ceiling: the environment is not the limit

The IK+PD controller, driven through the *same* action interface (normalised → ±20 Nm) and evaluated with the *same* metric suite, reaches **100 % success at every threshold down to 1 mm** (median error 0.0 mm).
**Significance:** millimetre accuracy is attainable at 30 Hz with ±20 Nm, so the learned policies' ≈ 30 mm floor is a property of the reward and optimiser, not the robot, control rate or physics. This control is what makes the reward comparison meaningful at all.

## 4. Where we differed from Stage 1 — and why

The report claims the human reward reaches **98.36 % at 2 cm and 79.74 % at 1 cm**. We did **not** reproduce this: vanilla reached 13.28 % / 2.22 % and was *worse* than Eureka at every threshold in the first comparison. Two measured reasons (the second investigated in doc 07):

1. The vanilla reward's precision incentive is also weak: going from 3 cm to 0.5 cm earns ≈ 0.025 per step ≈ 7.5 reward against an episode return of ≈ 1300 — about 0.6 %. The +5 bonus dominates in the same way Eureka's +100 does, only less extremely.
2. Action standard deviation collapses (entropy 2.83 → −4.66), so exploration dies before fine precision is found.

**Later correction (matched budget, 5 seeds, doc 10):** at equal 800-iteration budget the two rewards are statistically indistinguishable (mean error 37.75 vs 37.69 mm, p = 0.98; 2 cm 28.99 vs 26.87 %, p = 0.56). The Session-1 conclusion "Eureka beats the human reward" came from one seed at mismatched budgets (vanilla 5000 iterations with collapsed exploration vs Eureka 1500) and is **retracted**. What survives: *neither* reward approaches the IK+PD ceiling, and both are coarse-reaching rewards.

## 5. Robustness under domain randomization (evaluation-time DR, Session 1, single seed)

Policies trained *without* DR, evaluated under per-episode randomization (static DR). In-distribution lies inside the RAPP envelope; shifted is strictly harder and unseen.

| Policy | Physics | <5 cm | <2 cm | <1 cm | mean (mm) | median (mm) | mean |q̇| |
|---|---|---|---|---|---|---|---|
| Eureka @1500 | nominal | 99.05 % | 26.37 % | 7.91 % | 38.6 | 30.1 | 2.522 |
| Eureka @1500 | in-distribution DR | 97.88 % | 34.89 % | 9.30 % | 45.9 | 25.1 | 1.280 |
| Eureka @1500 | shifted DR | 91.14 % | **42.36 %** | **11.47 %** | 107.0 | **22.4** | 0.951 |
| Vanilla @5000 | nominal | 93.04 % | 13.28 % | 2.22 % | 101.9 | 34.7 | 1.534 |
| Vanilla @5000 | in-distribution DR | 92.14 % | 12.99 % | 2.73 % | 108.8 | 33.6 | 0.745 |
| Vanilla @5000 | shifted DR | 80.69 % | 13.18 % | 2.64 % | 194.8 | 35.2 | 1.169 |

**The observation:** for the Eureka policy, harder physics *degrades* coarse success (99.05 → 91.14 % at 5 cm) while *improving* fine precision (26.4 → 42.4 % at 2 cm; median 30.1 → 22.4 mm). Friction, damping and weaker actuators impose a slowness the reward never asked for (mean joint velocity 2.52 → 0.95 rad/s). The interpretation offered at the time: independent evidence that the precision gap is an objective-design problem, not a capability problem.

**Follow-up test (5 seeds, matched 800-iteration protocol, evaluation-time DR on policies trained without DR; data in `results/MATRIX_A_to_G.md`, per-seed values in doc 10).** The direction replicates for *both* rewards, contrary to the original statement that vanilla shows no such improvement:

| Policy | 2 cm nominal | 2 cm DRtrain | 2 cm DRshift | seeds improving nominal→DRshift |
|---|---|---|---|---|
| A (vanilla) | 28.99 ± 4.20 % | 34.19 ± 7.13 % | 37.57 ± 10.30 % | 5 of 5 (sign test one-sided p = 0.031) |
| B (Eureka) | 26.87 ± 6.26 % | 40.77 ± 11.27 % | 43.24 ± 11.81 % | 5 of 5 (one-sided p = 0.031) |

So "harder physics raises 2 cm precision of a no-DR policy" holds at n = 5 for both rewards (exact per-policy sign test p = 0.031 one-sided; magnitudes about +9 to +16 points), while mean/tail error under DRshift is much worse (A: 87 mm mean, 666 mm worst-decile). The slow-down explanation remains plausible but was not separately tested (mean joint velocity was not re-measured per seed).

## 6. Evaluation notes

- Throughput: vanilla and Eureka trained concurrently on one L4; wall-clock per run is therefore not a clean efficiency metric and is not used as one.
- All rows in this document use METRICS_VERSION 1.0.0, 4096 eval envs, eval seed 1234, terminal error at t = 9.93 s.
- Reproduction commands: doc 14.
