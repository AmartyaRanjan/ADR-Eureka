# 08 — The physics-informed observer (PINN): design, four protocol iterations, identifiability findings

**Purpose.** The roadmap's novel ingredient: a network that infers the arm's physical parameters ξ from a short window of motion, with uncertainty, so an adaptive-DR controller could prioritise physics regions where the model is least sure. This document explains what was built, how it was repeatedly debugged and improved, what was found, and why the final verdict is "useful for analysis, not a reliable controller signal".

## 1. Design

- **Parameters:** ξ = [m1, m2, f1, f2, b1, b2, α1, α2] — link masses, Coulomb friction, viscous damping, actuator effectiveness (per joint).
- **Physics (closed form, `source/pi_adr/pinn/dynamics.py`):** M(q;ξ) q̈ + C(q,q̇;ξ) + F(q̇;ξ) = α ⊙ τ_cmd. The closed form was verified against PhysX (2.27 % mean relative error on q̈; 0.036 Nm on recovered torque), so the physics residual is exact.
- **Design rule:** α stays on the *right-hand side* and the network is fed the **commanded** torque. Feeding it the already-scaled torque would make α unidentifiable by construction.
- **Model (`pinn/model.py`):** set-encoder `PhysicsObserver` over a window of 64 transitions (q, q̇, τ_cmd, q̈), deep ensemble of 5 members; predictive variance = epistemic spread between members + mean predicted (aleatoric) variance.
- **Loss:** data/acceleration term L_data, physics residual L_phys (weight 0.01), supervised parameter term L_sup (privileged ξ available in simulation).
- **Splits:** by *physics configuration* — never time samples from a configuration already seen — so numbers are not memorisation. Data: train 4096 envs (seed 0), val 1024 (seed 7, same ranges), extrapolation 1024 (seed 11, ranges outside training).

## 2. Defect found first: the labels did not satisfy physics (D4)

The first observer: parameter error 0.23, acceleration RMSE 23.6 rad/s². Checking the data: evaluating the closed form at the **ground-truth** ξ gave RMSE **78.8 rad/s²** — worse than the model. Fixes: step physics per 1/120 s step (not over a 4-substep control step), mask velocities above 3.5 rad/s (the velocity limit engages from ≈ 4 rad/s: median error 4.1–4.4 rad/s² below 4 rad/s but **68.2** at 4–4.9). After both: true-ξ RMSE **6.69 rad/s²** (5.3 % of mean |q̈|).

## 3. Baseline observer (single torque regime 0.25)

| | acc RMSE (rad/s²) | param error (normalised) | uncertainty↔error rank corr |
|---|---|---|---|
| interpolation | 2.79 | **0.065** | +0.11 |
| extrapolation | 12.28 | 0.209 | +0.10 |

Per-parameter absolute error (interp): m1 0.179 kg, m2 0.051, f1 0.095, f2 0.114, b1 0.042, b2 0.017, α1 0.064, α2 0.052.
**Two findings:** (i) extrapolation degrades ≈ 3× — the honest, expected result; the observer must not be trusted outside its training range, supporting the rule that adaptive DR stays inside C_RAPP; (ii) **uncertainty is nearly uninformative about error** (rank correlation +0.11), so the roadmap's priority score term U_PINN would be noise.

## 4. Diagnosing *why* uncertainty is uninformative: identifiability

**Hypothesis (tested directly on saved artifacts, no new training):** link mass and actuator effectiveness trade off — a heavier link driven by a stronger actuator gives the same acceleration, α/M being what the data constrain. If so, the ensemble members should disagree on m and α in a *correlated* way that a diagonal-only variance estimate hides.

`scripts/diagnose_pinn_covariance.py`: cross-member correlation of the 5 members' predictions, averaged over validation environments.

| pair | single-regime |
|---|---|
| corr(m1, a1) | **+0.535** |
| corr(m2, a2) | **+0.466** |
| corr(a1, a2) | **−0.568** |

All above the |r| > 0.3 threshold set *in advance*. The dominant eigenvector of the full covariance is 92 % aligned with m1 and mixes in α1/α2 (+0.308/−0.210): ≈ 72 % of total disagreement lives in one combined mass–effectiveness direction. Sum of diagonal variances equals the sum of eigenvalues (0.004507) — trace invariance — confirming it is about *how* uncertainty is distributed, not its total. **Meaning:** the observer says one thing loudly ("I cannot separate m1 from a1") and nearly nothing else; a controller reading marginal variances sees two moderate, seemingly independent numbers.

## 5. Four excitation-protocol iterations (what was tried, why, what happened)

### 5.1 Dual torque regimes — **tested and REFUTED** (10 Oct)
- *Rationale:* vary torque amplitude (0.15 and 0.45) so Coriolis/friction terms vary relative to α·τ/M, breaking the degeneracy.
- *Result:* the degeneracy got **worse**: corr(m1,a1) +0.652, corr(m2,a2) **+0.861**, new large corr(m1,m2) +0.504, total variance ×3.3 (0.0147), val RMSE 2.93 (no gain).
- *Why the idea was wrong (post-mortem):* the inertial term is α·τ/M — **linear** in τ for any fixed (α, M). Doubling torque doubles the acceleration regardless of how α/M splits; amplitude gives more points along the *same* degenerate direction, not a new direction of information. Mixed amplitudes plausibly made the fitting task harder.
- *Lesson:* a clean refutation is a result; the wrong physics reasoning is documented, not hidden.

### 5.2 Steady-state ("long-hold") excitation (10 Oct)
- *Rationale from the equations:* at constant torque held until q̈ → 0, mass drops out: α·τ = C + F — constraining friction/damping/α without mass. The existing protocol (hold 6 steps ≈ 0.2 s, resample 32 steps ≈ 1.07 s) never approaches terminal velocity.
- *Sizing (`scripts/estimate_settling_time.py`, no simulator):* time constant τ_c = M/b ranges from ≈ 158 physics steps (damping ×4) to **≈ 2521 steps (21 s)** at damping ×0.5 — two orders longer than the hold windows.
- *First attempt failed:* one fixed torque for all environments gave **0.2 %** usable steps. Steady-state velocity v_ss = α·τ/b exceeds the ±5 rad/s limit for low-damping/high-effectiveness environments — "steady state" there is "pinned against the velocity limit", masked out correctly but uselessly.
- *Fix:* because this is simulation data, size each environment's hold torque from its own ground-truth (b, α) to target a common 2 rad/s (`clamp(v·b/(α·20), 0.01, 1)`). Usable fraction: **96.3 %** in the final 30 steps.

Result (all numbers vs earlier protocols):

| | single-regime | dual-regime (failed) | **long-hold** |
|---|---|---|---|
| interp acc RMSE | 2.79 | 2.93 | **2.08** |
| interp param err | 0.065 | 0.071 | 0.064 |
| interp uncertainty↔error rank corr | +0.114 | +0.255 | **+0.293** |
| extrap acc RMSE | 12.28 | 12.98 | **5.93** |
| extrap param err | 0.209 | 0.223 | **0.161** |
| extrap rank corr | +0.097 | +0.000 | **−0.150** |
| corr(m1,a1) | +0.535 | +0.652 | **−0.175** (fixed) |
| corr(m2,a2) | +0.466 | +0.861 | **+0.731** (not fixed) |
| corr(a1,a2) | −0.568 | +0.636 | +0.018 (fixed) |
| corr(m1,m2) | −0.201 | +0.504 | **−0.576** (new) |

Accuracy improved substantially (extrapolation RMSE more than halved); joint 1's degeneracy was resolved; joint 2's was not; extrapolation uncertainty became *anti*-correlated with error (with only 1024 extrapolation environments this specific number is the least reliable).
- *Hypothesis for joint 2:* both joints were driven simultaneously to independently calibrated velocities, but the arm is a coupled 2-link system (joint 2 acceleration depends on joint 1 velocity through the −h·q̇1² term and full M⁻¹ mixing), so joint 1 motion contaminates joint 2's "steady state".

### 5.3 Sequential single-joint holds — `seq` observer (11 Oct)
- *Test of the coupling hypothesis:* first half of the hold drives joint 1 only (joint 2 torque zero), second half joint 2 only (per-env calibrated target 2 rad/s, 300 hold steps, transient steps 192, torque scales 0.15/0.45); 99.9 % valid transitions in the smoke test.
- *Result* (`results/pinn_observer_seq.json`, `pinn_covariance_diagnosis_seq.json`):

| | long-hold | **seq** |
|---|---|---|
| interp acc RMSE | 2.08 | **1.95** |
| extrap acc RMSE | 5.93 | 7.73 (worse) |
| interp param err | — | 0.0606 |
| corr(m2,a2) | +0.731 | **+0.561** |
| corr(m1,a1) | −0.175 | +0.078 |
| corr(m1,m2) | −0.576 | −0.361 |
| corr(a1,a2) | +0.018 | +0.035 |
| uncertainty↔error rank corr interp / extrap | +0.293 / −0.150 | **+0.267 / −0.155** |
| per-parameter interp abs error | | m1 0.185, m2 0.054, f1 0.085, f2 0.115, b1 0.027, b2 0.012, a1 0.071, a2 0.056 |

Sequential excitation reduced the m↔α and m1↔m2 degeneracies and improved interpolation accuracy and ranking over the original (+0.27 vs +0.11), but extrapolation did not improve and uncertainty remains anti-correlated with error there. The m1 error stays the largest. Decision: use `seq` for rung G, with the stated caveat; the shuffled-placebo arm is the control for exactly this weakness.

## 6. Probing the *training distribution* (`scripts/pinn_probe.py`)

For rung G, the frozen observer is applied to the controller's *current* range one dimension at a time (others nominal, same excitation protocol, 512 envs). It reports mean epistemic variance per dimension (normalised by the observer's XI_BOUNDS width²). Smoke-test values at the initial ranges: friction 2.0e-4, damping 5.3e-4, effectiveness 5.9e-4, mass 0.9e-4 (abs-error diagnostics 0.10, 0.06, 0.07, 0.05). In running G, damping scale's epistemic variance typically exceeded the others by up to ≈ 10×, saturating the clipped multipliers — a *weak test* of the idea (doc 10).

## 7. How we tested that the PINN findings are real

| Finding | Test | Verdict |
|---|---|---|
| Labels were wrong | Ground-truth ξ through the closed form → 78.8 vs model 23.6; fixed → 6.69 | Confirmed (oracle) |
| Mass–effectiveness degeneracy | Pre-set threshold |r| > 0.3; trace-invariance check; eigen-analysis | Confirmed |
| Dual-torque fix | Same protocol, same validation construction | Refuted (and explained) |
| Long-hold fix | Same diagnostic as acceptance test | Partially confirmed (joint 1 only) |
| Sequential fix | Same diagnostic | Partially confirmed; extrapolation not fixed |
| Weak uncertainty calibration | Rank correlation on held-out physics | Persistent in all four versions |
| Limitation | Single training seed per observer version; extrapolation split only 1024 envs | Reported, not hidden |

## 8. Verdict

The observer is **accurate in-range** (acceleration RMSE ≈ 2 rad/s² on held-out physics, normalised parameter error ≈ 0.06; note the closed form evaluated at the *true* parameters still has a 6.69 rad/s² RMSE against simulator labels, so the two RMSE figures are measured differently and should not be compared directly) and gave real insight into identifiability: mass/effectiveness cannot be separated from transient data, and sequential steady-state excitation partly separates them. It is **not** a reliable source of uncertainty for control (rank correlation +0.27 in-range, negative out-of-range; marginal variances dominated by one dimension). Consequently the downstream test (rung G) was always going to be a weak test of the roadmap's claim; its negative result should be read in that light.
