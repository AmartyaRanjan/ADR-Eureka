# Experiment log — Physics-Informed Adaptive DR

Append-only. Every run that produces a number gets an entry. Metrics are defined once in
`source/pi_adr/evaluation/metrics.py` (`METRICS_VERSION`); changing them invalidates
cross-run comparison and requires a version bump plus re-runs.

## Environment of record

| Component | Version | Source |
|---|---|---|
| GPU | 1x NVIDIA L4 (23 GB), driver 595.84 | — |
| OS / Python | Ubuntu 26.04, Python 3.11 (`../env_isaaclab`) | — |
| Isaac Sim | 5.1.0.0 (pip) | https://developer.nvidia.com/isaac/sim |
| Isaac Lab | 2.3.0 @ `3c6e67bb5` (2025-10-28) | https://github.com/isaac-sim/IsaacLab |
| rl_games | 1.6.1 | https://github.com/Denys88/rl_games |
| torch | 2.7.0+cu128 | — |
| IsaacLabEureka | `716657b` (2025-10-28), locally patched for Gemini | https://github.com/NVlabs/IsaacLabEureka |
| LEAP_Hand_Isaac_Lab | `c576e6a` (2025-09-04), one-line 2.3 compat patch | https://github.com/leap-hand/LEAP_Hand_Isaac_Lab |
| RAPP method | — | https://github.com/eureka-research/DrEureka |

---

## 2026-09-15 — Session 1: build both benchmarks, reproduce stage-1 baselines

### Infrastructure fixes (all required before anything ran)

| # | Problem | Fix | Touches |
|---|---|---|---|
| 1 | `import isaaclab` failed — venv editable installs point at `/home/.../projects/I`, project had moved to `/data/home/.../projects/I` | symlink `/home/amartyaranjan888/projects/I -> /data/home/...` | host fs (reversible) |
| 2 | URDF→USD importer missing `libxml2.so.2`, `libicu*.so.73` (absent on Ubuntu 26.04) | `.libshim/` with symlinks to the user's miniconda copies, used only by `scripts/convert_asset.sh`; cached USD makes train/eval shim-free | pi_adr only |
| 3 | LEAP `pip install -e` fails: `setup.py` reads `keywords`, absent from its `extension.toml` | not installed; `PYTHONPATH=source/LEAP_Isaaclab` instead | nothing |
| 4 | LEAP `scripts/rl_games/train.py` imports `dump_pickle`, removed from `isaaclab.utils.io` in 2.3 | use `pi_adr/scripts/train.py` (derived from Isaac Lab 2.3's own train.py + task registration) | pi_adr only |
| 5 | LEAP `EventCfg` omits `operation`, now required by `randomize_actuator_gains` | added `"operation": "abs"` (= the 2.1 default it was written against) | `leap_hand_env_cfg.py` (1 line) |

### Track A — 2-DOF reacher: environment built and verified

No 2-DOF task existed in any repo on this machine, so it was written from scratch against the
stage-1 report's constants (`source/pi_adr/tasks/two_dof_reach/`, asset `assets/two_dof_arm.urdf`).

`scripts/validate_env.py` — **10/10 checks passed**:

| Check | Result |
|---|---|
| A. 30 Hz control / 120 Hz physics / 300 steps per episode | `step_dt=0.033333 s`, 300 steps |
| B. simulated end-effector == analytic FK | max error **0.000 mm** |
| C. zero torque ⇒ no drift (planar, `g(q)=0`) | `|q|=0`, `|qd|=0` after 1 s |
| D. `a=+1` ⇒ +20 Nm on joint 1, positive rotation | applied **20.000 Nm** |
| E. joint velocity within ±5 rad/s under saturated torque | peak 5.101 rad/s |
| F. targets `r~U(0.20,0.85)` inside 1.0 m reach | r ∈ [0.202, 0.849] m |
| G. 8-D observation layout | `(N,8)`, all fields match |
| H. vanilla and Eureka rewards == report eqs. 7 and 8 | max deviation 0.00 |
| I. per-env DR actually reaches the simulator, differs per env | friction read-back std 0.054 |
| J. `tau = 20·a·actuator_effectiveness` | max deviation 0.0000 Nm |

Check I is the roadmap's "silent randomizer" failure mode (§25) — explicitly ruled out, DR is
read back from the simulator rather than assumed.

### Bug found and fixed in the evaluation harness

First eval of a partially-trained vanilla policy reported **0.39% success at 5 cm** while its
training return (~1208–1283) implied it was reaching and holding the target. Cause: `DirectRLEnv`
fires `time_out` at `episode_length_buf >= max_episode_length-1` and resets *within that same
step*, so a rollout of `horizon-1` steps records the **post-reset** pose as the terminal error —
i.e. a fresh random arm pose against a fresh random target, which is ~random-chance success.
Fixed by rolling out `horizon-2` steps (terminal sample at t=9.93 s of a 10 s episode).
Re-eval of the same checkpoint: **93.75% / 20.70% / 3.91%** at 5/2/1 cm — correct shape.

This is exactly the class of silent error that would have invalidated every downstream number,
so it is recorded here rather than quietly fixed.

### Track B — LEAP Hand

Stock upstream task `Isaac-Reorient-Cube-Leap` with **unmodified** config and hyperparameters
(GRU-256, `horizon_length 32`, `minibatch_size 32768`, `mini_epochs 5`, lr 5e-4 adaptive).
Smoke run: 1024 envs, 5 iterations, ~16k FPS, checkpoint written, return 33.09 — pipeline works.

### Run generation 1 (superseded — kept for the record)

First full set, trained before two environment defects were found. Numbers are **not** used as
baselines; they are what exposed the defects.

| Run | 5 cm | 2 cm | 1 cm | mean err | median | vel | effort |
|---|---|---|---|---|---|---|---|
| vanilla @5000 | 79.98% | 3.88% | 0.81% | 142.3 mm | 43.1 mm | 1.64 | 0.571 |
| vanilla @1000 | 87.11% | 23.14% | 5.96% | 92.6 mm | 33.4 mm | 2.22 | 0.773 |
| vanilla @200 | 92.43% | 14.75% | 2.29% | 55.0 mm | 32.8 mm | 3.07 | 1.290 |
| Eureka @500 | 95.56% | 26.15% | 7.74% | 50.3 mm | 32.2 mm | 3.09 | 1.372 |
| Eureka @1500 | 93.02% | 10.96% | 2.64% | 63.9 mm | 37.5 mm | 2.78 | 1.075 |

Two anomalies drove the investigation below: vanilla got **worse** with training (reward peaked
at epoch 200 = 1281 and fell monotonically to 1098 by epoch 5000), and every policy plateaued at
a **~32 mm median error** regardless of reward or training length.

### Defect 1 — joint damping was 0, not the documented 0.05

`ImplicitActuatorCfg(damping=0.0)` silently overrode the URDF's `<dynamics damping="0.05">`.
Read back from the simulator: `joint_damping = [0.0, 0.0]`.

This mattered beyond fidelity: DR samples damping as `nominal x scale`, so "DR at scale 1.0"
would have been 0.05 while the nominal baseline was 0.0 — every DR-vs-nominal comparison would
have carried a constant offset. Fixed to `damping=NOMINAL_JOINT_DAMPING`, verified by read-back.

### Defect 2 — the ±π joint limits cap the task at ~69% success

Diagnosed with a scripted IK + PD controller (`scripts/pd_baseline.py`), which measures what the
*environment* permits rather than what PPO learned:

| Controller | 5 cm | 1 cm | 1 mm | mean err | median | failures at a joint limit |
|---|---|---|---|---|---|---|
| IK+PD, ±π limits | 68.60% | 67.68% | 67.24% | 152.6 mm | 0.0 mm | **31.40% of 31.40% failures** |
| IK+PD, limits widened | **100%** | **100%** | **100%** | 0.0 mm | 0.0 mm | 0% |

Every single PD failure ended stalled against a joint stop. When a goal needs rotation past ±π
the shortest angular path is blocked, so the arm pushes into the limit and never arrives — a
wrap-around trap. It caps *any* controller near 69% at 5 cm, and it is the source of the heavy
tails in generation 1 (p90 = 600 mm, worst decile = 744 mm on vanilla @5000).

Two conclusions:
1. **The environment supports millimetre precision** (PD: 100% under 1 mm once unblocked), so the
   RL policies' 32 mm floor is a reward/learning result, not a physics limit. This is what makes
   the reward comparison meaningful.
2. The report's 99.49% at 5 cm is **unattainable under hard ±π limits**, so its limits cannot have
   been binding. Both joints are now `continuous`; observed angles are wrapped to [-π, π] to keep
   the 8-D observation bounded and identical in layout.

### Defect 3 (not fixed, recorded) — exploration collapse

From the run's TensorBoard scalars: `losses/entropy` falls 2.83 → −4.66, i.e. action σ collapses
from 1.0 to **≈0.024**, against the report's stated late-training σ of 0.10–0.11. Adaptive LR
drops to 1e-4 with KL held at ~0.01. This is consistent with the slow post-peak decay and is the
leading suspect for vanilla's failure to refine precision. Not yet addressed — it needs a
hyperparameter probe (entropy coefficient / σ floor), which is a separate experiment.

### Validation after the fixes

`validate_env.py` **10/10**, `validate_dynamics.py` **PASSED** (closed-form dynamics vs PhysX:
2.27% mean relative error on qdd, 0.036 Nm on recovered torque), `pd_baseline.py` **100%** at all
six precision thresholds. Three checks were themselves corrected during this work — D and J had
encoded an idealised torque law that ignored drive damping, Coulomb friction, and the fact that
per-joint actuator effectiveness stays coupled through `M(q)` (so q̇ is *not* componentwise
proportional to α). J now tests against the closed-form coupled dynamics.

### Run generation 2 (current baselines, corrected benchmark)

| Run | Task | Reward | Iterations | Envs | Seed |
|---|---|---|---|---|---|
| `A_vanilla_5000` | 2-DOF reach | vanilla (human) | 5000 | 4096 | 42 |
| `B_eureka_500` | 2-DOF reach | Eureka | 500 | 4096 | 42 |
| `B_eureka_1500` | 2-DOF reach | Eureka | 1500 | 4096 | 42 |
| `PD_reference` | 2-DOF reach | — (scripted IK+PD) | — | 2048 | 1234 |
| `LEAP_500` | Reorient-Cube-Leap | upstream, unmodified | 500 | 4096 | 42 |

Throughput note: vanilla and Eureka run concurrently on one L4 (~24% GPU for a single run, ~88%
for two). Iteration counts and all task metrics are unaffected, but **wall-clock per run is not
comparable across concurrent runs** and is not used as a headline metric.

Tooling note: `train_and_eval.sh` originally located the checkpoint via "most recent run
directory", which silently picked the *other* stream's folder when two trainings overlapped —
the first `A_vanilla_5000` eval failed this way. It now parses the run directory out of each
run's own log line (`Exact experiment name requested from command line:`).

---

## Phase 3 — physics observer (PINN)

`source/pi_adr/pinn/` — closed-form 2-link dynamics + a set-encoder ensemble that infers
ξ = [m1, m2, f1, f2, b1, b2, α1, α2] from a window of (q, q̇, τ, q̈) transitions.

Actuator effectiveness α is kept on the right-hand side of the EOM
(`M q̈ + C + F = α·τ_cmd`) and the observer is fed the *commanded* torque. Feeding it the
already-scaled torque would make α unidentifiable by construction.

### Defect 4 — the dataset labels did not satisfy the equations of motion

The first observer trained to a parameter error of 0.23 (23% of the parameter range) and an
acceleration RMSE of 23.6 rad/s². Checking the data rather than the model: evaluating the
closed-form dynamics at the **ground-truth** ξ gave an acceleration RMSE of **78.8 rad/s²** —
worse than the model's. The model was fitting label noise, not physics.

Two causes, both in data collection:

1. **q̈ differenced over a control step.** `env.step()` holds torque for 4 physics substeps, so
   `(q̇ₜ₊₁ − q̇ₜ)/0.033` is an *average* acceleration that does not satisfy the EOM at the
   recorded state. Fixed by stepping the simulator one physics step (1/120 s) at a time.
2. **The velocity limit engages from ~4 rad/s, not at 5.** Model error by speed band:

   | \|q̇\| | [0,1) | [1,2) | [2,3) | [3,4) | [4, 4.9) |
   |---|---|---|---|---|---|
   | median error (rad/s²) | 4.13 | 4.18 | 4.27 | 4.38 | **68.18** |

   The original mask (0.98 × 5 rad/s) let the whole saturated band through. Now masked at
   3.5 rad/s, with lower excitation torque to keep data in that regime.

After both fixes the ground-truth-ξ acceleration RMSE is **6.69 rad/s²** (5.3% of mean \|q̈\|).

### Observer results

| | acceleration RMSE | parameter error (normalised) | uncertainty↔error rank corr |
|---|---|---|---|
| interpolation (held-out physics, same ranges) | 2.79 rad/s² | **0.065** | +0.11 |
| extrapolation (ranges outside training) | 12.28 rad/s² | 0.209 | +0.10 |

Per-parameter absolute error, interpolation: m1 0.179 kg, m2 0.051 kg, f1 0.095, f2 0.114,
b1 0.042, b2 0.017, α1 0.064, α2 0.052.

Validation splits are by *physics configuration*, never by time samples from a configuration
already seen, so these are not memorisation numbers.

**Negative result worth flagging:** the deep ensemble's predictive variance is only weakly
related to its actual error (rank correlation +0.11 interpolation, +0.10 extrapolation). The
roadmap gives U_PINN(ξ) a term of its own in the adaptive-DR priority score
`w(ξ) = α_u U_PINN + α_f [1−S(π,ξ)] + α_s ‖∇_ξ J‖`, and on this evidence that term would be
close to noise. Before adaptive DR is allowed to depend on it, either the uncertainty needs to
be made calibrated or α_u should be set to zero and the claim dropped. This is exactly the
"negative result that explains why the PINN does not improve DR" the roadmap anticipates.

Extrapolation degrading (0.065 → 0.209) is the expected and honest outcome: the observer should
not be trusted to report physics outside the range it was trained on, which is an argument for
the roadmap's rule that adaptive DR stays inside C_RAPP.

---

## 2026-10-10 — Session 2: adaptive DR (rung F), no PINN

Gap since Session 1: none of the project state was lost — `pi_adr/` on disk is exactly as Session 1
left it (results, checkpoints, logs all intact; no process still running). One new fact about the
environment: this is a **shared machine** — `nvidia-smi` on arrival showed ~14 GB/23 GB already in
use by other users' unrelated workloads (web services, an image-generation backend, a camera
simulator), not reserved for this project. Session 2's experiments are scaled down from Session 1's
4096-env runs (to 2048 envs) specifically to be a reasonable citizen on shared infrastructure, not
because of any newly discovered compute limit of the task itself.

### Scope decision: build rung F, not rung G

The roadmap's experimental matrix (`2DOF_Physics_Informed_Adaptive_DR_Research_Roadmap`, Table 7)
defines:
- Rung D: Eureka + RAPP + static DR (already measured in Session 1 as `*_DRtrain`/`*_DRshift`, except
  those were DR applied **at evaluation time only** against a policy trained with no DR at all — not
  actually rung D as the roadmap defines it, which trains with DR active throughout)
- Rung F: Eureka + RAPP + **Adaptive** DR, no PINN
- Rung G: Eureka + RAPP + PINN-Adaptive DR (frozen observer feeds the controller)

Session 2 builds rung **F**, deliberately skipping the PINN-informed step to G, for a reason
established by Session 1's own evidence rather than by schedule pressure: `results/pinn_observer.json`
showed the ensemble's uncertainty only weakly tracks its actual parameter error (Spearman rank
correlation +0.114 interpolation, +0.097 extrapolation — both close to zero). The roadmap's proposed
priority score `w(xi) = a_u*U_PINN(xi) + a_f*(1-S(pi,xi)) + a_s*||grad_xi J||` would, on this evidence,
have its first term dominated by noise. Building G on an uncalibrated uncertainty signal would make
any D-vs-G comparison uninterpretable (a negative result could mean "adaptive DR doesn't help" or
"the PINN signal was noise" — no way to tell which). Rung F isolates the adaptive-DR question from
the PINN-calibration question, which is cleaner science even though it is a smaller claim than the
original roadmap's main contribution.

This is logged explicitly in `source/pi_adr/dr/adaptive.py`'s module docstring, not just here.

### Controller design (`source/pi_adr/dr/adaptive.py`)

Per-dimension (friction, damping_scale, effectiveness, mass_scale) independent control — carrying
over RAPP's own one-parameter-at-a-time discipline so a success drop is attributable to a single
cause:

```
S_d = P(d_T < 5cm) measured under the CURRENT training range for dimension d (others at nominal)
if S_d >= 0.90:  widen the moving bound 25% of the remaining gap toward its C_RAPP limit
else:            hold for up to 2 stalled outer iterations, then narrow 15% back toward the start
```

Each dimension widens in only one direction, fixed by Session 1's measured RAPP envelope (not
assumed): friction/damping/mass widen upward (harder = larger), effectiveness widens downward
(harder = weaker actuator) — matching the asymmetric mass finding and the effectiveness cliff below
0.3 from `rapp_eureka500_envelope.json`.

Starting range = the exact "in-distribution DR" range already used for Session 1's `*_DRtrain` eval
(friction[0,0.25], damping[0.5,4.0], effectiveness[0.5,1.0], mass[1.0,1.5]) — chosen so rung F starts
from a point this project has already measured once, rather than an arbitrary new one.

### Implementation finding: rl_games epoch semantics on `--checkpoint` resume

Needed for the orchestrator (`scripts/run_adaptive_dr.py`) to chain short training chunks with DR
ranges changing between chunks. Two things had to be verified against rl_games source, not assumed:

1. **Isaac Lab's hydra integration accepts nested dataclass overrides for free.** `DirectRLEnvCfg` is
   registered to Hydra's ConfigStore by `hydra_task_config` (`isaaclab_tasks/utils/hydra.py`), so
   `env.domain_rand.joint_friction_range=[0.0,0.25]` on the CLI reaches `TwoDofReachEnvCfg` with no
   new code in the environment — confirmed by inspecting the resulting run's dumped
   `params/env.yaml`, not just by the process exiting 0.
2. **`--checkpoint` resume is cumulative, not relative.** `a2c_common.py:639` restores
   `self.epoch_num = weights['epoch']` on load, and the training loop runs while
   `epoch_num < max_epochs` (an absolute target, not a budget). The orchestrator must therefore pass
   `--max_iterations = chunk_iters * (outer_step + 1)` on every chunk, not a fixed `chunk_iters` —
   the first version of `run_adaptive_dr.py` used a fixed per-chunk value and would have produced
   silent zero-iteration chunks starting from the second resume (epoch_num already >= max_epochs).
   Caught before the real run via a smoke test (2 outer steps, 3 iterations each, 64 envs) that
   confirmed checkpoints land at the cumulative epoch (`..._ep_3_...` then `..._ep_6_...`).

### Experiment D vs F — run configuration

Both trained Eureka-reward, seed 42, 2048 envs, 800 total PPO iterations (comparable budget to
Session 1's Eureka@500/@1500 pair), evaluated with the Session-1-frozen metric suite
(`METRICS_VERSION 1.0.0`) at the end.

- **D (static DR)**: one continuous `train.py` run, DR range fixed at the "in-distribution" range for
  all 800 iterations.
- **F (adaptive DR)**: `run_adaptive_dr.py`, 8 outer steps x 100-iteration chunks, controller
  observing `S_d` and updating ranges between chunks as above.

Run sequentially (not concurrently) to keep shared-GPU memory footprint low (~5-6 GB each at 2048
envs, versus ~9 GB free at the time of launch).

### Results: D (static DR) vs F (adaptive DR), 800 iterations, 2048 envs, seed 42

Evaluated with the frozen metric suite (`scripts/eval_DvF.sh`), 4096 eval envs, eval seed 1234,
three physics distributions each:

| Run | physics | <5cm | <2cm | <1cm | mean err | median err | settling |
|---|---|---|---|---|---|---|---|
| D | nominal | 97.78% | 32.50% | 8.76% | 30.5 mm | 26.0 mm | 1.17 s |
| D | in-distribution DR | 99.73% | 21.80% | 6.23% | 28.7 mm | 30.2 mm | 0.92 s |
| D | shifted (unseen) DR | 96.39% | 21.07% | 4.96% | 57.0 mm | 30.3 mm | 1.80 s |
| F | nominal | **72.88%** | 20.14% | 5.42% | 38.3 mm | 36.2 mm | **7.18 s** |
| F | in-distribution DR | 94.19% | **25.37%** | **8.15%** | 30.7 mm | 30.9 mm | 2.23 s |
| F | shifted (unseen) DR | 96.44% | 21.66% | 5.22% | 54.1 mm | 30.6 mm | 1.63 s |

### Controller trajectory (confirms the mechanism, not just the final ranges)

Full per-outer-step log, `results/adaptive_dr/F_adaptive_dr_controller.json`:

```
outer  friction S     damping S     effectiveness S   mass S      final ranges this step
0      0.848 hold     0.863 hold    0.978 widen        0.949 widen  eff:[.50,1]->[.45,1]  mass:[1,1.5]->[1,2.125]
1      0.893 hold     0.884 hold    0.957 widen        0.952 widen  eff->[.41,1]  mass->[1,2.59]
2      0.940 WIDEN    0.946 WIDEN   0.986 widen        0.982 widen  fric:[0,.25]->[0,1.19]  damp:[.5,4]->[.5,67]
3      0.932 widen    0.958 widen   0.960 widen         0.965 widen  fric->[0,1.89]  damp->[.5,114]  mass->[1,3.21]
4      0.893 hold     0.957 widen   0.978 widen         0.972 widen  damp->[.5,150]  mass->[1,3.41]
5      0.904 widen    0.967 widen   0.978 widen         0.979 widen  fric->[0,2.42]  damp->[.5,176]  mass->[1,3.56]
6      0.791 hold     0.960 widen   0.934 widen         0.953 widen  damp->[.5,196]  mass->[1,3.67]
7      0.748 hold     0.925 widen   0.908 widen         0.958 widen  damp->[.5,211]  mass->[1,3.75]
```

Final ranges reached by iteration 800: friction `[0, 2.42]` of C_RAPP `[0,4]` (60%), damping
`[0.5, 211]` of `[0.25,256]` (82%), effectiveness `[0.32, 1.0]` of `[0.3,1.0]` (97% — essentially
fully widened), mass `[1.0, 3.75]` of `[1,4]` (92%). The controller behaved exactly as designed:
mass and effectiveness — the two dimensions with the sharpest, most policy-relevant RAPP gradients
(§ RAPP section above) — widened fastest and furthest; friction repeatedly stalled once pushed past
~1.9–2.4 (success dipping to 0.79, 0.75 in the last two steps), correctly triggering hold-not-widen.

### Interpretation (single seed — read with Priority-1 caveat from `PROJECT_STATUS_AND_ROADMAP.md`)

1. **Shifted (hardest, unseen) distribution: F and D are statistically indistinguishable** (96.44%
   vs 96.39% @5cm; 21.66% vs 21.07% @2cm; 5.22% vs 4.96% @1cm; 54.1 vs 57.0 mm mean error). F is
   directionally better on every one of these six numbers, but the margins are small enough that a
   single seed cannot support a real-effect claim either way.
2. **In-distribution DR: F is more precise, D is more coarsely reliable.** F reaches more episodes
   under 2cm/1cm (25.4%/8.2% vs 21.8%/6.2%) but fewer under 5cm (94.2% vs 99.7%) — a genuinely mixed
   result, not a clean win for either rung.
3. **Nominal physics: F is clearly, substantially worse** (72.88% vs 97.78% @5cm; settling time
   7.18s vs 1.17s — a 6x difference, far outside plausible single-seed noise for this metric). This
   is the one result in this table large enough to trust without more seeds.

**Most likely mechanism for (3), directly supported by the trajectory table above:** the widening
rule (25% of the remaining C_RAPP gap per outer step, gated only by a single 90%-success threshold)
widened aggressively enough that by the end of the fixed 800-iteration budget, F's policy had spent
a shrinking fraction of total training time at or near any single narrow regime — including the
easy, zero-randomization nominal point it is evaluated against. This reads as a **widening-rate /
budget mismatch**, not evidence that adaptive DR itself cannot help: D's fixed range was *always*
the comparatively narrow "in-distribution" box for its full 800 iterations, while F's effective
training distribution was moving and, by the end, far broader than D's ever was — the two runs were
not actually training on comparably-sized distributions despite an identical iteration budget. This
is a controlled variable this experiment did not hold constant, and should be fixed before drawing
any conclusion about adaptive DR's value from this result alone (see `PROJECT_STATUS_AND_ROADMAP.md`
§5.1 follow-up: "tune curriculum rate" item).

**Bottom line for today:** rung F is built, runs correctly end-to-end, and the controller's
widening decisions are interpretable and consistent with the RAPP envelope's own asymmetries
(mass/effectiveness widen fastest, matching where RAPP found the real gradient). It does **not**,
in this single-seed, fixed-budget run, demonstrate a clear advantage over static DR — and surfaces a
concrete, fixable experimental-design gap (uncontrolled effective-distribution breadth) rather than
a dead end. This is reported as measured, not spun toward either "adaptive DR works" or "adaptive DR
doesn't work."

---

## 2026-10-10 continued — Priority-1 follow-up: F2 (slower widening, same day)

Same day, same session, in direct response to the D-vs-F result above. `run_adaptive_dr.py` gained
`--step_frac`/`--success_target`/`--narrow_frac`/`--patience` CLI flags (previously hardcoded).
F2 re-runs rung F identically except `step_frac: 0.25 -> 0.10`, same 800-iteration budget, same
seed, same starting range, same C_RAPP envelope.

**Final ranges reached, F1 vs F2** (both after 8 outer steps / 800 iterations):

| dim | F1 (step_frac=0.25) | F2 (step_frac=0.10) | C_RAPP |
|---|---|---|---|
| friction | [0, 2.42] | [0, 1.79] | [0, 4.0] |
| damping_scale | [0.5, 211] | [0.5, 122] | [0.25, 256] |
| effectiveness | [0.32, 1.0] | [0.39, 1.0] | [0.3, 1.0] |
| mass_scale | [1.0, 3.75] | [1.0, 2.92] | [1.0, 4.0] |

F2's range is narrower on every dimension, as designed — confirms the knob does what it says.

### Results: F2 vs F1 vs D

| Run | physics | <5cm | <2cm | <1cm | mean err | settling |
|---|---|---|---|---|---|---|
| D | nominal | 97.78% | 32.50% | 8.76% | 30.5 mm | 1.17 s |
| F1 (step=0.25) | nominal | 72.88% | 20.14% | 5.42% | 38.3 mm | 7.18 s |
| **F2 (step=0.10)** | nominal | **90.94%** | 20.00% | 4.32% | 32.4 mm | **4.04 s** |
| D | in-distribution | 99.73% | **21.80%** | 6.23% | 28.7 mm | 0.92 s |
| F1 | in-distribution | 94.19% | **25.37%** | 8.15% | 30.7 mm | 2.23 s |
| F2 | in-distribution | 99.41% | **15.43%** | 5.00% | 32.4 mm | 1.06 s |
| D | shifted | 96.39% | 21.07% | 4.96% | 57.0 mm | 1.80 s |
| F1 | shifted | 96.44% | 21.66% | 5.22% | 54.1 mm | 1.63 s |
| F2 | shifted | **98.17%** | 14.55% | 4.83% | 44.8 mm | 1.61 s |

### Interpretation: the fix worked partially, and exposed a second, more interesting problem

**The budget-mismatch hypothesis is confirmed directionally.** Nominal 5cm success:
D 97.78% > F2 90.94% > F1 72.88%. Slower widening recovers most (not all) of the nominal-physics
gap, exactly as predicted — a narrower, slower-growing training distribution leaves more effective
budget near the easy/nominal point.

**But it is not a clean fix, and the full picture is more interesting than "slower is better":**
F2 is the *worst* of all three runs at 2cm precision in **every** physics condition (20.00% nominal,
15.43% in-distribution, 14.55% shifted — each the lowest number in its row). F2 wins outright only
on 5cm coarse success under the shifted (hardest) distribution (98.17%, best of the three).

**A mechanistic explanation that connects back to this project's own central thesis.** The
controller's widening gate is `S_d = P(d_T < 5cm)` — exactly the coarse threshold this project's
Session-1 work showed can stay high while fine precision collapses underneath it (the original
Eureka-vs-vanilla finding). A *slower* widening schedule does not avoid that trap; if anything it
makes the controller spend *more* cumulative training iterations inside whatever physics band
currently clears 5cm but has already degraded at 2cm/1cm (the mid-widening history for F2 shows
coarse success holding at 0.90–0.99 across almost every outer step while the controller kept
advancing), rather than moving through that band quickly the way F1's faster schedule did. In other
words: **gating the curriculum on coarse success imports the same coarse/fine mismatch that drove
the original reward-comparison finding, directly into the domain-randomization controller.** This
was not anticipated when `AdaptiveDRController` was designed and is a genuinely new, controller-level
instance of the project's recurring theme (an objective that looks satisfied at a coarse threshold
can already be failing at the precision that actually matters).

**Concrete next experiment this motivates** (add to `PROJECT_STATUS_AND_ROADMAP.md` §5.1 as a
follow-up to the follow-up): gate the controller's widen/hold decision on a precision-aware signal
(e.g. `P(d_T < 2cm)` instead of `P(d_T < 5cm)`, or a weighted combination) rather than coarse success
alone, and re-run. If 2cm precision recovers across all three physics conditions without
re-introducing the nominal-performance collapse, that would be the first genuinely clean win for
adaptive DR in this project to date. Still single-seed; still needs §5.2 before any claim is final.

**Status: first attempt (F3) mis-specified, caught after 3 outer steps, corrected as F4.**
`--gate_threshold` wired through `run_adaptive_dr.py` -> `eval_dr_dims.py` cleanly (the latter
already had a `--success_eps` flag). F3 (`gate_threshold=0.02`, `success_target=0.90` — left at its
default) was launched and, after 3 outer steps, showed `S_d` stuck at 0.28–0.37 for every dimension
with the range frozen at its starting values the entire time (`EXPERIMENT_LOG.md` would have shown
8 identical "hold" rows). **Cause: `success_target=0.90` carried over from the coarse-threshold
runs, but no policy in this project has ever reached anywhere near 90% success at the 2cm
threshold** (every measured 2cm success rate across this project's runs is 12–42%, see the D/F/F1/F2
results above) — the controller could structurally never widen. Killed after noticing (no wasted
training beyond the 3 diagnostic outer steps), and relaunched as **F4** with `success_target: 0.90 ->
0.30`, matched to the actual observed 2cm-precision range. This is itself a small, worth-recording
finding: a curriculum's success threshold must be re-calibrated to whatever precision epsilon it is
gating on, not reused from a different epsilon's typical success rate — an easy mistake to repeat if
this project extends the gate-threshold idea further (e.g. to 1cm, where success rates are lower
still). F4 queued, results to follow in a further entry below once complete.

---

## 2026-10-10 continued — Priority 3: diagnosing the vanilla-reward sigma collapse

Two independent, cheap variants, both vanilla reward, seed 42, 2048 envs, 1500 iterations (smaller
and shorter than Session 1's `A_vanilla_5000` run on purpose — this is a diagnostic probe, not a
replacement headline number).

`source/pi_adr/tasks/two_dof_reach/two_dof_reach_env_cfg.py` gained a `VanillaRewardCfg` block
(`distance_weight`, `success_bonus`, `action_penalty_coef`, defaulting to the exact report equation)
so the reward-reweighting variant (V2) could run as a hydra override rather than a new hardcoded
reward branch. `validate_env.py` re-run after this change: still 10/10 (the defaults reproduce the
original equation exactly, confirmed by check H).

### V1: raise entropy_coef (0.005 -> 0.02), reward unchanged

| | <5cm | <2cm | <1cm | mean err | median err | entropy @100% |
|---|---|---|---|---|---|---|
| Session-1 baseline (`A_vanilla_5000`, 5000 iters, 4096 envs, entropy_coef=0.005) | 93.04% | 13.28% | 2.22% | 101.9 mm | 34.7 mm | -4.66 (σ≈0.024) |
| **V1** (1500 iters, 2048 envs, entropy_coef=0.02) | **97.83%** | **17.36%** | **3.91%** | 30.6 mm | 30.6 mm | **-1.23 (σ≈0.29)** |

**Confirmed as a real, fixable cause.** V1 beats the Session-1 baseline on every single metric
(coarse success, 2cm precision, 1cm precision, mean error) despite **3.3x fewer training iterations
and half the parallel environments** — i.e. this is not a "more compute" effect, it is specifically
about entropy not collapsing. `exp(-1.23)=0.29` vs `exp(-4.66)=0.0094`: a ~30x difference in
retained action standard deviation at the end of training.

**Not a complete fix.** V1's 17.36%/3.91% @2cm/1cm is still far short of the Stage-1 report's
claimed 98.36%/79.74%. Raising entropy_coef demonstrably helps and is at least part of the
explanation for the non-reproduction — but on this evidence it cannot be the *whole* explanation,
since even with exploration preserved, precision is nowhere near the report's numbers. This keeps
Session 1's open finding open: either V2 (reward reweighting) accounts for more of the remaining
gap, or some other factor (hyperparameters not yet tried, more iterations with entropy_coef=0.02,
or a genuine discrepancy with the report's own unreproduced setup) is still missing.

### V2: weaken the action penalty (0.005 -> 0.001), entropy_coef unchanged

| | <5cm | <2cm | <1cm | mean err | entropy @100% |
|---|---|---|---|---|---|
| Session-1 baseline | 93.04% | 13.28% | 2.22% | 101.9 mm | -4.66 |
| **V2** (action_penalty_coef=0.001, entropy_coef unchanged) | 99.24% | **12.23%** | **1.51%** | 38.0 mm | -4.12 |

**Did not help — if anything, slightly worse** on both precision thresholds (within likely
single-seed noise, but certainly not an improvement). Entropy still collapsed almost as much
(-4.12 vs baseline's -4.66) since `entropy_coef` was untouched in this variant — exactly as
expected if the action-penalty weighting is not what drives the collapse.

### Conclusion: the isolation is clean

V1 (entropy_coef changed, reward unchanged) fixed the collapse and improved precision on every
metric. V2 (reward changed, entropy_coef unchanged) left the collapse intact and did not improve
precision. **The dominant, confirmed cause of Session 1's vanilla-reward precision gap is the PPO
entropy coefficient, not the reward's action-penalty weighting.** The reward equation itself
(`-d + 5*I(d<0.05) - 0.005||a||^2`, exactly as reported) is not obviously the problem; the optimizer
configuration around it was. This resolves the Session-1 open question (§3, item 1) down to one
confirmed cause, with the residual gap to the Stage-1 report's claimed numbers (V1 still only
reaches 17.36%/3.91% @2cm/1cm vs the report's 98.36%/79.74%) left as a smaller, separately-flagged
open question — possibly entropy_coef=0.02 combined with a full 5000-iteration budget closes more
of it, which is now a well-defined, cheap follow-up rather than a vague "collapse" diagnosis.

### Side investigation (no new training, ran while D/F trained): PINN covariance diagnosis

`scripts/diagnose_pinn_covariance.py` — tests the identifiability-degeneracy hypothesis from
`PROJECT_STATUS_AND_ROADMAP.md` §5.3 step 1 directly against the already-trained ensemble and
already-saved validation data (`data/pinn_observer.pt`, `data/pinn_val.pt`); no simulator, no new
training, runs in seconds.

**Confirmed.** Cross-member correlation of the 5-ensemble's predictions, averaged over validation
environments:

```
corr(m1, a1) = +0.535     corr(m2, a2) = +0.466     corr(a1, a2) = -0.568
```

all three well above the |r|>0.3 threshold set in advance. The dominant eigenvector of the full
(non-diagonal) covariance matrix is 92% aligned with `m1` and mixes in `a1`/`a2` (`+0.308`/`-0.210`)
— i.e. the ensemble's single largest axis of disagreement is a **joint** mass–effectiveness
direction, not independent per-parameter uncertainty. `sum(diagonal variance) == sum(eigenvalues)`
by trace invariance (both 0.004507), confirming this is genuinely about how the SAME total
uncertainty is distributed across directions, not a difference in total uncertainty magnitude.

This directly explains the Session-1 negative result (uncertainty↔error rank corr. +0.11): the
diagonal-only estimate used in `ObserverEnsemble.forward()` reports `m1` and `a1` as if their
uncertainties were independent contributions, when in fact ~72% of the ensemble's total
disagreement lives in one combined direction. A controller reading the diagonal sees two
moderately-uncertain, seemingly-independent numbers; the ensemble is actually saying one thing very
loudly ("I can't tell `m1` and `a1` apart from this data") and almost nothing else.

Full numeric output: `results/pinn_covariance_diagnosis.json`.

---

## 2026-10-10 continued — Priority 4 attempt: dual-torque-regime fix, tested and REFUTED

`collect_pinn_data.py --torque_scale` changed from a single float to `nargs="+"`: with 2+ values
given, each hold-cycle now samples a regime per-environment-per-cycle from the given set (so the
SAME environment, same `xi`, sees multiple torque amplitudes across its 192-step window — not two
separate environment populations at different amplitudes, which was the original, flawed plan before
reconsidering what "the same environment at two regimes" actually requires). Re-collected all three
splits (train/val/extrap) with `--torque_scale 0.15 0.45`, retrained the observer
(`train_pinn.py --epochs 2000`), re-ran `diagnose_pinn_covariance.py`.

### Result: the degeneracy got WORSE, not better

| | single regime (0.25) | dual regime (0.15, 0.45) |
|---|---|---|
| corr(m1, a1) | +0.535 | **+0.652** |
| corr(m2, a2) | +0.466 | **+0.861** |
| corr(a1, a2) | -0.568 | +0.636 (sign flipped) |
| corr(m1, m2) | -0.201 | +0.504 (new, large) |
| sum(diagonal variance) | 0.00451 | **0.01468** (3.3x larger) |
| dominant eigenvector | 92% m1 | 82% m1, more mixed with a1/a2/m2 |
| val acceleration RMSE | 2.79 | 2.93 (no improvement) |
| val uncertainty↔error rank corr | +0.114 | +0.255 (improved, but off a much noisier base) |

Not only did `corr(m1,a1)`/`corr(m2,a2)` fail to shrink, new large correlations appeared between
parameters that weren't notably entangled before (`m1`-`m2`, `f1`-everything). The ensemble's total
uncertainty roughly tripled. This is a clean refutation, not an ambiguous result — the fix was
implemented correctly (confirmed: usable-transition fractions and per-split physics ranges in the
collection logs look exactly as intended) and tested properly (same evaluation protocol, same
validation set construction), and it made the diagnosed problem measurably worse.

### Why — the physics reasoning behind the original fix was incomplete

The original fix reasoned "varying torque amplitude changes the relative size of the
Coriolis/friction terms versus `alpha*tau/M`, which should break the degeneracy." **This is wrong,
or at least insufficient, and the experiment is why it's worth stating precisely:** in
`forward_dynamics`, the inertial contribution to acceleration is `alpha*tau_cmd / M(q)` — for *any*
fixed `(alpha, M)` pair, this term scales **linearly** with `tau_cmd`. Doubling the torque doubles
the predicted acceleration from this term regardless of how `alpha` and `M` are individually
divided — the *ratio* `alpha/M` that actually determines the observed acceleration is completely
unaffected by which torque amplitude was used to probe it. Varying torque amplitude bin-to-bin
provides more data *points* along the same degenerate direction, not an independent *direction* of
information. If anything, mixing two different-magnitude regimes in one training window likely
made the acceleration-prediction task itself harder (wider dynamic range per window for a small
MLP to fit), which plausibly explains both the higher validation RMSE and the larger total ensemble
variance — consistent with what was measured, not an unexplained anomaly.

**What would actually break the degeneracy, reasoned from the equations of motion themselves:**
`alpha` and `M` are only distinguishable through terms that do NOT scale with `alpha*tau/M` in the
same way — i.e. the velocity-dependent terms, `C(q,q̇)` (Coriolis, depends on `M` only, via `m2`)
and `F(q̇)` (friction/damping, independent of both `M` and `alpha`). These terms are relatively
largest either at high velocity (large Coriolis) or, most cleanly, **at steady state**: if torque is
held constant long enough for the arm to stop accelerating (`q̈ -> 0`), `inverse_dynamics` collapses
to `alpha*tau = C(q,q̇_terminal) + F(q̇_terminal)` — mass drops out of the equation entirely at that
point, leaving a relationship that constrains friction/damping/effectiveness without mass
contaminating it. The current collection protocol (`--hold 6` steps ≈ 0.2 s,
`--resample 32` steps ≈ 1.07 s) never holds a constant torque anywhere near long enough to approach
a terminal velocity (the arm's time constant at these torques and inertias is longer than either
window) — it is a transient-regime-only protocol, and the mass/effectiveness degeneracy specifically
lives in the transient regime. **This, not torque amplitude, is the dimension that needs to vary.**

**Corrected next experiment** (supersedes the "two torque regimes" plan in
`PROJECT_STATUS_AND_ROADMAP.md` §5.4 — update that section to match): add a third excitation mode to
`collect_pinn_data.py` that holds one torque value for a much longer window (enough physics steps,
at the arm's actual inertia/friction time constant, to approach terminal velocity — needs a quick
analytic estimate from `dynamics.py`'s `M`/`F` rather than a guess) interleaved with the existing
short-hold transient excitation, so each environment's window contains both transient and
near-steady-state samples. Re-run the same diagnostic afterward as the acceptance test.

---

## 2026-10-10 continued — Priority 4, attempt 2: long-hold (steady-state) excitation

`scripts/estimate_settling_time.py` (new, pure analysis of `dynamics.py`'s closed-form `M(q;xi)`,
no simulator) sized the fix before implementing it: settling time constant `tau_c = M/b` ranges
from ~158 physics steps (high-damping end of the PINN's own training range, `damping_scale=4.0`)
to **~2521 steps / 21 seconds** at the low-damping end (`damping_scale=0.5`) — two orders of
magnitude longer than the existing 6–32 step hold windows, confirming the mechanism identified
above and ruling out full (90–95%) settling across the whole range as impractical in one dataset.

`collect_pinn_data.py` gained a `--long_hold_steps` phase appended after the existing transient
phase: one sustained torque per environment (not resampled), long enough to partially settle most
of the range. First attempt used one fixed torque amplitude for every environment
(`--long_hold_torque_scale`) — a smoke test (256 envs, 300 long-hold steps) found only **0.2%** of
the final 30 (most-settled) steps were still below the velocity mask, versus 59.3% for the
transient phase. **Cause, confirmed by direct computation, not assumed:** steady-state velocity
`v_ss = alpha*tau/b` for a fixed torque scales with `1/b`, so low-damping/high-effectiveness
environments reach a steady-state velocity that **exceeds this arm's own ±5 rad/s hardware limit**
well before any natural settling — "steady state" for them is actually "pinned against the velocity
constraint," a different and uninformative regime the mask correctly (if unhelpfully) excludes.

**Fix:** since this is a simulation dataset (not a protocol the deployed, frozen PINN must itself
replicate on real hardware), the hold torque is legitimately sized **per-environment from that
environment's own ground-truth `(b, alpha)`** (`env._joint_damping`, `env._actuator_effectiveness`,
both already available as privileged simulator state during collection) to target one common safe
steady-state velocity (`--long_hold_target_velocity`, default 2.0 rad/s) regardless of the
environment's own damping. Re-ran the same smoke test with this fix: **93.4% overall usable, 96.3%
in the final 30 (most-settled) steps** — the structural problem is resolved.

**Status: queued, running.** Full train/val/extrap re-collection with `--long_hold_steps 300
--long_hold_target_velocity 2.0` (alongside the existing dual-torque-regime transient phase, since
that part of the earlier fix, while it didn't help on its own, is harmless and still provides
transient-regime data the long-hold phase doesn't), followed by observer retrain and a third
covariance diagnosis. Results in a further entry below once complete — this is the acceptance test
for whether the degeneracy identified at the start of this session is actually fixable.

---

## 2026-10-10 continued — major correction: Track B (LEAP) already has native adaptive DR

While the GPU was occupied by the long multi-seed/PINN queue, re-read
`LEAP_Hand_Isaac_Lab/.../leap_hand_env_cfg.py` and `.../utils/adr.py` closely while scoping what
"port adaptive DR to LEAP" (`PROJECT_STATUS_AND_ROADMAP.md` §5.6, previously "Priority 6") would
actually require. **Finding: it doesn't need to be ported — the upstream repo already ships a
working, OpenAI-ADR-style automatic domain randomization system, and it was ACTIVE during every
LEAP run this project has done, including the `LEAP_500` baseline reported as a "stock,
rung-1" result.**

Mechanism (`LeapHandADR.increase_ranges()`, `reorientation_env.py` lines ~267–390):
`adr_criteria = (consecutive_successes / z_rotation_steps) / episode_length_seconds` (rotations per
second, consistent with the "1 full rotation every 6.67s" comment in the cfg); whenever
`adr_criteria >= cfg.min_rot_adr_coeff` (0.15), every DR parameter in `adr_cfg_dict` widens by one
linear increment toward its configured target (`num_increments=25` total), each increment applied
to live `EventTermCfg.params` via the event manager. `cfg.enable_adr = True` and
`starting_adr_increments = 0` are the upstream defaults — "0" is a STARTING POINT, not "disabled".

**Verified directly against `LEAP_500`'s own TensorBoard log** (not assumed):
`Episode/num_adr_increases` rose from 0 to **8** (of 25) over the course of training;
`Episode/adr_criteria` rose from 0.0002 to a peak of 0.315, crossing the 0.15 threshold repeatedly.
The reported 7.52 rotations/20s result was measured on a policy trained under an actively-widening
domain-randomization distribution, not a static or zero-DR one.

### Consequence: `LEAP_500` needs re-labeling, and the Track B research question changes

`results/LEAP_500.json` and `RESULTS_2026-09-16.md`'s Track B section should be understood as
**"LEAP Hand, native ADR active from increment 0, 500 iterations"**, not "rung 1 / no DR" — the
latter does not exist yet as a result in this project. Confirmed (via a config-dump check, not a
full run — see below) that `env.enable_adr=false` is a valid hydra override that actually reaches
`LeapHandEnvCfg` (same mechanism as every 2-DOF hydra override this session), so a true no-ADR
baseline is one `--num_envs` fix away from running (the verification attempt used `--num_envs 64`
for speed and hit the unrelated, already-known `batch_size % minibatch_size` assertion — LEAP's
agent yaml needs `num_envs` large enough relative to its `horizon_length=32`/`minibatch_size=32768`,
exactly the constraint documented for the 2-DOF task in Session 1; not a new bug).

**This also changes what this project's own contribution to Track B should be.** The original plan
(`PROJECT_STATUS_AND_ROADMAP.md` §5.6, now superseded) was to port this project's
`AdaptiveDRController` to LEAP from scratch. That is no longer the right framing: LEAP already has a
mature, published-style adaptive DR mechanism. The actually-open, non-redundant research questions
for Track B are:
1. **Does LEAP's native ADR benefit from a RAPP-style feasibility-derived starting envelope**,
   rather than the hand-set ranges currently hardcoded in `adr_cfg_dict`/`adr_custom_cfg_dict`? This
   is a direct, non-redundant application of this project's actual novel idea (RAPP-initialized
   adaptive DR) to the real platform, rather than reinventing ADR a second time.
2. **How does LEAP's linear-threshold-gated widening rule compare to this project's
   success-rate/precision-gated rule** (the F1→F2→F4 progression this session)? LEAP's rule
   increments by a fixed step whenever a single coarse performance threshold is cleared — structurally
   very close to F1/F2 (the versions *shown this session to have a precision blind spot*), not F4.
   Whether LEAP's task has an analogous coarse/fine distinction (e.g. successful-but-sloppy rotations)
   is an open empirical question, not yet checked.
3. A true no-ADR baseline (`enable_adr=false`) is needed before either comparison means anything —
   this is the one new run actually required, and it is cheap (one more LEAP training run at the
   already-used 4096–8192 env scale).

**Status: scoped, not yet executed** — GPU was occupied by the 2-DOF multi-seed/PINN queue for the
remainder of this session. This is the concrete, ready-to-run starting point for the next session's
Track B work, replacing the vaguer "port the mechanism" plan.

---

## 2026-10-10 continued — Priority 2: multi-seed D vs F4 — the F4 result does NOT replicate

Two additional seeds (123, 7) run for both D and F4, identical protocol to seed 42 (`run_D_seed.sh`,
`run_F4_seed.sh` — new, thin wrappers around the exact same training/eval commands already used for
seed 42, so all three seeds are directly comparable). n=3 per condition — small, but enough to show
whether the seed-42 gap was within normal variance.

### Full results, mean ± std (n=3), with the raw per-seed values shown because the spread matters more than the mean here

| | 5cm | 2cm | 1cm | mean err (mm) |
|---|---|---|---|---|
| **D, nominal** | 97.44 ± 0.47% | 28.12 ± 5.24% | 8.05 ± 3.02% | 37.1 ± 7.2 |
| **F4, nominal** | 93.59 ± 6.38% | 25.81 ± 1.49% | 6.83 ± 1.47% | 47.4 ± 17.9 |
| **D, in-distribution** | 98.89 ± 0.73% | 24.10 ± 9.26% | 7.50 ± 4.46% | 37.8 ± 9.4 |
| **F4, in-distribution** | 98.58 ± 1.08% | 23.79 ± 5.02% | 4.99 ± 1.65% | 40.0 ± 10.1 |
| **D, shifted** | 96.28 ± 2.24% | 21.98 ± 8.27% | 6.61 ± 3.94% | 58.2 ± 20.2 |
| **F4, shifted** | 96.88 ± 1.61% | 20.82 ± 2.95% | 3.90 ± 1.46% | 53.6 ± 13.1 |

Raw 2cm values per seed (42, 123, 7) — D in-distribution: `[21.80, 16.21, 34.30]`; F4
in-distribution: `[29.42, 19.80, 22.14]`. The seed-42 pair that drove this session's "F4 beats D"
claim (29.42 vs 21.80) sits well inside the spread both rungs show across their *own* three seeds
(D alone ranges 16.21–34.30 — a wider spread than the D-vs-F4 gap at any single seed).

### Honest conclusion: the F4 finding does not hold up

**D and F4 are statistically indistinguishable on every precision threshold in every physics
condition, at n=3.** Where seed 42 alone showed F4 ahead (in-distribution and shifted 2cm/1cm), the
multi-seed mean is a near-tie with highly overlapping distributions; at 1cm, D's mean is actually
*higher* than F4's in all three conditions (though again within the overlapping spread). F4's
nominal-physics numbers are also noisier (±6.38% on 5cm success, ±17.9mm on mean error) than D's,
driven by one weak seed (123: 86.5% @5cm) rather than a consistent pattern.

**This supersedes the "F4 is the cleanest adaptive-DR result this session" conclusion drawn from
seed 42 alone (previous entry).** That conclusion was premature — exactly the failure mode Priority
2 existed to catch, and it caught it. The honest summary of this session's full F1→F2→F4 progression
is: **three different adaptive-DR controller designs were tried on this benchmark (coarse-gated
fast widening, coarse-gated slow widening, precision-gated widening), and none has been shown, with
proper multi-seed evaluation, to outperform simple static domain randomization.** This is a genuine
negative result for the specific adaptive-DR mechanisms implemented so far — not for adaptive DR as
a concept, and not a reason to stop (see the revised plan below), but it must be the headline, not a
footnote, of today's adaptive-DR work.

**What this does NOT rule out:**
- A larger seed count (5–10) might reveal a smaller real effect this n=3 sample is underpowered to
  detect — the single-seed point estimates are not wildly inconsistent with a small true effect
  existing, just not large enough to separate from PPO's own well-documented seed-to-seed variance.
- The curriculum-signal insight from the F1→F2→F4 progression (gating on coarse 5cm success lets
  the controller widen through precision-degraded regimes) may still be correct as a *mechanism*,
  even though F4's net effect on the final metric washed out across seeds — the mechanism and the
  net benefit are separate claims, and only the second one failed to replicate here.
- Rung G (PINN-informed) and the RAPP-vs-no-RAPP question (rung E, running as this is written) are
  untested by this result and still open.

**Revised priority ordering going forward:** given this result, sinking more compute into tuning
this specific controller's hyperparameters further (a 5th curriculum variant) is a weaker bet than
either (a) a proper n=5+ seed count on the current best candidate (F4) before concluding anything
further, or (b) checking whether rung E (no RAPP) or rung G (PINN-informed, once the PINN is fixed)
behave differently enough to be worth their own multi-seed check. Update
`PROJECT_STATUS_AND_ROADMAP.md` §5.1/§5.2 to reflect this.

---

## 2026-10-10 continued — Rung E (adaptive DR, no RAPP): the largest effect this session, and a clear trade-off

`scripts/run_E_adaptive_no_rapp.sh` — identical controller/gate/step-rate to F4
(`gate_threshold=0.02`, `success_target=0.30`, `step_frac=0.10`, 800 iterations), with `--no_rapp`
swapping RAPP's measured `C_RAPP` envelope for arbitrary wide bounds (friction [0,50], damping_scale
[0.1,1000], effectiveness [0.01,2], mass_scale [0.1,20]) — isolating whether RAPP's constraint
itself contributes anything, separate from the adaptive mechanism. Single seed (42) so far.

**Final ranges reached (vs F4's RAPP-bounded ones):** `mass_scale` widened to **`[1.0, 8.22]`** —
more than double F4's `[1.0, 1.61]` and over 2x past C_RAPP's own measured upper bound of 4.0.
`effectiveness` reached `[0.27, 1.0]`, past C_RAPP's lower bound of 0.3. Without a ceiling, the
controller pushed well outside the region RAPP determined the nominal policy could handle.

### Results

| Run | physics | <5cm | <2cm | <1cm | mean err | median err |
|---|---|---|---|---|---|---|
| D | nominal | 97.78% | 32.50% | 8.76% | 30.5 mm | 26.0 mm |
| F4 | nominal | 95.39% | 25.59% | 5.35% | 62.5 mm | 27.2 mm |
| **E (no RAPP)** | nominal | **65.11%** | 12.82% | 2.83% | 43.9 mm | 43.0 mm |
| D | in-distribution | 99.73% | 21.80% | 6.23% | 28.7 mm | 30.2 mm |
| F4 | in-distribution | 98.12% | 29.42% | 5.59% | 46.1 mm | 28.0 mm |
| **E** | in-distribution | 97.97% | **43.09%** | **13.31%** | **24.5 mm** | **22.2 mm** |
| D | shifted | 96.39% | 21.07% | 4.96% | 57.0 mm | 30.3 mm |
| F4 | shifted | 98.73% | 24.22% | 4.17% | 38.4 mm | 29.7 mm |
| **E** | shifted | **99.44%** | **33.67%** | **9.96%** | 29.2 mm | 26.6 mm |

### Interpretation: a genuine, large, and informative trade-off — not an unqualified win

**At nominal physics, E is dramatically worse than everything else tried this session** (65.11%
@5cm vs 95–98% for D/F4 — a collapse, not a small regression; settling time 5.89s vs ~1–2s
elsewhere). **At both DR conditions, E is the best result in the entire session by a wide margin** —
43.09%/13.31% @2cm/1cm in-distribution is nearly double D's 21.80%/6.23%, and the effect size here
(roughly 2x, on metrics in the tens-of-percent range) is far larger than the D-vs-F4 gaps that
turned out to be noise at n=3 — this is not guaranteed to replicate, but it is a qualitatively
different scale of effect, not a marginal one.

**Why this makes mechanistic sense, and what it says about RAPP's role:** without a RAPP-derived
ceiling, the controller kept widening mass toward 8x nominal as long as 2cm success stayed near
0.30, training the policy on a much harder, much wider physics distribution for most of its budget.
That produces a policy well-adapted to hard, varied physics (hence the DR-condition wins) at the
direct expense of the easy, zero-randomization case it spent comparatively little time training on
(hence the nominal collapse). **This is exactly the trade-off RAPP exists to prevent** — its
feasibility-derived envelope is supposed to keep the training distribution inside a region where the
nominal/easy case is not sacrificed. On this single data point, **RAPP's constraint cost most of
the DR-condition precision gain in exchange for preserving nominal performance** — a real,
interpretable, and non-obvious finding about what RAPP's envelope is actually doing to the
D-vs-F4-vs-E comparison, regardless of how the F4-vs-D question itself resolves with more seeds.

**Important asymmetry with the D-vs-F4 result:** D-vs-F4 differences were small relative to
cross-seed variance and did not replicate. E's differences from both D and F4 are large enough,
on metrics with comparable variance (the D/F4 multi-seed stds were mostly 1–10 percentage points;
E's gaps from D/F4 are 10–20+ percentage points at the DR conditions and >30 points at nominal) that
single-seed noise is a much less plausible explanation here — though this is a judgment call, not a
statistical test, until E itself is multi-seeded.

**Next step this directly motivates:** multi-seed rung E (currently single-seed) before treating
either side of this trade-off as established — if it holds, this would be the clearest, most
actionable finding of the entire project: *adaptive DR without a feasibility constraint trades
nominal performance for substantially better precision under physics uncertainty, and RAPP's role
is specifically to control how much of that trade a user is exposed to* — a genuinely interesting
research story, and one worth testing properly before the session ends rather than leaving as a
single data point.

---

## 2026-10-10 continued — long-hold PINN result: partial success, joint 1 fixed, joint 2 not

Full retrain (`train_pinn.py --epochs 2000`) on the corrected long-hold dataset (per-environment
calibrated steady-state excitation, §"F4" neighbor entry above), then
`diagnose_pinn_covariance.py` re-run.

### Accuracy and uncertainty calibration — clearly improved

| | single-regime (orig.) | dual-regime (failed) | **long-hold (this fix)** |
|---|---|---|---|
| interpolation acc. RMSE | 2.79 | 2.93 | **2.08** |
| interpolation param. error | 0.065 | 0.071 | 0.064 |
| interpolation uncertainty↔error rank corr | +0.114 | +0.255 | **+0.293** |
| extrapolation acc. RMSE | 12.28 | 12.98 | **5.93** |
| extrapolation param. error | 0.209 | 0.223 | **0.161** |
| extrapolation uncertainty↔error rank corr | +0.097 | +0.000 | **-0.150** |

Every accuracy number improved, several substantially (extrapolation RMSE more than halved). The
interpolation uncertainty calibration is the best of all three attempts. **Extrapolation
uncertainty calibration got worse** (now anti-correlated) — with only 1024 extrapolation
environments this specific number is the least statistically reliable one in this table, but it is
reported as measured, not dismissed.

### The degeneracy itself: joint 1 fixed, joint 2 not

| pair | single-regime | dual-regime (failed) | **long-hold** |
|---|---|---|---|
| corr(m1, a1) | +0.535 | +0.652 | **-0.175** (fixed) |
| corr(m2, a2) | +0.466 | +0.861 | **+0.731** (not fixed, still large) |
| corr(a1, a2) | -0.568 | +0.636 | +0.018 (fixed) |
| corr(m1, m2) | -0.201 | +0.504 | **-0.576** (new, large) |

Joint 1's mass/effectiveness degeneracy is resolved by the steady-state excitation, as hypothesized.
**Joint 2's is not**, and a new large `m1`-`m2` correlation appeared that wasn't present originally.

**Working hypothesis for why joint 2 differs (not yet confirmed):** the per-environment hold torque
was calibrated independently per joint, each targeting 2 rad/s from that joint's own `(b, alpha)` in
isolation — but the arm is a genuinely *coupled* 2-link system (`dynamics.py`'s `coriolis_vector`:
joint 2's acceleration depends on joint 1's velocity via the `-h*qd1^2` term, and both rows mix
through the full `M^{-1}` matrix inversion, not just through the diagonal). Driving both joints to
independently-calibrated steady velocities simultaneously does not give joint 2 a clean,
decoupled steady-state reading the way the single-joint analysis in
`scripts/estimate_settling_time.py` assumed — joint 1's motion contaminates what "steady state"
looks like for joint 2. This would also explain the new `m1`-`m2` correlation (both now entangled
through the same coupled-motion artifact) and is consistent with `m1` (the more heavily-weighted,
proximal link in the coupled dynamics) being the one that successfully decoupled from its own
actuator parameter while `m2` did not.

**Not pursued further today** (this is the third full iteration on this specific sub-problem this
session — dual-regime, fixed-torque long-hold, calibrated-torque long-hold — and genuine, measurable
progress was made each time; a fourth cycle is better scoped as deliberate follow-up work, not
squeezed into the remainder of this session). **Concrete next step, reasoned from the diagnosis
above:** hold joint 1 fixed (zero commanded torque, or lock it) while only joint 2 is driven to its
calibrated steady state, and vice versa — a genuinely decoupled single-joint probe, rather than
both joints held simultaneously. This directly tests the coupling hypothesis and should be a
small change to the existing long-hold phase (sequential rather than simultaneous per-joint holds).

**Net assessment for the roadmap:** this is real, measured progress (half the degeneracy resolved,
accuracy meaningfully improved, especially out-of-distribution) rather than a clean full fix. Rung G
should not yet be built on this observer — the interpolation uncertainty signal (+0.293) is
promising but the extrapolation one going negative is a real caution, and joint 2's degeneracy is
unresolved. The sequential single-joint hold is the next, well-motivated experiment before
attempting rung G again.

---

## 2026-10-10 continued — F4: precision-gated adaptive DR, recalibrated, the cleanest result today

F3's `success_target=0.90` was left over from the coarse-threshold (F1/F2) runs and never
recalibrated for a 2cm gate — no policy in this project reaches anywhere near 90% at 2cm, so F3
froze at its starting range for all 3 outer steps observed before being killed. Relaunched as **F4**
with `success_target: 0.90 -> 0.30` (matched to the actual observed range of 2cm success rates
across this project, 12–42%), same `gate_threshold=0.02`, `step_frac=0.10`, 800-iteration budget.

F4's trajectory oscillates rather than monotonically widening (unlike F1/F2): success hovered
around 0.26–0.37 against the 0.30 target, triggering alternating widen/hold/narrow decisions.
Final ranges are far more conservative than F1/F2's: `mass_scale` reached only `[1.0, 1.61]` (vs
F1's `[1.0, 3.75]`, F2's `[1.0, 2.92]`) — the precision-aware gate kept the controller close to its
starting point rather than pushing toward C_RAPP's bounds, which makes sense: precision degrades
well before coarse success does (RAPP's own finding, Session 1), so a controller honestly gated on
precision has much less room to widen into before it stops clearing its own bar.

### Full four-way comparison

| Run | physics | <5cm | <2cm | <1cm | mean err | median err |
|---|---|---|---|---|---|---|
| D (static) | nominal | 97.78% | 32.50% | 8.76% | 30.5 mm | 26.0 mm |
| F1 (coarse gate, fast widen) | nominal | 72.88% | 20.14% | 5.42% | 38.3 mm | 36.2 mm |
| F2 (coarse gate, slow widen) | nominal | 90.94% | 20.00% | 4.32% | 32.4 mm | 32.0 mm |
| **F4 (precision gate)** | nominal | 95.39% | **25.59%** | 5.35% | 62.5 mm | **27.2 mm** |
| D | in-distribution | 99.73% | 21.80% | 6.23% | 28.7 mm | 30.2 mm |
| F1 | in-distribution | 94.19% | 25.37% | 8.15% | 30.7 mm | 30.9 mm |
| F2 | in-distribution | 99.41% | 15.43% | 5.00% | 32.4 mm | 32.9 mm |
| **F4** | in-distribution | 98.12% | **29.42%** | 5.59% | 46.1 mm | 28.0 mm |
| D | shifted | 96.39% | 21.07% | 4.96% | 57.0 mm | 30.3 mm |
| F1 | shifted | 96.44% | 21.66% | 5.22% | 54.1 mm | 30.6 mm |
| F2 | shifted | 98.17% | 14.55% | 4.83% | 44.8 mm | 33.9 mm |
| **F4** | shifted | **98.73%** | **24.22%** | 4.17% | **38.4 mm** | 29.7 mm |

### Interpretation — the cleanest adaptive-DR result this session, with one clearly-diagnosed weak point

**F4 beats both F1 and F2 on 2cm precision in every single physics condition tested.** This is
exactly what gating on a precision-aware signal (rather than coarse 5cm success) was designed to
fix, and the trajectory mechanism matches: by not being allowed to widen past the point where 2cm
precision itself degrades, F4 never entered the wide, precision-poor regimes F1 and F2 spent much of
their training budget in.

**F4 is competitive with or better than static D in 2 of 3 conditions** — in-distribution 2cm
(29.42% vs D's 21.80%) and shifted 5cm/2cm/mean-error (all three better than D). The one condition
where D still clearly wins is nominal 5cm/mean-error (97.78%/30.5mm vs F4's 95.39%/62.5mm) — but
note F4's *median* nominal error (27.2mm) is essentially tied with D's (26.0mm); the mean-error gap
is driven by `worst_decile_error_m` (not tabulated above, see `results/F4_nominal.json`: 400.5mm —
a heavy-tailed minority of bad episodes, not a worse typical case). This is a specific, diagnosable
weak point (occasional large failures at exactly-zero randomization) rather than a general
regression, and a natural candidate for the first thing to check with per-episode failure analysis
in a follow-up, not a reason to discount the 2cm-precision result above.

**Still single-seed.** This is the most encouraging adaptive-DR result produced in this project to
date, and it is still exactly one data point per condition. Priority 2 (multi-seed) is more
important now than it was this morning, specifically to confirm the 2cm-precision advantage is real
before it is reported as a finding anywhere outside this log.

**UPDATE, same day, after multi-seeding (see entry below): this result did NOT replicate.** Read
the next entry before citing anything above this line.

### Deviations from the stage-1 report

- The report's documented results (99.49% / 98.36% / 79.74%) have **no corresponding code, logs
  or checkpoints anywhere on this machine** — this session rebuilt the benchmark from the
  report's equations, so our numbers are an independent reproduction, not a continuation.
- Physics engine settings (solver iteration counts, PhysX defaults) were not specified in the
  report and are Isaac Lab 2.3 defaults here.
- The report says PhysX integrates a `g(q)` gravity term; this build places the arm in the
  horizontal XY plane with joints about Z, so gravity produces no joint torque (verified, check C).

---

## 2026-10-11 — multi-seed results land: reward comparison and rung E (n=3 each)

All runs at the matched 800-iteration / 2048-env protocol, seeds 42/123/7, evaluation seed 1234,
`METRICS_VERSION 1.0.0`. Mean ± std; raw per-seed 2cm values shown because the spread is the story.

### 1. Eureka vs vanilla, matched budget, nominal physics — Session 1's headline does NOT hold

| reward | 5cm | 2cm | 1cm | mean err | raw 2cm (42,123,7) |
|---|---|---|---|---|---|
| vanilla | 98.70 ± 0.16 | **29.26 ± 1.30** | 8.02 ± 1.26 | 36.5 ± 3.9 mm | 27.9, 29.4, 30.5 |
| Eureka | 99.02 ± 0.36 | 25.59 ± 7.37 | 6.81 ± 1.79 | 38.1 ± 8.9 mm | 17.3, 28.0, 31.4 |

At equal training budget the two rewards are indistinguishable; vanilla is nominally ahead on 2cm/1cm
and far *more stable* across seeds (std 1.3 vs 7.4). Session 1's "Eureka beats the human reward at every
threshold" came from a single seed compared at mismatched budgets (vanilla 5000 iters with collapsed
exploration vs Eureka 1500). With exploration not collapsed and budgets matched, the gap disappears.
**Retract the Session-1 claim that Eureka outperforms the human reward on this benchmark.** What
survives: the Stage-1 *mechanism* (Eureka's +100 bonus makes it coarse-success-dominated) is
untested here by this comparison, not refuted; and neither reward gets near the IK+PD ceiling (100%).
Caveat: 800 iterations is shorter than Session 1's budgets, so this says nothing about 5000-iter behaviour.

### 2. Rung E (adaptive DR, no RAPP), n=3 — nominal cost is real, DR benefit is directional only

| | physics | 5cm | 2cm | 1cm | mean err |
|---|---|---|---|---|---|
| D | nominal | 97.44 ± 0.47 | 28.12 ± 5.24 | 8.05 ± 3.02 | 37.1 ± 7.2 |
| E | nominal | **82.06 ± 14.86** | 19.82 ± 6.67 | 4.87 ± 1.97 | 36.9 ± 6.2 |
| D | in-distribution | 98.89 ± 0.73 | 24.10 ± 9.26 | 7.50 ± 4.46 | 37.8 ± 9.4 |
| E | in-distribution | 98.65 ± 0.59 | 38.21 ± 15.98 | 10.91 ± 5.94 | **27.1 ± 2.4** |
| D | shifted | 96.28 ± 2.24 | 21.98 ± 8.27 | 6.61 ± 3.94 | 58.2 ± 20.2 |
| E | shifted | 98.80 ± 1.49 | 32.12 ± 17.17 | 8.29 ± 5.41 | **32.8 ± 3.9** |

Raw 2cm in-distribution, E: 43.1 / 20.4 / 51.2; D: 21.8 / 16.2 / 34.3. The seed-42 value (43.1) that
looked like a 2x win was the *second-best* of E's three seeds; seed 123 (20.4) is at D's level.

What holds up and what does not:
- **Nominal-physics cost: robust.** E's nominal 5cm success is below D's *minimum* seed in 3/3 seeds
  (65.1, 88.2, 92.9 vs D 96.9–97.8), settling time 3–6 s vs ~1–2 s. Training without a feasibility
  ceiling does sacrifice the easy case.
- **2cm/1cm precision gain under DR: NOT established.** Means are higher (38 vs 24, 32 vs 22 @2cm) but
  E's own std is 16–17 points; a rough Welch comparison gives t ≈ 1.3 at n=3. Directional, not significant.
- **Mean (tail) error under DR: the most consistent effect.** E's mean error is lower than D's in 3/3
  seeds in both DR conditions, and much tighter (27.1 ± 2.4 vs 37.8 ± 9.4 in-distribution; 32.8 ± 3.9
  vs 58.2 ± 20.2 shifted). Rough t ≈ 1.9–2.1 — suggestive, still n=3. So E's benefit looks like
  *fewer bad episodes* rather than *better typical precision*.

This **replaces** the "largest effect this session / clean trade-off" framing written on 10 Oct: the
trade-off is real in one direction (nominal cost) and only suggestive in the other. The mechanism story
(no ceiling → mass to ~8x → harder training distribution) stands as an explanation of the nominal
cost; it is not yet established as an explanation of a robustness gain.

### Bottom line after all multi-seeding
Of this session's three apparent adaptive-DR findings (F4 beats D; E beats D under DR; Eureka beats
vanilla), **none survives n=3 as a clear effect.** What does survive: (a) the PINN diagnosis chain,
(b) the entropy-coefficient fix for vanilla exploration collapse (V1 was single-seed — also still
needs multi-seeding), (c) the nominal-performance cost of an unconstrained curriculum, (d) the
tail-error reduction under DR for E (suggestive), (e) the environment/eval defect fixes, (f) LEAP
already having native ADR. Seed-to-seed std of 5–17 points on 2cm success is itself a finding:
**any future comparison here needs n ≥ 5 and should lead with tail error / worst-decile, not mean
2cm success.**

---
## 2026-10-11 — Sequential-hold PINN observer, rung G machinery, A–G fill launched

**Sequential single-joint hold observer (tag `seq`)** — data collected with `--long_hold_mode sequential` (first half of the hold drives joint 1 only, second half joint 2 only; per-env calibrated torque, target 2 rad/s), 2000 epochs. Artifacts: `data/pinn_observer_seq.pt`, `results/pinn_observer_seq.json`, `results/pinn_covariance_diagnosis_seq.json`, logs `logs/train_pinn_seq.log`, `logs/diagnose_pinn_covariance_seq.log`.

| | long-hold (simultaneous) | sequential (`seq`) |
|---|---|---|
| interp acc RMSE (rad/s²) | 2.08 | 1.95 |
| extrap acc RMSE | 5.93 | 7.73 (worse) |
| interp param err (norm) | — | 0.0606 |
| corr(m2,a2) across members | +0.731 | +0.561 |
| corr(m1,a1) | −0.175 | +0.078 |
| corr(m1,m2) | −0.576 | −0.361 |
| uncertainty↔error rank corr interp / extrap | — | +0.267 / **−0.155** |

Reading: sequential excitation reduced the m↔α and m1↔m2 degeneracies and improved interpolation accuracy and uncertainty ranking (+0.27 vs the original +0.11), but it did **not** fix extrapolation (worse, and uncertainty is anti-correlated with error there). The m1 error remains the largest (0.19 interp). Decision: use `seq` for rung G, with the caveat that observer uncertainty outside the training envelope is unreliable; the shuffled-multiplier placebo is the control for exactly this.

**Rung G design** (`scripts/pinn_probe.py`, `AdaptiveDRController.update(step_multiplier=)`, `run_adaptive_dr.py --pinn_observer/--pinn_mode/--pinn_strength`, `scripts/run_G_seed.sh`): same as F4 (step_frac 0.10, 2 cm gate, success_target 0.30) except each dimension's widening step is multiplied by clip(1 + strength·(u_d/mean u − 1), 0.5, 1.5) where u_d = mean epistemic variance (normalised by XI_BOUNDS width²) of the frozen observer's estimates of that dimension's parameters, probed with only that dimension randomised over its current range. Higher observer uncertainty ⇒ larger step (explore where the model knows least). Placebo (`shuffle`) uses the same multiplier values permuted across dimensions. Probe smoke test (initial ranges): epistemic var friction 2.0e-4, damping 5.3e-4, effectiveness 5.9e-4, mass 0.9e-4. Probe bug fixed: env.reset() must be inside `torch.inference_mode()` after the first rollout.

**Launched (two lanes max):** chain10 = G seeds 42,123,7,1,2 then Eureka-no-DR (rung B, 800-iter) seeds 1,2; chain11 (waits for chain8 "ALL DONE") = C seeds 123,7,1,2; F4 seeds 1,2; DRtrain/DRshift evals for existing 800-iter A and B checkpoints; G-placebo seeds 42,123,7. Logs: `logs/chain10_G_B.log`, `logs/chain11_fill.log`. Note: the C wrapper now suffixes run names for seed≠42 (`C_seed{S}_human_static_dr`).

**Not yet claimed:** nothing about G is known. The matrix compile (`analyze` over A–G, n=5) is pending these runs.

---
## 2026-10-11 (afternoon) — A–G matrix complete at n=5 (G placebo n=3)

All chains finished (chain8, chain10, chain11). Full tables with exact permutation tests: `results/MATRIX_A_to_G.md` (generated by `scripts/analyze_matrix.py`; 800-iter protocol, metrics v1.0.0, seeds 42/123/7/1/2). The G placebo (shuffled multipliers) was run for seeds 42/123/7 only.

**Findings (honest reading)**
- A, B, C are statistically indistinguishable on mean error and tail in all three evals (p≥0.18). Eureka-vs-human and static-DR-vs-none show no reliable effect on those metrics.
- B vs D: static RAPP DR lowers 2 cm success under DRtrain (40.8→21.4%, p=0.016) and DRshift (43.2→19.8%, p=0.016). E vs D: E loses nominal 5 cm success (80.4 vs 97.5%, p=0.008); its mean-error/tail advantage is not significant (p≥0.15). The earlier "E beats D" claim is retracted.
- F4 does not beat D on any primary metric (p≥0.2).
- G has the lowest mean error/worst decile and lowest across-seed spread in all evals (nominal 34.1 mm/82 mm; DRtrain 33.6/69; DRshift 45.4/174), but G−F4 is not significant (p 0.12–0.29), G−D reaches p=0.063 only for DRshift worst decile (post-hoc, ~30 comparisons), and G has the lowest 2 cm/1 cm success of the matrix.
- **Placebo:** shuffled multipliers (n=3) match G on nominal (32.4/72.6 mm) and DRtrain (32.0/56.5); only DRshift differs (62.2/335 vs 45.4/174), with n=3 and large placebo variance. Two of three placebo seeds also differ from G in final ranges (seed 7 placebo left damping at 4.0). Conclusion: no evidence that PINN-derived information, rather than the changed widening schedule, drives G's behaviour. Observer uncertainty was dominated by damping scale (~10x others at times), so multipliers were near-constant — a weak test.
- Not tested: LEAP hand (deferred), longer training budgets, placebo at n=5, an observer that is reliable outside its training envelope (extrapolation rank corr −0.16).

---
## 2026-10-12 — G placebo completed at n=5 (supersedes the n=3 placebo reading above)

Placebo seeds 1,2 finished (`logs/chain12_placebo.log`); matrix regenerated (`results/MATRIX_A_to_G.md`). G vs placebo (G − Gp, exact permutation):
- nominal: mean err +0.38 mm (p=0.85), worst-decile −5.4 mm (p=0.82), 2 cm −3.0 pts (p=0.32) — indistinguishable.
- DRtrain: +0.43 mm (p=0.79), worst decile −6.9 (p=0.74) — indistinguishable.
- DRshift: mean err −23.0 mm (p=0.071), worst decile −211 mm (p=0.024), 5 cm +4.0 pts (p=0.048). G better, driven by large placebo variance (68 ± 29 mm vs 45 ± 6).
Reading: on in-distribution evals the PINN-informed schedule is no better than a shuffled one. The DRshift advantage is the only sign of a PINN-specific effect; it is one of ~15 uncorrected contrasts on that eval and rests on placebo seeds that left some dimensions un-widened, so it is a hypothesis to retest, not a finding. Honest overall verdict for rung G: **not demonstrated**. Suggested follow-up (not run): ≥10 seeds on DRshift only, G vs placebo vs F4, with a pre-registered primary metric (worst-decile error).

---
## 2026-10-12 — Retest launched: G vs placebo on DRshift at 10 seeds
Pre-declared before seeing new data: primary metric = DRshift worst-decile error (G vs Gp, exact/Monte-Carlo permutation); secondary = DRshift mean error and 5 cm success. New seeds 3,4,5,6,8 added to the existing 42,123,7,1,2 for both arms (chain13 = G, chain14 = placebo; same protocol/observer `seq`). Unchanged code. Result will be appended below whatever it shows.

### Result of the 12 Oct retest (SUPERSEDED — see the 2026-10-13 data-integrity entry below: seeds 3–6 were contaminated)
DRshift, PRIMARY worst-decile error: G 256.8 ± 197.9 mm vs placebo 381.9 ± 289.2 mm, diff −125 mm, **p=0.281** (the 5-seed p=0.024 did not replicate). Mean error 53.0 vs 72.2 mm (p=0.280); 5 cm success 97.0 vs 93.7% (p=0.181). Direction still favours G, magnitude shrank, and variance is large in both arms. **Verdict: no demonstrated benefit of PINN-informed step scaling over a shuffled placebo; rung G closed as a negative/inconclusive result on the 2-DOF task.** The earlier DRshift signal was most likely seed noise.

---
## 2026-10-12 (evening) — LEAP Hand Track B queue launched (unattended, restart-safe)
`scripts/leap_queue.sh`: sequential, resumable (skips jobs whose `results/LEAP_<cond>_s<seed>_fullDR.json` exists; flock-guarded). Persistence: launched with setsid/nohup **and** user crontab (`@reboot` + every 15 min) so it resumes after a reboot or crash; an interrupted training job restarts from scratch (~1 h). Status: `logs/leap_queue_status.log`.
Conditions (500 PPO iters, 4096 envs, task `Isaac-Reorient-Cube-Leap`, seeds 42/123/7, order seed-major so partial results are usable): **L0** native ADR off (`env.enable_adr=false`); **L1** native ADR as shipped (seed 42 reuses the existing 2026-09-16 run); **L2** native ADR warm-started at 8/25 increments (`env.starting_adr_increments=8`). L2 is a *proxy* for "begin at the difficulty ADR reached" — **not RAPP**: no LEAP RAPP feasibility sweep has been run, so the roadmap's RAPP-initialised arm remains untested.
Evaluation (new, identical for all): `eval_leap.py --adr_increments {0,25}` with ADR frozen (`min_rot_adr_coeff=1e9`), 1024 envs, 600 steps, eval seed 1234; metric rotations/20 s. The older `LEAP_500.json` was evaluated without freezing and is not comparable. Summary: `scripts/analyze_leap.py` → `results/MATRIX_LEAP.md`. Expected total ≈ 8 h. Smoke-tested end to end (2-iter L0 train → frozen full-DR eval).

### 2026-10-13 — LEAP queue failure and fix
The first queue (20:47 12 Oct) failed every training job within ~3 min: CUDA out-of-memory / illegal-memory-access at environment init, because other users' jobs on the shared L4 held ~15.7 of 23 GB (LEAP at 4096 envs needs roughly 10 GB+). The runner logged "FAILED" and moved on, and the 15-min cron relaunched the whole loop ~25 times; one attempt also hung for hours (killed). Only the reused L1 seed-42 run has results. Fix: `leap_queue.sh` now waits for ≥12000 MiB free GPU memory before each job, retries training up to 4× (10 min apart), caps training at 3 h, and then gives up on that job and continues. No results from this queue should be read as comparable until jobs complete. Lesson: an unattended queue must treat resource contention as "wait", not "fail".

### 2026-10-13 (midday) — correction: the LEAP "failures" were an upstream early-stop, not crashes
Per-attempt logs showed attempts ending with rc=0 at epoch 272/331/384: the upstream LEAP rl_games config has `score_to_win: 100000`, and rl_games stops with "Maximum reward achieved. Network won!" once reward exceeds it. Without ADR (L0) the task is easy enough to hit this early; with ADR (L1) reward stayed ~84k, so it ran the full 500 epochs. My earlier host-RAM-OOM explanation for the 26–37 min "failures" was therefore wrong (it was a hypothesis from circumstantial evidence — nearly-full RAM — and is retracted; the first failures overnight were genuine GPU OOM from other users).
Decision: for matched budget all conditions now train exactly 500 iterations (`agent.params.config.score_to_win=1e9`), documented as a deviation from the upstream early-stop. L0 seed 42 was assembled by resuming (ep 272→331→384 by early-stop/resume cycles, then resumed from ep 384 to 500 with the stop disabled); resumption restarts the optimizer/env state, so L0 s42 is a slightly different training trajectory from the uninterrupted runs — flag when interpreting. Also fixed: an orphaned `sleep` from a killed queue held the flock and silently blocked the relaunch.

### 2026-10-13 — DATA-INTEGRITY FINDING: the n=10 G-vs-placebo retest was partly contaminated (seeds 3–6); verdict unchanged
While assembling documentation I noticed that G seed 4 and Gplacebo seed 4 had byte-identical nominal/DR metrics. Cause: chain13 (G) and chain14 (placebo) were launched concurrently, and Isaac Lab/rl_games names run directories by timestamp to the second, so when both arms started a chunk in the same second they shared one `logs/rl_games/two_dof_reach/<timestamp>/` directory; `find_checkpoint()` then resolved the checkpoint by glob in that directory and could return the *other arm's* file, which the next chunk resumed from. Audit (all `*_final_checkpoint.txt` + every adaptive chunk's experiment name): 8 shared-directory chunks, all in seeds 3, 4, 5, 6 (seed 4's final checkpoint is literally the same file for both arms). Seeds 42/123/7/1/2 (arms ran at different times) and seed 8 have no collisions; static (non-adaptive) 800-iter runs have none.
Action: contaminated results moved to `results/_quarantine_contaminated_retest_20261012/` (the earlier n=10 text kept there, labelled CONTAMINATED); retest recomputed on the 6 clean seeds (`results/G_vs_placebo_retest_clean_n6.txt`): DRshift worst-decile G 250.7 ± 194.7 vs placebo 487.1 ± 307.5 mm, diff −236 mm, p=0.147 (secondary: mean err p=0.169, 5 cm p=0.104). Verdict unchanged — no demonstrated PINN benefit — but power is lower than I stated (n=6, not 10). The 5-seed A–G matrix is unaffected (its seeds are clean). A clean rerun of seeds 3–6 with the two arms run *sequentially* is queued (chain15, starts after the LEAP queue finishes). The earlier statement "10 seeds/arm, p=0.281" in this log and in the roadmap/results/slides is superseded by this entry.
Lesson: concurrent launches must not share a timestamp-keyed output directory; checkpoint lookup should use a unique per-run path, and identical results across arms should be treated as a red flag, not a coincidence.
