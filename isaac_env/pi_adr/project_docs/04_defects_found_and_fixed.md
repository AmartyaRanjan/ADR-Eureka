# 04 — Defects found and fixed

Each item: **what happened, why it mattered, how it was detected, how it was fixed, and how the fix was verified.** These are recorded because every one would have silently invalidated downstream numbers.

## A. Environment/evaluation defects (Session 1, 15 Sep 2026)

### D1 — Evaluation recorded the post-reset distance
- **Symptom:** a partially trained vanilla policy scored **0.39 % success at 5 cm**, yet its training return (≈ 1208–1283) implied it reached and held the goal.
- **Cause:** `DirectRLEnv` fires `time_out` at `episode_length_buf >= max_episode_length − 1` and resets inside that same step. A rollout of `horizon − 1` steps therefore recorded the *post-reset* pose — a fresh random arm and fresh random goal — i.e. chance-level success.
- **Fix:** roll out `horizon − 2` steps (terminal sample at t = 9.93 s of 10 s).
- **Verification:** re-evaluation of the same checkpoint gave **93.75 % / 20.70 % / 3.91 %** at 5/2/1 cm — the expected curve shape.
- **Lesson:** a metric that is wildly inconsistent with training return is a bug until shown otherwise.

### D2 — Joint damping was 0, not the documented 0.05
- **Symptom/cause:** `ImplicitActuatorCfg(damping=0.0)` silently overrode the URDF's `<dynamics damping="0.05">`; read-back showed `[0.0, 0.0]`.
- **Why it mattered beyond fidelity:** DR samples damping as nominal × scale; "DR at scale 1.0" would have been 0.05 while the nominal baseline was 0.0, so **every DR-vs-nominal comparison would carry a constant offset.**
- **Fix:** `damping = NOMINAL_JOINT_DAMPING`; verified by read-back.

### D3 — ±π joint limits capped the task at ≈ 69 %
- **Detection:** a scripted IK + PD controller (`scripts/pd_baseline.py`) — an *optimal* reference measured through the same interface:

| Controller | 5 cm | 1 cm | 1 mm | mean err | failures at a joint stop |
|---|---|---|---|---|---|
| IK+PD, ±π limits | 68.60 % | 67.68 % | 67.24 % | 152.6 mm | **100 % of its 31.40 % failures** |
| IK+PD, limits widened | **100 %** | **100 %** | **100 %** | 0.0 mm | 0 % |

- **Cause:** when a goal needs rotation beyond ±π, the shortest angular path is blocked; the arm stalls against the stop (a wrap-around trap). This explained the heavy error tails of the first run generation (p90 = 600 mm, worst decile 744 mm).
- **Two conclusions:** (1) the environment supports millimetre precision once unblocked, so the learned 30 mm floor is a reward/learning result; (2) the report's 99.49 % at 5 cm is unattainable under hard ±π limits, so its limits cannot have been binding.
- **Fix:** both joints continuous; observed angles wrapped into [−π, π] so the 8-D observation layout and boundedness are unchanged.

### D4 — PINN dataset labels did not satisfy the equations of motion
- **Symptom:** the first observer reached parameter error 0.23 and acceleration RMSE 23.6 rad/s². Instead of tuning the model, the *data* was checked: evaluating the closed-form dynamics at the **ground-truth** parameters gave an RMSE of **78.8 rad/s²** — worse than the learned model. The model was fitting label noise.
- **Two causes:**
  1. q̈ was differenced over a *control* step. `env.step()` holds torque for 4 physics substeps, so (q̇ₜ₊₁ − q̇ₜ)/0.033 is an average acceleration that does not satisfy the EOM at the recorded state. **Fix:** step physics one 1/120 s step at a time.
  2. The velocity limit engages from ≈ 4 rad/s, not 5. Median model error by speed band: 4.13, 4.18, 4.27, 4.38 rad/s² for |q̇| in [0,1), [1,2), [2,3), [3,4), but **68.18** for [4, 4.9). The original mask (0.98 × 5 rad/s) let the saturated band through. **Fix:** mask at 3.5 rad/s with lower excitation torque.
- **Result:** ground-truth-ξ RMSE fell to **6.69 rad/s²** (5.3 % of mean |q̈|).

### D5 — Mass randomization skipped the write at scale 1.0
- **Effect:** RAPP's mass sweep inherited the previous candidate's masses; nominal scored **0.816** instead of **0.975**.
- **Fix:** `_randomize_physics` always writes masses; verified by reproducing 0.975 at nominal.

### D6 — Validation checks were themselves wrong
- Checks D and J of `validate_env.py` encoded an idealised torque law (ignored drive damping, Coulomb friction, and inter-joint coupling through M(q)). Corrected to test against the closed-form coupled dynamics.

## B. Tooling defects

| # | Defect | Effect | Fix |
|---|---|---|---|
| T1 | `train_and_eval.sh` located checkpoints by "most recent run directory" | With two overlapping trainings, the first `A_vanilla_5000` eval picked the *other* stream's folder | Parse each run's own log line `Exact experiment name requested from command line:` |
| T2 | Second rollout in one process raised "inference tensor" errors | Evaluation crashed on repeated resets | Run `reset()` inside `torch.inference_mode()` |
| T3 | `run_adaptive_dr.py` used a per-chunk `--max_iterations` | Silent zero-iteration chunks from the second resume | Cumulative iterations (doc 02) |
| T4 | F3 mis-specified: `success_target` 0.90 carried over to a 2 cm gate | No policy in the project exceeded ~42 % at 2 cm, so the controller could never widen; ranges froze | Killed after 3 outer steps; F4 with target 0.30 (doc 09) |
| T5 | `pkill -f` matched the invoking shell / left orchestrators alive | Tool calls killed; stale orchestrators | Kill by PID |
| T6 | Launcher "task completed" notifications are launcher returns, not job completion | Mistaken for finished runs | Poll logs/status files |

## C. The late, high-impact defects (12–13 Oct 2026)

### D7 — Shared timestamp-named run directories contaminated the G-vs-placebo retest
- **Detection (red-flag audit):** G seed 4 and Gplacebo seed 4 produced *byte-identical* nominal and DR metrics.
- **Cause:** the two arms (chain13 and chain14) were launched concurrently. Isaac Lab/rl_games names run directories `YYYY-MM-DD_HH-MM-SS`. When both arms started a chunk in the same second they wrote into the same directory, and `find_checkpoint()` located the checkpoint by glob in that directory, so it could return the other arm's file, which the next chunk resumed from. For seed 4, both arms' **final** checkpoint path was literally the same file.
- **Extent (full audit of all final-checkpoint paths and every adaptive chunk's experiment name):** 8 shared-directory chunks, all in seeds 3, 4, 5, 6. Seeds 42/123/7/1/2 (arms ran at different times) and seed 8 had none; static runs had none.
- **Resolution:** contaminated results moved to `results/_quarantine_contaminated_retest_20261012/`; retest recomputed on the 6 clean seeds; clean rerun of seeds 3–6 with arms run **sequentially** queued (`logs/chain15_retest_rerun.log`). The earlier "n=10, p=0.281" text is superseded everywhere.
- **Lesson:** two runs must never share a timestamp-keyed directory; identical results across arms are a defect signal.

### D8 — Misinterpreting the LEAP early stop as crashes (and a wrong OOM hypothesis)
Covered in doc 11: upstream `score_to_win: 100000` stops training when reward passes 100 k; the queue logged this as failure. An intermediate hypothesis ("host-RAM OOM kills") was circumstantial and **wrong**; it was retracted in the log.

### D9 — LEAP queue resource-contention handling
The first queue treated CUDA out-of-memory (other users' jobs) as job failure and cron relaunched the loop ~25 times; one attempt hung for hours. Redesigned to *wait* for resources (doc 11).

## D. Cross-cutting lessons

1. Compare against an oracle (IK+PD, analytic FK, closed-form dynamics) before trusting a result.
2. When a number contradicts another measurement of the same system (eval vs return, true-ξ RMSE vs learned RMSE), debug the measurement first.
3. Verify configuration by reading state back from the simulator.
4. Anything unattended must treat contention as "wait", not "fail", and keep one log per attempt so failure causes are not overwritten.
5. Never reuse a threshold calibrated for one regime (coarse 5 cm) in another (precision 2 cm).
