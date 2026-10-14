# 11 — LEAP Hand (Track B)

> **STATUS: COMPLETE (13 Oct 2026, 22:08 UTC).** All 9 runs (L0/L1/L2 × seeds 42, 123, 7) trained to 500 iterations and were evaluated at ADR = 0 and ADR = 25. Live table: `python scripts/analyze_leap.py` → `results/MATRIX_LEAP.md`; statistics: `results/LEAP_stats_n3.txt`; status log: `logs/leap_queue_status.log`.

**Purpose.** Run the roadmap's study on a real manipulation platform with the task upstream trains it for (`Isaac-Reorient-Cube-Leap`, in-hand cube reorientation; <https://github.com/leap-hand/LEAP_Hand_Isaac_Lab>), keeping the stock configuration so settings could be copied from the published project.

## 1. First result and the discovery that changed the plan (Sep–Oct 2026)

- **Stock run (`LEAP_500`, 16 Sep):** unmodified config (GRU-256, horizon 32, minibatch 32768, 5 mini-epochs, lr 5e-4 adaptive), 500 PPO iterations, 4096 envs, seed 42. Return 33.09 → **84,254.54**; **7.52 rotations per 20 s per hand**.
- **Setup fixes** needed to run at all on Isaac Lab 2.3: `train.py` import (`dump_pickle`), missing `operation` in `EventCfg`, `setup.py` keywords (doc 02).
- **Major correction (10 Oct):** while scoping "port adaptive DR to LEAP" I read `leap_hand_env_cfg.py` and `utils/adr.py` closely and found that **the upstream repo already ships a working OpenAI-style automatic domain randomization (`LeapHandADR`)**, enabled by default (`enable_adr = True`, `starting_adr_increments = 0`; 0 is a *starting point*, not "disabled"). Mechanism: `adr_criteria = (consecutive_successes / z_rotation_steps) / episode_seconds` (rotations per second); whenever it exceeds `min_rot_adr_coeff = 0.15` (one full rotation per 6.67 s) every parameter in `adr_cfg_dict` widens by one of 25 linear increments toward its configured target (friction, restitution, stiffness/damping, object mass/scale, wrench, latency, noise).
- **Verified against `LEAP_500`'s own TensorBoard** (not assumed): `Episode/num_adr_increases` rose **0 → 8 of 25**; `Episode/adr_criteria` rose from 0.0002 to a peak of 0.315, repeatedly crossing 0.15.

**Consequence.** `LEAP_500` is "native ADR active from increment 0", **not** a no-DR rung. The original plan (port our controller) was replaced with non-redundant questions: (1) a true no-ADR baseline (`env.enable_adr=false`, verified as a valid override reaching `LeapHandEnvCfg`); (2) does ADR benefit from a feasibility-derived (RAPP-style) starting envelope; (3) how does LEAP's coarse-threshold widening rule compare with a precision-aware rule (open).

## 2. The matrix (designed 12 Oct, running)

500 PPO iterations, 4096 envs, seeds 42/123/7, ordered seed-major so partial results are usable.

| Condition | Meaning | Config override |
|---|---|---|
| L0 | native ADR off | `env.enable_adr=false` |
| L1 | native ADR as shipped (seed 42 reuses the 16 Sep run) | none |
| L2 | native ADR warm-started at 8 of 25 increments | `env.starting_adr_increments=8` |

**L2 is a proxy, not RAPP.** It begins at the difficulty ADR reached by iteration 500 in L1 (8 increments). No LEAP RAPP feasibility sweep has been run; the roadmap's "RAPP-initialised ADR" arm remains **untested**.

**Evaluation (new, identical for all):** `scripts/eval_leap.py --adr_increments {0, 25}` — ADR **frozen** (`min_rot_adr_coeff = 1e9`), 1024 envs, 600 control steps (20 s), eval seed 1234, metric rotations / 20 s / hand, plus joint speed and effort. *Nominal* = 0 increments; *full DR* = 25 increments (the full native range). **Why frozen:** at eval the ADR criterion could otherwise advance the difficulty mid-evaluation, giving every policy a different test. The older `LEAP_500.json` was evaluated unfrozen and is not comparable (its re-evaluation under the frozen protocol is `LEAP_L1_s42_nominal`: 7.527 vs the original 7.517 — near-identical, a useful consistency check).

## 3. Results (3 seeds per condition, rotations per 20 s per hand, ADR frozen at evaluation)

| Condition | Nominal (ADR = 0) | Full DR (ADR = 25) | per-seed nominal (42, 123, 7) | per-seed full DR |
|---|---|---|---|---|
| L0 — native ADR off | 8.42 ± 2.22 | 0.56 ± 0.31 | 10.88, 6.57, 7.80 | 0.92, 0.37, 0.38 |
| L1 — native ADR (upstream) | 8.68 ± 2.76 | 0.75 ± 0.12 | 7.53, 11.83, 6.68 | 0.86, 0.79, 0.62 |
| L2 — ADR warm-start at 8/25 | 8.00 ± 2.11 | 1.53 ± 1.20 | 5.68, 9.80, 8.52 | 0.60, 1.11, 2.89 |

Exact permutation tests (3 vs 3; the smallest attainable two-sided p is 0.10):

| contrast | nominal diff (p) | full-DR diff (p) |
|---|---|---|
| L1 − L0 | +0.26 (0.90) | +0.19 (0.40) |
| L2 − L1 | −0.68 (0.70) | +0.78 (0.40) |
| L2 − L0 | −0.42 (0.80) | +0.98 (0.20) |

Secondary measures (mean over 3 seeds): joint speed ≈ 2.0 rad/s for all conditions at both difficulties (no difference); **control effort per joint** nominal L0 0.60 ± 0.02, L1 0.72 ± 0.02, L2 0.65 ± 0.07; full DR 0.52, 0.57, 0.60. Effort for L1 exceeds L0 in 3 of 3 seeds with complete separation (nominal 0.70/0.71/0.74 vs 0.61/0.58/0.61) — the minimum possible p for 3 vs 3 (0.10), so directional only.

**What this means, carefully**
1. **No condition effect can be claimed.** Every contrast has p ≥ 0.20; seed-to-seed spread inside a condition (2–3 rotations, e.g. L1 nominal 6.68–11.83) is larger than every between-condition difference. The single-seed impression after seed 42 (L0 ≫ L1) vanished when seeds were added: it was seed variance — the same lesson as on the 2-DOF track.
2. **At full randomization every policy collapses** (0.6–1.5 rotations per 20 s versus ≈ 8–9 at nominal, a ≈ 6–15× drop). Training for 500 iterations — during which native ADR advanced only to about 8 of 25 increments — does not produce a policy that survives the full range, regardless of how DR was handled. The L2 mean of 1.53 is driven by one seed (2.89); the other two (0.60, 1.11) are in the same band as L0/L1.
3. **ADR-trained policies spend more control effort** at equal speed (0.72 vs 0.60), consistent with training under actuator/action noise; whether this matters for hardware is untested.
4. **Budget is the main limitation.** 500 iterations is short relative to the published LEAP training; ADR's benefit presupposes training long enough to advance through the increments. These results therefore do **not** test whether ADR helps; they show that at this budget ADR neither helps nor hurts measurably.
5. **Consistency check on the evaluation:** re-evaluating the original 16 Sep checkpoint under the new frozen protocol gave 7.527 rotations vs the original unfrozen 7.517 — the frozen protocol reproduces the earlier number.

**Value for the report.** (i) The native-ADR confound (the stock "baseline" was an ADR run); (ii) a controlled demonstration that seed variance dominates at this budget; (iii) a frozen-difficulty evaluation protocol for LEAP; (iv) a specified next experiment (longer budgets, precision-aware ADR criterion — doc 13).

## 4. What went wrong while running it unattended (and what each failure was *actually*)

This sequence is documented because misdiagnosis cost hours.

1. **Queue v1 (12 Oct 20:47):** every training job failed within ~3 min. Cause (confirmed in the log): **CUDA out-of-memory** — other users' jobs on the shared L4 held ≈ 15.7 of 23 GB, and LEAP at 4096 envs needs ≈ 11 GB. The runner logged "FAILED" and moved on, and the cron entry relaunched the loop ≈ 25 times; one attempt also hung for hours (killed). *Fix:* gate each job on free GPU memory (≥ 12 GB), retry, cap training at 3 h.
2. **Queue v2/v3:** attempts still "failed" after 26–37 min with no error message. A **wrong hypothesis** was written down — silent host-RAM OOM kills (host RAM was nearly full from other users). It was circumstantial (kernel log unreadable) and untested.
3. **Per-attempt logs** (added in v3) revealed attempts ending with **return code 0** at epochs 272, 331, 384 — not crashes. The training log's last lines: *"Maximum reward achieved. Network won!"* Cause: upstream `rl_games_ppo_cfg.yaml` sets `score_to_win: 100000`; training stops when reward exceeds it. Without ADR the task is easy enough to hit it early; with ADR (L1) reward stayed ≈ 84 k and ran the full 500 epochs. **The RAM hypothesis was wrong and was retracted.**
4. **Matched budget decision:** to compare conditions at equal compute, `agent.params.config.score_to_win=1000000000` is now passed for every run, so all conditions train exactly 500 iterations — a **documented deviation** from the upstream early stop.
5. **Resume artifact:** L0 seed 42 was assembled by early-stop/resume cycles (epochs 272 → 331 → 384, each restarting optimizer and environment state), then resumed from epoch 384 to 500 with the stop disabled (`save_frequency` 50 for finer checkpoints). It is therefore a *slightly different training trajectory* from uninterrupted runs. If it looks like an outlier at the end, rerun it uninterrupted.
6. **Queue v4/v5 operational bugs:** an orphaned `sleep` from a killed queue held the `flock`, silently blocking the relaunch; and an 18 GB host-RAM gate (based on the retracted hypothesis) stalled the L2 evaluation by 400 MiB while the GPU sat idle. v5 reuses already-trained checkpoints, lowers the RAM gate to 10 GB for training and 3 GB for evals.

**Persistence design.** Launched with `setsid nohup` and a user crontab (`@reboot` + every 15 min) so it resumes after a disconnect or reboot; the `flock` prevents overlap. Truthful limits: if the machine is *powered off* the job stops until it boots; an interrupted training run restarts from scratch (or from its last 50-epoch checkpoint if resumed manually); the queue idles whenever other users' jobs occupy the GPU/host RAM. Observed in practice: after a host reboot at ≈ 10:49 on 13 Oct the cron `@reboot` entry restarted the queue automatically.

## 5. Caveats specific to these runs

- L0 seed 42 was assembled from early-stop/resume cycles (epochs 272 → 331 → 384 → 500) and L1 seed 123 resumed once from an epoch-250 checkpoint after a kill (return code 137, cause unknown). Resuming restarts optimizer/environment state; if any single-seed number looks like an outlier, this is a candidate cause. Neither run is an outlier in the final table (L0 s42 is the highest L0 nominal, 10.88, but L1 s123 at 11.83 is higher).
- L1 seed 42 is the 16 Sep run (upstream early-stop threshold was never reached, so identical to a 500-iteration run); L0 runs hit the upstream early-stop threshold at 272–384 epochs unless it was disabled (it was disabled for all new runs).
- L2 is an ADR warm-start proxy, **not RAPP**.
- 3 seeds cannot reach p < 0.10 on any contrast, by construction.

## 6. Open questions for this track (see doc 13)

1. Longer training (≥ 1500–5000 iterations) so native ADR can reach higher increments.
2. A genuine LEAP RAPP sweep to derive a feasibility envelope and test RAPP-initialised ADR.
3. Whether LEAP's coarse-threshold criterion hides a quality-versus-quantity trade-off analogous to the 2-DOF coarse/fine finding.
4. Sim-to-real (LEAP_Hand_API <https://github.com/leap-hand/LEAP_Hand_API>) — out of scope until simulation results are solid.
