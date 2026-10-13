# 11 — LEAP Hand (Track B)

> **STATUS: IN PROGRESS.** As of 13 Oct 2026 15:50 UTC: seed 42 is complete for L0 and L1, L2 seed-42 training is done and its evaluation is starting; seeds 123 and 7 for all three conditions are still queued (≈ 55–65 min of training each when the GPU is available). This document will be finalised when the queue ends. Live table: `python scripts/analyze_leap.py` → `results/MATRIX_LEAP.md`; live status: `logs/leap_queue_status.log`.

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

## 3. Results so far (single seed — **do not interpret as findings**)

| Cond (seed 42) | trained rotations/20 s nominal (ADR=0) | full DR (ADR=25) | mean joint speed (nom / DR) | effort (nom / DR) | final train reward |
|---|---|---|---|---|---|
| L0 (no ADR) | **10.88** | 0.92 | 2.17 / 2.20 rad/s | 0.607 / 0.528 | > 100,000 (early-stop threshold) |
| L1 (native ADR) | 7.53 | 0.86 | 1.86 / 2.07 | 0.696 / 0.591 | 84,254 |
| L2 (warm-start 8) | *pending* | *pending* | | | 36,728 at epoch 500 |

What these numbers say, carefully:
- **Nominal:** the no-ADR policy rotates more (10.9 vs 7.5) at nominal difficulty. That is expected: L1 spent its training budget on increasingly hard physics; L0 on the easy case. This mirrors the 2-DOF finding that adaptive/wide DR costs nominal performance.
- **Full DR:** *both* policies collapse to ≈ 0.9 rotations per 20 s at 25 increments — about a 12× drop for L0 and 8.8× for L1. After only 500 iterations, native ADR (which reached 8 of 25 increments) does not produce a policy that survives the full randomization range; the difference between L0 and L1 there (0.92 vs 0.86) is within what noise would produce. So **500 iterations is probably too short to show an ADR benefit** — a limitation of the design, to be stated in any write-up.
- The L2 training reward (36,728) is far below L1's (84,254) because it trains at a harder fixed starting difficulty; its frozen-eval numbers will show whether that investment pays at full DR.

## 4. What went wrong while running it unattended (and what each failure was *actually*)

This sequence is documented because misdiagnosis cost hours.

1. **Queue v1 (12 Oct 20:47):** every training job failed within ~3 min. Cause (confirmed in the log): **CUDA out-of-memory** — other users' jobs on the shared L4 held ≈ 15.7 of 23 GB, and LEAP at 4096 envs needs ≈ 11 GB. The runner logged "FAILED" and moved on, and the cron entry relaunched the loop ≈ 25 times; one attempt also hung for hours (killed). *Fix:* gate each job on free GPU memory (≥ 12 GB), retry, cap training at 3 h.
2. **Queue v2/v3:** attempts still "failed" after 26–37 min with no error message. A **wrong hypothesis** was written down — silent host-RAM OOM kills (host RAM was nearly full from other users). It was circumstantial (kernel log unreadable) and untested.
3. **Per-attempt logs** (added in v3) revealed attempts ending with **return code 0** at epochs 272, 331, 384 — not crashes. The training log's last lines: *"Maximum reward achieved. Network won!"* Cause: upstream `rl_games_ppo_cfg.yaml` sets `score_to_win: 100000`; training stops when reward exceeds it. Without ADR the task is easy enough to hit it early; with ADR (L1) reward stayed ≈ 84 k and ran the full 500 epochs. **The RAM hypothesis was wrong and was retracted.**
4. **Matched budget decision:** to compare conditions at equal compute, `agent.params.config.score_to_win=1000000000` is now passed for every run, so all conditions train exactly 500 iterations — a **documented deviation** from the upstream early stop.
5. **Resume artifact:** L0 seed 42 was assembled by early-stop/resume cycles (epochs 272 → 331 → 384, each restarting optimizer and environment state), then resumed from epoch 384 to 500 with the stop disabled (`save_frequency` 50 for finer checkpoints). It is therefore a *slightly different training trajectory* from uninterrupted runs. If it looks like an outlier at the end, rerun it uninterrupted.
6. **Queue v4/v5 operational bugs:** an orphaned `sleep` from a killed queue held the `flock`, silently blocking the relaunch; and an 18 GB host-RAM gate (based on the retracted hypothesis) stalled the L2 evaluation by 400 MiB while the GPU sat idle. v5 reuses already-trained checkpoints, lowers the RAM gate to 10 GB for training and 3 GB for evals.

**Persistence design.** Launched with `setsid nohup` and a user crontab (`@reboot` + every 15 min) so it resumes after a disconnect or reboot; the `flock` prevents overlap. Truthful limits: if the machine is *powered off* the job stops until it boots; an interrupted training run restarts from scratch (or from its last 50-epoch checkpoint if resumed manually); the queue idles whenever other users' jobs occupy the GPU/host RAM. Observed in practice: after a host reboot at ≈ 10:49 on 13 Oct the cron `@reboot` entry restarted the queue automatically.

## 5. Planned analysis (when finished)

- Table: mean ± std rotations/20 s per condition at ADR = 0 and ADR = 25, with per-seed values; exact permutation tests only for n = 3 (min attainable two-sided p = 2/20 = 0.10 for 3 vs 3 — **significance at conventional levels is impossible with 3 seeds**, so differences must be large and consistent across seeds to be discussed).
- Report separately: (a) L0 vs L1 at nominal, (b) L1 vs L2 at full DR, (c) joint speed/effort.
- State plainly that L2 is an ADR warm-start, not RAPP, and that 500 iterations is short relative to LEAP's published training.

## 6. Open questions for this track (see doc 13)

1. Longer training (≥ 1500–5000 iterations) so native ADR can reach higher increments.
2. A genuine LEAP RAPP sweep to derive a feasibility envelope and test RAPP-initialised ADR.
3. Whether LEAP's coarse-threshold criterion hides a quality-versus-quantity trade-off analogous to the 2-DOF coarse/fine finding.
4. Sim-to-real (LEAP_Hand_API <https://github.com/leap-hand/LEAP_Hand_API>) — out of scope until simulation results are solid.
