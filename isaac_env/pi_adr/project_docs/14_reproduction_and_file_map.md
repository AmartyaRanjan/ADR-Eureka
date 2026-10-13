# 14 — Reproduction commands and file map

All paths are under `ROOT=/data/home/amartyaranjan888/projects/I/isaac_env`; `PY=$ROOT/env_isaaclab/bin/python`; project directory `$ROOT/pi_adr`.

## 1. Code map

| Path | What it is |
|---|---|
| `source/pi_adr/tasks/two_dof_reach/two_dof_reach_env.py`, `_cfg.py` | The 2-DOF `DirectRLEnv`, rewards, DR, `VanillaRewardCfg`, `DomainRandCfg`, `physics_xi`, `commanded_torque` |
| `source/pi_adr/assets/two_dof_arm.urdf` (+ cached `usd/`) | Arm asset |
| `source/pi_adr/evaluation/metrics.py`, `runner.py` | Frozen metrics (v1.0.0), rollout (`horizon−2`), `set_domain_rand` |
| `source/pi_adr/dr/adaptive.py` | `AdaptiveDRController` incl. `step_multiplier` |
| `source/pi_adr/pinn/dynamics.py`, `model.py` | Closed-form dynamics, observer ensemble, `XI_BOUNDS`, losses |
| `scripts/validate_env.py`, `validate_dynamics.py`, `pd_baseline.py` | Environment validators and the IK+PD ceiling |
| `scripts/train.py`, `eval.py`, `train_and_eval.sh` | Training wrapper (Hydra), evaluation, combined launcher |
| `scripts/run_rapp.py` | RAPP sweep |
| `scripts/run_adaptive_dr.py`, `eval_dr_dims.py` | Adaptive outer loop and per-dimension evaluation |
| `scripts/collect_pinn_data.py`, `train_pinn.py`, `diagnose_pinn_covariance.py`, `estimate_settling_time.py`, `pinn_probe.py` | PINN data, training, diagnostics, probe for rung G |
| `scripts/run_{D,E,F4,G}_seed.sh`, `run_C_human_static_dr.sh`, `run_eureka_nodr_seed.sh`, `run_reward_compare_seed.sh`, `run_vanilla_entropy_seed.sh`, `eval_dr_for_run.sh` | Per-seed wrappers (see naming below) |
| `scripts/analyze_matrix.py`, `analyze_n5.py`, `analyze_g_retest.py`, `analyze_leap.py`, `summarize_results.py` | Statistics and tables |
| `scripts/eval_leap.py`, `leap_queue.sh` | LEAP evaluation (frozen ADR) and the unattended queue |
| `results/*.json`, `results/index.jsonl` | One JSON per evaluation; append-only index |
| `results/adaptive_dr/` | Controller trajectories, per-outer-step evals, probes, final checkpoint paths |
| `results/_quarantine_contaminated_retest_20261012/` | Contaminated G/Gp seeds 3–6 (kept for audit) |
| `data/pinn_*_{longhold,seq}.pt` | PINN datasets and observers (tags: none = single regime, `longhold`, `seq`) |
| `logs/` | Training/eval logs (per attempt for LEAP), chain logs, `leap_queue_status.log` |

## 2. Run-name conventions (needed to reproduce the tables)

| Rung / item | Result file pattern (`results/…`) |
|---|---|
| A | `vanilla_seed{42,123,7}_800iter_<eval>`; seeds 1,2: `vanilla_ent0.005_seed{S}_800iter_<eval>` |
| B | `eureka_seed{S}_800iter_<eval>` |
| C | `C_human_static_dr_<eval>` (seed 42); `C_seed{S}_human_static_dr_<eval>` |
| D | `D_<eval>` (42); `D_seed{S}_<eval>` |
| E | `E_adaptive_no_rapp_<eval>` (42); `E_seed{S}_no_rapp_<eval>` |
| F4 | `F4_<eval>` (42); `F4_seed{S}_<eval>` |
| G / placebo | `G_seed{S}_<eval>` / `Gplacebo_seed{S}_<eval>` |
| Entropy 0.02 | `vanilla_ent0.02_seed{S}_800iter_nominal` |
| LEAP | `LEAP_{L0,L1,L2}_s{S}_{nominal,fullDR}` |
`<eval>` ∈ {`nominal`, `DRtrain`, `DRshift`}. Seeds: 42, 123, 7, 1, 2 (matrix); 3, 4, 5, 6, 8 (retest).

## 3. Commands

```bash
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env; PY=$ROOT/env_isaaclab/bin/python; cd $ROOT/pi_adr

# Validation and ceiling
$PY -u scripts/validate_env.py --headless        # 10/10 checks
$PY -u scripts/validate_dynamics.py --headless   # closed form vs PhysX
$PY -u scripts/pd_baseline.py --headless         # IK+PD ceiling

# Baselines
./scripts/train_and_eval.sh PIADR-TwoDofReach-Vanilla-v0 5000 A_vanilla_5000
./scripts/train_and_eval.sh PIADR-TwoDofReach-Eureka-v0  1500 B_eureka_1500

# RAPP
$PY -u scripts/run_rapp.py --headless --checkpoint <ckpt> --run_name rapp_eureka500

# PINN (sequential-hold observer)
for spec in 'train 4096 0' 'val 1024 7' 'extrap 1024 11'; do set -- $spec
  $PY -u scripts/collect_pinn_data.py --headless --split $1 --num_envs $2 --seed $3 --steps 192 --long_hold_steps 300 \
     --torque_scale 0.15 0.45 --long_hold_target_velocity 2.0 --long_hold_mode sequential --tag seq; done
$PY -u scripts/train_pinn.py --epochs 2000 --tag seq
$PY scripts/diagnose_pinn_covariance.py --tag seq

# Matrix rungs (per seed S)
./scripts/run_reward_compare_seed.sh S            # A (vanilla) and B (Eureka) nominal
./scripts/run_eureka_nodr_seed.sh S               # B incl. DR evals
./scripts/run_C_human_static_dr.sh S
./scripts/run_D_seed.sh S;  ./scripts/run_E_seed.sh S;  ./scripts/run_F4_seed.sh S
./scripts/run_G_seed.sh S pinn seq;  ./scripts/run_G_seed.sh S shuffle seq    # run these SEQUENTIALLY, never concurrently
./scripts/eval_dr_for_run.sh vanilla_seed42_800iter vanilla   # DR evals for an existing checkpoint

# Statistics
python3 scripts/analyze_matrix.py      # -> results/MATRIX_A_to_G.md
python3 scripts/analyze_n5.py          # D vs E, entropy comparison
python3 scripts/analyze_g_retest.py    # G vs placebo (uses whichever seeds exist)
python3 scripts/analyze_leap.py        # -> results/MATRIX_LEAP.md

# LEAP (unattended, resumable)
nohup setsid scripts/leap_queue.sh > logs/leap_queue_stdout.log 2>&1 &
tail logs/leap_queue_status.log
```

## 4. Compute notes

- Shared L4: runs of this project need ≈ 5–6 GB (2-DOF, 2048 envs) and ≈ 11 GB (LEAP, 4096 envs); other users' jobs often occupy 14–16 GB, so LEAP waits for free memory.
- Typical wall clocks: static 800-iteration run ≈ 10–15 min; adaptive-DR run (with 8 evaluations and, for G, 8 probes) ≈ 30 min; LEAP 500 iterations ≈ 55–65 min at ≈ 20 k frames/s.
- rl_games `--checkpoint` resume restores the cumulative epoch: `--max_iterations` must be the *cumulative* target.

## 5. Known pitfalls (each cost time once)

1. Isaac Lab/rl_games run directories are timestamp-named to the second — never launch two trainings in the same second without distinct experiment names (doc 04, D7).
2. `pkill -f` can match the invoking shell; kill by PID. An orphaned `sleep` can keep a `flock` held.
3. Launcher "completed" notifications are not job completion; poll status files.
4. `env.reset()` after a first rollout must be inside `torch.inference_mode()`.
5. LEAP: `score_to_win: 100000` early-stops training; pass `agent.params.config.score_to_win=1000000000` for matched budgets.
6. Never print or commit the `.env` API key.
