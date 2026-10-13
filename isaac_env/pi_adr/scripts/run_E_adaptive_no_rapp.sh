#!/usr/bin/env bash
# Rung E (roadmap Table 7): Eureka + adaptive DR, WITHOUT RAPP's feasibility
# envelope (--no_rapp). Same controller/gate/step-rate as F4, so the only
# difference from F4 is whether the widening is clipped to a measured,
# task-feasible C_RAPP box or to arbitrary wide bounds -- isolates RAPP's own
# contribution from the adaptive mechanism's.
set -uo pipefail
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
PY=$ROOT/env_isaaclab/bin/python
cd "$ROOT/pi_adr"
SEED=${1:-42}
RUN=E_adaptive_no_rapp

$PY -u scripts/run_adaptive_dr.py \
  --task PIADR-TwoDofReach-Eureka-v0 --num_envs 2048 --eval_envs 1024 \
  --chunk_iters 100 --outer_steps 8 --step_frac 0.10 \
  --gate_threshold 0.02 --success_target 0.30 --seed "$SEED" --no_rapp \
  --run_name "$RUN" \
  > "logs/run_${RUN}.log" 2>&1

CKPT=$(cat "results/adaptive_dr/${RUN}_final_checkpoint.txt")
echo "[${RUN}] checkpoint: $CKPT"

for VARIANT in nominal DRtrain DRshift; do
  case $VARIANT in
    nominal) DRARGS="" ;;
    DRtrain) DRARGS="--dr --friction_range 0.0 0.25 --damping_scale_range 0.5 4.0 --effectiveness_range 0.5 1.0 --mass_scale_range 1.0 1.5" ;;
    DRshift) DRARGS="--dr --friction_range 0.25 0.5 --damping_scale_range 4.0 8.0 --effectiveness_range 0.3 0.5 --mass_scale_range 1.5 2.5" ;;
  esac
  $PY -u scripts/eval.py --task PIADR-TwoDofReach-Eureka-v0 --checkpoint "$CKPT" \
    --run_name "${RUN}_${VARIANT}" --num_envs 4096 --headless $DRARGS \
    --note "rung E: Eureka + adaptive DR, no RAPP, seed=${SEED}, variant=${VARIANT}" \
    > "logs/eval_${RUN}_${VARIANT}.log" 2>&1
  sed -n "/EVAL RESULT/,/^====/p" "logs/eval_${RUN}_${VARIANT}.log"
done
