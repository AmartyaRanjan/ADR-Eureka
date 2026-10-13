#!/usr/bin/env bash
# Evaluate an already-trained 800-iter checkpoint under DRtrain/DRshift (identical protocol to the other rungs).
# Usage: eval_dr_for_run.sh <run_name e.g. vanilla_seed42_800iter> <vanilla|eureka> <train_log_name (default train_<run>)>
set -uo pipefail
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
PY=$ROOT/env_isaaclab/bin/python
cd "$ROOT/pi_adr"
RUN=$1; TAGT=$2; LOG=${3:-train_${RUN}}
case $TAGT in vanilla) TASK=PIADR-TwoDofReach-Vanilla-v0;; eureka) TASK=PIADR-TwoDofReach-Eureka-v0;; esac
EXP=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" "logs/${LOG}.log" | tail -1 | awk '{print $NF}')
CKPT="logs/rl_games/two_dof_reach/${EXP}/nn/$(ls logs/rl_games/two_dof_reach/${EXP}/nn/ | grep '_ep_800_' | grep -v '^_' | head -1)"
echo "[${RUN}] checkpoint: $CKPT"
for VARIANT in DRtrain DRshift; do
  case $VARIANT in
    DRtrain) DRARGS="--dr --friction_range 0.0 0.25 --damping_scale_range 0.5 4.0 --effectiveness_range 0.5 1.0 --mass_scale_range 1.0 1.5" ;;
    DRshift) DRARGS="--dr --friction_range 0.25 0.5 --damping_scale_range 4.0 8.0 --effectiveness_range 0.3 0.5 --mass_scale_range 1.5 2.5" ;;
  esac
  $PY -u scripts/eval.py --task "$TASK" --checkpoint "$CKPT" --run_name "${RUN}_${VARIANT}" \
    --num_envs 4096 --headless $DRARGS --note "matrix fill: ${RUN} ${VARIANT}" > "logs/eval_${RUN}_${VARIANT}.log" 2>&1
  sed -n "/EVAL RESULT/,/^====/p" "logs/eval_${RUN}_${VARIANT}.log"
done
