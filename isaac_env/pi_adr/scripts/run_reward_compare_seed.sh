#!/usr/bin/env bash
# Multi-seed check of the Eureka-vs-vanilla gap, NOMINAL physics only (no DR).
# Uses today's 800-iter/2048-env protocol for speed -- NOT a reproduction of
# Session 1's 5000/1500-iter headline numbers at multi-seed (that would be the
# more expensive, exact check; this is the faster, still-informative one,
# explicitly logged as using a smaller budget).
# Usage: run_reward_compare_seed.sh <seed>
set -uo pipefail
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
PY=$ROOT/env_isaaclab/bin/python
cd "$ROOT/pi_adr"
SEED=$1

for TASK_TAG in "PIADR-TwoDofReach-Vanilla-v0:vanilla" "PIADR-TwoDofReach-Eureka-v0:eureka"; do
  TASK="${TASK_TAG%%:*}"
  TAG="${TASK_TAG##*:}"
  RUN="${TAG}_seed${SEED}_800iter"

  $PY -u scripts/train.py --task "$TASK" --headless \
    --num_envs 2048 --max_iterations 800 --seed "$SEED" \
    > "logs/train_${RUN}.log" 2>&1

  EXP=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" "logs/train_${RUN}.log" | tail -1 | awk '{print $NF}')
  CKPT="logs/rl_games/two_dof_reach/${EXP}/nn/$(ls logs/rl_games/two_dof_reach/${EXP}/nn/ | grep '_ep_800_' | grep -v '^_' | head -1)"
  echo "[${RUN}] checkpoint: $CKPT"

  $PY -u scripts/eval.py --task "$TASK" --checkpoint "$CKPT" \
    --run_name "${RUN}_nominal" --num_envs 4096 --headless \
    --note "multi-seed reward comparison, reward=${TAG}, seed=${SEED}, 800 iters (reduced budget, see script header)" \
    > "logs/eval_${RUN}_nominal.log" 2>&1
  sed -n "/EVAL RESULT/,/^====/p" "logs/eval_${RUN}_nominal.log"
done
