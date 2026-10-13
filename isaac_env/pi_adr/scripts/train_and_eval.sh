#!/usr/bin/env bash
# Train a 2-DOF variant then evaluate its final checkpoint with the frozen
# metric suite. Usage: train_and_eval.sh <TASK> <ITERS> <RUN_NAME> [NUM_ENVS]
set -uo pipefail

ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
PY=$ROOT/env_isaaclab/bin/python
cd "$ROOT/pi_adr"

TASK=$1
ITERS=$2
RUN_NAME=$3
NUM_ENVS=${4:-4096}
TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="logs/train_${RUN_NAME}_${TS}.log"

echo "[$(date +%H:%M:%S)] TRAIN $RUN_NAME  task=$TASK iters=$ITERS envs=$NUM_ENVS"
START=$(date +%s)
$PY -u scripts/train.py --task "$TASK" --headless \
    --num_envs "$NUM_ENVS" --max_iterations "$ITERS" --seed 42 > "$TRAIN_LOG" 2>&1
STATUS=$?
WALL=$(( $(date +%s) - START ))

if [ $STATUS -ne 0 ] || ! grep -q "MAX EPOCHS NUM" "$TRAIN_LOG"; then
  echo "[$(date +%H:%M:%S)] TRAIN FAILED ($RUN_NAME) status=$STATUS -- see $TRAIN_LOG"
  exit 1
fi
echo "[$(date +%H:%M:%S)] trained in ${WALL}s"

# Read the run directory from this run's own log. "Most recent directory" is
# wrong whenever two trainings run concurrently, which is how the first
# A_vanilla_5000 eval ended up looking in the Eureka run's folder.
EXP=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" "$TRAIN_LOG" | tail -1 | awk '{print $NF}')
RUN_DIR="logs/rl_games/two_dof_reach/${EXP}/"
CKPT=$(ls "${RUN_DIR}nn/"last_*_ep_${ITERS}_*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "[$(date +%H:%M:%S)] no checkpoint for iter $ITERS in ${RUN_DIR}nn/"
  exit 1
fi

echo "[$(date +%H:%M:%S)] EVAL $RUN_NAME  ckpt=$CKPT"
$PY -u scripts/eval.py --task "$TASK" --checkpoint "$CKPT" \
    --run_name "$RUN_NAME" --num_envs 4096 --headless \
    --note "iters=$ITERS train_envs=$NUM_ENVS train_wall_s=$WALL damping=0.05" \
    > "logs/eval_${RUN_NAME}.log" 2>&1

sed -n '/EVAL RESULT/,/^====/p' "logs/eval_${RUN_NAME}.log"
echo "[$(date +%H:%M:%S)] DONE $RUN_NAME"
