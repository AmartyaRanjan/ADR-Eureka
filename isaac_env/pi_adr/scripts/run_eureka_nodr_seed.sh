#!/usr/bin/env bash
# Rung B at the 800-iter protocol (Eureka reward, no DR), nominal eval only; DR evals via eval_dr_for_run.sh.
set -uo pipefail
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env; PY=$ROOT/env_isaaclab/bin/python; cd "$ROOT/pi_adr"
SEED=$1; RUN=eureka_seed${SEED}_800iter
$PY -u scripts/train.py --task PIADR-TwoDofReach-Eureka-v0 --headless --num_envs 2048 --max_iterations 800 --seed "$SEED" > "logs/train_${RUN}.log" 2>&1
bash scripts/eval_dr_for_run.sh "$RUN" eureka
EXP=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" "logs/train_${RUN}.log" | tail -1 | awk '{print $NF}')
CKPT="logs/rl_games/two_dof_reach/${EXP}/nn/$(ls logs/rl_games/two_dof_reach/${EXP}/nn/ | grep '_ep_800_' | grep -v '^_' | head -1)"
$PY -u scripts/eval.py --task PIADR-TwoDofReach-Eureka-v0 --checkpoint "$CKPT" --run_name "${RUN}_nominal" --num_envs 4096 --headless --note "rung B seed ${SEED}" > "logs/eval_${RUN}_nominal.log" 2>&1
