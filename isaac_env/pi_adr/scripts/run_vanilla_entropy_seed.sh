#!/usr/bin/env bash
# Vanilla reward, nominal physics, 800 iters / 2048 envs (matched to the reward-comparison runs).
# Usage: run_vanilla_entropy_seed.sh <seed> <entropy_coef>   (0.005 = default, 0.02 = fix)
set -uo pipefail
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
PY=$ROOT/env_isaaclab/bin/python
cd "$ROOT/pi_adr"
SEED=$1; ENT=$2
RUN="vanilla_ent${ENT}_seed${SEED}_800iter"
OV=()
[ "$ENT" != "0.005" ] && OV=("agent.params.config.entropy_coef=${ENT}")
$PY -u scripts/train.py --task PIADR-TwoDofReach-Vanilla-v0 --headless \
  --num_envs 2048 --max_iterations 800 --seed "$SEED" "${OV[@]}" > "logs/train_${RUN}.log" 2>&1
EXP=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" "logs/train_${RUN}.log" | tail -1 | awk '{print $NF}')
CKPT="logs/rl_games/two_dof_reach/${EXP}/nn/$(ls logs/rl_games/two_dof_reach/${EXP}/nn/ | grep '_ep_800_' | grep -v '^_' | head -1)"
$PY -u scripts/eval.py --task PIADR-TwoDofReach-Vanilla-v0 --checkpoint "$CKPT" \
  --run_name "${RUN}_nominal" --num_envs 4096 --headless \
  --note "n=5 entropy test, entropy_coef=${ENT}, seed=${SEED}, 800 iters" > "logs/eval_${RUN}_nominal.log" 2>&1
sed -n "/EVAL RESULT/,/^====/p" "logs/eval_${RUN}_nominal.log"
