#!/usr/bin/env bash
# E5 (pre-registered, project_docs/19): evaluate A (vanilla) and B (Eureka) checkpoints at iterations 100..800. Waits for E4step0 to finish first.
set -u
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env; PI=$ROOT/pi_adr; PY=$ROOT/env_isaaclab/bin/python
export HOME=/home/amartyaranjan888
exec 9>"$PI/logs/E5.lock"; flock -n 9 || exit 0
tail -n 1 "$PI/logs/E4step0_quality.log" 2>/dev/null | grep -q "ALL DONE" || exit 0
cd "$PI"
trainlog() { # rung seed
  case $1 in
    A) [ "$2" -eq 42 -o "$2" -eq 123 -o "$2" -eq 7 ] && echo "logs/train_vanilla_seed$2_800iter.log" || echo "logs/train_vanilla_ent0.005_seed$2_800iter.log";;
    B) echo "logs/train_eureka_seed$2_800iter.log";;
  esac; }
for N in 100 200 400 800 300 600; do for R in A B; do for S in 42 123 7 1 2; do
  RUN=E5_${R}_seed${S}_it${N}_nominal
  [ -f "results/${RUN}.json" ] && continue
  EXP=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" "$(trainlog $R $S)" | tail -1 | awk '{print $NF}')
  NN="logs/rl_games/two_dof_reach/$EXP/nn"
  F=$(ls "$NN" | grep "_ep_${N}_" | grep -v '^last_' | head -1); [ -z "$F" ] && F=$(ls "$NN" | grep "_ep_${N}_" | head -1)
  [ -z "$F" ] && { echo "no checkpoint $R $S $N"; continue; }
  [ $R = A ] && TASK=PIADR-TwoDofReach-Vanilla-v0 || TASK=PIADR-TwoDofReach-Eureka-v0
  for try in 1 2 3 4; do
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 3500 ]; do sleep 60; done
    $PY -u scripts/eval.py --task $TASK --checkpoint "$NN/$F" --run_name "$RUN" --num_envs 4096 --headless --note "E5 sample efficiency $R seed $S iter $N" > "logs/eval_${RUN}.log" 2>&1
    [ -f "results/${RUN}.json" ] && python3 scripts/valid_eval.py "results/${RUN}.json" && break; rm -f "results/${RUN}.json"; echo "[$(date '+%F %T')] retry $try $RUN"; sleep $((20 + RANDOM % 40))
  done
  echo "[$(date '+%F %T')] done $RUN"
done; done; done
echo "[E5] ALL DONE"
