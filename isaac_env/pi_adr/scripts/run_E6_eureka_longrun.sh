#!/usr/bin/env bash
# E6 (pre-registered, project_docs/22_E6_preregistration.md): Eureka reward, no DR, 2048 envs, long run.
# Usage: run_E1_entropy_longrun.sh <seed> <entropy_coef>
# Env overrides (smoke tests only): ITERS (default 5000), EVAL_ITS, DR_ITS, RUN_PREFIX (default E1)
set -uo pipefail
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env; PY=$ROOT/env_isaaclab/bin/python; cd "$ROOT/pi_adr"
SEED=$1; ENT=$2
ITERS=${ITERS:-5000}; EVAL_ITS=${EVAL_ITS:-"200 400 800 1500 3000 5000"}; DR_ITS=${DR_ITS:-"800 5000"}
RUN="${RUN_PREFIX:-E6}_ent${ENT}_seed${SEED}"
mkdir -p results/E6
if [ -f "results/E6/${RUN}_STATUS.txt" ]; then
  if head -c 2 "results/E6/${RUN}_STATUS.txt" | grep -q "OK"; then echo "skip $RUN (done)"; exit 0; fi
  rm -f "results/E6/${RUN}_STATUS.txt"   # TRUNCATED earlier: retry (resumes from its latest checkpoint)
fi
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 60; done
OV=(); [ "$ENT" != "0.005" ] && OV=("agent.params.config.entropy_coef=${ENT}")
DIRS="results/E6/${RUN}_experiment_dirs.txt"   # every attempt's run directory, in order
RES=""
if [ -f "logs/train_${RUN}.log" ] && [ ! -f "results/E6/${RUN}_STATUS.txt" ]; then
  PREV=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" "logs/train_${RUN}.log" | tail -1 | awk '{print $NF}')
  if [ -n "$PREV" ]; then
    grep -qx "$PREV" "$DIRS" 2>/dev/null || echo "$PREV" >> "$DIRS"
    BEST=$(for f in logs/rl_games/two_dof_reach/$PREV/nn/*.pth; do [ -f "$f" ] && echo "$(echo "$f" | grep -oE 'ep_[0-9]+' | tail -1 | cut -d_ -f2) $f"; done | sort -n | tail -1)
    if [ -n "$BEST" ]; then RES=${BEST#* }; N_ATT=$(ls logs/train_${RUN}_a*.log 2>/dev/null | wc -l); mv "logs/train_${RUN}.log" "logs/train_${RUN}_a$((N_ATT+1)).log"; fi
  fi
fi
echo "[$(date '+%F %T')] train $RUN iters=$ITERS ${RES:+RESUMING from $RES}"
for try in 1 2 3 4; do
  $PY -u scripts/train.py --task PIADR-TwoDofReach-Eureka-v0 --headless --num_envs 2048 \
    --max_iterations "$ITERS" --seed "$SEED" ${RES:+--checkpoint "$RES"} "${OV[@]}" > "logs/train_${RUN}.log" 2>&1
  # transient Isaac Sim start-up failure (no training happened): wait a random bit and retry
  if grep -q "create_articulation_view" "logs/train_${RUN}.log" && ! grep -q "epoch: " "logs/train_${RUN}.log"; then
    echo "[$(date '+%F %T')] sim start-up error for $RUN (try $try); retrying"; mv "logs/train_${RUN}.log" "logs/train_${RUN}_startupfail${try}.log"; sleep $((30 + RANDOM % 60)); continue
  fi
  break
done
EXP=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" "logs/train_${RUN}.log" | tail -1 | awk '{print $NF}')
grep -qx "$EXP" "$DIRS" 2>/dev/null || echo "$EXP" >> "$DIRS"
echo "$EXP" > "results/E6/${RUN}_experiment_dir.txt"
NN="logs/rl_games/two_dof_reach/${EXP}/nn"
ck() { for d in $(tac "$DIRS"); do n="logs/rl_games/two_dof_reach/$d/nn"; f=$(ls "$n" 2>/dev/null | grep "_ep_${1}_" | grep -v '^last_' | head -1); [ -z "$f" ] && f=$(ls "$n" 2>/dev/null | grep "_ep_${1}_" | head -1); [ -n "$f" ] && { echo "$n/$f"; return; }; done; }
LAST=$(ls "$NN" | grep -oE "ep_[0-9]+" | sed 's/ep_//' | sort -n | tail -1)
if [ "$LAST" != "$ITERS" ]; then echo "[$(date '+%F %T')] ABORT $RUN: last checkpoint epoch $LAST != $ITERS (truncated run)"; echo "TRUNCATED $LAST" > "results/E6/${RUN}_STATUS.txt"; exit 1; fi
for N in $EVAL_ITS; do
  C=$(ck $N); [ -z "$C" ] && { echo "no checkpoint at $N"; continue; }
  for try in 1 2 3; do
    $PY -u scripts/eval.py --task PIADR-TwoDofReach-Eureka-v0 --checkpoint "$C" --run_name "${RUN}_it${N}_nominal" \
      --num_envs 4096 --headless --note "E6 ent=${ENT} seed=${SEED} iter=${N} nominal" > "logs/eval_${RUN}_it${N}_nominal.log" 2>&1
    [ -f "results/${RUN}_it${N}_nominal.json" ] && python3 scripts/valid_eval.py "results/${RUN}_it${N}_nominal.json" && break; rm -f "results/${RUN}_it${N}_nominal.json"; echo "eval retry $try ($N nominal; missing or invalid)"; sleep 20
  done
done
for N in $DR_ITS; do
  C=$(ck $N); [ -z "$C" ] && continue
  for try in 1 2 3; do
    $PY -u scripts/eval.py --task PIADR-TwoDofReach-Eureka-v0 --checkpoint "$C" --run_name "${RUN}_it${N}_DRshift" \
      --num_envs 4096 --headless --dr --friction_range 0.25 0.5 --damping_scale_range 4.0 8.0 \
      --effectiveness_range 0.3 0.5 --mass_scale_range 1.5 2.5 --note "E6 ent=${ENT} seed=${SEED} iter=${N} DRshift" \
      > "logs/eval_${RUN}_it${N}_DRshift.log" 2>&1
    [ -f "results/${RUN}_it${N}_DRshift.json" ] && python3 scripts/valid_eval.py "results/${RUN}_it${N}_DRshift.json" && break; rm -f "results/${RUN}_it${N}_DRshift.json"; echo "eval retry $try ($N DRshift; missing or invalid)"; sleep 20
  done
done
echo "OK${RES:+ (RESUMED from $(basename "$RES"); flag in tables)}" > "results/E6/${RUN}_STATUS.txt"; echo "[$(date '+%F %T')] done $RUN"
