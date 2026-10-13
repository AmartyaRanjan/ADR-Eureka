#!/usr/bin/env bash
# Resumable, sequential LEAP Hand queue (Track B). Safe to start repeatedly (flock) and to kill/reboot:
# a job is skipped when its final result file exists; an interrupted job restarts from scratch.
# Conditions: L0 = native ADR off; L1 = native ADR (upstream default); L2 = native ADR warm-started at 8 increments
# (proxy for "start at the difficulty ADR reached"; NOT a RAPP-derived envelope -- no LEAP RAPP sweep was run).
# score_to_win (upstream 100000) is disabled so EVERY condition trains the full 500 iterations (L0 otherwise stops early at ~270-380 epochs: "Maximum reward achieved").
# Every checkpoint is evaluated with ADR FROZEN at 0 (nominal) and 25 (full native DR) increments.
set -u
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
PI=$ROOT/pi_adr; LEAP=$ROOT/LEAP_Hand_Isaac_Lab; PY=$ROOT/env_isaaclab/bin/python
export HOME=/home/amartyaranjan888 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONPATH=$LEAP/source/LEAP_Isaaclab
exec 9>"$PI/logs/leap_queue.lock"; flock -n 9 || exit 0
STATUS=$PI/logs/leap_queue_status.log
NEED_MIB=${NEED_MIB:-12000}   # free GPU memory required before launching a LEAP job (shared GPU)
NEED_RAM_MIB=${NEED_RAM_MIB:-10000}  # host RAM a LEAP job needs (~17 GB RSS observed); shared host, kernel OOM kills are silent
wait_gpu() {
  local n=0
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    ram=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    [ "${free:-0}" -ge "$NEED_MIB" ] && [ "${ram:-0}" -ge "$NEED_RAM_MIB" ] && return 0
    [ $((n % 20)) -eq 0 ] && log "waiting for resources (GPU free ${free}/${NEED_MIB} MiB, host RAM avail ${ram}/${NEED_RAM_MIB} MiB)"
    n=$((n+1)); sleep 60
  done
}
log() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

run_eval() { # ckpt name inc
  [ -f "$PI/results/$2.json" ] && return 0
  (cd "$LEAP" && $PY -u "$PI/scripts/eval_leap.py" --headless --checkpoint "$1" --run_name "$2" --adr_increments "$3" \
     --note "LEAP matrix, frozen ADR=$3" > "$PI/logs/eval_$2.log" 2>&1)
}
job() { # cond seed
  local cond=$1 seed=$2 run=LEAP_$1_s$2 extra=""
  [ -f "$PI/results/${run}_fullDR.json" ] && { log "skip $run (done)"; return; }
  case $cond in L0) extra="env.enable_adr=false";; L1) extra="";; L2) extra="env.starting_adr_increments=8";; esac
  local ckpt=""
  [ -f "$PI/results/${run}_checkpoint.txt" ] && ckpt=$(cat "$PI/results/${run}_checkpoint.txt") && [ -f "$ckpt" ] || ckpt=""
  if [ -n "$ckpt" ]; then log "reuse trained checkpoint for $run: $(basename "$ckpt")"
  elif [ "$run" = "LEAP_L1_s42" ]; then   # existing upstream-config 500-iter run (seed 42, ADR on), reuse
    ckpt=$(ls "$LEAP"/logs/rl_games/leap_hand_reorient/2026-09-16_03-32-28/nn/*ep_500_* | head -1)
  else
    local attempt exp=""; local rv="RESUME_${run}"; local resume="${!rv:-}"
    for attempt in 1 2 3 4 5 6; do
      wait_gpu; log "train $run (attempt $attempt${resume:+, resuming from $(basename "$resume")})"
      local alog="$PI/logs/train_${run}_a${attempt}.log"
      (cd "$LEAP" && timeout 10800 $PY -u "$PI/scripts/train.py" --task Isaac-Reorient-Cube-Leap --headless --num_envs 4096 \
         --max_iterations 500 --seed "$seed" ${resume:+--checkpoint "$resume"} $extra agent.params.config.save_frequency=50 agent.params.config.score_to_win=1000000000 > "$alog" 2>&1)
      rc=$?
      exp=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" "$alog" | tail -1 | awk '{print $NF}')
      ckpt=$(ls "$LEAP/logs/rl_games/leap_hand_reorient/$exp/nn/"*ep_500_* 2>/dev/null | head -1)
      [ -n "$ckpt" ] && break
      last=$(for f in "$LEAP/logs/rl_games/leap_hand_reorient/$exp/nn/"*.pth; do [ -f "$f" ] && echo "$(echo "$f" | grep -oE 'ep_[0-9]+' | tail -1 | cut -d_ -f2) $f"; done | sort -n | tail -1 | cut -d' ' -f2-)
      [ -n "$last" ] && resume=$last
      log "train $run attempt $attempt ended rc=$rc epoch=$(grep -oE 'epoch: [0-9]+/500' "$alog" | tail -1) cause: $(grep -E 'out of memory|Killed|Error' "$alog" | tail -1 | cut -c1-120); next resume=$(basename "${resume:-none}"); retry in 5 min"
      sleep 300
    done
  fi
  if [ -z "$ckpt" ]; then log "GAVE UP on $run after retries"; return; fi
  echo "$ckpt" > "$PI/results/${run}_checkpoint.txt"
  NEED_RAM_MIB=3000 wait_gpu; run_eval "$ckpt" "${run}_nominal" 0; run_eval "$ckpt" "${run}_fullDR" 25
  log "done $run"
}
log "queue start (pid $$)"
for seed in 42 123 7; do for cond in L1 L0 L2; do job $cond $seed; done; done
log "queue finished"
