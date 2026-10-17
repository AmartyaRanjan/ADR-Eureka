#!/usr/bin/env bash
# E2 (pre-registered): evaluate the 9 LEAP checkpoints at frozen ADR 5,10,15,20 (sequential; retries on transient sim errors)
set -u
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env; PI=$ROOT/pi_adr; LEAP=$ROOT/LEAP_Hand_Isaac_Lab; PY=$ROOT/env_isaaclab/bin/python
export HOME=/home/amartyaranjan888 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONPATH=$LEAP/source/LEAP_Isaaclab
exec 9>"$PI/logs/E2.lock"; flock -n 9 || exit 0
cd "$LEAP"
for A in 10 15 5 20; do for c in L0 L1 L2; do for s in 42 123 7; do
  RUN=LEAP_${c}_s${s}_adr${A}
  [ -f "$PI/results/${RUN}.json" ] && continue
  CK=$(cat "$PI/results/LEAP_${c}_s${s}_checkpoint.txt")
  for try in 1 2 3; do
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 5000 ]; do sleep 60; done
    $PY -u "$PI/scripts/eval_leap.py" --headless --checkpoint "$CK" --run_name "$RUN" --adr_increments $A --note "E2 robustness curve ADR=$A" > "$PI/logs/eval_${RUN}.log" 2>&1
    [ -f "$PI/results/${RUN}.json" ] && break; echo "[$(date '+%F %T')] retry $try $RUN"; sleep 30
  done
  echo "[$(date '+%F %T')] done $RUN"
done; done; done
echo "[E2] ALL DONE"
