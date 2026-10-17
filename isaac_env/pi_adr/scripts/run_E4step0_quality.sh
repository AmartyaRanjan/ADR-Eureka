#!/usr/bin/env bash
# E4 step-0 (pre-registered, project_docs/18): rotations + falls on the 9 LEAP checkpoints at frozen ADR 0,5,10,15,20,25.
set -u
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env; PI=$ROOT/pi_adr; LEAP=$ROOT/LEAP_Hand_Isaac_Lab; PY=$ROOT/env_isaaclab/bin/python
export HOME=/home/amartyaranjan888 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONPATH=$LEAP/source/LEAP_Isaaclab
exec 9>"$PI/logs/E4step0.lock"; flock -n 9 || exit 0
cd "$LEAP"
for A in 0 10 15 5 20 25; do for c in L0 L1 L2; do for s in 42 123 7; do
  RUN=LEAPQ_${c}_s${s}_adr${A}
  [ -f "$PI/results/${RUN}.json" ] && continue
  CK=$(cat "$PI/results/LEAP_${c}_s${s}_checkpoint.txt")
  for try in 1 2 3 4; do
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 5000 ]; do sleep 60; done
    $PY -u "$PI/scripts/eval_leap_quality.py" --headless --checkpoint "$CK" --run_name "$RUN" --adr_increments $A --note "E4 step0 quality diagnostic ADR=$A" > "$PI/logs/eval_${RUN}.log" 2>&1
    [ -f "$PI/results/${RUN}.json" ] && python3 "$PI/scripts/valid_eval.py" "$PI/results/${RUN}.json" && break; rm -f "$PI/results/${RUN}.json"; echo "[$(date '+%F %T')] retry $try $RUN"; sleep $((20 + RANDOM % 40))
  done
  echo "[$(date '+%F %T')] done $RUN"
done; done; done
echo "[E4step0] ALL DONE"
