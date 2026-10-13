#!/usr/bin/env bash
set -u
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
PY=$ROOT/env_isaaclab/bin/python
while kill -0 1547535 2>/dev/null; do sleep 20; done
cd $ROOT/LEAP_Hand_Isaac_Lab
TS=$(date +%Y%m%d_%H%M%S)
PYTHONPATH=$ROOT/LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab \
  $PY -u $ROOT/pi_adr/scripts/train.py --task Isaac-Reorient-Cube-Leap --headless \
  --num_envs 4096 --max_iterations 500 --seed 42 \
  > $ROOT/pi_adr/logs/train_leap_500_$TS.log 2>&1
