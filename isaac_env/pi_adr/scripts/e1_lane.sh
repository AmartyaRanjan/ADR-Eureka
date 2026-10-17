#!/usr/bin/env bash
# one lane = one entropy value; seeds sequential; up to 3 passes so interrupted/truncated runs are retried (resuming)
cd /data/home/amartyaranjan888/projects/I/isaac_env/pi_adr
ENT=$1
for pass in 1 2 3; do for s in 42 123 7 1 2; do bash scripts/run_E1_entropy_longrun.sh $s $ENT; done; done
echo "[lane ent=$ENT] ALL DONE"
