#!/usr/bin/env bash
# Restart-safe supervisor for E1 (cron every 10 min; flock prevents overlap). Lanes start >= 100 s apart.
cd /data/home/amartyaranjan888/projects/I/isaac_env/pi_adr
exec 9>logs/E6_supervisor.lock; flock -n 9 || exit 0
export HOME=/home/amartyaranjan888 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PATH=$PATH:/usr/bin:/bin
bash scripts/e6_lane.sh 0.005 >> logs/E6_lane_ent0.005.log 2>&1 &
sleep 100
bash scripts/e6_lane.sh 0.02 >> logs/E6_lane_ent0.02.log 2>&1 &
wait
