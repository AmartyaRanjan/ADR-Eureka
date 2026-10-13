#!/usr/bin/env bash
# Evaluate the D (static DR) and F (adaptive DR) checkpoints under three
# distributions each, using the exact frozen metric suite (scripts/eval.py),
# so these numbers sit in the same table as every Session-1 result.
set -uo pipefail

ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
PY=$ROOT/env_isaaclab/bin/python
cd "$ROOT/pi_adr"

D_CKPT=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" logs/train_D_static_dr_800.log | tail -1 | awk '{print $NF}')
D_CKPT="logs/rl_games/two_dof_reach/${D_CKPT}/nn/$(ls logs/rl_games/two_dof_reach/${D_CKPT}/nn/ | grep '_ep_800_' | head -1)"
F_CKPT=$(cat results/adaptive_dr/F_adaptive_dr_final_checkpoint.txt)

echo "D checkpoint: $D_CKPT"
echo "F checkpoint: $F_CKPT"

for TAG in D F; do
  CKPT_VAR="${TAG}_CKPT"
  CKPT="${!CKPT_VAR}"
  if [ ! -f "$CKPT" ]; then
    echo "  SKIP $TAG: checkpoint not found ($CKPT)"
    continue
  fi

  # nominal
  $PY -u scripts/eval.py --task PIADR-TwoDofReach-Eureka-v0 --checkpoint "$CKPT" \
    --run_name "${TAG}_nominal" --num_envs 4096 --headless \
    --note "rung ${TAG}, nominal physics, 800 total PPO iters, 2048 train envs" \
    > "logs/eval_${TAG}_nominal.log" 2>&1

  # in-distribution (same range D trained on / F started from)
  $PY -u scripts/eval.py --task PIADR-TwoDofReach-Eureka-v0 --checkpoint "$CKPT" \
    --run_name "${TAG}_DRtrain" --num_envs 4096 --headless --dr \
    --friction_range 0.0 0.25 --damping_scale_range 0.5 4.0 \
    --effectiveness_range 0.5 1.0 --mass_scale_range 1.0 1.5 \
    --note "rung ${TAG}, in-distribution DR" \
    > "logs/eval_${TAG}_DRtrain.log" 2>&1

  # shifted / unseen (identical to Session-1 B_eureka_1500_DRshift distribution)
  $PY -u scripts/eval.py --task PIADR-TwoDofReach-Eureka-v0 --checkpoint "$CKPT" \
    --run_name "${TAG}_DRshift" --num_envs 4096 --headless --dr \
    --friction_range 0.25 0.5 --damping_scale_range 4.0 8.0 \
    --effectiveness_range 0.3 0.5 --mass_scale_range 1.5 2.5 \
    --note "rung ${TAG}, shifted unseen DR" \
    > "logs/eval_${TAG}_DRshift.log" 2>&1

  for RUN in "${TAG}_nominal" "${TAG}_DRtrain" "${TAG}_DRshift"; do
    sed -n "/EVAL RESULT/,/^====/p" "logs/eval_${RUN}.log" 2>/dev/null
  done
done
