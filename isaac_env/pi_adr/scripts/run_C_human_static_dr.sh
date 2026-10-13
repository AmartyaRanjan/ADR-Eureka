#!/usr/bin/env bash
# Rung C (roadmap Table 7): human (vanilla) reward + static DR.
# Mirrors rung D's protocol exactly, reward swapped for vanilla -- isolates
# whether static DR's effect (seen for Eureka in D vs B_eureka_1500) also
# applies to the human reward.
set -uo pipefail
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
PY=$ROOT/env_isaaclab/bin/python
cd "$ROOT/pi_adr"
SEED=${1:-42}
RUN=C_human_static_dr
[ "$SEED" != "42" ] && RUN=C_seed${SEED}_human_static_dr

$PY -u scripts/train.py --task PIADR-TwoDofReach-Vanilla-v0 --headless \
  --num_envs 2048 --max_iterations 800 --seed "$SEED" \
  env.domain_rand.enabled=true \
  'env.domain_rand.joint_friction_range=[0.0,0.25]' \
  'env.domain_rand.joint_damping_scale_range=[0.5,4.0]' \
  'env.domain_rand.actuator_effectiveness_range=[0.5,1.0]' \
  'env.domain_rand.link_mass_scale_range=[1.0,1.5]' \
  > "logs/train_${RUN}.log" 2>&1

EXP=$(grep -oE "Exact experiment name requested from command line: [0-9_-]+" "logs/train_${RUN}.log" | tail -1 | awk '{print $NF}')
CKPT="logs/rl_games/two_dof_reach/${EXP}/nn/$(ls logs/rl_games/two_dof_reach/${EXP}/nn/ | grep '_ep_800_' | grep -v '^_' | head -1)"
echo "[${RUN}] checkpoint: $CKPT"

for VARIANT in nominal DRtrain DRshift; do
  case $VARIANT in
    nominal) DRARGS="" ;;
    DRtrain) DRARGS="--dr --friction_range 0.0 0.25 --damping_scale_range 0.5 4.0 --effectiveness_range 0.5 1.0 --mass_scale_range 1.0 1.5" ;;
    DRshift) DRARGS="--dr --friction_range 0.25 0.5 --damping_scale_range 4.0 8.0 --effectiveness_range 0.3 0.5 --mass_scale_range 1.5 2.5" ;;
  esac
  $PY -u scripts/eval.py --task PIADR-TwoDofReach-Vanilla-v0 --checkpoint "$CKPT" \
    --run_name "${RUN}_${VARIANT}" --num_envs 4096 --headless $DRARGS \
    --note "rung C: human reward + static DR, seed=${SEED}, variant=${VARIANT}" \
    > "logs/eval_${RUN}_${VARIANT}.log" 2>&1
  sed -n "/EVAL RESULT/,/^====/p" "logs/eval_${RUN}_${VARIANT}.log"
done
