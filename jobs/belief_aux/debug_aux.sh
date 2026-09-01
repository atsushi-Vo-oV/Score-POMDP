#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# DEBUG gate for predictive-sufficiency belief supervision
# (model.observation_prediction_coef): one update per score-family method on
# the demo-seeded WML config (light_dark) plus one PPO CartPole arm, so both
# update paths execute the auxiliary loss on the GPU. MIG slice.
# Submit from the repository root: pjsub jobs/belief_aux/debug_aux.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for METHOD in score_transformer score_alpha score_deepsets; do
  echo "=== debug aux arm: $METHOD (light_dark demo-WML) ==="
  python3.11 -m sb_pomdp.compare \
    --config config/ld2d_p3o_wml_demo_aux.json \
    --gradient-modes full \
    --methods "$METHOD" \
    --tasks light_dark \
    --seeds 0 \
    --override "experiment.seeds=[0]" \
    --override experiment.eval_episodes=1 \
    --override ppo.num_envs=16 \
    --override ppo.minibatch_size=480 \
    --override ppo.sequence_microbatch_size=120 \
    --max-updates 1 \
    --label "aux-debug-$STAMP-$METHOD"
done
echo "=== debug aux arm: cartpole (PPO) ==="
python3.11 -m sb_pomdp.compare \
  --config config/two_task_full_v2.json \
  --gradient-modes full \
  --methods score_transformer \
  --tasks masked_cartpole \
  --seeds 0 \
  --override "experiment.seeds=[0]" \
  --override experiment.eval_episodes=1 \
  --override ppo.num_envs=4 \
  --override ppo.epochs=1 \
  --override model.observation_prediction_coef=1.0 \
  --max-updates 1 \
  --label "aux-debug-$STAMP-cartpole"
echo "aux debug completed all arms."
