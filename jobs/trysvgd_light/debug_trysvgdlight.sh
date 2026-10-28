#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=03:00:00
#PJM -j
#PJM -S

# DEBUG gate for try-svgd-light (mirror2 and tps; disk tag is tuned locally first): one PPO update (4 envs, 1 epoch) preceded by one random-action prediction
# update and followed by one current-policy prediction update, for every task and model.
# Submit from the repository root: pjsub jobs/trysvgd_light/debug_trysvgdlight.sh

set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
run_arm () {
  CONFIG="$1"; METHOD="$2"; TASK="$3"; TAG="$4"
  echo "=== debug try-svgd-light arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks "$TASK"     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --override ppo.total_steps=1024     --override model.prediction_pretrain_steps=512     --override model.prediction_interval_updates=1     --override model.prediction_phase_updates=1     --label "trysvgdlight-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/trysvgdlight_mirror2_tf-ula.json score_transformer mountain_hike mirror2-ula
run_arm config/trysvgdlight_mirror2_tf-svgd.json score_transformer mountain_hike mirror2-svgd
run_arm config/trysvgdlight_mirror2_tf-smi.json score_transformer mountain_hike mirror2-smi
run_arm config/trysvgdlight_tps_tf-ula.json score_transformer mountain_hike tps-ula
run_arm config/trysvgdlight_tps_tf-svgd.json score_transformer mountain_hike tps-svgd
run_arm config/trysvgdlight_tps_tf-smi.json score_transformer mountain_hike tps-smi
echo "try-svgd-light debug completed all arms."
