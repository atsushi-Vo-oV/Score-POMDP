#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=03:00:00
#PJM -j
#PJM -S

# DEBUG gate for try-svgd: one PPO update (4 envs, 1 epoch) preceded by one random-action prediction
# update and followed by one current-policy prediction update, for every task and model.
# Submit from the repository root: pjsub jobs/trysvgd/debug_trysvgd.sh

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
  echo "=== debug try-svgd arm: $TAG / $METHOD ==="
  python3.11 -m sb_pomdp.compare     --config "$CONFIG"     --gradient-modes full     --methods "$METHOD"     --tasks "$TASK"     --seeds 0     --override "experiment.seeds=[0]"     --override experiment.eval_episodes=1     --override ppo.num_envs=4     --override ppo.epochs=1     --override ppo.total_steps=1024     --override model.prediction_pretrain_steps=512     --override model.prediction_interval_updates=1     --override model.prediction_phase_updates=1     --label "trysvgd-debug-$STAMP-$TAG-$METHOD"
}
run_arm config/trysvgd_mirror2_base.json gru mountain_hike mirror2-gru
run_arm config/trysvgd_mirror2_base.json rnn mountain_hike mirror2-rnn
run_arm config/trysvgd_mirror2_base.json particle_filter mountain_hike mirror2-pf
run_arm config/trysvgd_mirror2_tf-ula.json score_transformer mountain_hike mirror2-ula
run_arm config/trysvgd_mirror2_tf-svgd.json score_transformer mountain_hike mirror2-svgd
run_arm config/trysvgd_mirror2_tf-smi.json score_transformer mountain_hike mirror2-smi
run_arm config/trysvgd_tps_base.json gru mountain_hike tps-gru
run_arm config/trysvgd_tps_base.json rnn mountain_hike tps-rnn
run_arm config/trysvgd_tps_base.json particle_filter mountain_hike tps-pf
run_arm config/trysvgd_tps_tf-ula.json score_transformer mountain_hike tps-ula
run_arm config/trysvgd_tps_tf-svgd.json score_transformer mountain_hike tps-svgd
run_arm config/trysvgd_tps_tf-smi.json score_transformer mountain_hike tps-smi
run_arm config/trysvgd_tag_base.json gru disk_tag tag-gru
run_arm config/trysvgd_tag_base.json rnn disk_tag tag-rnn
run_arm config/trysvgd_tag_base.json particle_filter disk_tag tag-pf
run_arm config/trysvgd_tag_tf-ula.json score_transformer disk_tag tag-ula
run_arm config/trysvgd_tag_tf-svgd.json score_transformer disk_tag tag-svgd
run_arm config/trysvgd_tag_tf-smi.json score_transformer disk_tag tag-smi
echo "try-svgd debug completed all arms."
