#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# full / score_deepsets (cold belief) / masked_cartpole / seed 10
# Middle rung of the encoder ladder on CartPole (alpha_pool < deep_sets <
# transformer), protocol-identical to two-task-full-v2 (200 updates, no
# critic transform, so it is directly comparable with the existing gru 500,
# pf 114.6, score_transformer 40.4, rnn 34.6). Runs on the MIG queue;
# full-mode score CartPole is ~18 min/update so expect 2-3 wall-time stops -
# resubmit this same file to resume from checkpoints/latest.pt.
# Submit from the repository root: pjsub jobs/two_task_alpha/shard07_ds_cartpole.sh

set -euo pipefail
export SB_POMDP_CONFIG="config/two_task_alpha_cold.json"
export SB_POMDP_SHARD_INDEX=7
export SB_POMDP_CAMPAIGN_ID="two-task-alpha-cold-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
