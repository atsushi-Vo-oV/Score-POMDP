#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# full / gru / masked_cartpole / seed 10
# Submit from the repository root: pjsub jobs/two_task_full/shard07_gru_cartpole.sh
# Fixed campaign id + segmented updates: resubmitting this exact file resumes
# from checkpoints/latest.pt; a completed shard exits 0 without touching
# artifacts. Do not edit the campaign id between submissions of the same set.

set -euo pipefail
export SB_POMDP_CONFIG="config/two_task_full.json"
export SB_POMDP_SHARD_INDEX=7
export SB_POMDP_CAMPAIGN_ID="two-task-full-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
