#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# one-step belief gradients / disk tag with range observations, ghosts at 0.5-0.7 of the player's top speed / light TF-SVGD, v2 default / seed 13.
# Stored one-step replay for the PPO update, full backpropagation through time for the prediction phases (experiment 28).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_tagrs_tf-svgd-stored-predfull.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-tagrs-tf-svgd-stored-predfull-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
