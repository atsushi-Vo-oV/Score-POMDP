#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# one-step belief gradients / Hike |s| observation + teleport 0.03 / light TF-SVGD, v1 default / seed 13.
# One-step gradients from the stored incoming particles, all rows at once, for the PPO update and the prediction phases (experiment 28).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_tps_tf-svgd-stored.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-tps-tf-svgd-stored-v1"
export SB_POMDP_SEGMENT_UPDATES=213
exec bash jobs/genkai_production.sh
