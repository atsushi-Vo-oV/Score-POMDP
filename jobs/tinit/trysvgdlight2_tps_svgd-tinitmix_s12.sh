#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# transition initialisation (experiment 30) / Hike |s| observation + teleport 0.03 / light TF-SVGD, v2 default / seed 12.
# The transition network starts as 4 * (previous particle - x) per pair with attention logit -4 * sum_d z_d tanh(z_d / 2), every query is pulled to the previous particles near it (pair units);
# nothing is added to the score and the weights stay trainable.
# 213 updates.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_tps_tf-svgd-tinitmix.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-tps-tf-svgd-tinitmix-v1"
export SB_POMDP_SEGMENT_UPDATES=213
exec bash jobs/genkai_production.sh
