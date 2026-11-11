#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd KAN arms / mirror2 / score_transformer / seed 10.
# kanscore: the direct score networks are KANs (B-spline, grid 8, order 3, parameters matched to the MLP); needs a tree without the analytic-score override.
# 333 updates.  Segmented: resubmit to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_mirror2_tf-svgd-kanscore.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-mirror2-tf-svgd-kanscore-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
