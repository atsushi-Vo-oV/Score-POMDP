#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=72:00:00
#PJM -j
#PJM -S

# core6.1 / Hike |s| observation + teleport 0.03 / score transformer k16f, alpha_l and beta_l learned from 0.2 / 0.2 with the
# schedule learning rate 0.0025 (core6 blearn used 0.025) / seed 14, 100 updates.
# Everything else is core6 blearn (K=16, L=12, energy [8], d_model 32, ff 64, v2 belief, sigmoid bound).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/core61_tps_blslow.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="core61-tps-blslow-v1"
export SB_POMDP_SEGMENT_UPDATES=100
exec bash jobs/genkai_production.sh
