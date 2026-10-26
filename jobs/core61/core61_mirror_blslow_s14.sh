#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# core6.1 / mirror-world Hike (ridge 1, corners, w=2) / score transformer k16f, alpha_l and beta_l learned from 0.2 / 0.2 with the
# schedule learning rate 0.0025 (core6 blearn used 0.025) / seed 14, 200 updates.
# Everything else is core6 blearn (K=16, L=12, energy [8], d_model 32, ff 64, v2 belief, sigmoid bound).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/core61_mirror_blslow.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="core61-mirror-blslow-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
