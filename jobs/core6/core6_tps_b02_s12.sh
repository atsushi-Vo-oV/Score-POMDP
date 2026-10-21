#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=72:00:00
#PJM -j
#PJM -S

# core6 / Hike |s| observation + teleport 0.03 / score transformer k16f, alpha 0.2, beta 0.2 fixed / seed 12, 100 updates.
# Same as the core4 transformer (K=16, L=12, energy [8], d_model 32, ff 64, v2 belief) except that the
# ULA chain is x + alpha_l * score + beta_l * xi with alpha and beta set independently (core4/core5 had
# beta = sqrt(2 alpha tau)).  Estimated ~32 min/update.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/core6_tps_b02.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="core6-tps-b02-v1"
export SB_POMDP_SEGMENT_UPDATES=100
exec bash jobs/genkai_production.sh
