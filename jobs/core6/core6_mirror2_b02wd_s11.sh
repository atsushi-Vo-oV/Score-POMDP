#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# core6 / mirror-world Hike (w=4, outside -5) / score transformer k16f, alpha 0.2, beta 0.2 fixed, AdamW weight decay 0.01 / seed 11, 200 updates.
# Same as the core4 transformer (K=16, L=12, energy [8], d_model 32, ff 64, v2 belief) except that the
# ULA chain is x + alpha_l * score + beta_l * xi with alpha and beta set independently (core4/core5 had
# beta = sqrt(2 alpha tau)).  Estimated ~36 min/update.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/core6_mirror2_b02wd.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="core6-mirror2-b02wd-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
