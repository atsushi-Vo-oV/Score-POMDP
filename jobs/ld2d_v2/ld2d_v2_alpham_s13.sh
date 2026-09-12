#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S

# Light-Dark 2D (P3O-symlog task) / v2 / score_alpha parameter-matched / seed 13, 200 updates.
# alpha_feedforward_dim 704 -> 75,269 params (vs 35,525 default; transformer 75,286).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/ld2d_v2_alphamatch.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="ld2d-v2-alphamatch-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
