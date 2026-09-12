#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=60:00:00
#PJM -j
#PJM -S

# Mountain Hike / v3 / score_alpha parameter-matched / seed 10, 200 updates.
# alpha_feedforward_dim 704 -> 75,269 params (vs 35,525 default; transformer 75,286).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_v3_alphamatch.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v3-alphamatch-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
