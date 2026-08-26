#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# full / score_deepsets (warm belief) / light_dark bonus3 / seed 10
# Middle rung of the encoder ladder: alpha_pool (linear+lse head, positions
# only) < deep_sets (free MLP after pooling, positions+scores) < transformer
# (attention). Separates "removing attention" from "removing the post-pool
# nonlinearity and the score input" if score_alpha keeps leading.
# Protocol identical to the other ld_bonus3 shards (200 updates, seed 10,
# goal_bonus 3.0). Segmented: on a wall-time stop resubmit this same file.
# Submit from the repository root:
#   pjsub jobs/ld_bonus3/warmds_lightdark.sh

set -euo pipefail
export SB_POMDP_CONFIG="config/ld_bonus3_warm.json"
export SB_POMDP_SHARD_INDEX=7
export SB_POMDP_CAMPAIGN_ID="ld-bonus3-warm-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
