#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S

# Masked pendulum (angular velocity hidden, angle observed as (cos, sin) + N(0, 0.05^2), 200-step swing-up)
# / v6 / score_alpha / seed 12, 200 updates. v3 with the set-level mixture-density aux predictor.
# Same protocol as the Light-Dark 2D grid (symlog value targets, 200 PPO updates).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/pendulum_v6.json"
export SB_POMDP_SHARD_INDEX=8
export SB_POMDP_CAMPAIGN_ID="pendulum-v6-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
