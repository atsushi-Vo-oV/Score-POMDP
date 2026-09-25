#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S

# Masked pendulum (angular velocity hidden, angle observed as (cos, sin) + N(0, 0.05^2), 200-step swing-up)
# / v3 / score_alpha / seed 13, 200 updates. cold chain + per-particle aux losses.
# Same protocol as the Light-Dark 2D grid (symlog value targets, 200 PPO updates).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/pendulum_v3.json"
export SB_POMDP_SHARD_INDEX=9
export SB_POMDP_CAMPAIGN_ID="pendulum-v3-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
