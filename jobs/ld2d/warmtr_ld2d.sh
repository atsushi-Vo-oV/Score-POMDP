#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# full / score_transformer (warm) / light_dark 2D (dimension=2, goal_bonus 3.0) / seed 10
# Exploration diagnostic: in 2D an ideal passive policy hits the goal ball
# ~12% of episodes (vs 3% in 5D and zero events observed in learned runs),
# so the goal bonus becomes visible to policy-gradient learning. Model dims
# (K=16, L=4, d_model, hiddens) are identical to the 5D runs; the particle
# space follows the environment (2D) by construction. Otherwise the
# two-task-full-v2 protocol (200 updates, seed 10, full mode).
# Segmented: on a wall-time stop resubmit this same file.
# Submit from the repository root: pjsub jobs/ld2d/warmtr_ld2d.sh

set -euo pipefail
export SB_POMDP_CONFIG="config/ld2d_warm.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="ld2d-warm-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
