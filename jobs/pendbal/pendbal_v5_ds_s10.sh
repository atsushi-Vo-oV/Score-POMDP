#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=96:00:00
#PJM -j
#PJM -S

# Masked pendulum BALANCING variant (start within +-0.5 rad of upright, speed +-1 rad/s, velocity
# hidden, 200 steps) with the PPO settings that solve the pendulum with full state (gamma 0.9, lr 1e-3, 10 epochs, minibatch 128, entropy 0, raw value targets,
# GAE 0.9; the original settings never leave the zero-torque level).
# v5 / score_deepsets / seed 10, 200 updates. v2 with the set-level mixture-density aux predictor.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/pendbal_v5.json"
export SB_POMDP_SHARD_INDEX=11
export SB_POMDP_CAMPAIGN_ID="pendbal-v5-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
