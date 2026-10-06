#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=72:00:00
#PJM -j
#PJM -S

# Masked pendulum BALANCING variant (start within +-0.5 rad of upright, speed +-1 rad/s, velocity
# hidden, 200 steps) with the PPO settings that solve the pendulum with full state (gamma 0.9, lr 1e-3, 10 epochs, minibatch 128, entropy 0, raw value targets,
# GAE 0.9; the original settings never leave the zero-torque level).
# v6lf / score_alpha / seed 10, 200 updates. v3 with the set-level mixture-density aux predictor. Score chain K=4, L=2, energy [32,32] (FLOPs-matched to the baselines).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/pendbal_v6lf.json"
export SB_POMDP_SHARD_INDEX=6
export SB_POMDP_CAMPAIGN_ID="pendbal-v6lf-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
