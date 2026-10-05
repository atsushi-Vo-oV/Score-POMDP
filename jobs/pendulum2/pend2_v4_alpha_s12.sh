#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=96:00:00
#PJM -j
#PJM -S

# Masked pendulum swing-up (velocity hidden, 200 steps) with the PPO settings that solve the task
# with full state (gamma 0.9, lr 1e-3, 10 epochs, minibatch 128, entropy 0, raw value targets,
# GAE 0.9; the original settings never leave the zero-torque level).
# v4 / score_alpha / seed 12, 200 updates. warm start + observation anchor, no aux loss.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/pendulum2_v4.json"
export SB_POMDP_SHARD_INDEX=8
export SB_POMDP_CAMPAIGN_ID="pendulum2-v4-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
