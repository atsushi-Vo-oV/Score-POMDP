#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# full / rnn / light_dark 2D "P3O-style" / seed 10
# The reward design follows the successful P3O light-dark setup
# (arXiv 2505.16732): NO per-step position cost (free detours), a dense
# terminal grading 10*||x_T||^2 whose belief expectation prices residual
# uncertainty directly (no rare goal event to discover), a much darker dark
# (noise_gain 5.0, floor 0.01), wider prior (initial_std 1.0), and process
# noise 0.01 so dead reckoning keeps degrading. First-order dynamics and the
# v2 PPO protocol are kept (200 updates, seed 10, full mode).
# Segmented: on a wall-time stop resubmit this same file.
# Submit from the repository root AFTER debug_p3o.sh has passed:
#   pjsub jobs/ld2d_p3o/rnn_p3o.sh

set -euo pipefail
export SB_POMDP_CONFIG="config/ld2d_p3o.json"
export SB_POMDP_SHARD_INDEX=10
export SB_POMDP_CAMPAIGN_ID="ld2d-p3o-refs-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
