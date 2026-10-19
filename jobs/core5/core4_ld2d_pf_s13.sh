#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# core4 / Light-Dark 2D / DPFRL particle filter / seed 13, 200 updates (rerun).
# The original arm died at update 3 from a non-finite gradient norm (clip_grad_norm_ with
# error_if_nonfinite).  This tree skips such minibatches with a warning instead.
set -euo pipefail
export SB_POMDP_CONFIG="config/core4_ld2d.json"
export SB_POMDP_SHARD_INDEX=19
export SB_POMDP_CAMPAIGN_ID="core4-ld2d-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
