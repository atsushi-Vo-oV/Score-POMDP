#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# P3O-style light_dark, scale-fixed second attempt (campaign ld2d-p3o2-refs-v1).
# The first attempt (terminal_cost 10) reproduced the v1 reward-scale
# pathology: terminal spikes of O(1e4) exploded the critic (value_loss 5.6M)
# and the policy diverged to ||x_T|| ~ 30-40. This version grades with
# terminal_cost 1.0 (returns O(10), passive-vs-light margin ~1.1 preserved)
# and adds state_cost 0.001 so mid-episode runaway is no longer free.
# Otherwise identical to ld2d_p3o (index map unchanged).
# Segmented: on a wall-time stop resubmit this same file.
# Submit from the repository root: pjsub jobs/ld2d_p3o2/gru_p3o2.sh

set -euo pipefail
export SB_POMDP_CONFIG="config/ld2d_p3o2.json"
export SB_POMDP_SHARD_INDEX=7
export SB_POMDP_CAMPAIGN_ID="ld2d-p3o2-refs-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
