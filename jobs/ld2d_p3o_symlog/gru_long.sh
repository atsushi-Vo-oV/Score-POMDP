#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# gru on the P3O-identical light_dark task with symlog critic targets,
# 600 updates (3x the standard budget). The 200-update run was still
# improving monotonically at the end (-39 -> -28.5) without visiting the
# light; this run separates "needs more budget" from "exploration never
# finds the light" (passive floor ~ -12, light-visit policy ~ -1..-3).
# Segmented: on a wall-time stop resubmit this same file.
# Submit from the repository root: pjsub jobs/ld2d_p3o_symlog/gru_long.sh

set -euo pipefail
export SB_POMDP_CONFIG="config/ld2d_p3o_symlog_long.json"
export SB_POMDP_SHARD_INDEX=7
export SB_POMDP_CAMPAIGN_ID="ld2d-p3o-symlog-long-v1"
export SB_POMDP_SEGMENT_UPDATES=600
exec bash jobs/genkai_production.sh
