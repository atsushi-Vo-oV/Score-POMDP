#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# P3O-identical light_dark task + PPO with symlog critic targets
# (ppo.value_target_transform=symlog). Rewards, GAE, and advantages stay in
# raw units; only the critic regresses symlog(return) and its prediction is
# mapped back with symexp, which removes the O(1e5-1e6) value-loss blow-up
# that collapsed every method on this task (smoke: gru 31,527 -> 9.8,
# score 7.4M -> 30.5 at identical rollouts). Runs on the MIG queue.
# Segmented: on a wall-time stop resubmit this same file.
# Submit from the repository root AFTER debug_symlog.sh has passed:
#   pjsub jobs/ld2d_p3o_symlog/warmtr_symlog.sh

set -euo pipefail
export SB_POMDP_CONFIG="config/ld2d_p3o_symlog_warm.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="ld2d-p3o-symlog-warm-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
