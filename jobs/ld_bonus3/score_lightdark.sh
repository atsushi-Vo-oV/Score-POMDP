#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# full / score_transformer (cold belief) / light_dark bonus3 / seed 10
# light_dark with goal_bonus raised 1.0 -> 3.0: the task-adequacy audit
# showed information gathering is NOT optimal at bonus 1.0 (margin -0.11 for
# a scripted light-visit policy) and becomes clearly optimal at 3.0
# (margin +1.54). Everything else is the two-task-full-v2 protocol
# (200 updates, seed 10, full mode). No debug gate needed: every mechanism
# here passed its gate already; the only change is a reward constant.
# Segmented checkpoints: on a wall-time stop resubmit this same file.
# Submit from the repository root:
#   pjsub jobs/ld_bonus3/score_lightdark.sh

set -euo pipefail
export SB_POMDP_CONFIG="config/ld_bonus3_refs.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="ld-bonus3-refs-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
