#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=36:00:00
#PJM -j
#PJM -S

# Mountain Hike / action-value critic study / q_gru / gru / seed 10, 200 updates.
# gru baseline with the action-value critic (fairness control)
# Segmented: resubmit this same file to resume from latest.pt.
# Submit from the repository root AFTER debug_q.sh has passed: pjsub jobs/mountain_hike_q/q_gru_gru.sh
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_q_gru.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="mountain-hike-q-gru-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
