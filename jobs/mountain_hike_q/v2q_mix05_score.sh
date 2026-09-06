#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=36:00:00
#PJM -j
#PJM -S

# Mountain Hike / action-value critic study / v2q_mix05 / score_transformer / seed 10, 200 updates.
# v2q with the Q-advantage mixed 50/50 with GAE (Q-Prop/IPG style)
# Segmented: resubmit this same file to resume from latest.pt.
# Submit from the repository root AFTER debug_q.sh has passed: pjsub jobs/mountain_hike_q/v2q_mix05_score.sh
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_v2q_mix05.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v2q-mix05-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
