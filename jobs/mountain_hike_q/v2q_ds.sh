#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=36:00:00
#PJM -j
#PJM -S

# Mountain Hike / action-value critic study / v2q / score_deepsets / seed 10, 200 updates.
# v2 (no g) with the action-value critic Q(b,a); advantage = Q - E_pi Q
# Segmented: resubmit this same file to resume from latest.pt.
# Submit from the repository root AFTER debug_q.sh has passed: pjsub jobs/mountain_hike_q/v2q_ds.sh
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_v2q.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="mountain-hike-v2q-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
