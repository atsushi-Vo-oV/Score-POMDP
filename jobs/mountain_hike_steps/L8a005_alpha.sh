#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# Mountain Hike / Langevin budget variant 'L8a005': langevin_steps=8, step_size=0.05
# (per-env-step particle displacement bound L*alpha*|score| = 0.40*|score| vs
# 0.08*|score| in the reference; the measured |score| is 0.3-0.5 for transformers).
# score_alpha seed 10, 50 updates (the full-mode PPO update costs ~15 min at L=4
# and scales with L, so this arm is capped; compare with mountain-hike-v1 @50).
# alpha uses the parameter-matched trunk. Segmented: resubmit to resume.
# Submit from the repository root: pjsub jobs/mountain_hike_steps/L8a005_alpha.sh
set -euo pipefail
export SB_POMDP_CONFIG="config/mountain_hike_L8a005.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="mountain-hike-L8a005-v1"
export SB_POMDP_SEGMENT_UPDATES=50
exec bash jobs/genkai_production.sh
