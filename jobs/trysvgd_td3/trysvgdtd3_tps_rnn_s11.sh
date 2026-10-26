#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=72:00:00
#PJM -j
#PJM -S

# try-svgd-td3 / Hike |s| observation + teleport 0.03 / rnn (FLOPs-matched), TD3 / seed 11.
# Same belief models, FLOPs matching and prediction schedule as try-svgd; TD3 replaces PPO.
# 213 updates (each collects 2048 steps; TD3 updates then take 16 gradient steps of 16 x 32-step windows).
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdtd3_tps_base.json"
export SB_POMDP_SHARD_INDEX=7
export SB_POMDP_CAMPAIGN_ID="trysvgdtd3-tps-base-v1"
export SB_POMDP_SEGMENT_UPDATES=213
exec bash jobs/genkai_production.sh
