#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# every-step observation anchor / Hike |s| observation + teleport 0.03 / light TF-SVGD, v2 default / seed 11.
# observation anchor at every step, 12 transport stages: the chain of every step starts at mu(o_t) + sigma(o_t) * noise (experiment 27).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_tps_tf-svgd-anchor-all.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-tps-tf-svgd-anchor-all-v1"
export SB_POMDP_SEGMENT_UPDATES=213
exec bash jobs/genkai_production.sh
