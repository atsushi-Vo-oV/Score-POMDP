#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# every-step observation anchor / disk tag with range observations, ghosts at 0.5-0.7 of the player's top speed / light TF-SVGD, v2 default / seed 11.
# observation anchor at every step, 1 transport stage: the chain of every step starts at mu(o_t) + sigma(o_t) * noise (experiment 27).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_tagrs_tf-svgd-anchor-all-l1.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-tagrs-tf-svgd-anchor-all-l1-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
