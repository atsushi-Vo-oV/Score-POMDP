#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# tagres (slow elliptic tag) / light TF-SVGD, v2 default / seed 13: disk tag with elliptic range observations as tagre, but the ghosts
# pursue at 0.5-0.7 of the player's top speed instead of 0.8-1.0 (experiment 25).
# 2420 updates = 25 random-action prediction + 2000 PPO + 395 interleaved prediction (64 environments, 64 particles).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_tagres_tf-svgd.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-tagres-tf-svgd-env64-k64-v1"
export SB_POMDP_SEGMENT_UPDATES=2420
exec bash jobs/genkai_production.sh
