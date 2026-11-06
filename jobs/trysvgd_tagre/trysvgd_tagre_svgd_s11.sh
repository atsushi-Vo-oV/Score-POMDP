#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd / disk tag (tagre), elliptic range observations (body-frame distance with the sideways axis scaled by 0.6: the observation-consistent set is a heading-aligned ellipse, so looking around resolves the bearing) / score transformer SVGD / seed 11.
# 2420 updates = 25 random-action prediction + 2000 PPO + 395 interleaved prediction (twice the tagr PPO updates, 64 environments per process = 4x the samples, 64 particles).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_tagre_tf-svgd.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgd-tagre-tf-svgd-env64-k64-v1"
export SB_POMDP_SEGMENT_UPDATES=2420
exec bash jobs/genkai_production.sh
