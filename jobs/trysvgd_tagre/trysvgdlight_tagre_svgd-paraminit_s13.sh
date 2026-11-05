#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / disk tag (tagre), elliptic range observations (body-frame distance with the sideways axis scaled by 0.6: the observation-consistent set is a heading-aligned ellipse, so looking around resolves the bearing) / score transformer SVGD with the direct score / seed 13.
# param_init: chain start xi -> mu + exp(log sigma) * xi with one learned (mu, log sigma).
# 2493 updates = 98 random-action prediction + 2000 PPO + 395 interleaved prediction (twice the tagr length).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_tagre_tf-svgd-param-init.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-tagre-tf-svgd-param-init-v1"
export SB_POMDP_SEGMENT_UPDATES=2493
exec bash jobs/genkai_production.sh
