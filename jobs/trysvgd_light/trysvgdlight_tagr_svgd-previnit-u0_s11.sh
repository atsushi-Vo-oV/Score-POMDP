#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / disk tag, range observations (noisy distance to every ghost, no view cone) / score transformer SVGD with the direct score / seed 11.
# prev_init_u0: chain start = previous particles (langevin_warm_start, no transition proposal); the t = 0 start is the
# observation anchor applied to U(-sqrt(3), sqrt(3)) noise instead of N(0, I) (model.langevin_initial_noise=uniform).
# 1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_tagr_tf-svgd-prev-init-u0.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-tagr-tf-svgd-prev-init-u0-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
