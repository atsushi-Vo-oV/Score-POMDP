#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / Car-Flag (green flag at +1 or -1, side revealed only within 0.2 of the priest at 0.5) / score transformer SVGD with the direct score / seed 14.
# uniform_init_u0: chain start U(-sqrt(3), sqrt(3)) per latent coordinate at every step; at t = 0 the observation anchor is
# applied to that uniform noise instead of N(0, I) (model.langevin_initial_noise=uniform).
# 333 updates = 98 random-action prediction + 200 PPO + 35 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_carflag_tf-svgd-uniform-init-u0.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-carflag-tf-svgd-uniform-init-u0-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
