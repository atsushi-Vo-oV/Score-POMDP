#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / disk tag, range observations (noisy distance to every ghost, no view cone) / score transformer SVGD with the direct score / seed 14.
# prior: no initial network and no observation anchor (initial_belief=prior); t = 0 is a recursive step
# over K learned prior particles with a zero previous action, and its chain starts at N(0, I) like every t >= 1.
# 1293 updates = 98 random-action prediction + 1000 PPO + 195 interleaved prediction.
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_tagr_tf-svgd-prior.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-tagr-tf-svgd-prior-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
