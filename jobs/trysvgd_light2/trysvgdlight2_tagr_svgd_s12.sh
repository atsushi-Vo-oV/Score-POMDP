#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light2 / disk tag, range observations (noisy distance to every ghost, no view cone) / score transformer SVGD with the direct score / seed 12.
# Default v2 of the light TF-SVGD (2026-11-14): the chain starts from U(-sqrt(3), sqrt(3)) at every step including t = 0,
# no observation anchor, no bound on the direct score and no -x term at t = 0; everything else as try-svgd-light.
# 1293 updates.  Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_tagr_tf-svgd.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-tagr-tf-svgd-v1"
export SB_POMDP_SEGMENT_UPDATES=1293
exec bash jobs/genkai_production.sh
