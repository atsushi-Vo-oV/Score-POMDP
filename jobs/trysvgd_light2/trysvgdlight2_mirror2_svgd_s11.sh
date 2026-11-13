#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light2 / mirror-world Hike (w=4, outside -5) / score transformer SVGD with the directly parameterised score / seed 11.
# Default v2 of the light TF-SVGD (2026-11-14): the chain starts from U(-sqrt(3), sqrt(3)) at every step including t = 0,
# no observation anchor, no bound on the direct score and no -x term at t = 0; everything else as try-svgd-light.
# 333 updates.  Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight2_mirror2_tf-svgd.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgdlight2-mirror2-tf-svgd-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
