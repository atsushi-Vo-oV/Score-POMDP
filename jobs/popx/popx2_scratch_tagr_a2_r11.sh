#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S
# Population exploration on the v2 default of the light TF-SVGD (uniform chain start at every step, no observation
# anchor, unbounded score without the t = 0 -x term); otherwise identical to popx_scratch_tagr_a2.
set -euo pipefail
export SB_POMDP_ENTRY=popexplore
export SB_POMDP_CONFIG=config/popx2_scratch_tagr_a2.json
export SB_POMDP_CAMPAIGN_ID=popx2-tagr-a2-v1
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_BULK_INDEX=2
export SB_POMDP_SEGMENT_UPDATES="${SB_POMDP_SEGMENT_UPDATES:-1293}"
if [[ "${SB_POMDP_SOURCE_ONLY:-0}" != 1 ]]; then
  cd "${PJM_O_WORKDIR:-$PWD}"
  exec bash jobs/popx/launch.sh
fi
