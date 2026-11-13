#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S
set -euo pipefail
export SB_POMDP_ENTRY=popexplore
export SB_POMDP_CONFIG=config/popx_scratch_tagr_full.json
export SB_POMDP_CAMPAIGN_ID=popx-tagr-full-v1
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_BULK_INDEX=3
export SB_POMDP_SEGMENT_UPDATES="${SB_POMDP_SEGMENT_UPDATES:-1293}"
if [[ "${SB_POMDP_SOURCE_ONLY:-0}" != 1 ]]; then
  cd "${PJM_O_WORKDIR:-$PWD}"
  exec bash jobs/popx/launch.sh
fi
