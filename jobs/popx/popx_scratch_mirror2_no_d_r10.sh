#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S
set -euo pipefail
export SB_POMDP_ENTRY=popexplore
export SB_POMDP_CONFIG=config/popx_scratch_mirror2_no_d.json
export SB_POMDP_CAMPAIGN_ID=popx-mirror2-no_d-v1
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_BULK_INDEX=1
export SB_POMDP_SEGMENT_UPDATES="${SB_POMDP_SEGMENT_UPDATES:-333}"
if [[ "${SB_POMDP_SOURCE_ONLY:-0}" != 1 ]]; then
  cd "${PJM_O_WORKDIR:-$PWD}"
  exec bash jobs/popx/launch.sh
fi
