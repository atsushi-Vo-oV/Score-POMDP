#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S
set -euo pipefail
cd "${PJM_O_WORKDIR:-$PWD}"
export SB_POMDP_CONFIG=config/bcresume_tagr_s10_bcreset.json
export SB_POMDP_CAMPAIGN_ID=bcresume-tagr-bcreset-v1
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_SEGMENT_UPDATES=110
export SB_POMDP_BULK_INDEX=1
exec bash jobs/genkai_production.sh
