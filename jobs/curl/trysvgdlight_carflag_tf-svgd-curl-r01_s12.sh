#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S
set -euo pipefail
cd "${PJM_O_WORKDIR:-$PWD}"
export SB_POMDP_CONFIG=config/trysvgdlight_carflag_tf-svgd-curl-r01.json
export SB_POMDP_CAMPAIGN_ID=trysvgdlight-carflag-tf-svgd-curl-r01-v1
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_SEGMENT_UPDATES=333
export SB_POMDP_BULK_INDEX=3
exec bash jobs/genkai_production.sh
