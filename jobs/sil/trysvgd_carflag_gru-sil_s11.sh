#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S
set -euo pipefail
cd "${PJM_O_WORKDIR:-$PWD}"
export SB_POMDP_CONFIG=config/trysvgd_carflag_gru-sil.json
export SB_POMDP_CAMPAIGN_ID=trysvgd-carflag-gru-sil-v1
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_SEGMENT_UPDATES=333
export SB_POMDP_BULK_INDEX=2
exec bash jobs/genkai_production.sh
