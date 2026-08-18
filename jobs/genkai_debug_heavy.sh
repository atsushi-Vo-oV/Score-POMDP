#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -S

set -euo pipefail

PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

if [[ -n "${PJM_BULKNUM:-}" ]]; then
  echo "Bulk submission is disabled; submit this script as one normal job." >&2
  exit 2
fi

CAMPAIGN_ID="${SB_POMDP_CAMPAIGN_ID:-}"
if (( ${#CAMPAIGN_ID} > 128 )) || \
   [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]*[A-Za-z0-9_-])?$ ]]; then
  echo "SB_POMDP_CAMPAIGN_ID must contain one safe ASCII identifier." >&2
  exit 2
fi

# Debug shard 3 is fixed to the heaviest calibration condition:
# full / score_transformer / masked_mountain_car_continuous / seed 0.
# genkai_debug.sh supplies the production module stack and changes only the
# number of updates and evaluation episodes from the production experiment.
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="$CAMPAIGN_ID"
exec bash jobs/genkai_debug.sh
