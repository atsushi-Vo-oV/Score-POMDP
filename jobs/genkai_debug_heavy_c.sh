#!/bin/bash
#PJM -L rscgrp=c-batch
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

# C-batch provides 14 CPU cores per H100.  The child script otherwise uses
# exactly the same production config and one-update heavy condition as the
# b-batch calibration job.
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="$CAMPAIGN_ID"
export SB_POMDP_OMP_NUM_THREADS=14
exec bash jobs/genkai_debug.sh
