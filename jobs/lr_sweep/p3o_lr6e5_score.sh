#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# LD2D P3O-constant task (symlog) / learning rate 6.25e-5 (x1/4) / score_transformer / seed 10.
# The score arms oscillate by +-300 between checkpoints at 2.5e-4; a lower rate
# tests whether the oscillation is a step-size artefact. gru is the control.
set -euo pipefail
export SB_POMDP_CONFIG="config/ld2d_p3o_symlog_lr6e5.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="ld2d-p3o-symlog-lr6e5-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
