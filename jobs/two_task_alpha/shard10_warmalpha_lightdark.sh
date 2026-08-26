#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# full / score_alpha (warm+tau0.1+learned alpha_l) / light_dark / seed 10
# alpha-vector (PWLC head over pooled particles) vs transformer comparison,
# protocol-identical to two-task-full-v2 (200 updates, lr 2.5e-4, entropy
# 0.01, max_grad_norm 10, rescaled light_dark rewards). The cold/transformer
# reference is the completed v2 score shard. Segmented checkpoints: on a
# wall-time stop resubmit this same file to resume from latest.pt.
# Submit from the repository root AFTER debug_alpha.sh has passed:
#   pjsub jobs/two_task_alpha/shard10_warmalpha_lightdark.sh

set -euo pipefail
export SB_POMDP_CONFIG="config/two_task_alpha_warm.json"
export SB_POMDP_SHARD_INDEX=10
export SB_POMDP_CAMPAIGN_ID="two-task-alpha-warm-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
