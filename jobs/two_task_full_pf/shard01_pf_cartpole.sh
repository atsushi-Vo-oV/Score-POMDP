#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# full / particle_filter / masked_cartpole / seed 10
# Particle-filter baseline production shard, protocol-identical to the
# two-task-full-v2 campaign (lr 2.5e-4, entropy 0.01, max_grad_norm 10,
# rescaled light_dark rewards, 200 updates) with matched parameters
# (pf max diff vs score_transformer: 0.085%).
# Submit from the repository root AFTER debug_pf.sh has passed:
#   pjsub jobs/two_task_full_pf/shard01_pf_cartpole.sh
# Short elapse request for earlier backfill; on a wall-time stop resubmit
# this same file to resume from checkpoints/latest.pt.

set -euo pipefail
export SB_POMDP_CONFIG="config/two_task_full_pf.json"
export SB_POMDP_SHARD_INDEX=1
export SB_POMDP_CAMPAIGN_ID="two-task-full-pf-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
