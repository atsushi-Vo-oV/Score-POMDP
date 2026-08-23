#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -j
#PJM -S

# full / gru / masked_cartpole / seed 10
# v2 fixes over two_task_full: max_grad_norm 0.5 -> 10 (the shared global
# clip was crushing the policy step for the score method, whose energy-net
# gradients dominate the norm), and light_dark rewards rescaled
# (state/action cost 0.005, goal_bonus 1.0) so value targets are O(1-100)
# and information gathering is optimal.
# Submit from the repository root: pjsub jobs/two_task_full_v2/shard07_gru_cartpole.sh
# Short elapse request for earlier backfill scheduling; if the wall time
# runs out mid-run, resubmit this same file to resume.

set -euo pipefail
export SB_POMDP_CONFIG="config/two_task_full_v2.json"
export SB_POMDP_SHARD_INDEX=7
export SB_POMDP_CAMPAIGN_ID="two-task-full-v2"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
