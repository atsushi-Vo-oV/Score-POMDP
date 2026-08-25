#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

# full / rnn / light_dark / seed 10
# Elman-RNN (gate-free tanh recurrence) production shard, protocol-identical
# to the two-task-full-v2 campaign (lr 2.5e-4, entropy 0.01, max_grad_norm 10,
# rescaled light_dark rewards, 200 updates) with matched parameters
# (rnn [149,177]/hidden 64: +0.02% vs score_transformer on this task).
# Submit from the repository root AFTER debug_rnn.sh has passed:
#   pjsub jobs/two_task_full_rnn/shard04_rnn_lightdark.sh
# Short elapse request for earlier backfill; on a wall-time stop resubmit
# this same file to resume from checkpoints/latest.pt.

set -euo pipefail
export SB_POMDP_CONFIG="config/two_task_full_rnn.json"
export SB_POMDP_SHARD_INDEX=4
export SB_POMDP_CAMPAIGN_ID="two-task-full-rnn-v1"
export SB_POMDP_SEGMENT_UPDATES=200
exec bash jobs/genkai_production.sh
