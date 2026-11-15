#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# tagresn (slow elliptic tag, noisy ranges) / GRU baseline / seed 14: as tagres (ghosts at 0.5-0.7 of the
# player's top speed), with the range noise doubled to std = 0.10 + 0.40 d (experiment 26).
# 2420 updates = 25 random-action prediction + 2000 PPO + 395 interleaved prediction (64 environments, 64 particles).
# Segmented: resubmit this same file (or run_shard.sh on the math server) to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_tagresn_gru.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgd-tagresn-gru-env64-k64-v1"
export SB_POMDP_SEGMENT_UPDATES=2420
exec bash jobs/genkai_production.sh
