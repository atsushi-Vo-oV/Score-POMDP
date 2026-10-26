#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd / disk tag (3 ghosts with 180-degree view, player cone 60 degrees) / score transformer SMI / seed 14.
# 573 updates = 98 random-action prediction + 400 PPO + 75 interleaved prediction.
# Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgd_tag_tf-smi.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgd-tag-tf-smi-v1"
export SB_POMDP_SEGMENT_UPDATES=573
exec bash jobs/genkai_production.sh
