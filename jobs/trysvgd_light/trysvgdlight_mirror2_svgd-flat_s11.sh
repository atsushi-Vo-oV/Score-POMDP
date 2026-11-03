#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / mirror-world Hike (w=4, outside -5) / score transformer SVGD-JOINT with the directly parameterised score / seed 11.
# -FLAT (joint without structure): one MLP maps (o, x, a, x_{t-1,1..K} in stored order) to the score;
# no attention, no permutation invariance; hidden width matched to the joint arm's per-query MACs.
# As try-svgd (no warm start, learned alpha, prediction phases, diffusion PPO policy) except that the
# networks output the score itself instead of a scalar energy differentiated by autograd.
# 333 updates.  Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_mirror2_tf-svgd-flat.json"
export SB_POMDP_SHARD_INDEX=2
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-mirror2-tf-svgd-flat-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
