#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / mirror-world Hike (w=4, outside -5) / score transformer SVGD-JOINT with the directly parameterised score / seed 12.
# -JOINT: one network maps (o, x, a, attention summary of the previous particles) to the score;
# otherwise the observation term and the attention transition term are separate networks, summed.
# As try-svgd (no warm start, learned alpha, prediction phases, diffusion PPO policy) except that the
# networks output the score itself instead of a scalar energy differentiated by autograd.
# 333 updates.  Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_mirror2_tf-svgd-joint.json"
export SB_POMDP_SHARD_INDEX=3
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-mirror2-tf-svgd-joint-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
