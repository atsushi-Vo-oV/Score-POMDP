#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S

# try-svgd-light / mirror-world Hike (w=4, outside -5) / score transformer SVGD with the directly parameterised score / seed 14.
# prev_init_u0: chain start = previous particles (langevin_warm_start, no transition proposal); the t = 0 start is the
# observation anchor applied to U(-sqrt(3), sqrt(3)) noise instead of N(0, I) (model.langevin_initial_noise=uniform).
# -JOINT: one network maps (o, x, a, attention summary of the previous particles) to the score;
# otherwise the observation term and the attention transition term are separate networks, summed.
# As try-svgd (no warm start, learned alpha, prediction phases, diffusion PPO policy) except that the
# networks output the score itself instead of a scalar energy differentiated by autograd.
# 333 updates.  Segmented: resubmit this same file to resume from latest.pt.
set -euo pipefail
export SB_POMDP_CONFIG="config/trysvgdlight_mirror2_tf-svgd-prev-init-u0.json"
export SB_POMDP_SHARD_INDEX=5
export SB_POMDP_CAMPAIGN_ID="trysvgdlight-mirror2-tf-svgd-prev-init-u0-v1"
export SB_POMDP_SEGMENT_UPDATES=333
exec bash jobs/genkai_production.sh
