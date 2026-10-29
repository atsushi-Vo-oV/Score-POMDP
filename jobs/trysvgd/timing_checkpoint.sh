#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=03:00:00
#PJM -j
#PJM -S

# Timing on the production MIG slice: full-size try-svgd updates (16 envs x 128 steps, 4 epochs) with
# the belief-replay activation checkpoint off (2 updates each for ULA / SVGD / SMI and the direct-score
# ULA) and on (1 ULA update for comparison).  Reads: phase=ppo_update seconds and gpu_peak_memory_mb.
# Submit from the repository root: pjsub jobs/trysvgd/timing_checkpoint.sh
set -euo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"
module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
run () {
  CONFIG="$1"; CKPT="$2"; UPDATES="$3"; TAG="$4"
  echo "=== timing $TAG checkpoint=$CKPT ==="
  python3.11 -m sb_pomdp.compare --config "$CONFIG" --methods score_transformer --seeds 10 --max-updates "$UPDATES" \
    --override experiment.eval_episodes=1 --override model.belief_activation_checkpoint=$CKPT \
    --label "timing-$STAMP-$TAG-ckpt$CKPT" 2>&1 | grep -E "phase=(rollout_collection|ppo_update) status=end|Model parameters|Error|failed"
  d=$(ls -d results/*timing-$STAMP-$TAG-ckpt$CKPT*/full/score_transformer/mountain_hike/seed_010 | head -1)
  python3.11 -c "import csv,sys; [print('metrics update', r['update'], 'gpu_peak_mb', r['gpu_peak_memory_mb'], 'steps_per_s', r['steps_per_second']) for r in csv.DictReader(open(sys.argv[1]))]" "$d/metrics.csv"
}
run config/trysvgd_mirror2_tf-ula.json false 2 ula
run config/trysvgd_mirror2_tf-svgd.json false 2 svgd
run config/trysvgd_mirror2_tf-smi.json false 2 smi
run config/trysvgdlight_mirror2_tf-ula.json false 2 light-ula
run config/trysvgd_mirror2_tf-ula.json true 1 ula
echo "timing completed."
