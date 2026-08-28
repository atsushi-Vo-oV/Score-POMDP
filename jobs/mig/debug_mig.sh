#!/bin/bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S

# MIG feasibility gate (1/7 of a full-GPU point rate). Earlier score jobs on
# b-batch-mig died with "NVML_SUCCESS == r INTERNAL ASSERT FAILED
# (CUDACachingAllocator.cpp:844)", a PyTorch caching-allocator/NVML issue on
# MIG devices. This job reproduces the plain run and then tries the
# cudaMallocAsync allocator backend, which bypasses that code path, and
# records wall time per arm so MIG throughput can be compared with the full
# GPU (alpha/p3o debug: ~40 s per light_dark update).
# Submit from the repository root: pjsub jobs/mig/debug_mig.sh

set -uo pipefail
PROJECT_ROOT="${PJM_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.2.2 cudnn/8.9.7 nccl/2.22.3 pytorch-cuda/2.3.1-12.2.2
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$(nproc)"

echo "=== environment ==="
echo "nproc=$(nproc)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
env | grep -E "^PJM_(RSCGRP|NODE|GPU|MPI)" || true
nvidia-smi -L || true
python3.11 -c "import torch; print('torch', torch.__version__, 'device', torch.cuda.get_device_name(0), 'mem_GiB', torch.cuda.get_device_properties(0).total_memory/2**30)"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
run_arm () {
  ARM="$1"; METHOD="$2"; shift 2
  echo "=== mig arm: $ARM (method=$METHOD, PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-default}) ==="
  START=$(date +%s)
  python3.11 -m sb_pomdp.compare \
    --config config/ld2d_p3o.json \
    --gradient-modes full \
    --methods "$METHOD" \
    --tasks light_dark \
    --seeds 0 \
    --override "experiment.seeds=[0]" \
    --override experiment.eval_episodes=1 \
    --override ppo.num_envs=4 \
    --override ppo.epochs=1 \
    --max-updates 1 \
    --label "mig-debug-$STAMP-$ARM" "$@"
  STATUS=$?
  echo "=== mig arm $ARM finished: exit=$STATUS wall=$(( $(date +%s) - START ))s ==="
}

# A: plain caching allocator (expected to reproduce the NVML assertion)
run_arm plain_score score_transformer
# B: cudaMallocAsync backend (candidate fix)
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync run_arm async_score score_transformer
# C: baseline path under the same fix
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync run_arm async_gru gru
echo "mig debug finished (see per-arm exit codes above)."
