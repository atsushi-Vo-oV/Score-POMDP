#!/bin/bash
# Submit existing job files on the MIG queue (1/7 of the full-GPU point rate)
# without editing them: command-line -L options override the #PJM directives.
# Usage: bash jobs/mig/pjsub_mig.sh jobs/<set>/<shard>.sh [more job files...]
# Verify with pjstat: a MIG allocation shows CORE 4 (a full GPU shows 30).
set -euo pipefail
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <job.sh> [...]" >&2
  exit 1
fi
for job in "$@"; do
  pjsub -L rscgrp=b-batch-mig -L gpu=1 "$job"
done
