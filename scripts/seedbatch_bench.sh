#!/usr/bin/env bash
# From the repository root. Does not submit jobs or contact another host.
set -euo pipefail
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
"${SB_POMDP_PYTHON:-python3.11}" -m sb_pomdp.seedbatch.benchmark \
  --config "${SB_POMDP_CONFIG:-config/trysvgdlight_mirror2_tf-svgd.json}" \
  --device "${SB_POMDP_DEVICE:-cuda}" --counts 1 2 5 \
  --repeats "${SB_POMDP_BENCH_REPEATS:-3}" \
  --json-output "${SB_POMDP_BENCH_REPORT:-results/seedbatch-bench/benchmark.json}" "$@"
