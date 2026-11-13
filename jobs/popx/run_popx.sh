#!/usr/bin/env bash
# scheduler adapter: run_popx.sh TREE JOB GPU; logs stdout, training exit status.
set -euo pipefail
exec 2>&1
[[ $# == 3 ]] || { echo 'usage: run_popx.sh TREE JOB GPU'; exit 2; }
tree=$(realpath -e -- "$1")
job=$2
gpu=$3
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a nonnegative integer'; exit 2; }
cd "$tree"
[[ "$job" == jobs/popx/*.sh && "$job" != *..* && -f "$job" ]] || { echo "Invalid population job: $job"; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
export SB_POMDP_SOURCE_ONLY=1
source "$job"
unset SB_POMDP_SOURCE_ONLY
[[ "${SB_POMDP_ENTRY:-}" == popexplore ]] || { echo 'Not a population job'; exit 2; }
exec bash jobs/popx/launch.sh
