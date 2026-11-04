#!/usr/bin/env bash
# Run from the repository root. Output directories must not already exist.
set -euo pipefail
output_root="${1:-/opt/geospsr-runtime/variance-probability-reproduction}"
common=(--case B --cutoff 50 --grid 400 --moment-method gradient
        --moment-dtype float32 --source-dtype float32 --paired-sources
        --inverse-batch 512 --probability --shared-mean --fast-reference --query-batch 256
        --dense-batch 2048 --queries 1024 65536 --repeats 2
        --spatial-screen-queries 8192 --spatial-order 5
        --query-domain mixed --spatial-domain periodic)
for n in 2048 4096 8192; do
    sdd=()
    if [[ "$n" != 8192 ]]; then sdd=(--sdd --sdd-shared-mean --sdd-measure-max 1024); fi
    ./reproduction/run_env.sh -m reproduction.benchmark_variance_reuse \
        "${common[@]}" "${sdd[@]}" --n "$n" --variance-spatial-grids 80 96 112 128 144 160 \
        --output "$output_root/quintic-B$n"
done
./reproduction/run_env.sh -m reproduction.benchmark_variance_reuse \
    "${common[@]}" --n 16384 --variance-spatial-grids 112 128 144 160 176 \
    --sdd --sdd-shared-mean --sdd-measure-max 1024 \
    --replay-batches 32 --replay-queries 65536 --replay-budget 180 \
    --output "$output_root/shared-B16384"
