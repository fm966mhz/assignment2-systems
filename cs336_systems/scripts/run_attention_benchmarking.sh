#! /bin/bash

set -euo pipefail

BASE_DIR="$1"

RANDOM_SEED=42
NUM_ITERS=10
OUTPUT_PATH_PREFIX="${BASE_DIR}/assignment2_output/basic_attention_benchmarking_jit_compiled"
COMPILE=true

uv run cs336_systems/benchmark_attention_main.py \
    --random_seed=${RANDOM_SEED} \
    --num_iters=${NUM_ITERS} \
    --output_path_prefix=${OUTPUT_PATH_PREFIX} \
    --compile=${COMPILE}