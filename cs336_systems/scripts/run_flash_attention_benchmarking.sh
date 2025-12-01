#! /bin/bash

set -euo pipefail

BASE_DIR="$1"
BENCHMARKING_OUTPUT_PATH_PREFIX="${BASE_DIR}/assignment2_output/flash_attention_with_optimizations_benmarking"
MAX_SEQ_LEN=32768
MAX_MODEL_DIM=128
TRITON_BENCHMARKING_REP=2000
TRITON_BENCHMARKING_WARMUP=2000

uv run cs336_systems/flash_attention_benmarking_main.py \
    --max_seq_len=${MAX_SEQ_LEN} \
    --max_model_dim=${MAX_MODEL_DIM} \
    --triton_benmarking_rep=${TRITON_BENCHMARKING_REP} \
    --triton_benmarking_warmup=${TRITON_BENCHMARKING_WARMUP} \
    --output_path_prefix=${BENCHMARKING_OUTPUT_PATH_PREFIX}