#! /bin/bash

set -euo pipefail

BASE_DIR="$1"
BENCHMARKING_OUTPUT_PATH_PREFIX="${BASE_DIR}/assignment2_output/flash_attention_benmarking"
MAX_SEQ_LEN=65536
MAX_MODEL_DIM=128
TRITON_BENCHMARKING_REP=1000
TRITON_BENCHMARKING_WARMUP=1000

uv run cs336_systems/flash_attention_benmarking_main.py \
    --max_seq_len=${MAX_SEQ_LEN} \
    --max_model_dim=${MAX_MODEL_DIM} \
    --triton_benmarking_rep=${TRITON_BENCHMARKING_REP} \
    --triton_benmarking_warmup=${TRITON_BENCHMARKING_WARMUP} \
    --output_path_prefix=${BENCHMARKING_OUTPUT_PATH_PREFIX}