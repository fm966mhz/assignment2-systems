#!/bin/bash

set -euo pipefail

BASE_DIR="$1"

BATCH_SIZE=4
VOCAB_SIZE=10000
MAX_CONTEXT_LENGTH=1024
PREDEFINED_MODEL_CONFIG="small"
DEVICE="cuda:0"
FORWARD_PASS_ONLY=true
RANDOM_SEED=42
NUM_BENCHMARKING_STEPS=10
NUM_WARMUP_STEPS=5
TORCH_COMPILE=false
PROFILE_OUTPUT_PATH="${BASE_DIR}/assignment2_output/profile_output_bs128_small_forward_no_torch_compile.nsys-rep"

uv run nsys profile -o ${PROFILE_OUTPUT_PATH} --python-backtrace=cuda \
    python -m cs336_systems.end_to_end_benchmarking_main \
    --random_seed=${RANDOM_SEED} \
    --batch_size=${BATCH_SIZE} \
    --vocab_size=${VOCAB_SIZE} \
    --max_context_length=${MAX_CONTEXT_LENGTH} \
    --num_benchmarking_steps=${NUM_BENCHMARKING_STEPS} \
    --num_warmup_steps=${NUM_WARMUP_STEPS} \
    --predefined_model_config=${PREDEFINED_MODEL_CONFIG} \
    --device=${DEVICE} \
    --forward_pass_only=${FORWARD_PASS_ONLY} \
    --torch_compile=${TORCH_COMPILE}