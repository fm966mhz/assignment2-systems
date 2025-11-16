#!/bin/bash

set -euo pipefail

BATCH_SIZE=4
VOCAB_SIZE=10000
MAX_CONTEXT_LENGTH=32
PREDEFINED_MODEL_CONFIG="large"
DEVICE="cuda:0"
FORWARD_PASS_ONLY=false
RANDOM_SEED=42
NUM_BENCHMARKING_STEPS=10
NUM_WARMUP_STEPS=5
TORCH_COMPILE=true
AUTOCAST=true
AUTOCAST_DTYPE="bfloat16"

uv run cs336_systems/end_to_end_benchmarking_main.py \
    --random_seed=${RANDOM_SEED} \
    --batch_size=${BATCH_SIZE} \
    --vocab_size=${VOCAB_SIZE} \
    --max_context_length=${MAX_CONTEXT_LENGTH} \
    --num_benchmarking_steps=${NUM_BENCHMARKING_STEPS} \
    --num_warmup_steps=${NUM_WARMUP_STEPS} \
    --predefined_model_config=${PREDEFINED_MODEL_CONFIG} \
    --device=${DEVICE} \
    --forward_pass_only=${FORWARD_PASS_ONLY} \
    --torch_compile=${TORCH_COMPILE} \
    --autocast=${AUTOCAST} \
    --autocast_dtype=${AUTOCAST_DTYPE}