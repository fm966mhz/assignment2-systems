#! /bin/bash

set -euo pipefail

dist_backend="gloo"
world_size=2
batch_size=32
max_context_length=256
vocab_size=5000
learning_rate=0.001
predefined_model_config="small"
warmup_steps=5
benchmarking_steps=5
flatten_before_communication=true

uv run cs336_systems/benchmark_ddp.py \
    --dist_backend=${dist_backend} \
    --world_size=${world_size} \
    --batch_size=${batch_size} \
    --max_context_length=${max_context_length} \
    --vocab_size=${vocab_size} \
    --learning_rate=${learning_rate} \
    --predefined_model_config=${predefined_model_config} \
    --warmup_steps=${warmup_steps} \
    --benchmarking_steps=${benchmarking_steps} \
    --flatten_before_communication=${flatten_before_communication}