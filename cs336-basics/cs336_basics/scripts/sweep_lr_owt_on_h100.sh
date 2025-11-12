#!/bin/bash

set -euo pipefail

# Note: BASE_DIR is the script path; change to dirname if you want the script directory.
BASE_DIR="$1"
EXP_NAME="sweep_owt_on_h100"

# Input data.
TRAINING_DATASET_PATH="${BASE_DIR}/data/owt_train_tokens.npy"
VALIDATION_DATASET_PATH="${BASE_DIR}/data/owt_valid_tokens.npy"

# Configs of the transformer (placeholders - edit as needed).
VOCAB_SIZE=32000
MAX_CONTEXT_LENGTH=1024
NUM_LAYERS=12
NUM_HEADS=32
ROPE_THETA=10000.0
D_MODEL=1024
D_FF=2752

# Optimizer / LR configs (placeholders).
WEIGHT_DECAY=0.001
ADAMW_BETA_1=0.9
ADAMW_BETA_2=0.999
ADAMW_EPS=1e-8

# WANDB (placeholders)
WANDB_ENTITY="fm966hz"
WANDB_PROJECT="cs336-assignment-1-owt-params-sweep"
WANDB_SWEEP_METHOD="bayes"
WANDB_SWEEP_NAME="${EXP_NAME}"

# Training config placeholders
NUM_STEPS=1000
BATCH_SIZE=12
VALIDATION_BATCH_SIZE=64
VALIDATION_FREQ=25
DEVICE="cuda:0"

# Gradient clipping
MAX_TOTAL_GRADIENT_L2_NORM=4.0

# Misc
LOG_METRICS_TO_CONSOLE=false

TRAIN_CMD="uv run cs336_basics/parameter_sweeps_main.py"

$TRAIN_CMD  \
	--training_dataset_path="${TRAINING_DATASET_PATH}" \
	--validation_dataset_path="${VALIDATION_DATASET_PATH}" \
	--vocab_size=${VOCAB_SIZE} \
	--max_context_length=${MAX_CONTEXT_LENGTH} \
	--num_layers=${NUM_LAYERS} \
	--num_heads=${NUM_HEADS} \
	--rope_theta=${ROPE_THETA} \
	--d_model=${D_MODEL} \
	--d_ff_to_d_model=${D_FF_TO_D_MODEL} \
    --d_ff=${D_FF} \
	--weight_decay=${WEIGHT_DECAY} \
	--adamw_beta_1=${ADAMW_BETA_1} \
	--adamw_beta_2=${ADAMW_BETA_2} \
	--adamw_eps=${ADAMW_EPS} \
	--wandb_entity="${WANDB_ENTITY}" \
	--wandb_project="${WANDB_PROJECT}" \
	--wandb_sweep_method="${WANDB_SWEEP_METHOD}" \
    --wandb_sweep_name="${WANDB_SWEEP_NAME}" \
	--num_steps=${NUM_STEPS} \
	--batch_size=${BATCH_SIZE} \
	--validation_batch_size=${VALIDATION_BATCH_SIZE} \
	--validation_freq=${VALIDATION_FREQ} \
	--device="${DEVICE}" \
	--log_metrics_to_console=${LOG_METRICS_TO_CONSOLE} \
	--max_total_gradient_l2_norm=${MAX_TOTAL_GRADIENT_L2_NORM}

