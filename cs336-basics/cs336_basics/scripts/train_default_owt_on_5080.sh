#!/bin/bash

set -euo pipefail

# Note: BASE_DIR is the script path; change to dirname if you want the script directory.
BASE_DIR="$1"
EXP_NAME="default_owt_on_5080_debug_memory_test"

# Input data.
TRAINING_DATASET_PATH="${BASE_DIR}/data/owt_train_tokens.npy"
VALIDATION_DATASET_PATH="${BASE_DIR}/data/owt_valid_tokens.npy"

# Configs of the transformer (placeholders - edit as needed).
VOCAB_SIZE=32000
MAX_CONTEXT_LENGTH=512
NUM_LAYERS=10
NUM_HEADS=32
ROPE_THETA=10000.0
D_MODEL=512
D_FF=1408
# Manually using `bfloat16` seems brittle and is giving me
# """"
#   torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
#   RuntimeError: expected scalar type Float but found BFloat16
# """"
# Should try `torch.amp.autocast` first.
DTYPE="bfloat16"

# Optimizer / LR configs (placeholders).
WEIGHT_DECAY=0.001
ADAMW_BETA_1=0.9
ADAMW_BETA_2=0.999
ADAMW_EPS=1e-8
MAX_LEARNING_RATE=0.002
MIN_LEARNING_RATE=0.0001
LR_WARMUP_ITERS=1000
LR_COSINE_CYCLE_ITERS=21000

# Checkpointing (placeholders).
CHECKPOINT_DIR_PATH="${BASE_DIR}/assignment1_output/experiments/${EXP_NAME}/checkpoint"
MAX_NUM_CHECKPOINTS=5
CHECKPOINT_FREQ=20

# WANDB (placeholders)
WANDB_ENTITY="fm966hz"
WANDB_PROJECT="cs336-assignment-1"
WANDB_RUN_NAME="${EXP_NAME}"

# Training config placeholders
NUM_STEPS=22000
BATCH_SIZE=12
VALIDATION_BATCH_SIZE=64
VALIDATION_FREQ=25
DEVICE="cuda:0"

# Gradient clipping
MAX_TOTAL_GRADIENT_L2_NORM=4.0

# Misc
LOG_METRICS_TO_CONSOLE=false
TORCH_CUDA_EMPTY_CACHE=false

TRAIN_CMD="uv run cs336_basics/train_transformer_lm_main.py"

$TRAIN_CMD  \
	--training_dataset_path="${TRAINING_DATASET_PATH}" \
	--validation_dataset_path="${VALIDATION_DATASET_PATH}" \
	--vocab_size=${VOCAB_SIZE} \
	--max_context_length=${MAX_CONTEXT_LENGTH} \
	--num_layers=${NUM_LAYERS} \
	--num_heads=${NUM_HEADS} \
	--rope_theta=${ROPE_THETA} \
	--d_model=${D_MODEL} \
    --d_ff=${D_FF} \
    --dtype=${DTYPE} \
	--weight_decay=${WEIGHT_DECAY} \
	--adamw_beta_1=${ADAMW_BETA_1} \
	--adamw_beta_2=${ADAMW_BETA_2} \
	--adamw_eps=${ADAMW_EPS} \
	--max_learning_rate=${MAX_LEARNING_RATE} \
	--min_learning_rate=${MIN_LEARNING_RATE} \
	--lr_warmup_iters=${LR_WARMUP_ITERS} \
	--lr_cosine_cycle_iters=${LR_COSINE_CYCLE_ITERS} \
	--checkpoint_dir_path="${CHECKPOINT_DIR_PATH}" \
	--max_num_checkpoints=${MAX_NUM_CHECKPOINTS} \
	--checkpoint_freq=${CHECKPOINT_FREQ} \
	--wandb_entity="${WANDB_ENTITY}" \
	--wandb_project="${WANDB_PROJECT}" \
    --wandb_run_name="${WANDB_RUN_NAME}" \
	--num_steps=${NUM_STEPS} \
	--batch_size=${BATCH_SIZE} \
	--validation_batch_size=${VALIDATION_BATCH_SIZE} \
	--validation_freq=${VALIDATION_FREQ} \
	--device="${DEVICE}" \
	--log_metrics_to_console=${LOG_METRICS_TO_CONSOLE} \
	--torch_cuda_empty_cache=${TORCH_CUDA_EMPTY_CACHE}

