"""Train a Transfomer LM."""

import pickle

from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from absl import app
from absl import logging
from absl import flags
from torch import nn
from torch import optim

import wandb

from cs336_basics import checkpoint
from cs336_basics import optimizers
from cs336_basics import train_model
from cs336_basics import transformer


FLAGS = flags.FLAGS

# Input data.
flags.DEFINE_string("training_dataset_path", "", "The training data path.")
flags.DEFINE_string(
    "validation_dataset_path", "", "The path to the validation dataset."
)
# Configs of the transformer.
flags.DEFINE_integer("vocab_size", None, "The vocab size.")
flags.DEFINE_integer("max_context_length", None, "The max context length.")
flags.DEFINE_integer("num_layers", None, "Number of layers.")
flags.DEFINE_integer("num_heads", None, "Number of heads.")
flags.DEFINE_float("rope_theta", None, "RoPE theta.")
flags.DEFINE_integer("d_model", None, "d_model.")
flags.DEFINE_float("d_ff_to_d_model", 8.0 / 3.0, "d_ff_to_d_model.")
flags.DEFINE_integer(
    "d_ff", None, "d_ff. This one takes precedence over `d_ff_to_d_model."
)
flags.DEFINE_enum("dtype", "float32", ["float32", "bfloat16"], "dtype.")
# Configs of the optimizer.
flags.DEFINE_float("weight_decay", 0.001, "Weight decay.")
flags.DEFINE_float("adamw_beta_1", 0.9, "AdamW beta_1.")
flags.DEFINE_float("adamw_beta_2", 0.999, "AdamW beta_2.")
flags.DEFINE_float("adamw_eps", 1e-8, "AdamW's eps.")
flags.DEFINE_float("max_learning_rate", 2e-3, "Max learning rate.")
flags.DEFINE_float("min_learning_rate", 1e-4, "Min learning rate.")
flags.DEFINE_integer(
    "lr_warmup_iters", 20, "Learning rate warm up number of iteration."
)
flags.DEFINE_integer(
    "lr_cosine_cycle_iters", 100, "Learning rate cosine cycle number of iterations."
)
#  Configs of the checkpointing.
flags.DEFINE_string("checkpoint_dir_path", "", "Path to the checkpoint directory.")
flags.DEFINE_integer("max_num_checkpoints", None, "Max number of checkpoints to store.")
flags.DEFINE_integer("checkpoint_freq", None, "Checkpointing frequency.")
# Configs of WANDB.
flags.DEFINE_string("wandb_entity", "cs336-assignment-1", "wandb entity.")
flags.DEFINE_string("wandb_project", "test_train", "wandb project.")
flags.DEFINE_string("wandb_run_name", None, "wandb run name.")
# Configs of training.
flags.DEFINE_integer("num_steps", None, "Number of training steps.")
flags.DEFINE_integer("batch_size", None, "Training batch size.")
flags.DEFINE_integer("validation_batch_size", None, "Validation batch size.")
flags.DEFINE_integer("validation_freq", None, "Validation frequency.")
flags.DEFINE_string("device", "cpu", "Device of the training.")
# Gradient clipping
flags.DEFINE_float("max_total_gradient_l2_norm", None, "Max total gradient L2 norm.")
# Misc
flags.DEFINE_bool("log_metrics_to_console", False, "Log metrics to the console.")
flags.DEFINE_bool(
    "torch_cuda_empty_cache", False, "If true, aggresively empty CUDA cache."
)


def _get_dtype() -> torch.dtype:
    _dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    return _dtype_map[FLAGS.dtype]


def _get_train_and_validaton_datasets() -> tuple[npt.NDArray, npt.NDArray]:
    return (
        np.load(FLAGS.training_dataset_path, mmap_mode="r"),
        np.load(FLAGS.validation_dataset_path, mmap_mode="r"),
    )


def _load_or_create_checkpoint_manager() -> checkpoint.CheckpointManager:
    return checkpoint.CheckpointManager(
        checkpoint_dir=FLAGS.checkpoint_dir_path,
        max_num_checkpoints=FLAGS.max_num_checkpoints,
    )


def _get_wandb_config() -> dict[str, Any]:
    return {
        "vocab_size": FLAGS.vocab_size,
        "max_content_length": FLAGS.max_context_length,
        "num_layers": FLAGS.num_layers,
        "num_heads": FLAGS.num_heads,
        "rope_theta": FLAGS.rope_theta,
        "d_model": FLAGS.d_model,
        "d_ff": FLAGS.d_ff,
        "d_ff_to_d_model": FLAGS.d_ff_to_d_model,
        "dtype": FLAGS.dtype,
        "weight_decay": FLAGS.weight_decay,
        "adamw_beta_1": FLAGS.adamw_beta_1,
        "adamw_beta_2": FLAGS.adamw_beta_2,
        "max_learning_rate": FLAGS.max_learning_rate,
        "min_learning_rate": FLAGS.min_learning_rate,
        "lr_warmup_iters": FLAGS.lr_warmup_iters,
        "lr_cosine_cycle_iters": FLAGS.lr_cosine_cycle_iters,
        "num_steps": FLAGS.num_steps,
        "batch_size": FLAGS.batch_size,
    }


def _global_performance_tuning() -> None:
    if FLAGS.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")


def _load_or_init_state(
    checkpoint_manager: checkpoint.CheckpointManager,
) -> tuple[nn.Module, optim.Optimizer, int]:
    model = transformer.TransformerLm(
        transformer.TransformerConfig(
            vocab_size=FLAGS.vocab_size,
            context_length=FLAGS.max_context_length,
            num_layers=FLAGS.num_layers,
            num_heads=FLAGS.num_heads,
            rope_theta=FLAGS.rope_theta,
            d_model=FLAGS.d_model,
            d_ff_to_d_model=FLAGS.d_ff_to_d_model,
            d_ff=FLAGS.d_ff,
        ),
        device=FLAGS.device,
    )
    optimizer = optimizers.AdamW(
        model.parameters(),
        lr=FLAGS.min_learning_rate,
        weight_decay=FLAGS.weight_decay,
        betas=(FLAGS.adamw_beta_1, FLAGS.adamw_beta_2),
        eps=FLAGS.adamw_eps,
    )
    # `torch.compile` will change the keys in the state_dict (prepending by `_orig_mod_`), so we
    # need to compile the mode before loading checkpoints.
    if FLAGS.device.startswith("mps"):
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)

    latest_checkpointed_iteration = checkpoint_manager.load_checkpoint(
        model=model, optimizer=optimizer, device=FLAGS.device
    )
    return (model, optimizer, latest_checkpointed_iteration)


def main(argv):
    """Runs the training."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line args,")

    # Global performance tuning.
    _global_performance_tuning()

    logging.info(f"Loading checkpoint from {FLAGS.checkpoint_dir_path}...")
    checkpoint_manager = _load_or_create_checkpoint_manager()
    model, optimizer, latest_checkpointed_iteration = _load_or_init_state(
        checkpoint_manager=checkpoint_manager
    )
    if latest_checkpointed_iteration == 0:
        logging.info(
            f"No existing checkpoints in {FLAGS.checkpoint_dir_path}. Model and optimizer "
            "initialized."
        )
    else:
        logging.info(
            "Model and optimizer loaded from the latest checkpoint at iteration "
            f"{latest_checkpointed_iteration}."
        )
    lr_scheduler = optimizers.CosineLrScheduler(
        optimizer=optimizer,
        max_learning_rate=FLAGS.max_learning_rate,
        min_learning_rate=FLAGS.min_learning_rate,
        warmup_iters=FLAGS.lr_warmup_iters,
        cosine_cycle_iters=FLAGS.lr_cosine_cycle_iters,
        last_epoch=latest_checkpointed_iteration - 1,
    )

    logging.info(
        f"Mapping training and validation dataset from {FLAGS.training_dataset_path} and "
        f"{FLAGS.validation_dataset_path}."
    )
    training_dataset, validation_dataset = _get_train_and_validaton_datasets()
    logging.info("Training and validation datasets created.")

    logging.info("Creating wandb run...")
    wandb_run = wandb.init(
        entity=FLAGS.wandb_entity,
        project=FLAGS.wandb_project,
        name=FLAGS.wandb_run_name,
        config=_get_wandb_config(),
    )
    logging.info("wandb run created.")

    logging.info(f"Running main training loop for {FLAGS.num_steps} steps...")
    train_model.train_loop(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        config=train_model.TrainingConfig(
            num_steps=FLAGS.num_steps,
            training_batch_size=FLAGS.batch_size,
            context_length=FLAGS.max_context_length,
            checkpoint_freq=FLAGS.checkpoint_freq,
            validation_batch_size=FLAGS.validation_batch_size,
            validation_freq=FLAGS.validation_freq,
            max_total_gradient_l2_norm=FLAGS.max_total_gradient_l2_norm,
            device=FLAGS.device,
        ),
        dtype=_get_dtype(),
        checkpoint_manager=checkpoint_manager,
        wandb_run=wandb_run,
        log_metrics_to_console=FLAGS.log_metrics_to_console,
        torch_cuda_empty_cache=FLAGS.torch_cuda_empty_cache,
    )
    logging.info("Main training loop completed.")
    wandb_run.finish()


if __name__ == "__main__":
    app.run(main)
