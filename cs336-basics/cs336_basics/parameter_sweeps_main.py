"""Transformer LM hyperparameters sweep using wandb."""

import random

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
# Note that learning rates are not defined as flags since they will be swept.
flags.DEFINE_float("weight_decay", 0.001, "Weight decay.")
flags.DEFINE_float("adamw_beta_1", 0.9, "AdamW beta_1.")
flags.DEFINE_float("adamw_beta_2", 0.999, "AdamW beta_2.")
flags.DEFINE_float("adamw_eps", 1e-8, "AdamW's eps.")
# Configs of WANDB.
flags.DEFINE_string("wandb_entity", "cs336-assignment-1", "wandb entity.")
flags.DEFINE_string("wandb_project", "test_train", "wandb project.")
flags.DEFINE_enum(
    "wandb_sweep_method", "random", ["grid", "random", "bayes"], "Wandb sweep method."
)
flags.DEFINE_string("wandb_sweep_name", None, "Wandb sweep name.")
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


def _get_dtype() -> torch.dtype:
    _DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    return _DTYPE_MAP[FLAGS.dtype]


def _get_train_and_validaton_datasets() -> tuple[npt.NDArray, npt.NDArray]:
    return (
        np.load(FLAGS.training_dataset_path, mmap_mode="r"),
        np.load(FLAGS.validation_dataset_path, mmap_mode="r"),
    )


def _global_performance_tuning() -> None:
    if FLAGS.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")


def _init_state(min_learning_rate: float) -> tuple[nn.Module, optim.Optimizer, int]:
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
        dtype=_get_dtype(),
    )
    optimizer = optimizers.AdamW(
        model.parameters(),
        lr=min_learning_rate,
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

    return (model, optimizer, 0)


def train_and_get_validation_loss() -> None:
    """Trains a Transformer LM and returns its validation loss."""
    # Fix all the random seeds.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)

    logging.info("Creating wandb run...")
    wandb_run = wandb.init(
        entity=FLAGS.wandb_entity,
        project=FLAGS.wandb_project,
    )
    logging.info(f"wandb run created. Run config: {wandb_run.config}.")

    logging.info("Init model and optimizer...")
    model, optimizer, _ = _init_state(wandb_run.config["min_learning_rate"])
    logging.info("Model and optimizer initialized.")

    lr_scheduler = optimizers.CosineLrScheduler(
        optimizer=optimizer,
        max_learning_rate=wandb_run.config["max_learning_rate"],
        min_learning_rate=wandb_run.config["min_learning_rate"],
        warmup_iters=wandb_run.config["lr_warmup_iters"],
        cosine_cycle_iters=wandb_run.config["lr_cosine_cycle_iters"],
        last_epoch=-1,
    )

    logging.info(
        f"Mapping training and validation dataset from {FLAGS.training_dataset_path} and "
        f"{FLAGS.validation_dataset_path}."
    )
    training_dataset, validation_dataset = _get_train_and_validaton_datasets()
    logging.info("Training and validation datasets created.")

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
            checkpoint_freq=1_000,
            validation_batch_size=FLAGS.validation_batch_size,
            validation_freq=FLAGS.validation_freq,
            max_total_gradient_l2_norm=FLAGS.max_total_gradient_l2_norm,
            device=FLAGS.device,
        ),
        checkpoint_manager=None,
        wandb_run=wandb_run,
        log_metrics_to_console=FLAGS.log_metrics_to_console,
    )
    logging.info("Main training loop completed.")
    wandb_run.finish()


def main(argv):
    """Runs the sweep."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line args,")

    # Global performance tuning.
    _global_performance_tuning()

    sweep_configuration = {
        "method": FLAGS.wandb_sweep_method,
        "name": FLAGS.wandb_sweep_name,
        "metric": {"goal": "minimize", "name": "validation_loss"},
        "parameters": {
            "max_learning_rate": {"max": 2e-2, "min": 5e-4},
            "min_learning_rate": {"max": 2e-4, "min": 1e-5},
            "lr_warmup_iters": {"values": [50, 100, 150, 200]},
            "lr_cosine_cycle_iters": {"values": [4000, 5000, 6000, 7000]},
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=FLAGS.wandb_project,
    )

    wandb.agent(sweep_id=sweep_id, function=train_and_get_validation_loss, count=20)


if __name__ == "__main__":
    app.run(main)
