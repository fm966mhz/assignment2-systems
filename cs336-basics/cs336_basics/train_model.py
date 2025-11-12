"""The training utils."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import wandb

from absl import logging
from jaxtyping import Float
from torch import nn
from torch import optim
from tqdm import tqdm

from cs336_basics.checkpoint import CheckpointManager
from cs336_basics.data_loader import get_batch
from cs336_basics.functions import cross_entropy


@dataclass(frozen=True)
class TrainingConfig:
    """Training config."""

    num_steps: int
    training_batch_size: int
    context_length: int
    checkpoint_freq: int

    validation_batch_size: int
    validation_freq: int

    device: str

    max_total_gradient_l2_norm: float | None = None


def train_loop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    training_dataset: npt.NDArray,
    validation_dataset: npt.NDArray,
    config: TrainingConfig,
    dtype: torch.dtype,
    checkpoint_manager: CheckpointManager | None = None,
    wandb_run: wandb.Run | None = None,
    log_metrics_to_console: bool = False,
    torch_cuda_empty_cache: bool = False,
):
    """The main training loop."""
    latest_checkpointed_iteration = (
        checkpoint_manager.checkpoint_metadata.latest_checkpointed_iteration
        if checkpoint_manager is not None
        else 0
    )
    scaler = (
        torch.amp.grad_scaler.GradScaler() if config.device.startswith("cuda") else None
    )
    logging.info(f"Use AMP GradScaler: {scaler}.")
    for t in tqdm(
        range(
            latest_checkpointed_iteration,
            config.num_steps,
        ),
        initial=latest_checkpointed_iteration,
        total=config.num_steps,
    ):
        optimizer.zero_grad()
        # TODO(djwenren): pinning CPU memories?
        input_seq, label_seq = get_batch(
            dataset=training_dataset,
            batch_size=config.training_batch_size,
            context_length=config.context_length,
            device=config.device,
        )
        # Forward pass.
        if config.device.startswith("cuda"):
            if t == latest_checkpointed_iteration:
                logging.info(
                    f"Training using autocast with device type {config.device}, dtype: {dtype}."
                )
            with torch.autocast(device_type=config.device, dtype=dtype):
                logits: Float[torch.Tensor, "batch_size seq_len vocab_size"] = model(
                    input_seq
                )
                loss = cross_entropy(logits=logits, targets=label_seq)
        else:
            logits: Float[torch.Tensor, "batch_size seq_len vocab_size"] = model(
                input_seq
            )
            loss = cross_entropy(logits=logits, targets=label_seq)
        loss_val = loss.detach().cpu().item()
        if wandb_run is not None:
            wandb_run.log(
                {
                    "training_loss": loss_val,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=t + 1,
            )

        # Backward pass.
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        del loss, loss_val, logits, input_seq, label_seq

        if wandb_run and (
            (t + 1 - latest_checkpointed_iteration) % config.validation_freq == 0
        ):
            if str(config.device).startswith("cuda") and torch_cuda_empty_cache:
                torch.cuda.empty_cache()
            run_validation(
                model=model,
                validation_dataset=validation_dataset,
                config=config,
                wandb_run=wandb_run,
                step=t + 1,
            )

        if checkpoint_manager and (
            (t + 1 - latest_checkpointed_iteration) % config.checkpoint_freq == 0
        ):
            checkpoint_manager.save_checkpoint(
                model=model, optimizer=optimizer, iteration=t + 1
            )

        if str(config.device).startswith("cuda") and torch_cuda_empty_cache:
            torch.cuda.empty_cache()


def run_validation(
    model: nn.Module,
    validation_dataset: npt.NDArray,
    config: TrainingConfig,
    wandb_run: wandb.Run | None,
    step: int,
):
    """Runs valiation.

    Returns the validation loss and perplexity.
    """
    input_seq, label_seq = get_batch(
        dataset=validation_dataset,
        batch_size=config.validation_batch_size,
        context_length=config.context_length,
        device=config.device,
    )
    with torch.no_grad():
        logits: Float[torch.Tensor, "batch_size seq_len vocab_size"] = model(input_seq)
    loss = cross_entropy(logits=logits, targets=label_seq).detach().cpu().item()
    # Perplexity is just the exponential of the cross entropy loss.
    perplexity = np.exp(loss)
    if wandb_run is not None:
        wandb_run.log(
            {
                "validation_loss": loss,
                "validation_perplexity": perplexity,
            },
            step=step,
        )
    del input_seq, label_seq, logits, loss, perplexity
