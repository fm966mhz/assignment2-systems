"""Bench mark DDP."""

import os
import random
import timeit

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from absl import app
from absl import logging
from absl import flags
from jaxtyping import Int

from cs336_basics import functions as F
from cs336_basics import transformer
from cs336_systems import predefined_model_configs

FLAGS = flags.FLAGS


_random_seed = flags.DEFINE_integer(
    "random_seed",
    42,
    "The random seed to use.",
)
_dist_backend = flags.DEFINE_enum(
    "dist_backend",
    "gloo",
    ["gloo", "nccl"],
    "The distributed backend to use.",
)
_world_size = flags.DEFINE_integer(
    "world_size",
    4,
    "The number of processes to use.",
)
_max_context_length = flags.DEFINE_integer(
    "max_context_length",
    1024,
    "The maximum context length to use.",
)
_vocab_size = flags.DEFINE_integer(
    "vocab_size",
    10_000,
    "The vocabulary size to use.",
)
_batch_size = flags.DEFINE_integer(
    "batch_size",
    128,
    "The batch size to use.",
)
_learning_rate = flags.DEFINE_float(
    "learning_rate",
    0.001,
    "The learning rate to use.",
)
_predefined_model_config = flags.DEFINE_enum(
    "predefined_model_config",
    "xl",
    ["small", "medium", "large", "xl", "2p7B"],
    "The predefined model config.",
)
_warmup_steps = flags.DEFINE_integer(
    "warmup_steps",
    5,
    "The number of warmup steps to run.",
)
_benchmarking_steps = flags.DEFINE_integer(
    "benchmarking_steps",
    10,
    "The number of benchmarking steps to run.",
)
_flatten_before_communication = flags.DEFINE_bool(
    "flatten_before_communication",
    False,
    "If true, flattens the input and label sequences before communication.",
)


def _set_random_seeds(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def _get_random_test_batch(
    batch_size: int,
    max_context_length: int,
    vocab_size: int,
    device: str,
    rank: int,
    world_size: int,
) -> tuple[
    Int[torch.Tensor, "batch_size context_length"],
    Int[torch.Tensor, "batch_size context_length"],
]:
    """Get the random test batch."""
    sequence = torch.randint(0, vocab_size, (batch_size, max_context_length + 1))
    per_rank_batch_size = batch_size // world_size
    input_seq = sequence[
        rank * per_rank_batch_size : (rank + 1) * per_rank_batch_size, :-1
    ]
    label_seq = sequence[
        rank * per_rank_batch_size : (rank + 1) * per_rank_batch_size, 1:
    ]
    if device.startswith("cuda"):
        input_seq = input_seq.pin_memory().to(device)
        label_seq = label_seq.pin_memory().to(device)
    else:
        input_seq = input_seq.to(device)
        label_seq = label_seq.to(device)
    return input_seq, label_seq


def _all_gather_time_measurements(
    step_times: list[float],
    communication_times: list[float],
    world_size: int,
    device: str,
) -> tuple[float, float]:
    """Measure the time taken for all-reduce."""
    step_times_tensor = torch.tensor(step_times, device=device)
    communication_times_tensor = torch.tensor(communication_times, device=device)
    all_step_times = [torch.zeros_like(step_times_tensor) for _ in range(world_size)]
    all_communication_times = [
        torch.zeros_like(communication_times_tensor) for _ in range(world_size)
    ]
    dist.all_gather(all_step_times, step_times_tensor)
    dist.all_gather(all_communication_times, communication_times_tensor)
    return (
        torch.cat(all_step_times).mean().item(),
        torch.cat(all_communication_times).mean().item(),
    )


def setup(rank: int, world_size: int, backend: str, random_seed: int) -> str:
    """Setup the distributed environment.

    Args:
        rank: The rank of the process.
        world_size: The number of processes to use.
        backend: The distributed backend to use.
        random_seed: The random seed to use.

    Returns:
        The device to use.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    logging.set_verbosity(logging.INFO)
    _set_random_seeds(random_seed)

    device = f"cuda:{rank}" if backend == "nccl" else f"cpu:{rank}"
    return device


def cleanup():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()


def get_model(
    device: str,
    config: transformer.TransformerConfig,
) -> transformer.TransformerLm:
    """Get the model.

    Args:
        device: The device to use.
        config: The model config.

    Returns:
        The model.
    """
    model = transformer.TransformerLm(config, device=device).to(device)  # type: ignore
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    return model


def get_optimizer(
    model: transformer.TransformerLm,
    learning_rate: float,
) -> optim.Optimizer:
    """Get the optimizer."""
    return optim.AdamW(model.parameters(), lr=learning_rate)


def run_one_step(
    model: transformer.TransformerLm,
    optimizer: optim.Optimizer,
    input_seq: Int[torch.Tensor, "batch_size context_length"],
    label_seq: Int[torch.Tensor, "batch_size context_length"],
    flatten_before_communication: bool,
    device: str,
) -> tuple[float, float]:
    """Runs one step.

    Args:
        model: The model to run.
        optimizer: The optimizer to use.
        input_seq: The input sequence.
        label_seq: The label sequence.
        flatten_before_communication: If true, flattens the input and label sequences before
            communication.
        device: The device to use.

    Returns:
        A tuple of the step time and the communication time.
    """
    optimizer.zero_grad()
    # Forward and backward pass.
    start_time = timeit.default_timer()
    logits = model(input_seq)
    loss = F.cross_entropy(logits=logits, targets=label_seq)
    loss.backward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    logging.info(f"Device {device} forward and backward completed.")
    # Communication (all-reduce).
    communication_start_time = timeit.default_timer()
    if flatten_before_communication:
        flattened_grads = (
            torch._utils._flatten_dense_tensors(  # pylint: disable=protected-access
                [param.grad for param in model.parameters()]
            )
        )
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.AVG, async_op=False)
        torch._utils._unflatten_dense_tensors(  # pylint: disable=protected-access
            flattened_grads, [param.grad for param in model.parameters()]
        )
    else:
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    logging.info(f"Device {device} communication completed.")
    communication_end_time = timeit.default_timer()
    communication_time = communication_end_time - communication_start_time
    # Optimization step.
    optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    logging.info(f"Device {device} optimization step completed.")
    end_time = timeit.default_timer()
    return end_time - start_time, communication_time


def run_benchmarking_steps(
    rank: int,
    world_size: int,
    random_seed: int,
    dist_backend: str,
    warmup_steps: int,
    benchmarking_steps: int,
    batch_size: int,
    max_context_length: int,
    vocab_size: int,
    learning_rate: float,
    predefined_model_config: str,
    flatten_before_communication: bool,
) -> None:
    """Run benchmarking steps."""
    device = setup(rank, world_size, dist_backend, random_seed)
    model_config = predefined_model_configs.get_predefined_model_configs(
        vocab_size, max_context_length
    )[predefined_model_config]
    model = get_model(device, model_config)
    optimizer = get_optimizer(model, learning_rate)
    logging.info("Rank %d, running %d warmup steps...", rank, warmup_steps)
    for _ in range(warmup_steps):
        input_seq, label_seq = _get_random_test_batch(
            batch_size, max_context_length, vocab_size, device, rank, world_size
        )
        run_one_step(
            model, optimizer, input_seq, label_seq, flatten_before_communication, device
        )
    logging.info("Rank %d, running %d benchmarking steps...", rank, benchmarking_steps)
    local_step_times, local_communication_times = [], []
    for step_idx in range(benchmarking_steps):
        input_seq, label_seq = _get_random_test_batch(
            batch_size, max_context_length, vocab_size, device, rank, world_size
        )
        step_time, communication_time = run_one_step(
            model, optimizer, input_seq, label_seq, flatten_before_communication, device
        )
        local_step_times.append(step_time)
        local_communication_times.append(communication_time)
        logging.info(
            "Rank %d, step %d, step time: %.6f, communication time: %.6f",
            rank,
            step_idx,
            step_time,
            communication_time,
        )
    avg_step_time, avg_communication_time = _all_gather_time_measurements(
        local_step_times, local_communication_times, world_size, device
    )
    if rank == 0:
        logging.info(
            "Average step time over all ranks: %.6f seconds",
            avg_step_time,
        )
        logging.info(
            "Average communication time over all ranks: %.6f seconds",
            avg_communication_time,
        )
    cleanup()


def main(argv: list[str]) -> None:
    """Main function."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    mp.spawn(
        run_benchmarking_steps,
        args=(
            _world_size.value,
            _random_seed.value,
            _dist_backend.value,
            _warmup_steps.value,
            _benchmarking_steps.value,
            _batch_size.value,
            _max_context_length.value,
            _vocab_size.value,
            _learning_rate.value,
            _predefined_model_config.value,
            _flatten_before_communication.value,
        ),
        nprocs=_world_size.value,
        join=True,
    )


if __name__ == "__main__":
    app.run(main)
