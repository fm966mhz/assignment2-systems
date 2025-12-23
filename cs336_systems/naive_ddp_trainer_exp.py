"""Naive DDP trainer experiment."""

import os

from copy import deepcopy

import numpy.typing as npt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from absl import app
from absl import flags
from absl import logging
from jaxtyping import Float

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
_test_dataset_size = flags.DEFINE_integer(
    "test_dataset_size",
    1024,
    "The size of the test dataset to use.",
)
_test_x_size = flags.DEFINE_integer(
    "test_x_size",
    10,
    "The size of the test x to use.",
)
_test_y_size = flags.DEFINE_integer(
    "test_y_size",
    5,
    "The size of the test y to use.",
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


class ToyModel(torch.nn.Module):
    """Toy model."""

    def __init__(self, x_size: int, y_size: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(x_size, 16)
        self.fc2 = torch.nn.Linear(16, y_size)
        self.relu = torch.nn.ReLU()

    def forward(
        self, x: Float[npt.NDArray, "batch_size x_size"]
    ) -> Float[npt.NDArray, "batch_size y_size"]:
        """Forward pass."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def setup(rank: int, world_size: int, backend: str) -> None:
    """Setup."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    logging.set_verbosity(logging.INFO)


def cleanup() -> None:
    """Cleanup."""
    dist.destroy_process_group()


def get_model(
    device: str, x_size: int, y_size: int
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Gets model.

    Args:
        rank: The rank of the process.
        world_size: The number of processes.
        device: The device to use.

    Returns:
        A tuple of the parallel model and the non-parallel model.
    """
    model = ToyModel(x_size, y_size)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    non_parallel_model = deepcopy(model)
    return model.to(device), non_parallel_model.to(device)


def get_optimizer(
    model: torch.nn.Module, non_parallel_model: torch.nn.Module, learning_rate: float
) -> tuple[optim.Optimizer, optim.Optimizer]:
    """Gets optimizer.

    Args:
        model: The parallel model.
        non_parallel_model: The non-parallel model.
        learning_rate: The learning rate.

    Returns:
        A tuple of the optimizer and the non-parallel optimizer.
    """
    return (
        optim.Adam(model.parameters(), lr=learning_rate),
        optim.Adam(non_parallel_model.parameters(), lr=learning_rate),
    )


def naive_ddp_train(
    rank: int,
    world_size: int,
    backend: str,
    x_size: int,
    y_size: int,
    batch_size: int,
    learning_rate: float,
    x: Float[torch.Tensor, "dataset_size x_size"],
    y: Float[torch.Tensor, "dataset_size y_size"],
) -> None:
    """Naive DDP train step."""
    setup(rank, world_size, backend)
    logging.info("Rank %d, setup", rank)
    device = f"cuda:{rank}" if backend == "nccl" else f"cpu:{rank}"
    model, non_parallel_model = get_model(device, x_size, y_size)
    optimizer, non_parallel_optimizer = get_optimizer(
        model, non_parallel_model, learning_rate
    )

    local_batch_size = batch_size // world_size
    for batch_idx, (x_batch, y_batch) in enumerate(
        zip(x.split(batch_size), y.split(batch_size))
    ):
        logging.info("Rank %d, batch %d", rank, batch_idx)
        optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Step for the non-parallel model.
        y_pred_non_parallel = non_parallel_model(x_batch)
        loss_non_parallel = ((y_batch - y_pred_non_parallel) ** 2).mean()
        loss_non_parallel.backward()
        non_parallel_optimizer.step()

        # Step for the parallel model.
        x_batch_local = x_batch[rank * local_batch_size : (rank + 1) * local_batch_size]
        y_batch_local = y_batch[rank * local_batch_size : (rank + 1) * local_batch_size]
        y_pred_parallel = model(x_batch_local)
        loss_parallel = ((y_batch_local - y_pred_parallel) ** 2).mean()
        loss_parallel.backward()
        # Before al-reduce, the gradients should be different from the non-parallel model.
        for param, non_parallel_param in zip(
            model.parameters(), non_parallel_model.parameters()
        ):
            assert not torch.allclose(param.grad, non_parallel_param.grad), (
                f"Rank {rank}, batch {batch_idx}, param {param.grad} is close to "
                f"non_parallel_param {non_parallel_param.grad}"
            )
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
        # After al-reduce, the gradients should be the same as the non-parallel model.
        for param, non_parallel_param in zip(
            model.parameters(), non_parallel_model.parameters()
        ):
            assert torch.allclose(param.grad, non_parallel_param.grad), (
                f"Rank {rank}, batch {batch_idx}, param grad {param.grad} is not close to "
                f"non_parallel_param grad {non_parallel_param.grad}. "
            )
        optimizer.step()
        logging.info(
            "Rank %d, batch %d, loss_parallel: %f, loss_non_parallel: %f",
            rank,
            batch_idx,
            loss_parallel.detach().cpu().item(),
            loss_non_parallel.detach().cpu().item(),
        )
        del (
            x_batch,
            y_batch,
            x_batch_local,
            y_batch_local,
            y_pred_non_parallel,
            loss_non_parallel,
            y_pred_parallel,
            loss_parallel,
        )
    cleanup()


def main(argv: list[str]) -> None:
    """Main function."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line args,")

    test_x = torch.randn(_test_dataset_size.value, _test_x_size.value)
    test_y = torch.randn(_test_dataset_size.value, _test_y_size.value)
    mp.spawn(
        naive_ddp_train,
        args=(
            _world_size.value,
            _dist_backend.value,
            _test_x_size.value,
            _test_y_size.value,
            _batch_size.value,
            _learning_rate.value,
            test_x,
            test_y,
        ),
        nprocs=_world_size.value,
        join=True,
    )


if __name__ == "__main__":
    app.run(main)
