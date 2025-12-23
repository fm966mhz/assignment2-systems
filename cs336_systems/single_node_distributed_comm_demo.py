"""Single node distributed communication demo."""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank: int, world_size: int, backend: str):
    """Setup the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def distributed_demo(rank: int, world_size: int, backend: str):
    """Distributed demo."""
    setup(rank, world_size, backend)
    device = f"cuda:{rank}" if backend == "nccl" else f"cpu:{rank}"
    data = torch.randint(0, 10, (3,), device=device)
    print(f"Rank {rank} has data (before all-reduce): {data}")
    dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} has data (after all-reduce): {data}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD_SIZE = 4
    mp.spawn(
        fn=distributed_demo, args=(WORLD_SIZE, "gloo"), nprocs=WORLD_SIZE, join=True
    )
    WORLD_SIZE = 1
    mp.spawn(
        fn=distributed_demo, args=(WORLD_SIZE, "nccl"), nprocs=WORLD_SIZE, join=True
    )
