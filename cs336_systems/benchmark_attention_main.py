"""Benchmark attention."""

import collections
import timeit

import numpy as np
import pandas as pd
import torch
import tqdm

from absl import app
from absl import flags
from absl import logging
from jaxtyping import Float

from torch import optim

from cs336_basics import layers
from cs336_basics import optimizers


FLAGS = flags.FLAGS


flags.DEFINE_integer("random_seed", 42, "The random seed.")
flags.DEFINE_integer("num_iters", 10, "The number of iterations to run the benchmark.")
flags.DEFINE_bool("compile", False, "If true, compiles the model.")
flags.DEFINE_string(
    "output_path_prefix", "", "The prefix of the output paths of the pickle files."
)


def _set_random_seeds() -> None:
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)
    torch.cuda.manual_seed(FLAGS.random_seed)


def get_test_data(batch_size: int, d_model: int, seq_len: int) -> tuple[
    Float[torch.Tensor, "batch_size seq_len d_model"],
    Float[torch.Tensor, "batch_size seq_len d_model"],
]:
    return (
        torch.randn((batch_size, seq_len, d_model)).to("cuda"),
        torch.randn((batch_size, seq_len, d_model)).to("cuda"),
    )


def run_one_forward_pass(
    model: layers.MultiHeadSelfAttention,
    in_features: Float[torch.Tensor, "batch_size seq_len d_model"],
    labels: Float[torch.Tensor, "batch_size seq_len d_model"],
) -> torch.Tensor:
    """Runs one forward pass."""
    out_features = model(in_features)
    loss = torch.nn.functional.mse_loss(out_features, labels)
    torch.cuda.synchronize()
    return loss


def time_forward_pass(
    model: layers.MultiHeadSelfAttention,
    in_features: Float[torch.Tensor, "batch_size seq_len d_model"],
    labels: Float[torch.Tensor, "batch_size seq_len d_model"],
    num_iters: int,
) -> tuple[float, float]:
    """Times the forward pass."""
    times, memory_usage = [], []
    for _ in range(num_iters):
        start_time = timeit.default_timer()
        _ = run_one_forward_pass(model, in_features, labels)
        memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
        torch.cuda.empty_cache()
    return float(np.mean(times)), float(np.mean(memory_usage))


def time_backward_pass(
    model: layers.MultiHeadSelfAttention,
    optimizer: optim.Optimizer,
    in_features: Float[torch.Tensor, "batch_size seq_len d_model"],
    labels: Float[torch.Tensor, "batch_size seq_len d_model"],
    num_iters: int,
) -> float:
    """Times the backward pass."""
    times = []
    for _ in range(num_iters):
        optimizer.zero_grad()
        loss = run_one_forward_pass(model, in_features, labels)
        start_time = timeit.default_timer()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(timeit.default_timer() - start_time)
    return float(np.mean(times))


def main(argv):
    """Main function to benchmark attention."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    _set_random_seeds()

    logging.info("Running benchmark attention...")
    param_combos = []
    # Keyed by d_model, then kieyed by seq_len.
    forward_times, backward_times, memory_usage = (
        collections.defaultdict(dict),
        collections.defaultdict(dict),
        collections.defaultdict(dict),
    )
    # 16384 would crash my laptop.
    for seq_len in [256, 1024, 4096, 8192]:  # , 16384]:for seq_len in []
        for d_model in [16, 32, 64, 128]:
            param_combos.append((d_model, seq_len))

    for d_model, seq_len in tqdm.tqdm(param_combos):
        model = layers.MultiHeadSelfAttention(
            d_model, num_heads=1, device=torch.device("cuda")
        ).to("cuda")
        if FLAGS.compile:
            model = torch.compile(model)
        optimizer = optimizers.AdamW(model.parameters(), lr=1e-3)
        in_features, labels = get_test_data(
            batch_size=8, d_model=d_model, seq_len=seq_len
        )
        # Warm up.
        for _ in range(5):
            loss = run_one_forward_pass(model, in_features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()
        # Benchmark.
        forward_time, forward_memory_usage = time_forward_pass(
            model, in_features, labels, num_iters=FLAGS.num_iters
        )
        forward_times[d_model][seq_len] = forward_time
        memory_usage[d_model][seq_len] = forward_memory_usage
        backward_times[d_model][seq_len] = time_backward_pass(
            model, optimizer, in_features, labels, num_iters=FLAGS.num_iters
        )
        torch.cuda.empty_cache()

    pd.DataFrame(forward_times).to_pickle(
        f"{FLAGS.output_path_prefix}_forward_times.pickle"
    )
    pd.DataFrame(backward_times).to_pickle(
        f"{FLAGS.output_path_prefix}_backward_times.pickle"
    )
    pd.DataFrame(memory_usage).to_pickle(
        f"{FLAGS.output_path_prefix}_memory_usage.pickle"
    )

    logging.info("Benchmark attention completed.")


if __name__ == "__main__":
    app.run(main)
