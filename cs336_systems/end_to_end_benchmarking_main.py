"""End-to-end benchmarking main."""

import contextlib
import random
import timeit

from typing import Any
from typing import Callable

import cs336_basics
import numpy as np
import torch
import torch.cuda.nvtx as nvtx

from absl import app
from absl import flags
from absl import logging
from jaxtyping import Int
from torch import optim

import cs336_basics.functions
from cs336_systems import annotated_functions

cs336_basics.functions.scaled_dot_product_attention = (
    annotated_functions.annotated_scaled_dot_product_attention
)

from cs336_basics import functions as F
from cs336_basics import optimizers as O
from cs336_basics import transformer
from cs336_systems import predefined_model_configs

FLAGS = flags.FLAGS


flags.DEFINE_integer("random_seed", 42, "The random seed.")
flags.DEFINE_integer("batch_size", 4, "The batch size.")
flags.DEFINE_integer("num_layers", 12, "The number of layers.")
flags.DEFINE_integer("num_heads", 12, "The number of heads.")
flags.DEFINE_integer("d_model", 768, "The model dimension.")
flags.DEFINE_integer("d_ff", 3072, "The feedforward dimension.")
flags.DEFINE_float("rope_theta", 10000.0, "The RoPE theta.")
flags.DEFINE_enum(
    "predefined_model_config",
    None,
    ["small", "medium", "large", "xl", "2p7B"],
    "The predefined model config.",
)
flags.DEFINE_integer(
    "num_benchmarking_steps", 10, "The number of benchmarking steps to run the model."
)
flags.DEFINE_integer(
    "num_warmup_steps",
    10,
    "The number of warmup steps to run the model before benchmarking.",
)
flags.DEFINE_string("device", "cpu", "The device to run the model on.")
flags.DEFINE_bool(
    "forward_pass_only",
    True,
    "If true, only runs the forward pass in warm up and benchmarking steps.",
)
flags.DEFINE_bool(
    "torch_compile", False, "If true, uses torch.compile to compile the model."
)
flags.DEFINE_bool("autocast", False, "If true, uses torch.autocast to run the model.")
flags.DEFINE_enum(
    "autocast_dtype",
    "bfloat16",
    ["float32", "bfloat16"],
    "The dtype to use for the autocast.",
)
flags.DEFINE_bool(
    "profile_memory", False, "If true, profiles the memory usage of the model."
)
flags.DEFINE_string(
    "profile_memory_output_path", None, "The path to save the profile memory output."
)


def _get_model_config() -> transformer.TransformerConfig:
    if FLAGS.predefined_model_config is not None:
        return predefined_model_configs.get_predefined_model_configs()[
            FLAGS.predefined_model_config
        ]
    return transformer.TransformerConfig(
        vocab_size=FLAGS.vocab_size,
        num_layers=FLAGS.num_layers,
        num_heads=FLAGS.num_heads,
        d_model=FLAGS.d_model,
        d_ff=FLAGS.d_ff,
        rope_theta=FLAGS.rope_theta,
        context_length=FLAGS.max_context_length,
    )


def _set_random_seeds() -> None:
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)


def _get_random_test_batch() -> tuple[
    Int[torch.Tensor, "batch_size context_length"],
    Int[torch.Tensor, "batch_size context_length"],
]:
    sequence = torch.randint(
        0, FLAGS.vocab_size, (FLAGS.batch_size, FLAGS.max_context_length + 1)
    )
    input_seq = sequence[:, :-1]
    label_seq = sequence[:, 1:]
    if FLAGS.device.startswith("cuda"):
        input_seq = input_seq.pin_memory().to(FLAGS.device)
        label_seq = label_seq.pin_memory().to(FLAGS.device)
    else:
        input_seq = input_seq.to(FLAGS.device)
        label_seq = label_seq.to(FLAGS.device)
    return input_seq, label_seq


def _get_auto_cast_type() -> torch.dtype:
    if FLAGS.autocast_dtype == "bfloat16":
        return torch.bfloat16
    elif FLAGS.autocast_dtype == "float32":
        return torch.float32
    else:
        raise ValueError(
            f"Invalid autocast dtype: {FLAGS.autocast_dtype}. Must be one of: float32, bfloat16."
        )


def run_one_step(
    model: Callable[..., Any] | transformer.TransformerLm,
    optimizer: optim.Optimizer,
    input_seq: Int[torch.Tensor, "batch_size context_length"],
    label_seq: Int[torch.Tensor, "batch_size context_length"],
) -> None:
    if FLAGS.autocast:
        run_context = torch.autocast(
            device_type=FLAGS.device, dtype=_get_auto_cast_type()
        )
    else:
        run_context = contextlib.nullcontext()
    if FLAGS.forward_pass_only:
        grad_context = torch.no_grad()
    else:
        grad_context = contextlib.nullcontext()
    with run_context, grad_context:
        optimizer.zero_grad()
        logits = model(input_seq)
        loss = F.cross_entropy(logits=logits, targets=label_seq)
        if FLAGS.forward_pass_only:
            if FLAGS.device.startswith("cuda"):
                torch.cuda.synchronize()
            return
        loss.backward()
        optimizer.step()
        if FLAGS.device.startswith("cuda"):
            torch.cuda.synchronize()


def run_warmup_steps(
    model: Callable[..., Any] | transformer.TransformerLm,
    optimizer: optim.Optimizer,
    input_seq: Int[torch.Tensor, "batch_size context_length"],
    label_seq: Int[torch.Tensor, "batch_size context_length"],
) -> None:
    """Run the warmup steps."""
    with nvtx.range("run_warmup_steps"):
        for _ in range(FLAGS.num_warmup_steps):
            run_one_step(
                model=model,
                optimizer=optimizer,
                input_seq=input_seq,
                label_seq=label_seq,
            )


def run_benchmarking_steps(
    model: Callable[..., Any] | transformer.TransformerLm,
    optimizer: optim.Optimizer,
    input_seq: Int[torch.Tensor, "batch_size context_length"],
    label_seq: Int[torch.Tensor, "batch_size context_length"],
) -> tuple[float, float]:
    """Runs the benchmarking steps.

    Returns:
        Tuple of average and standard deviation of one step time.
    """
    if FLAGS.profile_memory and FLAGS.device.startswith("cuda"):
        assert (
            FLAGS.profile_memory_output_path is not None
        ), "Profile memory output path must be provided if profile memory is enabled."
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    with nvtx.range("run_benchmarking_steps"):
        step_times = timeit.repeat(
            lambda: run_one_step(
                model=model,
                optimizer=optimizer,
                input_seq=input_seq,
                label_seq=label_seq,
            ),
            number=1,
            repeat=FLAGS.num_benchmarking_steps,
        )
    if FLAGS.profile_memory and FLAGS.device.startswith("cuda"):
        torch.cuda.memory._dump_snapshot(FLAGS.profile_memory_output_path)
        torch.cuda.memory._record_memory_history(enabled=None)
    return float(np.mean(step_times)), float(np.std(step_times))


def main(argv):
    """Main function to run the end-to-end benchmarking."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Running end-to-end benchmarking...")
    _set_random_seeds()

    model_config = _get_model_config()
    model = transformer.TransformerLm(model_config, device=FLAGS.device)
    optimizer = O.AdamW(model.parameters(), lr=1e-3)
    if FLAGS.torch_compile:
        model = torch.compile(model)
    input_seq, label_seq = _get_random_test_batch()
    logging.info(f"Running {FLAGS.num_warmup_steps} warmup steps...")
    run_warmup_steps(
        model=model,
        optimizer=optimizer,
        input_seq=input_seq,
        label_seq=label_seq,
    )
    logging.info("Warmup steps completed.")

    logging.info(f"Running {FLAGS.num_benchmarking_steps} benchmarking steps...")
    mean, std = run_benchmarking_steps(
        model=model,
        optimizer=optimizer,
        input_seq=input_seq,
        label_seq=label_seq,
    )
    logging.info(
        f"Benchmarking steps completed. Mean step time: {mean}, standard deviation: {std}."
    )


if __name__ == "__main__":
    app.run(main)
