"""Run FlashAttention2 benchmarking."""

# import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TRITON_INTERPRET"] = "1"

from typing import Callable

import pandas as pd
import torch
import tqdm
import triton

from absl import app
from absl import flags
from absl import logging
from jaxtyping import Float

from cs336_basics import functions
from cs336_systems import triton_functions

_max_seq_len = flags.DEFINE_integer(
    "max_seq_len", 1024, "The maximum sequence length of the query and key."
)
_max_model_dim = flags.DEFINE_integer(
    "max_model_dim", 1024, "The maximum model dimension."
)
_triton_benchmarking_rep = flags.DEFINE_integer(
    "triton_benmarking_rep",
    10000,
    "The number of repetitions for the Triton benchmarking.",
)
_triton_benchmarking_warmup = flags.DEFINE_integer(
    "triton_benmarking_warmup",
    1000,
    "The number of warmup for the Triton benchmarking.",
)
_output_path_prefix = flags.DEFINE_string(
    "output_path_prefix",
    "flash_attention_benmarking",
    "The prefix for the output path.",
)


def _get_test_inputs(B: int, S: int, T: int, D: int, dtype: torch.dtype) -> tuple[
    Float[torch.Tensor, "B S D"],
    Float[torch.Tensor, "B T D"],
    Float[torch.Tensor, "B T D"],
    Float[torch.Tensor, "B S D"],
]:
    torch.manual_seed(42)
    # Q, K, V, dO.
    return (
        torch.randn(
            (B, S, D),
            dtype=dtype,
            device=torch.device("cuda"),
            requires_grad=True,
        ),
        torch.randn(
            (B, T, D),
            dtype=dtype,
            device=torch.device("cuda"),
            requires_grad=True,
        ),
        torch.randn(
            (B, T, D),
            dtype=dtype,
            device=torch.device("cuda"),
            requires_grad=True,
        ),
        torch.randn(
            (B, S, D),
            dtype=dtype,
            device=torch.device("cuda"),
            requires_grad=False,
        ),
    )


def _get_forward_pass_fn(
    attn_fn: Callable[
        [
            Float[torch.Tensor, "B S D"],
            Float[torch.Tensor, "B T D"],
            Float[torch.Tensor, "B T D"],
        ],
        Float[torch.Tensor, "B S D"],
    ],
    Q: Float[torch.Tensor, "B S D"],
    K: Float[torch.Tensor, "B T D"],
    V: Float[torch.Tensor, "B T D"],
) -> Callable[[], Float[torch.Tensor, "B S D"]]:
    def _forward_pass_fn():
        return attn_fn(Q, K, V)

    return _forward_pass_fn


def _get_backward_pass_fn(
    attn_fn: Callable[
        [
            Float[torch.Tensor, "B S D"],
            Float[torch.Tensor, "B T D"],
            Float[torch.Tensor, "B T D"],
        ],
        Float[torch.Tensor, "B S D"],
    ],
    Q: Float[torch.Tensor, "B S D"],
    K: Float[torch.Tensor, "B T D"],
    V: Float[torch.Tensor, "B T D"],
    dO: Float[torch.Tensor, "B S D"],
) -> Callable[
    [],
    tuple[
        Float[torch.Tensor, "B S D"],
        Float[torch.Tensor, "B T D"],
        Float[torch.Tensor, "B T D"],
    ],
]:
    def _backward_pass_fn() -> tuple[
        Float[torch.Tensor, "B S D"],
        Float[torch.Tensor, "B T D"],
        Float[torch.Tensor, "B T D"],
    ]:
        attn_fn(Q, K, V).backward(dO)
        return Q.grad, K.grad, V.grad  # type: ignore

    return _backward_pass_fn


def _get_e2e_fn(
    attn_fn: Callable[
        [
            Float[torch.Tensor, "B S D"],
            Float[torch.Tensor, "B T D"],
            Float[torch.Tensor, "B T D"],
        ],
        Float[torch.Tensor, "B S D"],
    ],
    Q: Float[torch.Tensor, "B S D"],
    K: Float[torch.Tensor, "B T D"],
    V: Float[torch.Tensor, "B T D"],
) -> Callable[
    [],
    tuple[
        Float[torch.Tensor, "B S D"],
        Float[torch.Tensor, "B T D"],
        Float[torch.Tensor, "B T D"],
    ],
]:
    def _e2e_fn() -> tuple[
        Float[torch.Tensor, "B S D"],
        Float[torch.Tensor, "B T D"],
        Float[torch.Tensor, "B T D"],
    ]:
        o = attn_fn(Q, K, V)
        loss = o.sum()
        loss.backward()
        return Q.grad, K.grad, V.grad  # type: ignore

    return _e2e_fn


def _flash_attention_wrapper(
    Q: Float[torch.Tensor, "B S D"],
    K: Float[torch.Tensor, "B T D"],
    V: Float[torch.Tensor, "B T D"],
) -> Float[torch.Tensor, "B S D"]:
    return torch.compile(triton_functions.FlashAttention2.apply)(Q, K, V, True)  # type: ignore


def _regular_attention_wrapper(
    Q: Float[torch.Tensor, "B S D"],
    K: Float[torch.Tensor, "B T D"],
    V: Float[torch.Tensor, "B T D"],
) -> Float[torch.Tensor, "B S D"]:
    causal_mask = torch.tril(torch.ones((Q.shape[-2], K.shape[-2]))).to(
        dtype=torch.bool, device=Q.device
    )
    return torch.compile(functions.scaled_dot_product_attention)(
        q=Q, k=K, v=V, mask=causal_mask
    )


def main(argv):
    """Main function to run the FlashAttention2 benchmarking."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    batch_size, seq_len = 1, 128
    params = []
    while seq_len <= _max_seq_len.value:
        model_dim = 16
        while model_dim <= _max_model_dim.value:
            params.append((batch_size, seq_len, model_dim, torch.float32))
            params.append((batch_size, seq_len, model_dim, torch.bfloat16))
            model_dim *= 2
        seq_len *= 2

    logging.info(f"Running benchmarking for {len(params)} params combos...")
    all_results_df = {
        "seq_len": [],
        "model_dim": [],
        "dtype": [],
        "flash_attn_forward": [],
        "regular_attn_forward": [],
        "flash_attn_backward": [],
        "regular_attn_backward": [],
        "flash_attn_e2e": [],
        "regular_attn_e2e": [],
    }
    for batch_size, seq_len, model_dim, dtype in tqdm.tqdm(params):
        Q, K, V, dO = _get_test_inputs(batch_size, seq_len, seq_len, model_dim, dtype)
        flash_attn_forward_result = triton.testing.do_bench(
            _get_forward_pass_fn(_flash_attention_wrapper, Q, K, V),
            rep=_triton_benchmarking_rep.value,
            warmup=_triton_benchmarking_warmup.value,
        )
        flash_attn_backward_result = triton.testing.do_bench(
            _get_backward_pass_fn(_flash_attention_wrapper, Q, K, V, dO),
            rep=_triton_benchmarking_rep.value,
            warmup=_triton_benchmarking_warmup.value,
        )
        flash_attn_e2e_result = triton.testing.do_bench(
            _get_e2e_fn(_flash_attention_wrapper, Q, K, V),
            rep=_triton_benchmarking_rep.value,
            warmup=_triton_benchmarking_warmup.value,
        )
        regular_attn_forward_result = triton.testing.do_bench(
            _get_forward_pass_fn(_regular_attention_wrapper, Q, K, V),
            rep=_triton_benchmarking_rep.value,
            warmup=_triton_benchmarking_warmup.value,
        )
        regular_attn_backward_result = triton.testing.do_bench(
            _get_backward_pass_fn(_regular_attention_wrapper, Q, K, V, dO),
            rep=_triton_benchmarking_rep.value,
            warmup=_triton_benchmarking_warmup.value,
        )
        regular_attn_e2e_result = triton.testing.do_bench(
            _get_e2e_fn(_regular_attention_wrapper, Q, K, V),
            rep=_triton_benchmarking_rep.value,
            warmup=_triton_benchmarking_warmup.value,
        )
        all_results_df["seq_len"].append(seq_len)
        all_results_df["model_dim"].append(model_dim)
        all_results_df["dtype"].append(str(dtype))
        all_results_df["flash_attn_forward"].append(flash_attn_forward_result)
        all_results_df["regular_attn_forward"].append(regular_attn_forward_result)
        all_results_df["flash_attn_backward"].append(flash_attn_backward_result)
        all_results_df["regular_attn_backward"].append(regular_attn_backward_result)
        all_results_df["flash_attn_e2e"].append(flash_attn_e2e_result)
        all_results_df["regular_attn_e2e"].append(regular_attn_e2e_result)

    pd.DataFrame(all_results_df).to_pickle(
        f"{_output_path_prefix.value}_results.pickle"
    )
    logging.info("Benchmarking completed.")
    print(pd.DataFrame(all_results_df))


if __name__ == "__main__":
    app.run(main)
