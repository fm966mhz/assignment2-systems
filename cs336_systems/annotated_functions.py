"""Annotated functions."""

import einops
import numpy as np
import torch
import torch.cuda.nvtx as nvtx

from jaxtyping import Bool
from jaxtyping import Float

from cs336_basics.functions import softmax


def annotated_scaled_dot_product_attention(
    q: Float[torch.Tensor, "... queries_len d_k"],
    k: Float[torch.Tensor, "... keys_len d_k"],
    v: Float[torch.Tensor, "... values_len d_v"],
    mask: Bool[torch.Tensor, "... queries_len keys_len"] | None = None,
) -> Float[torch.Tensor, "... queries_len d_v"]:
    """Annotated scaled dot product attention."""
    d_k = q.shape[-1]
    with nvtx.range("compute_scaled_dot_product"):
        scaled_dot_product = einops.einsum(
            q, k, "... queries_len d_k, ... keys_len d_k -> ... queries_len keys_len"
        ) / np.sqrt(d_k)
    if mask is not None:
        scaled_dot_product.masked_fill_(~mask, float("-inf"))
    with nvtx.range("compute_attention_softmax"):
        softmax_result = softmax(scaled_dot_product, dim=-1)
    with nvtx.range("matmul_with_v"):
        output = einops.einsum(
            softmax_result,
            v,
            "... queries_len keys_len, ... keys_len d_v -> ... queries_len d_v",
        )
    return output
