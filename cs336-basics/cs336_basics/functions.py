"""Functions."""

import einops
import numpy as np
import torch

from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int


def silu(x: Float[torch.Tensor, "..."]):
    """Runs SiLU."""
    return x * torch.sigmoid(x)


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    """Runs softmax."""
    x_diffed = x - x.max(dim=dim, keepdim=True)[0]
    x_exped = torch.exp(x_diffed)
    x_exped_sumed = torch.sum(x_exped, dim=dim, keepdim=True)
    return x_exped / x_exped_sumed


def scaled_dot_product_attention(
    q: Float[torch.Tensor, "... queries_len d_k"],
    k: Float[torch.Tensor, "... keys_len d_k"],
    v: Float[torch.Tensor, "... values_len d_v"],
    mask: Bool[torch.Tensor, "... queries_len keys_len"] | None = None,
) -> Float[torch.Tensor, "... queries_len d_v"]:
    """Runs scaled dot product attention."""
    d_k = q.shape[-1]
    scaled_dot_product = einops.einsum(
        q, k, "... queries_len d_k, ... keys_len d_k -> ... queries_len keys_len"
    ) / np.sqrt(d_k)
    if mask is not None:
        scaled_dot_product.masked_fill_(~mask, float("-inf"))
    return einops.einsum(
        softmax(scaled_dot_product, dim=-1),
        v,
        "... queries_len keys_len, ... keys_len d_v -> ... queries_len d_v",
    )


def cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, ""]:
    """Computes the cross entropy."""
    logits_diffed: Float[torch.Tensor, "... vocab_size"] = (
        logits - logits.max(dim=-1, keepdim=True)[0]
    )
    return torch.mean(
        -torch.gather(logits_diffed, dim=-1, index=targets.unsqueeze(dim=-1)).squeeze()
        + torch.log(torch.sum(torch.exp(logits_diffed), dim=-1))
    )
