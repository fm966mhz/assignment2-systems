"""Decode."""

import torch

from jaxtyping import Int
from jaxtyping import Float
from torch import nn

import cs336_basics.functions as F


def decode(
    input_tokens: Int[torch.Tensor, "seq_len"],
    model: nn.Module,
    stop_token: int,
    max_output_length: int,
    temperature: float,
    top_p: float,
    eps: float = 1e-8,
    max_model_context_length: int = 1024,
    device: torch.device | str | None = None,
) -> Int[torch.Tensor, "seq_len"]:
    """Runs decoding."""
    assert temperature >= 0, f"Temperator must be non-negative, but got {temperature}."
    assert 0 <= top_p <= 1, f"top_p must be in [0, 1], but got {top_p}."
    output = []
    # TODO(djwenren): this clearly calls for KV cache.
    if device is not None:
        model = model.to(device)
        input_tokens = input_tokens.to(device)
    for _ in range(max_output_length):
        model_input_tokens = torch.cat(
            (input_tokens, torch.tensor(output, dtype=torch.int64, device=device))
        )
        if model_input_tokens.shape[-1] > max_model_context_length:
            model_input_tokens = model_input_tokens[-max_model_context_length:]
        logits: Float[torch.Tensor, "vocab_size"] = model(model_input_tokens)[-1]
        probs = F.softmax(x=logits / (temperature + eps), dim=-1)
        new_token_idx = nucleus_sample(probs, top_p).cpu().item()
        output.append(new_token_idx)
        if new_token_idx == stop_token:
            break
    return torch.tensor(output, dtype=torch.int64, device=device)


def nucleus_sample(
    input_probs: Float[torch.Tensor, "vocab_size"], top_p: float
) -> Int[torch.Tensor, ""]:
    """Runs one nucleus sampling."""
    sorted_probs, token_ids = torch.sort(input_probs, dim=-1, descending=True)
    cdf_probs = torch.cumsum(sorted_probs, dim=-1)
    last_eligible_idx = torch.searchsorted(cdf_probs, top_p, right=False)
    normalized_probs = sorted_probs[: last_eligible_idx + 1]
    normalized_probs = normalized_probs / normalized_probs.sum()
    return token_ids[torch.distributions.Categorical(probs=normalized_probs).sample()]
