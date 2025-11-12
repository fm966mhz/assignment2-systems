"""The transformer."""

import math

from dataclasses import dataclass

import torch

from jaxtyping import Int
from jaxtyping import Float
from torch import nn

from cs336_basics import layers as L


@dataclass(frozen=True)
class TransformerConfig:
    """Tranformer config."""

    vocab_size: int
    context_length: int
    num_layers: int
    num_heads: int

    rope_theta: float

    d_model: int
    d_ff_to_d_model: float | None = None
    d_ff: int | None = None


class TransformerLm(nn.Module):
    """Transformer LM."""

    def __init__(
        self,
        config: TransformerConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = L.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            device=device,
            dtype=dtype,
        )
        self.rope = L.Rope(
            theta=config.rope_theta,
            d_k=config.d_model // config.num_heads,
            max_seq_len=config.context_length,
            device=device,
        )
        transfomer_blocks = []
        assert (
            config.num_layers > 0
        ), f"Transformer number of layers must be positive, but got {config.num_layers}."
        for _ in range(config.num_layers):
            transfomer_blocks.append(
                L.TransfomerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=self._get_d_ff(
                        d_model=config.d_model,
                        d_ff_to_d_model=config.d_ff_to_d_model,
                        d_ff=config.d_ff,
                    ),
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                )
            )
        self.layers = nn.ModuleList(transfomer_blocks)
        self.ln_final = L.RMSNorm(d_model=config.d_model, device=device, dtype=dtype)
        self.lm_head = L.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self, input_tokens: Int[torch.LongTensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        """Run the forward pass."""
        activation = self.token_embeddings(input_tokens)
        token_positions = torch.arange(input_tokens.shape[-1])
        for transformer_block in self.layers:
            activation = transformer_block(
                in_features=activation, token_positions=token_positions
            )
        activation = self.ln_final(activation)
        return self.lm_head(activation)

    def _get_d_ff(
        self, d_model: int, d_ff_to_d_model: float | None, d_ff: int | None
    ) -> int:
        if d_ff is not None:
            return d_ff
        assert (
            d_ff_to_d_model is not None
        ), "d_ff and d_ff_to_d_model cannot be both None."
        return math.ceil(d_model * d_ff_to_d_model / 64) * 64
