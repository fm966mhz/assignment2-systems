"""Layers."""

import einops
import numpy as np
import torch

from jaxtyping import Int
from jaxtyping import Float
from torch import nn

from cs336_basics.functions import scaled_dot_product_attention
from cs336_basics.functions import silu


class Linear(nn.Module):
    """Linear module."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, dtype=dtype, device=device),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_out"]:
        """Forward pass."""
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Embedding module."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device),
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(
        self, token_ids: Int[torch.LongTensor, "..."]
    ) -> Float[torch.Tensor, "... embedding_dim"]:
        """Foward pass."""
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm module."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[torch.Tensor, "... d_model"]):
        """Forward pass."""
        assert x.shape[-1] == self.weight.shape[0], (
            f"Input shape {x.shape}'s last dimension is different from what this module is expected"
            " {self.gain.shape}"
        )
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.eps + einops.reduce(x**2, "... d_model -> ...", "mean"))
        result = einops.einsum(
            x, 1.0 / rms, self.weight, "... d_model, ..., d_model -> ... d_model"
        )
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """SwiGLU module."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.in_projection_layer_1 = Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.in_projection_layer_3 = Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.out_projection_layer_2 = Linear(
            in_features=d_ff, out_features=d_model, device=device, dtype=dtype
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        """Forward."""
        out_1 = self.in_projection_layer_1(x)
        out_3 = self.in_projection_layer_3(x)
        return self.out_projection_layer_2(silu(out_1) * out_3)


class Rope(nn.Module):
    """RoPE module."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        theta_tensor = einops.einsum(
            # Positions are zero-indexed.
            torch.arange(max_seq_len, device=device),
            1.0 / theta ** (2 * torch.arange(d_k // 2, device=device) / d_k),
            "seq_len, half_d_k -> seq_len half_d_k",
        )
        cosine_matrix = einops.einsum(
            torch.cos(theta_tensor),
            torch.tensor([[1.0, 0], [0, 1.0]], device=device),
            "seq_len half_d_k, r_out r_in -> seq_len half_d_k r_out r_in",
        )
        sine_matrix = einops.einsum(
            torch.sin(theta_tensor),
            torch.tensor([[0, -1.0], [1.0, 0]], device=device),
            "seq_len half_d_k, r_out r_in -> seq_len half_d_k r_out r_in",
        )
        self.register_buffer(
            "rope_matrix", cosine_matrix + sine_matrix, persistent=False
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """Forward pass."""
        position_embeddings: Float[torch.Tensor, "... seq_len half_d_k r_out r_in"] = (
            self.rope_matrix[token_positions]
        )
        x_rearanged = einops.rearrange(
            x, "... seq_len (half_d_k r_in) -> ... seq_len half_d_k r_in", r_in=2
        )
        output = einops.einsum(
            x_rearanged,
            position_embeddings,
            (
                "... seq_len half_d_k r_in, ... seq_len half_d_k r_out r_in -> "
                "... seq_len half_d_k r_out"
            ),
        )
        return einops.rearrange(
            output,
            "... seq_len half_d_out r_out -> ... seq_len (half_d_out r_out)",
        )


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention modeul."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: Rope | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model {d_model} must be divisible by num_heads {num_heads}."
        d_head = d_model // num_heads
        self.num_heads = num_heads
        self.d_head = d_head
        self.rope = rope
        self.combined_in_projection = Linear(
            in_features=d_model, out_features=3 * d_model, device=device, dtype=dtype
        )
        self.out_projection = Linear(
            in_features=d_model, out_features=d_model, device=device, dtype=dtype
        )
        self.device = device

    def forward(
        self,
        in_features: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """Runs forward pass."""
        qkv_combined = einops.rearrange(
            self.combined_in_projection(in_features),
            "... seq_len (num_vars d_model) -> ... seq_len num_vars d_model",
            num_vars=3,
        )
        qkv_combined = einops.rearrange(
            qkv_combined, " ... num_vars d_model -> ... d_model num_vars"
        )
        q, k, v = qkv_combined[..., 0], qkv_combined[..., 1], qkv_combined[..., 2]
        q = einops.rearrange(
            q,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
        )
        k = einops.rearrange(
            k,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
        )
        v = einops.rearrange(
            v,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
        )
        seq_len = q.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).to(
            torch.bool
        )
        if token_positions is not None:
            assert self.rope is not None, (
                "`token_positions provided but the MultiHeadSelfAttention module doesn't a have "
                "RoPE config."
            )
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        scaled_dot_product_result = scaled_dot_product_attention(
            q=q, k=k, v=v, mask=mask
        )
        return self.out_projection(
            einops.rearrange(
                scaled_dot_product_result,
                "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)",
            )
        )


class TransfomerBlock(nn.Module):
    """The Transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: Rope | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.rms_norm_pre_attn = RMSNorm(
            d_model=d_model, eps=eps, device=device, dtype=dtype
        )
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )
        self.rms_norm_pre_ff = RMSNorm(
            d_model=d_model, eps=eps, device=device, dtype=dtype
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        in_features: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """Runs the forward pass."""
        assert (
            in_features.dim() >= 2
        ), f"`in_features` must have at least 2 dimensions, but got {in_features.dim()}"
        activation = self.rms_norm_pre_attn(in_features)
        activation = self.attn(in_features=activation, token_positions=token_positions)
        post_attn_block_activation = in_features + activation
        activation = self.rms_norm_pre_ff(post_attn_block_activation)
        activation = self.ffn(activation)
        return post_attn_block_activation + activation
