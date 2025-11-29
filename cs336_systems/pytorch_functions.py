"""PyTorch functions."""

import math

import einops
import numpy as np
import torch

from jaxtyping import Float


class FlashAttention2Torch(torch.autograd.Function):
    """PyTorch implementation of FlashAttention2."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: Float[torch.Tensor, "... S D"],
        K: Float[torch.Tensor, "... T D"],
        V: Float[torch.Tensor, "... T D"],
        is_causal: bool = False,
        q_tile_size: int = 16,
        k_tile_size: int = 16,
    ) -> Float[torch.Tensor, "... S D"]:
        """Forward pass."""
        input_batch_shape = Q.shape[:-2]
        assert (
            K.shape[:-2] == input_batch_shape
        ), f"K batch shape {K.shape[:-2]} mismatch with Q {input_batch_shape}"
        assert (
            V.shape[:-2] == input_batch_shape
        ), f"V batch shape {V.shape[:-2]} mismatch with Q {input_batch_shape}"
        assert (
            K.shape == V.shape
        ), f"Mismatch shape between K ({K.shape}) and V ({V.shape})"
        S = Q.shape[-2]
        T = K.shape[-2]
        D = Q.shape[-1]
        Q = Q.view((-1, S, D))
        K = K.view((-1, T, D))
        V = V.view((-1, T, D))
        sqrt_D = np.sqrt(D)
        num_tiles_q = math.ceil(S / q_tile_size)
        num_tiles_kv = math.ceil(T / k_tile_size)
        O: Float[torch.Tensor, "B S D"] = torch.empty_like(Q)
        L: Float[torch.Tensor, "B S"] = torch.empty(Q.shape[:-1])
        for s in range(num_tiles_q):
            q_s: Float[torch.Tensor, "B q_tile_size D"] = Q[
                :, s * q_tile_size : min(S, (s + 1) * q_tile_size), :
            ]
            o_s: Float[torch.Tensor, "B q_tile_size D"] = torch.zeros_like(q_s)
            l_s: Float[torch.Tensor, "B q_tile_size"] = torch.zeros(q_s.shape[:-1])
            m_s: Float[torch.Tensor, "B q_tile_size"] = -1.0e6 * torch.ones(
                q_s.shape[:-1]
            )
            for t in range(num_tiles_kv):
                k_t: Float[torch.Tensor, "... k_tile_size D"] = K[
                    :, t * k_tile_size : (t + 1) * k_tile_size, :
                ]
                v_t: Float[torch.Tensor, "... k_tile_size D"] = V[
                    :, t * k_tile_size : (t + 1) * k_tile_size, :
                ]
                S_st = (
                    einops.einsum(
                        q_s,
                        k_t,
                        "B q_tile_size D, B k_tile_size D -> B q_tile_size k_tile_size",
                    )
                    / sqrt_D
                )
                m_s_old = m_s.clone().detach()
                m_s = torch.maximum(
                    m_s,
                    einops.reduce(
                        S_st,
                        "B q_tile_size k_tile_size -> B q_tile_size",
                        reduction="max",
                    ),
                )
                m_s_diff = m_s - m_s_old
                P_tilde_st = torch.exp(S_st - m_s.unsqueeze(dim=-1))
                l_s = einops.reduce(
                    P_tilde_st,
                    "B q_tile_size k_tile_size -> B q_tile_size",
                    reduction="sum",
                ) + l_s * torch.exp(-m_s_diff)
                o_s = einops.einsum(
                    P_tilde_st,
                    v_t,
                    "B q_tile_size k_tile_size, B k_tile_size D -> B q_tile_size D",
                ) + einops.einsum(
                    torch.exp(-m_s_diff),
                    o_s,
                    "B q_tile_size, B q_tile_size D -> B q_tile_size D",
                )
            O[:, s * q_tile_size : (s + 1) * q_tile_size, :] = einops.einsum(
                1.0 / l_s,
                o_s,
                "B q_tile_size, B q_tile_size D -> B q_tile_size D",
            )
            L[:, s * q_tile_size : (s + 1) * q_tile_size] = m_s + torch.log(l_s)
        ctx.save_for_backward(O, L, Q, K, V)
        ctx.is_causal = is_causal  # type: ignore
        return O.view(input_batch_shape + (S, D))

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: Float[torch.Tensor, "... S D"],
    ) -> tuple[
        Float[torch.Tensor, "... S D"],
        Float[torch.Tensor, "... T D"],
        Float[torch.Tensor, "... T D"],
        None,
        None,
        None,
    ]:
        """Backward pass."""
        (dO,) = grad_outputs
        O, L, Q, K, V = ctx.saved_tensors  # type: ignore
        input_batch_shape = dO.shape[:-2]
        S, D = dO.shape[-2:]
        T = K.shape[-2]
        dO = dO.view((-1, S, D))

        def _backward_impl(
            dO: Float[torch.Tensor, "B S D"],
            O: Float[torch.Tensor, "B S D"],
            L: Float[torch.Tensor, "B S"],
            Q: Float[torch.Tensor, "B S D"],
            K: Float[torch.Tensor, "B T D"],
            V: Float[torch.Tensor, "B T D"],
            sqrt_d: float,
            is_causal: bool = False,
        ) -> tuple[
            Float[torch.Tensor, "B S D"],
            Float[torch.Tensor, "B T D"],
            Float[torch.Tensor, "B T D"],
        ]:
            S = einops.einsum(Q, K, "B S D, B T D -> B S T") / sqrt_d
            q_seq_len, k_seq_len = S.shape[-2:]
            # Let S' = S + causal_mask, we have dS' = dS, so we only need to update S for the
            # situations with `is_causal = True`.
            if is_causal:
                S = torch.where(
                    torch.arange(q_seq_len, device=S.device)[None, :, None]
                    >= torch.arange(k_seq_len, device=S.device)[None, None, :],
                    S,
                    -1e6,
                )
            P = torch.exp(S - L.unsqueeze(dim=-1))
            dV = einops.einsum(P, dO, "B S T, B S D -> B T D")
            dP = einops.einsum(dO, V, "B S D, B T D -> B S T")
            dS = P * (dP - einops.reduce(O * dO, "B S D -> B S 1", reduction="sum"))
            dQ = einops.einsum(dS, K, "B S T, B T D -> B S D") / sqrt_d
            dK = einops.einsum(dS, Q, "B S T, B S D -> B T D") / sqrt_d
            return dQ, dK, dV

        compiled_backward_impl = torch.compile(_backward_impl)
        dQ, dK, dV = compiled_backward_impl(
            dO, O, L, Q, K, V, np.sqrt(D), ctx.is_causal  # type:ignore
        )
        return (
            dQ.view(input_batch_shape + (S, D)),
            dK.view(input_batch_shape + (T, D)),
            dV.view(input_batch_shape + (T, D)),
            None,
            None,
            None,
        )
