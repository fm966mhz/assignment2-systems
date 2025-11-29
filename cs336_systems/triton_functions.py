"""Triton functions."""

import einops
import numpy as np
from pandas._libs.tslibs import is_unitless
import torch
import triton
import triton.language as tl

from jaxtyping import Float


@triton.jit
def weighted_sum_fwd(
    x_ptr: tl.tensor,
    w_ptr: tl.tensor,
    output_ptr: tl.tensor,
    x_stride_row: int,
    x_stride_dim: int,
    weight_stride_row: int,
    output_stride_row: int,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """Weighted sum forward pass."""
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        # Note this is different from the provided code in the assignment writeup.
        # The order should be from major to minor. In the case of PyTorch tensors, it is row-major,
        # which can be easily visualized.
        order=(0, 1),
    )

    weight_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(D,),
        strides=(weight_stride_row,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=x_ptr.dtype.element_ty)  # type: ignore

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        output += tl.sum(row * weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    tl.store(output_block_ptr, output)


@triton.jit
def weighted_sum_bwd(
    x_ptr: tl.tensor,
    w_ptr: tl.tensor,
    grad_output_ptr: tl.tensor,
    grad_x_ptr: tl.tensor,
    partial_grad_weight_ptr: tl.tensor,
    stride_xr: int,
    stride_xd: int,
    stride_weight_d: int,
    stride_grad_output_r: int,
    stride_grad_x_r: int,
    stride_grad_x_d: int,
    stride_partial_grad_weight_block: int,
    stride_partial_grad_weight_d: int,
    NUM_ROWS: tl.constexpr,
    D: tl.constexpr,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """Weighted sum backward pass."""
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_grad_output_r,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(0, 1),
    )
    weight_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(D,),
        strides=(stride_weight_d,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_grad_x_r, stride_grad_x_d),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(0, 1),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_partial_grad_weight_block, stride_partial_grad_weight_d),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(0, 1),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(
            grad_output_block_ptr, boundary_check=(0,), padding_option="zero"
        )
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(
            partial_grad_weight_block_ptr,
            grad_weight_row,
            boundary_check=(1,),
        )

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance(
            (0, D_TILE_SIZE)
        )
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    """Weighted sum function."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Float[torch.Tensor, "... ROWS D"],
        weight: Float[torch.Tensor, "D"],
    ) -> Float[torch.Tensor, "... ROWS"]:
        """Forward pass."""
        D, output_dims = x.shape[-1], x.shape[:-1]

        assert (
            len(weight.shape) == 1 and weight.shape[0] == D
        ), "Weight must be a 1D tensor of shape (D,)"
        assert x.is_cuda and weight.is_cuda, "x and weight must be on GPU"
        assert x.is_contiguous(), "x must be contiguous"

        input_shape = x.shape
        x = einops.rearrange(x, "... ROWS D -> (... ROWS) D")
        ctx.save_for_backward(x, weight)

        ctx.D_TILE_SIZE = triton.next_power_of_2(D)  # type: ignore
        ctx.ROWS_TILE_SIZE = 16  # type: ignore
        ctx.input_shape = input_shape  # type: ignore

        y = torch.empty(output_dims, device=x.device, dtype=x.dtype)

        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](  # type: ignore
            x,  # type: ignore
            weight,  # type: ignore
            y,  # type: ignore
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows,  # type: ignore
            D=D,  # type: ignore
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,  # type: ignore
            D_TILE_SIZE=ctx.D_TILE_SIZE,  # type: ignore
        )

        return y.view(input_shape[:-1])

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: Float[torch.Tensor, "... ROWS"],
    ) -> tuple[Float[torch.Tensor, "... ROWS D"], Float[torch.Tensor, "D"]]:
        """Backward pass."""
        (grad_output,) = grad_outputs
        x, weight = ctx.saved_tensors  # type: ignore
        assert len(x.shape) == 2, "x must be a 2D tensor"
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE  # type: ignore
        n_rows, D = x.shape

        partial_grad_weight = torch.empty(
            (int(triton.cdiv(n_rows, ROWS_TILE_SIZE)), D),
            device=x.device,
            dtype=x.dtype,
        )
        grad_x = torch.empty_like(x, dtype=x.dtype)
        grad_output = einops.rearrange(grad_output, "... ROWS -> (... ROWS)")

        assert grad_output.shape == (
            n_rows,
        ), "grad_output must be a 1D tensor of shape (n_rows,)"

        weighted_sum_bwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x,
            weight,
            grad_output,  # type: ignore
            grad_x,  # type: ignore
            partial_grad_weight,  # type: ignore
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            grad_output.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(dim=0)
        input_shape = ctx.input_shape  # type: ignore
        return grad_x.view(input_shape), grad_weight


@triton.jit
def flash_fwd_kernal(
    Q_ptr: tl.tensor,
    K_ptr: tl.tensor,
    V_ptr: tl.tensor,
    O_ptr: tl.tensor,
    L_ptr: tl.tensor,
    # Q strides.
    stride_q_b: int,
    stride_q_s: int,
    stride_q_d: int,
    # K strides.
    stride_k_b: int,
    stride_k_t: int,
    stride_k_d: int,
    # V strides
    stride_v_b: int,
    stride_v_t: int,
    stride_v_d: int,
    # O strides.
    stride_o_b: int,
    stride_o_s: int,
    stride_o_d: int,
    # L strides.
    stride_l_b: int,
    stride_l_s: int,
    sqrt_D: float,
    is_causal: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    D_BLOCK_SHAPE: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """Flash Attention 2 foward pass."""
    q_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_q_b,
        shape=(S, D),
        strides=(stride_q_s, stride_q_d),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK_SHAPE),
        order=(0, 1),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_k_b,
        shape=(T, D),
        strides=(stride_k_t, stride_k_d),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK_SHAPE),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_v_b,
        shape=(T, D),
        strides=(stride_v_t, stride_v_d),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK_SHAPE),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_o_b,
        shape=(S, D),
        strides=(stride_o_s, stride_o_d),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK_SHAPE),
        order=(0, 1),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_l_b,
        shape=(S,),
        strides=(stride_l_s,),
        offsets=(Q_TILE_SIZE * q_tile_index,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q_s = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o_s = tl.zeros((Q_TILE_SIZE, D_BLOCK_SHAPE), dtype=tl.float32)
    l_s = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_s = tl.full((Q_TILE_SIZE,), -1.0e6, dtype=tl.float32)
    if is_causal:
        q_indices = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    for k_tile_index in range(tl.cdiv(T, K_TILE_SIZE)):
        k_t = tl.trans(
            tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero"), (1, 0)
        )  # D K_TILE_SIZE
        v_t = tl.load(
            V_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # K_TILE_SIZE D
        S_st = tl.dot(q_s, k_t) / sqrt_D  # Q_TILE_SIZE K_TILE_SIZE
        if is_causal:
            k_indices = k_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask_matrix = tl.where(
                q_indices[:, None] >= k_indices[None, :], 0.0, -1.0e6
            )
            S_st += causal_mask_matrix
        m_s_old = m_s
        m_s = tl.maximum(m_s, tl.max(S_st, axis=-1))
        exp_neg_m_s_diff = tl.exp(m_s_old - m_s)  # Q_TILE_SIZE
        P_tilde_st = tl.exp(
            S_st - tl.expand_dims(m_s, axis=-1)
        )  # Q_TILE_SIZE K_TILE_SIZE
        l_s = tl.sum(P_tilde_st, axis=-1) + l_s * exp_neg_m_s_diff
        o_s = tl.dot(P_tilde_st.cast(v_t.dtype), v_t) + o_s * tl.expand_dims(
            exp_neg_m_s_diff, axis=-1
        )  # Q_TILE_SIZE D

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    o_s = o_s / tl.expand_dims(l_s, axis=-1)
    l_s = m_s + tl.log(l_s)

    # boundary_check is crucial since `D_BLOCK_SHAPE` can be greater than the `D` shape of
    # `O_block_ptr`.
    tl.store(O_block_ptr, o_s.cast(O_block_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l_s.cast(L_block_ptr.dtype.element_ty), boundary_check=(0,))


class FlashAttention2(torch.autograd.Function):
    """Triton implementation of FlashAttention2."""

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
        num_tiles_q = triton.cdiv(S, q_tile_size)
        # 16 is the min size of the inner dimension for dot product.
        d_block_shape = max(16, triton.next_power_of_2(D))  # type: ignore
        O = torch.empty_like(Q).to(Q.device)
        L = torch.empty(Q.shape[:-1]).to(Q.device)
        flash_fwd_kernal[(num_tiles_q, Q.shape[0])](
            Q,  # type: ignore
            K,  # type: ignore
            V,  # type: ignore
            O,  # type: ignore
            L,  # type: ignore
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            sqrt_D,
            is_causal,  # type: ignore
            S,  # type: ignore
            T,  # type: ignore
            D,  # type: ignore
            d_block_shape,  # type:ignore
            q_tile_size,  # type: ignore
            k_tile_size,  # type: ignore
        )
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
