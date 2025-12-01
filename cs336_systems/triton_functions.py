"""Triton functions."""

import einops
import numpy as np
from pandas._libs.tslibs import is_unitless
import torch
import triton
import triton.language as tl

from jaxtyping import Float
from torch.library import triton_op
from torch.library import wrap_triton


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


def flash_attention_get_configs():
    """Gets the configs for the flash attention forward and backward passes.

    Turned out that the SRAM on my 5080 if very limited. The max tile size I can use is 16 as tested
    in `flash_attention_benchmarking_main.py`.
    """
    return [
        triton.Config(
            {"Q_TILE_SIZE": 2**i, "K_TILE_SIZE": 2**i}, num_stages=3, num_warps=4
        )
        for i in range(4, 5)
    ]


@triton.autotune(
    configs=flash_attention_get_configs(),
    key=["S", "T"],
)
@triton.jit
def flash_fwd_kernel(
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
    S: int,
    T: int,
    D: int,
    # scale factor is 1.0 / sqrt(D). We use a `tl.constexpr` instead of a `float` because the latter
    # doesn't work well with `torch.compile`. Somehow with a float, the program will crash when
    # launching the CUDA kernal with the error of
    # "TypeError: 'float' object cannot be interpreted as an integer".
    scale_factor: tl.constexpr,
    is_causal: tl.constexpr,
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
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_k_b,
        shape=(T, D),
        strides=(stride_k_t, stride_k_d),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_v_b,
        shape=(T, D),
        strides=(stride_v_t, stride_v_d),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_o_b,
        shape=(S, D),
        strides=(stride_o_s, stride_o_d),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
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
    q_indices = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    # Early exit if the current max Q index is smaller than the smallest k index in the current K
    # tile. Mathematically, this means `k_tile_index` needs to satisfy the following inequality:
    #
    #     k_tile_index * K_TILE_SIZE < (q_tile_index + 1) * Q_TILE_SIZE
    #
    # in order for there to be any meaningful computation for that particular k tile.
    max_k_index = T
    if is_causal:
        max_k_index = tl.minimum(max_k_index, (q_tile_index + 1) * Q_TILE_SIZE)
    for k_tile_index in tl.range(tl.cdiv(max_k_index, K_TILE_SIZE)):
        k_t = tl.trans(
            tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero"), (1, 0)
        )  # D K_TILE_SIZE
        v_t = tl.load(
            V_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # K_TILE_SIZE D
        k_indices = k_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        mask_t = k_indices < T
        S_st = tl.dot(q_s, k_t) * scale_factor  # Q_TILE_SIZE K_TILE_SIZE
        S_st = tl.where(mask_t[None, :], S_st, -1.0e6)
        if is_causal:
            S_st = tl.where(q_indices[:, None] >= k_indices[None, :], S_st, -1.0e6)
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


@triton.autotune(
    configs=flash_attention_get_configs(),
    key=["S", "T"],
)
@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr: tl.tensor,
    K_ptr: tl.tensor,
    V_ptr: tl.tensor,
    O_ptr: tl.tensor,
    L_ptr: tl.tensor,
    dO_ptr: tl.tensor,
    dQ_ptr: tl.tensor,
    # Q strides.
    stride_q_b: int,
    stride_q_s: int,
    stride_q_d: int,
    # K strides.
    stride_k_b: int,
    stride_k_t: int,
    stride_k_d: int,
    # V strides.
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
    # dO strides.
    stride_do_b: int,
    stride_do_s: int,
    stride_do_d: int,
    # dQ strides.
    stride_dq_b: int,
    stride_dq_s: int,
    stride_dq_d: int,
    S: int,
    T: int,
    D: int,
    # scale factor is 1.0 / sqrt(D). We use a `tl.constexpr` instead of a `float` because the latter
    # doesn't work well with `torch.compile`. Somehow with a float, the program will crash when
    # launching the CUDA kernal with the error of
    # "TypeError: 'float' object cannot be interpreted as an integer".
    scale_factor: tl.constexpr,
    is_causal: tl.constexpr,
    D_BLOCK_SHAPE: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """Flash Attention 2 backward pass for dQ."""
    q_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_q_b,
        shape=(S, D),
        strides=(stride_q_s, stride_q_d),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_k_b,
        shape=(T, D),
        strides=(stride_k_t, stride_k_d),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_v_b,
        shape=(T, D),
        strides=(stride_v_t, stride_v_d),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_o_b,
        shape=(S, D),
        strides=(stride_o_s, stride_o_d),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_l_b,
        shape=(S,),
        strides=(stride_l_s,),
        offsets=(Q_TILE_SIZE * q_tile_index,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_do_b,
        shape=(S, D),
        strides=(stride_do_s, stride_do_d),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dq_b,
        shape=(S, D),
        strides=(stride_dq_s, stride_dq_d),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    q_s = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o_s = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    l_s = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    do_s = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dq_s = tl.zeros((Q_TILE_SIZE, D_BLOCK_SHAPE), dtype=tl.float32)
    q_indices = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    max_k_index = T
    if is_causal:
        max_k_index = tl.minimum(max_k_index, (q_tile_index + 1) * Q_TILE_SIZE)
    for k_tile_index in tl.range(tl.cdiv(max_k_index, K_TILE_SIZE)):
        k_t = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_t = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        k_indices = k_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        S_st = tl.dot(q_s, tl.trans(k_t, (1, 0))) * scale_factor
        S_st = tl.where(k_indices < T, S_st, -1.0e6)
        if is_causal:
            S_st = tl.where(q_indices[:, None] >= k_indices[None, :], S_st, -1.0e6)
        P_st = tl.exp(S_st - tl.expand_dims(l_s, axis=-1))
        dP_st = tl.dot(do_s, tl.trans(v_t, (1, 0)))
        D_s = tl.sum(o_s * do_s, axis=-1, keep_dims=True)
        dS_st = P_st * (dP_st - D_s)
        dq_s += tl.dot(dS_st.cast(k_t.dtype), k_t) * scale_factor

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(
        dQ_block_ptr, dq_s.cast(dQ_block_ptr.dtype.element_ty), boundary_check=(0, 1)
    )


@triton.autotune(
    configs=flash_attention_get_configs(),
    key=["S", "T"],
)
@triton.jit
def flash_bwd_dk_dv_kernel(
    Q_ptr: tl.tensor,
    K_ptr: tl.tensor,
    V_ptr: tl.tensor,
    O_ptr: tl.tensor,
    L_ptr: tl.tensor,
    dO_ptr: tl.tensor,
    dK_ptr: tl.tensor,
    dV_ptr: tl.tensor,
    # Q strides.
    stride_q_b: int,
    stride_q_s: int,
    stride_q_d: int,
    # K strides.
    stride_k_b: int,
    stride_k_t: int,
    stride_k_d: int,
    # V strides.
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
    # dO strides.
    stride_do_b: int,
    stride_do_s: int,
    stride_do_d: int,
    # dK strides.
    stride_dk_b: int,
    stride_dk_t: int,
    stride_dk_d: int,
    # dV strides.
    stride_dv_b: int,
    stride_dv_t: int,
    stride_dv_d: int,
    S: int,
    T: int,
    D: int,
    # scale factor is 1.0 / sqrt(D). We use a `tl.constexpr` instead of a `float` because the latter
    # doesn't work well with `torch.compile`. Somehow with a float, the program will crash when
    # launching the CUDA kernal with the error of
    # "TypeError: 'float' object cannot be interpreted as an integer".
    scale_factor: tl.constexpr,
    is_causal: tl.constexpr,
    D_BLOCK_SHAPE: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """Flash Attention 2 backward pass for dQ."""
    k_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Similar to the forward pass, we can skip tiles that are not needed for the current k tile due
    # to the causal mask.
    min_q_tile_index = 0
    if is_causal:
        min_q_tile_index = tl.floor(k_tile_index * K_TILE_SIZE / Q_TILE_SIZE).to(
            tl.int32
        )

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_q_b,
        shape=(S, D),
        strides=(stride_q_s, stride_q_d),
        offsets=(min_q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_k_b,
        shape=(T, D),
        strides=(stride_k_t, stride_k_d),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_v_b,
        shape=(T, D),
        strides=(stride_v_t, stride_v_d),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_o_b,
        shape=(S, D),
        strides=(stride_o_s, stride_o_d),
        offsets=(min_q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_l_b,
        shape=(S,),
        strides=(stride_l_s,),
        offsets=(min_q_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_do_b,
        shape=(S, D),
        strides=(stride_do_s, stride_do_d),
        offsets=(min_q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dk_b,
        shape=(T, D),
        strides=(stride_dk_t, stride_dk_d),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dv_b,
        shape=(T, D),
        strides=(stride_dv_t, stride_dv_d),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_BLOCK_SHAPE),
        order=(1, 0),
    )
    k_t = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v_t = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dk_t = tl.zeros((K_TILE_SIZE, D_BLOCK_SHAPE), dtype=tl.float32)
    dv_t = tl.zeros((K_TILE_SIZE, D_BLOCK_SHAPE), dtype=tl.float32)
    k_indices = k_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    for q_tile_index in tl.range(min_q_tile_index, tl.cdiv(S, Q_TILE_SIZE)):
        q_s = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        o_s = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
        l_s = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        do_s = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        q_indices = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        S_st = tl.dot(q_s, tl.trans(k_t, (1, 0))) * scale_factor
        S_st = tl.where(k_indices < T, S_st, -1.0e6)
        if is_causal:
            S_st = tl.where(q_indices[:, None] >= k_indices[None, :], S_st, -1.0e6)
        P_st = tl.exp(S_st - l_s[:, None])
        dP_st = tl.dot(do_s, tl.trans(v_t, (1, 0)))
        D_s = tl.sum(o_s * do_s, axis=-1, keep_dims=True)
        dS_st = P_st * (dP_st - D_s)
        dk_t += tl.dot(tl.trans(dS_st.cast(q_s.dtype), (1, 0)), q_s) * scale_factor
        dv_t += tl.dot(tl.trans(P_st.cast(do_s.dtype), (1, 0)), do_s)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))

    tl.store(
        dK_block_ptr, dk_t.cast(dK_block_ptr.dtype.element_ty), boundary_check=(0, 1)
    )
    tl.store(
        dV_block_ptr, dv_t.cast(dV_block_ptr.dtype.element_ty), boundary_check=(0, 1)
    )


class FlashAttention2(torch.autograd.Function):
    """Triton implementation of FlashAttention2."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: Float[torch.Tensor, "... S D"],
        K: Float[torch.Tensor, "... T D"],
        V: Float[torch.Tensor, "... T D"],
        is_causal: bool = False,
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
        scale_factor = 1.0 / (D**0.5)
        # 16 is the min size of the inner dimension for dot product.
        d_block_shape = max(16, triton.next_power_of_2(D))  # type: ignore
        O = torch.empty_like(Q, dtype=Q.dtype).to(Q.device)
        # CRITICAL NOTE FOR USING `torch.zeros` instead of `torch.empty`.
        # `torch.empty((B, S))` creates a FAKE TENSOR that has no dependency in the main computation
        # graph. It will be aggressively optimized away by `torch.compile` (Inductor) and never gets
        # materialized. When the Triton kernel tries to write to it, it will cause an illegal memory
        # access error and crash the program!
        L = torch.zeros(Q.shape[:-1], dtype=Q.dtype).to(Q.device)
        grid = lambda meta: (triton.cdiv(S, meta["Q_TILE_SIZE"]), Q.shape[0])
        flash_fwd_kernel[grid](
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
            S,  # type: ignore
            T,  # type: ignore
            D,  # type: ignore
            scale_factor=scale_factor,  # type: ignore
            is_causal=is_causal,  # type: ignore
            D_BLOCK_SHAPE=d_block_shape,  # type:ignore
        )
        # print(f"Best config: {flash_fwd_kernal.best_config}")
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
    ]:
        """Backward pass."""
        (dO,) = grad_outputs
        O, L, Q, K, V = ctx.saved_tensors  # type: ignore
        input_batch_shape = dO.shape[:-2]
        S, D = dO.shape[-2:]
        T = K.shape[-2]
        dO = dO.view((-1, S, D))
        dQ = torch.empty_like(Q, dtype=Q.dtype).to(Q.device)
        dK = torch.empty_like(K, dtype=K.dtype).to(K.device)
        dV = torch.empty_like(V, dtype=V.dtype).to(V.device)
        scale_factor = 1.0 / (D**0.5)
        # 16 is the min size of the inner dimension for dot product.
        d_block_shape = max(16, triton.next_power_of_2(D))  # type: ignore
        grid_dq = lambda meta: (triton.cdiv(S, meta["Q_TILE_SIZE"]), Q.shape[0])
        flash_bwd_dq_kernel[grid_dq](
            Q,  # type: ignore
            K,  # type: ignore
            V,  # type: ignore
            O,  # type: ignore
            L,  # type: ignore
            dO,  # type: ignore
            dQ,  # type: ignore
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
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            S,  # type: ignore
            T,  # type: ignore
            D,  # type: ignore
            scale_factor=scale_factor,  # type: ignore
            is_causal=ctx.is_causal,  # type: ignore
            D_BLOCK_SHAPE=d_block_shape,  # type:ignore
        )
        grid_dk_dv = lambda meta: (triton.cdiv(T, meta["K_TILE_SIZE"]), Q.shape[0])
        flash_bwd_dk_dv_kernel[grid_dk_dv](
            Q,  # type: ignore
            K,  # type: ignore
            V,  # type: ignore
            O,  # type: ignore
            L,  # type: ignore
            dO,  # type: ignore
            dK,  # type: ignore
            dV,  # type: ignore
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
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dK.stride(0),
            dK.stride(1),
            dK.stride(2),
            dV.stride(0),
            dV.stride(1),
            dV.stride(2),
            S,  # type: ignore
            T,  # type: ignore
            D,  # type: ignore
            scale_factor=scale_factor,  # type: ignore
            is_causal=ctx.is_causal,  # type: ignore
            D_BLOCK_SHAPE=d_block_shape,  # type:ignore
        )
        return (
            dQ.view(input_batch_shape + (S, D)),
            dK.view(input_batch_shape + (T, D)),
            dV.view(input_batch_shape + (T, D)),
            None,
        )
