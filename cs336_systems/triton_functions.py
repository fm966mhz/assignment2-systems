"""Triton functions."""

import einops
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
