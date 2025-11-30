"""Debug Triton and Torch compile."""

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import einops
import torch
import triton
import triton.language as tl

from jaxtyping import Float


@triton.jit
def load_and_store_kernel(
    Q_ptr: tl.tensor,
    O_ptr: tl.tensor,
    L_ptr: tl.tensor,
    stride_q_b,
    stride_q_s,
    stride_q_d,
    stride_o_b,
    stride_o_s,
    stride_o_d,
    stride_l_b,
    stride_l_s,
    S,
    D,
    D_BLOCK_SHAPE: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    """Load and store kernel."""
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
        offsets=(q_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    q_s = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o_s = tl.zeros((Q_TILE_SIZE, D_BLOCK_SHAPE), dtype=tl.float32)
    l_s = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    tl.store(O_block_ptr, o_s.cast(O_block_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l_s.cast(L_block_ptr.dtype.element_ty), boundary_check=(0,))


class LoadAndStoreFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: Float[torch.Tensor, "B S D"],
        q_tile_size: int = 16,
    ) -> tuple[Float[torch.Tensor, "B S D"], Float[torch.Tensor, "B S"]]:
        B, S, D = Q.shape
        print(B, S, D)
        O = torch.empty_like(Q).contiguous().to(Q.device)
        # CRITICAL NOTE FOR USING `torch.zeros` instead of `torch.empty`.
        # `torch.empty((B, S))` creates a FAKE TENSOR that has no dependency in the main computation
        # graph. It will be aggressively optimized away by `torch.compile` (Inductor) and never gets
        # materialized. When the Triton kernel tries to write to it, it will cause an illegal memory
        # access error and crash the program!
        L = torch.zeros((B, S)).contiguous().to(Q.device)
        print(f"In forward 1: {O.shape}\n{L.shape}\n{L}")
        num_tiles_q = triton.cdiv(S, q_tile_size)
        d_block_shape = max(16, triton.next_power_of_2(D))  # type:ignore
        print(f"d_block_shape: {d_block_shape}")
        load_and_store_kernel[(num_tiles_q, B)](
            Q,  # type:ignore
            O,  # type:ignore
            L,  # type:ignore
            stride_q_b=Q.stride(0),
            stride_q_s=Q.stride(1),
            stride_q_d=Q.stride(2),
            stride_o_b=O.stride(0),
            stride_o_s=O.stride(1),
            stride_o_d=O.stride(2),
            stride_l_b=L.stride(0),
            stride_l_s=L.stride(1),
            S=S,
            D=D,
            D_BLOCK_SHAPE=int(d_block_shape),  # type: ignore
            Q_TILE_SIZE=int(q_tile_size),  # type:ignore
        )
        print(f"In forward 2: {O.shape}\n{L.shape}\n{L}")
        return O, L

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: Float[torch.Tensor, "B S D"],
    ) -> tuple[Float[torch.Tensor, "B S D"], None]:
        raise NotImplementedError("")


def main():
    B, S, D = 1, 16, 16
    Q = torch.arange(B * S * D).view((B, S, D)).to(dtype=torch.float32, device="cuda")
    O, L = torch.compile(LoadAndStoreFn.apply)(Q)
    print(f"In main: {L}")


if __name__ == "__main__":
    main()
