"""Test Triton functions."""

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pytest
import torch

from jaxtyping import Float
from torch.autograd import gradcheck

from cs336_basics import functions
from cs336_systems import triton_functions


def test_weighted_sum_func():
    x = torch.randn(10, 10).to("cuda")
    weight = torch.randn(10).to("cuda")
    y = triton_functions.WeightedSumFunc.apply(x, weight)
    assert y is not None
    assert y.shape == (10,)
    assert y.device == x.device
    assert y.dtype == x.dtype
    np.testing.assert_allclose(
        y.cpu().numpy(), x.cpu().numpy() @ weight.cpu().numpy(), atol=1e-6
    )


def test_weighted_sum_func_backward():
    def f(x, weight):
        return triton_functions.WeightedSumFunc.apply(x, weight)

    # Float64 is required for using `gradcheck` due to the numerical errors of Float32.
    x = torch.randn(10, 10, dtype=torch.float64, requires_grad=True).to("cuda")
    weight = torch.randn(10, dtype=torch.float64, requires_grad=True).to("cuda")
    test_passed = gradcheck(f, (x, weight), eps=1e-6, atol=1e-4)
    print(f"Test passed: {test_passed}")


def get_attention_inputs(with_head: bool, device: torch.device) -> tuple[
    Float[torch.Tensor, "... S D"],
    Float[torch.Tensor, "... T D"],
    Float[torch.Tensor, "... T D"],
    Float[torch.Tensor, "... S D"],
]:
    torch.manual_seed(42)
    B, S, T, D = 4, 64, 32, 6
    H = 8
    if with_head:
        return (
            torch.randn((B, H, S, D), device=device, requires_grad=True),
            torch.randn((B, H, T, D), device=device, requires_grad=True),
            torch.randn((B, H, T, D), device=device, requires_grad=True),
            torch.randn((B, H, S, D), device=device, requires_grad=False),
        )
    return (
        torch.randn((B, S, D), device=device, requires_grad=True),
        torch.randn((B, T, D), device=device, requires_grad=True),
        torch.randn((B, T, D), device=device, requires_grad=True),
        torch.randn((B, S, D), device=device, requires_grad=False),
    )


@pytest.mark.parametrize(
    "with_head, is_causal", [(False, False), (True, False), (False, True), (True, True)]
)
def test_flash_attention_foward(with_head, is_causal):
    q, k, v, _ = get_attention_inputs(with_head, device=torch.device("cuda"))
    causal_mask = (
        torch.tril(torch.ones((q.shape[-2], k.shape[-2]))).to(
            dtype=torch.bool, device="cuda"
        )
        if is_causal
        else None
    )
    expected_o = functions.scaled_dot_product_attention(q=q, k=k, v=v, mask=causal_mask)
    actual_o = triton_functions.FlashAttention2.apply(q, k, v, is_causal)

    np.testing.assert_allclose(
        actual_o.detach().cpu().numpy(),
        expected_o.detach().cpu().numpy(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "with_head, is_causal",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_flash_attention_backward(with_head, is_causal):
    q, k, v, do = get_attention_inputs(with_head, device=torch.device("cuda"))
    causal_mask = (
        torch.tril(torch.ones((q.shape[-2], k.shape[-2]))).to(
            dtype=torch.bool, device="cuda"
        )
        if is_causal
        else None
    )
    functions.scaled_dot_product_attention(q=q, k=k, v=v, mask=causal_mask).backward(do)
    expected_dq, expected_dk, expected_dv = q.grad, k.grad, v.grad
    q, k, v, do = get_attention_inputs(with_head, device=torch.device("cuda"))

    triton_functions.FlashAttention2.apply(q, k, v, is_causal).backward(do)

    torch.testing.assert_close(q.grad, expected_dq, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k.grad, expected_dk, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v.grad, expected_dv, rtol=1e-2, atol=1e-2)
