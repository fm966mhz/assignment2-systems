"""Test Triton functions."""

import numpy as np
import torch

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


def test_flash_attention_without_head_without_causal():
    q = torch.randn((4, 64, 6)).to("cuda")
    k = torch.randn((4, 32, 6)).to("cuda")
    v = torch.randn((4, 32, 6)).to("cuda")
    expected_o = functions.scaled_dot_product_attention(q=q, k=k, v=v)

    actual_o = triton_functions.FlashAttention2.apply(q, k, v)

    np.testing.assert_allclose(
        actual_o.detach().cpu().numpy(),
        expected_o.detach().cpu().numpy(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_flash_attention_with_head_without_causal():
    q = torch.randn((4, 8, 64, 6)).to("cuda")
    k = torch.randn((4, 8, 32, 6)).to("cuda")
    v = torch.randn((4, 8, 32, 6)).to("cuda")
    expected_o = functions.scaled_dot_product_attention(q=q, k=k, v=v)

    actual_o = triton_functions.FlashAttention2.apply(q, k, v)

    np.testing.assert_allclose(
        actual_o.detach().cpu().numpy(),
        expected_o.detach().cpu().numpy(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_flash_attention_without_head_with_causal():
    q = torch.randn((4, 64, 6)).to("cuda")
    k = torch.randn((4, 32, 6)).to("cuda")
    v = torch.randn((4, 32, 6)).to("cuda")
    causal_mask = torch.tril(torch.ones((q.shape[-2], k.shape[-2]))).to(
        dtype=torch.bool, device="cuda"
    )
    expected_o = functions.scaled_dot_product_attention(q=q, k=k, v=v, mask=causal_mask)

    actual_o = triton_functions.FlashAttention2.apply(q, k, v, True)

    np.testing.assert_allclose(
        actual_o.detach().cpu().numpy(),
        expected_o.detach().cpu().numpy(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_flash_attention_with_head_with_causal():
    q = torch.randn((4, 8, 64, 6)).to("cuda")
    k = torch.randn((4, 8, 32, 6)).to("cuda")
    v = torch.randn((4, 8, 32, 6)).to("cuda")
    causal_mask = torch.tril(torch.ones((q.shape[-2], k.shape[-2]))).to(
        dtype=torch.bool, device="cuda"
    )
    expected_o = functions.scaled_dot_product_attention(q=q, k=k, v=v, mask=causal_mask)

    actual_o = triton_functions.FlashAttention2.apply(q, k, v, True)

    np.testing.assert_allclose(
        actual_o.detach().cpu().numpy(),
        expected_o.detach().cpu().numpy(),
        atol=1e-2,
        rtol=1e-2,
    )
