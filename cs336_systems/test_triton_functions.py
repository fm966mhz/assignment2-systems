"""Test Triton functions."""

import numpy as np
import torch

from torch.autograd import gradcheck

from cs336_systems.triton_functions import WeightedSumFunc


def test_weighted_sum_func():
    x = torch.randn(10, 10).to("cuda")
    weight = torch.randn(10).to("cuda")
    y = WeightedSumFunc.apply(x, weight)
    assert y is not None
    assert y.shape == (10,)
    assert y.device == x.device
    assert y.dtype == x.dtype
    np.testing.assert_allclose(
        y.cpu().numpy(), x.cpu().numpy() @ weight.cpu().numpy(), atol=1e-6
    )


def test_weighted_sum_func_backward():
    def f(x, weight):
        return WeightedSumFunc.apply(x, weight)

    # Float64 is required for using `gradcheck` due to the numerical errors of Float32.
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True).to("cuda")
    weight = torch.randn(2, dtype=torch.float64, requires_grad=True).to("cuda")
    test_passed = gradcheck(f, (x, weight), eps=1e-6, atol=1e-4)
    print(f"Test passed: {test_passed}")
