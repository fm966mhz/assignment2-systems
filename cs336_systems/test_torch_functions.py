"""Test PyTorch functions."""

import numpy as np
import torch

from cs336_basics import functions
from cs336_systems import pytorch_functions


def test_pytorch_flash_attention_2_forward_no_head_dim():
    q = torch.randn((4, 64, 6))
    k = torch.randn((4, 32, 6))
    v = torch.randn((4, 32, 6))
    expected_o = functions.scaled_dot_product_attention(q=q, k=k, v=v)

    actual_o = pytorch_functions.FlashAttention2Torch.apply(q, k, v)

    np.testing.assert_allclose(
        actual_o.detach().cpu().numpy(), expected_o.detach().cpu().numpy(), atol=1e-6
    )


def test_pytorch_flash_attention_2_forward_with_head_dim():
    q = torch.randn((4, 2, 64, 6))
    k = torch.randn((4, 2, 32, 6))
    v = torch.randn((4, 2, 32, 6))
    expected_o = functions.scaled_dot_product_attention(q=q, k=k, v=v)

    actual_o = pytorch_functions.FlashAttention2Torch.apply(q, k, v)

    np.testing.assert_allclose(
        actual_o.detach().cpu().numpy(), expected_o.detach().cpu().numpy(), atol=1e-6
    )
