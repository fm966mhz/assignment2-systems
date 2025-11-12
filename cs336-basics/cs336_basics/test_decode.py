"""Test decode."""

from collections import Counter

import numpy as np
import torch

from cs336_basics.decode import nucleus_sample


def test_nucleus_sample():
    """Test nucleus sampling."""
    test_probs = torch.tensor([0.1, 0.3, 0.2, 0.3, 0.1])
    sampled_idx = []
    total_samples = 1000
    for _ in range(total_samples):
        sampled_idx.append(nucleus_sample(test_probs, 0.6).cpu().item())
    counts = Counter(sampled_idx)
    ratios = {idx: count / total_samples for idx, count in counts.items()}
    assert ratios.keys() == {3, 1}
    ratio_vals = np.array(list(ratios.values()))
    np.testing.assert_allclose(ratio_vals, [0.5, 0.5], atol=1e-2)
