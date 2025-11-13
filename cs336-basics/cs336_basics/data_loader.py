"""Data loader."""

import numpy as np
import numpy.typing as npt
import torch

from jaxtyping import Int


def get_batch(
    dataset: Int[npt.NDArray, "dataset_size"],
    batch_size: int,
    context_length: int,
    device: str | torch.device | None,
) -> tuple[
    Int[torch.Tensor, "batch_size context_length"],
    Int[torch.Tensor, "batch_size context_length"],
]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    dataset_size = dataset.shape[0]
    rng = np.random.default_rng()
    # It can go up to `dataset_size - context_length - 1`, so that the target sequence can go up to
    # `[dataset_size - CL - 1 + 1 : dataset_size - CL + CL - 1]`.
    selected_start_ids = rng.integers(
        low=0, high=dataset_size - context_length, size=batch_size
    )
    selected_ranges = []
    for i in range(0, context_length):
        selected_ranges.append(selected_start_ids + i)
    selected_ranges = np.stack(selected_ranges, axis=-1)
    inputs = torch.tensor(dataset[selected_ranges], dtype=torch.int64)
    labels = torch.tensor(dataset[selected_ranges + 1], dtype=torch.int64)
    if str(device).startswith("cuda"):
        inputs = inputs.pin_memory().to(device)
        labels = labels.pin_memory().to(device)
    else:
        inputs = inputs.to(device)
        labels = labels.to(device)

    return inputs, labels
