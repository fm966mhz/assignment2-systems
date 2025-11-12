"""Checkpoint."""

import os
import pathlib
import pickle
import typing

from dataclasses import dataclass

import torch

from torch import nn
from torch import optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the
            model, optimizer, and iteration to.
    """
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    cpu_model_state = {k: v.cpu() for k, v in model_state.items()}
    cpu_optimizer_state = {
        "state": {},
        "param_groups": optimizer_state["param_groups"],
    }
    for param_id, state in optimizer_state["state"].items():
        cpu_optimizer_state["state"][param_id] = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                cpu_optimizer_state["state"][param_id][k] = v.cpu()
            else:
                cpu_optimizer_state["state"][param_id][k] = v

    torch.save(
        {
            "iteration": iteration,
            "model": cpu_model_state,
            "optimizer": cpu_optimizer_state,
        },
        out,
    )
    del model_state, optimizer_state, cpu_model_state, cpu_optimizer_state


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device | None = None,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint_data = torch.load(src, map_location=device)
    assert "iteration" in checkpoint_data, "`iteration` not found in checkpoint data."
    assert "model" in checkpoint_data, "`model` not found in checkpoint data."
    assert "optimizer" in checkpoint_data, "`optimizer` not found in checkpoint data."
    model.load_state_dict(checkpoint_data["model"])
    optimizer.load_state_dict(checkpoint_data["optimizer"])
    return checkpoint_data["iteration"]


@dataclass
class CheckpointMetadata:
    """Metadata for the checkpoints in this directory."""

    # Map from iteration to checkpoint (relative) filename.
    iteration_to_filename: dict[int, str]
    # The iteration of the latest checkpoint.
    latest_checkpointed_iteration: int
    # At most this many checkpoints should be kept.
    max_num_checkpoints: int


class CheckpointManager:
    """Checkpoint manager."""

    def __init__(self, checkpoint_dir: str, max_num_checkpoints: int):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        assert (
            not self.checkpoint_dir.is_file()
        ), f"The checkpoint dir {checkpoint_dir} cannot be a file."
        if not self.checkpoint_dir.exists():
            os.makedirs(self.checkpoint_dir)

        self.metadata_file = self.checkpoint_dir / "METADATA"
        if self.metadata_file.exists():
            with open(self.metadata_file, "rb") as f:
                self.checkpoint_metadata: CheckpointMetadata = pickle.load(f)
                self.checkpoint_metadata.max_num_checkpoints = max_num_checkpoints
            self._trim_checkpoint_files()
        else:
            self.checkpoint_metadata = CheckpointMetadata(
                iteration_to_filename={},
                latest_checkpointed_iteration=0,
                max_num_checkpoints=max_num_checkpoints,
            )

    def _trim_checkpoint_files(self) -> None:
        sorted_iterations = sorted(
            self.checkpoint_metadata.iteration_to_filename.keys()
        )
        i = 0
        while (
            len(self.checkpoint_metadata.iteration_to_filename)
            > self.checkpoint_metadata.max_num_checkpoints
        ):
            iteration = sorted_iterations[i]
            os.remove(
                self.checkpoint_dir
                / self.checkpoint_metadata.iteration_to_filename[iteration]
            )
            del self.checkpoint_metadata.iteration_to_filename[iteration]
            i += 1
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.checkpoint_metadata, f)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        iteration: int,
    ) -> None:
        """Saves a checkpoint."""
        assert iteration > self.checkpoint_metadata.latest_checkpointed_iteration
        output_filename = f"{iteration}.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            out=self.checkpoint_dir / output_filename,
        )
        self.checkpoint_metadata.iteration_to_filename[iteration] = output_filename
        self.checkpoint_metadata.latest_checkpointed_iteration = iteration
        self._trim_checkpoint_files()

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        iteration: int | None = None,
        device: torch.device | None = None,
    ) -> int:
        """Loads a checkpoint."""
        if iteration is None:
            iteration = self.checkpoint_metadata.latest_checkpointed_iteration
        if iteration == 0:
            return 0
        src_filename = self.checkpoint_metadata.iteration_to_filename[iteration]
        iteration_from_ckpt = load_checkpoint(
            src=self.checkpoint_dir / src_filename,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        assert iteration == iteration_from_ckpt, (
            f"Checkpoint data corrupted. Expected iteration from metadata: {iteration}, but "
            "checkpoint file indicates {iteration_from_ckpt}."
        )
        return iteration
