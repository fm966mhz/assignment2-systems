"""The DDP container implementation."""

import torch
import torch.distributed as dist


class DDPContainer(torch.nn.Module):
    """A container for DistributedDataParallel models."""

    def __init__(self, model: torch.nn.Module):
        """Initializes the DDP container.

        Args:
            model: The model to be wrapped with DDP.
        """
        super().__init__()
        self.module = model
        self._all_reduce_handles = []
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._all_reduce_gradient)

    def forward(self, *args, **kwargs):
        """Performs a forward pass using the DDP model."""
        return self.module(*args, **kwargs)

    def finish_gradient_syncronization(self):
        """Ensures that all gradients are synchronized across processes."""
        for handle in self._all_reduce_handles:
            handle.wait()
        self._all_reduce_handles.clear()

    def _all_reduce_gradient(self, param) -> None:
        """All-reduces the gradient of a parameter.

        Args:
            param: The parameter whose gradient is to be all-reduced.
        """
        if param.grad is not None:
            self._all_reduce_handles.append(
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            )
