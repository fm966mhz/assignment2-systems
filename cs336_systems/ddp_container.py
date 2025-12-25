"""The DDP container implementation."""

import threading

import torch
import torch.distributed as dist


class ParamBucket:
    """A bucket for grouping parameter tensors."""

    def __init__(self, max_bucket_size_bytes: int):
        """Initializes the parameter bucket.

        Args:
            max_bucket_size_bytes: The maximum size of the bucket in bytes.
        """
        self.max_bucket_size_bytes = max_bucket_size_bytes
        self.params = []
        self.bucket_size_bytes = 0
        self.num_grad_ready_params = 0
        self._flattened_grads = None
        self.lock = threading.Lock()

    def add_param(self, param: torch.nn.Parameter) -> bool:
        """Attempts to add a parameter to the bucket.

        Args:
            param: The parameter tensor to add.

        Returns:
            True if the bucket is full after the param is added, False otherwise.
        """
        param_size = param.numel() * param.element_size()
        self.params.append(param)
        self.bucket_size_bytes += param_size
        return self.bucket_size_bytes >= self.max_bucket_size_bytes

    def reset_bucket_state(self) -> None:
        """Resets the count of gradient-ready parameters."""
        self.num_grad_ready_params = 0
        self._flattened_grads = None

    def increment_num_grad_ready_params(self) -> None:
        """Increments the count of gradient-ready parameters."""
        self.num_grad_ready_params += 1

    def all_params_ready(self) -> bool:
        """Checks if all parameters in the bucket have their gradients ready.

        Returns:
            True if all parameters are ready, False otherwise.
        """
        return self.num_grad_ready_params == len(self.params)

    @property
    def flattened_grads(self) -> torch.Tensor:
        """Returns the flattened gradients of the parameters in the bucket."""
        # print(
        #     f"Rank {dist.get_rank()} Flattening grads for bucket with {len(self.params)} params."
        # )
        if self._flattened_grads is None:
            self._flattened_grads = (
                torch._utils._flatten_dense_tensors(  # pylint: disable=protected-access
                    [p.grad for p in self.params]
                )
            )
        return self._flattened_grads

    def unflatten_grads(self) -> None:
        """Unflattens the flattened gradients back into the individual parameter gradients."""
        # print(
        #     f"Rank {dist.get_rank()} Unflattening grads for bucket with {len(self.params)} params. Flattened grads shape: {self._flattened_grads.shape}"
        # )
        new_grads = (
            torch._utils._unflatten_dense_tensors(  # pylint: disable=protected-access
                self._flattened_grads, [p.grad for p in self.params]
            )
        )
        for param, new_grad in zip(self.params, new_grads):
            param.grad = new_grad


class DDPContainer(torch.nn.Module):
    """A container for DistributedDataParallel models."""

    def __init__(self, model: torch.nn.Module, bucket_size_mb: float | None = None):
        """Initializes the DDP container.

        Args:
            model: The model to be wrapped with DDP.
            bucket_size_mb: The bucket size in megabytes for gradient bucketing (optional).
        """
        super().__init__()
        self.module = model
        self._all_reduce_handles = []
        self._param_bucket_by_param = None
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        if bucket_size_mb is None:
            for param in self.module.parameters():
                if not param.requires_grad:
                    continue
                param.register_post_accumulate_grad_hook(self._all_reduce_gradient)
        else:
            max_bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
            self._param_bucket_by_param = {}
            current_bucket = ParamBucket(max_bucket_size_bytes)
            for param in self.module.parameters():
                if not param.requires_grad:
                    continue
                is_bucket_full = current_bucket.add_param(param)
                self._param_bucket_by_param[param] = current_bucket
                param.register_post_accumulate_grad_hook(
                    self._all_reduce_bucketed_params
                )
                if is_bucket_full:
                    current_bucket = ParamBucket(max_bucket_size_bytes)

    def forward(self, *args, **kwargs):
        """Performs a forward pass using the DDP model."""
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """Ensures that all gradients are synchronized across processes."""
        for handle in self._all_reduce_handles:
            handle.wait()
        self._all_reduce_handles.clear()
        if self._param_bucket_by_param is not None:
            for bucket in set(self._param_bucket_by_param.values()):
                bucket.unflatten_grads()
                bucket.reset_bucket_state()

    def _all_reduce_gradient(self, param) -> None:
        """All-reduces the gradient of a parameter.

        Args:
            param: The parameter whose gradient is to be all-reduced.
        """
        if param.grad is not None:
            self._all_reduce_handles.append(
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            )

    def _all_reduce_bucketed_params(self, param) -> None:
        if param.grad is None:
            return
        bucket: ParamBucket = self._param_bucket_by_param[param]
        with bucket.lock:
            bucket.increment_num_grad_ready_params()
            # print(
            #     f"Rank {dist.get_rank()} Bucket has {bucket.num_grad_ready_params}/{len(bucket.params)} ready params."
            # )
            if bucket.all_params_ready():
                self._all_reduce_handles.append(
                    dist.all_reduce(
                        bucket.flattened_grads, op=dist.ReduceOp.AVG, async_op=True
                    )
                )
