"""Optimizers."""

import math

from collections.abc import Callable
from typing import Any, List
from typing import Iterable
from typing import Optional
from typing import TypeAlias
from typing import Union

import torch

from torch import nn
from torch import optim

from jaxtyping import Float

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]


class SGD(optim.Optimizer):
    """Example SGD."""

    def __init__(self, params: ParamsT, lr: float = 1e-3):
        assert lr > 0, f"Invalid learning rate: {lr}."
        defaults = {"lr": lr}
        super().__init__(params=params, defaults=defaults)

    def step(  # type: ignore
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """Takes one step."""
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)  # The iteration number.
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(optim.Optimizer):
    """AdamW."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        assert lr >= 0, f"Invalid learning rate: {lr}."
        assert weight_decay >= 0, f"Invalid weight decay: {weight_decay}"
        assert eps > 0, f"Invalid eps: {eps}"
        assert 1 >= betas[0] > 0 and 1 >= betas[1] > 0, f"Invalid betas: {betas}"
        super().__init__(
            params=params,
            defaults={
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": betas,
                "eps": eps,
            },
        )

    def step(  # type: ignore
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            self._step_one_group(group)
        return loss

    def _step_one_group(self, param_group: dict[str, Any]):
        lr = param_group["lr"]
        weigtht_decay = param_group["weight_decay"]
        beta_1, beta_2 = param_group["betas"]
        eps = param_group["eps"]
        for p in param_group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            t = state.get("t", 1)
            m = state.get("m", 0)
            v = state.get("v", 0)
            grad = p.grad.data
            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * (grad**2)
            alpha_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)
            p.data -= alpha_t * m / torch.sqrt(v + eps)
            p.data *= 1 - lr * weigtht_decay
            state["t"] = t + 1
            state["m"] = m
            state["v"] = v


def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        return (
            0.5
            * (
                1
                + math.cos(
                    (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
                )
            )
            * (max_learning_rate - min_learning_rate)
            + min_learning_rate
        )
    return min_learning_rate


class CosineLrScheduler(optim.lr_scheduler.LRScheduler):
    """The cosine LR scheduler."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
        last_epoch: int = -1,
    ):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        return [
            get_cosine_lr(
                it=self.last_epoch,
                max_learning_rate=self.max_learning_rate,
                min_learning_rate=self.min_learning_rate,
                warmup_iters=self.warmup_iters,
                cosine_cycle_iters=self.cosine_cycle_iters,
            )
            for _ in self.optimizer.param_groups
        ]


def clip_gradient(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
    device: torch.device | None = None,
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) are modified in-place.
    """
    l2_norm = get_total_gradient_l2_norm(parameters, device=device)
    if l2_norm <= max_l2_norm:
        return
    scaling_factor = max_l2_norm / (l2_norm + eps)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.data *= scaling_factor


def get_total_gradient_l2_norm(
    parameters: Iterable[nn.Parameter], device: torch.device | None = None
) -> float:
    """Gets the total gradient l2 norm."""
    l2_norm_squared = torch.zeros((), device=device)
    for p in parameters:
        if p.grad is None:
            continue
        l2_norm_squared += torch.sum(p.grad.detach().data ** 2)
    return torch.sqrt(l2_norm_squared).item()
