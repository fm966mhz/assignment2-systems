"""Test optimizer."""

import numpy as np

from torch import nn

from cs336_basics.optimizers import AdamW
from cs336_basics.optimizers import CosineLrScheduler


def test_cosine_lr_scheduler():
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    expected_lrs = [
        # Despite `last_epoch` is init to -1, the scheduler initializer already called `step()` once
        # https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/torch/optim/lr_scheduler.py#L147,
        # which incremented `last_epoch` by 1.
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    test_model = nn.Linear(in_features=10, out_features=20)
    test_optimizer = AdamW(test_model.parameters(), lr=0)
    test_lr_schuduler = CosineLrScheduler(
        optimizer=test_optimizer,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
        last_epoch=-1,
    )

    actual_lrs = []
    for _ in range(25):
        actual_lrs.append(test_optimizer.param_groups[0]["lr"])
        test_optimizer.step()
        test_lr_schuduler.step()

    np.testing.assert_allclose(
        actual=actual_lrs,
        desired=expected_lrs,
    )
