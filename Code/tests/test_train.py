from __future__ import annotations

from src.irc_vit.train import learning_rate


def test_learning_rate_can_use_longer_schedule_horizon() -> None:
    short = {
        "lr": 0.0006,
        "min_lr": 0.0,
        "warmup_steps": 500,
        "steps": 2000,
    }
    long_horizon = dict(short, lr_total_steps=20000)

    assert learning_rate(1999, short) < 1e-7
    assert learning_rate(1999, long_horizon) > 0.00058
