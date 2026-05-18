"""
SchedulerResample – randomly resample collocation points every N epochs.
"""

import numpy as np
from .scheduler_base import Scheduler


class SchedulerResample(Scheduler):
    """
    Resample collocation (PDE interior + BC) points every ``every_n`` epochs.

    Parameters
    ----------
    every_n : int
        Resample interval in epochs.
    """

    def __init__(self, every_n: int = 100):
        if every_n <= 0:
            raise ValueError("every_n must be a positive integer")
        self.every_n = every_n

    def on_compile(self, trainer) -> None:
        pass

    def on_epoch_start(self, trainer, epoch: int) -> None:
        if epoch > 0 and epoch % self.every_n == 0:
            trainer._sample_train_data()
            _ts = trainer.test_samples
            if any(v > 0 for v in (_ts.values() if isinstance(_ts, dict) else _ts)):
                trainer._sample_test_data()


__all__ = ["SchedulerResample"]
