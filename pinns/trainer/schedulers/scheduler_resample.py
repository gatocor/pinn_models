"""
SchedulerResample – randomly resample collocation points every N epochs.

Optionally maintains a pre-sampled *pool* of points (pool_size × the
training count) so that resampling is a cheap index-selection operation
rather than a full domain-sampling call.
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
    pool_size : int
        Pre-sample ``pool_size * n_train`` points at compile time.
        Resampling then draws ``n_train`` indices at random from this pool
        (fast, no domain calls).  Set to 1 to disable pooling and do a
        full domain resample at each interval.
    pool_refresh_every : int
        Rebuild the entire pool with fresh domain samples every this many
        epochs (0 = never).  Useful when ``pool_size * n_train < epochs``.
    """

    def __init__(
        self,
        every_n: int = 100,
        pool_size: int = 10,
        pool_refresh_every: int = 0,
    ):
        if every_n <= 0:
            raise ValueError("every_n must be a positive integer")
        self.every_n = every_n
        self.pool_size = max(1, int(pool_size))
        self.pool_refresh_every = int(pool_refresh_every)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_compile(self, trainer) -> None:
        """Pre-sample the pool (if pool_size > 1)."""
        if self.pool_size > 1:
            self._build_pool(trainer)

    def on_epoch_start(self, trainer, epoch: int) -> None:
        if epoch <= 0:
            return

        # Optionally refresh the pool with entirely new points
        if (self.pool_refresh_every > 0
                and epoch % self.pool_refresh_every == 0
                and self.pool_size > 1):
            self._build_pool(trainer)

        # Resample at the configured interval
        if epoch % self.every_n == 0:
            if self.pool_size > 1 and getattr(trainer, '_train_pool', None):
                self._select_from_pool(trainer)
            else:
                trainer._sample_train_data()
                _ts = trainer.test_samples
                _has_test = any(
                    v > 0
                    for v in (_ts.values() if isinstance(_ts, dict) else _ts)
                )
                if _has_test:
                    trainer._sample_test_data()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pool(self, trainer) -> None:
        """Sample a large pool for cheap index-based resampling."""
        trainer._train_pool = {}
        for name, n in trainer.train_samples.items():
            if n > 0:
                np_data = trainer._sample_points_np(name, n * self.pool_size)
                trainer._train_pool[name] = trainer._to_tensor(np_data)

    def _select_from_pool(self, trainer) -> None:
        """Draw n_train random indices from the pool (no domain calls)."""
        pool = getattr(trainer, '_train_pool', None)
        if not pool:
            trainer._sample_train_data()
            return

        trainer._train_data = {}
        trainer._train_targets = {}
        rng = trainer.rng

        for name, n in trainer.train_samples.items():
            if n > 0 and name in pool:
                pool_tensor = pool[name]
                pool_n = len(pool_tensor)
                indices = rng.choice(pool_n, size=n, replace=False)
                trainer._train_data[name] = trainer._index_tensor(pool_tensor, indices)


__all__ = ["SchedulerResample"]
