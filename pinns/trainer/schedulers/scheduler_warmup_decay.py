"""
Warmup + exponential-decay learning rate scheduler.

Linearly ramps the learning rate from 0 to ``peak_value`` over the first
``warmup_steps`` steps, then applies exponential decay:

    step < warmup_steps:
        lr = start_value + (peak_value - start_value) * step / warmup_steps

    step >= warmup_steps:
        lr = peak_value * decay_rate ^ ((step - warmup_steps) / decay_steps)

This mirrors the ``optax.warmup_exponential_decay_schedule`` used in jaxpi.

Usage
-----

    trainer.compile(
        ...
        schedulers=[
            SchedulerWarmupDecay(
                peak_value=1e-3,
                warmup_steps=5000,
                decay_rate=0.9,
                decay_steps=2000,
            ),
        ],
    )
"""

import math

from .scheduler_base import Scheduler


class SchedulerWarmupDecay(Scheduler):
    """
    Linear warmup followed by exponential decay.

    Parameters
    ----------
    peak_value : float
        Peak (maximum) learning rate reached at the end of warmup.
    warmup_steps : int
        Number of steps for the linear warmup phase (default 5000).
    decay_rate : float
        Multiplicative decay factor per ``decay_steps`` (default 0.9).
    decay_steps : int
        Number of steps per decay period (default 2000).
    start_value : float
        Learning rate at step 0, before warmup (default 0.0).
    """

    def __init__(
        self,
        peak_value: float = 1e-3,
        warmup_steps: int = 5000,
        decay_rate: float = 0.9,
        decay_steps: int = 2000,
        start_value: float = 0.0,
    ):
        if peak_value <= 0:
            raise ValueError("peak_value must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if decay_rate <= 0:
            raise ValueError("decay_rate must be positive")
        if decay_steps <= 0:
            raise ValueError("decay_steps must be positive")
        if start_value < 0:
            raise ValueError("start_value must be non-negative")

        self.peak_value = float(peak_value)
        self.warmup_steps = int(warmup_steps)
        self.decay_rate = float(decay_rate)
        self.decay_steps = int(decay_steps)
        self.start_value = float(start_value)

    # ------------------------------------------------------------------
    # LR computation
    # ------------------------------------------------------------------

    def compute_lr(self, step: int) -> float:
        """Return the learning rate at the given (global) step."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.start_value + (self.peak_value - self.start_value) * step / self.warmup_steps
        decay_step = step - self.warmup_steps
        return self.peak_value * (self.decay_rate ** (decay_step / self.decay_steps))

    # ------------------------------------------------------------------
    # Lifecycle hook — override base to also work with SOAP
    # ------------------------------------------------------------------

    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Apply the lr schedule regardless of optimizer type."""
        global_epoch = trainer.get_global_epoch() + epoch
        new_lr = self.compute_lr(global_epoch)
        trainer.set_learning_rate(new_lr)

    def needs_epoch_end_at(self, epoch: int) -> bool:
        return False


__all__ = ["SchedulerWarmupDecay"]
