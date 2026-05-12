"""
Base Scheduler class for PINN training strategies.

All active Schedulers receive lifecycle hooks from the Trainer:

  on_compile(trainer)            – called once, after initial data is sampled
  on_epoch_start(trainer, epoch) – called at the start of every training epoch
  on_epoch_end(trainer, epoch, loss) – called after each gradient step
  on_training_end(trainer)       – called once, after the last epoch

Schedulers may read and mutate any public trainer attribute
(``trainer._train_data``, ``trainer.problem.domain``, etc.).
"""

from abc import ABC


class Scheduler(ABC):
    """
    Abstract base class for PINN training schedulers.

    Override whichever lifecycle hooks you need; the default
    implementation for each hook is a no-op.
    """

    def on_compile(self, trainer) -> None:
        """Called once at compile time after initial data has been sampled."""
        pass

    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Called at the start of each training epoch (before the gradient step)."""
        pass

    def on_epoch_end(self, trainer, epoch: int, loss: float) -> None:
        """Called after each gradient step."""
        pass

    def on_training_end(self, trainer) -> None:
        """Called once when ``train()`` completes (before the function returns)."""
        pass


__all__ = ["Scheduler"]
