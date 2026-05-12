"""
SchedulerCurriculum – progressive time/domain expansion.

Implements *causal curriculum learning*: the upper bound of a chosen input
dimension (typically the time axis) starts at ``t_ends[0]`` and advances to
``t_ends[1]``, ``t_ends[2]``, … after every ``epochs_per_stage`` epochs.

After training ends the domain bound is restored to its original value.
"""

from .scheduler_base import Scheduler


class SchedulerCurriculum(Scheduler):
    """
    Progressive domain-expansion curriculum.

    Parameters
    ----------
    t_ends : list[float]
        Sequence of upper-bound values for the chosen input dimension,
        e.g. ``[2.0, 5.0, 10.0]``.  The first stage uses ``t_ends[0]``;
        the last stage is held until training finishes.
    epochs_per_stage : int
        Number of epochs to train with each upper bound before advancing.
    dim : int
        Index of the input dimension to schedule (default 0, typically time).
    """

    def __init__(
        self,
        t_ends: list,
        epochs_per_stage: int = 1000,
        dim: int = 0,
    ):
        if not t_ends:
            raise ValueError("t_ends must be a non-empty list")
        self.t_ends = list(t_ends)
        self.epochs_per_stage = int(epochs_per_stage)
        self.dim = int(dim)
        self._original_xmax: float = None
        self._current_stage: int = -1

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_compile(self, trainer) -> None:
        """Snapshot the original domain bound and reset stage counter."""
        self._original_xmax = float(trainer.problem.domain.xmax[self.dim])
        self._current_stage = -1
        # Apply stage 0 immediately so the first resample uses the curriculum bound
        self._advance_to_stage(trainer, 0)

    def on_epoch_start(self, trainer, epoch: int) -> None:
        stage = min(epoch // self.epochs_per_stage, len(self.t_ends) - 1)
        if stage != self._current_stage:
            self._advance_to_stage(trainer, stage)

    def on_training_end(self, trainer) -> None:
        """Restore the original domain upper bound."""
        if self._original_xmax is not None:
            trainer.problem.domain.xmax[self.dim] = self._original_xmax

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _advance_to_stage(self, trainer, stage: int) -> None:
        self._current_stage = stage
        new_end = float(self.t_ends[stage])
        trainer.problem.domain.xmax[self.dim] = new_end
        trainer._sample_train_data()
        if trainer._test_data:
            trainer._sample_test_data()

        # If a SchedulerLagrange is active, resize its λ vectors
        for s in getattr(trainer, '_schedulers', []):
            if hasattr(s, 'reinitialize_if_needed'):
                s.reinitialize_if_needed(trainer)

        if stage > 0:
            print(f"  [SchedulerCurriculum] Stage {stage}: "
                  f"dim {self.dim} upper bound = {new_end}")


__all__ = ["SchedulerCurriculum"]
