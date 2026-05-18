"""
SchedulerCurriculum – progressive time-window curriculum learning.

The trainer's ``t_max`` (and optionally ``t_min``) attributes control the
upper (and lower) time bound used when sampling collocation points.  This
scheduler advances ``t_max`` through a user-supplied sequence of values,
widening the training window over time without ever mutating the domain.

Example
-------
    SchedulerCurriculum([0.25, 0.5, 0.75, 1.0], epochs_per_stage=5000)

Starts training on t ∈ [t_min, 0.25], then expands to [t_min, 0.5] after
5000 epochs, etc.  The final stage is held for the rest of training.
"""

from .scheduler_base import Scheduler


class SchedulerCurriculum(Scheduler):
    """
    Progressive time-window curriculum.

    Parameters
    ----------
    t_ends : list[float]
        Sequence of ``t_max`` values.  Training starts with ``t_ends[0]``
        and advances to each subsequent value after ``epochs_per_stage``
        epochs.  The last value is held until training finishes.
    epochs_per_stage : int
        Number of epochs per stage (default 1000).
    t_starts : list[float] | None
        Optional sequence of ``t_min`` values (same length as ``t_ends``).
        If ``None``, ``trainer.t_min`` is left unchanged throughout.
    reset_optimizer : bool
        If ``True`` (default), the optimizer state (momentum, variance
        accumulators, preconditioner) is reset at every stage transition.
        This prevents stale curvature estimates from slowing adaptation to
        the new, larger time window.
    """

    def __init__(
        self,
        t_ends: list,
        epochs_per_stage: int = 1000,
        t_starts: list = None,
        reset_optimizer: bool = True,
    ):
        if not t_ends:
            raise ValueError("t_ends must be a non-empty list")
        if t_starts is not None and len(t_starts) != len(t_ends):
            raise ValueError("t_starts and t_ends must have the same length")
        self.t_ends = list(t_ends)
        self.t_starts = list(t_starts) if t_starts is not None else None
        self.epochs_per_stage = int(epochs_per_stage)
        self.reset_optimizer = bool(reset_optimizer)
        self._current_stage: int = -1
        self._original_t_min: float = None
        self._original_t_max: float = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_compile(self, trainer) -> None:
        """Snapshot original bounds and immediately apply stage 0."""
        self._original_t_min = trainer.t_min
        self._original_t_max = trainer.t_max
        self._current_stage = -1
        self._advance_to_stage(trainer, 0)

    def on_epoch_start(self, trainer, epoch: int) -> None:
        stage = min(epoch // self.epochs_per_stage, len(self.t_ends) - 1)
        if stage != self._current_stage:
            self._advance_to_stage(trainer, stage)

    def on_training_end(self, trainer) -> None:
        """Restore original time bounds."""
        trainer.t_min = self._original_t_min
        trainer.t_max = self._original_t_max

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _advance_to_stage(self, trainer, stage: int) -> None:
        self._current_stage = stage
        trainer.t_max = float(self.t_ends[stage])
        if self.t_starts is not None:
            trainer.t_min = float(self.t_starts[stage])

        trainer._sample_train_data()
        if trainer._test_data:
            trainer._sample_test_data()

        # Resize Lagrange λ vectors for the new point count
        for s in getattr(trainer, '_schedulers', []):
            if hasattr(s, 'reinitialize_if_needed'):
                s.reinitialize_if_needed(trainer)

        # Reset optimizer momentum/variance so stale curvature estimates from
        # the old (narrower) window don't impede learning on the new window.
        # Skip stage 0 — the optimizer is freshly initialised at compile time.
        if self.reset_optimizer and stage > 0:
            trainer._init_optimizer_state()




__all__ = ["SchedulerCurriculum"]
