"""
ReduceLROnPlateau learning rate scheduler.
"""

from .scheduler_base import Scheduler


class SchedulerReduceLROnPlateau(Scheduler):
    """
    Reduce learning rate when training loss plateaus.

    Plateau detection uses a sliding window over the **actual loss history**
    stored by the trainer (via ``trainer.get_loss_history()``).  On each epoch
    end the scheduler computes the relative change of an exponential moving
    average (EMA) of the loss over the last ``window`` steps:

        relative_change = |EMA_new - EMA_old| / EMA_old

    If ``relative_change < epsilon`` and the cooldown period has elapsed, the
    learning rate is multiplied by ``factor`` and applied immediately via
    ``trainer.set_learning_rate()``.

    The base learning rate is captured the first time the scheduler is called
    (via ``trainer.get_learning_rate()``) so that reductions are always
    relative to the *original* value, not the already-reduced one.

    Example::

        scheduler = ReduceLROnPlateau(window=1000, epsilon=1e-3, factor=0.5)
        trainer.compile(schedulers=[scheduler], ...)

    Attributes:
        window: Number of loss history steps used to evaluate the EMA slope
                (default: 1000).
        epsilon: Relative-change threshold that triggers a reduction
                 (default: 1e-3).
        factor: LR multiplier applied on each plateau detection (default: 0.5).
        ema_alpha: EMA smoothing factor — higher = smoother (default: 0.99).
        min_lr: Hard floor on the learning rate (default: 1e-8).
        cooldown: Epochs to wait after a reduction before checking again
                  (default: same as ``window``).
    """

    def __init__(
        self,
        window: int = 1000,
        epsilon: float = 1e-3,
        factor: float = 0.5,
        ema_alpha: float = 0.99,
        min_lr: float = 1e-8,
        cooldown: int = None,
    ):
        if window <= 0:
            raise ValueError("window must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < factor < 1:
            raise ValueError("factor must be between 0 and 1")
        if not 0 < ema_alpha < 1:
            raise ValueError("ema_alpha must be between 0 and 1")

        self.window = window
        self.epsilon = epsilon
        self.factor = factor
        self.ema_alpha = ema_alpha
        self.min_lr = min_lr
        self.cooldown = cooldown if cooldown is not None else window

        # Mutable state (intentionally preserved across train() calls so the
        # scheduler keeps its history when train() is called multiple times)
        self._base_lr: float = None
        self._reduction_count: int = 0
        self._last_reduction_step: int = -(10 ** 9)

    # ------------------------------------------------------------------
    # Scheduler interface
    # ------------------------------------------------------------------

    def lr(self, base_lr: float, step: int) -> float:
        """Return the current LR based on accumulated reduction count."""
        effective_base = self._base_lr if self._base_lr is not None else base_lr
        return max(effective_base * (self.factor ** self._reduction_count), self.min_lr)

    def on_epoch_start(self, trainer, epoch: int) -> None:
        """No-op — learning rate is managed reactively in ``on_epoch_end``."""
        pass

    def on_epoch_end(self, trainer, epoch: int, loss: float) -> None:
        """Detect plateau from actual trainer loss history; reduce LR if needed."""
        if trainer.optimizer_name in ("lbfgs", "soap"):
            return

        # Capture the initial learning rate once so all reductions are relative to it
        if self._base_lr is None:
            self._base_lr = trainer.get_learning_rate()

        global_epoch = trainer.get_global_epoch() + epoch

        # Read the loss values recorded by the trainer
        losses = trainer.get_loss_history().get("loss", [])
        if len(losses) < self.window + 1:
            return  # not enough history yet

        # Compute EMA over the last (window + 1) values
        window_losses = losses[-(self.window + 1):]
        ema = window_losses[0]
        for v in window_losses[1:]:
            ema = self.ema_alpha * ema + (1 - self.ema_alpha) * v

        ema_old = window_losses[0]
        ema_new = ema

        # Respect cooldown
        if (global_epoch - self._last_reduction_step) < self.cooldown:
            return

        # Plateau check: relative change of the EMA over the window
        if ema_old > 1e-10:
            relative_change = abs(ema_new - ema_old) / ema_old
            if relative_change < self.epsilon:
                self._reduction_count += 1
                self._last_reduction_step = global_epoch
                new_lr = max(
                    self._base_lr * (self.factor ** self._reduction_count),
                    self.min_lr,
                )
                trainer.set_learning_rate(new_lr)

    def reset(self):
        """Reset all internal scheduler state."""
        self._base_lr = None
        self._reduction_count = 0
        self._last_reduction_step = -(10 ** 9)


__all__ = ["SchedulerReduceLROnPlateau"]
