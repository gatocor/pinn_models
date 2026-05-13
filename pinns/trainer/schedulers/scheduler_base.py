"""
Base Scheduler class for PINN training strategies.

All active Schedulers receive lifecycle hooks from the Trainer:

  on_compile(trainer)                 – called once after initial data is sampled
  on_epoch_start(trainer, epoch)      – called before each gradient step
  on_epoch_end(trainer, epoch, loss)  – called after each gradient step
  on_training_end(trainer)            – called once after the last epoch

Schedulers may use the full trainer helper API
(``trainer.set_learning_rate()``, ``trainer.get_loss_history()``, etc.).

**Learning-rate schedulers** optionally implement ``lr(base_lr, step) -> float``.
When present, the base ``on_epoch_start`` will call it and apply the result via
``trainer.set_learning_rate()``.  Override ``on_epoch_start`` to suppress this
behaviour (e.g. when the LR update happens reactively in ``on_epoch_end``).
"""

from abc import ABC


def is_notebook() -> bool:
    """Return True when running inside a Jupyter notebook or qtconsole."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except (NameError, AttributeError):
        return False


class Scheduler(ABC):
    """
    Abstract base class for all PINN training schedulers.

    Override whichever lifecycle hooks you need; every hook has a no-op
    default.

    **Optional LR interface**: implement ``lr(base_lr, step) -> float`` to
    create a learning-rate scheduler.  The base ``on_epoch_start`` detects
    this method and applies the returned value automatically.
    """

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_compile(self, trainer) -> None:
        """Called once at compile time after initial data has been sampled."""
        pass

    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Called before each gradient step.

        If ``self.lr`` is defined, computes the new learning rate and applies
        it via ``trainer.set_learning_rate()``.  Override to suppress this.
        """
        if not hasattr(self, 'lr'):
            return
        if trainer.optimizer_name in ("lbfgs", "soap"):
            return
        global_epoch = trainer.get_global_epoch() + epoch
        new_lr = self.lr(trainer.get_learning_rate(), global_epoch)
        trainer.set_learning_rate(new_lr)

    def on_epoch_end(self, trainer, epoch: int, loss: float) -> None:
        """Called after each gradient step."""
        pass

    def on_training_end(self, trainer) -> None:
        """Called once when ``train()`` completes."""
        pass

    # ------------------------------------------------------------------
    # JIT-state protocol (optional – override for in-step contributions)
    # ------------------------------------------------------------------

    def get_jit_state(self) -> dict:
        """Return JAX arrays that must be passed *inside* the JIT training step.

        The trainer collects ``get_jit_state()`` from every scheduler before
        each gradient step and passes the aggregated dict to ``extra_loss``.
        Return an empty dict (default) to opt out.

        The returned pytree structure **must be stable** across epochs (same
        keys and array shapes/dtypes); value changes are fine and will be
        picked up without recompilation.
        """
        return {}

    def extra_loss(self, residuals: dict, jit_state: dict):
        """Additional loss term computed *inside* the JIT step.

        Parameters
        ----------
        residuals : dict[str, jax.Array]
            Per-term residual vectors ``{name: R_k}`` produced by the problem's
            ``make_residual_fn``.
        jit_state : dict
            The dict returned by ``self.get_jit_state()`` for this scheduler.

        Returns
        -------
        jax.Array – scalar extra loss to add to the total.  Default: 0.
        """
        import jax.numpy as jnp
        return jnp.array(0.0)


__all__ = ["Scheduler", "is_notebook"]
