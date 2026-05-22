"""L-BFGS optimizer wrapper (jaxopt)."""
from __future__ import annotations

from typing import Optional

from .base import BaseOptimizer


class LBFGSOptimizer(BaseOptimizer):
    """L-BFGS optimizer via jaxopt.

    :py:meth:`build` returns *None* (no optax ``GradientTransformation`` needed).
    The custom :py:meth:`train_loop` override handles the full training loop via
    ``jaxopt.LBFGS.update``, so ``Trainer.train()`` delegates entirely to this class.

    Parameters
    ----------
    max_iter:
        Maximum number of L-BFGS iterations per training step (default 5).
    history_size:
        Number of past iterates stored for the Hessian approximation (default 50).
    tol:
        Convergence tolerance on the gradient norm (default 1e-9).
    line_search:
        Line-search method — ``"strong_wolfe"`` or ``None`` (default
        ``"strong_wolfe"``).
    """

    name = "lbfgs"

    def __init__(
        self,
        learning_rate: float = 1e-3,
        max_iter: int = 5,
        history_size: int = 50,
        tol: float = 1e-9,
        line_search: str = "strong_wolfe",
        **kwargs,
    ) -> None:
        super().__init__(learning_rate, **kwargs)
        self.max_iter = max_iter
        self.history_size = history_size
        self.tol = tol
        self.line_search = line_search

    def build(
        self,
        *,
        grad_clip: Optional[float] = None,   # ignored
        lr_scheduler: bool = False,           # ignored
    ):
        """Return *None* — signal that the lbfgs training path should be used."""
        return None

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------

    def train_loop(self, trainer, epochs, print_each, params_dict, weights) -> bool:
        """Full L-BFGS training loop, delegated from ``Trainer.train()``."""
        import time
        import jax
        import jaxopt
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None

        from pinns.trainer.schedulers import is_notebook

        # Build loss function ------------------------------------------
        _train_targets = dict(getattr(trainer, '_train_targets', {}))
        _schedulers = getattr(trainer, '_schedulers', None) or []
        compute_loss_core, _ = trainer._make_compute_loss_fn(weights, params_dict,
                                                              schedulers=_schedulers)

        def compute_loss_lbfgs(params, train_data):
            loss, _residuals = compute_loss_core(params, train_data, _train_targets)
            return loss

        # Build solver -------------------------------------------------
        solver = jaxopt.LBFGS(
            fun=compute_loss_lbfgs,
            maxiter=self.max_iter,
            history_size=self.history_size,
            tol=self.tol,
        )

        print(f"Starting L-BFGS training for {epochs} epochs "
              f"({self.max_iter} iterations per epoch, "
              f"history={self.history_size})...")

        start_time = time.time()
        start_epoch = trainer._global_epoch

        auto_save_path = trainer._setup_training_plot()

        state = solver.init_state(trainer.model.params, trainer._train_data)

        if print_each > 0:
            trainer._log_epoch(
                start_epoch, start_epoch, epochs + start_epoch, None, 0.0,
                weights, params_dict, auto_save_path,
                do_plot=False,
            )

        epoch_times = []
        for epoch in range(epochs):
            trainer._current_epoch = epoch
            t0 = time.time()
            global_epoch = start_epoch + epoch

            trainer.model.params, state = solver.update(
                trainer.model.params, state, trainer._train_data
            )
            loss = state.value
            epoch_times.append(time.time() - t0)

            if print_each > 0 and (
                (global_epoch + 1) % print_each == 0 or epoch == epochs - 1
            ):
                elapsed = time.time() - start_time
                trainer._log_epoch(
                    global_epoch, global_epoch + 1, epochs + start_epoch,
                    loss, elapsed,
                    weights, params_dict, auto_save_path,
                )

        trainer.history['epoch_times'].extend(epoch_times)
        trainer._global_epoch += epochs
        print(f"L-BFGS training complete in {time.time() - start_time:.1f}s")

        if (is_notebook()
                and trainer._plotter is not None
                and trainer._plotter._fig is not None
                and plt is not None):
            plt.close(trainer._plotter._fig)

        return True
