"""L-BFGS optimizer wrapper (jaxopt)."""
from __future__ import annotations

from typing import Optional

from .base import BaseOptimizer


class LBFGSOptimizer(BaseOptimizer):
    """L-BFGS optimizer via jaxopt.

    L-BFGS uses a completely different training loop (``jaxopt.LBFGS.update``),
    so :py:meth:`build` returns *None*.  The trainer detects ``None`` and
    delegates to its dedicated ``_train_lbfgs`` method.  The constructor
    parameters are stored and read back by the trainer when it builds the
    ``jaxopt.LBFGS`` solver.

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
