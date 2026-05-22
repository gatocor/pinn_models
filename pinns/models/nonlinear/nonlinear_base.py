"""Base class for nonlinear solvers used by ModelSpectralSolver."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any


class NonlinearSolverBase(ABC):
    """Abstract base for nonlinear solvers.

    Subclasses implement :meth:`solve` which finds ``u`` such that
    ``residual(u) = 0``, given a linear solver for the inner linear system.

    The ``residual`` callable has signature::

        residual(u_flat) -> r_flat

    The ``linear_solve`` callable has signature::

        linear_solve(matvec, b) -> x   # solves matvec(x) = b
    """

    @abstractmethod
    def solve(
        self,
        residual: Callable,
        u0_flat,
        linear_solve: Callable,
    ):
        """Find ``u`` such that ``residual(u) = 0``.

        Args:
            residual:      fn(u_flat) → r_flat
            u0_flat:       initial guess (flat JAX array)
            linear_solve:  fn(matvec, b) → x  (inner linear solver)

        Returns:
            Solution ``u_flat``.
        """
