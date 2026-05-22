"""Base class for linear solvers used by ModelSpectralSolver."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable


class LinearSolverBase(ABC):
    """Abstract base for linear solvers.

    Subclasses implement :meth:`solve` which solves ``matvec(x) = b``.

    The ``matvec`` callable has signature::

        matvec(v) -> w   # applies the linear operator to v
    """

    @abstractmethod
    def solve(self, matvec: Callable, b):
        """Solve ``matvec(x) = b`` for ``x``.

        Args:
            matvec:  fn(v) → w  (matrix-vector product, matrix-free)
            b:       right-hand side (flat JAX array)

        Returns:
            Solution ``x``.
        """
