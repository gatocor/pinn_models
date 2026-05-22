"""LinearSolverGMRES — matrix-free GMRES via jax.scipy."""
from __future__ import annotations
from .linear_base import LinearSolverBase


class LinearSolverGMRES(LinearSolverBase):
    """GMRES linear solver (matrix-free).

    Uses ``jax.scipy.sparse.linalg.gmres``.  Only requires ``matvec``
    applications — no explicit matrix is formed.  Fully differentiable
    via implicit differentiation of the linear solve.

    Args:
        tol:       Convergence tolerance.
        restart:   GMRES restart cycle length (Arnoldi basis size).
        max_iter:  Maximum number of outer restarts.

    Example::

        model = pinns.ModelSpectralSolver(
            domain, ["u"],
            linear=LinearSolverGMRES(tol=1e-10, restart=30),
            shape=32, bc="chebyshev",
        )
    """

    def __init__(self, tol: float = 1e-10, restart: int = 30, max_iter: int = 100):
        self.tol = tol
        self.restart = restart
        self.max_iter = max_iter

    def solve(self, matvec, b):
        import jax.scipy.sparse.linalg as jssl
        x, info = jssl.gmres(
            matvec, b,
            tol=self.tol,
            restart=self.restart,
            maxiter=self.max_iter,
        )
        return x
