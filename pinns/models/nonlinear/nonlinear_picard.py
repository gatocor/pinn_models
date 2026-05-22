"""NonlinearSolverPicard — fixed-point / Picard iteration."""
from __future__ import annotations
from .nonlinear_base import NonlinearSolverBase


class NonlinearSolverPicard(NonlinearSolverBase):
    """Picard / fixed-point iteration for mildly nonlinear problems.

    At each iteration solves the linearised system::

        L u^{k+1} = f(u^k)

    where ``L`` is the linear part and ``f(u^k)`` is the source evaluated
    at the previous iterate.  Converges linearly; simpler than Newton.

    Args:
        tol:       Convergence tolerance on ``||u^{k+1} - u^k||``.
        max_iter:  Maximum Picard iterations.
    """

    def __init__(self, tol: float = 1e-8, max_iter: int = 50):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, residual, u0_flat, linear_solve):
        import jax
        import jax.numpy as jnp

        u = u0_flat
        for _ in range(self.max_iter):
            r = residual(u)
            # Linear part of residual: J_L v = d(Lu)/dv  (ignore nonlinear part)
            def matvec(v):
                _, jv = jax.jvp(residual, (u,), (v,))
                return jv
            delta = linear_solve(matvec, -r)
            u_new = u + delta
            if float(jnp.linalg.norm(u_new - u)) < self.tol:
                return u_new
            u = u_new
        return u
