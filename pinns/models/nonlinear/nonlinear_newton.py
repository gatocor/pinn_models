"""NonlinearSolverNewton — Newton-Krylov nonlinear solver."""
from __future__ import annotations
from typing import Optional
from .nonlinear_base import NonlinearSolverBase


class NonlinearSolverNewton(NonlinearSolverBase):
    """Newton-Krylov solver for nonlinear systems ``R(u) = 0``.

    At each Newton iteration solves the linear system::

        J(u^k) Δu = -R(u^k)

    using Jacobian-free Newton-Krylov (JFNK): the Jacobian-vector product
    ``J v`` is computed via ``jax.jvp`` — no explicit Jacobian is formed.

    Args:
        tol:       Convergence tolerance on ``||R(u)||``.
        max_iter:  Maximum Newton iterations.

    Example::

        model = pinns.ModelSpectralSolver(
            domain, ["u"],
            nonlinear=NonlinearSolverNewton(tol=1e-8, max_iter=20),
            linear=LinearSolverGMRES(tol=1e-10),
            shape=32, bc="chebyshev",
        )
    """

    def __init__(self, tol: float = 1e-8, max_iter: int = 20):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, residual, u0_flat, linear_solve):
        import jax
        import jax.numpy as jnp

        u = u0_flat
        for _ in range(self.max_iter):
            r = residual(u)
            if float(jnp.linalg.norm(r)) < self.tol:
                break
            # Jacobian-vector product via forward-mode AD
            def matvec(v):
                _, jv = jax.jvp(residual, (u,), (v,))
                return jv
            delta = linear_solve(matvec, -r)
            u = u + delta
        return u
