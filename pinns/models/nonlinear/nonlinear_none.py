"""NonlinearSolverNone — pass-through for purely linear problems."""
from __future__ import annotations
from .nonlinear_base import NonlinearSolverBase


class NonlinearSolverNone(NonlinearSolverBase):
    """No nonlinear iteration — for linear problems only.

    Solves ``L u = f`` directly via the inner linear solver without
    any outer nonlinear loop.  Use when ``set_source_fn`` does not
    depend on the state ``U``.
    """

    def solve(self, residual, u0_flat, linear_solve):
        # residual(u) = L(u) - f  →  solve L u = f once
        # We use the linear solver directly: matvec = Jacobian of residual
        import jax
        def matvec(v):
            _, jv = jax.jvp(residual, (u0_flat,), (v,))
            return jv
        rhs = -residual(u0_flat)
        return u0_flat + linear_solve(matvec, rhs)
