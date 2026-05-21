"""
pinns/integrators/integrator_dopri5.py — Dormand-Prince 4(5) adaptive integrator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .integrator_base import Integrator
from .stepsize_controller import PIDController

__all__ = ["IntegratorDopri5"]


# ── Butcher tableau — Dormand-Prince (1980) / DOPRI5 ─────────────────────────
# 5th-order solution weights (b5):
_DP_B1 =    35.0 /   384.0
_DP_B3 =   500.0 /  1113.0
_DP_B4 =   125.0 /   192.0
_DP_B5 = -2187.0 /  6784.0
_DP_B6 =    11.0 /    84.0

# Inter-stage coefficients
_DP_A21 =  1.0 / 5.0

_DP_A31 =  3.0 / 40.0
_DP_A32 =  9.0 / 40.0

_DP_A41 =  44.0 /  45.0
_DP_A42 = -56.0 /  15.0
_DP_A43 =  32.0 /   9.0

_DP_A51 =  19372.0 /  6561.0
_DP_A52 = -25360.0 /  2187.0
_DP_A53 =  64448.0 /  6561.0
_DP_A54 =   -212.0 /   729.0

_DP_A61 =   9017.0 /  3168.0
_DP_A62 =   -355.0 /    33.0
_DP_A63 =  46732.0 /  5247.0
_DP_A64 =     49.0 /   176.0
_DP_A65 =  -5103.0 / 18656.0

# Error = b5 − b4*  (b4* = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
_DP_E1 =     71.0 /   57600.0
_DP_E3 =    -71.0 /   16695.0
_DP_E4 =     71.0 /    1920.0
_DP_E5 = -17253.0 /  339200.0
_DP_E6 =     22.0 /     525.0
_DP_E7 =     -1.0 /      40.0   # FSAL: k7 = f(y_{n+1})


class IntegratorDopri5(Integrator):
    """Adaptive Dormand-Prince Runge-Kutta 4(5) integrator (native JAX).

    Uses the classic 7-stage DOPRI5 tableau with an embedded 4th-order
    solution to estimate the local truncation error.  Step sizes are
    controlled by a :class:`~pinns.integrators.PIDController`.

    The integration loop is implemented with ``jax.lax.scan`` (fixed
    ``max_steps`` budget), making it compatible with ``jax.grad`` and JIT.

    Args:
        rtol:       Relative tolerance for the error controller.
        atol:       Absolute tolerance for the error controller.
        dt0:        Initial step size.
        max_steps:  Maximum number of attempted steps (accepted + rejected).
        checkpoint: Wrap ``_one_step`` with ``jax.remat`` to save memory.
    """

    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        dt0: float = 1e-2,
        max_steps: int = 4096,
        checkpoint: bool = False,
    ):
        self.rtol       = float(rtol)
        self.atol       = float(atol)
        self.dt0        = float(dt0)
        self.max_steps  = int(max_steps)
        self.checkpoint = checkpoint

    def _one_step(self, problem, state_hat, t, dt, L, params):
        """Single Dormand-Prince step.

        Returns:
            ``(sh_5th, err_abs)`` where ``sh_5th`` is the 5th-order solution
            and ``err_abs`` is the elementwise |b5 - b4*| error estimate.
        """
        state_names = problem.state_names

        def _rhs(sh):
            Nhat = problem._nonlinear_op(sh, params)
            return {name: L[name] * sh[name] + Nhat[name] for name in state_names}

        k1 = _rhs(state_hat)
        k2 = _rhs({name: state_hat[name] + dt * _DP_A21 * k1[name] for name in state_names})
        k3 = _rhs({
            name: state_hat[name] + dt * (_DP_A31 * k1[name] + _DP_A32 * k2[name])
            for name in state_names
        })
        k4 = _rhs({
            name: state_hat[name] + dt * (
                _DP_A41 * k1[name] + _DP_A42 * k2[name] + _DP_A43 * k3[name]
            )
            for name in state_names
        })
        k5 = _rhs({
            name: state_hat[name] + dt * (
                _DP_A51 * k1[name] + _DP_A52 * k2[name]
                + _DP_A53 * k3[name] + _DP_A54 * k4[name]
            )
            for name in state_names
        })
        k6 = _rhs({
            name: state_hat[name] + dt * (
                _DP_A61 * k1[name] + _DP_A62 * k2[name] + _DP_A63 * k3[name]
                + _DP_A64 * k4[name] + _DP_A65 * k5[name]
            )
            for name in state_names
        })

        sh_new = {
            name: state_hat[name] + dt * (
                _DP_B1 * k1[name]
                + _DP_B3 * k3[name]
                + _DP_B4 * k4[name]
                + _DP_B5 * k5[name]
                + _DP_B6 * k6[name]
            )
            for name in state_names
        }

        k7 = _rhs(sh_new)  # FSAL

        import jax.numpy as jnp
        err_abs = {
            name: jnp.abs(dt * (
                _DP_E1 * k1[name]
                + _DP_E3 * k3[name]
                + _DP_E4 * k4[name]
                + _DP_E5 * k5[name]
                + _DP_E6 * k6[name]
                + _DP_E7 * k7[name]
            ))
            for name in state_names
        }

        return sh_new, err_abs

    def solve(
        self,
        problem,
        inferred_params=None,
        t_obs=None,
    ):
        """Adaptive-step solve using Dormand-Prince 4(5)."""
        controller = PIDController(
            rtol=self.rtol, atol=self.atol, order=5,
            safety=0.9, fac_min=0.2, fac_max=10.0,
        )
        return self._adaptive_solve(
            problem, inferred_params, controller,
            dt0=self.dt0, max_steps=self.max_steps, checkpoint=self.checkpoint,
            t_obs=t_obs,
        )

    def __repr__(self) -> str:
        return (
            f"IntegratorDopri5(rtol={self.rtol}, atol={self.atol}, "
            f"dt0={self.dt0}, max_steps={self.max_steps}, "
            f"checkpoint={self.checkpoint})"
        )
