"""
pinns/integrators/integrator_tsit5.py — Tsitouras 4(5) adaptive integrator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import jax.numpy as jnp

from .integrator_base import Integrator, _apply_L
from .stepsize_controller import PIDController

__all__ = ["IntegratorTsit5"]


# ── Butcher tableau — Tsitouras (2011) ────────────────────────────────────────
# "Runge–Kutta pairs of order 5(4) satisfying only the first column simplifying
#  assumption", Computers & Mathematics with Applications
# c = [0, 0.161, 0.327, 0.9, 0.9800255409045097, 1, 1]

_TS_A21 =  0.161

_TS_A31 = -0.008480655492356990
_TS_A32 =  0.335480655492357

_TS_A41 =  2.8971530571054935
_TS_A42 = -6.359448489975075
_TS_A43 =  4.3622954328695815

_TS_A51 =  5.325864828439257
_TS_A52 = -11.748883564062828
_TS_A53 =  7.4955393428898365
_TS_A54 = -0.09249506636175525

_TS_A61 =  5.86145544294642
_TS_A62 = -12.92096931784711
_TS_A63 =  8.159367898576159
_TS_A64 = -0.071584973281401
_TS_A65 = -0.02826905039406838

# 5th-order weights  (b7 = 0, FSAL)
_TS_B1 =  0.09646076681806523
_TS_B2 =  0.01
_TS_B3 =  0.4798896504144996
_TS_B4 =  1.379008574103742
_TS_B5 = -3.290069515436081
_TS_B6 =  2.324710524099774

# Embedded 4th-order weights b4*
_TS_B4s1 = -0.001780011052226
_TS_B4s2 = -0.000816434459657
_TS_B4s3 = -0.007880878010262
_TS_B4s4 =  0.144711007173263
_TS_B4s5 = -0.582357165452555
_TS_B4s6 =  0.458082105929187
_TS_B4s7 =  1.0 / 66.0

# Error = b5 - b4*
_TS_E1 = _TS_B1 - _TS_B4s1
_TS_E2 = _TS_B2 - _TS_B4s2
_TS_E3 = _TS_B3 - _TS_B4s3
_TS_E4 = _TS_B4 - _TS_B4s4
_TS_E5 = _TS_B5 - _TS_B4s5
_TS_E6 = _TS_B6 - _TS_B4s6
_TS_E7 =        - _TS_B4s7   # b7 = 0


class IntegratorTsit5(Integrator):
    """Adaptive Tsitouras Runge-Kutta 4(5) integrator (native JAX).

    Uses the 7-stage Tsit5 (Tsitouras 2011) tableau with an embedded 4th-order
    solution.  For non-stiff problems Tsit5 is often slightly faster than
    Dopri5 because of better free-parameter optimisation, while using the same
    number of function evaluations per step.

    Args:
        rtol:       Relative tolerance.
        atol:       Absolute tolerance.
        dt0:        Initial step size.
        max_steps:  Maximum number of attempted steps.
        checkpoint: Wrap ``_one_step`` with ``jax.remat`` (memory vs speed).
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

    @property
    def dt(self) -> float:
        """Alias for ``dt0``, required by :meth:`_get_obs_times`."""
        return self.dt0

    def _one_step(self, problem, state_hat, t, dt, L, params):
        """Single Tsitouras step.

        Returns:
            ``(sh_5th, err_abs)`` where ``sh_5th`` is the 5th-order solution
            and ``err_abs`` carries the |b5 - b4*| error estimate.
        """
        state_names = problem.state_names

        def _rhs(sh):
            Nhat = problem._call_nonlinear_op(sh, params)
            return {name: _apply_L(L[name], sh[name]) + Nhat[name] for name in state_names}

        k1 = _rhs(state_hat)
        k2 = _rhs({
            name: state_hat[name] + dt * _TS_A21 * k1[name]
            for name in state_names
        })
        k3 = _rhs({
            name: state_hat[name] + dt * (_TS_A31 * k1[name] + _TS_A32 * k2[name])
            for name in state_names
        })
        k4 = _rhs({
            name: state_hat[name] + dt * (
                _TS_A41 * k1[name] + _TS_A42 * k2[name] + _TS_A43 * k3[name]
            )
            for name in state_names
        })
        k5 = _rhs({
            name: state_hat[name] + dt * (
                _TS_A51 * k1[name] + _TS_A52 * k2[name]
                + _TS_A53 * k3[name] + _TS_A54 * k4[name]
            )
            for name in state_names
        })
        k6 = _rhs({
            name: state_hat[name] + dt * (
                _TS_A61 * k1[name] + _TS_A62 * k2[name] + _TS_A63 * k3[name]
                + _TS_A64 * k4[name] + _TS_A65 * k5[name]
            )
            for name in state_names
        })

        sh_new = {
            name: state_hat[name] + dt * (
                _TS_B1 * k1[name]
                + _TS_B2 * k2[name]
                + _TS_B3 * k3[name]
                + _TS_B4 * k4[name]
                + _TS_B5 * k5[name]
                + _TS_B6 * k6[name]
            )
            for name in state_names
        }

        k7 = _rhs(sh_new)  # FSAL

        err_abs = {
            name: jnp.abs(dt * (
                _TS_E1 * k1[name]
                + _TS_E2 * k2[name]
                + _TS_E3 * k3[name]
                + _TS_E4 * k4[name]
                + _TS_E5 * k5[name]
                + _TS_E6 * k6[name]
                + _TS_E7 * k7[name]
            ))
            for name in state_names
        }

        return sh_new, err_abs

    def solve(
        self,
        problem,
        inferred_params=None,
    ):
        """Adaptive-step solve using Tsitouras 4(5)."""
        controller = PIDController(
            rtol=self.rtol, atol=self.atol, order=5,
            safety=0.9, fac_min=0.2, fac_max=10.0,
        )
        return self._adaptive_solve(
            problem, inferred_params, controller,
            dt0=self.dt0, max_steps=self.max_steps, checkpoint=self.checkpoint,
        )

    def __repr__(self) -> str:
        return (
            f"IntegratorTsit5(rtol={self.rtol}, atol={self.atol}, "
            f"dt0={self.dt0}, max_steps={self.max_steps}, "
            f"checkpoint={self.checkpoint})"
        )
