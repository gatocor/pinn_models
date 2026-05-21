"""
pinns/integrators/integrator_diffrax.py — Diffrax-based adaptive integrator.

Wraps any `diffrax` solver (Dopri5, Dopri8, Kvaerno5, …) inside the standard
``Integrator`` interface.  The full RHS (linear + nonlinear) is built from the
``ModelSolver`` operators and passed to ``diffrax.diffeqsolve``.

Gradient strategy
-----------------
Because adaptive solvers use a **data-dependent number of steps**, the forward
trajectory cannot be represented as a static ``jax.lax.scan`` — standard
reverse-mode AD would require storing every intermediate step, which is
impractical.  Instead, ``IntegratorDiffrax`` uses diffrax's built-in adjoint
methods:

* ``"backsolve"`` *(default for adaptive)*  — ``diffrax.BacksolveAdjoint``.
  Integrates the adjoint ODE backwards in time with a fresh adaptive solver.
  O(1) memory in the number of steps.  Standard Neural ODE adjoint.

* ``"recursive"`` *(default for fixed-step)*  — ``diffrax.RecursiveCheckpointAdjoint``.
  Differentiates through the unrolled steps but uses binary-tree checkpointing,
  keeping memory O(log N).  Equivalent to ``jax.remat`` across the scan.

* ``"direct"`` — ``diffrax.DirectAdjoint``.  Full backprop through all steps.
  Only practical for very short integrations.

Usage
-----
::

    import diffrax
    integrator = IntegratorDiffrax(
        solver=diffrax.Dopri5(),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
        adjoint="backsolve",   # gradient via adjoint ODE
        dt0=1e-3,              # initial step hint
    )
    model = pinns.ModelSolver(domain, ["u"], integrator, shape=64)
    ...
    states = model.solve(t_obs=t_ref)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp

from .integrator_base import Integrator

__all__ = ["IntegratorDiffrax"]


class IntegratorDiffrax(Integrator):
    """Diffrax-backed adaptive integrator.

    Args:
        solver:               A diffrax solver instance, e.g. ``diffrax.Dopri5()``.
                              Defaults to ``diffrax.Dopri5()``.
        stepsize_controller:  A diffrax step-size controller.  Defaults to
                              ``diffrax.PIDController(rtol=1e-6, atol=1e-8)``.
                              Pass ``diffrax.ConstantStepSize()`` for fixed steps.
        dt0:                  Initial step size hint (required by diffrax).
        adjoint:              Gradient strategy — ``"backsolve"``, ``"recursive"``,
                              or ``"direct"``.  Defaults to ``"backsolve"``.
        max_steps:            Maximum number of solver steps (default 2^20).

    Example — adaptive, adjoint backprop::

        integrator = IntegratorDiffrax(
            solver=diffrax.Dopri5(),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
            adjoint="backsolve",
            dt0=1e-3,
        )

    Example — fixed step, scan-style backprop with checkpointing::

        integrator = IntegratorDiffrax(
            solver=diffrax.Euler(),
            stepsize_controller=diffrax.ConstantStepSize(),
            dt0=1e-4,
            adjoint="recursive",
        )
    """

    def __init__(
        self,
        solver=None,
        stepsize_controller=None,
        dt0: float = 1e-3,
        adjoint: str = "backsolve",
        max_steps: int = 2**20,
        throw: bool = True,
    ):
        try:
            import diffrax as _dfx
        except ImportError as e:
            raise ImportError(
                "IntegratorDiffrax requires diffrax.  Install it with:\n"
                "    pip install diffrax"
            ) from e

        self.solver              = solver              or _dfx.Dopri5()
        self.stepsize_controller = stepsize_controller or _dfx.PIDController(rtol=1e-6, atol=1e-8)
        self.dt0                 = float(dt0)
        self.adjoint_name        = adjoint
        self.max_steps           = max_steps
        self.throw               = throw

        # Resolve adjoint object once at construction time.
        if adjoint == "backsolve":
            self._adjoint = _dfx.BacksolveAdjoint()
        elif adjoint == "recursive":
            self._adjoint = _dfx.RecursiveCheckpointAdjoint()
        elif adjoint == "direct":
            self._adjoint = _dfx.DirectAdjoint()
        else:
            raise ValueError(
                f"Unknown adjoint='{adjoint}'.  "
                "Choose 'backsolve', 'recursive', or 'direct'."
            )

    # ──────────────────────────────────────────────────────────────────── #
    #  Public API                                                          #
    # ──────────────────────────────────────────────────────────────────── #

    def solve(
        self,
        problem,
        inferred_params: Optional[Dict[str, Any]] = None,
        t_obs=None,
    ) -> Dict[str, Any]:
        """Forward-simulate and return states at observation times.

        Args:
            problem:         A fully-configured :class:`~pinns.models.ModelSolver`.
            inferred_params: Dict of JAX-differentiable parameter values.
            t_obs:           1-D array of snapshot times.

        Returns:
            Dict ``{state_name: array(n_obs, *shape)}`` in physical space.
        """
        import diffrax as dfx

        self._check_problem(problem)

        state_names = problem.state_names
        t0          = float(problem._t_min)
        t1          = float(problem._t_max)

        resolved_t_obs = self._resolve_obs_times(problem, t_obs)

        # ── Build params (gradient flows through inferred_params) ──────────
        params = problem._build_params(inferred_params)
        K2     = problem.K2

        # ── Initial state in spectral space ───────────────────────────────
        y0_hat = problem.get_initial_hat()

        # ── Split complex → real/imag to avoid diffrax complex-dtype warning.
        # For each state we store two real arrays: "{name}_re" and "{name}_im".
        # The RHS and recombination step are transparent to the user operators.
        is_complex = {name: jnp.iscomplexobj(y0_hat[name]) for name in state_names}

        def _to_real(y_hat):
            out = {}
            for name in state_names:
                if is_complex[name]:
                    out[f"{name}_re"] = y_hat[name].real
                    out[f"{name}_im"] = y_hat[name].imag
                else:
                    out[name] = y_hat[name]
            return out

        def _to_complex(y_real):
            out = {}
            for name in state_names:
                if is_complex[name]:
                    out[name] = y_real[f"{name}_re"] + 1j * y_real[f"{name}_im"]
                else:
                    out[name] = y_real[name]
            return out

        y0_real = _to_real(y0_hat)

        # ── Full RHS operating on real split state ─────────────────────────
        # IMPORTANT: `params` is passed via `args` (not closed over) so that
        # BacksolveAdjoint's custom_vjp can differentiate through it.
        # `K2` is a static domain array (never differentiated) — safe to close over.
        # `L` is recomputed inside rhs from `_params` to avoid passing complex
        # arrays through `args`, which would trigger diffrax's complex-dtype warning.
        def rhs(t, y_real, args):
            _params = args
            _L = problem._linear_op(K2, _params)
            y_hat = _to_complex(y_real)
            N_hat = problem._nonlinear_op(y_hat, _params)
            dydt_hat = {
                name: _L[name] * y_hat[name] + N_hat[name]
                for name in state_names
            }
            return _to_real(dydt_hat)

        # ── Observation save-points ────────────────────────────────────────
        saveat = dfx.SaveAt(ts=jnp.asarray(resolved_t_obs))

        # ── Solve ──────────────────────────────────────────────────────────
        solution = dfx.diffeqsolve(
            dfx.ODETerm(rhs),
            self.solver,
            t0=t0,
            t1=t1,
            dt0=self.dt0,
            y0=y0_real,
            args=params,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
            adjoint=self._adjoint,
            max_steps=self.max_steps,
            throw=self.throw,
        )
        # solution.ys: {key: (n_obs, *shape)} in real split form

        # ── Recombine real/imag and map back to physical space ─────────────
        # solution.ys values have shape (n_obs, *spatial_shape)
        ys_hat = {
            name: (
                solution.ys[f"{name}_re"] + 1j * solution.ys[f"{name}_im"]
                if is_complex[name]
                else solution.ys[name]
            )
            for name in state_names
        }
        result = {
            name: jax.vmap(problem.inverse)(ys_hat[name])
            for name in state_names
        }

        self._store_result(problem, result, resolved_t_obs)
        return result

    def __repr__(self) -> str:
        return (
            f"IntegratorDiffrax(solver={self.solver!r}, "
            f"adjoint='{self.adjoint_name}', dt0={self.dt0})"
        )
