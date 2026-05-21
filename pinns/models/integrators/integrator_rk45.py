"""
pinns/integrators/integrator_rk45.py — Fixed-step Dormand-Prince (RK45) integrator.

Implements the Dormand-Prince 5th-order Runge-Kutta scheme (the same tableau used
by scipy's ``RK45`` and MATLAB's ``ode45``).  The step size is **fixed** so the
method is compatible with ``jax.lax.scan``, JIT compilation, and reverse-mode AD.

Butcher tableau (DOPRI5 / Dormand-Prince 1980)::

    0      |
    1/5    | 1/5
    3/10   | 3/40        9/40
    4/5    | 44/45      -56/15       32/9
    8/9    | 19372/6561 -25360/2187  64448/6561  -212/729
    1      | 9017/3168  -355/33      46732/5247    49/176   -5103/18656
    1      | 35/384       0          500/1113     125/192   -2187/6784    11/84
    -------+-------------------------------------------------------------------
    b5     | 35/384       0          500/1113     125/192   -2187/6784    11/84    0

The method is explicit and 5th-order accurate per step; no adaptive step control
is performed.  For stiff PDEs prefer :class:`IntegratorETD2RK` or
:class:`IntegratorIMEX`.

The forward loop uses ``jax.lax.scan`` for JIT compilation and efficient AD.
Optional ``checkpoint=True`` wraps the inner step with ``jax.remat`` to trade
recomputation for memory during the backward pass.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp

from .integrator_base import Integrator

__all__ = ["IntegratorRK45"]

# ── Dormand-Prince coefficients ────────────────────────────────────────────────
_C2 = 1.0 / 5.0
_C3 = 3.0 / 10.0
_C4 = 4.0 / 5.0
_C5 = 8.0 / 9.0

_A21 = 1.0 / 5.0

_A31 = 3.0 / 40.0
_A32 = 9.0 / 40.0

_A41 =  44.0 / 45.0
_A42 = -56.0 / 15.0
_A43 =  32.0 / 9.0

_A51 =  19372.0 / 6561.0
_A52 = -25360.0 / 2187.0
_A53 =  64448.0 / 6561.0
_A54 =   -212.0 /  729.0

_A61 =   9017.0 / 3168.0
_A62 =   -355.0 /   33.0
_A63 =  46732.0 / 5247.0
_A64 =     49.0 /  176.0
_A65 =  -5103.0 / 18656.0

# 5th-order weights
_B1 =    35.0 /   384.0
_B3 =   500.0 /  1113.0
_B4 =   125.0 /   192.0
_B5 = -2187.0 /  6784.0
_B6 =    11.0 /    84.0


class IntegratorRK45(Integrator):
    """Fixed-step Dormand-Prince (RK45) integrator.

    Uses the 6-stage, 5th-order Dormand-Prince explicit Runge-Kutta tableau.
    The step size is **fixed** (no adaptive control), making the method fully
    compatible with ``jax.lax.scan``, JIT compilation, and reverse-mode AD.

    Args:
        dt:         Time step size.
        n_steps:    Total number of time steps (optional; inferred if ``None``).
        checkpoint: Wrap inner step with ``jax.remat`` to save memory during AD.

    Note:
        This is an explicit method.  For stiff problems (e.g. Allen-Cahn with
        fine spatial grids) a small ``dt`` will be required for stability.
        Prefer :class:`IntegratorETD2RK` for stiff spectral PDEs.

    Example::

        integrator = IntegratorRK45(dt=1e-4)
        states = integrator.solve(problem)
        # states["u"].shape == (n_obs, Nx)
    """

    def __init__(
        self,
        dt: float,
        n_steps: Optional[int] = None,
        checkpoint: bool = False,
    ):
        self.dt = float(dt)
        self.n_steps = n_steps
        self.checkpoint = checkpoint
        self._jit_cache: dict = {}

    def solve(
        self,
        problem,
        inferred_params: Optional[Dict[str, Any]] = None,
        t_obs=None,
    ) -> Dict[str, Any]:
        """Forward-simulate using fixed-step Dormand-Prince RK45.

        Args:
            problem:         A fully-configured :class:`~pinns.models.ModelSolver`.
            inferred_params: Dict of JAX-differentiable parameter values.
            t_obs:           1-D array of snapshot times.  If ``None``, uses
                             times from ``problem.add_observations()``.

        Returns:
            Dict ``{state_name: array(n_obs, *shape)}`` in physical space.
        """
        self._check_problem(problem)

        dt         = self.dt
        state_names = problem.state_names
        checkpoint  = self.checkpoint

        # ── Python-level observation schedule (not traced by JAX) ──────────
        t0 = problem._t_min
        resolved_t_obs = self._resolve_obs_times(problem, t_obs)
        obs_steps = [int(round((t - t0) / dt)) for t in resolved_t_obs]
        n_obs = len(obs_steps)
        segment_lengths = [obs_steps[0]] + [
            obs_steps[i] - obs_steps[i - 1] for i in range(1, n_obs)
        ]
        uniform = all(s == segment_lengths[0] for s in segment_lengths)
        steps_per_seg = segment_lengths[0]

        # ── JIT-compiled core ───────────────────────────────────────────────
        def _core(inferred):
            params = problem._build_params(inferred)
            K2 = problem.K2
            L  = problem._linear_op(K2, params)

            # Full RHS in spectral space: linear + nonlinear
            def _rhs(sh):
                Nhat = problem._nonlinear_op(sh, params)
                return {
                    name: L[name] * sh[name] + Nhat[name]
                    for name in state_names
                }

            # Weighted sum helper
            def _axpy(sh, k, alpha):
                return {name: sh[name] + alpha * k[name] for name in state_names}

            # Dormand-Prince single step
            def _step(sh, _):
                k1 = _rhs(sh)
                k2 = _rhs(_axpy(sh, k1, dt * _A21))
                k3 = _rhs({
                    name: sh[name] + dt * (_A31 * k1[name] + _A32 * k2[name])
                    for name in state_names
                })
                k4 = _rhs({
                    name: sh[name] + dt * (_A41 * k1[name] + _A42 * k2[name] + _A43 * k3[name])
                    for name in state_names
                })
                k5 = _rhs({
                    name: sh[name] + dt * (
                        _A51 * k1[name] + _A52 * k2[name] + _A53 * k3[name] + _A54 * k4[name]
                    )
                    for name in state_names
                })
                k6 = _rhs({
                    name: sh[name] + dt * (
                        _A61 * k1[name] + _A62 * k2[name] + _A63 * k3[name]
                        + _A64 * k4[name] + _A65 * k5[name]
                    )
                    for name in state_names
                })
                new_sh = {
                    name: sh[name] + dt * (
                        _B1 * k1[name]
                        + _B3 * k3[name]
                        + _B4 * k4[name]
                        + _B5 * k5[name]
                        + _B6 * k6[name]
                    )
                    for name in state_names
                }
                return new_sh, None

            step_fn = jax.remat(_step) if checkpoint else _step
            state_hat_0 = problem.get_initial_hat()

            if uniform and n_obs > 1:
                def _segment(sh, _):
                    final, _ = jax.lax.scan(step_fn, sh, None, length=steps_per_seg)
                    return final, final
                _, seg_states = jax.lax.scan(_segment, state_hat_0, None, length=n_obs)
            else:
                current = state_hat_0
                seg_states_list = []
                for seg_len in segment_lengths:
                    if seg_len <= 0:
                        seg_states_list.append(current)
                        continue
                    current, _ = jax.lax.scan(step_fn, current, None, length=seg_len)
                    seg_states_list.append(current)
                seg_states = {
                    name: jnp.stack([s[name] for s in seg_states_list], axis=0)
                    for name in state_names
                }

            return {
                name: jax.vmap(problem.inverse)(seg_states[name])
                for name in state_names
            }

        cache_key = (id(problem), tuple(obs_steps), self.checkpoint)
        if cache_key not in self._jit_cache:
            self._jit_cache[cache_key] = jax.jit(_core)
        result = self._jit_cache[cache_key](inferred_params)
        self._store_result(problem, result, resolved_t_obs)
        return result

    def __repr__(self) -> str:
        return (
            f"IntegratorRK45(dt={self.dt}, n_steps={self.n_steps}, "
            f"checkpoint={self.checkpoint})"
        )
