"""
pinns/integrators/integrator_rk4.py — Classical RK4 spectral integrator.

Treats the full RHS (linear + nonlinear) explicitly::

    k1 = L·û + N̂(û)
    k2 = L·(û + k1·dt/2) + N̂(û + k1·dt/2)
    k3 = L·(û + k2·dt/2) + N̂(û + k2·dt/2)
    k4 = L·(û + k3·dt)   + N̂(û + k3·dt)
    û_{n+1} = û + dt/6 · (k1 + 2k2 + 2k3 + k4)

The linear operator contributes pointwise (``L * û``) at each stage.
This is simple and accurate but requires ``dt`` small enough for stability
when large eigenvalues are present (i.e. fine grids / large diffusion).

For stiff problems with spectral discretisation, prefer :class:`IntegratorETD2RK`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp

from .integrator_base import Integrator

__all__ = ["IntegratorRK4"]


class IntegratorRK4(Integrator):
    """Classical 4th-order Runge-Kutta integrator.

    Args:
        dt:         Time step size.
        n_steps:    Total number of time steps (optional; inferred if ``None``).
        checkpoint: Wrap inner step with ``jax.remat`` to save memory during AD.
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
        # Cache: maps (problem_id, obs_schedule_tuple) -> jitted _core fn
        self._jit_cache: dict = {}

    def solve(
        self,
        problem,
        inferred_params: Optional[Dict[str, Any]] = None,
        t_obs=None,
    ) -> Dict[str, Any]:
        """Forward-simulate using RK4 and return states at observation times.

        Args:
            problem:         A fully-configured :class:`~pinns.models.ModelSolver`.
            inferred_params: Dict of JAX-differentiable parameter values.
            t_obs:           1-D array of snapshot times.  If ``None``, uses
                             times from ``problem.add_observations()``.

        Returns:
            Dict ``{state_name: array(n_obs, *shape)}`` in physical space.
        """
        self._check_problem(problem)

        dt = self.dt
        state_names = problem.state_names
        checkpoint = self.checkpoint

        # ── Python-level observation schedule (not traced by JAX) ────
        t0 = problem._t_min
        resolved_t_obs = self._resolve_obs_times(problem, t_obs)
        obs_steps = [int(round((t - t0) / dt)) for t in resolved_t_obs]
        n_obs = len(obs_steps)
        segment_lengths = [obs_steps[0]] + [
            obs_steps[i] - obs_steps[i - 1] for i in range(1, n_obs)
        ]
        uniform = all(s == segment_lengths[0] for s in segment_lengths)
        steps_per_seg = segment_lengths[0]

        # ── JIT-compiled core: takes inferred params as explicit arg ──
        def _core(inferred):
            params = problem._build_params(inferred)
            K2 = problem.K2
            L = problem._linear_op(K2, params)

            def _full_rhs(sh):
                Nhat = problem._nonlinear_op(sh, params)
                return {name: L[name] * sh[name] + Nhat[name] for name in state_names}

            def _add(a, b, scale=1.0):
                return {name: a[name] + scale * b[name] for name in state_names}

            def _step(sh, _):
                k1 = _full_rhs(sh)
                k2 = _full_rhs(_add(sh, k1, dt / 2.0))
                k3 = _full_rhs(_add(sh, k2, dt / 2.0))
                k4 = _full_rhs(_add(sh, k3, dt))
                new_sh = {
                    name: sh[name]
                           + (dt / 6.0) * (k1[name] + 2 * k2[name] + 2 * k3[name] + k4[name])
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
            f"IntegratorRK4(dt={self.dt}, n_steps={self.n_steps}, "
            f"checkpoint={self.checkpoint})"
        )
