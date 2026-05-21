"""
pinns/integrators/integrator_euler.py — Forward Euler (fixed-step) integrator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp

from .integrator_base import Integrator

__all__ = ["IntegratorEuler"]


class IntegratorEuler(Integrator):
    """Forward Euler integrator (1st order, fixed step).

    Advances the spectral state by::

        û_{n+1} = û_n + dt · [L · û_n + N̂(û_n)]

    This is explicit and only first-order accurate.  It requires a very small
    ``dt`` for stability on stiff problems.  Prefer :class:`IntegratorETD2RK`
    or :class:`IntegratorIMEX` for spectral PDEs with large linear eigenvalues.

    Args:
        dt:         Fixed time step size.
        n_steps:    Total number of steps (optional; inferred if ``None``).
        checkpoint: Wrap inner step with ``jax.remat`` to reduce peak memory
                    during reverse-mode AD.
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

    def _one_step(self, problem, state_hat, t, dt, L, params):
        """Single Euler step; error estimate is zero (no embedded method)."""
        state_names = problem.state_names
        Nhat = problem._nonlinear_op(state_hat, params)
        sh_new = {
            name: state_hat[name] + dt * (L[name] * state_hat[name] + Nhat[name])
            for name in state_names
        }
        err_abs = {name: jnp.zeros_like(sh_new[name]) for name in state_names}
        return sh_new, err_abs

    def solve(
        self,
        problem,
        inferred_params: Optional[Dict[str, Any]] = None,
        t_obs=None,
    ) -> Dict[str, Any]:
        """Forward-simulate using forward Euler and return states at obs times."""
        self._check_problem(problem)

        dt          = self.dt
        state_names = problem.state_names
        checkpoint  = self.checkpoint

        t0 = problem._t_min
        resolved_t_obs = self._resolve_obs_times(problem, t_obs)
        obs_steps = [int(round((t - t0) / dt)) for t in resolved_t_obs]
        n_obs = len(obs_steps)
        segment_lengths = [obs_steps[0]] + [
            obs_steps[i] - obs_steps[i - 1] for i in range(1, n_obs)
        ]
        uniform = all(s == segment_lengths[0] for s in segment_lengths)
        steps_per_seg = segment_lengths[0]

        def _core(inferred):
            params = problem._build_params(inferred)
            K2 = problem.K2
            L  = problem._linear_op(K2, params)

            def _step(sh, _):
                Nhat = problem._nonlinear_op(sh, params)
                new_sh = {
                    name: sh[name] + dt * (L[name] * sh[name] + Nhat[name])
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
            f"IntegratorEuler(dt={self.dt}, n_steps={self.n_steps}, "
            f"checkpoint={self.checkpoint})"
        )
