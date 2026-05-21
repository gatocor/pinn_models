"""
pinns/integrators/integrator_imex.py — IMEX Euler spectral integrator.

Implicit-Explicit (IMEX) Euler method.  The linear part is treated
**implicitly** (A-stable for all dt) and the nonlinear part explicitly::

    û_{n+1} = (1 - dt · L)^{-1} · (û_n + dt · N̂(û_n))

Since ``L`` is a diagonal operator in the spectral basis, the implicit
inversion ``(1 - dt · L)^{-1}`` is a pointwise scalar division::

    û_{n+1}[k] = (û_n[k] + dt · N̂(û_n)[k]) / (1 - dt · L[k])

This makes IMEX Euler **unconditionally stable** for the linear (diffusion)
part at any ``dt``, while remaining explicit for the nonlinear reactions.
It is only first-order accurate in time; for better accuracy combine with
IMEX midpoint or use :class:`IntegratorETD2RK`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp

from .integrator_base import Integrator

__all__ = ["IntegratorIMEX"]


class IntegratorIMEX(Integrator):
    """First-order IMEX Euler spectral integrator.

    The linear part (diffusion) is treated implicitly — stable for any
    ``dt`` — while the nonlinear part is treated explicitly.

    Args:
        dt:         Time step size.
        n_steps:    Total number of time steps (optional).
        checkpoint: Wrap inner step with ``jax.remat`` for reduced memory.

    Note:
        Only first-order accurate.  Use :class:`IntegratorETD2RK` for better
        accuracy on stiff spectral PDEs.
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
        """Forward-simulate using IMEX Euler and return states at observation times.

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
            denom = {name: 1.0 - dt * L[name] for name in state_names}

            def _step(sh, _):
                Nhat = problem._nonlinear_op(sh, params)
                new_sh = {
                    name: (sh[name] + dt * Nhat[name]) / denom[name]
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

    # ──────────────────────────────────────────────────────────────────── #
    #  Single-step interface (used by _adaptive_solve in the base class)  #
    # ──────────────────────────────────────────────────────────────────── #

    def _one_step(self, problem, state_hat, t, dt, L, params):
        """One IMEX-Euler step with Richardson error estimate.

        Uses Richardson extrapolation (one full step of ``dt`` vs two half-steps
        of ``dt/2``) to obtain a 2nd-order-accurate error surrogate from a
        1st-order scheme.  The full step is returned as the solution.

        Note:
            This doubles the cost per step (3 nonlinear evaluations instead of
            1) but provides a reliable error estimate for adaptive control.

        Args:
            problem:   :class:`~pinns.models.ModelSolver`.
            state_hat: Fourier-space state dict.
            t:         Current time (unused by IMEX-Euler but kept for API
                       consistency).
            dt:        Step size.
            L:         Dict of linear-operator eigenvalue arrays.
            params:    Full parameter dict.

        Returns:
            ``(state_hat_new, err_abs)`` — both in Fourier space.
        """
        state_names = problem.state_names

        # ── full step ────────────────────────────────────────────────────
        Nhat   = problem._nonlinear_op(state_hat, params)
        denom  = {name: jnp.float32(1.0) - dt * L[name] for name in state_names}
        sh_full = {
            name: (state_hat[name] + dt * Nhat[name]) / denom[name]
            for name in state_names
        }

        # ── two half-steps (Richardson) ──────────────────────────────────
        dt2     = dt / jnp.float32(2.0)
        denom2  = {name: jnp.float32(1.0) - dt2 * L[name] for name in state_names}
        sh_mid  = {
            name: (state_hat[name] + dt2 * Nhat[name]) / denom2[name]
            for name in state_names
        }
        Nhat2   = problem._nonlinear_op(sh_mid, params)
        sh_fine = {
            name: (sh_mid[name] + dt2 * Nhat2[name]) / denom2[name]
            for name in state_names
        }

        # Error = |full − fine|, scaled to match tolerance units
        err_abs = {
            name: jnp.abs(sh_full[name] - sh_fine[name])
            for name in state_names
        }

        return sh_full, err_abs

    def __repr__(self) -> str:
        return (
            f"IntegratorIMEX(dt={self.dt}, n_steps={self.n_steps}, "
            f"checkpoint={self.checkpoint})"
        )
