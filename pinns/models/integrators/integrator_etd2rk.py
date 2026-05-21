"""
pinns/integrators/integrator_etd2rk.py — ETD2RK spectral time integrator.

Exponential Time Differencing 2nd-order Runge-Kutta (Cox-Matthews scheme,
Hochbruck & Ostermann 2005).  The linear part is treated exactly via matrix
exponentials; in the spectral / Fourier basis these reduce to pointwise
scalar multiplications ``exp(L * dt)``.

Scheme (per time step)::

    k1       = N̂(û_n)
    û_★      = E½ · û_n  + (dt/2) · φ₁(L dt/2) · k1
    k2       = N̂(û_★)
    û_{n+1}  = E  · û_n  + dt · [φ₁(L dt) · k1 + φ₂(L dt) · (k2 − k1)]

where:
    E        = exp(L dt)
    E½       = exp(L dt / 2)
    φ₁(z)    = (e^z − 1) / z    [safe near z=0]
    φ₂(z)    = (φ₁(z) − 1) / z  [safe near z=0]

The forward loop is implemented with ``jax.lax.scan`` for JIT compilation
and efficient reverse-mode AD.  Optional ``checkpoint=True`` wraps the inner
step with ``jax.remat`` to trade recomputation for memory.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp

from .integrator_base import Integrator

__all__ = ["IntegratorETD2RK"]


def _phi1(z):
    """φ₁(z) = (e^z − 1) / z, numerically safe near z=0 (including JAX AD)."""
    # Use a safe z to avoid 0/0 in the non-selected branch during backward pass.
    safe_z = jnp.where(jnp.abs(z) < 1e-8, jnp.ones_like(z), z)
    return jnp.where(
        jnp.abs(z) < 1e-8,
        1.0 + z / 2.0 + z ** 2 / 6.0,
        (jnp.exp(safe_z) - 1.0) / safe_z,
    )


def _phi2(z):
    """φ₂(z) = (φ₁(z) − 1) / z, numerically safe near z=0 (including JAX AD)."""
    p1 = _phi1(z)
    safe_z = jnp.where(jnp.abs(z) < 1e-8, jnp.ones_like(z), z)
    return jnp.where(jnp.abs(z) < 1e-8, 0.5 + z / 6.0, (p1 - 1.0) / safe_z)


class IntegratorETD2RK(Integrator):
    """Exponential Time Differencing 2nd-order Runge-Kutta integrator.

    Attributes:
        dt:          Time step size.
        n_steps:     Total number of time steps.  If ``None``, inferred from
                     ``problem.domain._t_max - problem.domain._t_min`` and ``dt``.
        checkpoint:  If ``True``, wrap the inner scan body with ``jax.remat``
                     to reduce peak memory during reverse-mode AD at the cost
                     of recomputing activations during the backward pass.

    Example::

        integrator = IntegratorETD2RK(dt=1e-4, n_steps=20_000, checkpoint=True)
        states = integrator.solve(problem, inferred_params={"eps1": 0.18, "eps2": 0.09})
        # states["u"].shape == (n_obs, 64, 64)
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

    # ──────────────────────────────────────────────────────────────────── #
    #  Public API                                                          #
    # ──────────────────────────────────────────────────────────────────── #

    def solve(
        self,
        problem,
        inferred_params: Optional[Dict[str, Any]] = None,
        t_obs=None,
    ) -> Dict[str, Any]:
        """Forward-simulate using ETD2RK and return states at observation times.

        Args:
            problem:          A fully-configured :class:`~pinns.models.ModelSolver`.
            inferred_params:  Dict of JAX-differentiable parameter values.
            t_obs:            1-D array of times at which to snapshot the state.
                              If ``None``, uses times from ``problem.add_observations()``.

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
        # Capturing inferred as an explicit pytree argument means
        # jax.grad / jax.value_and_grad will differentiate through it.
        def _core(inferred):
            params = problem._build_params(inferred)
            K2 = problem.K2

            # Linear operator eigenvalues
            L = problem._linear_op(K2, params)

            # ETD2RK precomputed coefficients
            coeffs = {}
            for name in state_names:
                Ln = L[name]
                z    = Ln * dt
                z2   = Ln * (dt / 2.0)
                coeffs[name] = (
                    jnp.exp(z),      # E
                    jnp.exp(z2),     # E2
                    _phi1(z),        # ph1
                    _phi1(z2),       # ph12
                    _phi2(z),        # ph2
                )

            state_hat_0 = problem.get_initial_hat()

            def _step(sh, _):
                N1 = problem._nonlinear_op(sh, params)
                sh_star = {
                    name: coeffs[name][1] * sh[name]
                           + (dt / 2.0) * coeffs[name][3] * N1[name]
                    for name in state_names
                }
                N2 = problem._nonlinear_op(sh_star, params)
                new_sh = {
                    name: coeffs[name][0] * sh[name]
                           + dt * (coeffs[name][2] * N1[name]
                                   + coeffs[name][4] * (N2[name] - N1[name]))
                    for name in state_names
                }
                return new_sh, None

            step_fn = jax.remat(_step) if checkpoint else _step

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

        # ── Cache the jitted core to avoid recompiling on every call ──
        # The cache key captures all static structure that determines the
        # compiled XLA program: which problem object, what observation
        # schedule, and whether gradient checkpointing is active.
        # The trainable parameters are passed as explicit JAX arguments so
        # JAX's XLA cache handles them correctly.
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
        """One ETD2RK step with embedded ETD1 error estimate.

        The ETD2-RK (Cox-Matthews) scheme is 2nd-order; the embedded
        ETD1 (1st-order) scheme omits the φ₂ correction::

            û_{n+1}^{ETD1}  = E·û_n + dt·φ₁(Ldt)·k1
            û_{n+1}^{ETD2}  = E·û_n + dt·[φ₁(Ldt)·k1 + φ₂(Ldt)·(k2−k1)]
            err              = û_{n+1}^{ETD2} − û_{n+1}^{ETD1}
                             = dt · φ₂(Ldt) · (k2 − k1)

        Args:
            problem:   :class:`~pinns.models.ModelSolver`.
            state_hat: Fourier-space state dict at time ``t``.
            t:         Current time (ignored by ETD2RK but kept for uniformity).
            dt:        Step size (JAX scalar).
            L:         Dict of linear-operator eigenvalue arrays.
            params:    Full parameter dict.

        Returns:
            ``(state_hat_new, err_abs)`` — both in Fourier space.
        """
        state_names = problem.state_names

        # Pre-compute ETD2RK coefficients for the current dt
        coeffs = {}
        for name in state_names:
            Ln  = L[name]
            z   = Ln * dt
            z2  = Ln * (dt / 2.0)
            coeffs[name] = (
                jnp.exp(z),    # E
                jnp.exp(z2),   # E2  (half-step)
                _phi1(z),      # ph1
                _phi1(z2),     # ph12
                _phi2(z),      # ph2
            )

        N1 = problem._nonlinear_op(state_hat, params)
        sh_star = {
            name: (
                coeffs[name][1] * state_hat[name]
                + (dt / 2.0) * coeffs[name][3] * N1[name]
            )
            for name in state_names
        }
        N2 = problem._nonlinear_op(sh_star, params)

        sh_new = {
            name: (
                coeffs[name][0] * state_hat[name]
                + dt * (
                    coeffs[name][2] * N1[name]
                    + coeffs[name][4] * (N2[name] - N1[name])
                )
            )
            for name in state_names
        }

        # Error = φ₂ correction = difference between ETD2 and ETD1
        err_abs = {
            name: jnp.abs(dt * coeffs[name][4] * (N2[name] - N1[name]))
            for name in state_names
        }

        return sh_new, err_abs

    def __repr__(self) -> str:
        return (
            f"IntegratorETD2RK(dt={self.dt}, n_steps={self.n_steps}, "
            f"checkpoint={self.checkpoint})"
        )
