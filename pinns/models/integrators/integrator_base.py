"""
pinns/integrators/integrator_base.py — Abstract base class for integrators.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp


class Integrator(ABC):
    """Abstract base class for spectral time integrators.

    All integrators expose a single :meth:`solve` method that takes a
    :class:`~pinns.models.ModelSpectralSolver` and an optional dict of inferred
    parameter values, and returns the simulated state at the observation
    times registered via ``problem.add_observations()``.

    Subclasses must implement :meth:`solve`; they may optionally override
    :meth:`_one_step` for readability.
    """

    @abstractmethod
    def solve(
        self,
        problem,
        inferred_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Forward-simulate the PDE and return states at observation times.

        Args:
            problem: A fully-configured :class:`~pinns.models.ModelSpectralSolver`.
            inferred_params: Dict of differentiable parameter values.
                If ``None``, the values stored in ``problem.inferred_params``
                are used (as plain Python scalars, not JAX arrays).

        Returns:
            Dict mapping ``state_name`` → JAX array of shape ``(n_obs, *shape)``
            with the physical-space state at each observation time.
        """
        ...

    def _check_problem(self, problem) -> None:
        """Validate problem type and configuration.

        Accepts any object carrying the ``_is_solver_problem = True`` marker,
        which covers :class:`~pinns.models.ModelSpectralSolver`.
        """
        if not getattr(problem, '_is_solver_problem', False):
            raise TypeError(
                f"{type(self).__name__}.solve() expects a ModelSpectralSolver, "
                f"got {type(problem).__name__}."
            )
        problem._validate()

    def _resolve_obs_times(self, problem, t_obs):
        """Return observation times, preferring the ``t_obs`` argument.

        Args:
            problem: A :class:`~pinns.models.ModelSpectralSolver`.
            t_obs:   Optional array of observation times passed directly to
                     ``solve()``.  If ``None``, falls back to the times
                     registered via ``problem.add_observations()``.

        Raises:
            RuntimeError: If neither source provides observation times.
        """
        import numpy as np
        if t_obs is not None:
            return np.asarray(t_obs)
        if problem._obs_times is not None:
            return problem._obs_times
        raise RuntimeError(
            "No observation times available.  Either call "
            "problem.add_observations() or pass t_obs= to solve()."
        )

    # ──────────────────────────────────────────────────────────────────── #
    #  Single-step interface (override in adaptive-capable integrators)   #
    # ──────────────────────────────────────────────────────────────────── #

    def _one_step(self, problem, state_hat, t, dt, L, params):
        """Perform a single time step and return an embedded error estimate.

        Override in subclasses that support adaptive time stepping.

        Args:
            problem:   A :class:`~pinns.models.ModelSpectralSolver`.
            state_hat: Dict of Fourier-space state arrays at time ``t``.
            t:         Current time (JAX scalar).
            dt:        Step size to attempt (JAX scalar).
            L:         Dict of linear-operator eigenvalue arrays.
            params:    Full parameter dict (physical + inferred).

        Returns:
            ``(state_hat_new, err_abs)`` where ``err_abs`` is a dict with the
            same structure as ``state_hat`` containing element-wise absolute
            error estimates.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _one_step. "
            "Override _one_step to use _adaptive_solve."
        )

    # ──────────────────────────────────────────────────────────────────── #
    #  Generic adaptive solve loop (shared by all adaptive integrators)   #
    # ──────────────────────────────────────────────────────────────────── #

    def _adaptive_solve(
        self,
        problem,
        inferred_params,
        controller,
        dt0: float,
        max_steps: int,
        checkpoint: bool = False,
        t_obs=None,
    ) -> Dict[str, Any]:
        """Adaptive-step loop driven by *controller*.

        Uses ``jax.lax.scan`` over ``max_steps`` iterations (required for
        ``jax.grad`` compatibility).  A ``done`` flag causes the body to
        no-op once the end time is reached; the remaining iterations do not
        affect the output.

        All branching uses ``jnp.where`` (never ``lax.cond``) to avoid the
        VJP issue where both branches of ``lax.cond`` are evaluated during
        the backward pass.

        Args:
            problem:         A :class:`~pinns.models.ModelSpectralSolver`.
            inferred_params: Differentiable parameter dict (passed as explicit
                             JAX pytree so ``jax.grad`` traces through it).
            controller:      A :class:`~pinns.integrators.StepsizeController`
                             instance.
            dt0:             Initial step size.
            max_steps:       Maximum number of attempted steps (both accepted
                             and rejected).
            checkpoint:      Wrap ``_one_step`` with ``jax.remat``.

        Returns:
            Dict ``{state_name: array(n_obs, *shape)}`` in physical space, one
            snapshot per registered observation time.
        """
        self._check_problem(problem)

        state_names = problem.state_names
        t0 = jnp.float32(problem._t_min)
        t1 = jnp.float32(problem._t_max)

        # Observation times as a JAX array (static length, traced values)
        resolved_t_obs = self._resolve_obs_times(problem, t_obs)
        import numpy as np
        t_obs_np  = np.asarray(resolved_t_obs, dtype=np.float32)
        n_obs     = len(t_obs_np)

        # Tolerances for error normalisation (from controller if available)
        rtol = float(getattr(controller, 'rtol', 1e-4))
        atol = float(getattr(controller, 'atol', 1e-6))

        one_step_fn = (
            jax.remat(
                lambda sh, t, dt, L, p: self._one_step(problem, sh, t, dt, L, p)
            )
            if checkpoint
            else
            lambda sh, t, dt, L, p: self._one_step(problem, sh, t, dt, L, p)
        )

        def _core(inferred):
            params = problem._build_params(inferred)
            K2     = problem.K2
            L      = problem._linear_op(K2, params)

            t_obs_jax = jnp.array(t_obs_np)  # (n_obs,) — traced but static shape

            # Initial spectral state
            sh0 = problem.get_initial_hat()

            # Observation buffer — shape (n_obs, *mode_shape) per state
            # Initialised to zeros; will be filled as integration proceeds.
            # Pre-fill any observation time that equals t0 with the IC so
            # the crossing condition (t < t_obs[i]) never misses t=t0.
            obs_buf = {}
            for name in state_names:
                buf = jnp.zeros(
                    (n_obs,) + sh0[name].shape, dtype=sh0[name].dtype
                )
                for i, t_i in enumerate(t_obs_np):
                    if abs(float(t_i) - float(t0)) < 1e-12 * max(abs(float(t1) - float(t0)), 1e-30):
                        buf = buf.at[i].set(sh0[name])
                obs_buf[name] = buf

            ctrl_state0 = controller.init()

            # Carry: (state_hat, t, dt, obs_buf, ctrl_state, done)
            init_carry = (
                sh0,
                t0,
                jnp.float32(dt0),
                obs_buf,
                ctrl_state0,
                jnp.bool_(False),
            )

            def body(carry, _):
                sh, t, dt, obs_buf, ctrl_state, done = carry

                # ── clamp dt to land exactly on the next observation time ──
                # Find the minimum upcoming obs time (greater than current t)
                future_mask = t_obs_jax > t + jnp.float32(1e-9) * (t1 - t0)
                next_obs_t  = jnp.where(
                    jnp.any(future_mask),
                    jnp.min(jnp.where(future_mask, t_obs_jax, t1 + jnp.float32(1.0))),
                    t1,
                )
                dt_actual = jnp.minimum(dt, next_obs_t - t)
                dt_actual = jnp.minimum(dt_actual, t1 - t)
                dt_actual = jnp.maximum(dt_actual, jnp.float32(1e-30))
                # Step-size scheduling decisions should NOT carry gradients.
                # Differentiating through the controller produces NaN gradients
                # because accept/reject decisions are discontinuous.  The correct
                # approach (analogous to the continuous adjoint) is to treat the
                # step schedule as fixed during the backward pass.
                dt_actual = jax.lax.stop_gradient(dt_actual)

                # ── step ────────────────────────────────────────────────
                sh_new, err_abs = one_step_fn(sh, t, dt_actual, L, params)

                # ── error normalisation (RMS over all states & modes) ──
                err_sq_sum = sum(
                    jnp.mean(err_abs[name] ** 2)
                    for name in state_names
                )
                ref_sq_sum = sum(
                    jnp.mean((rtol * jnp.abs(sh[name]) + atol) ** 2)
                    for name in state_names
                )
                err_norm = jnp.sqrt(err_sq_sum / (ref_sq_sum + jnp.float32(1e-30)))

                # ── controller ──────────────────────────────────────────
                dt_prop, accept, ctrl_state_new = controller.step(
                    err_norm, dt_actual, ctrl_state
                )
                # Stop gradient through controller outputs: accept/reject
                # decisions are discontinuous scheduling choices and must not
                # participate in the backward pass (same principle as the
                # continuous adjoint method for adaptive solvers).
                dt_prop        = jax.lax.stop_gradient(dt_prop)
                accept         = jax.lax.stop_gradient(accept)
                ctrl_state_new = jax.lax.stop_gradient(ctrl_state_new)

                # ── state update — jnp.where is AD-safe, lax.cond is not
                t_next = jnp.where(accept, t + dt_actual, t)
                sh_next = {
                    name: jnp.where(accept, sh_new[name], sh[name])
                    for name in state_names
                }

                # Carry dt: use controller proposal, don't overshoot t1
                dt_next = jnp.minimum(dt_prop, t1 - t_next + jnp.float32(1e-30))
                dt_next = jnp.maximum(dt_next, jnp.float32(1e-30))

                # ── observation recording ────────────────────────────────
                # We record obs[i] when the accepted step lands exactly on it.
                # Since dt_actual is clamped to land on the next obs time,
                # t_next == t_obs[i] (within floating-point tolerance) at
                # the appropriate step.
                t_after = t + dt_actual   # proposed next t (before accept decision)
                for i in range(n_obs):
                    crossed_i = (
                        (~done)
                        & accept
                        & (t < t_obs_jax[i])
                        & (t_after >= t_obs_jax[i])
                    )
                    new_obs_buf = {}
                    for name in state_names:
                        new_val = jnp.where(
                            crossed_i,
                            sh_new[name],
                            obs_buf[name][i],
                        )
                        new_obs_buf[name] = obs_buf[name].at[i].set(new_val)
                    obs_buf = new_obs_buf

                # ── done flag ────────────────────────────────────────────
                done_new  = done | (t_next >= t1 - jnp.float32(1e-7) * (t1 - t0))
                # No-op when already done (mask all updates)
                sh_out    = {name: jnp.where(done, sh[name], sh_next[name])
                             for name in state_names}
                # t, dt, ctrl, done are scheduling scalars — stop their grads
                # so the backward pass only flows through the state arrays.
                t_out     = jax.lax.stop_gradient(jnp.where(done, t, t_next))
                dt_out    = jax.lax.stop_gradient(jnp.where(done, dt, dt_next))
                ctrl_out  = jax.lax.stop_gradient(jnp.where(done, ctrl_state, ctrl_state_new))
                done_out  = jax.lax.stop_gradient(done_new)

                return (sh_out, t_out, dt_out, obs_buf, ctrl_out, done_out), None

            (_, _, _, obs_buf_final, _, _), _ = jax.lax.scan(
                body, init_carry, None, length=max_steps
            )

            # Convert Fourier-space snapshots to physical space
            return {
                name: jax.vmap(problem.inverse)(obs_buf_final[name])
                for name in state_names
            }

        result = jax.jit(_core)(inferred_params)
        self._store_result(problem, result, resolved_t_obs)
        return result

    # ──────────────────────────────────────────────────────────────────── #
    #  Post-solve interpolation cache                                      #
    # ──────────────────────────────────────────────────────────────────── #

    def _store_result(self, problem, result: Dict[str, Any], t_obs) -> None:
        """Cache the solved solution and build RegularGridInterpolants.

        Called by subclass ``solve()`` before returning.  After this,
        :meth:`apply` can be used to evaluate the solution at arbitrary
        space-time points, exactly like a trained network.

        Args:
            problem: The :class:`~pinns.models.ModelSpectralSolver` that was solved.
            result:  Dict ``{state_name: array(n_obs, *shape)}`` (physical space).
            t_obs:   1-D numpy array of observation times used in the solve.
        """
        import numpy as np
        from scipy.interpolate import RegularGridInterpolator
        import jax.core

        # Skip when called inside a JAX tracing context (e.g. jax.grad).
        # The interpolant is only useful for concrete arrays.
        leaves = jax.tree_util.tree_leaves(result)
        if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
            return

        self._last_result = result
        self._last_problem = problem
        self._last_t_obs = t_obs

        # Build one interpolant per state variable.
        # Grid axes: (t, x) for 1-D, (t, x, y) for 2-D, etc.
        grids = problem._grids                    # list of 1-D physical coord arrays
        t_ax = np.asarray(t_obs)
        axes = (t_ax,) + tuple(grids)            # (t, x[, y, ...])

        self._interpolants: Dict[str, Any] = {}
        for name, arr in result.items():
            arr_np = np.array(arr)               # (n_obs, *shape)
            self._interpolants[name] = RegularGridInterpolator(
                axes, arr_np,
                method="linear", bounds_error=False, fill_value=None,
            )

    def apply(self, X) -> "np.ndarray":
        """Evaluate the last solved solution at arbitrary space-time points.

        Mirrors the ``u_reference(X)`` / ``network.apply(params, X)`` interface
        used by the PINN trainer.

        Args:
            X: Array of shape ``(N, n_dims + 1)`` where the first ``n_dims``
               columns are spatial coordinates and the last column is time.
               Matches the convention used by :class:`~pinns.domain.DomainCubic`.

        Returns:
            Array of shape ``(N, n_states)``.  For single-state problems this
            is ``(N, 1)``, matching ``u_reference(X)``.

        Raises:
            RuntimeError: If :meth:`solve` has not been called yet.
        """
        import numpy as np

        if not hasattr(self, "_interpolants"):
            raise RuntimeError(
                "Call solve() before apply().  No solution is cached yet."
            )

        n_dims = self._last_problem.n_dims
        X = np.asarray(X)
        # Build query points in (t, x[, y, ...]) order expected by the interpolant
        # X columns: [x0, x1, ..., t]  →  reorder to [t, x0, x1, ...]
        space_cols = X[:, :n_dims]               # (N, n_dims)
        t_col      = X[:, n_dims : n_dims + 1]  # (N, 1)
        query = np.concatenate([t_col, space_cols], axis=1)  # (N, n_dims+1)

        state_names = self._last_problem.state_names
        cols = [self._interpolants[name](query)[:, None] for name in state_names]
        return np.concatenate(cols, axis=1)      # (N, n_states)
