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

from .integrator_base import (
    Integrator, _apply_L, _build_kronecker_matrix, _KronMatApplicator,
)

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


def _phi_mats(M):
    """Compute matrix φ₁(M) and φ₂(M) via a single augmented-matrix expm.

    Uses the identity (traceable by JAX, works inside ``jax.jit``)::

        exp([[M, I, 0],    =  [[e^M,  φ₁(M),  φ₂(M)],
             [0, 0, I],        [0,    I,       I     ],
             [0, 0, 0]])        [0,    0,       I     ]]

    Args:
        M: Square JAX array of shape ``(N, N)``.

    Returns:
        Tuple ``(φ₁(M), φ₂(M))`` — both real ``(N, N)`` JAX arrays.
    """
    N = M.shape[0]
    I = jnp.eye(N, dtype=M.dtype)
    Z = jnp.zeros((N, N), dtype=M.dtype)
    block = jnp.block([[M, I, Z],
                       [Z, Z, I],
                       [Z, Z, Z]])
    E = jax.scipy.linalg.expm(block)
    return E[:N, N:2*N].real, E[:N, 2*N:].real


def _phi1_mat(M):
    """Matrix φ₁(M) = (e^M − I) M⁻¹.  Computed via augmented expm (JAX-traceable)."""
    ph1, _ = _phi_mats(M)
    return ph1


def _phi2_mat(M):
    """Matrix φ₂(M) = (φ₁(M) − I) M⁻¹.  Computed via augmented expm (JAX-traceable)."""
    _, ph2 = _phi_mats(M)
    return ph2


def _compute_kron_coeffs(Ln_list, dt):
    """Precompute ETD2RK coefficients for a Kronecker-sum linear operator.

    Builds the full ``(N_total, N_total)`` Kronecker-sum matrix, then computes
    matrix exponentials and \u03c6 functions via the augmented-block ``expm`` trick
    (same as the 1-D Chebyshev path).  All operations are differentiable inside
    ``jax.jit`` / ``jax.grad`` — no eigenvector decompositions needed.

    Args:
        Ln_list: List of ``(Nᵢ, Nᵢ)`` linear-op matrices, one per spatial dim.
        dt:      Scalar time-step size.

    Returns:
        Tuple ``(E, E₂, φ₁(dt), φ₁(dt/2), φ₂(dt))`` of
        :class:`_KronMatApplicator`.
    """
    spatial_shape = tuple(Li.shape[0] for Li in Ln_list)
    C  = _build_kronecker_matrix(Ln_list)  # (N_total, N_total)
    z  = C * dt
    z2 = C * (dt / 2.0)
    ph1,  ph2  = _phi_mats(z)
    ph12, _    = _phi_mats(z2)
    mk = lambda M: _KronMatApplicator(M, spatial_shape)  # noqa: E731
    return (
        mk(jax.scipy.linalg.expm(z)),   # E
        mk(jax.scipy.linalg.expm(z2)),  # E2
        mk(ph1),                         # ph1
        mk(ph12),                        # ph12
        mk(ph2),                         # ph2
    )


def _apply(A, v):
    """Multiply ``A`` by ``v``.

    Dispatch rules:

    * :class:`_KronMatApplicator` — full Kronecker matrix path (flatten, multiply,
      reshape); fully differentiable via ``jax.scipy.linalg.expm``.
    * **Same shape** as ``v`` — diagonal / Fourier: elementwise ``A * v``.
    * **Dense matrix** — 1-D Chebyshev: ``A @ v``.
    """
    if isinstance(A, _KronMatApplicator):
        return (A.mat @ v.ravel()).reshape(v.shape)
    if A.shape == v.shape:
        return A * v          # diagonal / Fourier (any spatial dimension)
    return A @ v              # dense matrix operator (Chebyshev, etc.)


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
    ) -> Dict[str, Any]:
        """Forward-simulate using ETD2RK and return states at observation times.

        Observation times are derived from the domain bounds and ``self.dt``.

        Args:
            problem:          A fully-configured :class:`~pinns.models.ModelSpectralSolver`.
            inferred_params:  Dict of JAX-differentiable parameter values.

        Returns:
            Dict ``{state_name: array(n_obs, *shape)}`` in physical space.
        """
        self._check_problem(problem)

        dt = self.dt
        state_names = problem.state_names
        checkpoint = self.checkpoint

        # ── Python-level observation schedule (not traced by JAX) ────
        t0 = problem._t_min
        resolved_t_obs = self._get_obs_times(problem)
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

            # Linear operator eigenvalues (diagonal) or matrices (Chebyshev)
            L = problem._call_linear_op(params)

            # ETD2RK precomputed coefficients
            coeffs = {}
            for name in state_names:
                Ln = L[name]
                if isinstance(Ln, list):
                    # Kronecker-sum path (2-D+ Chebyshev)
                    coeffs[name] = _compute_kron_coeffs(Ln, dt)
                elif Ln.ndim == 2:
                    # Dense-matrix path (1-D Chebyshev)
                    z    = Ln * dt
                    z2   = Ln * (dt / 2.0)
                    coeffs[name] = (
                        jax.scipy.linalg.expm(z),   # E
                        jax.scipy.linalg.expm(z2),  # E2
                        _phi1_mat(z),               # ph1
                        _phi1_mat(z2),              # ph12
                        _phi2_mat(z),               # ph2
                    )
                else:
                    # Diagonal path (Fourier / DST / DCT)
                    z    = Ln * dt
                    z2   = Ln * (dt / 2.0)
                    coeffs[name] = (
                        jnp.exp(z),      # E
                        jnp.exp(z2),     # E2
                        _phi1(z),        # ph1
                        _phi1(z2),       # ph12
                        _phi2(z),        # ph2
                    )

            state_hat_0 = problem.get_initial_state()

            def _step(sh, _):
                N1 = problem._call_nonlinear_op(sh, params)
                sh_star = {
                    name: _apply(coeffs[name][1], sh[name])
                           + (dt / 2.0) * _apply(coeffs[name][3], N1[name])
                    for name in state_names
                }
                N2 = problem._call_nonlinear_op(sh_star, params)
                new_sh = {
                    name: _apply(coeffs[name][0], sh[name])
                           + dt * (_apply(coeffs[name][2], N1[name])
                                   + _apply(coeffs[name][4], N2[name] - N1[name]))
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

            return problem._to_physical_batch(seg_states)

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
            problem:   :class:`~pinns.models.ModelSpectralSolver`.
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
            if isinstance(Ln, list):
                coeffs[name] = _compute_kron_coeffs(Ln, dt)
            elif Ln.ndim == 2:
                z   = Ln * dt
                z2  = Ln * (dt / 2.0)
                coeffs[name] = (
                    jax.scipy.linalg.expm(z),   # E
                    jax.scipy.linalg.expm(z2),  # E2  (half-step)
                    _phi1_mat(z),               # ph1
                    _phi1_mat(z2),              # ph12
                    _phi2_mat(z),               # ph2
                )
            else:
                z   = Ln * dt
                z2  = Ln * (dt / 2.0)
                coeffs[name] = (
                    jnp.exp(z),    # E
                    jnp.exp(z2),   # E2  (half-step)
                    _phi1(z),      # ph1
                    _phi1(z2),     # ph12
                    _phi2(z),      # ph2
                )

        N1 = problem._call_nonlinear_op(state_hat, params)
        sh_star = {
            name: (
                _apply(coeffs[name][1], state_hat[name])
                + (dt / 2.0) * _apply(coeffs[name][3], N1[name])
            )
            for name in state_names
        }
        N2 = problem._call_nonlinear_op(sh_star, params)

        sh_new = {
            name: (
                _apply(coeffs[name][0], state_hat[name])
                + dt * (
                    _apply(coeffs[name][2], N1[name])
                    + _apply(coeffs[name][4], N2[name] - N1[name])
                )
            )
            for name in state_names
        }

        # Error = φ₂ correction = difference between ETD2 and ETD1
        err_abs = {
            name: jnp.abs(dt * _apply(coeffs[name][4], N2[name] - N1[name]))
            for name in state_names
        }

        return sh_new, err_abs

    def __repr__(self) -> str:
        return (
            f"IntegratorETD2RK(dt={self.dt}, n_steps={self.n_steps}, "
            f"checkpoint={self.checkpoint})"
        )
