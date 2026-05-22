"""Tests for integrators — accuracy, JAX differentiability, parameter recovery.

Uses the 1-D heat equation as a ground truth:

    u_t = α · u_xx,    u(x,0) = sin(2πx),    periodic BCs on [0,1]

Analytic solution:  u(x,t) = exp(−(2π)² α t) · sin(2πx)
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from pinns import DomainCubic, ModelSpectralSolver
from pinns import IntegratorETD2RK, IntegratorRK4, IntegratorIMEX, AdaptiveIntegrator, PIDController

try:
    import diffrax as _dfx
    from pinns import IntegratorDiffrax
    HAS_DIFFRAX = True
except ImportError:
    HAS_DIFFRAX = False

requires_diffrax = pytest.mark.skipif(not HAS_DIFFRAX, reason="diffrax not installed")


# ============================================================================
# Problem factory — 1D heat equation
# ============================================================================

ALPHA_TRUE = 0.01
N = 32
DT_STIFF   = 1e-2   # safe for ETD2RK and IMEX
DT_EXPLICIT = 1e-3  # needed for RK4 stability (diffusion CFL)


def make_heat_problem(alpha_init=ALPHA_TRUE, N=N, integrator=None):
    """Pure diffusion: u_t = alpha * u_xx, periodic, analytic solution known."""
    if integrator is None:
        integrator = IntegratorETD2RK(dt=DT_STIFF)
    domain = DomainCubic(
        space=[(0.0, 1.0)],
        time=(0.0, 0.1),
    )
    problem = ModelSpectralSolver(domain, state_names=["u"], integrator=integrator, shape=N)

    problem.set_linear_op(
        lambda X, p: {"u": -p["parameter"]["alpha"] * X**2}
    )
    problem.set_source_fn(
        lambda X, U, p: {"u": jnp.zeros_like(U["u"])}
    )

    problem.add_parameter("alpha", alpha_init)

    x = np.array(problem.x)
    u0 = np.sin(2 * np.pi * x)
    problem.add_initial(u0)
    return domain, problem, x


def analytic_heat(x, t, alpha):
    """Analytic solution u(x,t) = exp(-(2π)² α t) sin(2πx)."""
    return np.exp(-(2 * np.pi) ** 2 * alpha * t) * np.sin(2 * np.pi * x)


def add_uniform_obs(problem, domain, n_obs=5, alpha=ALPHA_TRUE):
    """Return reference data aligned with the problem integrator's obs time grid."""
    t_obs = np.array(problem._integrator._get_obs_times(problem))
    x = np.array(problem.x)
    U_obs = np.stack([analytic_heat(x, t, alpha) for t in t_obs], axis=0)
    return t_obs, U_obs


# ============================================================================
# Forward accuracy tests
# ============================================================================

class TestForwardAccuracy:
    """Verify that each integrator reproduces the analytic heat solution."""

    def _run(self, integrator, dt_factor=1.0):
        domain, problem, x = make_heat_problem(integrator=integrator)
        t_obs, U_obs = add_uniform_obs(problem, domain)
        inferred = {"alpha": jnp.array(ALPHA_TRUE)}
        result = integrator.solve(problem, inferred_params=inferred)
        return result["u"], U_obs

    def test_etd2rk_accuracy(self):
        integrator = IntegratorETD2RK(dt=DT_STIFF)
        pred, ref = self._run(integrator)
        np.testing.assert_allclose(np.array(pred), ref, atol=1e-3, rtol=1e-3)

    def test_rk4_accuracy(self):
        integrator = IntegratorRK4(dt=DT_EXPLICIT)
        pred, ref = self._run(integrator)
        np.testing.assert_allclose(np.array(pred), ref, atol=5e-3, rtol=5e-3)

    def test_imex_accuracy(self):
        # IMEX is 1st order; use small dt for accuracy
        integrator = IntegratorIMEX(dt=DT_EXPLICIT)
        pred, ref = self._run(integrator)
        # Looser tolerance than ETD2RK/RK4 due to 1st-order accuracy
        np.testing.assert_allclose(np.array(pred), ref, atol=2e-2, rtol=2e-2)


# ============================================================================
# Output shape
# ============================================================================

class TestOutputShape:
    def _solve(self, integrator):
        domain, problem, x = make_heat_problem(integrator=integrator)
        inferred = {"alpha": jnp.array(ALPHA_TRUE)}
        return integrator.solve(problem, inferred_params=inferred)

    @pytest.mark.parametrize("cls,dt", [
        (IntegratorETD2RK, DT_STIFF),
        (IntegratorRK4,    DT_EXPLICIT),
        (IntegratorIMEX,   DT_STIFF),
    ])
    def test_output_shape(self, cls, dt):
        integrator = cls(dt=dt)
        result = self._solve(integrator)
        expected_nt = int(round(0.1 / dt)) + 1  # domain time=(0,0.1)
        assert "u" in result
        assert result["u"].shape == (expected_nt, N)


# ============================================================================
# JAX differentiability
# ============================================================================

class TestDifferentiability:
    """Verify that jax.grad flows through all integrators."""

    def _loss(self, integrator, alpha_val, domain, problem, U_obs):
        inferred = {"alpha": alpha_val}
        pred = integrator.solve(problem, inferred_params=inferred)
        return jnp.mean((pred["u"] - jnp.array(U_obs)) ** 2)

    @pytest.mark.parametrize("cls,dt", [
        (IntegratorETD2RK, DT_STIFF),
        (IntegratorRK4,    DT_EXPLICIT),
        (IntegratorIMEX,   DT_STIFF),
    ])
    def test_grad_wrt_alpha(self, cls, dt):
        """jax.grad must produce a finite gradient for all integrators."""
        integrator = cls(dt=dt)
        domain, problem, x = make_heat_problem(integrator=integrator)
        _, U_obs = add_uniform_obs(problem, domain)

        alpha = jnp.array(ALPHA_TRUE)
        grad_fn = jax.grad(lambda a: self._loss(integrator, a, domain, problem, U_obs))
        g = grad_fn(alpha)

        assert g.shape == ()
        assert not jnp.isnan(g), f"{cls.__name__}: gradient is NaN"
        assert not jnp.isinf(g), f"{cls.__name__}: gradient is Inf"

    def test_grad_direction_etd2rk(self):
        """Gradient w.r.t. alpha should have correct sign for parameter recovery."""
        integrator = IntegratorETD2RK(dt=DT_STIFF)
        domain, problem, x = make_heat_problem(alpha_init=0.008, integrator=integrator)
        _, U_obs = add_uniform_obs(problem, domain, alpha=ALPHA_TRUE)

        # alpha_init < alpha_true → loss decreases when alpha increases → gradient < 0
        alpha = jnp.array(0.008)
        grad_fn = jax.grad(
            lambda a: jnp.mean((
                integrator.solve(problem, {"alpha": a})["u"] - jnp.array(U_obs)
            ) ** 2)
        )
        g = grad_fn(alpha)
        assert float(g) < 0.0, (
            f"Expected negative gradient (loss should decrease with larger alpha), got {float(g)}"
        )


# ============================================================================
# Gradient descent parameter recovery
# ============================================================================

class TestParameterRecovery:
    """Verify that gradient-based optimisation recovers the true parameter."""

    @pytest.mark.parametrize("cls,dt,lr,n_steps", [
        (IntegratorETD2RK, DT_STIFF,    0.005, 50),
        (IntegratorIMEX,   DT_EXPLICIT, 0.005, 50),
    ])
    def test_recover_alpha(self, cls, dt, lr, n_steps):
        """Run simple gradient descent; alpha should converge toward ALPHA_TRUE."""
        integrator = cls(dt=dt)
        domain, problem, x = make_heat_problem(alpha_init=0.005, integrator=integrator)
        _, U_obs = add_uniform_obs(problem, domain, alpha=ALPHA_TRUE)
        U_obs_jnp = jnp.array(U_obs)

        alpha = jnp.array(0.005)

        @jax.jit
        @jax.grad
        def grad_loss(a):
            pred = integrator.solve(problem, {"alpha": a})
            return jnp.mean((pred["u"] - U_obs_jnp) ** 2)

        for _ in range(n_steps):
            g = grad_loss(alpha)
            alpha = alpha - lr * g

        assert abs(float(alpha) - ALPHA_TRUE) < 1e-2, (
            f"{cls.__name__}: recovered alpha={float(alpha):.6f}, "
            f"expected {ALPHA_TRUE}"
        )


# ============================================================================
# Multi-state problem (2 species, 2D heat)
# ============================================================================

class TestMultiState:
    """Two coupled species, each pure diffusion — no coupling in nonlinear term."""

    def test_two_state_etd2rk(self):
        N = 16
        domain = DomainCubic(
            space=[(0.0, 1.0), (0.0, 1.0)],
            time=(0.0, 0.1),
        )
        integrator = IntegratorETD2RK(dt=1e-3)
        problem = ModelSpectralSolver(domain, state_names=["u", "v"], integrator=integrator, shape=(N, N))

        alpha1, alpha2 = 0.01, 0.02
        problem.set_linear_op(
            lambda X, p: {
                "u": -p["parameter"]["a1"] * (X[0][:, None]**2 + X[1][None, :]**2),
                "v": -p["parameter"]["a2"] * (X[0][:, None]**2 + X[1][None, :]**2),
            }
        )
        problem.set_source_fn(
            lambda X, U, p: {"u": jnp.zeros_like(U["u"]), "v": jnp.zeros_like(U["v"])}
        )
        problem.add_parameter(["a1", "a2"], [alpha1, alpha2])

        XX, YY = np.array(problem.x), np.array(problem.y)
        u0 = np.sin(2 * np.pi * XX) * np.cos(2 * np.pi * YY)
        v0 = np.cos(2 * np.pi * XX) * np.sin(2 * np.pi * YY)
        problem.add_initial(u0, v0)

        integrator = IntegratorETD2RK(dt=1e-3)
        result = integrator.solve(problem, {"a1": jnp.array(alpha1), "a2": jnp.array(alpha2)})

        # domain time=(0,0.1), dt=1e-3 -> 101 snapshots
        assert result["u"].shape == (101, N, N)
        assert result["v"].shape == (101, N, N)
        assert not jnp.any(jnp.isnan(result["u"]))
        assert not jnp.any(jnp.isnan(result["v"]))


# ============================================================================
# Validation errors
# ============================================================================

class TestValidationErrors:
    def test_solve_without_initial(self):
        domain, problem, _ = make_heat_problem()
        problem._initial = None
        with pytest.raises(RuntimeError, match="initial"):
            IntegratorETD2RK(dt=DT_STIFF).solve(problem)

    def test_solve_wrong_problem_type(self):
        from pinns import ProblemStrong, DomainCubic
        d = DomainCubic(space=[(0.0, 1.0)])
        p = ProblemStrong(domain=d, output_names=["u"])
        with pytest.raises(TypeError, match="ModelSpectralSolver"):
            IntegratorETD2RK(dt=DT_STIFF).solve(p)


# ============================================================================
# IntegratorDiffrax tests
# ============================================================================

@requires_diffrax
class TestIntegratorDiffrax:
    """Tests for IntegratorDiffrax using the 1D heat equation (analytic solution known)."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_problem(self, alpha_init=ALPHA_TRUE, N=32, dt0=1e-2):
        """Small heat problem for fast compilation.

        t_obs and U_obs are aligned with the dt0 grid over the full domain.
        """
        domain, problem, x = make_heat_problem(alpha_init=alpha_init, N=N)
        n_steps = int(round(0.1 / dt0))  # domain time=(0.0, 0.1)
        t_obs = np.linspace(0.0, 0.1, n_steps + 1)
        x = np.array(problem.x)
        U_obs = np.stack([analytic_heat(x, t, ALPHA_TRUE) for t in t_obs], axis=0)
        return domain, problem, x, t_obs, U_obs

    # ── forward accuracy ─────────────────────────────────────────────────────

    def test_forward_accuracy_dopri5(self):
        """Dopri5 with PID controller should match analytic solution to 1e-3."""
        domain, problem, x, t_obs, U_obs = self._make_problem()
        integrator = IntegratorDiffrax(
            solver=_dfx.Dopri5(),
            stepsize_controller=_dfx.PIDController(rtol=1e-6, atol=1e-8),
            adjoint="recursive",
            dt0=1e-2,
        )
        result = integrator.solve(problem, {"alpha": jnp.array(ALPHA_TRUE)})
        np.testing.assert_allclose(np.array(result["u"]), U_obs, atol=1e-3, rtol=1e-3)

    def test_forward_accuracy_euler_fixed(self):
        """Euler with constant stepsize should give approximate (lower accuracy) solution."""
        domain, problem, x, t_obs, U_obs = self._make_problem(N=16, dt0=1e-3)
        integrator = IntegratorDiffrax(
            solver=_dfx.Euler(),
            stepsize_controller=_dfx.ConstantStepSize(),
            adjoint="recursive",
            dt0=1e-3,   # fine enough for 1D heat with alpha=0.01, N=16
        )
        result = integrator.solve(problem, {"alpha": jnp.array(ALPHA_TRUE)})
        # Euler 1st order — loose tolerance
        np.testing.assert_allclose(np.array(result["u"]), U_obs, atol=5e-2, rtol=5e-2)

    def test_output_shape(self):
        """Output dict must have key 'u' with shape (n_steps+1, N)."""
        N = 16; dt0 = 1e-2
        domain, problem, x, t_obs, U_obs = self._make_problem(N=N, dt0=dt0)
        integrator = IntegratorDiffrax(
            solver=_dfx.Dopri5(),
            adjoint="direct",
            dt0=dt0,
        )
        result = integrator.solve(problem, {"alpha": jnp.array(ALPHA_TRUE)})
        assert "u" in result
        assert result["u"].shape == (len(t_obs), N)

    def test_no_nans(self):
        """Forward solution must not contain NaN or Inf."""
        domain, problem, x, t_obs, U_obs = self._make_problem()
        integrator = IntegratorDiffrax(
            solver=_dfx.Dopri5(),
            stepsize_controller=_dfx.PIDController(rtol=1e-6, atol=1e-8),
            adjoint="recursive",
            dt0=1e-2,
        )
        result = integrator.solve(problem, {"alpha": jnp.array(ALPHA_TRUE)})
        assert not jnp.any(jnp.isnan(result["u"])), "NaN in diffrax output"
        assert not jnp.any(jnp.isinf(result["u"])), "Inf in diffrax output"

    # ── differentiability ────────────────────────────────────────────────────

    @pytest.mark.parametrize("adjoint", ["recursive", "direct"])
    def test_grad_finite(self, adjoint):
        """jax.grad must produce finite (non-NaN, non-Inf) gradient."""
        domain, problem, x, t_obs, U_obs = self._make_problem(N=16, dt0=1e-2)
        integrator = IntegratorDiffrax(
            solver=_dfx.Dopri5(),
            stepsize_controller=_dfx.PIDController(rtol=1e-5, atol=1e-7),
            adjoint=adjoint,
            dt0=1e-2,
        )
        U_ref = jnp.array(U_obs)

        def loss(alpha):
            pred = integrator.solve(problem, {"alpha": alpha})
            return jnp.mean((pred["u"] - U_ref) ** 2)

        g = jax.grad(loss)(jnp.array(ALPHA_TRUE))
        assert g.shape == (), f"Expected scalar gradient, got {g.shape}"
        assert not jnp.isnan(g), f"adjoint={adjoint}: gradient is NaN"
        assert not jnp.isinf(g), f"adjoint={adjoint}: gradient is Inf"

    def test_grad_direction(self):
        """Gradient sign: alpha_init < alpha_true → gradient must be negative."""
        domain, problem, x, t_obs, U_obs = self._make_problem(alpha_init=0.005, N=16, dt0=1e-2)
        integrator = IntegratorDiffrax(
            solver=_dfx.Dopri5(),
            stepsize_controller=_dfx.PIDController(rtol=1e-5, atol=1e-7),
            adjoint="recursive",
            dt0=1e-2,
        )
        U_ref = jnp.array(U_obs)

        def loss(alpha):
            pred = integrator.solve(problem, {"alpha": alpha})
            return jnp.mean((pred["u"] - U_ref) ** 2)

        g = jax.grad(loss)(jnp.array(0.005))
        assert float(g) < 0.0, (
            f"Expected negative gradient when alpha_init < alpha_true, got {float(g):.6f}"
        )

    def test_grad_backsolve(self):
        """BacksolveAdjoint must also produce a finite gradient (may be slow first call)."""
        domain, problem, x, t_obs, U_obs = self._make_problem(N=16, dt0=1e-2)
        integrator = IntegratorDiffrax(
            solver=_dfx.Dopri5(),
            stepsize_controller=_dfx.PIDController(rtol=1e-5, atol=1e-7),
            adjoint="backsolve",
            dt0=1e-2,
            max_steps=2**16,
        )
        U_ref = jnp.array(U_obs)

        def loss(alpha):
            pred = integrator.solve(problem, {"alpha": alpha})
            return jnp.mean((pred["u"] - U_ref) ** 2)

        g = jax.grad(loss)(jnp.array(ALPHA_TRUE))
        assert not jnp.isnan(g), "backsolve adjoint: gradient is NaN"
        assert not jnp.isinf(g), "backsolve adjoint: gradient is Inf"

    # ── parameter recovery ───────────────────────────────────────────────────

    def test_recover_alpha_recursive(self):
        """Gradient descent with recursive adjoint should recover alpha to within 5e-3."""
        domain, problem, x, t_obs, U_obs = self._make_problem(alpha_init=0.005, N=16, dt0=1e-2)
        integrator = IntegratorDiffrax(
            solver=_dfx.Dopri5(),
            stepsize_controller=_dfx.PIDController(rtol=1e-5, atol=1e-7),
            adjoint="recursive",
            dt0=1e-2,
        )
        U_ref = jnp.array(U_obs)

        @jax.jit
        @jax.grad
        def grad_loss(alpha):
            pred = integrator.solve(problem, {"alpha": alpha})
            return jnp.mean((pred["u"] - U_ref) ** 2)

        alpha = jnp.array(0.005)
        for _ in range(50):
            alpha = alpha - 5e-3 * grad_loss(alpha)

        assert abs(float(alpha) - ALPHA_TRUE) < 1e-2, (
            f"Diffrax recursive: recovered alpha={float(alpha):.6f}, expected {ALPHA_TRUE}"
        )


# ============================================================================
# AdaptiveIntegrator (ETD2RK) tests
# ============================================================================

class TestAdaptiveIntegrator:
    """Tests for AdaptiveIntegrator wrapping IntegratorETD2RK.

    Uses the same 1D heat / Allen-Cahn fixtures defined at module level.
    """

    def _make_problem(self, alpha_init=ALPHA_TRUE, N=N, n_obs=5,
                      max_steps=512):
        integrator = AdaptiveIntegrator(
            IntegratorETD2RK(dt=1e-2),
            PIDController(rtol=1e-4, atol=1e-6),
            dt0=1e-2, max_steps=max_steps,
        )
        domain, problem, x = make_heat_problem(
            alpha_init=alpha_init, N=N, integrator=integrator,
        )
        # add_uniform_obs uses problem._integrator._get_obs_times (dt0=1e-2 -> 101 points)
        t_obs, U_obs = add_uniform_obs(problem, domain, n_obs=n_obs,
                                       alpha=alpha_init)
        return domain, problem, x, t_obs, U_obs

    # ── forward accuracy ─────────────────────────────────────────────────────

    def test_forward_accuracy(self):
        """Adaptive ETD2RK should match analytic heat solution within 1e-3."""
        domain, problem, x, t_obs, U_obs = self._make_problem(n_obs=3)
        integrator = AdaptiveIntegrator(
            IntegratorETD2RK(dt=1e-2), PIDController(rtol=1e-5, atol=1e-7),
            dt0=1e-2, max_steps=512,
        )
        result = integrator.solve(problem)
        err = float(jnp.max(jnp.abs(result["u"] - jnp.array(U_obs))))
        assert err < 1e-3, f"adaptive ETD2RK forward error {err:.2e} > 1e-3"

    def test_output_shape(self):
        """Output shape should be (n_obs, N)."""
        n_obs = 4
        domain, problem, x, t_obs, U_obs = self._make_problem(n_obs=n_obs)
        integrator = AdaptiveIntegrator(
            IntegratorETD2RK(dt=1e-2), dt0=1e-2, max_steps=256,
        )
        result = integrator.solve(problem)
        expected_nt = int(round(0.1 / 1e-2)) + 1  # dt0=1e-2, domain=(0,0.1) -> 11
        assert result["u"].shape == (expected_nt, N), (
            f"expected ({expected_nt},{N}), got {result['u'].shape}"
        )

    def test_no_nans(self):
        """No NaN values should appear in the output."""
        domain, problem, x, t_obs, U_obs = self._make_problem(n_obs=5)
        integrator = AdaptiveIntegrator(
            IntegratorETD2RK(dt=1e-2), PIDController(rtol=1e-4, atol=1e-6),
            dt0=1e-2, max_steps=512,
        )
        result = integrator.solve(problem)
        assert not jnp.any(jnp.isnan(result["u"])), "NaN in adaptive ETD2RK output"

    # ── differentiability ────────────────────────────────────────────────────

    def test_grad_finite(self):
        """jax.grad through adaptive ETD2RK should return a finite gradient."""
        domain, problem, x, t_obs, U_obs = self._make_problem(n_obs=3, N=16)
        integrator = AdaptiveIntegrator(
            IntegratorETD2RK(dt=1e-2), PIDController(rtol=1e-4, atol=1e-6),
            dt0=1e-2, max_steps=256,
        )
        U_ref = jnp.array(U_obs)

        def loss(alpha):
            pred = integrator.solve(problem, {"alpha": alpha})
            return jnp.mean((pred["u"] - U_ref) ** 2)

        g = jax.grad(loss)(jnp.array(ALPHA_TRUE))
        assert jnp.isfinite(g), f"adaptive ETD2RK gradient not finite: {g}"

    def test_grad_direction(self):
        """Gradient should point in the right direction (decreasing loss)."""
        domain, problem, x, t_obs, U_obs = self._make_problem(n_obs=3, N=16)
        integrator = AdaptiveIntegrator(
            IntegratorETD2RK(dt=1e-2), PIDController(rtol=1e-4, atol=1e-6),
            dt0=1e-2, max_steps=256,
        )
        U_ref = jnp.array(U_obs)

        def loss(alpha):
            pred = integrator.solve(problem, {"alpha": alpha})
            return jnp.mean((pred["u"] - U_ref) ** 2)

        alpha0 = jnp.array(0.005)   # wrong value
        l0 = loss(alpha0)
        g  = jax.grad(loss)(alpha0)
        l1 = loss(alpha0 - 1e-3 * g)
        assert l1 < l0, "gradient step did not decrease loss"

    # ── parameter recovery ───────────────────────────────────────────────────

    def test_recover_alpha(self):
        """Gradient descent should recover alpha from wrong initial guess."""
        domain, problem, x, t_obs, U_obs = self._make_problem(
            alpha_init=0.005, N=16, n_obs=4, max_steps=256,
        )
        integrator = AdaptiveIntegrator(
            IntegratorETD2RK(dt=1e-2), PIDController(rtol=1e-4, atol=1e-6),
            dt0=1e-2, max_steps=256,
        )
        U_ref = jnp.array(U_obs)

        @jax.jit
        @jax.grad
        def grad_loss(alpha):
            pred = integrator.solve(problem, {"alpha": alpha})
            return jnp.mean((pred["u"] - U_ref) ** 2)

        alpha = jnp.array(0.005)
        for _ in range(50):
            alpha = alpha - 5e-3 * grad_loss(alpha)

        assert abs(float(alpha) - ALPHA_TRUE) < 1e-2, (
            f"AdaptiveIntegrator(ETD2RK): recovered alpha={float(alpha):.6f}, "
            f"expected {ALPHA_TRUE}"
        )
