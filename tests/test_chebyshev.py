"""Tests for Chebyshev pseudospectral support in ModelSpectralSolver."""

import numpy as np
import jax.numpy as jnp
import pytest

import pinns
from pinns.models.model_solver import _cheb_diff_matrix


# ─────────────────────────────────────────────────────────────────────────── #
#  _cheb_diff_matrix unit tests                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestChebDiffMatrix:
    """Tests for the Chebyshev differentiation matrix helper."""

    def test_shape(self):
        D = _cheb_diff_matrix(8, 2.0)
        assert D.shape == (8, 8)

    def test_derivative_of_x(self):
        """D * x should give 1 at interior nodes (derivative of x is 1)."""
        N = 16
        L = 2.0
        D = _cheb_diff_matrix(N, L)
        j = np.arange(N)
        x = -np.cos(j * np.pi / (N - 1))  # Gauss-Lobatto on [-1, 1]
        Dx = D @ x
        # Interior nodes should be close to 1/(L/2) * 1 = 1 (D is scaled to domain length)
        # Actually D already accounts for 2/L, and x is on [-1,1], so D@x = 1/(L/2) = 1 when L=2
        np.testing.assert_allclose(Dx, np.ones(N), atol=1e-10)

    def test_derivative_of_x_squared(self):
        """D * x² should give 2x at all nodes."""
        N = 16
        L = 2.0
        D = _cheb_diff_matrix(N, L)
        j = np.arange(N)
        x = -np.cos(j * np.pi / (N - 1))  # on [-1, 1]
        Dx2 = D @ (x ** 2)
        # d/dx(x²) = 2x on [-1,1] (L=2, so no extra scaling needed since D has 2/L * L/2 = 1)
        np.testing.assert_allclose(Dx2, 2 * x, atol=1e-10)

    def test_negative_sum_diagonal(self):
        """Row sums of D should be zero (D * constants = 0)."""
        D = _cheb_diff_matrix(12, 3.0)
        row_sums = D.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.zeros(12), atol=1e-12)

    def test_derivative_of_sin_scaled(self):
        """D applied to sin(pi*X) on [-1,1] should give pi*cos(pi*X)."""
        N = 32
        L = 2.0
        D = _cheb_diff_matrix(N, L)
        j = np.arange(N)
        x = -np.cos(j * np.pi / (N - 1))  # on [-1, 1]
        f = np.sin(np.pi * x)
        Df = D @ f
        expected = np.pi * np.cos(np.pi * x)
        np.testing.assert_allclose(Df, expected, atol=1e-10)

    def test_physical_domain_scaling(self):
        """D on [0, π] applied to sin(x) should give cos(x)."""
        N = 32
        xmin, xmax = 0.0, np.pi
        L = xmax - xmin
        D = _cheb_diff_matrix(N, L)
        j = np.arange(N)
        x_ref = -np.cos(j * np.pi / (N - 1))            # on [-1, 1]
        x = xmin + (xmax - xmin) * (x_ref + 1.0) / 2.0  # on [0, π]
        f = np.sin(x)
        Df = D @ f
        np.testing.assert_allclose(Df, np.cos(x), atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────── #
#  ModelSpectralSolver with bc="chebyshev"                                    #
# ─────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def cheb_model():
    """A minimal Chebyshev ModelSpectralSolver for 1-D testing."""
    N = 16
    domain = pinns.DomainCubic(space=[(-1.0, 1.0)], time=(0.0, 0.1))
    integrator = pinns.IntegratorETD2RK(dt=1e-4)
    model = pinns.ModelSpectralSolver(domain, ["u"], integrator, shape=N, bc="chebyshev")
    return model, N


class TestChebModel:
    """Tests for ModelSpectralSolver initialised with bc='chebyshev'."""

    def test_construction(self, cheb_model):
        model, N = cheb_model
        assert model.bc == "chebyshev"

    def test_grid_nodes_are_gauss_lobatto(self, cheb_model):
        """model.x should be Gauss-Lobatto nodes on [-1, 1]."""
        model, N = cheb_model
        x = np.asarray(model.x)
        j = np.arange(N)
        expected = -np.cos(j * np.pi / (N - 1))
        np.testing.assert_allclose(x, expected, atol=1e-12)

    def test_k_is_diff_matrix(self, cheb_model):
        """model.k should be the (N, N) Chebyshev D matrix."""
        model, N = cheb_model
        assert model.k.shape == (N, N)

    def test_D_alias(self, cheb_model):
        """model.D should be the same object as model.k."""
        model, N = cheb_model
        assert hasattr(model, "D")
        np.testing.assert_array_equal(np.asarray(model.D), np.asarray(model.k))

    def test_K2_is_minus_D_squared(self, cheb_model):
        """model.K2 should equal -D²."""
        model, N = cheb_model
        D = np.asarray(model.k, dtype=np.float64)
        expected = -(D @ D)
        np.testing.assert_allclose(np.asarray(model.K2, dtype=np.float64), expected, rtol=1e-5)

    def test_forward_is_identity(self, cheb_model):
        """forward() should return the input unchanged (collocation basis)."""
        model, N = cheb_model
        u = jnp.ones(N) * 0.5
        u_hat = model.forward(u)
        np.testing.assert_array_equal(np.asarray(u_hat), np.asarray(u))

    def test_inverse_is_identity(self, cheb_model):
        """inverse() should return the input unchanged."""
        model, N = cheb_model
        u_hat = jnp.arange(N, dtype=float)
        u = model.inverse(u_hat)
        np.testing.assert_array_equal(np.asarray(u), np.asarray(u_hat))

    def test_diff_first_order(self, cheb_model):
        """model.diff(x, order=1) should equal D @ x."""
        model, N = cheb_model
        x = model.x  # Gauss-Lobatto nodes as physical coordinate
        Dx = model.diff(x, order=1)
        D = np.asarray(model.k)
        # Compare in float64 to avoid float32 precision issues
        expected = D @ np.asarray(x, dtype=np.float64)
        np.testing.assert_allclose(np.asarray(Dx, dtype=np.float64), expected, atol=1e-5)

    def test_diff_second_order(self, cheb_model):
        """model.diff(x², order=2) should equal D² @ x²."""
        model, N = cheb_model
        x = model.x
        x2 = x ** 2
        D2x2 = model.diff(x2, order=2)
        D = np.asarray(model.k, dtype=np.float64)
        x2_np = np.asarray(x2, dtype=np.float64)
        expected = D @ (D @ x2_np)
        np.testing.assert_allclose(np.asarray(D2x2, dtype=np.float64), expected, rtol=1e-3)

    def test_invalid_bc_rejected(self):
        """bc='unsupported' should raise ValueError."""
        domain = pinns.DomainCubic(space=[(-1.0, 1.0)], time=(0.0, 0.1))
        integrator = pinns.IntegratorETD2RK(dt=1e-4)
        with pytest.raises(ValueError):
            pinns.ModelSpectralSolver(domain, ["u"], integrator, shape=8, bc="unsupported")


# ─────────────────────────────────────────────────────────────────────────── #
#  Integration test: simple 1-D heat equation u_t = ν u_xx                  #
#  with homogeneous Dirichlet BCs and known analytical solution               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestChebHeatEquation:
    """Solve u_t = ν u_xx via Chebyshev ETD2RK and compare to analytic solution.

    The PDE u_t = ν u_xx on [-1, 1] with u(±1, t) = 0 and u(x, 0) = sin(πx)
    has the exact solution u(x, t) = exp(-ν π² t) sin(πx).

    Here we enforce the zero Dirichlet BCs by zeroing the boundary rows of the
    linear operator matrix (tau method: the boundary nodes are never updated).
    """

    def _build_model(self, nu, N, dt, t_end):
        domain = pinns.DomainCubic(space=[(-1.0, 1.0)], time=(0.0, t_end))
        integrator = pinns.IntegratorETD2RK(dt=dt)
        model = pinns.ModelSpectralSolver(
            domain, ["u"], integrator, shape=N, bc="chebyshev"
        )

        # Build the linear operator: L = ν D² with boundary rows zeroed.
        def linear_op(X, p):
            D = X  # X is the Chebyshev D matrix
            L = p["parameter"]["nu"] * (D @ D)
            # Tau method: zero BC rows so boundary values are never modified
            L = L.at[0, :].set(0.0)
            L = L.at[-1, :].set(0.0)
            return {"u": L}

        # Zero nonlinear term
        def nonlinear_op(X, U, p):
            return {"u": jnp.zeros_like(U["u"])}

        model.set_linear_op(linear_op)
        model.set_source_fn(nonlinear_op)
        model.add_parameter("nu", nu)

        x = np.asarray(model.x)
        u0 = np.sin(np.pi * x)
        u0[0] = 0.0
        u0[-1] = 0.0
        model.add_initial(u0)
        return model, np.asarray(model.x)

    def test_heat_equation_accuracy(self):
        """ETD2RK Chebyshev should match analytic heat solution to <1% rel. error."""
        import jax
        nu = 0.1
        N = 24
        dt = 5e-4
        t_end = 0.5

        # Force CPU to avoid cuBLAS GPU-memory issues with small matrix operations
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            model, x = self._build_model(nu, N, dt, t_end)
            result = model.solve()

        u_num = np.asarray(result["u"])[-1]           # last snapshot = t_end
        u_exact = np.exp(-nu * np.pi ** 2 * t_end) * np.sin(np.pi * x)

        # Compare interior values only (boundary is enforced to 0 by tau rows)
        interior = slice(1, N - 1)
        rel_err = np.max(np.abs(u_num[interior] - u_exact[interior])) / np.max(np.abs(u_exact[interior]))
        assert rel_err < 0.01, f"Relative error {rel_err:.4f} >= 1%"
