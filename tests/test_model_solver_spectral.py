"""Tests for ModelSolver spectral infrastructure — grid, transforms, K2."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import pinns
from pinns import DomainCubic, ModelSolver, IntegratorETD2RK


# ============================================================================
# Helper
# ============================================================================

def _make(space, shape, bc="periodic", time=(0.0, 1.0)):
    """Create a ModelSolver with minimal operators (for spectral tests)."""
    domain = DomainCubic(space=space, time=time)
    integrator = IntegratorETD2RK(dt=1e-3)
    model = ModelSolver(domain, ["u"], integrator, shape=shape, bc=bc)
    return model


# ============================================================================
# Construction
# ============================================================================

class TestConstruction:
    def test_1d_periodic(self):
        m = _make([(-1.0, 1.0)], 16)
        assert m.n_dims == 1
        assert m.shape == (16,)
        assert m.bc == "periodic"
        assert m._t_min == 0.0
        assert m._t_max == 1.0
        assert m.x.shape == (16,)

    def test_2d_periodic(self):
        m = _make([(-1.0, 1.0), (-1.0, 1.0)], (32, 32))
        assert m.n_dims == 2
        assert m.shape == (32, 32)
        assert m.x.shape == (32, 32)
        assert m.y.shape == (32, 32)
        assert m.K2.shape == (32, 32)

    def test_2d_dirichlet(self):
        m = _make([(0.0, 1.0), (0.0, 1.0)], (16, 16), bc="dirichlet")
        assert m.bc == "dirichlet"
        assert m.K2.shape == (16, 16)

    def test_2d_neumann(self):
        m = _make([(0.0, 1.0), (0.0, 1.0)], (16, 16), bc="neumann")
        assert m.bc == "neumann"
        assert m.K2.shape == (16, 16)

    def test_invalid_bc(self):
        with pytest.raises(ValueError, match="not supported"):
            _make([(0.0, 1.0)], 8, bc="Robin")

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="len\\(shape\\)"):
            _make([(0.0, 1.0)], (8, 8))

    def test_broadcast_scalar_shape(self):
        m = _make([(0.0, 1.0), (0.0, 2.0)], 8)
        assert m.shape == (8, 8)

    def test_time_tuple(self):
        m = _make([(0.0, 1.0)], 8, time=(0.5, 3.0))
        assert m._t_min == 0.5
        assert m._t_max == 3.0

    def test_repr(self):
        m = _make([(0.0, 1.0)], 8)
        assert "ModelSolver" in repr(m)
        assert "periodic" in repr(m)

    def test_wrong_domain_type(self):
        integrator = IntegratorETD2RK(dt=1e-3)
        with pytest.raises(TypeError, match="DomainCubic"):
            ModelSolver(object(), ["u"], integrator, shape=8)


# ============================================================================
# K2 eigenvalues
# ============================================================================

class TestK2:
    def test_periodic_1d_k2(self):
        N, L = 8, 2.0
        m = _make([(0.0, L)], N, bc="periodic")
        expected = (np.fft.fftfreq(N, d=L / N) * 2 * np.pi) ** 2
        np.testing.assert_allclose(np.array(m.K2), expected, atol=1e-10)

    def test_periodic_2d_k2_shape(self):
        m = _make([(0.0, 1.0), (0.0, 1.0)], (8, 8), bc="periodic")
        assert m.K2.shape == (8, 8)
        assert float(jnp.min(m.K2)) >= -1e-12

    def test_dirichlet_1d_k2_positive(self):
        m = _make([(0.0, 1.0)], 8, bc="dirichlet")
        assert float(jnp.min(m.K2)) > 0.0

    def test_neumann_1d_k2_non_negative(self):
        m = _make([(0.0, 1.0)], 8, bc="neumann")
        k2 = np.array(m.K2)
        assert k2[0] == pytest.approx(0.0, abs=1e-12)
        assert np.all(k2[1:] > 0.0)

    def test_periodic_2d_k2_symmetry(self):
        N = 16
        m = _make([(0.0, 1.0), (0.0, 1.0)], N, bc="periodic")
        kx = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        np.testing.assert_allclose(np.array(m.K2), KX**2 + KY**2, atol=1e-10)


# ============================================================================
# Forward / Inverse transforms: round-trip accuracy
# ============================================================================

class TestTransformsRoundTrip:
    def _smooth(self, m):
        if m.n_dims == 1:
            return jnp.sin(2 * np.pi * np.array(m.x) / m._lengths[0])
        x, y = np.array(m.x), np.array(m.y)
        return (jnp.sin(2 * np.pi * x / m._lengths[0])
                * jnp.cos(2 * np.pi * y / m._lengths[1]))

    def test_periodic_1d(self):
        m = _make([(0.0, 2.0)], 16, bc="periodic")
        u = self._smooth(m)
        np.testing.assert_allclose(np.array(m.inverse(m.forward(u))),
                                   np.array(u), atol=1e-5)

    def test_periodic_2d(self):
        m = _make([(0.0, 2.0), (0.0, 2.0)], (16, 16), bc="periodic")
        u = self._smooth(m)
        np.testing.assert_allclose(np.array(m.inverse(m.forward(u))),
                                   np.array(u), atol=1e-5)

    def test_dirichlet_1d(self):
        m = _make([(0.0, 1.0)], 8, bc="dirichlet")
        u = jnp.array(np.sin(np.pi * np.array(m.x)))
        np.testing.assert_allclose(np.array(m.inverse(m.forward(u))),
                                   np.array(u), atol=1e-5)

    def test_neumann_1d(self):
        m = _make([(0.0, 1.0)], 8, bc="neumann")
        u = jnp.array(np.cos(np.pi * np.array(m.x)))
        np.testing.assert_allclose(np.array(m.inverse(m.forward(u))),
                                   np.array(u), atol=1e-5)

    def test_neumann_2d(self):
        m = _make([(0.0, 1.0), (0.0, 1.0)], (8, 8), bc="neumann")
        u = self._smooth(m)
        np.testing.assert_allclose(np.array(m.inverse(m.forward(u))),
                                   np.array(u), atol=1e-5)


# ============================================================================
# Spectral derivative accuracy (periodic)
# ============================================================================

class TestSpectralDerivative:
    def test_minus_laplacian_1d_periodic(self):
        N, L = 32, 2.0
        m = _make([(0.0, L)], N, bc="periodic")
        k = 1
        u = jnp.sin(2 * np.pi * k * np.array(m.x) / L)
        lap_u = m.inverse(m.K2 * m.forward(u))
        expected = (2 * np.pi * k / L) ** 2 * u
        np.testing.assert_allclose(np.array(lap_u.real), np.array(expected), atol=1e-3)

    def test_minus_laplacian_2d_periodic(self):
        N = 32
        m = _make([(0.0, 1.0), (0.0, 1.0)], N, bc="periodic")
        u = jnp.sin(2 * np.pi * m.x) * jnp.cos(2 * np.pi * m.y)
        lap_u = m.inverse(m.K2 * m.forward(u))
        expected = (4 * np.pi**2 + 4 * np.pi**2) * u
        np.testing.assert_allclose(np.array(lap_u.real), np.array(expected), atol=2e-3)


# ============================================================================
# JAX differentiability
# ============================================================================

class TestJaxDifferentiability:
    def test_grad_through_forward(self):
        m = _make([(0.0, 1.0)], 8, bc="periodic")
        u0 = jnp.array(np.sin(2 * np.pi * np.array(m.x)))

        def loss(u):
            return jnp.sum(jnp.abs(m.forward(u)) ** 2).real

        g = jax.grad(loss)(u0)
        assert g.shape == u0.shape
        assert not jnp.any(jnp.isnan(g))

    def test_grad_through_inverse(self):
        m = _make([(0.0, 1.0)], 8, bc="periodic")
        u0 = jnp.array(np.sin(2 * np.pi * np.array(m.x)))
        u_hat0 = m.forward(u0)

        def loss(u_hat):
            return jnp.sum(m.inverse(u_hat) ** 2).real

        g = jax.grad(loss)(u_hat0)
        assert g.shape == u_hat0.shape
        assert not jnp.any(jnp.isnan(g))
