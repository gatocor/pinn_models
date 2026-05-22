"""Unit tests for pinns.functional.make_derivative_fn / derivative.

Each test uses a simple polynomial or trigonometric function whose exact
derivatives are known, and verifies that ``make_derivative_fn`` returns the
correct values for every axis combination.

Column layout (matches DomainCubic convention):
  1-D spatial only     → [x]       cols: x=0
  1-D spatial + time   → [x, t]    cols: x=0, t=1
  2-D spatial only     → [x, y]    cols: x=0, y=1
  2-D spatial + time   → [x, y, t] cols: x=0, y=1, t=2
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from pinns.functional import make_derivative_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(fn):
    """Wrap a JAX function as a model: apply(params, X) -> (N, 1)."""
    class _M:
        def apply(self, params, x):
            return fn(x).reshape(-1, 1)
    m = _M()
    return lambda params, x: m.apply(x, params), {}


def _pts(*cols):
    """Build a float32 point array from column vectors (as 1-D arrays)."""
    return jnp.stack([jnp.asarray(c, dtype=jnp.float32) for c in cols], axis=1)


RNG = np.random.default_rng(0)
N = 40   # batch size for all tests


# ---------------------------------------------------------------------------
# 1-D spatial only  (cols: x=0)
# ---------------------------------------------------------------------------

class TestDerivative1DSpatial:
    """Domain has a single spatial column x (col 0). No time axis."""

    def setup_method(self):
        self.xs = RNG.uniform(-1, 1, N).astype(np.float32)
        self.X  = _pts(self.xs)

    def test_first_deriv_x(self):
        """∂(x²)/∂x = 2x."""
        apply_fn, params = _model(lambda X: X[:, 0] ** 2)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0,)).ravel()
        np.testing.assert_allclose(got, 2 * self.xs, atol=1e-5)

    def test_second_deriv_x(self):
        """∂²(x³)/∂x² = 6x."""
        apply_fn, params = _model(lambda X: X[:, 0] ** 3)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0, 0)).ravel()
        np.testing.assert_allclose(got, 6 * self.xs, atol=1e-4)

    def test_third_deriv_x(self):
        """∂³(x⁴)/∂x³ = 24x."""
        apply_fn, params = _model(lambda X: X[:, 0] ** 4)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0, 0, 0)).ravel()
        np.testing.assert_allclose(got, 24 * self.xs, atol=1e-3)

    def test_sin_deriv_x(self):
        """∂sin(πx)/∂x = π cos(πx)."""
        apply_fn, params = _model(lambda X: jnp.sin(jnp.pi * X[:, 0]))
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0,)).ravel()
        expected = jnp.pi * jnp.cos(jnp.pi * self.xs)
        np.testing.assert_allclose(got, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 1-D spatial + time  (cols: x=0, t=1)
# ---------------------------------------------------------------------------

class TestDerivative1DSpatialTime:
    """Domain: x (col 0), t (col 1)."""

    def setup_method(self):
        self.xs = RNG.uniform(-1, 1, N).astype(np.float32)
        self.ts = RNG.uniform(0, 1, N).astype(np.float32)
        self.X  = _pts(self.xs, self.ts)

    # -- x derivatives -------------------------------------------------------

    def test_du_dx(self):
        """∂(x²·t)/∂x = 2x·t  — must NOT vary with t column."""
        apply_fn, params = _model(lambda X: X[:, 0] ** 2 * X[:, 1])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0,)).ravel()
        expected = 2 * self.xs * self.ts
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_d2u_dx2(self):
        """∂²(x³·t)/∂x² = 6x·t."""
        apply_fn, params = _model(lambda X: X[:, 0] ** 3 * X[:, 1])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0, 0)).ravel()
        expected = 6 * self.xs * self.ts
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_du_dx_independent_of_t(self):
        """∂(x²)/∂x = 2x — function does NOT depend on t; result is still 2x."""
        apply_fn, params = _model(lambda X: X[:, 0] ** 2)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0,)).ravel()
        np.testing.assert_allclose(got, 2 * self.xs, atol=1e-5)

    # -- t derivatives -------------------------------------------------------

    def test_du_dt(self):
        """∂(x·t²)/∂t = 2x·t  — must NOT vary with x column."""
        apply_fn, params = _model(lambda X: X[:, 0] * X[:, 1] ** 2)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (1,)).ravel()
        expected = 2 * self.xs * self.ts
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_d2u_dt2(self):
        """∂²(t³)/∂t² = 6t  — x column irrelevant."""
        apply_fn, params = _model(lambda X: X[:, 1] ** 3)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (1, 1)).ravel()
        expected = 6 * self.ts
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_du_dt_independent_of_x(self):
        """∂(t²)/∂t = 2t — function does NOT depend on x; result is 2t."""
        apply_fn, params = _model(lambda X: X[:, 1] ** 2)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (1,)).ravel()
        np.testing.assert_allclose(got, 2 * self.ts, atol=1e-5)

    # -- cross (mixed) partial -----------------------------------------------

    def test_mixed_partial_x_then_t(self):
        """∂²(x·t)/∂x∂t = 1 everywhere."""
        apply_fn, params = _model(lambda X: X[:, 0] * X[:, 1])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0, 1)).ravel()
        np.testing.assert_allclose(got, np.ones(N, dtype=np.float32), atol=1e-5)

    def test_mixed_partial_t_then_x(self):
        """∂²(x·t)/∂t∂x = 1 everywhere (same by Clairaut's theorem)."""
        apply_fn, params = _model(lambda X: X[:, 0] * X[:, 1])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (1, 0)).ravel()
        np.testing.assert_allclose(got, np.ones(N, dtype=np.float32), atol=1e-5)

    def test_x_deriv_zero_for_t_only_fn(self):
        """∂(t²)/∂x = 0 — t-only function has no x-dependence."""
        apply_fn, params = _model(lambda X: X[:, 1] ** 2)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0,)).ravel()
        np.testing.assert_allclose(got, 0.0, atol=1e-6)

    def test_t_deriv_zero_for_x_only_fn(self):
        """∂(x²)/∂t = 0 — x-only function has no t-dependence."""
        apply_fn, params = _model(lambda X: X[:, 0] ** 2)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (1,)).ravel()
        np.testing.assert_allclose(got, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 2-D spatial  (cols: x=0, y=1)
# ---------------------------------------------------------------------------

class TestDerivative2DSpatial:
    """Domain: x (col 0), y (col 1). No time axis."""

    def setup_method(self):
        self.xs = RNG.uniform(-1, 1, N).astype(np.float32)
        self.ys = RNG.uniform(-1, 1, N).astype(np.float32)
        self.X  = _pts(self.xs, self.ys)

    def test_du_dx(self):
        """∂(x²+y³)/∂x = 2x — y column irrelevant."""
        apply_fn, params = _model(lambda X: X[:, 0] ** 2 + X[:, 1] ** 3)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0,)).ravel()
        np.testing.assert_allclose(got, 2 * self.xs, atol=1e-5)

    def test_du_dy(self):
        """∂(x²+y³)/∂y = 3y²."""
        apply_fn, params = _model(lambda X: X[:, 0] ** 2 + X[:, 1] ** 3)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (1,)).ravel()
        expected = 3 * self.ys ** 2
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_d2u_dy2(self):
        """∂²(y³)/∂y² = 6y."""
        apply_fn, params = _model(lambda X: X[:, 1] ** 3)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (1, 1)).ravel()
        np.testing.assert_allclose(got, 6 * self.ys, atol=1e-4)

    def test_du_dx_zero_for_y_only(self):
        """∂(y²)/∂x = 0."""
        apply_fn, params = _model(lambda X: X[:, 1] ** 2)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0,)).ravel()
        np.testing.assert_allclose(got, 0.0, atol=1e-6)

    def test_du_dy_zero_for_x_only(self):
        """∂(x²)/∂y = 0."""
        apply_fn, params = _model(lambda X: X[:, 0] ** 2)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (1,)).ravel()
        np.testing.assert_allclose(got, 0.0, atol=1e-6)

    def test_mixed_xy(self):
        """∂²(x·y)/∂x∂y = 1."""
        apply_fn, params = _model(lambda X: X[:, 0] * X[:, 1])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0, 1)).ravel()
        np.testing.assert_allclose(got, np.ones(N, dtype=np.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# 2-D spatial + time  (cols: x=0, y=1, t=2)
# ---------------------------------------------------------------------------

class TestDerivative2DSpatialTime:
    """Domain: x (col 0), y (col 1), t (col 2)."""

    def setup_method(self):
        self.xs = RNG.uniform(-1, 1, N).astype(np.float32)
        self.ys = RNG.uniform(-1, 1, N).astype(np.float32)
        self.ts = RNG.uniform(0,  1, N).astype(np.float32)
        self.X  = _pts(self.xs, self.ys, self.ts)

    def test_du_dx(self):
        """∂(x²·y·t)/∂x = 2x·y·t — y and t columns don't bleed in."""
        apply_fn, params = _model(lambda X: X[:, 0]**2 * X[:, 1] * X[:, 2])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0,)).ravel()
        expected = 2 * self.xs * self.ys * self.ts
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_du_dy(self):
        """∂(x·y²·t)/∂y = 2x·y·t — x and t columns don't bleed in."""
        apply_fn, params = _model(lambda X: X[:, 0] * X[:, 1]**2 * X[:, 2])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (1,)).ravel()
        expected = 2 * self.xs * self.ys * self.ts
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_du_dt(self):
        """∂(x·y·t²)/∂t = 2x·y·t — x and y columns don't bleed in."""
        apply_fn, params = _model(lambda X: X[:, 0] * X[:, 1] * X[:, 2]**2)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (2,)).ravel()
        expected = 2 * self.xs * self.ys * self.ts
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_d2u_dx2(self):
        """∂²(x³·t)/∂x² = 6x·t."""
        apply_fn, params = _model(lambda X: X[:, 0]**3 * X[:, 2])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0, 0)).ravel()
        expected = 6 * self.xs * self.ts
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_d2u_dt2(self):
        """∂²(y·t³)/∂t² = 6y·t."""
        apply_fn, params = _model(lambda X: X[:, 1] * X[:, 2]**3)
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (2, 2)).ravel()
        expected = 6 * self.ys * self.ts
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_x_deriv_zero_for_yt_fn(self):
        """∂(y·t)/∂x = 0 — function has no x-dependence."""
        apply_fn, params = _model(lambda X: X[:, 1] * X[:, 2])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0,)).ravel()
        np.testing.assert_allclose(got, 0.0, atol=1e-6)

    def test_t_deriv_zero_for_xy_fn(self):
        """∂(x·y)/∂t = 0 — function has no t-dependence."""
        apply_fn, params = _model(lambda X: X[:, 0] * X[:, 1])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (2,)).ravel()
        np.testing.assert_allclose(got, 0.0, atol=1e-6)

    def test_mixed_xt(self):
        """∂²(x·t)/∂x∂t = 1 everywhere."""
        apply_fn, params = _model(lambda X: X[:, 0] * X[:, 2])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (0, 2)).ravel()
        np.testing.assert_allclose(got, np.ones(N, dtype=np.float32), atol=1e-5)

    def test_mixed_yt(self):
        """∂²(y·t)/∂y∂t = 1 everywhere."""
        apply_fn, params = _model(lambda X: X[:, 1] * X[:, 2])
        dfn = make_derivative_fn(apply_fn, params)
        Y = apply_fn(params, self.X)
        got = dfn(Y, self.X, 0, (1, 2)).ravel()
        np.testing.assert_allclose(got, np.ones(N, dtype=np.float32), atol=1e-5)
