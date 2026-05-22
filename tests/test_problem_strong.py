"""Test suite for ProblemStrong."""
import numpy as np
import pytest
import jax.numpy as jnp

from pinns.domain import DomainCubic
from pinns.problems.problem_strong import ProblemStrong



# ---------------------------------------------------------------------------
# Helpers — minimal domains
# ---------------------------------------------------------------------------

def _domain_1d():
    """1-D spatial domain [0, 1]."""
    return DomainCubic(space=[(0.0, 1.0)])


def _domain_2d():
    """2-D spatial domain [0,1]²."""
    return DomainCubic(space=[(0.0, 1.0), (0.0, 1.0)])


def _domain_1d_time():
    """1-D space + time domain, x∈[0,1], t∈[0,1] (plain bounds)."""
    return DomainCubic(space=[(0.0, 1.0)], time=(0.0, 1.0))


def _domain_1d_step():
    """1-D+time domain with a partitioned time axis — required by StepperStep."""
    return DomainCubic(space=[(0.0, 1.0)], time=np.linspace(0.0, 1.0, 6))




# Trivial residual callable
def _res(x, u, pars, diff):
    return u[:, 0]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        assert p.n_outputs == 1
        assert p.output_names == ['u']

    def test_multi_output(self):
        p = ProblemStrong(_domain_2d(), ['u', 'v', 'p'])
        assert p.n_outputs == 3
        assert p.output_names == ['u', 'v', 'p']

    def test_bad_domain_raises(self):
        with pytest.raises(TypeError, match="domain must be"):
            ProblemStrong("not_a_domain", ['u'])

    def test_empty_outputs_raises(self):
        with pytest.raises(ValueError):
            ProblemStrong(_domain_1d(), [])


# ---------------------------------------------------------------------------
# input_names auto-derivation
# ---------------------------------------------------------------------------

class TestInputNames:
    def test_1d_no_time(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        assert p.input_names == ['x']

    def test_2d_no_time(self):
        p = ProblemStrong(_domain_2d(), ['u'])
        assert p.input_names == ['x', 'y']

    def test_1d_with_time(self):
        p = ProblemStrong(_domain_1d_time(), ['u'])
        assert p.input_names == ['x', 't']

    def test_n_dims(self):
        p = ProblemStrong(_domain_2d(), ['u'])
        assert p.n_dims == 2


# ---------------------------------------------------------------------------
# add_inner
# ---------------------------------------------------------------------------

class TestAddInner:
    def test_single_term(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_inner(_res, name='pde')
        assert len(p.inner_terms) == 1
        assert p.inner_terms[0].name == 'pde'

    def test_multi_eq_names(self):
        p = ProblemStrong(_domain_1d(), ['u', 'v'])
        p.add_inner(_res, name=['pde_u', 'pde_v'])
        assert len(p.inner_terms) == 2
        names = [t.name for t in p.inner_terms]
        assert names == ['pde_u', 'pde_v']

    def test_multi_eq_name_list_too_short_raises(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        with pytest.raises(ValueError, match="at least 2 entries"):
            p.add_inner(_res, name=['only_one'])

    def test_chaining(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        result = p.add_inner(_res, name='pde')
        assert result is p


# ---------------------------------------------------------------------------
# add_boundary / add_dirichlet / add_neumann / add_robin
# ---------------------------------------------------------------------------

class TestBoundaryTerms:
    def test_add_boundary(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_boundary(_res, name='bc')
        assert len(p.boundary_terms) == 1

    def test_add_dirichlet_scalar(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_dirichlet(0.0, name='left_bc')
        assert len(p.boundary_terms) == 1
        assert p.boundary_terms[0].kind == 'dirichlet'
        assert p.boundary_terms[0].value == 0.0

    def test_add_dirichlet_callable(self):
        g = lambda x, pars: np.zeros(x.shape[0])
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_dirichlet(g, name='left_bc')
        assert callable(p.boundary_terms[0].value)

    def test_add_dirichlet_multi_output(self):
        p = ProblemStrong(_domain_2d(), ['u', 'v'])
        p.add_dirichlet(0.0, name='zero_bc', outputs=['u', 'v'])
        assert len(p.boundary_terms) == 2

    def test_add_dirichlet_ambiguous_multi_output_raises(self):
        p = ProblemStrong(_domain_2d(), ['u', 'v'])
        with pytest.raises(ValueError, match="must specify"):
            p.add_dirichlet(0.0, name='bc')

    def test_add_neumann(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_neumann(1.0, name='flux_bc')
        t = p.boundary_terms[0]
        assert t.kind == 'neumann'
        assert callable(t.fn)

    def test_add_robin(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_robin(alpha=1.0, beta=1.0, g=0.0, name='robin_bc')
        t = p.boundary_terms[0]
        assert t.kind == 'robin'
        assert callable(t.fn)


# ---------------------------------------------------------------------------
# add_periodic
# ---------------------------------------------------------------------------

class TestAddPeriodic:
    def test_valid(self):
        d = _domain_1d()
        p = ProblemStrong(d, ['u'])
        d.add_periodic('x', name='per')
        p.add_periodic('per', name='per')
        assert len(p.boundary_terms) == 1

    def test_invalid_axis_raises(self):
        d = _domain_1d()
        with pytest.raises(ValueError):
            d.add_periodic('w', name='per')

    def test_unregistered_region_raises(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        with pytest.raises(ValueError, match="not registered"):
            p.add_periodic('missing', name='per')


# ---------------------------------------------------------------------------
# add_initial
# ---------------------------------------------------------------------------

class TestAddInitial:
    def test_requires_time(self):
        p = ProblemStrong(_domain_1d(), ['u'])  # no time axis
        with pytest.raises(ValueError, match="time-dependent"):
            p.add_initial(_res)

    def test_registers_ic(self):
        p = ProblemStrong(_domain_1d_time(), ['u'])
        p.add_initial(_res, name='ic')
        assert len(p.initial_terms) == 1
        assert p.initial_terms[0].kind == 'initial'

    def test_multi_output_ic(self):
        d = DomainCubic(space=[(0.0, 1.0), (0.0, 1.0)], time=(0.0, 1.0))
        p = ProblemStrong(d, ['u', 'v'])
        p.add_initial(_res, name='ic', outputs=['u', 'v'])
        assert len(p.initial_terms) == 2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset:
    def test_register(self):
        from pinns import Dataset
        ds = Dataset()
        x = np.linspace(0, 1, 10)[:, None]
        u = np.zeros(10)
        ds.add_points(x, u, name='obs')
        assert len(ds) == 1
        assert ds.data_terms[0].name == 'obs'

    def test_len_mismatch_raises(self):
        from pinns import Dataset
        ds = Dataset()
        x = np.zeros((5, 1))
        u = np.zeros(3)
        with pytest.raises(ValueError, match="row"):
            ds.add_points(x, u, name='obs')

    def test_multiple_pointsets(self):
        from pinns import Dataset
        ds = Dataset()
        x = np.zeros((5, 1))
        ds.add_points(x, np.zeros(5), name='a')
        ds.add_points(x, np.ones(5),  name='b')
        assert len(ds) == 2
        assert ds.names == ['a', 'b']


# ---------------------------------------------------------------------------
# add_parameter / update_params
# ---------------------------------------------------------------------------

class TestParams:
    def test_add_parameter_scalar(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_parameter('alpha', 0.01)
        assert p._params['alpha'] == 0.01

    def test_add_parameter_list(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_parameter(['a', 'b'], [1.0, 2.0])
        assert p._params == {'a': 1.0, 'b': 2.0}

    def test_add_parameter_list_length_mismatch_raises(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        with pytest.raises(ValueError, match="value"):
            p.add_parameter(['a', 'b'], [1.0])

    def test_add_parameter_trainable(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_parameter('k', 1.5, trainable=True)
        assert p._params['k'] == 1.5
        assert 'k' in p._trainable

    def test_add_parameter_trainable_multiple(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_parameter(['k', 'm'], [1.0, 2.0], trainable=True)
        assert p._params['k'] == 1.0
        assert p._params['m'] == 2.0
        assert 'k' in p._trainable
        assert 'm' in p._trainable

    def test_update_params(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_parameter('alpha', 0.0)
        p.update_params(alpha=0.5)
        assert p._params['alpha'] == 0.5

    def test_chaining_add_parameter(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        result = p.add_parameter('a', 1.0)
        assert result is p


# ---------------------------------------------------------------------------
# add_dependency
# ---------------------------------------------------------------------------

class TestAddDependency:
    def _problem(self):
        return ProblemStrong(_domain_1d_step(), ['u'])

    def test_basic(self):
        p = self._problem()
        p.add_dependency('u_prev')
        assert len(p.dependencies) == 1
        assert p.dependencies[0].name == 'u_prev'
        assert p.dependencies[0].component == 0
        assert p.dependencies[0].order == ()

    def test_with_derivative(self):
        p = self._problem()
        p.add_dependency('du_dx_prev', component=0, order=(0,))
        dep = p.dependencies[0]
        assert dep.order == (0,)

    def test_chaining(self):
        p = self._problem()
        result = p.add_dependency('u_prev')
        assert result is p

    def test_duplicate_name_raises(self):
        p = self._problem()
        p.add_dependency('u_prev')
        with pytest.raises(ValueError, match="already registered"):
            p.add_dependency('u_prev')

    def test_bad_component_raises(self):
        p = self._problem()
        with pytest.raises(ValueError, match="component"):
            p.add_dependency('bad', component=99)

    def test_bad_order_dim_raises(self):
        p = self._problem()  # n_dims=2 (x,t)
        with pytest.raises(ValueError, match="dimension index"):
            p.add_dependency('bad', order=(99,))

    def test_empty_name_raises(self):
        p = self._problem()
        with pytest.raises(ValueError, match="non-empty"):
            p.add_dependency('')



# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Term accessor properties
# ---------------------------------------------------------------------------

class TestTermAccessors:
    def test_inner_terms_filtered(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_inner(_res, name='pde')
        p.add_boundary(_res, name='bc')
        assert len(p.inner_terms) == 1
        assert len(p.boundary_terms) == 1


# ---------------------------------------------------------------------------
# _resolve_outputs
# ---------------------------------------------------------------------------

class TestResolveOutputs:
    def test_none_single_output(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        result = p._resolve_outputs(None)
        assert result == [(0, '')]

    def test_none_multi_output_raises(self):
        p = ProblemStrong(_domain_1d(), ['u', 'v'])
        with pytest.raises(ValueError, match="outputs"):
            p._resolve_outputs(None)

    def test_name_lookup(self):
        p = ProblemStrong(_domain_1d(), ['u', 'v'])
        result = p._resolve_outputs('u')
        assert result == [(0, '')]  # single → suffix suppressed

    def test_index_lookup(self):
        p = ProblemStrong(_domain_1d(), ['u', 'v'])
        result = p._resolve_outputs([0, 1])
        assert len(result) == 2

    def test_bad_name_raises(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        with pytest.raises(ValueError, match="not found"):
            p._resolve_outputs('z')

    def test_bad_index_raises(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        with pytest.raises(ValueError, match="out of range"):
            p._resolve_outputs(5)


# ---------------------------------------------------------------------------
# Periodic residual — functional correctness
# ---------------------------------------------------------------------------

class _FakeNet:
    """Minimal JAX-traceable network whose output is defined by *fn*.

    ``apply(params, x, pdict)`` ignores *params* and *pdict* and simply
    evaluates ``fn(x)`` as a JAX array with shape ``(n, 1)``.
    """
    def __init__(self, fn):
        self.fn = fn
        self.params = {}

    def apply(self, x, params=None, pdict=None):
        return self.fn(x).reshape(-1, 1)


class TestPeriodicResidual:
    """Verify that the periodic residual is zero for a truly periodic function
    and non-zero for a non-periodic one, for both value and derivative matching.

    Domain: x ∈ [-1, 1], t ∈ [0, 1].
    Periodic axis: x  →  pairs (-1, t) ↔ (+1, t).

    Periodic function  :  u(x, t) = cos(π x)
        u(-1) = cos(-π) = -1,  u(1) = cos(π) = -1   → diff = 0 ✓
        u_x(-1) = -π sin(-π) = 0,  u_x(1) = -π sin(π) = 0  → diff = 0 ✓

    Non-periodic function:  u(x, t) = x² + x
        u(-1) = 0,  u(1) = 2          → diff = -2 ✗
        u_x(-1) = -1,  u_x(1) = 3     → diff = -4 ✗
    """

    RNG = np.random.default_rng(42)
    N = 50  # number of paired samples

    def _setup(self, net_fn, match_x_derivative=1):
        """Build domain, problem, network and return (residual_fn, data)."""
        d = DomainCubic(space=[(-1.0, 1.0)], time=(0.0, 1.0))
        d.add_periodic('x', name='per')
        p = ProblemStrong(d, ['u'])
        p.add_periodic('per', name='per', component=0,
                        match_x_derivative=match_x_derivative)
        net = _FakeNet(net_fn)
        res_fn = p.make_residual_fn(net)

        # Build paired data: x_a at x=-1, x_b at x=+1, same t
        t = self.RNG.uniform(0, 1, (self.N, 1))
        x_a = np.hstack([np.full((self.N, 1), -1.0), t])
        x_b = np.hstack([np.full((self.N, 1),  1.0), t])
        data = {'per': np.vstack([x_a, x_b]).astype(np.float32)}
        return res_fn, data

    # -- periodic function: u = cos(π x) ------------------------------------

    def test_periodic_value_residual_is_zero(self):
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0])
        res_fn, data = self._setup(net_fn, match_x_derivative=0)
        R = res_fn({}, data)['per']
        np.testing.assert_allclose(np.array(R), 0.0, atol=1e-5)

    def test_periodic_value_and_deriv_residual_is_zero(self):
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0])
        res_fn, data = self._setup(net_fn, match_x_derivative=1)
        R = res_fn({}, data)['per']
        np.testing.assert_allclose(np.array(R), 0.0, atol=1e-5)

    def test_periodic_second_deriv_residual_is_zero(self):
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0])
        res_fn, data = self._setup(net_fn, match_x_derivative=2)
        R = res_fn({}, data)['per']
        np.testing.assert_allclose(np.array(R), 0.0, atol=1e-5)

    # -- non-periodic function: u = x² + x ----------------------------------

    def test_non_periodic_value_residual_nonzero(self):
        # u(-1)=0, u(1)=2  → residual = -2
        net_fn = lambda x: x[:, 0] ** 2 + x[:, 0]
        res_fn, data = self._setup(net_fn, match_x_derivative=0)
        R = np.array(res_fn({}, data)['per'][:, 0])
        np.testing.assert_allclose(R, -2.0, atol=1e-5)

    def test_non_periodic_deriv_residual_nonzero(self):
        # u_x(-1)=-1, u_x(1)=3  → derivative residual column = -4
        net_fn = lambda x: x[:, 0] ** 2 + x[:, 0]
        res_fn, data = self._setup(net_fn, match_x_derivative=1)
        R = np.array(res_fn({}, data)['per'][:, 1])
        np.testing.assert_allclose(R, -4.0, atol=1e-5)

    # -- residual shape checks -----------------------------------------------

    def test_residual_shape_value_only(self):
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0])
        res_fn, data = self._setup(net_fn, match_x_derivative=0)
        R = res_fn({}, data)['per']
        assert R.shape == (self.N, 1)

    def test_residual_shape_with_first_deriv(self):
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0])
        res_fn, data = self._setup(net_fn, match_x_derivative=1)
        R = res_fn({}, data)['per']
        assert R.shape == (self.N, 2)  # value + 1st deriv

    def test_residual_shape_with_second_deriv(self):
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0])
        res_fn, data = self._setup(net_fn, match_x_derivative=2)
        R = res_fn({}, data)['per']
        assert R.shape == (self.N, 3)  # value + 1st + 2nd deriv


# ---------------------------------------------------------------------------
# Periodic residual — 2-D spatial + time
# ---------------------------------------------------------------------------

class TestPeriodicResidual2D:
    """
    Domain: x ∈ [-1, 1], y ∈ [0, 1], t ∈ [0, 1].
    Columns: [x, y, t]  (col 0 = x, col 1 = y, col 2 = t).
    Periodic BC on the x-axis pairs (-1, y, t) ↔ (+1, y, t).

    Time- and y-varying functions are used deliberately so that any
    mis-alignment of the non-periodic coordinates (y, t) between the two
    sides of a pair would produce a non-zero residual — exposing incorrect
    point pairing in the sampling code.
    """

    RNG = np.random.default_rng(7)
    N = 60

    def _setup(self, net_fn, match_x_derivative=1):
        """Build 2-D+time domain/problem, return (residual_fn, network)."""
        d = DomainCubic(space=[(-1.0, 1.0), (0.0, 1.0)], time=(0.0, 1.0))
        d.add_periodic('x', name='per_x')
        p = ProblemStrong(d, ['u'])
        p.add_periodic('per_x', name='per_x', component=0,
                        match_x_derivative=match_x_derivative)
        net = _FakeNet(net_fn)
        res_fn = p.make_residual_fn(net)
        return res_fn, net

    def _make_data(self, y=None, t=None, t_b_override=None, y_b_override=None):
        """Build (2N, 3) paired array; optionally override y/t on the b side."""
        rng = self.RNG
        _y = rng.uniform(0, 1, (self.N, 1)) if y is None else y
        _t = rng.uniform(0, 1, (self.N, 1)) if t is None else t
        x_a = np.hstack([np.full((self.N, 1), -1.0), _y, _t])
        _y_b = _y if y_b_override is None else y_b_override
        _t_b = _t if t_b_override is None else t_b_override
        x_b = np.hstack([np.full((self.N, 1), 1.0), _y_b, _t_b])
        return {'per_x': np.vstack([x_a, x_b]).astype(np.float32)}

    # -- periodic in x, time-varying: u = cos(πx) · t² ----------------------

    def test_time_varying_periodic_zero_residual(self):
        """cos(πx)·t² is periodic in x at any t; value residual must be zero."""
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0]) * x[:, 2] ** 2
        res_fn, net = self._setup(net_fn, match_x_derivative=0)
        R = res_fn(net.params, self._make_data())['per_x']
        np.testing.assert_allclose(np.array(R), 0.0, atol=1e-5)

    def test_time_varying_periodic_deriv_zero_residual(self):
        """cos(πx)·t² is periodic including its x-derivative at any t."""
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0]) * x[:, 2] ** 2
        res_fn, net = self._setup(net_fn, match_x_derivative=1)
        R = res_fn(net.params, self._make_data())['per_x']
        np.testing.assert_allclose(np.array(R), 0.0, atol=1e-4)

    def test_time_misaligned_gives_nonzero(self):
        """Mismatched t between a/b sides turns a spatially-periodic function
        into a nonzero residual, proving t alignment is required.

        u = cos(πx)·t²:
          u_a = cos(-π)·t_a² = -t_a²
          u_b = cos(+π)·t_b² = -t_b²
          residual = u_a - u_b = t_b² - t_a²  (non-zero when t_b ≠ t_a)
        """
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0]) * x[:, 2] ** 2
        res_fn, net = self._setup(net_fn, match_x_derivative=0)
        t_a = self.RNG.uniform(0, 0.7, (self.N, 1)).astype(np.float32)
        t_b = (t_a + 0.3).astype(np.float32)   # constant offset guarantees t_a ≠ t_b
        data = self._make_data(t=t_a, t_b_override=t_b)
        R = np.array(res_fn(net.params, data)['per_x'][:, 0])
        expected = (t_b ** 2 - t_a ** 2).ravel()
        np.testing.assert_allclose(R, expected, atol=1e-5)

    # -- periodic in x with y and t: u = cos(πx) · sin(πy) · t --------------

    def test_y_t_periodic_zero_residual(self):
        """cos(πx)·sin(πy)·t is periodic in x for any aligned (y, t) pair."""
        net_fn = (lambda x:
                  jnp.cos(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1]) * x[:, 2])
        res_fn, net = self._setup(net_fn, match_x_derivative=0)
        R = res_fn(net.params, self._make_data())['per_x']
        np.testing.assert_allclose(np.array(R), 0.0, atol=1e-5)

    def test_y_misaligned_gives_nonzero(self):
        """Mismatched y between a/b sides makes residual nonzero when u depends on y.

        u = cos(πx) + y:
          u(-1, y_a, t) = -1 + y_a,  u(+1, y_b, t) = -1 + y_b
          residual = (y_a - y_b) = -0.3  for all points.
        """
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0]) + x[:, 1]
        res_fn, net = self._setup(net_fn, match_x_derivative=0)
        y_a = self.RNG.uniform(0, 0.7, (self.N, 1)).astype(np.float32)
        y_b = (y_a + 0.3).astype(np.float32)
        data = self._make_data(y=y_a, y_b_override=y_b)
        R = np.array(res_fn(net.params, data)['per_x'][:, 0])
        np.testing.assert_allclose(R, -0.3, atol=1e-5)

    def test_y_aligned_periodic_plus_y_zero(self):
        """cos(πx) + y is periodic when y is aligned between pairs."""
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0]) + x[:, 1]
        res_fn, net = self._setup(net_fn, match_x_derivative=0)
        R = res_fn(net.params, self._make_data())['per_x']
        np.testing.assert_allclose(np.array(R), 0.0, atol=1e-5)

    # -- non-periodic in x: u = x + y · t -----------------------------------

    def test_non_periodic_2d_value_residual(self):
        """u = x + y·t: u(-1, y, t) = -1+yt, u(+1, y, t) = 1+yt → diff = -2."""
        net_fn = lambda x: x[:, 0] + x[:, 1] * x[:, 2]
        res_fn, net = self._setup(net_fn, match_x_derivative=0)
        R = np.array(res_fn(net.params, self._make_data())['per_x'][:, 0])
        np.testing.assert_allclose(R, -2.0, atol=1e-5)

    # -- output shape --------------------------------------------------------

    def test_residual_shape_2d_with_deriv(self):
        """With match_x_derivative=1 the output has 2 columns (value + 1st deriv)."""
        net_fn = lambda x: jnp.cos(jnp.pi * x[:, 0]) * x[:, 2]
        res_fn, net = self._setup(net_fn, match_x_derivative=1)
        R = res_fn(net.params, self._make_data())['per_x']
        assert R.shape == (self.N, 2)
