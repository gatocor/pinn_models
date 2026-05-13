"""Test suite for ProblemStrong."""
import numpy as np
import pytest

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
# add_fixed / add_inferred / update_params
# ---------------------------------------------------------------------------

class TestParams:
    def test_add_fixed_scalar(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_fixed('alpha', 0.01)
        assert p.fixed_params['alpha'] == 0.01

    def test_add_fixed_list(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_fixed(['a', 'b'], [1.0, 2.0])
        assert p.fixed_params == {'a': 1.0, 'b': 2.0}

    def test_add_fixed_list_length_mismatch_raises(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        with pytest.raises(ValueError, match="value"):
            p.add_fixed(['a', 'b'], [1.0])

    def test_add_inferred(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_inferred('k', init=1.5)
        assert p.inferred_params['k'] == 1.5

    def test_add_inferred_multiple(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_inferred(['k', 'm'], init=[1.0, 2.0])
        assert p.inferred_params == {'k': 1.0, 'm': 2.0}

    def test_update_params(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        p.add_fixed('alpha', 0.0)
        p.update_params(alpha=0.5)
        assert p.fixed_params['alpha'] == 0.5

    def test_chaining_add_fixed(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        result = p.add_fixed('a', 1.0)
        assert result is p

    def test_add_solution(self):
        p = ProblemStrong(_domain_1d(), ['u'])
        sol = lambda x, pars: np.zeros(x.shape[0])
        p.add_solution(sol)
        assert callable(p.solution)


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
