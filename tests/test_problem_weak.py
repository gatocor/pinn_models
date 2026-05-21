"""Test suite for ProblemWeak (new ProblemStrong-style interface)."""
import numpy as np
import pytest

# Import directly to avoid triggering the torch-dependent pinns/__init__.py
from pinns.domain.domain_mesh  import DomainMesh
from pinns.problems.problem_weak import ProblemWeak

# ---------------------------------------------------------------------------
# Tiny mesh helpers
# ---------------------------------------------------------------------------

def _unit_square_mesh(n=5):
    """Return (verts, faces) for a regular triangulation of [0,1]²."""
    try:
        import pinns.meshes as meshes
        return meshes.square(x_max=1.0, y_max=1.0, mesh_size=1.0 / n)
    except Exception:
        # Fallback: hand-built 2-triangle mesh
        verts = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        return verts, faces


def _domain(n=5):
    verts, faces = _unit_square_mesh(n)
    d = DomainMesh((verts, faces))
    tol = 1e-10
    d.add_boundary(lambda v: v[:, 1] < tol,       name="bottom")
    d.add_boundary(lambda v: v[:, 0] > 1.0 - tol, name="right")
    d.add_boundary(lambda v: v[:, 1] > 1.0 - tol, name="top")
    d.add_boundary(lambda v: v[:, 0] < tol,       name="left")
    return d


def _volume_fn_poisson(x, y, params, phi, derivative=None):
    """∇u·∇φ − f·φ  for −∇²u = f,  f = 2π²sin(πx)sin(πy)."""
    import jax.numpy as jnp
    du_dx   = derivative(y,   x, 0, (0,))
    du_dy   = derivative(y,   x, 0, (1,))
    dphi_dx = derivative(phi, x, 0, (0,))
    dphi_dy = derivative(phi, x, 0, (1,))
    f = 2.0 * jnp.pi**2 * jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1])
    return du_dx * dphi_dx + du_dy * dphi_dy - f * phi


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        assert p.n_outputs == 1
        assert p.output_names == ["u"]
        assert p.domain is d

    def test_multi_output(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u", "v"])
        assert p.n_outputs == 2

    def test_bad_domain_raises(self):
        with pytest.raises(TypeError, match="DomainMesh"):
            ProblemWeak("not_a_domain", output_names=["u"])

    def test_input_names_auto_derived(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        # 2-D spatial domain → auto input_names = ['x', 'y']
        assert p.input_names == ["x", "y"]

    def test_cubature_order_stored(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"], cubature_order=2)
        assert p.cubature_order == 2

    def test_lagrange_order_stored(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"], lagrange_order=1)
        assert p.lagrange_order == 1

    def test_cubature_data_populated(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        assert "pts"      in p.cubature_data
        assert "weights"  in p.cubature_data
        assert "phi"      in p.cubature_data
        assert "grad_phi" in p.cubature_data
        assert "node_ids" in p.cubature_data

    def test_free_nodes_set(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        assert p.free_nodes is not None
        assert len(p.free_nodes) > 0

    def test_repr(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        r = repr(p)
        assert "ProblemWeak" in r
        assert "n_free" in r


# ---------------------------------------------------------------------------
# add_inner
# ---------------------------------------------------------------------------

class TestAddInner:
    def test_sets_volume_fn(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        p.add_inner(_volume_fn_poisson, name="pde")
        assert p.volume_fn is _volume_fn_poisson

    def test_name_stored(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        p.add_inner(_volume_fn_poisson, name="my_pde")
        assert p._volume_name == "my_pde"

    def test_default_name(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        p.add_inner(_volume_fn_poisson)
        assert p._volume_name == "pde"

    def test_returns_self(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        ret = p.add_inner(_volume_fn_poisson)
        assert ret is p


# ---------------------------------------------------------------------------
# add_parameter
# ---------------------------------------------------------------------------

class TestAddParameter:
    def test_adds_params(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        p.add_parameter("kappa", 1.0)
        p.add_parameter("alpha", 0.5)
        assert p.params["kappa"] == 1.0
        assert p.params["alpha"] == 0.5

    def test_returns_self(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        ret = p.add_parameter("k", 2.0)
        assert ret is p

    def test_constructor_params(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"], params={"k": 3.0})
        assert p.params["k"] == 3.0


# ---------------------------------------------------------------------------
# add_dirichlet
# ---------------------------------------------------------------------------

class TestAddDirichlet:
    def test_new_style_region(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        p.add_dirichlet(0.0, name="bc_bottom", region="bottom")
        assert len(p.boundary_conditions) == 1
        bc = p.boundary_conditions[0]
        from pinns.problems.terms import TermDirichletBC
        assert isinstance(bc, TermDirichletBC)
        assert bc.name == "bc_bottom"

    def test_all_four_sides(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        p.add_dirichlet(0.0, name="bc_bottom", region="bottom")
        p.add_dirichlet(0.0, name="bc_right",  region="right")
        p.add_dirichlet(0.0, name="bc_top",    region="top")
        p.add_dirichlet(0.0, name="bc_left",   region="left")
        assert len(p.boundary_conditions) == 4

    def test_returns_self(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        ret = p.add_dirichlet(0.0, name="bc_bottom", region="bottom")
        assert ret is p

    def test_region_all(self):
        """region='all' selects the entire boundary."""
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        p.add_dirichlet(0.0, name="bc_all")
        assert len(p.boundary_conditions) == 1

    def test_callable_value(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        p.add_dirichlet(lambda x: np.zeros(len(x)), name="bc_bottom", region="bottom")
        assert len(p.boundary_conditions) == 1

    def test_component_stored(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u", "v"])
        p.add_dirichlet(0.0, name="bc_u", region="bottom", outputs="u")
        p.add_dirichlet(0.0, name="bc_v", region="bottom", outputs="v")
        assert p.boundary_conditions[0].component == 0
        assert p.boundary_conditions[1].component == 1

    def test_node_indices_non_empty(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        p.add_dirichlet(0.0, name="bc_bottom", region="bottom")
        bc = p.boundary_conditions[0]
        # Geometry lives on the domain, not patched onto the Term
        ni = d._boundary_regions[bc.region]['node_indices']
        assert ni is not None
        assert len(ni) > 0


# ---------------------------------------------------------------------------
# add_neumann
# ---------------------------------------------------------------------------

class TestAddNeumann:
    def test_basic(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        p.add_neumann(0.0, name="nbc_right", region="right")
        assert len(p.boundary_conditions) == 1
        bc = p.boundary_conditions[0]
        from pinns.problems.terms import TermNeumannBC
        assert isinstance(bc, TermNeumannBC)
        assert bc.name == "nbc_right"

    def test_returns_self(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        ret = p.add_neumann(0.0, name="nbc_right", region="right")
        assert ret is p


# ---------------------------------------------------------------------------
# Method chaining
# ---------------------------------------------------------------------------

class TestChaining:
    def test_full_chain(self):
        d = _domain()
        p = (
            ProblemWeak(d, output_names=["u"])
            .add_inner(_volume_fn_poisson, name="pde")
            .add_dirichlet(0.0, name="bc_bottom", region="bottom")
            .add_dirichlet(0.0, name="bc_right",  region="right")
            .add_dirichlet(0.0, name="bc_top",    region="top")
            .add_dirichlet(0.0, name="bc_left",   region="left")
            .add_parameter("dummy", 1.0)
        )
        assert p.volume_fn is _volume_fn_poisson
        assert len(p.boundary_conditions) == 4
        assert p.params["dummy"] == 1.0


# ---------------------------------------------------------------------------
# Cubature data sanity checks
# ---------------------------------------------------------------------------

class TestCubatureData:
    """Quick sanity checks on pre-computed cubature arrays."""

    def test_shapes_consistent(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        cd = p.cubature_data
        F, Q = cd['pts'].shape[:2]
        L = cd['phi'].shape[2]
        assert cd['weights'].shape  == (F, Q)
        assert cd['phi'].shape      == (F, Q, L)
        assert cd['grad_phi'].shape == (F, Q, L, 2)
        assert cd['node_ids'].shape == (F, L)

    def test_weights_positive(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        assert (p.cubature_data['weights'] > 0).all()

    def test_phi_partition_of_unity(self):
        """For P1 elements: Σ_a φ_a(x) = 1 at every quadrature point."""
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        phi = p.cubature_data['phi']   # (F, Q, L)
        sums = phi.sum(axis=2)         # (F, Q)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_weights_sum_to_area(self):
        """Total weight should equal domain area ≈ 1.0 for unit square."""
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        total = p.cubature_data['weights'].sum()
        assert abs(total - 1.0) < 0.05   # 5% tolerance for coarse mesh


# ---------------------------------------------------------------------------
# n_free_nodes property
# ---------------------------------------------------------------------------

class TestFreeNodes:
    def test_all_nodes_free_without_dirichlet(self):
        """Without Dirichlet BCs enforced as hard constraints, all DOFs are free."""
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        assert p.n_free_nodes == p.n_dofs

    def test_n_dofs_positive(self):
        d = _domain()
        p = ProblemWeak(d, output_names=["u"])
        assert p.n_dofs > 0
