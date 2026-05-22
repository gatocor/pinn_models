"""
Tests for ModelPartitioned.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

import pinns
from pinns import ModelPartitioned, create_model as Model
from pinns import PartitionFB, PartitionX
from pinns.domain import DomainCubic

RNG = jax.random.PRNGKey(0)
BATCH = 8


# ---------------------------------------------------------------------------
# Fixtures — domains built in partition mode (grid_positions set)
# ---------------------------------------------------------------------------

@pytest.fixture
def domain_1d_spatial():
    # 4 equal subdomains: [0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1]
    return DomainCubic(space=[np.linspace(0.0, 1.0, 5)])


@pytest.fixture
def domain_1d_time():
    # 4 spatial subdomains + continuous time [0,1]
    return DomainCubic(space=[np.linspace(0.0, 1.0, 5)], time=(0.0, 1.0))


@pytest.fixture
def domain_1d_partitioned_time():
    # 2 spatial subdomains + 2 time subdomains
    return DomainCubic(space=[np.array([0.0, 0.5, 1.0])], time=[0.0, 0.5, 1.0])


@pytest.fixture
def domain_2d():
    # 2×3 spatial grid
    return DomainCubic(space=[np.array([0.0, 0.5, 1.0]),
                               np.array([0.0, 1/3, 2/3, 1.0])])


@pytest.fixture
def model_1d_spatial(domain_1d_spatial):
    return Model(domain_1d_spatial, output_dim=1, hidden_dims=[16, 16], normalize=False)


@pytest.fixture
def model_1d_time(domain_1d_time):
    return Model(domain_1d_time, output_dim=1, hidden_dims=[16, 16], normalize=False)


@pytest.fixture
def model_1d_partitioned_time(domain_1d_partitioned_time):
    return Model(domain_1d_partitioned_time, output_dim=1, hidden_dims=[16, 16], normalize=False)


@pytest.fixture
def model_2d(domain_2d):
    return Model(domain_2d, output_dim=1, hidden_dims=[16, 16], normalize=False)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_fb_1d_spatial(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB(overlap=0.3))
        assert ens.n_models == 4
        assert ens.shape == (4,)

    def test_x_1d_spatial(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionX(interface_weight=2.0))
        assert ens.n_models == 4
        assert ens.shape == (4,)

    def test_2d_spatial(self, model_2d):
        ens = ModelPartitioned(model_2d, PartitionFB())
        assert ens.n_models == 6   # 2×3
        assert ens.shape == (2, 3)

    def test_partition_time_false(self, model_1d_time):
        ens = ModelPartitioned(model_1d_time, PartitionFB(), partition_time=False)
        assert ens.n_models == 4  # spatial only, no time split

    def test_partition_time_continuous(self, model_1d_time):
        """Continuous time interval → single temporal slab, spatial still split."""
        ens = ModelPartitioned(model_1d_time, PartitionFB(), partition_time=True)
        # 4 spatial × 1 temporal = 4
        assert ens.n_models == 4

    def test_partition_time_partitioned(self, model_1d_partitioned_time):
        """Partitioned time axis → time subdomains × spatial partitions."""
        ens = ModelPartitioned(model_1d_partitioned_time, PartitionFB(), partition_time=True)
        # 2 spatial × 2 time = 4
        assert ens.n_models == 4
        assert ens.shape == (2, 2)

    def test_wrong_strategy_type(self, model_1d_spatial):
        with pytest.raises(TypeError):
            ModelPartitioned(model_1d_spatial, "bad_strategy")

    def test_no_partition_grid_falls_back_to_single(self):
        """Domain without grid_positions → single spatial region."""
        domain = DomainCubic(space=[(0.0, 1.0)])
        model = Model(domain, output_dim=1, hidden_dims=[16, 16], normalize=False)
        ens = ModelPartitioned(model, PartitionFB())
        assert ens.n_models == 1


# ---------------------------------------------------------------------------
# Strategy assignment
# ---------------------------------------------------------------------------

class TestStrategyBounds:

    def test_fb_strategy_assigned(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB(overlap=0.4))
        for s in ens._strategies:
            assert isinstance(s, PartitionFB)
            assert s.overlap == pytest.approx(0.4)

    def test_x_strategy_assigned(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionX(interface_weight=5.0))
        for s in ens._strategies:
            assert isinstance(s, PartitionX)
            assert s.interface_weight == pytest.approx(5.0)

    def test_bounds_cover_domain(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        lo_all = min(s._xmin[0] for s in ens._strategies)
        hi_all = max(s._xmax[0] for s in ens._strategies)
        assert lo_all == pytest.approx(0.0)
        assert hi_all == pytest.approx(1.0)

    def test_subfoundaries_contiguous(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        bounds = sorted([(s._xmin[0], s._xmax[0]) for s in ens._strategies])
        for i in range(len(bounds) - 1):
            assert bounds[i][1] == pytest.approx(bounds[i + 1][0])

    def test_time_included_in_bounds(self, model_1d_partitioned_time):
        ens = ModelPartitioned(model_1d_partitioned_time, PartitionFB(), partition_time=True)
        for s in ens._strategies:
            # bounds should have 2 dims: spatial + time
            assert s._xmin.shape[0] == 2
            assert s._xmax.shape[0] == 2

    def test_time_excluded_when_false(self, model_1d_partitioned_time):
        ens = ModelPartitioned(model_1d_partitioned_time, PartitionFB(), partition_time=False)
        for s in ens._strategies:
            assert s._xmin.shape[0] == 1  # spatial only

    def test_template_not_modified(self, model_1d_spatial):
        # Strategies in the ensemble should be independent from the prototype
        proto = PartitionFB(overlap=0.3)
        ens = ModelPartitioned(model_1d_spatial, proto)
        assert all(s is not proto for s in ens._strategies)


# ---------------------------------------------------------------------------
# init / apply
# ---------------------------------------------------------------------------

class TestInitApply:

    def test_init_returns_dict(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        params = ens.init(RNG)
        assert isinstance(params, dict)
        assert set(params.keys()) == {f"sub_{i}" for i in range(ens.n_models)}

    def test_apply_shape(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        params = ens.init(RNG)
        x = jnp.ones((BATCH, 1))
        y = ens.apply(x, params)
        assert y.shape == (BATCH, 1)

    def test_apply_shape_2d(self, model_2d):
        ens = ModelPartitioned(model_2d, PartitionFB())
        params = ens.init(RNG)
        x = jnp.ones((BATCH, 2))
        y = ens.apply(x, params)
        assert y.shape == (BATCH, 1)

    def test_apply_shape_with_time(self, model_1d_partitioned_time):
        ens = ModelPartitioned(model_1d_partitioned_time, PartitionFB(), partition_time=True)
        params = ens.init(RNG)
        x = jnp.ones((BATCH, 2))   # spatial + time
        y = ens.apply(x, params)
        assert y.shape == (BATCH, 1)

    def test_apply_x_strategy(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionX())
        params = ens.init(RNG)
        x = jnp.linspace(0, 1, BATCH)[:, None]
        y = ens.apply(x, params)
        assert y.shape == (BATCH, 1)

    def test_independent_params_per_model(self, model_1d_spatial):
        """Each sub-model has independent parameters."""
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        params = ens.init(RNG)
        p0 = params["sub_0"]
        p1 = params["sub_1"]
        leaves0 = jax.tree_util.tree_leaves(p0)
        leaves1 = jax.tree_util.tree_leaves(p1)
        assert not all(jnp.allclose(a, b) for a, b in zip(leaves0, leaves1))


# ---------------------------------------------------------------------------
# Container interface
# ---------------------------------------------------------------------------

class TestContainerInterface:

    def test_len(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        assert len(ens) == 4

    def test_getitem(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        assert ens[0] is ens.models[0]

    def test_iter(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        assert len(list(ens)) == 4

    def test_output_dim(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        assert ens.output_dim == 1

    def test_domain_same(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        assert ens.domain._spatial_dims == model_1d_spatial.domain._spatial_dims

    def test_repr(self, model_1d_spatial):
        ens = ModelPartitioned(model_1d_spatial, PartitionFB())
        r = repr(ens)
        assert "ModelPartitioned" in r
        assert "PartitionFB" in r
        assert "4" in r


# ---------------------------------------------------------------------------
# Accessibility from top-level import
# ---------------------------------------------------------------------------

def test_top_level_import():
    assert hasattr(pinns, "ModelPartitioned")
    assert pinns.ModelPartitioned is ModelPartitioned

