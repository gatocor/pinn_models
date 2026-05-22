"""Tests for pinns.models.layers — all composable layer types.

Run with:
    conda run -n pinn python -m pytest tests/test_layers.py -v --override-ini addopts=
"""

import os
os.environ.setdefault("PINNS_BACKEND", "jax")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from pinns.models.layers import (
    RandomFourierFeatures, FourierFeatures, GNNFeatures, LaplacianFeatures, AlphaTransform,
    Normalize, Denormalize,
    FNN, WFFNN, ResNet, PirateNet,
)

# ---------------------------------------------------------------------------
# Tiny [0,1]^2 mesh shared by GNN / Laplacian tests
#
#   3──2
#   |\ |
#   | \|
#   0──1
# ---------------------------------------------------------------------------
VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

RNG = jax.random.PRNGKey(0)
BATCH = 5

@pytest.fixture
def simple_domain():
    from pinns import DomainMesh
    return DomainMesh((VERTS, FACES))


@pytest.fixture
def query_spatial():
    """Interior query points (x, y) for the unit square mesh."""
    return jnp.array(
        [[0.2, 0.2], [0.5, 0.5], [0.7, 0.3], [0.3, 0.8], [0.6, 0.6]],
        dtype=jnp.float32,
    )


@pytest.fixture
def query_spacetime():
    """Query points (x, y, t) for space-time tests."""
    return jnp.array(
        [[0.2, 0.2, 0.1], [0.5, 0.5, 0.5], [0.7, 0.3, 0.9],
         [0.3, 0.8, 0.2], [0.6, 0.6, 0.7]],
        dtype=jnp.float32,
    )


@pytest.fixture
def spacetime_domain():
    """Unit-square mesh with a [0,1] time interval."""
    from pinns import DomainMesh
    return DomainMesh((VERTS, FACES), time=(0.0, 1.0))


# ---------------------------------------------------------------------------
# Helper: configure a RandomFourierFeatures via a minimal ModelBase so that
# _configure() is called, just as it would be in production code.
# ---------------------------------------------------------------------------
def _make_ff(n_features=8, sigma=1.0, seed=0, include_input=False,
             encode_time=None, spatial_dims=2, has_time=False, n_context=0):
    from pinns.domain import DomainCubic
    from pinns.models.model_base import ModelBase
    space = [[0.0, 1.0]] * spatial_dims
    domain = DomainCubic(space=space, time=[0.0, 1.0] if has_time else None)
    net = ModelBase(domain, output_dim=1, n_context=n_context)
    ff  = RandomFourierFeatures(n_features=n_features, sigma=sigma, seed=seed,
                          include_input=include_input, encode_time=encode_time)
    net.add(ff)
    return ff


# ===========================================================================
# FourierFeatures
# ===========================================================================

class TestRandomFourierFeatures:

    def test_output_dim_no_include(self):
        ff = _make_ff(spatial_dims=3, n_features=16)
        assert ff.output_dim == 32

    def test_output_dim_with_include_input(self):
        ff = _make_ff(spatial_dims=3, n_features=16, include_input=True)
        assert ff.output_dim == 35   # 32 + 3

    def test_call_shape(self):
        ff = _make_ff(spatial_dims=3, n_features=16)
        out = ff(jnp.ones((BATCH, 3)))
        assert out.shape == (BATCH, 32)

    def test_call_with_params_dict(self):
        """params_dict argument is ignored — should not raise."""
        ff  = _make_ff(spatial_dims=2, n_features=8)
        out = ff(jnp.ones((BATCH, 2)), params_dict={"anything": 1})
        assert out.shape == (BATCH, 16)

    def test_transform_alias(self):
        ff = _make_ff(spatial_dims=2, n_features=8)
        x  = jnp.ones((BATCH, 2))
        assert jnp.allclose(ff(x), ff.transform(x))

    def test_deterministic_across_seeds(self):
        """Same seed → same B → identical output."""
        ff1 = _make_ff(spatial_dims=2, n_features=8, seed=42)
        ff2 = _make_ff(spatial_dims=2, n_features=8, seed=42)
        x   = jax.random.normal(RNG, (BATCH, 2))
        assert jnp.allclose(ff1(x), ff2(x))

    def test_different_seeds_differ(self):
        ff1 = _make_ff(spatial_dims=2, n_features=8, seed=0)
        ff2 = _make_ff(spatial_dims=2, n_features=8, seed=1)
        x   = jax.random.normal(RNG, (BATCH, 2))
        assert not jnp.allclose(ff1(x), ff2(x))

    def test_include_input_shape(self):
        ff  = _make_ff(spatial_dims=2, n_features=8, include_input=True)
        out = ff(jnp.ones((BATCH, 2)))
        assert out.shape == (BATCH, 18)   # 16 + 2

    def test_cos_sin_range(self):
        """All values should lie in [-1, 1] since they are cos/sin."""
        ff  = _make_ff(spatial_dims=3, n_features=32)
        out = ff(jax.random.normal(RNG, (100, 3)))
        assert jnp.all(out >= -1.0 - 1e-5)
        assert jnp.all(out <=  1.0 + 1e-5)

    def test_repr(self):
        ff = _make_ff(spatial_dims=2, n_features=8, sigma=5.0)
        r  = repr(ff)
        assert "RandomFourierFeatures" in r
        assert "n_features=8" in r
        assert "sigma=5.0" in r

    def test_compose_with_fnn(self):
        """RandomFourierFeatures output feeds correctly into an FNN layer."""
        from pinns.models.layers.fnn import FNNModule
        ff      = _make_ff(spatial_dims=2, n_features=8)
        module  = FNNModule(layer_sizes=(ff.output_dim, 32, 1))
        feats   = ff(jnp.ones((BATCH, 2)))
        params  = module.init(RNG, feats[:1])
        y       = module.apply(feats, params)
        assert y.shape == (BATCH, 1)

    def test_aliased_in_layers_module(self):
        import pinns.models.layers as T
        assert T.RandomFourierFeatures is RandomFourierFeatures

    # --- space-time domain, encode_time=False (spatial Fourier + raw t) -----

    def test_time_no_encode_requires_flag(self):
        """Must raise if domain has time and encode_time is not set."""
        with pytest.raises(ValueError, match="encode_time"):
            _make_ff(spatial_dims=2, n_features=8, has_time=True)  # encode_time=None

    def test_time_encode_false_output_dim(self):
        ff = _make_ff(spatial_dims=2, n_features=8, has_time=True, encode_time=False)
        # 2*8 (spatial Fourier) + 1 (raw t) = 17
        assert ff.output_dim == 17

    def test_time_encode_false_call_shape(self, query_spacetime):
        ff  = _make_ff(spatial_dims=2, n_features=8, has_time=True, encode_time=False)
        out = ff(query_spacetime)
        assert out.shape == (BATCH, 17)

    def test_time_encode_false_t_passthrough(self, query_spacetime):
        """Last column of output should equal the raw t input."""
        ff  = _make_ff(spatial_dims=2, n_features=8, has_time=True, encode_time=False)
        out = ff(query_spacetime)
        assert jnp.allclose(out[:, -1:], query_spacetime[:, 2:3])

    def test_time_encode_false_include_input(self, query_spacetime):
        ff = _make_ff(spatial_dims=2, n_features=8, has_time=True,
                      encode_time=False, include_input=True)
        # 2 (spatial) + 2*8 + 1 (raw t) = 19
        assert ff.output_dim == 19
        assert ff(query_spacetime).shape == (BATCH, 19)

    # --- space-time domain, encode_time=True (full spacetime Fourier) -------

    def test_time_encode_true_output_dim(self):
        ff = _make_ff(spatial_dims=2, n_features=8, has_time=True, encode_time=True)
        # Fourier input dim = 3 (x, y, t), output = 2*8 = 16
        assert ff.output_dim == 16

    def test_time_encode_true_call_shape(self, query_spacetime):
        ff  = _make_ff(spatial_dims=2, n_features=8, has_time=True, encode_time=True)
        out = ff(query_spacetime)
        assert out.shape == (BATCH, 16)

    def test_time_encode_true_include_input(self, query_spacetime):
        ff = _make_ff(spatial_dims=2, n_features=8, has_time=True,
                      encode_time=True, include_input=True)
        # 3 (x,y,t) + 2*8 = 19
        assert ff.output_dim == 19
        assert ff(query_spacetime).shape == (BATCH, 19)

    def test_repr_shows_encode_time(self):
        ff1 = _make_ff(spatial_dims=2, n_features=8, has_time=True, encode_time=False)
        assert "encode_time=False" in repr(ff1)
        ff2 = _make_ff(spatial_dims=2, n_features=8, has_time=True, encode_time=True)
        assert "encode_time=True" in repr(ff2)



# ===========================================================================
# GNNFeatures
# ===========================================================================

class TestGNNFeatures:

    def test_construction(self, simple_domain):
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1)
        assert enc.spatial_dims == 2
        assert enc.hidden_dim == 8
        assert enc.output_dim == 8  # no time

    def test_output_dim_no_time(self, simple_domain):
        enc = GNNFeatures(simple_domain, hidden_dim=16)
        assert enc.output_dim == 16

    def test_init_returns_params(self, simple_domain):
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1)
        params = enc.init(RNG)
        assert isinstance(params, dict)

    def test_call_shape(self, simple_domain, query_spatial):
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1)
        params = enc.init(RNG)
        out = enc(params, query_spatial)
        assert out.shape == (BATCH, 8)

    def test_call_with_params_dict(self, simple_domain, query_spatial):
        """params_dict argument is ignored — should not raise."""
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1)
        params = enc.init(RNG)
        out = enc(params, query_spatial, params_dict={"anything": 1})
        assert out.shape == (BATCH, 8)

    def test_repr(self, simple_domain):
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1)
        r = repr(enc)
        assert "GNNFeatures" in r
        assert "hidden_dim=8" in r

    def test_compose_with_fnn(self, simple_domain, query_spatial):
        """GNNFeatures can be used as input to an FNN layer."""
        from pinns.models.layers.fnn import FNNModule
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1)
        module = FNNModule(layer_sizes=(enc.output_dim, 16, 1))
        enc_params = enc.init(jax.random.PRNGKey(1))
        features = enc(enc_params, query_spatial)
        net_params = module.init(RNG, features[:1])
        y = module.apply(features, net_params)
        assert y.shape == (BATCH, 1)

    def test_aliased_in_layers_module(self):
        import pinns.models.layers as T
        assert T.GNNFeatures is GNNFeatures


# ===========================================================================
# LaplacianFeatures  (= AlphaTransform)
# ===========================================================================

class TestLaplacianFeatures:

    def test_construction(self, simple_domain):
        enc = LaplacianFeatures(simple_domain, n_features=3)
        assert enc.n_features == 3
        assert enc.output_dim == 3

    def test_output_dim(self, simple_domain):
        enc = LaplacianFeatures(simple_domain, n_features=3)
        assert enc.output_dim == 3

    def test_eigenvector_shape(self, simple_domain):
        enc = LaplacianFeatures(simple_domain, n_features=3)
        assert enc._Phi.shape == (4, 3)  # 4 nodes, 3 eigenvectors

    def test_call_shape(self, simple_domain, query_spatial):
        enc = LaplacianFeatures(simple_domain, n_features=3)
        out = enc(query_spatial)
        assert out.shape == (BATCH, 3)

    def test_call_with_params_dict(self, simple_domain, query_spatial):
        """params_dict is ignored — no raise."""
        enc = LaplacianFeatures(simple_domain, n_features=3)
        out = enc(query_spatial, params_dict={"x": 1})
        assert out.shape == (BATCH, 3)

    def test_transform_alias(self, simple_domain, query_spatial):
        enc = LaplacianFeatures(simple_domain, n_features=3)
        assert jnp.allclose(enc(query_spatial), enc.transform(query_spatial))

    def test_deterministic(self, simple_domain, query_spatial):
        # Eigenvectors are unique only up to sign, so compare absolute values.
        enc1 = LaplacianFeatures(simple_domain, n_features=3)
        enc2 = LaplacianFeatures(simple_domain, n_features=3)
        assert jnp.allclose(jnp.abs(enc1(query_spatial)), jnp.abs(enc2(query_spatial)))

    def test_repr(self, simple_domain):
        enc = LaplacianFeatures(simple_domain, n_features=3)
        r = repr(enc)
        assert "LaplacianFeatures" in r
        assert "3 eigenvectors" in r

    def test_alpha_transform_alias(self, simple_domain, query_spatial):
        """AlphaTransform is the same class as LaplacianFeatures."""
        enc = AlphaTransform(simple_domain, n_features=3)
        out = enc(query_spatial)
        assert out.shape == (BATCH, 3)
        assert AlphaTransform is LaplacianFeatures

    def test_compose_with_fnn(self, simple_domain, query_spatial):
        """LaplacianFeatures can be used as input to an FNN layer."""
        from pinns.models.layers.fnn import FNNModule
        enc = LaplacianFeatures(simple_domain, n_features=3)
        features = enc(query_spatial)
        module = FNNModule(layer_sizes=(enc.output_dim, 16, 1))
        params = module.init(RNG, features[:1])
        y = module.apply(features, params)
        assert y.shape == (BATCH, 1)

    def test_aliased_in_layers_module(self):
        import pinns.models.layers as T
        assert T.LaplacianFeatures is LaplacianFeatures
        assert T.AlphaTransform is AlphaTransform


# ===========================================================================
# Context fields (n_context > 0) — step-integration use case
# ===========================================================================

N_CTX = 2  # number of context columns (e.g. [u, v] from previous step)


@pytest.fixture
def query_with_context(query_spatial):
    """Spatial query points with N_CTX extra context columns appended."""
    ctx = jnp.ones((BATCH, N_CTX), dtype=jnp.float32) * 0.5
    return jnp.concatenate([query_spatial, ctx], axis=-1)


@pytest.fixture
def query_spacetime_with_context(query_spacetime):
    """Spacetime query points with N_CTX extra context columns appended."""
    ctx = jnp.ones((BATCH, N_CTX), dtype=jnp.float32) * 0.5
    return jnp.concatenate([query_spacetime, ctx], axis=-1)


class TestContextFields:
    """n_context > 0: context columns are forwarded / encoded by each transformer."""

    # ------------------------------------------------------------------
    # RandomFourierFeatures
    # ------------------------------------------------------------------

    def test_fourier_context_output_dim(self):
        ff = _make_ff(spatial_dims=2, n_features=8, n_context=N_CTX)
        assert ff.output_dim == 16 + N_CTX

    def test_fourier_context_call_shape(self, query_with_context):
        ff = _make_ff(spatial_dims=2, n_features=8, n_context=N_CTX)
        out = ff(query_with_context)
        assert out.shape == (BATCH, 16 + N_CTX)

    def test_fourier_context_passthrough(self, query_with_context):
        """Context columns appear unchanged at the end of the output."""
        ff = _make_ff(spatial_dims=2, n_features=8, n_context=N_CTX)
        out = ff(query_with_context)
        ctx_in  = query_with_context[:, -N_CTX:]
        ctx_out = out[:, -N_CTX:]
        assert jnp.allclose(ctx_in, ctx_out)

    def test_fourier_domain_context_output_dim(self):
        ff = _make_ff(spatial_dims=2, n_features=8, n_context=N_CTX)
        assert ff.output_dim == 16 + N_CTX

    def test_fourier_domain_time_context_encode_false(
        self, query_spacetime_with_context
    ):
        """spatial Fourier + raw t + N_CTX context columns."""
        ff = _make_ff(spatial_dims=2, n_features=8, has_time=True,
                      encode_time=False, n_context=N_CTX)
        # 2*8 (spatial) + 1 (t) + N_CTX
        assert ff.output_dim == 17 + N_CTX
        out = ff(query_spacetime_with_context)
        assert out.shape == (BATCH, 17 + N_CTX)
        # context passthrough
        assert jnp.allclose(out[:, -N_CTX:], query_spacetime_with_context[:, -N_CTX:])

    def test_fourier_context_repr(self):
        ff = _make_ff(spatial_dims=2, n_features=8, n_context=N_CTX)
        assert f"n_context={N_CTX}" in repr(ff)

    # ------------------------------------------------------------------
    # GNNFeatures
    # ------------------------------------------------------------------

    def test_gnn_context_output_dim(self, simple_domain):
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1, n_context=N_CTX)
        assert enc.output_dim == 8 + N_CTX

    def test_gnn_context_init(self, simple_domain):
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1, n_context=N_CTX)
        params = enc.init(RNG)
        assert isinstance(params, dict)

    def test_gnn_context_call_shape(self, simple_domain, query_with_context):
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1, n_context=N_CTX)
        params = enc.init(RNG)
        out = enc(params, query_with_context)
        assert out.shape == (BATCH, 8 + N_CTX)

    def test_gnn_context_passthrough(self, simple_domain, query_with_context):
        """Raw context appears at the end of the GNN output (skip connection)."""
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1, n_context=N_CTX)
        params = enc.init(RNG)
        out = enc(params, query_with_context)
        ctx_in  = query_with_context[:, -N_CTX:]
        ctx_out = out[:, -N_CTX:]
        assert jnp.allclose(ctx_in, ctx_out)

    def test_gnn_context_repr(self, simple_domain):
        enc = GNNFeatures(simple_domain, hidden_dim=8, message_steps=1, n_context=N_CTX)
        assert f"n_context={N_CTX}" in repr(enc)

    # ------------------------------------------------------------------
    # LaplacianFeatures
    # ------------------------------------------------------------------

    def test_laplacian_context_output_dim_encoded(self, simple_domain):
        """encode_context=True (default): output_dim = n_eig * (1 + n_context)."""
        enc = LaplacianFeatures(simple_domain, n_features=3, n_context=N_CTX)
        assert enc.output_dim == 3 * (1 + N_CTX)

    def test_laplacian_context_output_dim_raw(self, simple_domain):
        """encode_context=False: output_dim = n_eig + n_context."""
        enc = LaplacianFeatures(
            simple_domain, n_features=3, n_context=N_CTX, encode_context=False
        )
        assert enc.output_dim == 3 + N_CTX

    def test_laplacian_context_call_shape_encoded(self, simple_domain, query_with_context):
        enc = LaplacianFeatures(simple_domain, n_features=3, n_context=N_CTX)
        out = enc(query_with_context)
        assert out.shape == (BATCH, 3 * (1 + N_CTX))

    def test_laplacian_context_call_shape_raw(self, simple_domain, query_with_context):
        enc = LaplacianFeatures(
            simple_domain, n_features=3, n_context=N_CTX, encode_context=False
        )
        out = enc(query_with_context)
        assert out.shape == (BATCH, 3 + N_CTX)

    def test_laplacian_context_raw_passthrough(self, simple_domain, query_with_context):
        """encode_context=False: context columns appear unchanged at the end."""
        enc = LaplacianFeatures(
            simple_domain, n_features=3, n_context=N_CTX, encode_context=False
        )
        out = enc(query_with_context)
        ctx_in  = query_with_context[:, -N_CTX:]
        ctx_out = out[:, -N_CTX:]
        assert jnp.allclose(ctx_in, ctx_out)

    def test_laplacian_context_phi_u_encoding(self, simple_domain, query_with_context):
        """encode_context=True: each context field is φ(x)*u_j(x).

        Use the φ block from the same encoder (out[:, :3]) as reference so
        per-column sign ambiguity is not an issue.
        """
        enc = LaplacianFeatures(simple_domain, n_features=3, n_context=N_CTX)
        out = enc(query_with_context)
        phi_x = out[:, :3]                       # φ(x) from this encoder
        u1 = query_with_context[:, 2:3]          # (5, 1) = all 0.5
        u2 = query_with_context[:, 3:4]          # (5, 1) = all 0.5
        # Output layout: [φ(x) | φ(x)*u1 | φ(x)*u2]
        assert jnp.allclose(out[:, 3:6], phi_x * u1, atol=1e-6)
        assert jnp.allclose(out[:, 6:9], phi_x * u2, atol=1e-6)

    def test_laplacian_context_repr(self, simple_domain):
        enc = LaplacianFeatures(simple_domain, n_features=3, n_context=N_CTX)
        r = repr(enc)
        assert f"n_context={N_CTX}" in r
        assert "φᵀu" in r

    def test_laplacian_context_repr_raw(self, simple_domain):
        enc = LaplacianFeatures(
            simple_domain, n_features=3, n_context=N_CTX, encode_context=False
        )
        assert "raw" in repr(enc)


# ===========================================================================
# Normalize / Denormalize (standalone — no ModelBase required)
# ===========================================================================

class TestNormalizeLayer:

    def test_spatial_normalize(self, simple_domain):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = Normalize()
        net.add(layer)
        params = net.init(RNG)
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=jnp.float32)
        out = layer.apply(x)
        assert out.shape == (3, 2)
        assert float(out.min()) >= -1.0 - 1e-5
        assert float(out.max()) <=  1.0 + 1e-5

    def test_init_returns_empty(self, simple_domain):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = Normalize()
        net.add(layer)
        assert layer.init(RNG) == {}

    def test_repr(self, simple_domain):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = Normalize()
        net.add(layer)
        assert "Normalize" in repr(layer)

    def test_denormalize_noop_without_range(self, simple_domain):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = Denormalize()
        net.add(layer)
        x = jnp.ones((BATCH, 1))
        out = layer.apply(x)
        assert jnp.allclose(x, out)

    def test_denormalize_rescales(self, simple_domain):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1, output_range=(0.0, 2.0))
        layer = Denormalize()
        net.add(layer)
        # Input -1 → 0.0, input 1 → 2.0
        x = jnp.array([[-1.0], [1.0]])
        out = layer.apply(x)
        assert jnp.allclose(out, jnp.array([[0.0], [2.0]]), atol=1e-5)


# ===========================================================================
# FNN / WFFNN composable layers
# ===========================================================================

class TestFNNLayer:

    def test_configure_sets_layer_sizes(self, simple_domain, query_spatial):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = FNN([32, 32])
        net.add(layer)
        assert layer._layer_sizes == [2, 32, 32, 1]

    def test_forward_shape(self, simple_domain, query_spatial):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=2)
        layer = FNN([64])
        net.add(layer)
        params = layer.init(RNG)
        out = layer.apply(query_spatial, params)
        assert out.shape == (BATCH, 2)

    def test_repr(self, simple_domain):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = FNN([32])
        net.add(layer)
        assert "FNN" in repr(layer)
        assert "[2, 32, 1]" in repr(layer)

    def test_wffnn_configure_and_forward(self, simple_domain, query_spatial):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = WFFNN([32])
        net.add(layer)
        params = layer.init(RNG)
        out = layer.apply(query_spatial, params)
        assert out.shape == (BATCH, 1)


# ===========================================================================
# ResNet composable layer
# ===========================================================================

class TestResNetLayer:

    def test_configure_and_forward(self, simple_domain, query_spatial):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = ResNet(hidden_dim=16, n_blocks=2)
        net.add(layer)
        params = layer.init(RNG)
        out = layer.apply(query_spatial, params)
        assert out.shape == (BATCH, 1)

    def test_repr(self, simple_domain):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = ResNet(hidden_dim=16, n_blocks=2)
        net.add(layer)
        assert "ResNet" in repr(layer)
        assert "hidden_dim=16" in repr(layer)

    def test_spacetime_input(self, spacetime_domain, query_spacetime):
        from pinns.models.model_base import ModelBase
        net = ModelBase(spacetime_domain, output_dim=1)
        layer = ResNet(hidden_dim=16, n_blocks=2)
        net.add(layer)
        params = layer.init(RNG)
        out = layer.apply(query_spacetime, params)
        assert out.shape == (BATCH, 1)

    def test_multi_output(self, simple_domain, query_spatial):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=3)
        layer = ResNet(hidden_dim=32, n_blocks=1)
        net.add(layer)
        params = layer.init(RNG)
        out = layer.apply(query_spatial, params)
        assert out.shape == (BATCH, 3)


# ===========================================================================
# PirateNet composable layer
# ===========================================================================

class TestPirateNetLayer:

    def test_configure_and_forward(self, simple_domain, query_spatial):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = PirateNet(hidden_dim=16, n_blocks=2)
        net.add(layer)
        params = layer.init(RNG)
        out = layer.apply(query_spatial, params)
        assert out.shape == (BATCH, 1)

    def test_repr(self, simple_domain):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=1)
        layer = PirateNet(hidden_dim=16, n_blocks=2)
        net.add(layer)
        assert "PirateNet" in repr(layer)
        assert "hidden_dim=16" in repr(layer)

    def test_spacetime_input(self, spacetime_domain, query_spacetime):
        from pinns.models.model_base import ModelBase
        net = ModelBase(spacetime_domain, output_dim=1)
        layer = PirateNet(hidden_dim=16, n_blocks=2)
        net.add(layer)
        params = layer.init(RNG)
        out = layer.apply(query_spacetime, params)
        assert out.shape == (BATCH, 1)

    def test_multi_output(self, simple_domain, query_spatial):
        from pinns.models.model_base import ModelBase
        net = ModelBase(simple_domain, output_dim=3)
        layer = PirateNet(hidden_dim=32, n_blocks=1)
        net.add(layer)
        params = layer.init(RNG)
        out = layer.apply(query_spatial, params)
        assert out.shape == (BATCH, 3)


# ===========================================================================
# Cross-module imports
# ===========================================================================


def test_layers_module_exports():
    import pinns.models.layers as L
    for name in [
        "RandomFourierFeatures", "FourierFeatures", "GNNFeatures", "LaplacianFeatures", "AlphaTransform",
        "Normalize", "Denormalize",
        "FNN", "WFFNN", "ResNet", "PirateNet",
    ]:
        assert hasattr(L, name), f"pinns.models.layers missing {name}"


def test_pinns_has_layers_attribute():
    import pinns
    assert hasattr(pinns, "layers")
    assert hasattr(pinns.models.layers, "RandomFourierFeatures")
    assert hasattr(pinns.models.layers, "FourierFeatures")  # backward-compat alias
    assert hasattr(pinns.models.layers, "GNNFeatures")
    assert hasattr(pinns.models.layers, "LaplacianFeatures")
    assert hasattr(pinns.models.layers, "Normalize")
    assert hasattr(pinns.models.layers, "FNN")
    assert hasattr(pinns.models.layers, "ResNet")
    assert hasattr(pinns.models.layers, "PirateNet")


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
