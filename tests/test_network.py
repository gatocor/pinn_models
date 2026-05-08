"""
Tests for pinns.network: Network, Normalize, Denormalize, FNN.

Run with:
    conda run -n pinn python -m pytest tests/test_network.py -v --override-ini addopts=
"""

import os
os.environ.setdefault("PINNS_BACKEND", "jax")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from pinns import DomainMesh
from pinns.network import Network
from pinns.layers import (
    Normalize, Denormalize,
    FNN, WFFNN, ResNet, PirateNet,
    GNNFeatures, LaplacianFeatures, FourierFeatures,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
RNG   = jax.random.PRNGKey(42)
BATCH = 5

@pytest.fixture
def mesh_domain():
    return DomainMesh((VERTS, FACES))

@pytest.fixture
def spacetime_domain():
    return DomainMesh((VERTS, FACES), time=(0.0, 1.0))

@pytest.fixture
def xy_pts():
    return jnp.array(
        [[0.2, 0.2], [0.5, 0.5], [0.7, 0.3], [0.3, 0.8], [0.6, 0.6]],
        dtype=jnp.float32,
    )

@pytest.fixture
def xyt_pts():
    return jnp.array(
        [[0.2, 0.2, 0.1], [0.5, 0.5, 0.5], [0.7, 0.3, 0.9],
         [0.3, 0.8, 0.2], [0.6, 0.6, 0.7]],
        dtype=jnp.float32,
    )

@pytest.fixture
def xy_ctx_pts():
    """(x, y, u_t) — 2 spatial + 1 context."""
    return jnp.array(
        [[0.2, 0.2, 0.5], [0.5, 0.5, 0.1], [0.7, 0.3, 0.8],
         [0.3, 0.8, 0.3], [0.6, 0.6, 0.9]],
        dtype=jnp.float32,
    )


# ===========================================================================
# Normalize
# ===========================================================================

class TestNormalize:

    def test_spatial_only_range(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        # x in [0,1] → should map to [-1, 1]
        assert out.shape == (BATCH, 2)
        assert float(out.min()) >= -1.0 - 1e-5
        assert float(out.max()) <=  1.0 + 1e-5

    def test_spacetime_range(self, spacetime_domain, xyt_pts):
        net = Network(spacetime_domain, output_dim=1)
        net.add(Normalize())
        params = net.init(RNG)
        out = net.apply(params, xyt_pts)
        assert out.shape == (BATCH, 3)
        assert float(out.min()) >= -1.0 - 1e-5
        assert float(out.max()) <=  1.0 + 1e-5

    def test_context_passthrough_no_range(self, mesh_domain, xy_ctx_pts):
        """Context columns with no context_range pass through unchanged."""
        net = Network(mesh_domain, output_dim=1, n_context=1)
        net.add(Normalize())
        params = net.init(RNG)
        out = net.apply(params, xy_ctx_pts)
        # Last column (context) should equal input context column unchanged
        np.testing.assert_allclose(
            np.array(out[:, -1]), np.array(xy_ctx_pts[:, -1]), rtol=1e-5
        )

    def test_context_scaled_with_range(self, mesh_domain, xy_ctx_pts):
        """Context columns are rescaled when context_range is provided."""
        net = Network(mesh_domain, output_dim=1, n_context=1,
                      context_range=[(0.0, 1.0)])
        net.add(Normalize())
        params = net.init(RNG)
        out = net.apply(params, xy_ctx_pts)
        assert float(out[:, -1].min()) >= -1.0 - 1e-5
        assert float(out[:, -1].max()) <=  1.0 + 1e-5


# ===========================================================================
# Denormalize
# ===========================================================================

class TestDenormalize:

    def test_rescales_output(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1, output_range=(0.0, 2.0))
        net.add(Normalize())
        net.add(FNN([32]))
        net.add(Denormalize())
        params = net.init(RNG)
        xy = jnp.ones((BATCH, 2)) * 0.5
        out = net.apply(params, xy)
        assert out.shape == (BATCH, 1)
        # With tanh output from FNN and denorm to [0,2], values should be in range
        assert float(out.min()) >= 0.0 - 0.1
        assert float(out.max()) <= 2.0 + 0.1

    def test_noop_without_output_range(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(Denormalize())
        params = net.init(RNG)
        x = jnp.ones((BATCH, 2)) * 0.3
        out = net.apply(params, x)
        # No-op: input == output
        np.testing.assert_allclose(np.array(out), np.array(x), rtol=1e-6)


# ===========================================================================
# FNN (composable)
# ===========================================================================

class TestFNNLayer:

    def test_dimension_injection(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        fnn = FNN([32, 32])
        net.add(fnn)
        assert fnn._layer_sizes == [2, 32, 32, 1]

    def test_forward_shape(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=3)
        net.add(Normalize())
        net.add(FNN([64]))
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 3)

    def test_spacetime_dim_injection(self, spacetime_domain, xyt_pts):
        net = Network(spacetime_domain, output_dim=1)
        net.add(Normalize())
        fnn = FNN([32])
        net.add(fnn)
        # input_dim = 3 (x, y, t)
        assert fnn._layer_sizes[0] == 3
        params = net.init(RNG)
        out = net.apply(params, xyt_pts)
        assert out.shape == (BATCH, 1)


# ===========================================================================
# Full Network composition
# ===========================================================================

class TestNetworkComposition:

    def test_normalize_fnn_denormalize(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=1, output_range=(0.0, 1.0))
        net.add(Normalize())
        net.add(FNN([32]))
        net.add(Denormalize())
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)

    def test_fourier_encoding(self, mesh_domain, xy_pts):
        enc = FourierFeatures(domain=mesh_domain, n_features=8, encode_time=False)
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(enc)
        fnn = FNN([32])
        net.add(fnn)
        # After Normalize: dim=2; FourierFeatures(n_features=8): dim=16
        assert fnn._layer_sizes[0] == 16
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)

    def test_gnn_features_composable(self, mesh_domain, xy_pts):
        gnn = GNNFeatures(hidden_dim=16)
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(gnn)
        fnn = FNN([32])
        net.add(fnn)
        # GNN output_dim = 16 (no time, no context)
        assert fnn._layer_sizes[0] == 16
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)

    def test_laplacian_features_composable(self, mesh_domain, xy_pts):
        lap = LaplacianFeatures(n_features=4)
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(lap)
        fnn = FNN([32])
        net.add(fnn)
        # Laplacian output_dim = 4 (spatial only)
        assert fnn._layer_sizes[0] == 4
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)

    def test_repr(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1, output_range=(0.0, 1.0))
        net.add(Normalize())
        net.add(FNN([32]))
        net.add(Denormalize())
        r = repr(net)
        assert "Network" in r
        assert "Normalize" in r
        assert "FNN" in r
        assert "Denormalize" in r


# ===========================================================================
# Context fields in Network
# ===========================================================================

class TestNetworkContext:

    def test_context_passthrough_dim(self, mesh_domain, xy_ctx_pts):
        """n_context=1: initial dim = 2 + 1 = 3."""
        net = Network(mesh_domain, output_dim=1, n_context=1)
        fnn = FNN([32])
        net.add(Normalize())
        net.add(fnn)
        # input_dim to FNN should be 3
        assert fnn._layer_sizes[0] == 3

    def test_gnn_with_context(self, mesh_domain, xy_ctx_pts):
        gnn = GNNFeatures(hidden_dim=16, n_context=1)
        net = Network(mesh_domain, output_dim=1, n_context=1)
        net.add(Normalize())
        net.add(gnn)
        fnn = FNN([32])
        net.add(fnn)
        # gnn.output_dim = 16 + 1 (context skip) = 17
        assert gnn.output_dim == 17
        assert fnn._layer_sizes[0] == 17
        params = net.init(RNG)
        out = net.apply(params, xy_ctx_pts)
        assert out.shape == (BATCH, 1)

    def test_context_range_normalizes_context(self, mesh_domain, xy_ctx_pts):
        net = Network(mesh_domain, output_dim=1, n_context=1,
                      context_range=[(0.0, 1.0)])
        net.add(Normalize())
        fnn = FNN([32])
        net.add(fnn)
        params = net.init(RNG)
        out = net.apply(params, xy_ctx_pts)
        assert out.shape == (BATCH, 1)


# ===========================================================================
# GNN gets normalised mesh coordinates
# ===========================================================================

class TestNormalizeMeshCoordinates:

    def test_gnn_mesh_nodes_are_normalized(self, mesh_domain):
        """After Normalize, the GNN should build its mesh in [-1,1]^2."""
        gnn = GNNFeatures(hidden_dim=16)
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(gnn)

        # Mesh nodes in [0,1]^2 → after normalize should be in [-1,1]^2
        nodes = gnn._nodes_np
        assert float(nodes.min()) >= -1.0 - 1e-5
        assert float(nodes.max()) <=  1.0 + 1e-5

    def test_laplacian_mesh_nodes_are_normalized(self, mesh_domain):
        lap = LaplacianFeatures(n_features=4)
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(lap)

        nodes = lap._nodes_np
        assert float(nodes.min()) >= -1.0 - 1e-5
        assert float(nodes.max()) <=  1.0 + 1e-5


# ===========================================================================
# Dimension propagation through the stack
# ===========================================================================

class TestDimensionPropagation:
    """Check _current_dim is correctly updated after every add() call."""

    def test_initial_dim_spatial(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        assert net._current_dim == 2

    def test_initial_dim_spacetime(self, spacetime_domain):
        net = Network(spacetime_domain, output_dim=1)
        assert net._current_dim == 3

    def test_initial_dim_with_context(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1, n_context=2)
        assert net._current_dim == 4

    def test_normalize_preserves_dim(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        assert net._current_dim == 2

    def test_fourier_doubles_dim(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(FourierFeatures(domain=mesh_domain, n_features=8, encode_time=False))
        assert net._current_dim == 16  # 2*8

    def test_gnn_sets_hidden_dim(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(GNNFeatures(hidden_dim=24))
        assert net._current_dim == 24

    def test_laplacian_sets_n_features(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(LaplacianFeatures(n_features=6))
        assert net._current_dim == 6

    def test_fnn_sets_output_dim(self, mesh_domain):
        net = Network(mesh_domain, output_dim=3)
        net.add(Normalize())
        net.add(FNN([64, 64]))
        assert net._current_dim == 3

    def test_resnet_sets_output_dim(self, mesh_domain):
        net = Network(mesh_domain, output_dim=2)
        net.add(Normalize())
        net.add(ResNet(hidden_dim=32))
        assert net._current_dim == 2

    def test_piratenet_sets_output_dim(self, mesh_domain):
        net = Network(mesh_domain, output_dim=4)
        net.add(Normalize())
        net.add(PirateNet(hidden_dim=32))
        assert net._current_dim == 4

    def test_denormalize_preserves_dim(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1, output_range=(0.0, 1.0))
        net.add(Normalize())
        net.add(FNN([32]))
        net.add(Denormalize())
        assert net._current_dim == 1

    def test_chain_dim_progression(self, mesh_domain):
        """Step-by-step dim check across a full pipeline."""
        net = Network(mesh_domain, output_dim=2)
        assert net._current_dim == 2

        net.add(Normalize())
        assert net._current_dim == 2

        net.add(FourierFeatures(domain=mesh_domain, n_features=4, encode_time=False))
        assert net._current_dim == 8  # 2*4

        fnn = FNN([64])
        net.add(fnn)
        assert fnn._layer_sizes == [8, 64, 2]
        assert net._current_dim == 2


# ===========================================================================
# Params dict structure
# ===========================================================================

class TestParamsStructure:

    def test_no_param_layers_absent_from_params(self, mesh_domain):
        """Normalize and Denormalize must NOT create keys in params."""
        net = Network(mesh_domain, output_dim=1, output_range=(0.0, 1.0))
        net.add(Normalize())
        net.add(FNN([32]))
        net.add(Denormalize())
        params = net.init(RNG)
        keys = list(params.keys())
        # Only FNN should be present
        assert len(keys) == 1
        assert keys[0].startswith("FNN_")

    def test_multiple_fnn_layers_get_unique_keys(self, mesh_domain):
        net = Network(mesh_domain, output_dim=8)
        net.add(FNN([32]))     # outputs 8
        # Can't stack two terminal FNNs normally; use Fourier first
        enc = FourierFeatures(domain=mesh_domain, n_features=4, encode_time=False)
        net2 = Network(mesh_domain, output_dim=1)
        net2.add(Normalize())
        net2.add(enc)
        net2.add(FNN([32]))
        params = net2.init(RNG)
        assert "FNN_2" in params  # Normalize_0, FourierFeatures_1, FNN_2

    def test_gnn_creates_param_key(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(GNNFeatures(hidden_dim=16))
        net.add(FNN([32]))
        params = net.init(RNG)
        assert any(k.startswith("GNNFeatures_") for k in params)

    def test_laplacian_no_params(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(LaplacianFeatures(n_features=4))
        net.add(FNN([32]))
        params = net.init(RNG)
        # LaplacianFeatures has no trainable params
        assert not any(k.startswith("LaplacianFeatures_") for k in params)


# ===========================================================================
# WFFNN, ResNet, PirateNet integration
# ===========================================================================

class TestAlternativeNetworks:

    def test_wffnn_forward_shape(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(WFFNN([32, 32]))
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)

    def test_wffnn_dim_injection(self, mesh_domain):
        net = Network(mesh_domain, output_dim=3)
        net.add(Normalize())
        wffnn = WFFNN([64])
        net.add(wffnn)
        assert wffnn._layer_sizes[0] == 2
        assert wffnn._layer_sizes[-1] == 3

    def test_resnet_forward_shape(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=2)
        net.add(Normalize())
        net.add(ResNet(hidden_dim=32, n_blocks=3))
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 2)

    def test_resnet_spacetime(self, spacetime_domain, xyt_pts):
        net = Network(spacetime_domain, output_dim=1)
        net.add(Normalize())
        rnet = ResNet(hidden_dim=16)
        net.add(rnet)
        params = net.init(RNG)
        out = net.apply(params, xyt_pts)
        assert out.shape == (BATCH, 1)

    def test_piratenet_forward_shape(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(PirateNet(hidden_dim=32, n_blocks=2))
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)

    def test_piratenet_multi_output(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=3)
        net.add(Normalize())
        net.add(PirateNet(hidden_dim=32))
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 3)

    def test_fourier_then_resnet(self, mesh_domain, xy_pts):
        enc = FourierFeatures(domain=mesh_domain, n_features=8, encode_time=False)
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(enc)
        rnet = ResNet(hidden_dim=32)
        net.add(rnet)
        # ResNet input_dim must match FourierFeatures output (16)
        assert rnet._input_dim == 16
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)

    def test_gnn_then_piratenet(self, mesh_domain, xy_pts):
        gnn = GNNFeatures(hidden_dim=16)
        net = Network(mesh_domain, output_dim=2)
        net.add(Normalize())
        net.add(gnn)
        pnet = PirateNet(hidden_dim=32)
        net.add(pnet)
        assert pnet._input_dim == 16
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 2)
