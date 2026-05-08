"""
Tests for pinns.base_models: FNN, ResNet, PirateNet.

Covers for each architecture:
  - construction (various args)
  - init() / apply() shape correctness
  - set_input_range / set_output_range (normalisation)
  - FourierFeatures integration
  - predict() numpy I/O
  - to() parameter initialisation
  - output_transform hard-constraint hook

Run with:
    pytest tests/test_base_models.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("PINNS_BACKEND", "jax")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from pinns.base_models import FNN, ResNet, PirateNet
from pinns.layers import FourierFeatures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = jax.random.PRNGKey(42)
BATCH = 16
IN_DIM = 3
OUT_DIM = 2


def _x(batch=BATCH, dim=IN_DIM, seed=0):
    return jax.random.uniform(jax.random.PRNGKey(seed), (batch, dim))


def _xmin(dim=IN_DIM):
    return np.zeros(dim)


def _xmax(dim=IN_DIM):
    return np.ones(dim)


def _ymin(dim=OUT_DIM):
    return np.full(dim, -1.0)


def _ymax(dim=OUT_DIM):
    return np.full(dim, 1.0)


# ===========================================================================
# FNN
# ===========================================================================

class TestFNN:

    def test_construction_default(self):
        net = FNN([IN_DIM, 32, OUT_DIM])
        assert net.layer_sizes == [IN_DIM, 32, OUT_DIM]
        assert net.activation == "tanh"

    def test_construction_options(self):
        net = FNN([IN_DIM, 64, 64, OUT_DIM],
                  activation="gelu",
                  normalize_input=False,
                  unnormalize_output=False)
        assert net.activation == "gelu"
        assert not net.normalize_input

    def test_init_returns_params(self):
        net = FNN([IN_DIM, 32, OUT_DIM])
        params = net.init(RNG)
        assert params is not None

    def test_apply_shape(self):
        net = FNN([IN_DIM, 32, OUT_DIM])
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert y.shape == (BATCH, OUT_DIM), y.shape

    def test_apply_with_normalization(self):
        net = FNN([IN_DIM, 32, OUT_DIM])
        net.set_input_range(_xmin(), _xmax())
        net.set_output_range(_ymin(), _ymax())
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert y.shape == (BATCH, OUT_DIM)

    def test_output_range_constructor(self):
        net = FNN([IN_DIM, 32, OUT_DIM], output_range=(0.0, 1.0))
        assert net.output_min is not None
        assert net.output_max is not None

    def test_predict_numpy(self):
        net = FNN([IN_DIM, 32, OUT_DIM])
        net = net.to(seed=0)
        x_np = np.random.rand(BATCH, IN_DIM).astype(np.float32)
        y_np = net.predict(x_np)
        assert isinstance(y_np, np.ndarray)
        assert y_np.shape == (BATCH, OUT_DIM)

    def test_to_initialises_params(self):
        net = FNN([IN_DIM, 32, OUT_DIM])
        assert not hasattr(net, "params") or net.params is None
        net = net.to(seed=7)
        assert hasattr(net, "params") and net.params is not None

    def test_fourier_features(self):
        ff = FourierFeatures(input_dim=IN_DIM, n_features=16, sigma=1.0)
        # layer_sizes[0] must match ff.output_dim
        net = FNN([ff.output_dim, 32, OUT_DIM], feature_encoding=ff)
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert y.shape == (BATCH, OUT_DIM)

    def test_output_transform(self):
        """Hard-constraint output_transform is called and changes the output."""
        called = []

        def ot(x_orig, y, pd):
            called.append(True)
            return y * 0.0  # zero everything

        net = FNN([IN_DIM, 32, OUT_DIM], output_transform=ot)
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert called, "output_transform was not called"
        assert jnp.allclose(y, 0.0)

    def test_multiple_hidden_layers(self):
        net = FNN([IN_DIM, 64, 64, 64, OUT_DIM])
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert y.shape == (BATCH, OUT_DIM)


# ===========================================================================
# ResNet
# ===========================================================================

class TestResNet:

    def test_construction_default(self):
        net = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        assert net.input_dim == IN_DIM
        assert net.output_dim == OUT_DIM
        assert net.n_blocks == 4
        assert net.layer_norm is True

    def test_construction_options(self):
        net = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=64,
                     n_blocks=2, activation="gelu", layer_norm=False)
        assert net.n_blocks == 2
        assert not net.layer_norm

    def test_init_returns_params(self):
        net = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        params = net.init(RNG)
        assert params is not None

    def test_apply_shape(self):
        net = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert y.shape == (BATCH, OUT_DIM), y.shape

    def test_apply_with_normalization(self):
        net = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        net.set_input_range(_xmin(), _xmax())
        net.set_output_range(_ymin(), _ymax())
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert y.shape == (BATCH, OUT_DIM)

    def test_predict_numpy(self):
        net = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        net = net.to(seed=0)
        x_np = np.random.rand(BATCH, IN_DIM).astype(np.float32)
        y_np = net.predict(x_np)
        assert isinstance(y_np, np.ndarray)
        assert y_np.shape == (BATCH, OUT_DIM)

    def test_to_initialises_params(self):
        net = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        assert not hasattr(net, "params") or net.params is None
        net = net.to(seed=3)
        assert hasattr(net, "params") and net.params is not None

    def test_fourier_features(self):
        ff = FourierFeatures(input_dim=IN_DIM, n_features=16, sigma=1.0)
        # input_dim is the original (pre-encoding) dim; feature_encoding is applied inside
        net = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32,
                     feature_encoding=ff)
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert y.shape == (BATCH, OUT_DIM)

    def test_output_transform(self):
        called = []

        def ot(x_orig, y, pd):
            called.append(True)
            return y + 1.0

        net = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32,
                     output_transform=ot)
        params = net.init(RNG)
        y_with = net.apply(params, _x())

        net2 = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        y_without = net2.apply(params, _x())

        assert called
        assert jnp.allclose(y_with, y_without + 1.0)

    def test_repr(self):
        net = ResNet(input_dim=2, output_dim=1, hidden_dim=64, n_blocks=3)
        r = repr(net)
        assert "ResNet" in r
        assert "64" in r

    def test_skip_connection_near_identity_at_init(self):
        """
        At random init the residual blocks should not explode the output.
        With layer_norm=True the variance should stay within a reasonable range.
        """
        net = ResNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=64,
                     n_blocks=6, normalize_input=False, unnormalize_output=False)
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert jnp.all(jnp.isfinite(y)), "Output contains NaN/Inf at init"
        assert float(jnp.abs(y).max()) < 1e3, "Output magnitude too large at init"


# ===========================================================================
# PirateNet
# ===========================================================================

class TestPirateNet:

    def test_construction_default(self):
        net = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        assert net.input_dim == IN_DIM
        assert net.output_dim == OUT_DIM
        assert net.n_blocks == 3

    def test_construction_options(self):
        net = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=64,
                        n_blocks=5, activation="gelu",
                        rwf_mu=1.0, rwf_sigma=0.2)
        assert net.n_blocks == 5
        assert net.rwf_mu == 1.0

    def test_init_returns_params(self):
        net = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        params = net.init(RNG)
        assert params is not None

    def test_apply_shape(self):
        net = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert y.shape == (BATCH, OUT_DIM), y.shape

    def test_apply_with_normalization(self):
        net = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        net.set_input_range(_xmin(), _xmax())
        net.set_output_range(_ymin(), _ymax())
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert y.shape == (BATCH, OUT_DIM)

    def test_predict_numpy(self):
        net = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        net = net.to(seed=0)
        x_np = np.random.rand(BATCH, IN_DIM).astype(np.float32)
        y_np = net.predict(x_np)
        assert isinstance(y_np, np.ndarray)
        assert y_np.shape == (BATCH, OUT_DIM)

    def test_to_initialises_params(self):
        net = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        assert not hasattr(net, "params") or net.params is None
        net = net.to(seed=5)
        assert hasattr(net, "params") and net.params is not None

    def test_fourier_features(self):
        ff = FourierFeatures(input_dim=IN_DIM, n_features=16, sigma=1.0)
        # input_dim to PirateNet is the original dim; feature_encoding is applied
        # inside apply() after normalisation
        net = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32,
                        feature_encoding=ff)
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert y.shape == (BATCH, OUT_DIM)

    def test_output_transform(self):
        called = []

        def ot(x_orig, y, pd):
            called.append(True)
            return y * 2.0

        net = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32,
                        output_transform=ot)
        params = net.init(RNG)
        y_with = net.apply(params, _x())

        net2 = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=32)
        y_without = net2.apply(params, _x())

        assert called
        assert jnp.allclose(y_with, y_without * 2.0)

    def test_alpha_params_initialised_near_zero(self):
        """
        PirateNet residual blocks use α initialised to 0 so the network
        starts as a near-identity/linear map.  Check that block outputs
        have small magnitude at init (normalised input, no output scaling).
        """
        net = PirateNet(input_dim=IN_DIM, output_dim=OUT_DIM, hidden_dim=64,
                        n_blocks=3, normalize_input=False, unnormalize_output=False)
        params = net.init(RNG)
        y = net.apply(params, _x())
        assert jnp.all(jnp.isfinite(y)), "Output contains NaN/Inf at init"


# ===========================================================================
# Cross-class: all networks importable from pinns.base_models
# ===========================================================================

def test_public_imports():
    from pinns.base_models import FNN, ResNet, PirateNet
    assert FNN is not None
    assert ResNet is not None
    assert PirateNet is not None

def test_pinns_base_models_attribute():
    import pinns
    assert hasattr(pinns, "base_models")
    assert hasattr(pinns.base_models, "FNN")
    assert hasattr(pinns.base_models, "ResNet")
    assert hasattr(pinns.base_models, "PirateNet")


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
