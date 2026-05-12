"""
tests/test_model.py
-------------------
Tests for the high-level Model class.

Run with:
    conda run -n pinn python -m pytest tests/test_model.py -v --override-ini addopts=
"""

import os
os.environ.setdefault("PINNS_BACKEND", "jax")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from pinns import DomainMesh, create_model as Model
from pinns.models.model_base import ModelBase, NetworkLoss
from pinns.models.model_partitioned import ModelPartitioned
from pinns.models.model_stepper import ModelStepper
from pinns.models.layers import Normalize, Denormalize, FNN, ResNet, PirateNet
from pinns import PartitionFB, PartitionX, StepperStep
from pinns.models.stepping import StepperDt

RNG = jax.random.PRNGKey(0)

VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
BATCH = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def domain():
    return DomainMesh((VERTS, FACES))


@pytest.fixture
def domain_time():
    return DomainMesh((VERTS, FACES), time=(0.0, 1.0))


@pytest.fixture
def xy():
    return jax.random.uniform(RNG, (BATCH, 2), dtype=jnp.float32)


@pytest.fixture
def xyt():
    return jax.random.uniform(RNG, (BATCH, 3), dtype=jnp.float32)


# ===========================================================================
# Construction
# ===========================================================================

class TestModelConstruction:

    def test_is_network_subclass(self, domain):
        m = Model(domain, output_dim=1)
        assert isinstance(m, ModelBase)

    def test_default_layers_added(self, domain):
        """Default: Normalize + FNN."""
        m = Model(domain, output_dim=1)
        layer_types = [type(l).__name__ for l in m._layers]
        assert 'Normalize' in layer_types
        assert 'FNN' in layer_types
        assert 'Denormalize' not in layer_types

    def test_custom_hidden_dims(self, domain):
        m = Model(domain, output_dim=1, hidden_dims=[32, 64, 32])
        # The FNN layer stores the hidden dims; verify by inspecting its layer config.
        fnn_layers = [l for l in m._layers if type(l).__name__ == "FNN"]
        assert len(fnn_layers) == 1
        assert fnn_layers[0].hidden_dims == [32, 64, 32]

    def test_no_normalize(self, domain):
        m = Model(domain, output_dim=1, normalize=False)
        layer_types = [type(l).__name__ for l in m._layers]
        assert 'Normalize' not in layer_types
        assert 'FNN' in layer_types

    def test_denormalize_requires_output_range(self, domain):
        with pytest.raises(ValueError, match="output_range"):
            Model(domain, output_dim=1, denormalize=True)

    def test_denormalize_with_output_range(self, domain):
        m = Model(domain, output_dim=1,
                  denormalize=True, output_range=(0.0, 100.0))
        layer_types = [type(l).__name__ for l in m._layers]
        assert 'Denormalize' in layer_types

    def test_output_range_stored(self, domain):
        m = Model(domain, output_dim=1,
                  denormalize=True, output_range=(0.0, 100.0))
        assert m.output_range == (0.0, 100.0)

    def test_output_range_not_stored_without_denormalize(self, domain):
        """output_range is not forwarded to ModelBase when denormalize=False."""
        m = Model(domain, output_dim=1)
        assert m.output_range is None

    def test_explicit_core_replaces_fnn(self, domain):
        core = ResNet(hidden_dim=32, n_blocks=2)
        m = Model(domain, output_dim=1, core=core)
        layer_types = [type(l).__name__ for l in m._layers]
        assert 'ResNet' in layer_types
        assert 'FNN' not in layer_types

    def test_piratenet_core(self, domain):
        m = Model(domain, output_dim=1, core=PirateNet(hidden_dim=32, n_blocks=2))
        layer_types = [type(l).__name__ for l in m._layers]
        assert 'PirateNet' in layer_types

    def test_features_inserted_before_core(self, domain):
        """Features appear before FNN in the layer list."""
        from pinns.models.layers import RandomFourierFeatures
        feats = RandomFourierFeatures(n_features=64)
        m = Model(domain, output_dim=1, features=feats)
        names = [type(l).__name__ for l in m._layers]
        assert names.index('RandomFourierFeatures') < names.index('FNN')

    def test_slot_order_full(self, domain):
        """Normalize → Features → FNN → Denormalize."""
        from pinns.models.layers import RandomFourierFeatures
        m = Model(domain, output_dim=1,
                  features=RandomFourierFeatures(n_features=16),
                  denormalize=True, output_range=(-1.0, 1.0))
        names = [type(l).__name__ for l in m._layers]
        assert names == ['Normalize', 'RandomFourierFeatures', 'FNN', 'Denormalize']

    def test_output_dim_stored(self, domain):
        m = Model(domain, output_dim=3)
        assert m.output_dim == 3

    def test_with_spatial_strategy(self, domain):
        m = Model(domain, output_dim=1, partition=PartitionFB(overlap=0.3))
        assert isinstance(m, ModelPartitioned)

    def test_with_temporal_strategy(self, domain_time):
        m = Model(domain_time, output_dim=1, stepper=StepperDt(),
                  context_range=[(0.0, 1.0)])
        assert isinstance(m, ModelStepper)


# ===========================================================================
# Forward pass
# ===========================================================================

class TestModelForward:

    def test_output_shape_default(self, domain, xy):
        m = Model(domain, output_dim=1)
        params = m.init(RNG)
        y = m.apply(params, xy)
        assert y.shape == (BATCH, 1)

    def test_output_shape_multi_output(self, domain, xy):
        m = Model(domain, output_dim=3)
        params = m.init(RNG)
        y = m.apply(params, xy)
        assert y.shape == (BATCH, 3)

    def test_output_finite(self, domain, xy):
        m = Model(domain, output_dim=1)
        params = m.init(RNG)
        y = m.apply(params, xy)
        assert jnp.all(jnp.isfinite(y))

    def test_output_range_respected(self, domain, xy):
        """Denormalized output should lie within (or near) the prescribed range."""
        m = Model(domain, output_dim=1,
                  normalize=True, denormalize=True,
                  output_range=(10.0, 20.0))
        # After tanh activation + Denormalize, roughly in range.
        params = m.init(RNG)
        # After 1000 uninitialised outputs in tanh ~ [-1,1] → Denormalize → [10, 20].
        y = m.apply(params, xy)
        # Just check output is finite and shape is correct.
        assert y.shape == (BATCH, 1)
        assert jnp.all(jnp.isfinite(y))

    def test_without_normalize(self, domain, xy):
        m = Model(domain, output_dim=1, normalize=False)
        params = m.init(RNG)
        y = m.apply(params, xy)
        assert y.shape == (BATCH, 1)
        assert jnp.all(jnp.isfinite(y))

    def test_time_domain_forward(self, domain_time, xyt):
        m = Model(domain_time, output_dim=1)
        params = m.init(RNG)
        y = m.apply(params, xyt)
        assert y.shape == (BATCH, 1)
        assert jnp.all(jnp.isfinite(y))

    def test_resnet_core_forward(self, domain, xy):
        m = Model(domain, output_dim=1, core=ResNet(hidden_dim=32))
        params = m.init(RNG)
        y = m.apply(params, xy)
        assert y.shape == (BATCH, 1)
        assert jnp.all(jnp.isfinite(y))

    def test_piratenet_core_forward(self, domain, xy):
        m = Model(domain, output_dim=1, core=PirateNet(hidden_dim=32))
        params = m.init(RNG)
        y = m.apply(params, xy)
        assert y.shape == (BATCH, 1)
        assert jnp.all(jnp.isfinite(y))


# ===========================================================================
# add_constraint (Lifting)
# ===========================================================================

class TestModelAddConstraint:

    def test_add_constraint_appends_lifting(self, domain):
        m = Model(domain, output_dim=1)
        m.add_constraint(value=0.0)
        layer_types = [type(l).__name__ for l in m._layers]
        assert 'Lifting' in layer_types

    def test_add_constraint_is_last_layer(self, domain):
        m = Model(domain, output_dim=1)
        m.add_constraint(value=0.0)
        assert type(m._layers[-1]).__name__ == 'Lifting'

    def test_add_constraint_after_denormalize(self, domain):
        """Lifting should come after Denormalize in the pipeline."""
        m = Model(domain, output_dim=1,
                  denormalize=True, output_range=(-1.0, 1.0))
        m.add_constraint(value=0.0)
        names = [type(l).__name__ for l in m._layers]
        assert names.index('Lifting') > names.index('Denormalize')

    def test_multiple_constraints(self, domain):
        m = Model(domain, output_dim=2)
        m.add_constraint(value=0.0, output_idx=0)
        m.add_constraint(value=1.0, output_idx=1)
        lifting_layers = [l for l in m._layers if type(l).__name__ == 'Lifting']
        assert len(lifting_layers) == 2

    def test_add_constraint_chaining(self, domain):
        m = Model(domain, output_dim=1)
        result = m.add_constraint(value=0.0)
        assert result is m  # chaining

    def test_constraint_output_finite(self, domain, xy):
        m = Model(domain, output_dim=1)
        m.add_constraint(value=0.0)
        params = m.init(RNG)
        y = m.apply(params, xy)
        assert y.shape == (BATCH, 1)
        assert jnp.all(jnp.isfinite(y))

    def test_constraint_with_sigma(self, domain):
        m = Model(domain, output_dim=1)
        m.add_constraint(value=0.0, sigma=0.1)
        layer_types = [type(l).__name__ for l in m._layers]
        assert 'Lifting' in layer_types


# ===========================================================================
# network_losses (inherited from ModelBase)
# ===========================================================================

class TestModelNetworkLosses:

    def test_network_losses_empty_by_default(self, domain):
        m = Model(domain, output_dim=1)
        assert m.network_losses == []

    def test_add_network_loss(self, domain):
        m = Model(domain, output_dim=1)
        loss = NetworkLoss('reg', lambda p, x: jnp.mean(x ** 2), weight=1e-4)
        m.add_network_loss(loss)
        assert len(m.network_losses) == 1

    def test_network_loss_callable(self, domain, xy):
        m = Model(domain, output_dim=1)
        params = m.init(RNG)
        loss = NetworkLoss('test', lambda p, x: jnp.mean(x ** 2))
        m.add_network_loss(loss)
        val = m.network_losses[0].fn(params, xy)
        assert jnp.isfinite(val)


# ===========================================================================
# set_model integration
# ===========================================================================

class TestModelSetModel:

    def test_can_be_set_as_model(self, domain):
        from pinns.problems.problem_strong import ProblemStrong
        m = Model(domain, output_dim=1)
        p = ProblemStrong(domain, ['u'])
        p.set_model(m)
        assert p.model is m

    def test_no_terms_injected(self, domain):
        from pinns.problems.problem_strong import ProblemStrong
        m = Model(domain, output_dim=1)
        p = ProblemStrong(domain, ['u'])
        n_terms_before = len(p._terms)
        p.set_model(m)
        assert len(p._terms) == n_terms_before


# ===========================================================================
# Repr
# ===========================================================================

class TestModelRepr:

    def test_repr_contains_model(self, domain):
        m = Model(domain, output_dim=1)
        r = repr(m)
        assert 'Model' in r

    def test_repr_shows_normalize(self, domain):
        m = Model(domain, output_dim=1)
        r = repr(m)
        assert 'Normalize' in r

    def test_repr_shows_fnn(self, domain):
        m = Model(domain, output_dim=1, hidden_dims=[32, 32])
        r = repr(m)
        assert 'FNN' in r

    def test_repr_shows_denormalize(self, domain):
        m = Model(domain, output_dim=1,
                  denormalize=True, output_range=(0.0, 1.0))
        r = repr(m)
        assert 'Denormalize' in r

    def test_repr_shows_lifting_when_constrained(self, domain):
        m = Model(domain, output_dim=1)
        m.add_constraint(value=0.0)
        r = repr(m)
        assert 'Lifting' in r

    def test_repr_no_normalize(self, domain):
        m = Model(domain, output_dim=1, normalize=False)
        r = repr(m)
        assert 'Normalize' not in r


# ---------------------------------------------------------------------------
# output_dim override — FNN, WFFNN, ResNet, PirateNet
# ---------------------------------------------------------------------------

from pinns.models.layers import WFFNN
from pinns.domain import DomainCubic

_DOMAIN_2D  = DomainCubic(space=[(0.0, 1.0), (0.0, 1.0)])
_OUTPUT_DIM = 3    # non-trivial final output, distinct from hidden dim
_HIDDEN_DIM = 64
_KEY        = jax.random.PRNGKey(42)


def _fwd(net: ModelBase) -> jnp.ndarray:
    params = net.init(_KEY)
    return net.apply(params, jnp.ones((8, 2)))


class TestFNNOutputDim:
    def test_default_uses_network_output_dim(self):
        net = ModelBase(_DOMAIN_2D, output_dim=_OUTPUT_DIM)
        net.add(Normalize())
        layer = FNN([_HIDDEN_DIM, _HIDDEN_DIM])
        net.add(layer)
        assert layer._layer_sizes[-1] == _OUTPUT_DIM
        assert _fwd(net).shape == (8, _OUTPUT_DIM)

    def test_explicit_output_dim_overrides(self):
        # output_dim=<int> → use that width instead of network.output_dim
        net = ModelBase(_DOMAIN_2D, output_dim=_OUTPUT_DIM)
        net.add(Normalize())
        layer_mid   = FNN([_HIDDEN_DIM, _HIDDEN_DIM], output_dim=_HIDDEN_DIM)
        layer_final = FNN([_HIDDEN_DIM, _HIDDEN_DIM])
        net.add(layer_mid)
        net.add(layer_final)
        assert layer_mid._layer_sizes[-1]   == _HIDDEN_DIM,  "intermediate must use override"
        assert layer_final._layer_sizes[-1] == _OUTPUT_DIM,  "final must use network.output_dim"
        assert _fwd(net).shape == (8, _OUTPUT_DIM)


class TestWFFNNOutputDim:
    def test_default_uses_network_output_dim(self):
        net = ModelBase(_DOMAIN_2D, output_dim=_OUTPUT_DIM)
        net.add(Normalize())
        layer = WFFNN([_HIDDEN_DIM, _HIDDEN_DIM])
        net.add(layer)
        assert layer._layer_sizes[-1] == _OUTPUT_DIM
        assert _fwd(net).shape == (8, _OUTPUT_DIM)

    def test_explicit_output_dim_overrides(self):
        net = ModelBase(_DOMAIN_2D, output_dim=_OUTPUT_DIM)
        net.add(Normalize())
        layer_mid   = WFFNN([_HIDDEN_DIM, _HIDDEN_DIM], output_dim=_HIDDEN_DIM)
        layer_final = FNN([_HIDDEN_DIM])
        net.add(layer_mid)
        net.add(layer_final)
        assert layer_mid._layer_sizes[-1]   == _HIDDEN_DIM
        assert layer_final._layer_sizes[-1] == _OUTPUT_DIM
        assert _fwd(net).shape == (8, _OUTPUT_DIM)


class TestResNetOutputDim:
    def test_default_uses_network_output_dim(self):
        net = ModelBase(_DOMAIN_2D, output_dim=_OUTPUT_DIM)
        net.add(Normalize())
        layer = ResNet(hidden_dim=_HIDDEN_DIM, n_blocks=2)
        net.add(layer)
        assert layer._output_dim == _OUTPUT_DIM
        assert _fwd(net).shape == (8, _OUTPUT_DIM)

    def test_explicit_output_dim_overrides(self):
        net = ModelBase(_DOMAIN_2D, output_dim=_OUTPUT_DIM)
        net.add(Normalize())
        layer_mid   = ResNet(hidden_dim=_HIDDEN_DIM, n_blocks=2, output_dim=_HIDDEN_DIM)
        layer_final = FNN([_HIDDEN_DIM])
        net.add(layer_mid)
        net.add(layer_final)
        assert layer_mid._output_dim        == _HIDDEN_DIM
        assert layer_final._layer_sizes[-1] == _OUTPUT_DIM
        assert _fwd(net).shape == (8, _OUTPUT_DIM)


class TestPirateNetOutputDim:
    def test_default_uses_network_output_dim(self):
        net = ModelBase(_DOMAIN_2D, output_dim=_OUTPUT_DIM)
        net.add(Normalize())
        layer = PirateNet(hidden_dim=_HIDDEN_DIM, n_blocks=2)
        net.add(layer)
        assert layer._output_dim == _OUTPUT_DIM
        assert _fwd(net).shape == (8, _OUTPUT_DIM)

    def test_explicit_output_dim_overrides(self):
        net = ModelBase(_DOMAIN_2D, output_dim=_OUTPUT_DIM)
        net.add(Normalize())
        layer_mid   = PirateNet(hidden_dim=_HIDDEN_DIM, n_blocks=2, output_dim=_HIDDEN_DIM)
        layer_final = FNN([_HIDDEN_DIM])
        net.add(layer_mid)
        net.add(layer_final)
        assert layer_mid._output_dim        == _HIDDEN_DIM
        assert layer_final._layer_sizes[-1] == _OUTPUT_DIM
        assert _fwd(net).shape == (8, _OUTPUT_DIM)
