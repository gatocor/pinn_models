"""
tests/test_network_strategies.py
---------------------------------
Integration tests for Network strategies:
  - Strategy storage and defaults
  - _setup() / set_model() wiring to domain
  - StrategyStep time-grid initialisation
  - Forward pass (predict) works correctly after set_model
  - problem.strategy / problem.stepper properties

Run with:
    conda run -n pinn python -m pytest tests/test_network_strategies.py -v --override-ini addopts=
"""

import os
os.environ.setdefault("PINNS_BACKEND", "jax")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from pinns import DomainCubic, DomainMesh
from pinns.network import Network
from pinns.layers import Normalize, FNN, Denormalize
from pinns.strategies import (
    StrategyUnique, StrategyFB, StrategyX, StrategyStep,
)
from pinns.problem import ProblemStrong

RNG = jax.random.PRNGKey(0)
VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
BATCH = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mesh_domain():
    return DomainMesh((VERTS, FACES))


@pytest.fixture
def spacetime_domain():
    return DomainMesh((VERTS, FACES), time=(0.0, 1.0))


@pytest.fixture
def stepped_domain():
    """Domain with a partitioned time axis (required for StrategyStep)."""
    ts = np.linspace(0.0, 1.0, 6)         # 5 equal steps of dt=0.2
    return DomainMesh((VERTS, FACES), time=ts)


@pytest.fixture
def xy_pts():
    key = jax.random.PRNGKey(42)
    return jax.random.uniform(key, (BATCH, 2), dtype=jnp.float32)


@pytest.fixture
def xyt_pts():
    key = jax.random.PRNGKey(42)
    return jax.random.uniform(key, (BATCH, 3), dtype=jnp.float32)


# ===========================================================================
# Strategy defaults and storage
# ===========================================================================

class TestStrategyDefaults:

    def test_default_is_unique(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        assert isinstance(net.spatial, StrategyUnique)
        assert net.temporal is None

    def test_is_stepping_false_by_default(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        assert net.is_stepping is False
        assert net.stepper is None

    def test_set_spatial_fb(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1, spatial=StrategyFB(overlap=0.4))
        assert isinstance(net.spatial, StrategyFB)
        assert net.spatial.overlap == 0.4

    def test_set_spatial_x(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1,
                      spatial=StrategyX(interface_weight=5.0, flux_weight=1.0))
        assert isinstance(net.spatial, StrategyX)
        assert net.spatial.interface_weight == 5.0
        assert net.spatial.flux_weight == 1.0

    def test_set_temporal_step(self, stepped_domain):
        step = StrategyStep()
        net = Network(stepped_domain, output_dim=1, temporal=step)
        assert isinstance(net.temporal, StrategyStep)
        assert net.is_stepping is True
        assert net.stepper is step

    def test_invalid_spatial_raises(self, mesh_domain):
        with pytest.raises(TypeError, match="spatial must be"):
            Network(mesh_domain, output_dim=1, spatial="bad")

    def test_invalid_temporal_raises(self, mesh_domain):
        with pytest.raises(TypeError, match="temporal must be"):
            Network(mesh_domain, output_dim=1, temporal=StrategyFB())

    def test_repr_default_no_strategy_shown(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(FNN([32]))
        r = repr(net)
        # StrategyUnique is the default — should not appear in repr
        assert "StrategyUnique" not in r
        assert "temporal" not in r

    def test_repr_non_default_strategy_shown(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1,
                      spatial=StrategyFB(overlap=0.3))
        r = repr(net)
        assert "StrategyFB" in r

    def test_repr_temporal_shown(self, stepped_domain):
        net = Network(stepped_domain, output_dim=1, temporal=StrategyStep())
        net._setup(stepped_domain)
        r = repr(net)
        assert "StrategyStep" in r


# ===========================================================================
# _setup() wires strategies to domain
# ===========================================================================

class TestSetup:

    def test_unique_setup_is_noop(self, mesh_domain):
        """StrategyUnique.setup() is a no-op — should not raise."""
        net = Network(mesh_domain, output_dim=1)
        net._setup(mesh_domain)  # should not raise

    def test_fb_setup_is_noop(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1, spatial=StrategyFB())
        net._setup(mesh_domain)

    def test_x_setup_is_noop(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1, spatial=StrategyX())
        net._setup(mesh_domain)

    def test_step_setup_reads_time_grid(self, stepped_domain):
        step = StrategyStep()
        net = Network(stepped_domain, output_dim=1, temporal=step)
        net._setup(stepped_domain)
        assert step.ts is not None
        np.testing.assert_allclose(step.ts, np.linspace(0.0, 1.0, 6))

    def test_step_n_steps(self, stepped_domain):
        step = StrategyStep()
        net = Network(stepped_domain, output_dim=1, temporal=step)
        net._setup(stepped_domain)
        assert step.n_steps == 5

    def test_step_dt(self, stepped_domain):
        step = StrategyStep()
        net = Network(stepped_domain, output_dim=1, temporal=step)
        net._setup(stepped_domain)
        assert step.is_uniform
        np.testing.assert_allclose(step.dt, 0.2, rtol=1e-5)

    def test_step_setup_requires_time_grid(self, mesh_domain):
        """StrategyStep should raise at Network construction if domain has no time grid."""
        step = StrategyStep()
        with pytest.raises(ValueError, match="partitioned time axis"):
            Network(mesh_domain, output_dim=1, temporal=step)

    def test_step_n_steps_override(self, stepped_domain):
        step = StrategyStep(n_steps=3)
        net = Network(stepped_domain, output_dim=1, temporal=step)
        net._setup(stepped_domain)
        # override: only 3 steps even though domain has 5
        assert step.n_steps == 3


# ===========================================================================
# set_model integration with ProblemStrong
# ===========================================================================

class TestSetModel:

    def test_set_model_accepts_network(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(FNN([32]))
        problem = ProblemStrong(mesh_domain, ['u'])
        problem.set_model(net)
        assert problem.model is net

    def test_set_model_wires_step_strategy(self, stepped_domain):
        step = StrategyStep()
        net = Network(stepped_domain, output_dim=1, temporal=step)
        net.add(FNN([32]))
        problem = ProblemStrong(stepped_domain, ['u'])
        problem.set_model(net)
        # After set_model, _setup was called — step should be initialised
        assert step.ts is not None
        assert step.n_steps == 5

    def test_problem_strategy_returns_temporal_when_set(self, stepped_domain):
        step = StrategyStep()
        net = Network(stepped_domain, output_dim=1, temporal=step)
        net.add(FNN([32]))
        problem = ProblemStrong(stepped_domain, ['u'])
        problem.set_model(net)
        assert problem.strategy is step

    def test_problem_strategy_returns_spatial_when_no_temporal(self, mesh_domain):
        fb = StrategyFB(overlap=0.4)
        net = Network(mesh_domain, output_dim=1, spatial=fb)
        net.add(FNN([32]))
        problem = ProblemStrong(mesh_domain, ['u'])
        problem.set_model(net)
        assert problem.strategy is fb

    def test_problem_stepper_none_without_temporal(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(FNN([32]))
        problem = ProblemStrong(mesh_domain, ['u'])
        problem.set_model(net)
        assert problem.stepper is None

    def test_problem_stepper_is_step_strategy(self, stepped_domain):
        step = StrategyStep()
        net = Network(stepped_domain, output_dim=1, temporal=step)
        net.add(FNN([32]))
        problem = ProblemStrong(stepped_domain, ['u'])
        problem.set_model(net)
        assert problem.stepper is step

    def test_set_model_rejects_non_network(self, mesh_domain):
        problem = ProblemStrong(mesh_domain, ['u'])
        with pytest.raises(TypeError, match="Network instance"):
            problem.set_model("not a network")

    def test_set_model_chainable(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        net.add(FNN([32]))
        problem = ProblemStrong(mesh_domain, ['u'])
        result = problem.set_model(net)
        assert result is problem


# ===========================================================================
# Forward pass (predict) — init + apply give correct outputs
# ===========================================================================

class TestNetworkPredict:

    def test_predict_default_strategy(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(FNN([32]))
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)
        assert jnp.all(jnp.isfinite(out))

    def test_predict_fb_strategy_same_forward(self, mesh_domain, xy_pts):
        """StrategyFB does not change forward pass — same output shape."""
        net = Network(mesh_domain, output_dim=1, spatial=StrategyFB(overlap=0.3))
        net.add(Normalize())
        net.add(FNN([32]))
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)
        assert jnp.all(jnp.isfinite(out))

    def test_predict_x_strategy_same_forward(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=1,
                      spatial=StrategyX(interface_weight=2.0))
        net.add(Normalize())
        net.add(FNN([32]))
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)
        assert jnp.all(jnp.isfinite(out))

    def test_predict_step_strategy_spacetime(self, stepped_domain, xyt_pts):
        step = StrategyStep()
        net = Network(stepped_domain, output_dim=1, temporal=step)
        net.add(Normalize())
        net.add(FNN([32]))
        net._setup(stepped_domain)
        params = net.init(RNG)
        out = net.apply(params, xyt_pts)
        assert out.shape == (BATCH, 1)
        assert jnp.all(jnp.isfinite(out))

    def test_predict_multi_output(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=3)
        net.add(Normalize())
        net.add(FNN([64, 64]))
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 3)
        assert jnp.all(jnp.isfinite(out))

    def test_predict_with_denormalize(self, mesh_domain, xy_pts):
        net = Network(mesh_domain, output_dim=1, output_range=(0.0, 5.0))
        net.add(Normalize())
        net.add(FNN([32]))
        net.add(Denormalize())
        params = net.init(RNG)
        out = net.apply(params, xy_pts)
        assert out.shape == (BATCH, 1)
        # tanh output in [-1,1] maps to [0, 5] — should stay in range
        assert float(out.min()) >= 0.0 - 0.1
        assert float(out.max()) <= 5.0 + 0.1

    def test_predict_deterministic(self, mesh_domain, xy_pts):
        """Same params → same output every time."""
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(FNN([32]))
        params = net.init(RNG)
        out1 = net.apply(params, xy_pts)
        out2 = net.apply(params, xy_pts)
        np.testing.assert_array_equal(np.array(out1), np.array(out2))

    def test_predict_different_params_different_output(self, mesh_domain, xy_pts):
        """Different param inits → different outputs (probabilistically)."""
        net = Network(mesh_domain, output_dim=1)
        net.add(Normalize())
        net.add(FNN([32]))
        p1 = net.init(jax.random.PRNGKey(1))
        p2 = net.init(jax.random.PRNGKey(2))
        o1 = net.apply(p1, xy_pts)
        o2 = net.apply(p2, xy_pts)
        assert not jnp.allclose(o1, o2)

    def test_predict_via_problem_model(self, mesh_domain, xy_pts):
        """After set_model, network attached to problem still predicts correctly."""
        net = Network(mesh_domain, output_dim=2)
        net.add(Normalize())
        net.add(FNN([32]))
        problem = ProblemStrong(mesh_domain, ['u', 'v'])
        problem.set_model(net)
        params = net.init(RNG)
        out = problem.model.apply(params, xy_pts)
        assert out.shape == (BATCH, 2)
        assert jnp.all(jnp.isfinite(out))


# ===========================================================================
# NetworkLoss
# ===========================================================================

class TestNetworkLoss:
    from pinns.network import NetworkLoss as _NL

    def test_creation_basic(self):
        from pinns.network import NetworkLoss
        fn = lambda p, x: jnp.mean(x ** 2)
        loss = NetworkLoss('reg', fn, weight=0.01)
        assert loss.name == 'reg'
        assert loss.weight == 0.01
        assert loss.x is None
        assert callable(loss.fn)

    def test_creation_with_x(self):
        from pinns.network import NetworkLoss
        x = np.ones((10, 2), dtype=np.float32)
        fn = lambda p, x: jnp.mean(x ** 2)
        loss = NetworkLoss('iface', fn, weight=2.0, x=x)
        assert loss.x is not None
        assert loss.x.shape == (10, 2)

    def test_empty_name_raises(self):
        from pinns.network import NetworkLoss
        with pytest.raises(ValueError, match="non-empty"):
            NetworkLoss('', lambda p, x: jnp.array(0.0))

    def test_non_callable_fn_raises(self):
        from pinns.network import NetworkLoss
        with pytest.raises(TypeError, match="callable"):
            NetworkLoss('r', fn=42)

    def test_negative_weight_raises(self):
        from pinns.network import NetworkLoss
        with pytest.raises(ValueError, match="weight"):
            NetworkLoss('r', lambda p, x: x, weight=-1.0)

    def test_repr(self):
        from pinns.network import NetworkLoss
        loss = NetworkLoss('foo', lambda p, x: x, weight=3.0)
        r = repr(loss)
        assert 'foo' in r
        assert '3.0' in r


class TestNetworkAddNetworkLoss:

    def test_add_network_loss_and_retrieve(self, mesh_domain):
        from pinns.network import NetworkLoss
        net = Network(mesh_domain, output_dim=1)
        assert net.network_losses == []
        fn = lambda p, x: jnp.mean(x)
        loss = NetworkLoss('test', fn, weight=0.5)
        result = net.add_network_loss(loss)
        assert result is net  # chaining
        assert len(net.network_losses) == 1
        assert net.network_losses[0].name == 'test'

    def test_add_multiple_losses(self, mesh_domain):
        from pinns.network import NetworkLoss
        net = Network(mesh_domain, output_dim=1)
        for i in range(3):
            net.add_network_loss(NetworkLoss(f'loss_{i}', lambda p, x: jnp.array(0.0)))
        assert len(net.network_losses) == 3

    def test_network_losses_returns_copy(self, mesh_domain):
        from pinns.network import NetworkLoss
        net = Network(mesh_domain, output_dim=1)
        net.add_network_loss(NetworkLoss('a', lambda p, x: jnp.array(0.0)))
        lst1 = net.network_losses
        lst2 = net.network_losses
        assert lst1 is not lst2  # fresh list each time

    def test_add_non_networkloss_raises(self, mesh_domain):
        net = Network(mesh_domain, output_dim=1)
        with pytest.raises(TypeError, match="NetworkLoss"):
            net.add_network_loss("not a loss")

    def test_loss_fn_is_callable(self, mesh_domain):
        """NetworkLoss.fn can be evaluated with (params, x)."""
        from pinns.network import NetworkLoss
        net = Network(mesh_domain, output_dim=1)
        net.add(FNN([32]))
        params = net.init(RNG)
        x = jax.random.uniform(RNG, (8, 2), dtype=jnp.float32)
        loss = NetworkLoss('check', lambda p, xb: jnp.mean(xb ** 2))
        net.add_network_loss(loss)
        val = net.network_losses[0].fn(params, x)
        assert jnp.isfinite(val)


# ===========================================================================
# register_interface_loss
# ===========================================================================

class TestRegisterInterfaceLoss:

    def _make_x_nets(self, mesh_domain):
        net_a = Network(mesh_domain, output_dim=1,
                        spatial=StrategyX(xmin=[0.0], xmax=[0.5]))
        net_b = Network(mesh_domain, output_dim=1,
                        spatial=StrategyX(xmin=[0.5], xmax=[1.0]))
        net_a.add(FNN([16]))
        net_b.add(FNN([16]))
        return net_a, net_b

    def test_registers_on_both_networks(self, mesh_domain):
        from pinns.strategies import register_interface_loss
        net_a, net_b = self._make_x_nets(mesh_domain)
        x_iface = np.zeros((4, 2), dtype=np.float32)
        x_iface[:, 0] = 0.5
        register_interface_loss(net_a, net_b, x_interface=x_iface)
        assert len(net_a.network_losses) == 1
        assert len(net_b.network_losses) == 1
        assert 'interface' in net_a.network_losses[0].name
        assert 'interface' in net_b.network_losses[0].name

    def test_default_weight_from_strategy(self, mesh_domain):
        from pinns.strategies import register_interface_loss
        net_a = Network(mesh_domain, output_dim=1,
                        spatial=StrategyX(interface_weight=7.5, xmin=[0.0], xmax=[0.5]))
        net_b = Network(mesh_domain, output_dim=1,
                        spatial=StrategyX(xmin=[0.5], xmax=[1.0]))
        net_a.add(FNN([16])); net_b.add(FNN([16]))
        x_iface = np.zeros((4, 2), dtype=np.float32)
        register_interface_loss(net_a, net_b, x_iface)
        assert net_a.network_losses[0].weight == pytest.approx(7.5)

    def test_custom_weight(self, mesh_domain):
        from pinns.strategies import register_interface_loss
        net_a, net_b = self._make_x_nets(mesh_domain)
        x_iface = np.zeros((4, 2), dtype=np.float32)
        register_interface_loss(net_a, net_b, x_iface, weight=42.0)
        assert net_a.network_losses[0].weight == pytest.approx(42.0)
        assert net_b.network_losses[0].weight == pytest.approx(42.0)

    def test_non_strategyx_raises(self, mesh_domain):
        from pinns.strategies import register_interface_loss
        net_a = Network(mesh_domain, output_dim=1, spatial=StrategyFB())
        net_b = Network(mesh_domain, output_dim=1, spatial=StrategyX())
        with pytest.raises(TypeError, match="StrategyX"):
            register_interface_loss(net_a, net_b)

    def test_loss_evaluates(self, mesh_domain):
        from pinns.strategies import register_interface_loss
        net_a, net_b = self._make_x_nets(mesh_domain)
        # Initialise both networks
        pa = net_a.init(jax.random.PRNGKey(0))
        pb = net_b.init(jax.random.PRNGKey(1))
        net_a.params = pa
        net_b.params = pb
        x_iface = jnp.zeros((4, 2))
        x_iface = x_iface.at[:, 0].set(0.5)
        register_interface_loss(net_a, net_b, x_iface)
        # fn_a(params_a, x) → scalar, must be finite
        val_a = net_a.network_losses[0].fn(pa, x_iface)
        val_b = net_b.network_losses[0].fn(pb, x_iface)
        assert jnp.isfinite(val_a)
        assert jnp.isfinite(val_b)

    def test_x_stored_in_loss(self, mesh_domain):
        from pinns.strategies import register_interface_loss
        net_a, net_b = self._make_x_nets(mesh_domain)
        x_iface = np.linspace(0, 1, 8).reshape(4, 2).astype(np.float32)
        register_interface_loss(net_a, net_b, x_iface)
        assert net_a.network_losses[0].x is not None
        assert net_a.network_losses[0].x.shape == (4, 2)

    def test_custom_name(self, mesh_domain):
        from pinns.strategies import register_interface_loss
        net_a, net_b = self._make_x_nets(mesh_domain)
        register_interface_loss(net_a, net_b, name='my_iface')
        assert net_a.network_losses[0].name == 'my_iface_a'
        assert net_b.network_losses[0].name == 'my_iface_b'


# ===========================================================================
# set_model no longer couples to problem terms
# ===========================================================================

class TestSetModelDecoupled:

    def test_set_model_does_not_add_terms(self, mesh_domain):
        """set_model should only wire domain; no extra terms injected into problem."""
        net = Network(mesh_domain, output_dim=1)
        net.add(FNN([32]))
        problem = ProblemStrong(mesh_domain, ['u'])
        n_before = len(problem._terms)
        problem.set_model(net)
        assert len(problem._terms) == n_before

    def test_set_model_twice_still_clean(self, mesh_domain):
        """Swapping networks doesn't leave stale terms."""
        net1 = Network(mesh_domain, output_dim=1)
        net2 = Network(mesh_domain, output_dim=1)
        net1.add(FNN([16])); net2.add(FNN([16]))
        problem = ProblemStrong(mesh_domain, ['u'])
        problem.set_model(net1)
        n_after_first = len(problem._terms)
        problem.set_model(net2)
        assert len(problem._terms) == n_after_first
