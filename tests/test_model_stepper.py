"""
Tests for ModelStepper, set_context (ModelBase), and set_context (ModelPartitioned).
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

import pinns
from pinns import ModelBase, create_model as Model, ModelStepper, ModelPartitioned
from pinns.models.layers import Normalize, FNN
from pinns import PartitionFB
from pinns.domain import DomainCubic

RNG = jax.random.PRNGKey(42)
BATCH = 10


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def domain_1d():
    return DomainCubic(space=[(0.0, 1.0)])


@pytest.fixture
def domain_1d_time():
    return DomainCubic(space=[(0.0, 1.0)], time=(0.0, 1.0))


@pytest.fixture
def domain_1d_partitioned():
    return DomainCubic(space=[np.linspace(0.0, 1.0, 4)])


# ---------------------------------------------------------------------------
# set_context on ModelBase
# ---------------------------------------------------------------------------

class TestBaseModelSetContext:

    def test_set_context_before_layers(self, domain_1d):
        """set_context before add() adjusts n_context and _current_dim."""
        m = ModelBase(domain_1d, output_dim=2)
        assert m.n_context == 0
        m.set_context(2, [(0.0, 1.0), (0.0, 1.0)])
        assert m.n_context == 2
        assert m.context_range == [(0.0, 1.0), (0.0, 1.0)]
        # _current_dim: 1 spatial + 2 context = 3
        assert m._current_dim == 3

    def test_set_context_updates_current_dim(self, domain_1d_time):
        """_current_dim includes time + n_context."""
        m = ModelBase(domain_1d_time, output_dim=1)
        m.set_context(3)
        # 1 spatial + 1 time + 3 context = 5
        assert m._current_dim == 5

    def test_set_context_no_range(self, domain_1d):
        """set_context without context_range keeps context_range=None."""
        m = ModelBase(domain_1d, output_dim=1)
        m.set_context(1)
        assert m.n_context == 1
        assert m.context_range is None

    def test_set_context_returns_self(self, domain_1d):
        m = ModelBase(domain_1d, output_dim=1)
        result = m.set_context(1)
        assert result is m

    def test_set_context_patches_normalize(self, domain_1d):
        """When set_context is called after add(Normalize()), the layer is patched."""
        m = ModelBase(domain_1d, output_dim=2)
        m.add(Normalize())
        m.add(FNN([8]))
        m.set_context(2, [(-1.0, 1.0), (-1.0, 1.0)])

        norm_layer = m._layers[0]
        assert norm_layer._n_context == 2
        assert norm_layer._ctx_min is not None
        np.testing.assert_allclose(norm_layer._ctx_min, [-1.0, -1.0])
        np.testing.assert_allclose(norm_layer._ctx_max, [1.0, 1.0])

    def test_set_context_patches_normalize_no_range(self, domain_1d):
        """set_context with no range clears Normalize's ctx_min/max."""
        m = ModelBase(domain_1d, output_dim=1, n_context=1,
                      context_range=[(0.0, 1.0)])
        m.add(Normalize())
        m.add(FNN([8]))
        # Remove range
        m.set_context(1, None)
        norm_layer = m._layers[0]
        assert norm_layer._ctx_min is None
        assert norm_layer._ctx_max is None

    def test_set_context_forward_pass(self, domain_1d_time):
        """After set_context, apply() runs without error with the right input dim."""
        m = ModelBase(domain_1d_time, output_dim=1, n_context=1,
                      context_range=[(0.0, 1.0)])
        m.add(Normalize())
        m.add(FNN([16]))
        params = m.init(RNG)

        # Call set_context again to verify patching doesn't break inference.
        m.set_context(1, [(0.0, 1.0)])
        # x: (batch, 1 spatial + 1 time + 1 context) = (batch, 3)
        x = jnp.ones((BATCH, 3))
        y = m.apply(params, x)
        assert y.shape == (BATCH, 1)

    def test_set_context_bad_range_length(self, domain_1d):
        m = ModelBase(domain_1d, output_dim=1)
        with pytest.raises(ValueError, match="context_range must have 2 pairs"):
            m.set_context(2, [(0.0, 1.0)])  # wrong: only 1 pair

    def test_set_context_negative(self, domain_1d):
        m = ModelBase(domain_1d, output_dim=1)
        with pytest.raises(ValueError, match="n_context must be >= 0"):
            m.set_context(-1)

    def test_set_context_zero_clears(self, domain_1d):
        """set_context(0) removes context entirely."""
        m = ModelBase(domain_1d, output_dim=1, n_context=2,
                      context_range=[(0.0, 1.0), (0.0, 1.0)])
        m.add(Normalize())
        m.set_context(0)
        assert m.n_context == 0
        assert m._layers[0]._n_context == 0


# ---------------------------------------------------------------------------
# set_context on ModelPartitioned
# ---------------------------------------------------------------------------

class TestPartitionedModelSetContext:

    def test_set_context_delegates_to_submodels(self, domain_1d_partitioned):
        model = Model(domain_1d_partitioned, output_dim=2,
                      hidden_dims=[16], normalize=False)
        pm = ModelPartitioned(model, PartitionFB(overlap=0.2))
        pm.set_context(2, [(0.0, 1.0), (0.0, 1.0)])
        for sub in pm.models:
            assert sub.n_context == 2
            assert sub.context_range == [(0.0, 1.0), (0.0, 1.0)]

    def test_n_context_property(self, domain_1d_partitioned):
        model = Model(domain_1d_partitioned, output_dim=1, hidden_dims=[16], normalize=False)
        pm = ModelPartitioned(model, PartitionFB(overlap=0.2))
        assert pm.n_context == 0
        pm.set_context(1)
        assert pm.n_context == 1

    def test_set_context_returns_self(self, domain_1d_partitioned):
        model = Model(domain_1d_partitioned, output_dim=1, hidden_dims=[16], normalize=False)
        pm = ModelPartitioned(model, PartitionFB(overlap=0.2))
        assert pm.set_context(1) is pm


# ---------------------------------------------------------------------------
# ModelStepper construction
# ---------------------------------------------------------------------------

class TestStepperModelConstruct:

    def _make_model(self, domain_1d_time):
        m = ModelBase(domain_1d_time, output_dim=2, n_context=2,
                      context_range=[(0.0, 1.0), (0.0, 1.0)])
        m.add(Normalize())
        m.add(FNN([16]))
        return m

    def test_basic_construction(self, domain_1d_time):
        m = self._make_model(domain_1d_time)
        s = ModelStepper(m)
        assert s.output_dim == 2
        assert s.n_context == 2

    def test_construction_with_context_range(self, domain_1d_time):
        """Passing context_range triggers set_context on model."""
        m = ModelBase(domain_1d_time, output_dim=1)
        m.add(FNN([16]))
        s = ModelStepper(m, context_range=[(0.0, 1.0)])
        assert m.n_context == 1
        assert s.n_context == 1

    def test_wrong_n_context_raises(self, domain_1d_time):
        m = ModelBase(domain_1d_time, output_dim=2, n_context=1)
        m.add(FNN([16]))
        with pytest.raises(ValueError, match="n_context.*must.*equal.*output_dim"):
            ModelStepper(m)

    def test_wrong_context_range_length_raises(self, domain_1d_time):
        m = ModelBase(domain_1d_time, output_dim=2)
        m.add(FNN([16]))
        with pytest.raises(ValueError, match="context_range must have 2 pairs"):
            ModelStepper(m, context_range=[(0.0, 1.0)])

    def test_bad_model_type_raises(self):
        with pytest.raises(TypeError, match="ModelBase or ModelPartitioned"):
            ModelStepper("not a model")

    def test_repr(self, domain_1d_time):
        m = self._make_model(domain_1d_time)
        s = ModelStepper(m)
        r = repr(s)
        assert "ModelStepper" in r
        assert "output_dim=2" in r


# ---------------------------------------------------------------------------
# ModelStepper.init and apply (single step)
# ---------------------------------------------------------------------------

class TestStepperModelApply:

    @pytest.fixture
    def model_and_stepper(self, domain_1d_time):
        m = ModelBase(domain_1d_time, output_dim=2, n_context=2,
                      context_range=[(0.0, 1.0), (0.0, 1.0)])
        m.add(Normalize())
        m.add(FNN([16]))
        s = ModelStepper(m)
        return s

    def test_init_returns_params(self, model_and_stepper):
        params = model_and_stepper.init(RNG)
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_apply_output_shape(self, model_and_stepper):
        """Single-step apply produces shape (B, output_dim)."""
        s = model_and_stepper
        params = s.init(RNG)
        # x: spatial (1) + time (1) = 2
        x = jnp.ones((BATCH, 2))
        prev = jnp.zeros((BATCH, 2))
        y = s.apply(params, x, prev)
        assert y.shape == (BATCH, 2)

    def test_apply_different_prev(self, model_and_stepper):
        """Different prev_output → different output (model is sensitive to context)."""
        import jax.random as jr
        s = model_and_stepper
        params = s.init(RNG)
        x = jnp.ones((BATCH, 2))
        prev1 = jnp.zeros((BATCH, 2))
        prev2 = jnp.ones((BATCH, 2))
        y1 = s.apply(params, x, prev1)
        y2 = s.apply(params, x, prev2)
        # With normalisation the outputs should differ.
        assert not jnp.allclose(y1, y2)


# ---------------------------------------------------------------------------
# ModelStepper.rollout
# ---------------------------------------------------------------------------

class TestStepperModelRollout:

    @pytest.fixture
    def stepper(self, domain_1d_time):
        m = ModelBase(domain_1d_time, output_dim=1, n_context=1,
                      context_range=[(0.0, 1.0)])
        m.add(Normalize())
        m.add(FNN([16]))
        return ModelStepper(m)

    def test_rollout_output_shape(self, stepper):
        params = stepper.init(RNG)
        x_s = jnp.ones((BATCH, 1))           # spatial
        t   = jnp.linspace(0.0, 1.0, 5)      # 5 steps
        u0  = jnp.zeros((BATCH, 1))
        traj = stepper.rollout(params, x_s, t, u0)
        assert traj.shape == (5, BATCH, 1)

    def test_rollout_first_step_uses_u0(self, stepper):
        """Step 0 uses u0; step 1 uses step 0 output, etc."""
        params = stepper.init(RNG)
        x_s = jnp.ones((BATCH, 1))
        t   = jnp.array([0.0, 0.5])
        u0  = jnp.zeros((BATCH, 1))

        traj = stepper.rollout(params, x_s, t, u0)
        # Step 0 output comes from apply(x_spatial | t[0], u0)
        x0 = jnp.concatenate([x_s, jnp.full((BATCH, 1), 0.0)], axis=-1)
        y0_ref = stepper.apply(params, x0, u0)
        np.testing.assert_allclose(np.array(traj[0]), np.array(y0_ref), atol=1e-5)

    def test_rollout_t_values_none_raises(self, stepper):
        params = stepper.init(RNG)
        x_s = jnp.ones((BATCH, 1))
        u0  = jnp.zeros((BATCH, 1))
        with pytest.raises(ValueError, match="t_values"):
            stepper.rollout(params, x_s, None, u0)

    def test_rollout_deterministic(self, stepper):
        """Same inputs, same random key → same trajectory."""
        params = stepper.init(RNG)
        x_s = jnp.ones((BATCH, 1))
        t   = jnp.linspace(0.0, 1.0, 4)
        u0  = jnp.zeros((BATCH, 1))
        t1  = stepper.rollout(params, x_s, t, u0)
        t2  = stepper.rollout(params, x_s, t, u0)
        np.testing.assert_allclose(np.array(t1), np.array(t2))


# ---------------------------------------------------------------------------
# ModelStepper wrapping ModelPartitioned
# ---------------------------------------------------------------------------

class TestStepperWithPartitioned:

    def test_construction_with_partitioned(self, domain_1d_partitioned):
        model = Model(domain_1d_partitioned, output_dim=1,
                      hidden_dims=[16], normalize=False)
        model.set_context(1, [(0.0, 1.0)])
        s = ModelStepper(model)
        assert s.output_dim == 1
        assert s.n_context == 1

    def test_construction_with_partitioned_context_range(self, domain_1d_partitioned):
        pm = ModelPartitioned(
            Model(domain_1d_partitioned, output_dim=1, hidden_dims=[16], normalize=False),
            PartitionFB(overlap=0.2),
        )
        s = ModelStepper(pm, context_range=[(0.0, 1.0)])
        assert pm.n_context == 1
        assert s.n_context == 1

    def test_partitioned_rollout_shape(self):
        domain = DomainCubic(space=[np.linspace(0.0, 1.0, 4)], time=(0.0, 1.0))
        # Build model with context before creating ModelPartitioned so init uses correct dims.
        # Model.__init__ doesn't expose context_range; use set_context after construction.
        model = Model(domain, output_dim=1, n_context=1, hidden_dims=[16], normalize=False)
        model.set_context(1, [(0.0, 1.0)])
        pm = ModelPartitioned(model, PartitionFB(overlap=0.2))
        s = ModelStepper(pm)
        params = s.init(RNG)

        x_s = jnp.linspace(0.0, 1.0, BATCH)[:, None]
        t   = jnp.linspace(0.0, 1.0, 3)
        u0  = jnp.zeros((BATCH, 1))
        traj = s.rollout(params, x_s, t, u0)
        assert traj.shape == (3, BATCH, 1)
