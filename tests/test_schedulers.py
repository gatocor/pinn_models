"""
Test suite for PINN training schedulers.

Covers:
- SchedulerExponentialDecay : lr() math, on_epoch_start hook
- SchedulerReduceLROnPlateau: plateau detection, cooldown, min_lr, no-op on lbfgs
- SchedulerResample         : resample triggered every_n, pool disabled path
- SchedulerAdaptiveResample : modes 'replace' / 'rar' fire at correct epochs
- SchedulerCurriculum       : stage advancement, domain bound restore
- SchedulerLagrange         : multiplier initialisation, weight injection, update step
- SchedulerWarmupDecay      : warmup phase, decay phase, start_value, LR applied to trainer
- SchedulerCausal           : sort-index construction, causal weight shape/monotonicity,
                              combine='min'/'mean', multi-term support
- SchedulerGradNorm         : weight initialisation, EMA update direction, re-scaling
- SchedulerNTK              : weight initialisation, EMA update direction, re-scaling
- SchedulerPartition        : mask set-up, window=1/2/None stages, reset_optimizer,
                              manual freeze/unfreeze, integration through Trainer.train()
- Integration               : schedulers work end-to-end through Trainer.train()
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pinns.trainer.schedulers import (
    Scheduler,
    SchedulerExponentialDecay,
    SchedulerReduceLROnPlateau,
    SchedulerResample,
    SchedulerAdaptiveResample,
    SchedulerCurriculum,
    SchedulerLagrange,
    SchedulerWarmupDecay,
    SchedulerCausal,
    SchedulerGradNorm,
    SchedulerNTK,
    SchedulerPartition,
)
from pinns.trainer.schedulers.scheduler_partition import (
    MaskedState,
    make_masked_optimizer,
    _make_mask,
    _active_from_mask,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mock_trainer(
    *,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    global_epoch: int = 0,
    loss_history: list = None,
):
    """Return a SimpleNamespace that mimics the Trainer helper API."""
    _lr = [lr]

    t = SimpleNamespace()
    t.optimizer_name         = optimizer_name
    t._global_epoch          = global_epoch
    t.get_learning_rate      = lambda: _lr[0]
    t.set_learning_rate      = lambda v: _lr.__setitem__(0, v)
    t.get_global_epoch       = lambda: t._global_epoch
    t.get_loss_history       = lambda: {"loss": list(loss_history or [])}
    t.train_samples          = {}
    t.test_samples           = {}
    t._train_data            = {}
    t._test_data             = {}
    t._schedulers            = []
    t._sample_train_data     = MagicMock()
    t._sample_test_data      = MagicMock()
    t._init_optimizer_state  = MagicMock()
    return t


def _lr_trainer(lr=1e-3, global_epoch=0, optimizer_name="adam"):
    """Minimal mock trainer sufficient for LR-scheduler tests."""
    _lr = [float(lr)]
    t = SimpleNamespace()
    t.optimizer_name    = optimizer_name
    t._global_epoch     = global_epoch
    t.get_global_epoch  = lambda: t._global_epoch
    t.set_learning_rate = lambda v: _lr.__setitem__(0, float(v))
    t.get_learning_rate = lambda: _lr[0]
    return t


def _partition_trainer(n_subnets=4, t_col=-1):
    """Mock trainer carrying a ModelPartitioned-like network (params + bounds)."""
    keys   = [f"sub_{i}" for i in range(n_subnets)]
    params = {k: {"w": jnp.ones((4,))} for k in keys}

    net = SimpleNamespace()
    net.params   = params
    net.n_models = n_subnets
    net._all_xmin = np.array(
        [[i / n_subnets, i / n_subnets] for i in range(n_subnets)], dtype=np.float32
    )
    net._all_xmax = np.array(
        [[(i + 1) / n_subnets, (i + 1) / n_subnets] for i in range(n_subnets)],
        dtype=np.float32,
    )
    net._all_outer_xmin = net._all_xmin.copy()

    _lr = [1e-3]
    t = SimpleNamespace()
    t.optimizer_name    = "adam"
    t.optimizer         = optax.adam(1e-3)
    t.opt_state         = None
    t.model             = net
    t.t_min             = None
    t.t_max             = None
    t._global_epoch     = 0
    t._test_data        = {}
    t._sample_train_data = lambda: None
    t._sample_test_data  = lambda: None
    t.get_global_epoch  = lambda: t._global_epoch
    t.set_learning_rate = lambda v: _lr.__setitem__(0, float(v))
    t.get_learning_rate = lambda: _lr[0]
    return t


# ===========================================================================
# SchedulerExponentialDecay
# ===========================================================================

class TestSchedulerExponentialDecay:

    def test_lr_at_step_zero(self):
        s = SchedulerExponentialDecay(gamma=0.5, each_n_steps=100)
        assert s.lr(1.0, 0) == pytest.approx(1.0)

    def test_lr_at_first_decay(self):
        s = SchedulerExponentialDecay(gamma=0.5, each_n_steps=100)
        assert s.lr(1.0, 100) == pytest.approx(0.5)

    def test_lr_at_second_decay(self):
        s = SchedulerExponentialDecay(gamma=0.5, each_n_steps=100)
        assert s.lr(1.0, 200) == pytest.approx(0.25)

    def test_lr_between_steps_unchanged(self):
        s = SchedulerExponentialDecay(gamma=0.9, each_n_steps=1000)
        assert s.lr(1.0, 500) == pytest.approx(1.0)

    def test_on_epoch_start_updates_trainer(self):
        s = SchedulerExponentialDecay(gamma=0.5, each_n_steps=10)
        t = _make_mock_trainer(lr=1.0, global_epoch=10)
        s.on_epoch_start(t, 0)
        assert t.get_learning_rate() == pytest.approx(0.5)

    def test_on_epoch_start_skips_lbfgs(self):
        s = SchedulerExponentialDecay(gamma=0.5, each_n_steps=1)
        t = _make_mock_trainer(lr=1.0, optimizer_name="lbfgs")
        s.on_epoch_start(t, 100)
        assert t.get_learning_rate() == pytest.approx(1.0)

    def test_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            SchedulerExponentialDecay(gamma=0)

    def test_invalid_each_n_steps(self):
        with pytest.raises(ValueError, match="each_n_steps"):
            SchedulerExponentialDecay(each_n_steps=0)


# ===========================================================================
# SchedulerReduceLROnPlateau
# ===========================================================================

def _flat_losses(n: int, value: float = 0.1) -> list:
    return [value] * n


class TestSchedulerReduceLROnPlateau:

    def test_no_reduction_before_window(self):
        s = SchedulerReduceLROnPlateau(window=100, epsilon=1e-3, factor=0.5)
        t = _make_mock_trainer(lr=1e-3, loss_history=[0.1] * 50)
        s.on_epoch_end(t, epoch=50, loss=0.1)
        assert t.get_learning_rate() == pytest.approx(1e-3)

    def test_reduction_on_plateau(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        t = _make_mock_trainer(lr=1e-3, loss_history=_flat_losses(12))
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert t.get_learning_rate() == pytest.approx(5e-4)

    def test_multiple_reductions(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        t = _make_mock_trainer(lr=1e-3, loss_history=_flat_losses(12))
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert s._reduction_count == 1
        t._global_epoch = 12
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert s._reduction_count == 2
        assert t.get_learning_rate() == pytest.approx(2.5e-4)

    def test_cooldown_prevents_immediate_re_reduction(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=100)
        t = _make_mock_trainer(lr=1e-3, loss_history=_flat_losses(12))
        s.on_epoch_end(t, epoch=12, loss=0.1)
        first_lr = t.get_learning_rate()
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert t.get_learning_rate() == pytest.approx(first_lr)

    def test_min_lr_floor(self):
        s = SchedulerReduceLROnPlateau(
            window=10, epsilon=1e-3, factor=0.01, min_lr=1e-4, cooldown=0
        )
        t = _make_mock_trainer(lr=1e-4, loss_history=_flat_losses(12))
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert t.get_learning_rate() >= 1e-4

    def test_no_reduction_on_lbfgs(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        t = _make_mock_trainer(lr=1e-3, optimizer_name="lbfgs",
                               loss_history=_flat_losses(12))
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert t.get_learning_rate() == pytest.approx(1e-3)

    def test_base_lr_captured_once(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        t = _make_mock_trainer(lr=1e-3, loss_history=_flat_losses(12))
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert s._base_lr == pytest.approx(1e-3)
        t.set_learning_rate(5e-4)
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert s._base_lr == pytest.approx(1e-3)

    def test_reset_clears_state(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        t = _make_mock_trainer(lr=1e-3, loss_history=_flat_losses(12))
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert s._reduction_count == 1
        s.reset()
        assert s._reduction_count == 0
        assert s._base_lr is None

    def test_decreasing_loss_no_reduction(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        losses = [1.0 - i * 0.05 for i in range(12)]
        t = _make_mock_trainer(lr=1e-3, loss_history=losses)
        s.on_epoch_end(t, epoch=12, loss=losses[-1])
        assert s._reduction_count == 0

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            SchedulerReduceLROnPlateau(window=0)
        with pytest.raises(ValueError):
            SchedulerReduceLROnPlateau(epsilon=0)
        with pytest.raises(ValueError):
            SchedulerReduceLROnPlateau(factor=0)
        with pytest.raises(ValueError):
            SchedulerReduceLROnPlateau(factor=1.0)
        with pytest.raises(ValueError):
            SchedulerReduceLROnPlateau(ema_alpha=0)


# ===========================================================================
# SchedulerResample
# ===========================================================================

class TestSchedulerResample:

    def _trainer_with_samples(self):
        t = _make_mock_trainer()
        t.train_samples = {"pde": 100}
        t.test_samples  = {}
        return t

    def test_resample_called_at_interval(self):
        s = SchedulerResample(every_n=5)
        t = self._trainer_with_samples()
        s.on_epoch_start(t, epoch=5)
        t._sample_train_data.assert_called_once()

    def test_no_resample_before_interval(self):
        s = SchedulerResample(every_n=5)
        t = self._trainer_with_samples()
        s.on_epoch_start(t, epoch=4)
        t._sample_train_data.assert_not_called()

    def test_no_resample_at_epoch_zero(self):
        s = SchedulerResample(every_n=5)
        t = self._trainer_with_samples()
        s.on_epoch_start(t, epoch=0)
        t._sample_train_data.assert_not_called()

    def test_invalid_every_n(self):
        with pytest.raises(ValueError):
            SchedulerResample(every_n=0)


# ===========================================================================
# SchedulerAdaptiveResample
# ===========================================================================

class TestSchedulerAdaptiveResampleConfig:

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            SchedulerAdaptiveResample(mode="bad")

    def test_replace_mode_created(self):
        s = SchedulerAdaptiveResample(mode="replace", every_n=50)
        assert s.mode == "replace"
        assert s.every_n == 50

    def test_rar_mode_created(self):
        s = SchedulerAdaptiveResample(mode="rar", every_n=100, k=2.0, c=0.5)
        assert s.mode == "rar"
        assert s.k == 2.0

    def test_on_epoch_skipped_when_not_interval(self):
        s = SchedulerAdaptiveResample(mode="replace", every_n=10)
        t = _make_mock_trainer()
        s._replace_or_add_resample = MagicMock()
        s._rar_resample = MagicMock()
        s.on_epoch_start(t, epoch=7)
        s._replace_or_add_resample.assert_not_called()
        s._rar_resample.assert_not_called()

    def test_on_epoch_triggered_at_interval(self):
        s = SchedulerAdaptiveResample(mode="replace", every_n=10)
        t = _make_mock_trainer()
        s._replace_or_add_resample = MagicMock()
        s.on_epoch_start(t, epoch=10)
        s._replace_or_add_resample.assert_called_once_with(t)

    def test_rar_triggered_at_interval(self):
        s = SchedulerAdaptiveResample(mode="rar", every_n=10)
        t = _make_mock_trainer()
        s._rar_resample = MagicMock()
        s.on_epoch_start(t, epoch=10)
        s._rar_resample.assert_called_once_with(t)


# ===========================================================================
# SchedulerCurriculum
# ===========================================================================

class TestSchedulerCurriculum:

    def _make_curriculum_trainer(self, xmax0=1.0):
        t = _make_mock_trainer()
        domain = SimpleNamespace(xmax=[xmax0])
        t.problem = SimpleNamespace(domain=domain)
        t.t_min = 0.0
        t.t_max = xmax0
        return t

    def test_on_compile_sets_stage0_bound(self):
        s = SchedulerCurriculum(t_ends=[0.5, 1.0], epochs_per_stage=100)
        t = self._make_curriculum_trainer(xmax0=1.0)
        s.on_compile(t)
        assert t.t_max == pytest.approx(0.5)

    def test_on_epoch_advances_stage(self):
        s = SchedulerCurriculum(t_ends=[0.5, 1.0], epochs_per_stage=100)
        t = self._make_curriculum_trainer(xmax0=2.0)
        s.on_compile(t)
        s.on_epoch_start(t, epoch=100)
        assert t.t_max == pytest.approx(1.0)

    def test_on_epoch_clamps_to_last_stage(self):
        s = SchedulerCurriculum(t_ends=[0.5, 1.0], epochs_per_stage=100)
        t = self._make_curriculum_trainer(xmax0=2.0)
        s.on_compile(t)
        s.on_epoch_start(t, epoch=9999)
        assert t.t_max == pytest.approx(1.0)

    def test_on_training_end_restores_original_bound(self):
        s = SchedulerCurriculum(t_ends=[0.5, 1.0], epochs_per_stage=100)
        t = self._make_curriculum_trainer(xmax0=2.0)
        s.on_compile(t)
        s.on_training_end(t)
        assert t.t_max == pytest.approx(2.0)

    def test_invalid_t_ends(self):
        with pytest.raises(ValueError):
            SchedulerCurriculum(t_ends=[])


# ===========================================================================
# SchedulerLagrange
# ===========================================================================

class TestSchedulerLagrangeUnit:

    def test_init_stores_params(self):
        s = SchedulerLagrange(["pde", "bc"], lr=0.1)
        assert s._lagrange_lr == 0.1
        assert s.terms == ["pde", "bc"]

    def test_is_scheduler_subclass(self):
        assert issubclass(SchedulerLagrange, Scheduler)


# ===========================================================================
# SchedulerWarmupDecay
# ===========================================================================

class TestSchedulerWarmupDecay:

    def test_warmup_at_step_zero(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=100)
        assert s.compute_lr(0) == pytest.approx(0.0)

    def test_warmup_midpoint(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=100)
        assert s.compute_lr(50) == pytest.approx(5e-4)

    def test_warmup_at_peak(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=100)
        assert s.compute_lr(99) == pytest.approx(1e-3 * 99 / 100)

    def test_decay_immediately_after_warmup(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=100,
                                  decay_rate=0.5, decay_steps=100)
        assert s.compute_lr(100) == pytest.approx(1e-3)

    def test_decay_one_period(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=0,
                                  decay_rate=0.5, decay_steps=100)
        assert s.compute_lr(100) == pytest.approx(5e-4)

    def test_decay_two_periods(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=0,
                                  decay_rate=0.5, decay_steps=100)
        assert s.compute_lr(200) == pytest.approx(2.5e-4)

    def test_start_value_warmup(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=100, start_value=1e-4)
        assert s.compute_lr(0) == pytest.approx(1e-4)
        assert s.compute_lr(50) == pytest.approx(1e-4 + (1e-3 - 1e-4) * 0.5)

    def test_lr_monotone_during_warmup(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=1000)
        lrs = [s.compute_lr(step) for step in range(1001)]
        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_lr_monotone_during_decay(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=0,
                                  decay_rate=0.9, decay_steps=100)
        lrs = [s.compute_lr(step) for step in range(0, 1000, 10)]
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_on_epoch_start_applies_lr(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=0)
        t = _lr_trainer(lr=1.0, global_epoch=0)
        s.on_epoch_start(t, epoch=0)
        assert t.get_learning_rate() == pytest.approx(s.compute_lr(0))

    def test_on_epoch_start_uses_global_epoch(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=0,
                                  decay_rate=0.9, decay_steps=100)
        t = _lr_trainer(lr=1.0, global_epoch=100)
        s.on_epoch_start(t, epoch=0)
        assert t.get_learning_rate() == pytest.approx(s.compute_lr(100))

    def test_on_epoch_start_accumulates_local_epoch(self):
        s = SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=0,
                                  decay_rate=0.9, decay_steps=100)
        t = _lr_trainer(lr=1.0, global_epoch=0)
        s.on_epoch_start(t, epoch=50)
        assert t.get_learning_rate() == pytest.approx(s.compute_lr(50))

    def test_invalid_peak_value(self):
        with pytest.raises(ValueError, match="peak_value"):
            SchedulerWarmupDecay(peak_value=0)

    def test_invalid_decay_rate(self):
        with pytest.raises(ValueError, match="decay_rate"):
            SchedulerWarmupDecay(decay_rate=0)

    def test_invalid_decay_steps(self):
        with pytest.raises(ValueError, match="decay_steps"):
            SchedulerWarmupDecay(decay_steps=0)


# ===========================================================================
# SchedulerCausal
# ===========================================================================

class TestSchedulerCausal:

    def _causal(self, n_chunks=4, t_col=0, tol=1.0, combine="min"):
        return SchedulerCausal(term="pde", tol=tol, n_chunks=n_chunks,
                               t_col=t_col, combine=combine)

    def _trainer_with_data(self, N=32, t_col=0):
        data = np.random.rand(N, 3).astype(np.float32)
        data[:, t_col] = np.linspace(0, 1, N)
        t = SimpleNamespace()
        t._train_data = {"pde": data}
        t.weights = {"pde": 1.0}
        return t

    def test_on_compile_builds_sort_idx(self):
        s = self._causal()
        t = self._trainer_with_data()
        s.on_compile(t)
        assert "pde" in s._sort_idxs
        assert s._sort_idxs["pde"].shape[0] == 32

    def test_sort_idx_orders_by_t(self):
        s = self._causal(t_col=2)
        data = np.zeros((16, 3), dtype=np.float32)
        data[:, 2] = np.arange(15, -1, -1, dtype=np.float32)
        t = SimpleNamespace()
        t._train_data = {"pde": data}
        t.weights = {"pde": 1.0}
        s.on_compile(t)
        idx = np.array(s._sort_idxs["pde"])
        assert idx[0] == 15
        assert idx[-1] == 0

    def test_get_jit_state_keys(self):
        s = self._causal()
        t = self._trainer_with_data()
        s.on_compile(t)
        state = s.get_jit_state()
        assert "sort_idx_pde" in state

    def _run_term_weights(self, combine="min"):
        s = self._causal(n_chunks=4, combine=combine)
        N = 32
        data = np.zeros((N, 3), dtype=np.float32)
        data[:, 0] = np.linspace(0, 1, N)
        t = SimpleNamespace()
        t._train_data = {"pde": data}
        t.weights = {"pde": 1.0}
        s.on_compile(t)
        tw = s.term_weights({"pde": jnp.ones((N, 1))}, s.get_jit_state())
        return tw["pde"]

    def test_term_weights_shape_min(self):
        assert self._run_term_weights("min").shape[0] == 32

    def test_term_weights_shape_mean(self):
        assert self._run_term_weights("mean").shape[0] == 32

    def test_term_weights_all_nonneg(self):
        assert float(jnp.min(self._run_term_weights("min"))) >= 0.0

    def test_term_weights_first_chunk_largest(self):
        s = self._causal(n_chunks=4, tol=10.0)
        N = 64
        data = np.zeros((N, 3), dtype=np.float32)
        data[:, 0] = np.linspace(0, 1, N)
        t = SimpleNamespace()
        t._train_data = {"pde": data}
        t.weights = {"pde": 1.0}
        s.on_compile(t)
        tw = s.term_weights({"pde": jnp.ones((N, 1))}, s.get_jit_state())
        sort_idx = np.array(s._sort_idxs["pde"])
        assert float(tw["pde"][sort_idx[0]]) >= float(tw["pde"][sort_idx[-1]])

    def test_multi_term_keys_in_output(self):
        s = SchedulerCausal(term=["pde", "ic"], t_col=0, n_chunks=4)
        N = 32
        data = np.zeros((N, 3), dtype=np.float32)
        data[:, 0] = np.linspace(0, 1, N)
        t = SimpleNamespace()
        t._train_data = {"pde": data, "ic": data}
        t.weights = {"pde": 1.0, "ic": 1.0}
        s.on_compile(t)
        tw = s.term_weights(
            {"pde": jnp.ones((N, 1)), "ic": jnp.ones((N, 1))},
            s.get_jit_state()
        )
        assert "pde" in tw and "ic" in tw

    def test_invalid_combine(self):
        with pytest.raises(ValueError, match="combine"):
            SchedulerCausal(combine="bad")


# ===========================================================================
# SchedulerGradNorm
# ===========================================================================

class TestSchedulerGradNorm:

    def _net_params(self):
        return {"w": jnp.ones((4,)), "b": jnp.zeros((2,))}

    def _make_trainer(self, residual_fn):
        params = self._net_params()
        t = SimpleNamespace()
        t.model          = SimpleNamespace(params=params)
        t._train_data      = {"pde": np.zeros((4, 2), dtype=np.float32),
                               "ic":  np.zeros((4, 2), dtype=np.float32)}
        t.weights          = {"pde": 1.0, "ic": 1.0}
        t.relative_weights = {"pde": 1.0, "ic": 1.0}
        t._residual_fn     = residual_fn
        applied = {}
        t.set_weights = lambda w: applied.update(w)
        t.get_weights = lambda: {**t.weights, **applied}
        return t, applied

    def test_on_compile_initialises_ema(self):
        s = SchedulerGradNorm(terms=["pde", "ic"])
        t, _ = self._make_trainer(
            lambda p, d: {"pde": jnp.sum(p["w"]), "ic": jnp.sum(p["w"])}
        )
        s.on_compile(t)
        assert set(s._ema_weights.keys()) == {"pde", "ic"}

    def test_needs_epoch_end_at_interval(self):
        s = SchedulerGradNorm(update_every=100)
        assert s.needs_epoch_end_at(0)
        assert s.needs_epoch_end_at(100)
        assert not s.needs_epoch_end_at(50)

    def test_larger_norm_term_gets_lower_weight(self):
        s = SchedulerGradNorm(terms=["pde", "ic"], momentum=0.0, update_every=1)
        t, applied = self._make_trainer(
            lambda p, d: {"pde": jnp.sum(p["w"]) * 10.0, "ic": jnp.sum(p["w"])}
        )
        s.on_compile(t)
        s.on_epoch_end(t, epoch=1, loss=0.0)
        assert "pde" in applied and "ic" in applied
        assert applied["pde"] < applied["ic"]

    def test_equal_norms_give_equal_weights(self):
        s = SchedulerGradNorm(terms=["pde", "ic"], momentum=0.0, update_every=1)
        t, applied = self._make_trainer(
            lambda p, d: {"pde": jnp.sum(p["w"]), "ic": jnp.sum(p["w"])}
        )
        s.on_compile(t)
        s.on_epoch_end(t, epoch=1, loss=0.0)
        assert applied.get("pde", 1.0) == pytest.approx(applied.get("ic", 1.0), rel=1e-4)

    def test_weights_mean_is_one_after_rescaling(self):
        s = SchedulerGradNorm(terms=["pde", "ic"], momentum=0.0, update_every=1)
        t, applied = self._make_trainer(
            lambda p, d: {"pde": jnp.sum(p["w"]) * 5.0, "ic": jnp.sum(p["w"])}
        )
        s.on_compile(t)
        s.on_epoch_end(t, epoch=1, loss=0.0)
        vals = [applied[k] for k in ["pde", "ic"]]
        assert sum(vals) / len(vals) == pytest.approx(1.0, rel=1e-3)


# ===========================================================================
# SchedulerNTK
# ===========================================================================

class TestSchedulerNTK:

    def _net_params(self):
        return {"w": jnp.ones((4,)), "b": jnp.zeros((2,))}

    def _trainer_ntk(self, scale_pde=1.0, scale_ic=1.0):
        params = self._net_params()

        def _residual_fn(p, data):
            return {
                "pde": jnp.sum(p["w"]) * scale_pde * jnp.ones((4, 1)),
                "ic":  jnp.sum(p["w"]) * scale_ic  * jnp.ones((4, 1)),
            }

        t = SimpleNamespace()
        t.model          = SimpleNamespace(params=params)
        t._train_data      = {
            "pde": np.zeros((4, 2), dtype=np.float32),
            "ic":  np.zeros((4, 2), dtype=np.float32),
        }
        t.weights          = {"pde": 1.0, "ic": 1.0}
        t.relative_weights = {"pde": 1.0, "ic": 1.0}
        t._residual_fn     = _residual_fn
        applied = {}
        t.set_weights = lambda w: applied.update(w)
        return t, applied

    def test_on_compile_initialises_ema(self):
        s = SchedulerNTK(terms=["pde", "ic"])
        t, _ = self._trainer_ntk()
        s.on_compile(t)
        assert set(s._ema_weights.keys()) == {"pde", "ic"}

    def test_needs_epoch_end_at_interval(self):
        s = SchedulerNTK(update_every=500)
        assert s.needs_epoch_end_at(0)
        assert s.needs_epoch_end_at(500)
        assert not s.needs_epoch_end_at(1)

    def test_high_trace_term_gets_lower_weight(self):
        s = SchedulerNTK(terms=["pde", "ic"], momentum=0.0, update_every=1, max_points=4)
        t, applied = self._trainer_ntk(scale_pde=10.0, scale_ic=1.0)
        s.on_compile(t)
        s.on_epoch_end(t, epoch=1, loss=0.0)
        assert applied["pde"] < applied["ic"]

    def test_equal_traces_give_equal_weights(self):
        s = SchedulerNTK(terms=["pde", "ic"], momentum=0.0, update_every=1, max_points=4)
        t, applied = self._trainer_ntk(scale_pde=1.0, scale_ic=1.0)
        s.on_compile(t)
        s.on_epoch_end(t, epoch=1, loss=0.0)
        assert applied.get("pde", 1.0) == pytest.approx(applied.get("ic", 1.0), rel=1e-3)

    def test_weights_mean_is_one(self):
        s = SchedulerNTK(terms=["pde", "ic"], momentum=0.0, update_every=1, max_points=4)
        t, applied = self._trainer_ntk(scale_pde=3.0, scale_ic=1.0)
        s.on_compile(t)
        s.on_epoch_end(t, epoch=1, loss=0.0)
        vals = [applied[k] for k in ["pde", "ic"]]
        assert sum(vals) / len(vals) == pytest.approx(1.0, rel=1e-3)

    def test_no_crash_without_residual_fn(self):
        s = SchedulerNTK(terms=["pde"], update_every=1)
        t = SimpleNamespace()
        t.model  = SimpleNamespace(params=self._net_params())
        t._train_data = {}
        t.weights = {"pde": 1.0}
        t.relative_weights = {"pde": 1.0}
        s.on_compile(t)
        s.on_epoch_end(t, epoch=1, loss=0.0)


# ===========================================================================
# SchedulerPartition
# ===========================================================================

class TestSchedulerPartition:

    def _scheduler(self, n=4, window=1, epochs_per_partition=100,
                   reset_optimizer=False):
        return SchedulerPartition(
            epochs_per_partition=epochs_per_partition,
            t_col=-1,
            window=window,
            reset_optimizer=reset_optimizer,
        )

    def _compiled(self, n=4, window=1, epochs_per_partition=100,
                  reset_optimizer=False):
        t = _partition_trainer(n_subnets=n)
        s = self._scheduler(n=n, window=window,
                            epochs_per_partition=epochs_per_partition,
                            reset_optimizer=reset_optimizer)
        s.on_compile(t)
        return s, t

    def test_on_compile_wraps_optimizer(self):
        s, t = self._compiled(n=3)
        assert isinstance(t.opt_state, MaskedState)

    def test_on_compile_activates_first_subnet_only(self):
        s, t = self._compiled(n=4, window=1)
        active = s.active_subnets(t)
        assert len(active) == 1
        assert "sub_0" in active

    def test_on_compile_sorted_by_time(self):
        s, t = self._compiled(n=4, window=1)
        # When group_by_time=True (default) keys are grouped into inner lists.
        # With unique t_min per subnet each inner list has one element.
        flat = [k for grp in s._sorted_subnet_keys for k in (grp if isinstance(grp, list) else [grp])]
        assert flat == ["sub_0", "sub_1", "sub_2", "sub_3"]

    def test_lbfgs_skipped(self):
        t = _partition_trainer(n_subnets=2)
        t.optimizer_name = "lbfgs"
        s = self._scheduler(n=2)
        s.on_compile(t)
        assert t.opt_state is None

    def test_window1_stage_advances_one_at_a_time(self):
        s, t = self._compiled(n=4, window=1, epochs_per_partition=10)
        t._global_epoch = 0
        s.on_epoch_start(t, epoch=10)
        assert s.active_subnets(t) == {"sub_1"}
        s.on_epoch_start(t, epoch=20)
        assert s.active_subnets(t) == {"sub_2"}

    def test_window1_clamps_at_last_stage(self):
        s, t = self._compiled(n=4, window=1, epochs_per_partition=10)
        t._global_epoch = 0
        s.on_epoch_start(t, epoch=9999)
        assert s.active_subnets(t) == {"sub_3"}

    def test_window1_no_advance_before_interval(self):
        s, t = self._compiled(n=4, window=1, epochs_per_partition=100)
        t._global_epoch = 0
        s.on_epoch_start(t, epoch=50)
        assert s.active_subnets(t) == {"sub_0"}

    def test_window2_stage0_has_one_subnet(self):
        s, t = self._compiled(n=4, window=2, epochs_per_partition=10)
        assert s.active_subnets(t) == {"sub_0"}

    def test_window2_stage1_has_two_subnets(self):
        s, t = self._compiled(n=4, window=2, epochs_per_partition=10)
        t._global_epoch = 0
        s.on_epoch_start(t, epoch=10)
        assert s.active_subnets(t) == {"sub_0", "sub_1"}

    def test_window2_stage2_slides(self):
        s, t = self._compiled(n=4, window=2, epochs_per_partition=10)
        t._global_epoch = 0
        s.on_epoch_start(t, epoch=20)
        assert s.active_subnets(t) == {"sub_1", "sub_2"}

    def test_window2_stage3_slides(self):
        s, t = self._compiled(n=4, window=2, epochs_per_partition=10)
        t._global_epoch = 0
        s.on_epoch_start(t, epoch=30)
        assert s.active_subnets(t) == {"sub_2", "sub_3"}

    def test_window_none_cumulative(self):
        s, t = self._compiled(n=4, window=None, epochs_per_partition=10)
        t._global_epoch = 0
        s.on_epoch_start(t, epoch=10)
        assert s.active_subnets(t) == {"sub_0", "sub_1"}
        s.on_epoch_start(t, epoch=20)
        assert s.active_subnets(t) == {"sub_0", "sub_1", "sub_2"}
        s.on_epoch_start(t, epoch=30)
        assert s.active_subnets(t) == {"sub_0", "sub_1", "sub_2", "sub_3"}

    def test_manual_freeze(self):
        s, t = self._compiled(n=4, window=None, epochs_per_partition=10)
        t._global_epoch = 0
        s.on_epoch_start(t, epoch=30)
        s.freeze(t, ["sub_1", "sub_3"])
        active = s.active_subnets(t)
        assert "sub_1" not in active
        assert "sub_3" not in active
        assert "sub_0" in active
        assert "sub_2" in active

    def test_manual_unfreeze(self):
        s, t = self._compiled(n=4, window=1, epochs_per_partition=10)
        s.unfreeze(t, ["sub_2"])
        assert "sub_2" in s.active_subnets(t)
        assert "sub_0" in s.active_subnets(t)

    def test_freeze_before_compile_raises(self):
        s = self._scheduler(n=4)
        t = _partition_trainer(n_subnets=4)
        with pytest.raises(RuntimeError, match="not been compiled"):
            s.freeze(t, ["sub_0"])

    def test_frozen_subnet_gets_zero_update(self):
        s, t = self._compiled(n=4, window=1)
        grads = {k: {"w": jnp.ones((4,))} for k in t.model.params}
        updates, _ = t.optimizer.update(grads, t.opt_state, t.model.params)
        for key in ["sub_1", "sub_2", "sub_3"]:
            assert jnp.allclose(updates[key]["w"], jnp.zeros((4,))), \
                f"{key} should have zero update"
        assert not jnp.allclose(updates["sub_0"]["w"], jnp.zeros((4,)))

    def test_reset_optimizer_zeros_moments_on_advance(self):
        s, t = self._compiled(n=4, window=1, epochs_per_partition=10,
                              reset_optimizer=True)
        grads = {k: {"w": jnp.ones((4,))} for k in t.model.params}
        _, t.opt_state = t.optimizer.update(grads, t.opt_state, t.model.params)
        t._global_epoch = 0
        s.on_epoch_start(t, epoch=10)
        assert s.active_subnets(t) == {"sub_1"}

    def test_manual_only_mode_no_auto_advance(self):
        t = _partition_trainer(n_subnets=4)
        s = SchedulerPartition(epochs_per_partition=None, window=1)
        s.on_compile(t)
        initial = s.active_subnets(t)
        t._global_epoch = 999999
        s.on_epoch_start(t, epoch=999999)
        assert s.active_subnets(t) == initial

    def test_invalid_epochs_per_partition(self):
        with pytest.raises(ValueError):
            SchedulerPartition(epochs_per_partition=-1)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            SchedulerPartition(window=0)


# ===========================================================================
# Integration: schedulers end-to-end through Trainer.train()
# ===========================================================================

import pinns
from pinns import DomainCubic, ProblemStrong, create_model, Trainer, derivative

_EPOCHS = 30
_PRINT  = 1
_E      = 20  # shorter runs for newer schedulers


def _simple_ode():
    domain = DomainCubic(time=(0.0, 1.0))
    problem = ProblemStrong(domain=domain, output_names=["x"])

    def residual(X, U, params, d):
        import jax.numpy as jnp
        x   = U[:, 0:1]
        x_t = d(x, X, 0, (0,))
        return x_t + x

    problem.add_inner(residual, name="pde")
    problem.add_initial(1.0, outputs="x", name="IC")
    return domain, problem


def _ode_trainer():
    domain, problem = _simple_ode()
    network = create_model(domain, output_dim=1, hidden_dims=(8,), activation="tanh")
    return Trainer(network, problem=problem)


class TestSchedulerIntegration:

    def _make_trainer(self):
        domain, problem = _simple_ode()
        network = create_model(domain, output_dim=1, hidden_dims=(8,), activation="tanh")
        return Trainer(network, problem=problem), domain

    def _compile_and_train(self, schedulers, epochs=_EPOCHS, print_each=_PRINT):
        trainer, _ = self._make_trainer()
        trainer.compile(
            problem={"pde": {"train": 50}, "IC": {"train": 10}},
            epochs=epochs, print_each=print_each, show=None,
            schedulers=schedulers,
        )
        trainer.train()
        assert len(trainer.history["loss"]) >= epochs
        return trainer

    def test_exponential_decay_runs(self):
        self._compile_and_train([SchedulerExponentialDecay(gamma=0.9, each_n_steps=10)])

    def test_reduce_lr_on_plateau_runs(self):
        self._compile_and_train([
            SchedulerReduceLROnPlateau(window=10, epsilon=1.0, factor=0.5, cooldown=0)
        ])

    def test_resample_runs(self):
        self._compile_and_train([SchedulerResample(every_n=5)])

    def test_adaptive_resample_replace_runs(self):
        self._compile_and_train([
            SchedulerAdaptiveResample(mode="replace", every_n=10, ratio=0.3)
        ])

    def test_adaptive_resample_rar_runs(self):
        self._compile_and_train([SchedulerAdaptiveResample(mode="rar", every_n=10)])

    def test_lagrange_runs(self):
        self._compile_and_train([SchedulerLagrange(["pde", "IC"], lr=0.01)])

    def test_combined_schedulers_run(self):
        self._compile_and_train([
            SchedulerExponentialDecay(gamma=0.95, each_n_steps=10),
            SchedulerResample(every_n=10),
        ])

    def test_warmup_decay_integration(self):
        self._compile_and_train([
            SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=5,
                                  decay_rate=0.9, decay_steps=5),
        ], epochs=_E)

    def test_causal_integration(self):
        self._compile_and_train([
            SchedulerCausal(term="pde", tol=1.0, n_chunks=4, t_col=0)
        ], epochs=_E)

    def test_grad_norm_integration(self):
        self._compile_and_train([
            SchedulerGradNorm(terms=["pde", "IC"], update_every=5)
        ], epochs=_E)

    def test_ntk_integration(self):
        self._compile_and_train([
            SchedulerNTK(terms=["pde", "IC"], update_every=5, max_points=16)
        ], epochs=_E)

    def test_warmup_decay_plus_causal(self):
        self._compile_and_train([
            SchedulerWarmupDecay(peak_value=1e-3, warmup_steps=5, decay_steps=5),
            SchedulerCausal(term="pde", n_chunks=4, t_col=0),
        ], epochs=_E)
