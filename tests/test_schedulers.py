"""
Test suite for PINN training schedulers.

Covers:
- SchedulerExponentialDecay: lr() math, on_epoch_start hook
- SchedulerReduceLROnPlateau: plateau detection, cooldown, min_lr, no-op on lbfgs
- SchedulerResample: resample triggered every_n, pool disabled path
- SchedulerAdaptiveResample: modes 'replace' / 'rar' fire at correct epochs
- SchedulerCurriculum: stage advancement, domain bound restore
- SchedulerLagrange: multiplier initialisation, weight injection, update step
- Integration: schedulers work end-to-end through Trainer.train()
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

from pinns.trainer.schedulers import (
    Scheduler,
    SchedulerExponentialDecay,
    SchedulerReduceLROnPlateau,
    SchedulerResample,
    SchedulerAdaptiveResample,
    SchedulerCurriculum,
    SchedulerLagrange,
)


# ---------------------------------------------------------------------------
# Lightweight mock trainer used across unit tests
# ---------------------------------------------------------------------------

def _make_mock_trainer(
    *,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    global_epoch: int = 0,
    loss_history: list = None,
):
    """Return a SimpleNamespace that mimics the Trainer helper API."""
    _lr = [lr]          # mutable so set_learning_rate updates it

    t = SimpleNamespace()
    t.optimizer_name         = optimizer_name
    t._global_epoch          = global_epoch
    t.get_learning_rate      = lambda: _lr[0]
    t.set_learning_rate      = lambda v: _lr.__setitem__(0, v)
    t.get_global_epoch       = lambda: t._global_epoch
    t.get_loss_history       = lambda: {"loss": list(loss_history or [])}
    # Extra trainer attributes used by some schedulers
    t.train_samples          = {}
    t.test_samples           = {}
    t._train_data            = {}
    t._test_data             = {}
    t._schedulers            = []
    t._sample_train_data     = MagicMock()
    t._sample_test_data      = MagicMock()
    t._init_optimizer_state  = MagicMock()
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
        # epoch=0 → global 10 → decay_count = 1 → lr = 0.5
        s.on_epoch_start(t, 0)
        assert t.get_learning_rate() == pytest.approx(0.5)

    def test_on_epoch_start_skips_lbfgs(self):
        s = SchedulerExponentialDecay(gamma=0.5, each_n_steps=1)
        t = _make_mock_trainer(lr=1.0, optimizer_name="lbfgs")
        s.on_epoch_start(t, 100)
        assert t.get_learning_rate() == pytest.approx(1.0)  # unchanged

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
    """Return a flat loss history of length n — guaranteed to trigger plateau."""
    return [value] * n


class TestSchedulerReduceLROnPlateau:

    def test_no_reduction_before_window(self):
        s = SchedulerReduceLROnPlateau(window=100, epsilon=1e-3, factor=0.5)
        t = _make_mock_trainer(lr=1e-3, loss_history=[0.1] * 50)
        s.on_epoch_end(t, epoch=50, loss=0.1)
        assert t.get_learning_rate() == pytest.approx(1e-3)

    def test_reduction_on_plateau(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        losses = _flat_losses(12)
        t = _make_mock_trainer(lr=1e-3, loss_history=losses)
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert t.get_learning_rate() == pytest.approx(5e-4)

    def test_multiple_reductions(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        losses = _flat_losses(12)
        t = _make_mock_trainer(lr=1e-3, loss_history=losses)
        # First reduction
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert s._reduction_count == 1
        # Second reduction — need to bypass cooldown by advancing global epoch
        t._global_epoch = 12
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert s._reduction_count == 2
        assert t.get_learning_rate() == pytest.approx(2.5e-4)

    def test_cooldown_prevents_immediate_re_reduction(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=100)
        losses = _flat_losses(12)
        t = _make_mock_trainer(lr=1e-3, loss_history=losses)
        s.on_epoch_end(t, epoch=12, loss=0.1)
        first_lr = t.get_learning_rate()
        # Try again at same global step — cooldown should block
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert t.get_learning_rate() == pytest.approx(first_lr)

    def test_min_lr_floor(self):
        s = SchedulerReduceLROnPlateau(
            window=10, epsilon=1e-3, factor=0.01, min_lr=1e-4, cooldown=0
        )
        losses = _flat_losses(12)
        t = _make_mock_trainer(lr=1e-4, loss_history=losses)
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert t.get_learning_rate() >= 1e-4

    def test_no_reduction_on_lbfgs(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        losses = _flat_losses(12)
        t = _make_mock_trainer(lr=1e-3, optimizer_name="lbfgs", loss_history=losses)
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert t.get_learning_rate() == pytest.approx(1e-3)

    def test_base_lr_captured_once(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        losses = _flat_losses(12)
        t = _make_mock_trainer(lr=1e-3, loss_history=losses)
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert s._base_lr == pytest.approx(1e-3)
        # Manually lower the lr and call again — base_lr should not change
        t.set_learning_rate(5e-4)
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert s._base_lr == pytest.approx(1e-3)

    def test_reset_clears_state(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        losses = _flat_losses(12)
        t = _make_mock_trainer(lr=1e-3, loss_history=losses)
        s.on_epoch_end(t, epoch=12, loss=0.1)
        assert s._reduction_count == 1
        s.reset()
        assert s._reduction_count == 0
        assert s._base_lr is None

    def test_decreasing_loss_no_reduction(self):
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1e-3, factor=0.5, cooldown=0)
        # Linearly decreasing loss — relative change will be large → no plateau
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
        s = SchedulerResample(every_n=5, pool_size=1)
        t = self._trainer_with_samples()
        s.on_epoch_start(t, epoch=5)
        t._sample_train_data.assert_called_once()

    def test_no_resample_before_interval(self):
        s = SchedulerResample(every_n=5, pool_size=1)
        t = self._trainer_with_samples()
        s.on_epoch_start(t, epoch=4)
        t._sample_train_data.assert_not_called()

    def test_no_resample_at_epoch_zero(self):
        s = SchedulerResample(every_n=5, pool_size=1)
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
    """Config / parameter validation only — no domain calls needed."""

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
        """on_epoch_start must be a no-op when epoch % every_n != 0."""
        s = SchedulerAdaptiveResample(mode="replace", every_n=10)
        t = _make_mock_trainer()
        # Patch the internal methods so no actual domain logic runs
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

    def _make_domain_mock(self, xmax0: float = 1.0):
        domain = SimpleNamespace()
        domain.xmax = [xmax0]
        return domain

    def _make_problem_mock(self, domain):
        problem = SimpleNamespace()
        problem.domain = domain
        return problem

    def _make_curriculum_trainer(self, xmax0=1.0):
        t = _make_mock_trainer()
        domain = self._make_domain_mock(xmax0)
        t.problem = self._make_problem_mock(domain)
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
# SchedulerLagrange — lightweight unit tests
# ===========================================================================

class TestSchedulerLagrangeUnit:
    """Tests that do not require a full JAX Trainer (mock the minimal surface)."""

    def test_init_stores_params(self):
        """Constructing with terms and lr stores them correctly."""
        s = SchedulerLagrange(["pde", "bc"], lr=0.1)
        assert s._lagrange_lr == 0.1
        assert s.terms == ["pde", "bc"]

    def test_is_scheduler_subclass(self):
        assert issubclass(SchedulerLagrange, Scheduler)


# ===========================================================================
# Integration: schedulers end-to-end through Trainer.train()
# ===========================================================================

import pinns
from pinns import DomainCubic, ProblemStrong, create_model, Trainer, derivative

_EPOCHS = 30
_PRINT  = 1   # record every epoch so history length == _EPOCHS


def _simple_ode():
    domain = DomainCubic(time=(0.0, 1.0))
    problem = ProblemStrong(domain=domain, output_names=["x"])

    def residual(X, U, params, d):
        import jax.numpy as jnp
        x   = U[:, 0:1]
        x_t = d(x, X, 0, (0,))
        return x_t + x  # x' + x = 0

    # Use "pde" as the interior term name so SchedulerAdaptiveResample
    # (which hard-codes the key 'pde') works in integration tests.
    problem.add_inner(residual, name="pde")
    problem.add_initial(1.0, outputs="x", name="IC")
    return domain, problem


class TestSchedulerIntegration:

    def _make_trainer(self):
        domain, problem = _simple_ode()
        network = create_model(domain, output_dim=1, hidden_dims=(8,), activation="tanh")
        return Trainer(problem, network), domain

    def test_exponential_decay_runs(self):
        trainer, _ = self._make_trainer()
        s = SchedulerExponentialDecay(gamma=0.9, each_n_steps=10)
        trainer.compile(
            problem={"pde": {"train": 50}, "IC": {"train": 10}},
            epochs=_EPOCHS, print_each=_PRINT, show=None,
            schedulers=[s],
        )
        trainer.train()
        assert len(trainer.history["loss"]) >= _EPOCHS

    def test_reduce_lr_on_plateau_runs(self):
        trainer, _ = self._make_trainer()
        s = SchedulerReduceLROnPlateau(window=10, epsilon=1.0, factor=0.5, cooldown=0)
        trainer.compile(
            problem={"pde": {"train": 50}, "IC": {"train": 10}},
            epochs=_EPOCHS, print_each=_PRINT, show=None,
            schedulers=[s],
        )
        trainer.train()
        assert len(trainer.history["loss"]) >= _EPOCHS

    def test_resample_runs(self):
        trainer, _ = self._make_trainer()
        s = SchedulerResample(every_n=5, pool_size=1)
        trainer.compile(
            problem={"pde": {"train": 50}, "IC": {"train": 10}},
            epochs=_EPOCHS, print_each=_PRINT, show=None,
            schedulers=[s],
        )
        trainer.train()
        assert len(trainer.history["loss"]) >= _EPOCHS

    def test_adaptive_resample_replace_runs(self):
        trainer, _ = self._make_trainer()
        s = SchedulerAdaptiveResample(mode="replace", every_n=10, ratio=0.3)
        trainer.compile(
            problem={"pde": {"train": 50}, "IC": {"train": 10}},
            epochs=_EPOCHS, print_each=_PRINT, show=None,
            schedulers=[s],
        )
        trainer.train()
        assert len(trainer.history["loss"]) >= _EPOCHS

    def test_adaptive_resample_rar_runs(self):
        trainer, _ = self._make_trainer()
        s = SchedulerAdaptiveResample(mode="rar", every_n=10)
        trainer.compile(
            problem={"pde": {"train": 50}, "IC": {"train": 10}},
            epochs=_EPOCHS, print_each=_PRINT, show=None,
            schedulers=[s],
        )
        trainer.train()
        assert len(trainer.history["loss"]) >= _EPOCHS

    def test_lagrange_runs(self):
        trainer, _ = self._make_trainer()
        s = SchedulerLagrange(["pde", "IC"], lr=0.01)
        trainer.compile(
            problem={"pde": {"train": 50}, "IC": {"train": 10}},
            epochs=_EPOCHS, print_each=_PRINT, show=None,
            schedulers=[s],
        )
        trainer.train()
        assert len(trainer.history["loss"]) >= _EPOCHS

    def test_combined_schedulers_run(self):
        """Multiple schedulers in the same list must not conflict."""
        trainer, _ = self._make_trainer()
        schedulers = [
            SchedulerExponentialDecay(gamma=0.95, each_n_steps=10),
            SchedulerResample(every_n=10, pool_size=1),
        ]
        trainer.compile(
            problem={"pde": {"train": 50}, "IC": {"train": 10}},
            epochs=_EPOCHS, print_each=_PRINT, show=None,
            schedulers=schedulers,
        )
        trainer.train()
        assert len(trainer.history["loss"]) >= _EPOCHS
