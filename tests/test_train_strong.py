"""
Test suite for end-to-end training with ProblemStrong + JAX Trainer.

Covers:
- Trainer construction (ModelBase, no .to() crash)
- compile() accepts dict-style train/test/weights
- train() runs without error and populates history
- Loss is finite and decreases over training
- predict() returns correct shape
- Standard PINN, FB-PINN (PartitionFB), and AL-PINN paths
- 0-D (time-only ODE) and 1-D+time spatial PDE
- Multiple outputs
"""

import os
# Use CPU-only JAX to avoid spurious CUDA platform errors in test environments
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import pinns
from pinns import (
    DomainCubic,
    ProblemStrong,
    PartitionFB,
    create_model,
    Trainer,
    derivative,
)
from pinns.trainer import SchedulerLagrange

# ---------------------------------------------------------------------------
# Number of training epochs for tests (tiny — correctness, not accuracy)
# ---------------------------------------------------------------------------
_FAST_EPOCHS = 20
_PRINT_EACH  = 999  # silence mid-training prints


# ===========================================================================
# Shared problem factories
# ===========================================================================

def _make_ode_problem():
    """Damped oscillator  x'' + 0.1 x' + 10 x = 0,  t in [0, 1]."""
    domain = DomainCubic(time=(0.0, 1.0))

    problem = ProblemStrong(domain=domain, output_names=["x"])

    def residual(X, U, params, derivative):
        beta  = params["parameter"]["beta"]
        omega = params["parameter"]["omega"]
        x     = U[:, 0:1]
        x_t   = derivative(x, X, 0, (0,))
        x_tt  = derivative(x, X, 0, (0, 0))
        return x_tt + beta * x_t + omega * x

    problem.add_inner(residual, name="ode")
    problem.add_initial(1.0, outputs="x", name="Ix")   # x(0) = 1

    def ic_velocity(X, U, params, derivative):
        x   = U[:, 0:1]
        x_t = derivative(x, X, 0, (0,))
        return x_t                                      # x'(0) = 0

    problem.add_initial(ic_velocity, outputs="x", name="Ivx")

    problem.add_parameter("beta",  0.1)
    problem.add_parameter("omega", 10.0)
    return domain, problem


def _make_1d_pde_problem():
    """1-D Poisson: -u_xx = pi^2 sin(pi x),  x in [0, 1], u(0)=u(1)=0."""
    domain = DomainCubic(space=[(0.0, 1.0)])

    problem = ProblemStrong(domain=domain, output_names=["u"])

    def residual(X, U, params, derivative):
        u    = U[:, 0:1]
        u_xx = derivative(u, X, 0, (0, 0))
        import jax.numpy as jnp
        pi = jnp.pi
        return -u_xx - pi**2 * jnp.sin(pi * X[:, 0:1])

    problem.add_inner(residual, name="pde")
    # Both boundary values are 0 — enforce on the full boundary at once
    problem.add_dirichlet(0.0, name="bc", region="all")
    return domain, problem


def _make_multiout_problem():
    """Coupled system: u' = -v, v' = u  (harmonic pair), t in [0, 1]."""
    domain = DomainCubic(time=(0.0, 1.0))

    problem = ProblemStrong(domain=domain, output_names=["u", "v"])

    def residual(X, U, params, derivative):
        import jax.numpy as jnp
        u   = U[:, 0:1]
        v   = U[:, 1:2]
        u_t = derivative(u, X, 0, (0,))
        v_t = derivative(v, X, 0, (0,))
        return jnp.concatenate([u_t + v, v_t - u], axis=1)
        return jnp.concatenate([u_t + v, v_t - u], axis=1)

    # name as list → one term per equation column: "ode_u", "ode_v"
    problem.add_inner(residual, name=["ode_u", "ode_v"])
    # Use separate scalar initial conditions per output
    problem.add_initial(1.0, outputs="u", name="IC_u")   # u(0) = 1
    problem.add_initial(0.0, outputs="v", name="IC_v")   # v(0) = 0
    return domain, problem


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestTrainerConstruction:
    """Trainer can be constructed without errors."""

    def test_ode_no_to_crash(self):
        """ModelBase has no .to() — must not raise AttributeError."""
        domain, problem = _make_ode_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16,), activation="tanh")
        trainer = Trainer(network, problem=problem)   # should not raise
        assert trainer.problem is problem
        assert trainer.model is network

    def test_1d_pde_construction(self):
        domain, problem = _make_1d_pde_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16,))
        trainer = Trainer(network, problem=problem)
        assert trainer is not None

    def test_multiout_construction(self):
        domain, problem = _make_multiout_problem()
        network = create_model(domain, output_dim=2, hidden_dims=(16,))
        Trainer(network, problem=problem)

    def test_fb_construction(self):
        domain, problem = _make_ode_problem()
        domain.set_partition(time=4)
        network = create_model(
            domain, output_dim=1, hidden_dims=(8,),
            partition=PartitionFB(overlap=0.5),
        )
        Trainer(network, problem=problem)


# ===========================================================================
# 2. compile()
# ===========================================================================

class TestCompile:
    def test_basic_compile(self):
        domain, problem = _make_ode_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16,))
        trainer = Trainer(network, problem=problem)
        trainer.compile(
            problem={
                "ode": {"train": 50, "test": 50, "weight": 1.0},
                "Ix":  {"train": 1,  "weight": 1.0},
                "Ivx": {"train": 1,  "weight": 1.0},
            },
            epochs=_FAST_EPOCHS,
            print_each=_PRINT_EACH,
        )
        assert trainer._compiled

    def test_compile_sets_epochs(self):
        domain, problem = _make_ode_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16,))
        trainer = Trainer(network, problem=problem)
        trainer.compile(epochs=42, print_each=_PRINT_EACH)
        assert trainer._epochs == 42

    def test_compile_pde(self):
        domain, problem = _make_1d_pde_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16,))
        trainer = Trainer(network, problem=problem)
        trainer.compile(
            problem={
                "pde": {"train": 30, "test": 30, "weight": 1.0},
                "bc":  {"train": 5,  "weight": 1.0},
            },
            epochs=_FAST_EPOCHS,
            print_each=_PRINT_EACH,
        )
        assert trainer._compiled


# ===========================================================================
# 3. train() — basic smoke tests
# ===========================================================================

class TestTrainSmoke:
    """train() runs end-to-end without exception."""

    def _run(self, problem, network, term_names, n_train=30, n_test=30):
        trainer = Trainer(network, problem=problem)
        trainer.compile(
            problem={k: {"train": n_train, "test": n_test, "weight": 1.0} for k in term_names},
            epochs=_FAST_EPOCHS,
            print_each=_PRINT_EACH,
            show=None,
        )
        trainer.train()
        return trainer

    def test_ode_standard(self):
        domain, problem = _make_ode_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16, 16))
        self._run(problem, network, ["ode", "Ix", "Ivx"])

    def test_1d_pde_standard(self):
        domain, problem = _make_1d_pde_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16, 16))
        self._run(problem, network, ["pde", "bc"])

    def test_multiout(self):
        domain, problem = _make_multiout_problem()
        network = create_model(domain, output_dim=2, hidden_dims=(16,))
        self._run(problem, network, ["ode_u", "ode_v", "IC_u", "IC_v"])

    def test_fb_pinn(self):
        domain, problem = _make_ode_problem()
        domain.set_partition(time=4)
        network = create_model(
            domain, output_dim=1, hidden_dims=(8, 8),
            partition=PartitionFB(overlap=0.5),
        )
        self._run(problem, network, ["ode", "Ix", "Ivx"])


# ===========================================================================
# 4. History population
# ===========================================================================

class TestHistory:
    def _trained_trainer(self):
        domain, problem = _make_ode_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16,))
        trainer = Trainer(network, problem=problem)
        trainer.compile(
            problem={
                "ode": {"train": 50, "test": 50, "weight": 1.0},
                "Ix":  {"train": 1,  "weight": 1.0},
                "Ivx": {"train": 1,  "weight": 1.0},
            },
            epochs=_FAST_EPOCHS,
            print_each=_PRINT_EACH,
            show=None,
        )
        trainer.train()
        return trainer

    def test_history_keys_present(self):
        trainer = self._trained_trainer()
        for key in ("epoch", "loss", "train_loss"):
            assert key in trainer.history, f"missing history key: {key}"

    def test_history_length(self):
        trainer = self._trained_trainer()
        # History records entries at print_each intervals + final epoch.
        # With print_each > epochs, at minimum epoch-0 and the last epoch are recorded.
        assert len(trainer.history["epoch"]) >= 1

    def test_loss_is_finite(self):
        trainer = self._trained_trainer()
        losses = np.array(trainer.history["loss"])
        assert np.all(np.isfinite(losses)), "non-finite loss encountered"

    def test_loss_decreases(self):
        """Final loss should be lower than initial loss over 20 epochs."""
        trainer = self._trained_trainer()
        losses = np.array(trainer.history["loss"])
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4e}, last={losses[-1]:.4e}"
        )


# ===========================================================================
# 5. predict()
# ===========================================================================

class TestPredict:
    def _trained_trainer(self):
        domain, problem = _make_ode_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16,))
        trainer = Trainer(network, problem=problem)
        trainer.compile(
            problem={
                "ode": {"train": 50},
                "Ix":  {"train": 1},
                "Ivx": {"train": 1},
            },
            epochs=_FAST_EPOCHS,
            print_each=_PRINT_EACH,
            show=None,
        )
        trainer.train()
        return trainer

    def test_predict_shape_1d(self):
        trainer = self._trained_trainer()
        t = np.linspace(0, 1, 100).reshape(-1, 1)
        pred = trainer.predict(t)
        assert pred.shape == (100, 1)

    def test_predict_finite(self):
        trainer = self._trained_trainer()
        t = np.linspace(0, 1, 50).reshape(-1, 1)
        pred = trainer.predict(t)
        assert np.all(np.isfinite(pred)), "predict returned non-finite values"


# ===========================================================================
# 6. Augmented Lagrangian (AL) mode — supported for ProblemStrong
# ===========================================================================

class TestALMode:
    """AL mode works for ProblemStrong: Lagrange multipliers applied to IC terms."""

    def test_al_trains_successfully(self):
        """AL mode with lagrange_constraints trains without error."""
        domain, problem = _make_ode_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16,))
        trainer = Trainer(network, problem=problem)
        trainer.compile(
            problem={
                "ode": {"train": 50},
                "Ix":  {"train": 1},
                "Ivx": {"train": 1},
            },
            epochs=_FAST_EPOCHS,
            print_each=_PRINT_EACH,
            schedulers=[SchedulerLagrange(terms=["Ix", "Ivx"])],  
            show=None,
        )
        trainer.train()   # must not raise
        assert len(trainer.history["epoch"]) >= 1

    def test_al_disabled_trains_fine(self):
        """Explicitly disabling AL mode still trains without error."""
        domain, problem = _make_ode_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16,))
        trainer = Trainer(network, problem=problem)
        trainer.compile(
            problem={
                "ode": {"train": 50},
                "Ix":  {"train": 1},
                "Ivx": {"train": 1},
            },
            epochs=_FAST_EPOCHS,
            print_each=_PRINT_EACH,
            show=None,
        )
        trainer.train()  # must not raise


# ===========================================================================
# 7. Re-train (calling train() twice) accumulates history
# ===========================================================================

class TestRetrain:
    def test_retrain_accumulates_epochs(self):
        domain, problem = _make_ode_problem()
        network = create_model(domain, output_dim=1, hidden_dims=(16,))
        trainer = Trainer(network, problem=problem)
        trainer.compile(
            problem={
                "ode": {"train": 50},
                "Ix":  {"train": 1},
                "Ivx": {"train": 1},
            },
            epochs=_FAST_EPOCHS,
            print_each=_PRINT_EACH,
            show=None,
        )
        trainer.train()
        n_after_first = len(trainer.history["epoch"])
        trainer.train()
        n_after_second = len(trainer.history["epoch"])
        # History must grow with each train() call
        assert n_after_second > n_after_first
        # Epoch numbers must be strictly increasing
        epochs = trainer.history["epoch"]
        assert all(epochs[i] < epochs[i + 1] for i in range(len(epochs) - 1))
