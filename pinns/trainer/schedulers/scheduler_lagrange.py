"""
SchedulerLagrange -- per-element Lagrange multiplier scheduler.

Augments the trainer loss with a Lagrange term for each specified residual:

    L = sum_k  w_k * mean(r_k^2)            (trainer, unchanged)
      + sum_{k in terms}  mean(lam_k * r_k)  (added by this scheduler)

where r_k in R^{N_k} is the residual vector for term k and
lam_k in R^{N_k} is a per-element Lagrange multiplier vector updated
after every epoch via dual ascent:

    lam_k <- clip(lam_k + lr * r_k, -max_val, max_val)

The lam_k arrays are passed as explicit JIT arguments so the compiled
training step sees updated values every epoch without recompilation.

Usage
-----

    from pinns.schedulers import SchedulerLagrange

    trainer.compile(
        ...
        schedulers=[SchedulerLagrange(["pde", "boundary_up"], lr=1e-3)],
    )

Parameters
----------
terms : list[str]
    Names of the loss terms to apply Lagrange multipliers to.
lr : float
    Dual-ascent step size for multiplier updates (default 1e-3).
max_val : float
    Clip multipliers to [-max_val, max_val] after each update (default 1e6).
"""

import jax
import jax.numpy as jnp
import numpy as np

from .scheduler_base import Scheduler


class SchedulerLagrange(Scheduler):
    """
    Per-element Lagrange multiplier scheduler.

    See module docstring for full description.
    """

    def __init__(
        self,
        terms: list,
        lr: float = 1e-3,
        max_val: float = 1e6,
    ):
        self.terms = list(terms)
        self._lagrange_lr = float(lr)
        self.max_val = float(max_val)

        # lam_k vectors -- populated in on_compile, updated in on_epoch_end
        self._lambdas: dict = {}   # {term_name: jnp.ndarray shape (N_k,)}
        # Track JAX array identity per term; reset λ whenever the array is replaced
        # (i.e. after any resample, even same-size, because points have changed)
        self._data_ids: dict = {}  # {term_name: id(jax_array)}

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_compile(self, trainer) -> None:
        """Initialise lambda vectors and record initial array identities."""
        self._lambdas = {}
        self._data_ids = {}
        # For ProblemWeak, Galerkin residuals are sized by node count, not by
        # the training-data sample count.  Use eval_term_residuals when available.
        for t in self.terms:
            # Try to get the true residual size from the assembled problem.
            try:
                r = trainer.eval_term_residuals(t)
                self._lambdas[t] = jnp.zeros(len(r))
                try:
                    arr = trainer._train_data[t]
                    self._data_ids[t] = id(arr)
                except KeyError:
                    pass
                continue
            except Exception:
                pass
            # Fall back to training-data shape.
            try:
                arr = trainer._train_data[t]
                self._data_ids[t] = id(arr)
                self._lambdas[t] = jnp.zeros(arr.shape[0])
            except KeyError:
                self._lambdas[t] = jnp.zeros(0)

    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Reset λ whenever the training-data array was replaced (any resample)."""
        self.reinitialize_if_needed(trainer)

    def on_epoch_end(self, trainer, epoch: int, loss: float) -> None:
        """Dual-ascent update: lam_k <- clip(lam_k + lr * r_k, +/-max_val)."""
        for t in self.terms:
            try:
                r = trainer.eval_term_residuals(t)          # np.ndarray (N_k,)
            except (KeyError, RuntimeError):
                continue
            r_jax = jnp.array(r, dtype=jnp.float32)
            # Reinitialise λ when the residual size differs from the stored lambda
            # (e.g. ProblemWeak Galerkin residuals are sized by node count, not
            # by the training-data sample count used in on_compile).
            if self._lambdas.get(t, jnp.zeros(0)).shape[0] != r_jax.shape[0]:
                self._lambdas[t] = jnp.zeros(r_jax.shape[0])
            self._lambdas[t] = jnp.clip(
                self._lambdas[t] + self._lagrange_lr * r_jax,
                -self.max_val,
                self.max_val,
            )

    # ------------------------------------------------------------------
    # JIT-state protocol
    # ------------------------------------------------------------------

    def get_jit_state(self) -> dict:
        """Return the current lambda arrays keyed by term name.

        The trainer passes this dict into the JIT-compiled training step as an
        explicit argument so that updated values are used every epoch.
        """
        return dict(self._lambdas)

    def extra_loss(self, residuals: dict, jit_state: dict):
        """Lagrange contribution: sum_{k in terms} mean(stop_grad(lam_k) * r_k).

        Parameters
        ----------
        residuals : dict[str, jax.Array]
            Per-term residual vectors from the problem's make_residual_fn.
        jit_state : dict[str, jax.Array]
            The dict returned by get_jit_state() for this scheduler.
        """
        total = jnp.array(0.0)
        for t in self.terms:
            if t not in residuals or t not in jit_state:
                continue
            lam = jax.lax.stop_gradient(jit_state[t])
            r   = residuals[t]
            # Guard against shape mismatch (e.g. first epoch before reinit)
            if lam.shape[0] != r.shape[0]:
                continue
            total = total + jnp.mean(lam * r)
        return total

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reinitialize_if_needed(self, trainer) -> None:
        """Reset λ_k to zero if the data array for term k was replaced.

        A resample always creates a new JAX array (different ``id``), even when
        the number of points is unchanged.  Old multiplier values correspond to
        old collocation points and must not be reused.
        """
        for t in self.terms:
            try:
                arr = trainer._train_data[t]
            except KeyError:
                continue
            current_id = id(arr)
            if self._data_ids.get(t) != current_id or t not in self._lambdas:
                self._lambdas[t] = jnp.zeros(arr.shape[0])
                self._data_ids[t] = current_id

    def reset(self) -> None:
        """Zero all Lagrange multipliers."""
        self._lambdas = {t: jnp.zeros_like(v) for t, v in self._lambdas.items()}

    def get_statistics(self) -> dict:
        """Return mean / std / min / max for each lambda vector."""
        return {
            t: {
                "mean": float(jnp.mean(v)),
                "std":  float(jnp.std(v)),
                "min":  float(jnp.min(v)),
                "max":  float(jnp.max(v)),
            }
            for t, v in self._lambdas.items()
        }


__all__ = ["SchedulerLagrange"]
