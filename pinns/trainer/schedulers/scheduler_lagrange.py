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

        # lam_k vectors -- populated in on_compile, updated via jit_update inside JIT
        self._lambdas: dict = {}   # {term_name: jnp.ndarray shape (N_k,)}
        # Track JAX array identity per term; reset λ whenever the array is replaced
        # (i.e. after any resample, even same-size, because points have changed)
        self._data_ids: dict = {}  # {term_name: id(jax_array)}
        # Set by reinitialize_if_needed; tells the training loop to rebuild
        # _sched_states from get_jit_state() rather than reuse the JIT output.
        self._needs_state_rebuild: bool = True

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

    def needs_epoch_end_at(self, epoch: int) -> bool:
        """SchedulerLagrange updates λ inside JIT; on_epoch_end is never needed."""
        return False

    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Reset λ whenever the training-data array was replaced (any resample)."""
        self.reinitialize_if_needed(trainer)

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

    def jit_update(self, jit_state: dict, residuals: dict) -> dict:
        """Dual-ascent update run *inside* the JIT step — no host sync.

        lam_k <- clip(lam_k + lr * mean(r_k), -max_val, max_val)
        """
        new_state = dict(jit_state)
        for t in self.terms:
            if t not in jit_state or t not in residuals:
                continue
            lam = jit_state[t]
            r   = residuals[t].flatten()
            if lam.shape[0] != r.shape[0]:
                continue
            new_state[t] = jnp.clip(
                lam + self._lagrange_lr * r,
                -self.max_val,
                self.max_val,
            )
        return new_state

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
                self._needs_state_rebuild = True

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
