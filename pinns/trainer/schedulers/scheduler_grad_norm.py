"""
SchedulerGradNorm -- per-term gradient-norm adaptive loss weighting.

Balances PDE and boundary loss terms by keeping their gradient magnitudes
proportional.  For each tracked term k, the weight is updated as:

    ĝ_k  =  ||∇_θ L_k||₂          (gradient norm for term k alone)
    ḡ    =  (1/K) Σ_k ĝ_k          (mean gradient norm across all tracked terms)
    ŵ_k  =  ḡ / ĝ_k               (normalized target weight)
    w_k  ←  α * w_k + (1-α) * ŵ_k  (exponential moving average)

The normalised weights are re-scaled so their mean equals 1 (matching the
original unweighted scale).  Updates happen every *update_every* epochs.

Usage
-----

    from pinns.schedulers import SchedulerGradNorm

    trainer.compile(
        ...
        schedulers=[SchedulerGradNorm(momentum=0.9, update_every=1000)],
    )

Parameters
----------
terms : list[str] or None
    Names of the loss terms to balance.  ``None`` (default) balances *all*
    terms registered in the problem.
momentum : float
    EMA momentum for weight smoothing (default 0.9).
update_every : int
    Number of epochs between weight updates (default 1000).  Computing
    per-term gradient norms requires one backward pass per term, which can
    be expensive; setting a large value amortises the cost.
min_norm : float
    Gradient norms below this threshold are ignored to avoid division by
    near-zero values (default 1e-10).
"""

import jax
import jax.numpy as jnp

from .scheduler_base import Scheduler


class SchedulerGradNorm(Scheduler):
    """
    Adaptive per-term loss weighting via gradient-norm normalisation.

    See module docstring for algorithm details.
    """

    def __init__(
        self,
        terms=None,
        momentum: float = 0.9,
        update_every: int = 1000,
        min_norm: float = 1e-10,
    ):
        self.terms = list(terms) if terms is not None else None
        self.momentum = float(momentum)
        self.update_every = int(update_every)
        self.min_norm = float(min_norm)

        self._term_names: list = []
        self._ema_weights: dict = {}   # {name: float}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_compile(self, trainer) -> None:
        """Initialise EMA weights from the current relative weights (adaptive state)."""
        self._term_names = self.terms if self.terms is not None else list(trainer.weights.keys())
        # Initialise EMA from relative_weights (adaptive state, not user priority).
        rw = getattr(trainer, 'relative_weights', {})
        self._ema_weights = {name: float(rw.get(name, 1.0))
                             for name in self._term_names}

    def needs_epoch_end_at(self, epoch: int) -> bool:
        return (epoch % self.update_every) == 0

    def on_epoch_end(self, trainer, epoch: int, loss: float) -> None:
        """Compute per-term gradient norms and update trainer.weights via EMA."""
        residual_fn = getattr(trainer, '_residual_fn', None)
        if residual_fn is None:
            return
        train_data = trainer._train_data
        params     = trainer.network.params

        norms = {}
        for name in self._term_names:
            def _loss_k(p, _name=name):
                res = residual_fn(p, train_data)
                R = res.get(_name)
                if R is None:
                    return jnp.array(0.0)
                return jnp.mean(R ** 2)

            g = jax.grad(_loss_k)(params)
            flat, _ = jax.flatten_util.ravel_pytree(g)
            norms[name] = float(jnp.linalg.norm(flat))

        # Filter out terms with tiny norms
        valid = {k: v for k, v in norms.items() if v > self.min_norm}
        if not valid:
            return

        mean_norm = sum(valid.values()) / len(valid)

        new_weights = {}
        for name in self._term_names:
            if name not in valid:
                # No update for terms with near-zero gradients
                new_weights[name] = self._ema_weights[name]
                continue
            target = mean_norm / valid[name]
            updated = self.momentum * self._ema_weights[name] + (1.0 - self.momentum) * target
            self._ema_weights[name] = updated
            new_weights[name] = updated

        # Re-scale so the mean weight across tracked terms stays at the previous
        # mean (preserving the overall loss magnitude).
        vals = list(new_weights.values())
        if vals:
            mean_w = sum(vals) / len(vals)
            if mean_w > 0.0:
                scale = 1.0 / mean_w
                new_weights = {k: v * scale for k, v in new_weights.items()}
                self._ema_weights = {k: self._ema_weights[k] * scale
                                     for k in self._term_names}

        trainer.set_weights(new_weights)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_weights(self) -> dict:
        """Return the current EMA weight estimates (after rescaling)."""
        return dict(self._ema_weights)
