"""
SchedulerNTK -- NTK-trace adaptive loss weighting.

Balances PDE and boundary loss terms using the diagonal trace of the Neural
Tangent Kernel (NTK) for each term.  For term k with residual R_k(x_i):

    tr(K_k) = (1/N_k) * Σ_i  ||∇_θ R_k(x_i)||_2^2

This is the per-sample Jacobian squared norm, summed over collocation points.
It captures how "curved" the loss landscape is for each term in parameter
space, and is the quantity used in Wang et al. (2022) "When and why PINNs
fail to train" as well as in jaxpi.

Weights are updated every *update_every* epochs via EMA:

    ŵ_k  =  tr(K_total) / tr(K_k)   (target weight)
    w_k  ←  α * w_k + (1-α) * ŵ_k  (exponential moving average, α = momentum)

where tr(K_total) = Σ_k tr(K_k).

.. note::
    NTK weights are more expensive than GradNorm (requires per-sample
    Jacobians via ``jax.vmap`` + ``jax.jacrev``).  Use a large
    ``update_every`` (e.g. 1000) to amortise the cost.

Usage
-----

    from pinns.schedulers import SchedulerNTK

    trainer.compile(
        ...
        schedulers=[SchedulerNTK(momentum=0.9, update_every=1000)],
    )

Parameters
----------
terms : list[str] or None
    Names of loss terms to balance.  ``None`` (default) balances all terms.
momentum : float
    EMA momentum for weight smoothing (default 0.9).
update_every : int
    Epochs between NTK updates (default 1000).
max_points : int
    Maximum number of collocation points to use per term when computing
    the Jacobian.  The full batch (e.g. 4096) produces a Jacobian of
    shape ``(N*output_dim, N_params)`` which can exhaust GPU memory.
    Points are randomly subsampled to at most ``max_points`` before the
    Jacobian call (default 64).
"""

import jax
import jax.numpy as jnp
import numpy as np

from .scheduler_base import Scheduler


class SchedulerNTK(Scheduler):
    """
    Adaptive per-term loss weighting via NTK trace normalisation.

    See module docstring for algorithm details.
    """

    def __init__(
        self,
        terms=None,
        momentum: float = 0.9,
        update_every: int = 1000,
        min_trace: float = 1e-10,
        max_points: int = 64,
    ):
        self.terms = list(terms) if terms is not None else None
        self.momentum = float(momentum)
        self.update_every = int(update_every)
        self.min_trace = float(min_trace)
        self.max_points = int(max_points)

        self._term_names: list = []
        self._ema_weights: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_compile(self, trainer) -> None:
        self._term_names = self.terms if self.terms is not None else list(trainer.weights.keys())
        rw = getattr(trainer, 'relative_weights', {})
        self._ema_weights = {name: float(rw.get(name, 1.0))
                             for name in self._term_names}

    def needs_epoch_end_at(self, epoch: int) -> bool:
        return (epoch % self.update_every) == 0

    def on_epoch_end(self, trainer, epoch: int, loss: float) -> None:
        """Compute per-term NTK traces and update trainer weights via EMA."""
        residual_fn = getattr(trainer, '_residual_fn', None)
        if residual_fn is None:
            return
        train_data = trainer._train_data
        params     = trainer.model.params

        traces = {}
        for name in self._term_names:
            # Subsample training data to cap Jacobian size
            data = train_data
            # Find the data array for this term (exact or prefix match)
            arr = None
            if name in train_data:
                arr = train_data[name]
            else:
                prefix = name + "_"
                for k, v in train_data.items():
                    if k.startswith(prefix):
                        arr = v
                        break
            if arr is not None and arr.shape[0] > self.max_points:
                idx = np.random.choice(arr.shape[0], self.max_points, replace=False)
                data = {**train_data, name: arr[idx]}
                # Also update split sub-keys (pde_0, pde_1, ...)
                for k in list(data.keys()):
                    if k.startswith(name + "_") and hasattr(data[k], 'shape') and data[k].shape[0] > self.max_points:
                        data[k] = data[k][idx]

            # Per-sample residual flattened to 1-D
            def _per_sample_residual(p, _data=data, _name=name):
                res = residual_fn(p, _data)
                R = res.get(_name)
                if R is None:
                    return jnp.zeros(1)
                return R.reshape(-1)   # (N*output_dim,)

            # Jacobian: pytree with same structure as params,
            # each leaf has shape (N_out, *param_leaf_shape)
            J = jax.jacobian(_per_sample_residual)(params)
            # Flatten to (N_out, N_params_total) and compute mean squared row norm
            J_matrix = _flatten_jacobian(J)
            ntk_trace = float(jnp.mean(jnp.sum(J_matrix ** 2, axis=1)))
            traces[name] = ntk_trace

        valid = {k: v for k, v in traces.items() if v > self.min_trace}
        if not valid:
            return

        total_trace = sum(valid.values())

        new_weights = {}
        for name in self._term_names:
            if name not in valid:
                new_weights[name] = self._ema_weights[name]
                continue
            target = total_trace / valid[name]
            updated = self.momentum * self._ema_weights[name] + (1.0 - self.momentum) * target
            self._ema_weights[name] = updated
            new_weights[name] = updated

        # Rescale so mean weight = 1 (preserve overall loss scale)
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
        """Return the current EMA weight estimates."""
        return dict(self._ema_weights)


def _flatten_jacobian(J_tree):
    """Convert a pytree Jacobian J[output_i][param_leaf] → 2-D array (N_out, N_params).

    ``jax.jacobian(f)(params)`` returns a pytree that mirrors the structure of
    ``params`` at each output element.  We flatten both axes to get a proper
    matrix.
    """
    # J_tree has structure matching params, each leaf has shape (N_out, *param_shape)
    leaves = jax.tree_util.tree_leaves(J_tree)
    # Each leaf: (N_out, *param_leaf_shape) → (N_out, param_leaf_flat)
    cols = [l.reshape(l.shape[0], -1) for l in leaves]
    return jnp.concatenate(cols, axis=1)   # (N_out, N_params_total)


__all__ = ["SchedulerNTK"]
