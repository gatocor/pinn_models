"""
SchedulerCausal -- causal (time-respecting) loss weighting for time-dependent PDEs.

When solving time-dependent PDEs the network should satisfy the PDE at early
times before being pushed to satisfy it at later times.  This scheduler
implements the soft causal weighting scheme from

    Wang et al. (2022) "Respecting causality for training physics-informed
    neural networks", CMAME.

The time-axis collocation points are sorted and split into *n_chunks* ordered
chunks.  The per-chunk mean squared residual is computed, and each chunk
receives a weight that decays exponentially with the cumulative loss of
*preceding* chunks:

    l_i  =  mean( R[chunk_i]² )
    w_i  =  exp( -tol * Σ_{j < i}  l_j )

Earlier chunks (small t) have weight 1; later chunks are suppressed until the
network has learned the correct solution there.  Gradients flow through the
chunk losses ``l_i`` normally, but ``stop_gradient`` is applied to the causal
weights ``w_i`` so they act as constants for the current step.

The scheduler

* zeroes the regular scalar weight for the PDE term in ``on_compile``
  (so the default ``w * mean(R²)`` is disabled for that term), and
* adds the causal-weighted loss back via ``extra_loss`` inside the JIT step.

The time-sort permutation is recomputed from ``trainer._train_data[term]`` at
every ``on_epoch_start``, so it stays valid after resampling.

.. warning::
    Mini-batch training shuffles the data arrays each epoch; causal ordering
    applies to the *full* batch, so ``compile(batch_size=...)`` should not be
    used together with this scheduler.

Usage
-----

    from pinns.schedulers import SchedulerCausal

    trainer.compile(
        ...
        # Single term (original behaviour):
        schedulers=[SchedulerCausal(term="pde", tol=1.0, n_chunks=16)],
        # Multiple terms — same causal weights applied to each:
        schedulers=[SchedulerCausal(term=["pde", "initial"], tol=1.0, n_chunks=16)],
    )

Parameters
----------
term : str | list[str]
    Name(s) of the residual term(s) to apply causal weighting to.
    All listed terms must have their input arrays share the same time column
    (``t_col``).  The causal weights are computed independently per term
    from that term's own collocation points.
    Default: ``"pde"``.
tol : float
    Causality tolerance ε.  Larger values enforce stricter temporal ordering;
    smaller values reduce to standard MSE loss (default ``1.0``).
n_chunks : int
    Number of ordered time chunks (default ``16``).
t_col : int
    Column index of the time coordinate in the input arrays
    (default ``0`` — first column).
"""

import jax
import jax.numpy as jnp
import numpy as np

from .scheduler_base import Scheduler


class SchedulerCausal(Scheduler):
    """
    Soft causal loss weighting for time-dependent PDE terms.

    See module docstring for full description.
    """

    def __init__(
        self,
        term = "pde",
        tol: float = 1.0,
        n_chunks: int = 16,
        t_col: int = 0,
        combine: str = "min",
    ):
        # Accept a single string or a list of strings.
        if isinstance(term, str):
            self.terms = [term]
        else:
            self.terms = list(term)
        # Keep self.term for backward-compat (points to first term).
        self.term = self.terms[0]
        self.tol = float(tol)
        self.n_chunks = int(n_chunks)
        self.t_col = int(t_col)
        if combine not in ("min", "mean"):
            raise ValueError(f"combine must be 'min' or 'mean', got {combine!r}")
        self.combine = combine

        self._base_weights: dict = {}
        self._sort_idxs: dict = {}
        self._needs_state_rebuild: bool = True

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_compile(self, trainer) -> None:
        """Record base weights; compute initial sort permutations."""
        for t in self.terms:
            self._base_weights[t] = float(trainer.weights.get(t, 1.0))
        self._update_sort(trainer)
        self._needs_state_rebuild = True

    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Refresh sort permutations each epoch (handles resampling)."""
        self._update_sort(trainer)
        self._needs_state_rebuild = True

    # ------------------------------------------------------------------
    # Sort helper
    # ------------------------------------------------------------------

    def _update_sort(self, trainer) -> None:
        """Compute time-sort permutations from current training data for each term.

        Handles prefix-expanded keys: ``"initial"`` in train_data may appear
        as ``"initial_u"``, ``"initial_v"``, etc.  We use any matching key
        (they share the same collocation points, just different targets).
        """
        train_data = trainer._train_data or {}
        for t in self.terms:
            # Try exact match first, then prefix match
            if t in train_data:
                data = train_data[t]
            else:
                prefix = t + "_"
                matches = [v for k, v in train_data.items() if k.startswith(prefix)]
                data = matches[0] if matches else None
            if data is None:
                continue
            t_vals = np.asarray(data[:, self.t_col])
            self._sort_idxs[t] = jnp.asarray(
                np.argsort(t_vals, kind="stable"), dtype=jnp.int32
            )

    # ------------------------------------------------------------------
    # JIT-state protocol
    # ------------------------------------------------------------------

    def get_jit_state(self) -> dict:
        """Return sort permutations as JAX arrays, keyed by term name."""
        state = {}
        for t in self.terms:
            idx = self._sort_idxs.get(t)
            state[f"sort_idx_{t}"] = (
                idx if idx is not None else jnp.array([0], dtype=jnp.int32)
            )
        return state

    def extra_loss(self, residuals: dict, jit_state: dict):
        """No extra loss — weighting is handled via term_weights."""
        return jnp.array(0.0)

    @staticmethod
    def _matching_keys(base: str, residuals: dict):
        """Return all residual keys that correspond to a base term name.

        Matches the exact key *and* any split suffixes produced by
        ``_split_residual``, e.g. ``"pde"`` matches ``"pde"``, ``"pde_0"``,
        ``"pde_1"``, ``"pde_u"``, ``"pde_v"``, etc.
        """
        prefix = base + "_"
        return [k for k in residuals if k == base or k.startswith(prefix)]

    def _compute_causal_weights(self, sort_idx, residuals_for_term: list):
        """Compute causal per-sample weights from a list of residual arrays.

        ``combine='min'`` (default / jaxpi-style):
            Compute independent weights per sub-term then take element-wise min.
            The causal front advances only as fast as the *slowest* species.

        ``combine='mean'``:
            Drive the schedule from the mean squared residual across all
            sub-terms (original behaviour).
        """
        n_chunks = self.n_chunks
        N = residuals_for_term[0].flatten().shape[0]
        N_trim = (N // n_chunks) * n_chunks
        chunk_size = N_trim // n_chunks

        if self.combine == "min":
            # Per-sub-term independent causal weights, then element-wise min
            causal_ws = []
            for r in residuals_for_term:
                r_sorted = r.flatten()[sort_idx]
                chunk_losses = jnp.mean(
                    r_sorted[:N_trim].reshape(n_chunks, chunk_size) ** 2, axis=1
                )
                prev_cumsum = jnp.concatenate(
                    [jnp.zeros(1), jnp.cumsum(chunk_losses[:-1])]
                )
                causal_ws.append(
                    jax.lax.stop_gradient(jnp.exp(-self.tol * prev_cumsum))
                )
            causal_w = jnp.stack(causal_ws, axis=1).min(axis=1)  # (n_chunks,)
        else:  # "mean"
            r_stacked = jnp.stack(
                [r.flatten() for r in residuals_for_term], axis=1
            )  # (N, K)
            r_sorted = r_stacked[sort_idx]
            chunk_losses = jnp.mean(
                r_sorted[:N_trim].reshape(n_chunks, chunk_size, -1) ** 2, axis=(1, 2)
            )
            prev_cumsum = jnp.concatenate(
                [jnp.zeros(1), jnp.cumsum(chunk_losses[:-1])]
            )
            causal_w = jax.lax.stop_gradient(
                jnp.exp(-self.tol * prev_cumsum)
            )

        w_sorted = jnp.repeat(causal_w, chunk_size)  # (N_trim,)
        if N > N_trim:
            w_sorted = jnp.concatenate(
                [w_sorted, jnp.ones(N - N_trim) * causal_w[-1]]
            )

        unsort_idx = jnp.argsort(sort_idx)
        w = w_sorted[unsort_idx]
        w = w / (jnp.mean(w) + 1e-8)
        return w

    def term_weights(self, residuals: dict, jit_state: dict) -> dict:
        """Return per-sample causal weights for every registered term.

        Handles split sub-terms automatically: ``term="pde"`` will match
        ``pde_0``, ``pde_1``, etc. and apply the same causal weights to all.
        The causal schedule is computed from the combined (mean) residual of
        all sub-terms so all outputs advance in lock-step.
        """
        out = {}
        for t in self.terms:
            sort_idx = jax.lax.stop_gradient(jit_state[f"sort_idx_{t}"])

            # Find all sub-keys for this base term (exact or split suffixes)
            keys = self._matching_keys(t, residuals)
            if not keys:
                continue

            r_list = [residuals[k] for k in keys]
            w = self._compute_causal_weights(sort_idx, r_list)
            for k in keys:
                out[k] = w
        return out

    def jit_update(self, jit_state: dict, residuals: dict) -> dict:
        """Causal weighting has no persistent JIT state to update."""
        return jit_state

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def causal_weight_at(self, trainer, term: str = None) -> np.ndarray:
        """Return an array of per-chunk causal weights using the current model.

        Useful for visualising how far the causal front has advanced.

        Parameters
        ----------
        term : str, optional
            Which term to inspect.  Defaults to the first registered term.

        Returns
        -------
        np.ndarray, shape ``(n_chunks,)``
        """
        t = term or self.terms[0]
        residual_fn = getattr(trainer, '_residual_fn_jit', None) or getattr(trainer, '_residual_fn', None)
        if residual_fn is None:
            raise RuntimeError("Call trainer.compile() first.")

        residuals = residual_fn(trainer.network.params, trainer._train_data)

        state    = self.get_jit_state()
        sort_idx = state[f"sort_idx_{t}"]

        # Support split sub-terms (pde_0, pde_1, …)
        keys  = self._matching_keys(t, residuals)
        if not keys:
            return np.zeros(self.n_chunks)
        r_list = [residuals[k] for k in keys]
        w = self._compute_causal_weights(sort_idx, r_list)

        # Recover per-chunk weights from the per-sample array
        n_chunks  = self.n_chunks
        N         = w.shape[0]
        N_trim    = (N // n_chunks) * n_chunks
        chunk_size = N_trim // n_chunks
        sort_idx_np = np.asarray(sort_idx)
        # w is already un-sorted; re-sort to get chunk order
        w_sorted = np.asarray(w)[sort_idx_np]
        causal_w = np.array([w_sorted[i * chunk_size] for i in range(n_chunks)])
        return causal_w
