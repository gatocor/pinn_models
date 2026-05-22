"""
Domain-decomposed ensemble of models for FB-PINN / X-PINN training.

:class:`ModelPartitioned` takes a single :class:`~pinns.modelbase.ModelBase`
(or :class:`~pinns.model.Model`) as a template and replicates it across the
spatiotemporal subdomains **already defined in the domain**.  Each replica is
assigned its own instance of the chosen spatial strategy
(:class:`~pinns.strategies.PartitionFB` or :class:`~pinns.strategies.PartitionX`)
with bounds set to its subdomain.

The domain must be built in *partition mode* for spatial decomposition — i.e.
``DomainCubic(space=[array_of_breakpoints, …])`` so that ``domain.grid_positions``
is set. If the domain has no spatial partition grid ``ModelPartitioned`` uses a
single spatial region covering the full domain.

The global solution is the **sum** of all sub-model predictions:

* ``PartitionFB`` — each sub-model returns a windowed output; the sum
  automatically forms the partition-of-unity approximation.
* ``PartitionX``  — each sub-model returns a hard-masked output; the sum
  assembles the piecewise non-overlapping solution.

Parameters
----------
model : ModelBase | Model
    **Template** network.  Its layers, configuration and strategies are
    deep-copied for every subdomain.  The template itself is *not* modified.
strategy : PartitionFB | PartitionX
    Prototype strategy.  ``ModelPartitioned`` creates one clone per
    subdomain with the correct ``xmin`` / ``xmax`` bounds while preserving
    every other parameter of the prototype (e.g. ``overlap``,
    ``interface_weight``).
partition_time : bool
    When ``True`` (default) *and* the domain carries a time axis:

    * If the domain was built with a **partitioned** time axis
      (``time`` array with >2 breakpoints, i.e. ``domain.time_grid_positions``
      is set), each time subdomain becomes a separate model column.
    * If the domain has a **continuous** time interval ``[t_min, t_max]``,
      the strategy bounds of every replica are extended to cover the full
      time span (no temporal splitting, but the strategy window/mask is
      applied over space+time jointly).

    When ``False`` the strategy bounds contain only the spatial coordinates;
    time is not factored into the window / mask.

Attributes
----------
models : list of ModelBase
    Flat list of all sub-models, ordered as
    ``(s0_0, s0_1, …, t_0), (s0_0, s0_1, …, t_1), …``
    (spatial loops are *innermost*, time is *outermost*).
shape : tuple of int
    Grid dimensions ``(n_s0, n_s1, …, n_t)`` — time only appended when
    ``partition_time=True`` and domain has a partitioned time axis.
n_models : int
    Total number of sub-models (= product of ``shape``).
output_dim : int
    Output dimension (shared by all sub-models).
domain :
    Domain object (from the template model).

Examples
--------
FB-PINN using the domain's own spatial grid::

    from pinns import Model, ModelPartitioned
    from pinns.models.partition import PartitionFB

    domain   = DomainCubic(space=[np.linspace(0, 1, 5)])  # 4 spatial subdomains
    model    = Model(domain, output_dim=1)
    ensemble = ModelPartitioned(model, PartitionFB(overlap=0.3))
    params   = ensemble.init(jax.random.PRNGKey(0))
    y        = ensemble.apply(x, params)          # sum of 4 windowed predictions

X-PINN with 2×2 spatial grid and time partitioning::

    from pinns.models.partition import PartitionX

    domain   = DomainCubic(space=[np.array([0, 0.5, 1.0]), np.array([0, 0.5, 1.0])],
                           time=[0, 0.25, 0.5, 0.75, 1.0])
    model    = Model(domain, output_dim=1)
    ensemble = ModelPartitioned(model, PartitionX(interface_weight=10.0),
                                partition_time=True)
    # 2×2 spatial × 4 time = 16 sub-models
"""

from __future__ import annotations

import copy
import itertools
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from .model_base import ModelBase
from .partition import PartitionFB, PartitionX


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clone_strategy(
    prototype: Union[PartitionFB, PartitionX],
    xmin: np.ndarray,
    xmax: np.ndarray,
    wmin: Optional[np.ndarray] = None,
    wmax: Optional[np.ndarray] = None,
) -> Union[PartitionFB, PartitionX]:
    """Return a new strategy with the same hyper-parameters but new bounds."""
    if isinstance(prototype, PartitionFB):
        return PartitionFB(
            overlap=prototype.overlap,
            window=prototype.window,
            xmin=xmin,
            xmax=xmax,
            wmin=wmin,
            wmax=wmax,
        )
    elif isinstance(prototype, PartitionX):
        return PartitionX(
            interface_weight=prototype.interface_weight,
            flux_weight=prototype.flux_weight,
            xmin=xmin,
            xmax=xmax,
        )
    else:
        raise TypeError(
            f"ModelPartitioned: strategy must be PartitionFB or PartitionX, "
            f"got {type(prototype).__name__!r}."
        )


def _patch_normalize(model: ModelBase, sub_xmin: np.ndarray, sub_xmax: np.ndarray) -> None:
    """Update the Normalize layer in *model* to use the given bounds.

    The original FBPINNs uses the WINDOW (extended) bounds for normalisation,
    so each sub-network normalises its inputs relative to its full window extent.
    """
    from .layers.normalize import Normalize
    import jax.numpy as _jnp
    if not len(sub_xmin):  # nothing to patch (no bounds assigned)
        return
    sub_xmin_f32 = _jnp.array(sub_xmin, dtype=_jnp.float32)
    sub_xmax_f32 = _jnp.array(sub_xmax, dtype=_jnp.float32)
    for layer in getattr(model, '_layers', []):
        if isinstance(layer, Normalize):
            # Only update the coordinate dimensions that the subdomain covers.
            n = min(len(sub_xmin_f32), layer._n_coord)
            if n == layer._n_coord:
                layer._coords_min = sub_xmin_f32
                layer._coords_max = sub_xmax_f32
            else:
                # Partial update (subdomain covers fewer dims than normalizer)
                layer._coords_min = layer._coords_min.at[:n].set(sub_xmin_f32[:n])
                layer._coords_max = layer._coords_max.at[:n].set(sub_xmax_f32[:n])


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ModelPartitioned:
    """Domain-decomposed model ensemble — see module docstring for details."""

    def __init__(
        self,
        model: ModelBase,
        strategy: Union[PartitionFB, PartitionX],
        *,
        partition_time: bool = True,
    ):
        if not isinstance(strategy, (PartitionFB, PartitionX)):
            raise TypeError(
                f"ModelPartitioned: strategy must be a PartitionFB or PartitionX "
                f"instance, got {type(strategy).__name__!r}."
            )

        domain = model.domain
        n_s = domain._spatial_dims  # number of spatial-only dimensions

        # ── Spatial subdomain bounds from domain.grid_positions ──────────── #
        grid = getattr(domain, 'grid_positions', None)  # list of breakpoint arrays or None
        if grid is not None:
            # Domain was built in partition mode: one breakpoint array per spatial dim.
            spatial_ranges = [
                [(float(grid[d][i]), float(grid[d][i + 1])) for i in range(len(grid[d]) - 1)]
                for d in range(n_s)
            ]
            n_per_dim: List[int] = [len(r) for r in spatial_ranges]
        elif n_s > 0:
            # No spatial grid — treat whole domain as a single spatial region.
            spatial_ranges = [[(float(domain.xmin[d]), float(domain.xmax[d]))] for d in range(n_s)]
            n_per_dim = [1] * n_s
        else:
            # Pure-time domain: no spatial dimensions, single "empty" spatial combo.
            spatial_ranges = []
            n_per_dim = []

        # Cartesian product; each element is a tuple of (lo, hi) pairs, one per dim.
        # For pure-time domains (spatial_ranges is empty), product gives one empty tuple.
        spatial_combos: List[Tuple] = list(itertools.product(*spatial_ranges)) if spatial_ranges else [()]

        # ── Temporal subdomain bounds from domain.time_grid_positions ──────── #
        has_time = bool(getattr(domain, 'has_time', False))
        use_time_in_bounds = partition_time and has_time

        temporal_combos: List  # list of (t_lo, t_hi) | None
        if use_time_in_bounds:
            if getattr(domain, 'is_time_partitioned', False):
                # Take breakpoints from the domain's time grid.
                t_breaks = np.asarray(domain.time_grid_positions, dtype=float)
            else:
                # Continuous time interval — one temporal "partition".
                t_breaks = np.array([domain._t_min, domain._t_max], dtype=float)
            temporal_combos = [
                (float(t_breaks[i]), float(t_breaks[i + 1]))
                for i in range(len(t_breaks) - 1)
            ]
        else:
            temporal_combos = [None]  # single sentinel: no temporal splitting

        # ── Build all sub-models ─────────────────────────────────────────── #
        # Layout: iterate time (outer) × spatial (inner) so that consecutive
        # indices share the same temporal window.
        self._models: List[ModelBase] = []
        self._strategies: List = []

        # ── Normal grid-based build ──────────────────────────────────────── #
        for t_range in temporal_combos:
            for s_combo in spatial_combos:
                # Assemble home bounds for this subdomain (used for input normalisation).
                if n_s > 0 and s_combo:
                    sub_xmin = np.array([lo for lo, _hi in s_combo], dtype=float)
                    sub_xmax = np.array([hi for _lo, hi in s_combo], dtype=float)
                else:
                    sub_xmin = np.empty(0, dtype=float)
                    sub_xmax = np.empty(0, dtype=float)

                if t_range is not None:
                    sub_xmin = np.append(sub_xmin, t_range[0])
                    sub_xmax = np.append(sub_xmax, t_range[1])

                # For PartitionFB, compute window parameters based on overlap and
                # subdomain width.  For 'hann': bell centered at interval midpoint,
                # width = overlap × spacing.  For 'cosine'/'sigmoid': flat-top or
                # sigmoid window spanning [sub_xmin, sub_xmax] with transitions of
                # width overlap × home_width on each side.
                if isinstance(strategy, PartitionFB) and len(sub_xmin) > 0:
                    win = getattr(strategy, 'window', 'hann')
                    if win == "hann":
                        # ── Hann bell (original FBPINNs): center-based ── #
                        # center at interval midpoint; width = overlap × interval_spacing.
                        # Each sub-network is normalized to its window bounds [center-w/2, center+w/2],
                        # matching original FBPINNs unnorm=(subdomain_xmin, subdomain_xmax).
                        center = (sub_xmin + sub_xmax) / 2.0
                        spacing = sub_xmax - sub_xmin        # interval width per dim
                        total_w = strategy.overlap * spacing  # window width per dim
                        outer_xmin = center - total_w / 2.0
                        outer_xmax = center + total_w / 2.0
                        s_wmin = s_wmax = total_w
                        new_strategy = _clone_strategy(strategy, center, center, s_wmin, s_wmax)
                        new_strategy.setup(domain)
                        m = copy.deepcopy(model)
                        _patch_normalize(m, outer_xmin, outer_xmax)
                    else:
                        # ── Flat-top cosine / sigmoid: interval-based ─────────── #
                        ov = strategy.overlap
                        width = sub_xmax - sub_xmin
                        s_wmin = s_wmax = 2.0 * ov * width
                        outer_xmin = sub_xmin - ov * width
                        outer_xmax = sub_xmax + ov * width
                        new_strategy = _clone_strategy(strategy, sub_xmin, sub_xmax, s_wmin, s_wmax)
                        new_strategy.setup(domain)
                        m = copy.deepcopy(model)
                        _patch_normalize(m, outer_xmin, outer_xmax)
                else:
                    s_wmin = s_wmax = None
                    new_strategy = _clone_strategy(strategy, sub_xmin, sub_xmax)
                    new_strategy.setup(domain)
                    m = copy.deepcopy(model)
                self._models.append(m)
                self._strategies.append(new_strategy)

        # ── Metadata ────────────────────────────────────────────────────── #
        n_t = len(temporal_combos) if (use_time_in_bounds and len(temporal_combos) > 1) else None
        if n_s > 0:
            shape_parts = tuple(n_per_dim)
        else:
            shape_parts = ()
        if n_t is not None:
            shape_parts = shape_parts + (n_t,)
        self._shape: Tuple[int, ...] = shape_parts if shape_parts else (len(self._models),)

        self._strategy_proto = strategy
        self._partition_time = partition_time
        self._n_spatial_dims = n_s
        self._n_per_dim = n_per_dim
        self._n_temporal_parts = len(temporal_combos)

        # Pre-stack bounds for fast vmapped apply (shape: n_models × n_dims).
        self._all_xmin: np.ndarray = np.stack(
            [s._xmin if s._xmin is not None and len(s._xmin) > 0
             else np.empty(0) for s in self._strategies]
        )
        self._all_xmax: np.ndarray = np.stack(
            [s._xmax if s._xmax is not None and len(s._xmax) > 0
             else np.empty(0) for s in self._strategies]
        )
        # Sigmoid transition widths (wmin/wmax) per sub-model.
        # _all_xmin/_all_xmax are the home breakpoints a_i / b_i (sigmoid centres).
        # outer edges = a_i - wmin/2  and  b_i + wmax/2.
        def _w(s, attr):
            v = getattr(s, attr, None)
            return v if v is not None else (s._xmax - s._xmin) / 2.0 if s._xmax is not None else np.empty(0)
        self._all_wmin: np.ndarray = np.stack([_w(s, '_wmin') for s in self._strategies])
        self._all_wmax: np.ndarray = np.stack([_w(s, '_wmax') for s in self._strategies])
        # Pre-compute outer edges for reference (not used in forward pass,
        # normalization is baked into each sub-model's Normalize layer).
        self._all_outer_xmin: np.ndarray = self._all_xmin - self._all_wmin / 2.0
        self._all_outer_xmax: np.ndarray = self._all_xmax + self._all_wmax / 2.0

        # Pre-stack per-subdomain normalisation bounds for vmapped FB-PINN forward.
        # Each sub-model has its own Normalize layer (patched to window bounds);
        # we extract those bounds here so they can be passed as vmap arguments.
        from .layers.normalize import Normalize as _Normalize
        def _get_norm_bounds(m):
            for layer in getattr(m, '_layers', []):
                if isinstance(layer, _Normalize) and layer._coords_min is not None:
                    return np.array(layer._coords_min), np.array(layer._coords_max)
            dom = m.domain
            return np.asarray(dom.xmin, dtype=np.float32), np.asarray(dom.xmax, dtype=np.float32)
        _nb = [_get_norm_bounds(m) for m in self._models]
        self._all_norm_min: np.ndarray = np.stack([nb[0] for nb in _nb])  # (K, n_dims)
        self._all_norm_max: np.ndarray = np.stack([nb[1] for nb in _nb])  # (K, n_dims)

        # ── Cache JAX arrays and layer split for fast vmapped FB-PINN forward ── #
        # These are computed once here so `apply` never repeats numpy→JAX
        # conversions or Python layer-search on every call.
        if isinstance(strategy, PartitionFB):
            self._jax_norm_min = jnp.array(self._all_norm_min, dtype=jnp.float32)
            self._jax_norm_max = jnp.array(self._all_norm_max, dtype=jnp.float32)
            self._jax_xmin     = jnp.array(self._all_xmin,     dtype=jnp.float32)
            self._jax_xmax     = jnp.array(self._all_xmax,     dtype=jnp.float32)
            self._jax_wmin     = jnp.array(self._all_wmin,     dtype=jnp.float32)
            self._jax_wmax     = jnp.array(self._all_wmax,     dtype=jnp.float32)

            # Split layers at the Normalize boundary (done once).
            _norm_idx_init = next(
                (i for i, l in enumerate(self._models[0]._layers)
                 if isinstance(l, _Normalize)), None
            )
            if _norm_idx_init is not None:
                self._fb_post_layers = self._models[0]._layers[_norm_idx_init + 1:]
                self._fb_post_lnames = self._models[0]._layer_names[_norm_idx_init + 1:]
            else:
                self._fb_post_layers = self._models[0]._layers
                self._fb_post_lnames = self._models[0]._layer_names

        elif isinstance(strategy, PartitionX):
            # Cache arrays for vmapped X-PINN forward.
            self._jax_xmin = jnp.array(self._all_xmin, dtype=jnp.float32)
            self._jax_xmax = jnp.array(self._all_xmax, dtype=jnp.float32)

    # ── Public interface ─────────────────────────────────────────────── #

    @property
    def models(self) -> List[ModelBase]:
        """Flat list of all sub-models."""
        return list(self._models)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Grid shape ``(n_s0, …, n_t)``; time appended when partitioned."""
        return self._shape

    @property
    def n_models(self) -> int:
        """Total number of sub-models."""
        return len(self._models)

    @property
    def output_dim(self) -> int:
        """Output dimension (same for all sub-models)."""
        return self._models[0].output_dim

    @property
    def domain(self):
        """Domain object shared by all sub-models."""
        return self._models[0].domain

    def init(self, rng: "jax.random.PRNGKey") -> dict:
        """
        Initialise all sub-models and return a combined parameter dict.

        Parameters
        ----------
        rng :
            JAX PRNG key; split internally for each sub-model.

        Returns
        -------
        dict
            ``{"sub_0": params_0, "sub_1": params_1, …}``
        """
        params = {}
        for i, m in enumerate(self._models):
            rng, sub = jax.random.split(rng)
            params[f"sub_{i}"] = m.init(sub)
        return params

    def apply(
        self,
        x: "jnp.ndarray",
        params: dict = None,
        params_dict=None,
    ) -> "jnp.ndarray":
        """
        Evaluate the partitioned model at *x*.

        Calls each sub-model's :meth:`~pinns.modelbase.ModelBase.apply`
        (which internally applies the strategy window / mask) and **sums**
        all contributions.

        Parameters
        ----------
        x :
            Input array ``(batch, n_dims)``.
        params :
            Combined parameter dict returned by :meth:`init`.
        params_dict :
            Optional auxiliary dict forwarded to every layer.

        Returns
        -------
        jnp.ndarray  shape ``(batch, output_dim)``
        """
        n = len(self._models)
        # All sub-networks share the same architecture; use the first model's
        # sequential_apply as the vmapped kernel (params differ, structure identical).
        apply_fn = self._models[0]._sequential_apply

        if isinstance(self._strategy_proto, PartitionFB):
            # ── FB-PINN: window-weighted sum with dynamic POU, vmapped ──── #
            n_s = self._all_xmin.shape[1]

            # Use precomputed JAX arrays (no numpy→JAX conversion on every call).
            all_norm_min = self._jax_norm_min.astype(x.dtype)
            all_norm_max = self._jax_norm_max.astype(x.dtype)
            all_xmin_v   = self._jax_xmin.astype(x.dtype)
            all_xmax_v   = self._jax_xmax.astype(x.dtype)
            all_wmin_v   = self._jax_wmin.astype(x.dtype)
            all_wmax_v   = self._jax_wmax.astype(x.dtype)

            # Use precomputed layer split (no Python layer search on every call).
            _post_layers = self._fb_post_layers
            _post_lnames = self._fb_post_lnames

            sub_params_list = [params[f"sub_{i}"] for i in range(n)]
            stacked = jax.tree.map(
                lambda *arrs: jnp.stack(arrs, axis=0), *sub_params_list
            )

            _win_type = getattr(self._strategy_proto, 'window', 'hann')
            _use_hann = _win_type == 'hann'
            _use_cos  = _win_type in ('cosine', 'hann')

            # Hoist params_dict setup outside the vmapped closure.
            pd = dict(params_dict) if params_dict else {}
            pd.setdefault('_x_orig', x)

            def _single_fb(sub_p, nm_min, nm_max, a_i, b_i, wmin_i, wmax_i):
                # 1. Manual normalise to [-1, 1] using per-subdomain window bounds.
                x_n = 2.0 * (x - nm_min) / (nm_max - nm_min + 1e-8) - 1.0
                # 2. Run post-Normalize layers of model 0 (shared architecture).
                for lname, layer in zip(_post_lnames, _post_layers):
                    x_n = layer.apply(x_n, sub_p.get(lname, {}), pd)
                y_i = x_n
                # 3. Window function.
                x_s = x[:, :n_s]
                if _use_cos:
                    wl = wmin_i / 2.0
                    wr = wmax_i / 2.0
                    def _cos1d(x_d, a_d, b_d, wl_d, wr_d):
                        lo = a_d - wl_d
                        hi = b_d + wr_d
                        lr = 0.5 * (1.0 - jnp.cos(jnp.pi * (x_d - lo) / wl_d))
                        rr = 0.5 * (1.0 + jnp.cos(jnp.pi * (x_d - b_d) / wr_d))
                        return jnp.where(x_d < lo, 0.0,
                               jnp.where(x_d < a_d,  lr,
                               jnp.where(x_d <= b_d, 1.0,
                               jnp.where(x_d <= hi,  rr, 0.0))))
                    w = jnp.ones((x_s.shape[0], 1), dtype=x.dtype)
                    for d in range(n_s):
                        w = w * _cos1d(x_s[:, d:d+1], a_i[d], b_i[d], wl[d], wr[d])
                    if _use_hann:
                        w = w ** 2
                else:
                    tol   = 1e-8
                    t_val = jnp.log((1.0 - tol) / tol)
                    sd_l  = wmin_i / (2.0 * t_val)
                    sd_r  = wmax_i / (2.0 * t_val)
                    ws    = (jax.nn.sigmoid((x_s - a_i) / sd_l)
                             * jax.nn.sigmoid((b_i - x_s) / sd_r))
                    w     = jnp.prod(ws, axis=-1, keepdims=True)
                return w * y_i, w

            all_uw, all_w = jax.vmap(_single_fb)(
                stacked, all_norm_min, all_norm_max,
                all_xmin_v, all_xmax_v, all_wmin_v, all_wmax_v,
            )  # (K, batch, out), (K, batch, 1)

            sum_w  = jnp.sum(all_w,  axis=0)   # (batch, 1)
            sum_uw = jnp.sum(all_uw, axis=0)   # (batch, out)
            return sum_uw / jnp.where(sum_w > 0, sum_w, 1.0)

        elif isinstance(self._strategy_proto, PartitionX):
            # ── X-PINN: hard-masked sum, fully vmapped ─────────────────── #
            n_s = self._all_xmin.shape[1]
            all_xmin = self._jax_xmin.astype(x.dtype)
            all_xmax = self._jax_xmax.astype(x.dtype)

            sub_params_list = [params[f"sub_{i}"] for i in range(n)]
            stacked = jax.tree.map(
                lambda *arrs: jnp.stack(arrs, axis=0), *sub_params_list
            )

            def _single_x(sub_p, xmin_i, xmax_i):
                y_i = apply_fn(sub_p, x, params_dict)           # (batch, out)
                x_s = x[:, :n_s]
                inside = jnp.all(
                    (x_s >= xmin_i) & (x_s <= xmax_i),
                    axis=-1, keepdims=True,
                )                                               # (batch, 1)
                return jnp.where(inside, y_i, jnp.zeros_like(y_i))

            all_out = jax.vmap(_single_x)(stacked, all_xmin, all_xmax)
            return jnp.sum(all_out, axis=0)

        else:
            # ── Fallback: Python loop for unknown strategy types ─────────── #
            total = None
            for i, (m, strategy) in enumerate(zip(self._models, self._strategies)):
                y = strategy.predict(m._sequential_apply, params[f"sub_{i}"], x, params_dict)
                total = y if total is None else total + y
            return total  # type: ignore[return-value]

    @property
    def n_context(self) -> int:
        """Number of context columns (shared across all sub-models)."""
        return self._models[0].n_context

    def set_context(
        self,
        n_context: int,
        context_range=None,
    ) -> "ModelPartitioned":
        """Propagate :meth:`~pinns.models.model_base.ModelBase.set_context` to all sub-models."""
        for m in self._models:
            m.set_context(n_context, context_range)
        return self

    def add_constraint(
        self,
        value,
        *,
        region: str = "all",
        output_idx=None,
        sigma=None,
    ) -> "ModelPartitioned":
        """Append the same hard boundary constraint to all sub-models.

        Delegates to :meth:`~pinns.models.model_base.ModelBase.add_constraint` on
        every sub-model so each subdomain enforces the constraint independently.

        Returns
        -------
        ModelPartitioned
            ``self`` for method chaining.
        """
        for m in self._models:
            m.add_constraint(value, region=region,
                             output_idx=output_idx, sigma=sigma)
        return self

    def __len__(self) -> int:
        return len(self._models)

    def __getitem__(self, idx: int) -> ModelBase:
        return self._models[idx]

    def __iter__(self):
        return iter(self._models)

    # ── Repr ─────────────────────────────────────────────────────────── #

    def __repr__(self) -> str:
        strategy_name = type(self._strategy_proto).__name__
        shape_str = "×".join(str(n) for n in self._shape)
        header = (
            f"ModelPartitioned(shape={shape_str}, strategy={strategy_name}, "
            f"n_models={self.n_models}, output_dim={self.output_dim})"
        )
        lines = [header]
        show = min(self.n_models, 6)
        for i in range(show):
            s = self._strategies[i]
            prefix = "  └─ " if i == show - 1 and show == self.n_models else "  ├─ "
            lines.append(f"{prefix}sub_{i}: {s!r}")
        if self.n_models > show:
            lines.append(f"  └─ … ({self.n_models - show} more)")
        return "\n".join(lines)


__all__ = ["ModelPartitioned"]
