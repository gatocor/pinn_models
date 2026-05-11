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
    y        = ensemble.apply(params, x)          # sum of 4 windowed predictions

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
from typing import List, Tuple, Union

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
) -> Union[PartitionFB, PartitionX]:
    """Return a new strategy with the same hyper-parameters but new bounds."""
    if isinstance(prototype, PartitionFB):
        return PartitionFB(
            overlap=prototype.overlap,
            continuity_weight=prototype.continuity_weight,
            xmin=xmin,
            xmax=xmax,
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
            # Pure-time domain.
            spatial_ranges = [[]]
            n_per_dim = [1]

        # Cartesian product; each element is a tuple of (lo, hi) pairs, one per dim.
        spatial_combos: List[Tuple] = list(itertools.product(*spatial_ranges))

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

        for t_range in temporal_combos:
            for s_combo in spatial_combos:
                # Assemble bounds for this subdomain.
                if n_s > 0 and s_combo:
                    sub_xmin = np.array([lo for lo, _hi in s_combo], dtype=float)
                    sub_xmax = np.array([hi for _lo, hi in s_combo], dtype=float)
                else:
                    sub_xmin = np.empty(0, dtype=float)
                    sub_xmax = np.empty(0, dtype=float)

                if t_range is not None:
                    sub_xmin = np.append(sub_xmin, t_range[0])
                    sub_xmax = np.append(sub_xmax, t_range[1])

                # Create a new strategy instance with these bounds.
                new_strategy = _clone_strategy(strategy, sub_xmin, sub_xmax)
                new_strategy.setup(domain)

                # Deep-copy the template model.
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
        params: dict,
        x: "jnp.ndarray",
        params_dict=None,
    ) -> "jnp.ndarray":
        """
        Evaluate the partitioned model at *x*.

        Calls each sub-model's :meth:`~pinns.modelbase.ModelBase.apply`
        (which internally applies the strategy window / mask) and **sums**
        all contributions.

        Parameters
        ----------
        params :
            Combined parameter dict returned by :meth:`init`.
        x :
            Input array ``(batch, n_dims)``.
        params_dict :
            Optional auxiliary dict forwarded to every layer.

        Returns
        -------
        jnp.ndarray  shape ``(batch, output_dim)``
        """
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
