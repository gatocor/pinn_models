"""
Periodic coordinate embedding for Physics-Informed Neural Networks.

Replaces one or more coordinates with their Fourier harmonics:
    x  →  [cos(k₁ · π/L · x),  sin(k₁ · π/L · x),
            cos(k₂ · π/L · x),  sin(k₂ · π/L · x), ...]

where L = (xmax - xmin) / 2 is the half-period of the domain along that axis.
All other columns (other spatial dims, time) pass through unchanged.

This hard-encodes the periodicity into the input representation so that any
downstream layer (e.g. FourierFeatures, PirateNet) automatically respects it,
without needing a soft penalty loss term.
"""

from __future__ import annotations

import jax.numpy as jnp
from typing import Dict, List, Optional, Sequence, Tuple, Union


_SPATIAL_AXIS_NAMES = {"x": 0, "y": 1, "z": 2}

# A single component spec: name/int, or a list of them.
_ComponentSpec = Union[str, int, Sequence[Union[str, int]]]


class PeriodicEmbedding:
    """
    Replaces one or more coordinates with their Fourier harmonics.

    Parameters
    ----------
    component : str, int, or list of str/int
        Axis (or axes) to embed periodically.  Each entry is either a name
        (``"x"``, ``"y"``, ``"z"``) or a 0-based integer column index.
        A single value is treated as a one-element list.
    k : sequence of int, or list of sequences of int
        Harmonic mode numbers.  Can be:

        * A flat sequence (e.g. ``[1]`` or ``[1, 2, 3]``) — same modes applied
          to **every** component.
        * A list of sequences (e.g. ``[[1, 2], [1]]``) — per-component modes,
          must have the same length as *component*.

        Each mode ``k_i`` contributes a ``cos`` and a ``sin`` column, so the
        total number of new columns per component is ``2 * len(k_i)``.

    Notes
    -----
    The fundamental frequency for each axis is ``π / L`` where
    ``L = (xmax - xmin) / 2``.  For ``k=[1]`` and ``x ∈ [-1, 1]`` this gives
    ``cos(π x), sin(π x)``, exactly matching jaxpi's ``PeriodEmbs``.

    The output column order is: embedded components (in ascending axis order,
    replacing their original column) followed by all remaining (non-embedded)
    columns unchanged.

    Examples
    --------
    ::

        # Single axis
        net.add(PeriodicEmbedding("x"))
        # [x, y, t] → [cos(πx), sin(πx), y, t]

        # Two axes, shared modes
        net.add(PeriodicEmbedding(["x", "y"], k=[1]))
        # [x, y, t] → [cos(πx), sin(πx), cos(πy), sin(πy), t]

        # Two axes, different modes per axis
        net.add(PeriodicEmbedding(["x", "y"], k=[[1, 2], [1]]))
        # [x, y, t] → [cos(πx), sin(πx), cos(2πx), sin(2πx), cos(πy), sin(πy), t]
    """

    def __init__(
        self,
        component: _ComponentSpec = "x",
        k: Union[Sequence[int], Sequence[Sequence[int]]] = (1,),
    ):
        # Normalise component to a list
        if isinstance(component, (str, int)):
            self.components: List[Union[str, int]] = [component]
        else:
            self.components = list(component)

        # Normalise k: flat sequence → same for all; list of sequences → per-component
        if len(self.components) > 0 and hasattr(k[0], '__iter__') and not isinstance(k[0], int):
            # list of sequences
            if len(k) != len(self.components):
                raise ValueError(
                    f"PeriodicEmbedding: k has {len(k)} entries but {len(self.components)} "
                    "components were given; they must match."
                )
            self.k_per_component: List[List[int]] = [list(ki) for ki in k]
        else:
            # flat sequence — broadcast to all components
            self.k_per_component = [list(k)] * len(self.components)

        # Set by _configure:
        # list of (axis_index, freq) pairs, one per component, sorted by axis
        self._entries:    Optional[List[Tuple[int, float, List[int]]]] = None
        self._embed_axes: Optional[set]  = None
        self._input_dim:  Optional[int]  = None
        self._output_dim: Optional[int]  = None

    # ── ModelBase composable protocol ──────────────────────────────────────── #

    def _configure(self, network, input_dim: int) -> int:
        domain    = network.domain
        n_spatial = domain._spatial_dims
        t_interval = getattr(domain, "t_interval", None)

        entries: List[Tuple[int, float, List[int]]] = []
        seen_axes = set()

        for comp, k_modes in zip(self.components, self.k_per_component):
            # Resolve axis index and bounds
            if isinstance(comp, str):
                name = comp.lower()
                if name not in _SPATIAL_AXIS_NAMES:
                    raise ValueError(
                        f"PeriodicEmbedding: unknown component name '{comp}'. "
                        "Use 'x', 'y', 'z' or an integer index."
                    )
                axis = _SPATIAL_AXIS_NAMES[name]
                if axis >= n_spatial:
                    raise ValueError(
                        f"PeriodicEmbedding: named axis '{comp}' maps to index {axis} "
                        f"but domain only has {n_spatial} spatial dimension(s)."
                    )
                xmin = float(domain.xmin[axis])
                xmax = float(domain.xmax[axis])
            else:
                axis = int(comp)
                if axis < n_spatial:
                    xmin = float(domain.xmin[axis])
                    xmax = float(domain.xmax[axis])
                elif axis == n_spatial and t_interval is not None:
                    xmin = float(t_interval[0])
                    xmax = float(t_interval[1])
                else:
                    total = n_spatial + (1 if t_interval is not None else 0)
                    raise ValueError(
                        f"PeriodicEmbedding: column index {axis} is out of range; "
                        f"domain has {total} column(s)."
                    )

            if axis in seen_axes:
                raise ValueError(
                    f"PeriodicEmbedding: axis {axis} appears more than once."
                )
            seen_axes.add(axis)

            L    = (xmax - xmin) / 2.0
            freq = float(jnp.pi / L)
            entries.append((axis, freq, k_modes))

        # Sort by axis so output order is deterministic (ascending column order)
        entries.sort(key=lambda e: e[0])

        # Output dim: each embedded column → 2*len(k_modes) cols; everything else passes through
        n_embedded = len(entries)
        extra = sum(2 * len(km) - 1 for _, _, km in entries)   # net gain per component
        out_dim = input_dim + extra

        self._entries    = entries
        self._embed_axes = seen_axes
        self._input_dim  = input_dim
        self._output_dim = out_dim
        return out_dim

    def init(self, rng) -> dict:
        """No trainable parameters."""
        return {}

    def apply(self, params: dict, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        return self._forward(x)

    # ── Forward pass ───────────────────────────────────────────────────────── #

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        assert self._entries is not None, (
            "PeriodicEmbedding not configured — add to a ModelBase first"
        )
        n_cols = x.shape[1]
        embed_axes = self._embed_axes

        # Build a mapping axis → list of harmonic columns
        harmonics: Dict[int, List[jnp.ndarray]] = {}
        for axis, freq, k_modes in self._entries:
            x_col = x[:, axis : axis + 1]
            cols = []
            for ki in k_modes:
                angle = ki * freq * x_col
                cols.append(jnp.cos(angle))
                cols.append(jnp.sin(angle))
            harmonics[axis] = cols

        # Walk through columns in order, replacing embedded ones with harmonics
        parts = []
        for col in range(n_cols):
            if col in embed_axes:
                parts.extend(harmonics[col])
            else:
                parts.append(x[:, col : col + 1])

        return jnp.concatenate(parts, axis=-1)

    def __call__(self, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        return self._forward(x)

    def __repr__(self) -> str:
        if self._entries is None:
            return (
                f"PeriodicEmbedding(components={self.components!r}, "
                f"k={self.k_per_component})"
            )
        entries_str = ", ".join(
            f"axis={ax} freq={freq:.4f} k={km}"
            for ax, freq, km in self._entries
        )
        return (
            f"PeriodicEmbedding([{entries_str}], "
            f"input_dim={self._input_dim}, output_dim={self._output_dim})"
        )


__all__ = ["PeriodicEmbedding"]
