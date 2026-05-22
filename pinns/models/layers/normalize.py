"""
Normalize and Denormalize layers for composable ModelBase pipelines.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple


class Normalize:
    """
    Affine normalisation layer.

    Spatial and time coordinates are mapped to ``[-1, 1]`` using the domain
    bounds.  Trailing *context* columns (e.g. ``u_t``) are rescaled using
    ``context_range`` if provided, otherwise passed through unchanged.

    Parameters
    ----------
    context_range : list of (min, max) pairs, optional
        One pair per context column.  If omitted, context columns are
        forwarded as-is.
    """

    def __init__(self, context_range: Optional[List[Tuple[float, float]]] = None):
        self.context_range = context_range
        self._coords_min: Optional[jnp.ndarray] = None
        self._coords_max: Optional[jnp.ndarray] = None
        self._ctx_min: Optional[jnp.ndarray] = None
        self._ctx_max: Optional[jnp.ndarray] = None
        self._n_coord: int = 0
        self._n_context: int = 0

    def _configure(self, network, input_dim: int) -> int:
        domain  = network.domain
        n_ctx   = network.n_context

        spatial_dims = domain._spatial_dims
        if hasattr(domain, '_vertices'):
            xmin = domain._vertices[:, :spatial_dims].min(axis=0)
            xmax = domain._vertices[:, :spatial_dims].max(axis=0)
            # DomainMesh stores only spatial coords in _vertices; append time separately.
            t_interval = getattr(domain, "t_interval", None)
            if t_interval is not None:
                xmin = np.append(xmin, float(t_interval[0]))
                xmax = np.append(xmax, float(t_interval[1]))
        else:
            # DomainCubic and similar expose xmin/xmax that already include
            # the time dimension when a time axis is present.
            xmin = np.asarray(domain.xmin, dtype=np.float64)
            xmax = np.asarray(domain.xmax, dtype=np.float64)

        self._n_coord   = len(xmin)
        self._n_context = n_ctx
        self._coords_min = jnp.array(xmin, dtype=jnp.float32)
        self._coords_max = jnp.array(xmax, dtype=jnp.float32)

        if n_ctx > 0 and self.context_range is not None:
            assert len(self.context_range) == n_ctx, (
                f"context_range has {len(self.context_range)} pairs but "
                f"network.n_context={n_ctx}"
            )
            self._ctx_min = jnp.array([r[0] for r in self.context_range], dtype=jnp.float32)
            self._ctx_max = jnp.array([r[1] for r in self.context_range], dtype=jnp.float32)

        network._register_coord_transform(self._transform_coords)
        return input_dim  # Normalize is shape-preserving

    def _transform_coords(self, coords: np.ndarray) -> np.ndarray:
        """Numpy-level transform for mesh vertices (used by lazy mesh layers).

        Slices min/max to match the number of columns in ``coords`` so that
        purely-spatial arrays (shape ``(N, 2)``) work correctly even when the
        normalisation was configured for space+time inputs (shape ``(N, 3)``).
        """
        n_cols = coords.shape[1]
        mn = np.array(self._coords_min)[:n_cols]
        mx = np.array(self._coords_max)[:n_cols]
        return 2.0 * (coords - mn) / (mx - mn + 1e-8) - 1.0

    def init(self, rng) -> Dict:
        return {}

    def apply(self, x: jnp.ndarray, params: Dict = None, params_dict=None) -> jnp.ndarray:
        coords   = x[:, : self._n_coord]
        coords_n = (
            2.0 * (coords - self._coords_min)
            / (self._coords_max - self._coords_min + 1e-8)
            - 1.0
        )
        parts = [coords_n]
        if self._n_context > 0:
            ctx = x[:, -self._n_context:]
            if self._ctx_min is not None:
                ctx = (
                    2.0 * (ctx - self._ctx_min)
                    / (self._ctx_max - self._ctx_min + 1e-8)
                    - 1.0
                )
            parts.append(ctx)
        return jnp.concatenate(parts, axis=-1)

    def __repr__(self) -> str:
        return (
            f"Normalize(n_coord={self._n_coord}, n_context={self._n_context}, "
            f"context_scaled={self._ctx_min is not None})"
        )


class Denormalize:
    """
    Affine de-normalisation layer.

    Maps FNN output (assumed in ``[-1, 1]``) back to physical range using
    ``network.output_range``.  If no ``output_range`` is set, acts as
    identity.
    """

    def __init__(self):
        self._out_min: Optional[jnp.ndarray] = None
        self._out_max: Optional[jnp.ndarray] = None

    def _configure(self, network, input_dim: int) -> int:
        if network.output_range is not None:
            r = network.output_range
            if isinstance(r[0], (int, float)):
                self._out_min = jnp.array([float(r[0])], dtype=jnp.float32)
                self._out_max = jnp.array([float(r[1])], dtype=jnp.float32)
            else:
                self._out_min = jnp.array([x[0] for x in r], dtype=jnp.float32)
                self._out_max = jnp.array([x[1] for x in r], dtype=jnp.float32)
        return input_dim

    def init(self, rng) -> Dict:
        return {}

    def apply(self, x: jnp.ndarray, params: Dict = None, params_dict=None) -> jnp.ndarray:
        if self._out_min is None:
            return x
        return (x + 1.0) / 2.0 * (self._out_max - self._out_min) + self._out_min

    def __repr__(self) -> str:
        return f"Denormalize(scaled={self._out_min is not None})"


__all__ = ["Normalize", "Denormalize"]
