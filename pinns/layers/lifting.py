"""
Lifting layer for hard enforcement of boundary values.

Implements the formula:

    u(x) = g(x) + tanh(d(x) / σ) · NN(x)

where:
  - g(x) is the prescribed boundary value
  - d(x) is the distance from x to the target boundary (exactly 0 on boundary)
  - σ controls the width of the transition zone

At the boundary:  d = 0  →  tanh = 0  →  u = g  (exactly satisfied)
In the interior:  d ≫ σ  →  tanh → 1  →  u ≈ g + NN  (network contributes fully)

The nodal distance field is precomputed once via a KD-tree at layer configuration
time.  At inference, distances at arbitrary query points are obtained by
barycentric interpolation on the mesh, which is JAX-differentiable.

Usage::

    net = Network(domain, output_dim=1)
    net.add(Normalize())
    net.add(FNN([64, 64]))
    net.add(Lifting(value=0.0, region='all'))

    # Multiple output components, each with their own BC:
    net.add(Lifting(value=0.0,  region='wall',   output_idx=0))   # u = 0 on wall
    net.add(Lifting(value=1.0,  region='inlet',  output_idx=1))   # v = 1 on inlet
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from typing import Callable, Optional, Union


class Lifting:
    """Hard-enforcement lifting layer.

    Apply as the **last** layer in a :class:`~pinns.network.Network` pipeline.
    Transforms the raw network output :math:`\\hat{u}(x)` into:

    .. math::

        u(x) = g(x) + \\tanh\\!\\bigl(d(x) / \\sigma\\bigr) \\cdot \\hat{u}(x)

    This requires the original input coordinates, which
    :meth:`~pinns.network.Network.apply` injects automatically as
    ``params_dict['_x_orig']``.

    Parameters
    ----------
    value : scalar, array, or callable
        Target boundary value :math:`g`.  Forms accepted:

        * **scalar / array** — constant, broadcast to all outputs (or to
          ``output_idx`` only).
        * **callable** ``g(x_orig) -> array`` — spatially varying value;
          receives original coordinates ``(batch, n_dims)`` and should return
          ``(batch, 1)`` or ``(batch, n_outputs)``.

    region : str
        ``'all'`` (default) for the full domain boundary, or a named region
        registered via :meth:`~pinns.domain.DomainMesh.add_boundary`.
    output_idx : int, list[int], or None
        Which output component(s) to apply the lifting to.
        ``None`` (default) applies to all outputs.
    sigma : float or None
        Transition half-width in physical units.  Controls how quickly the
        network output is fully expressed away from the boundary.
        ``None`` (default) uses 10 % of the mean spatial domain extent.

    Example::

        net = Network(domain, output_dim=2)
        net.add(Normalize())
        net.add(FNN([128, 128]))
        # u = 0 on all boundaries for both outputs
        net.add(Lifting(value=0.0, region='all'))

        # Separate lifting per component on different regions
        net.add(Lifting(value=0.0, region='wall',  output_idx=0))
        net.add(Lifting(value=1.0, region='inlet', output_idx=1))
    """

    def __init__(
        self,
        value: Union[float, np.ndarray, Callable] = 0.0,
        region: str = 'all',
        output_idx=None,
        sigma: Optional[float] = None,
    ):
        self.value = value
        self.region = region
        self.output_idx = output_idx
        self.sigma = sigma

        # Populated by _configure:
        self._nodal_dist: Optional[np.ndarray] = None  # (n_nodes, 1) float32
        self._nodes: Optional[np.ndarray] = None       # (n_nodes, spatial_dims)
        self._faces: Optional[np.ndarray] = None       # (n_faces, 3) int32
        self._spatial_dims: int = 0
        self._sigma_val: float = 1.0
        self._output_dim: int = 0

    # ── layer protocol ────────────────────────────────────────────────── #

    def _configure(self, network, input_dim: int) -> int:
        """Precompute nodal distances from the boundary via KD-tree.

        Called automatically by :meth:`~pinns.network.Network.add`.

        Parameters
        ----------
        network : Network
            The parent network (provides ``domain``).
        input_dim : int
            Current data width (= network ``output_dim`` when Lifting is last).

        Returns
        -------
        int
            ``input_dim`` unchanged — Lifting does not change the output shape.
        """
        from scipy.spatial import KDTree

        domain = network.domain
        spatial_dims = domain._spatial_dims
        self._spatial_dims = spatial_dims

        verts = domain._vertices[:, :spatial_dims].astype(np.float64)  # (n_nodes, d)
        faces = domain._faces                                            # (n_faces, 3)

        # ── Identify which mesh nodes define the "zero" boundary ─────────
        if self.region == 'all':
            if not hasattr(domain, '_boundary_node_mask'):
                raise AttributeError(
                    "Lifting(region='all') requires domain._boundary_node_mask "
                    "(available on DomainMesh).  For DomainCubic supply a "
                    "distance_fn or use region='name'."
                )
            bnd_mask = domain._boundary_node_mask          # bool (n_nodes,)
            bnd_node_idx = np.where(bnd_mask)[0]
        else:
            regions = getattr(domain, '_boundary_regions', {})
            if self.region not in regions:
                raise ValueError(
                    f"Lifting: region {self.region!r} not registered on the domain. "
                    f"Available: {list(regions.keys())}"
                )
            # For DomainMesh, each region stores explicit node_indices.
            reg = regions[self.region]
            bnd_node_idx = np.unique(
                reg.get('node_indices',
                        np.unique(reg.get('edges', np.empty((0, 2), int)).ravel()))
            )

        if len(bnd_node_idx) == 0:
            raise ValueError(
                f"Lifting: region {self.region!r} contains no boundary nodes."
            )

        # ── Distance from every node to nearest boundary node (KD-tree) ──
        bnd_coords = verts[bnd_node_idx]          # (n_bnd, d)
        tree = KDTree(bnd_coords)
        dist, _ = tree.query(verts, workers=-1)   # (n_nodes,) float64
        dist = np.maximum(dist, 0.0)              # clamp negatives (float rounding)

        self._nodal_dist = dist.astype(np.float32)[:, None]  # (n_nodes, 1)
        self._nodes = verts.astype(np.float32)
        self._faces = faces.astype(np.int32)
        self._output_dim = input_dim

        # ── Sigma: default to 10 % of mean spatial extent ────────────────
        if self.sigma is not None:
            self._sigma_val = float(self.sigma)
        else:
            extents = domain.xmax[:spatial_dims] - domain.xmin[:spatial_dims]
            self._sigma_val = float(0.1 * np.mean(extents))

        return input_dim  # output shape unchanged

    def init(self, rng) -> dict:
        """No trainable parameters."""
        return {}

    def apply(
        self,
        params: dict,
        x: jnp.ndarray,
        params_dict: Optional[dict] = None,
    ) -> jnp.ndarray:
        """Apply the lifting transform.

        Parameters
        ----------
        params : dict
            Empty (no trainable parameters).
        x : jnp.ndarray, shape ``(batch, output_dim)``
            Raw network output from the preceding layers.
        params_dict : dict or None
            Must contain ``'_x_orig'`` — the original input coordinates
            ``(batch, n_dims)`` injected by :meth:`~pinns.network.Network.apply`.

        Returns
        -------
        jnp.ndarray, shape ``(batch, output_dim)``
            Lifted output satisfying ``u = g`` at the boundary nodes.
        """
        from pinns.backends.jax.gnn_network import _interpolate_mesh

        if params_dict is None or '_x_orig' not in params_dict:
            raise RuntimeError(
                "Lifting.apply() requires '_x_orig' in params_dict.  "
                "Use Network.apply() which injects this automatically."
            )

        x_orig    = params_dict['_x_orig']                   # (batch, n_dims)
        x_spatial = x_orig[:, :self._spatial_dims]           # (batch, d)

        nodes = jnp.array(self._nodes)
        faces = jnp.array(self._faces)

        # Interpolate precomputed distance at query points → (batch, 1)
        d = _interpolate_mesh(
            jnp.array(self._nodal_dist),
            nodes, faces, x_spatial,
        )

        # Blending factor: 0 on boundary, → 1 in interior
        f = jnp.tanh(d / self._sigma_val)   # (batch, 1)

        # Target boundary value g  →  (batch, output_dim) or (batch, 1)
        if callable(self.value):
            g = self.value(x_orig)
        else:
            g = jnp.full_like(x, float(self.value))

        transformed = g + f * x

        if self.output_idx is None:
            return transformed

        # Apply only to the selected output column(s)
        idx = ([self.output_idx]
               if isinstance(self.output_idx, int)
               else list(self.output_idx))
        mask = jnp.zeros(x.shape[-1], dtype=bool).at[jnp.array(idx)].set(True)
        return jnp.where(mask[None, :], transformed, x)

    def __repr__(self) -> str:
        v = self.value if not callable(self.value) else '<callable>'
        return (
            f"Lifting(value={v}, region={self.region!r}, "
            f"sigma={self._sigma_val:.3g}, output_idx={self.output_idx})"
        )
