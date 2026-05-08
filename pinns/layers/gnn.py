"""
GNN-based spatial feature encoder for mesh PINNs (JAX/Flax).

GNNFeatures runs a trainable Chebyshev-GNN on the fixed spatial mesh and
produces a hidden_dim-dimensional embedding at every mesh node; those
embeddings are then barycentric-interpolated at arbitrary query points.

This transform is the GNN analogue of LaplacianFeatures: it can be composed
with any base model (FNN, ResNet, PirateNet) as a ``feature_encoding``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional

from pinns.backends.jax.gnn_network import (
    _GNNEncoderModule,
    _build_mesh_arrays,
    _interpolate_mesh,
    _barycentric_coords_all_triangles,
)


def _query_context_to_nodes(
    ctx: jnp.ndarray,       # (n_query, n_context)
    x_spatial: jnp.ndarray, # (n_query, 2)
    nodes: jnp.ndarray,     # (n_nodes, 2)
    faces: jnp.ndarray,     # (n_faces, 3)  int
    n_nodes: int,
) -> jnp.ndarray:           # (n_nodes, n_context)
    """
    Scatter query-point context values onto mesh nodes using the transpose
    of barycentric interpolation (weighted scatter + normalise).

    For each query point p in triangle (n0, n1, n2) with barycentric weights
    (w0, u0, v0), contributes  w0*ctx_p → n0,  u0*ctx_p → n1,  v0*ctx_p → n2.
    Each node's value is then divided by its total accumulated weight, giving a
    weighted mean of nearby context values.  Nodes with no query points receive 0.
    """
    u, v, w = _barycentric_coords_all_triangles(nodes, faces, x_spatial)
    # u, v, w each (n_query, n_faces)

    min_coord = jnp.minimum(jnp.minimum(u, v), w)
    tri_idx   = jnp.argmax(min_coord, axis=-1)   # (n_query,)

    n_query = x_spatial.shape[0]
    row     = jnp.arange(n_query)

    u_sel = u[row, tri_idx]  # (n_query,)
    v_sel = v[row, tri_idx]
    w_sel = w[row, tri_idx]

    n0 = faces[tri_idx, 0]   # (n_query,)
    n1 = faces[tri_idx, 1]
    n2 = faces[tri_idx, 2]

    # Scatter weighted context values and weight sums.
    def scatter(node_ids, weights):
        vals  = jnp.zeros((n_nodes, ctx.shape[1]))
        wsum  = jnp.zeros((n_nodes, 1))
        vals  = vals.at[node_ids].add(weights[:, None] * ctx)
        wsum  = wsum.at[node_ids].add(weights[:, None])
        return vals, wsum

    v0a, w0a = scatter(n0, w_sel)
    v1a, w1a = scatter(n1, u_sel)
    v2a, w2a = scatter(n2, v_sel)

    total_val  = v0a + v1a + v2a
    total_wsum = w0a + w1a + w2a
    total_wsum = jnp.where(total_wsum < 1e-12, 1.0, total_wsum)
    return total_val / total_wsum

class GNNFeatures:
    """
    Trainable GNN-based spatial feature encoder for mesh PINNs.

    Runs a ChebConv GNN on the fixed triangular mesh and produces a
    ``hidden_dim``-wide embedding at every node; embeddings are then
    barycentric-interpolated at query points.

    **Time handling** — if *domain* carries a ``t_interval``:

    * The GNN sees only spatial node features ``[x_norm, y_norm]``.
    * After interpolation, the raw ``t`` coordinate is appended.
    * ``output_dim = hidden_dim + 1``.

    Parameters
    ----------
    domain : DomainMesh
        2-D triangular mesh domain.
    hidden_dim : int
        Width of GNN hidden layers and output embedding.
    poly_order : int
        Chebyshev polynomial order (higher = wider spectral range).
    message_steps : int
        Number of stacked ChebConv layers.
    activation : str
        Activation for the GNN encoder (default ``'relu'``).
    n_context : int
        Number of extra context columns in the input *after* spatial (+time)
        coordinates (e.g. solution values U from the previous time step).
        These are interpolated to mesh nodes and fed into the GNN message
        passing as additional node features, then also appended raw at the end
        of the output (skip connection).  ``output_dim`` increases by
        ``n_context``.

    Example::

        enc   = GNNFeatures(domain, hidden_dim=64)
        net   = FNN(layer_sizes=[enc.output_dim, 128, 1], feature_encoding=enc)
        enc_p = enc.init(jax.random.PRNGKey(0))
        net_p = net.init(jax.random.PRNGKey(1))
        # merge and pass as a combined params dict to the trainer
    """

    def __init__(
        self,
        domain=None,
        hidden_dim: int = 64,
        poly_order: int = 10,
        message_steps: int = 4,
        activation: str = "relu",
        n_context: int = 0,
    ):
        self.hidden_dim    = hidden_dim
        self.poly_order    = poly_order
        self.message_steps = message_steps
        self.activation    = activation
        self.n_context     = n_context

        # These are set by _build() or _configure()
        self.has_time     = None
        self.spatial_dims = None
        self.output_dim   = None
        self._nodes_np    = None
        self._faces_np    = None
        self._module      = None

        if domain is not None:
            self._build_from_domain(domain)

    def _build_from_domain(self, domain, coord_transform=None):
        """Extract mesh from domain and call _build(), optionally transforming verts."""
        t_interval = getattr(domain, "t_interval", None)
        self.has_time     = t_interval is not None
        self.spatial_dims = domain._spatial_dims
        if self.has_time:
            self.t_min = float(t_interval[0])
            self.t_max = float(t_interval[1])

        verts = domain._vertices[:, : self.spatial_dims].astype(np.float32)
        if coord_transform is not None:
            verts = coord_transform(verts)
        faces = domain._faces.astype(np.int32)
        self._build(verts, faces)

    def _build(self, verts: np.ndarray, faces: np.ndarray):
        """Build mesh arrays and GNN module from (optionally transformed) vertices."""
        self._nodes_np = verts
        self._faces_np = faces
        self.n_nodes   = len(verts)
        self.output_dim = (
            self.hidden_dim
            + (1 if self.has_time else 0)
            + self.n_context
        )

        (
            self._edge_src,
            self._edge_dst,
            self._edge_weights,
            self._nodes_norm,
        ) = _build_mesh_arrays(verts, faces)

        self._module = _GNNEncoderModule(
            encoder_in_dim=self.spatial_dims + self.n_context,
            hidden_dim=self.hidden_dim,
            poly_order=self.poly_order,
            message_steps=self.message_steps,
            activation=self.activation,
        )

    # ── Network composable protocol ───────────────────────────────────── #

    def _configure(self, network, input_dim: int) -> int:
        """Called by Network.add().  Builds mesh using network's domain and
        any accumulated coordinate transforms from preceding Normalize layers."""
        self._build_from_domain(
            network.domain,
            coord_transform=network._apply_coord_transforms
            if network._coord_transforms else None,
        )
        return self.output_dim

    def apply(
        self,
        params: Dict,
        x,
        params_dict=None,
    ):
        """Network-protocol alias: apply(params, x) → forward pass."""
        return self.__call__(params, x, params_dict)

    # ── Trainer-compatible API ─────────────────────────────────────────── #

    def init(self, rng: "jax.random.PRNGKey") -> Dict:
        """Initialise GNN encoder parameters."""
        if self.n_context > 0:
            # Pad _nodes_norm with zero context columns so the Dense encoder
            # kernel is initialised with the correct input dimension.
            ctx_zeros = jnp.zeros(
                (self.n_nodes, self.n_context), dtype=self._nodes_norm.dtype
            )
            node_feats = jnp.concatenate([self._nodes_norm, ctx_zeros], axis=-1)
        else:
            node_feats = self._nodes_norm
        return self._module.init(
            rng,
            node_feats,
            self._edge_src,
            self._edge_dst,
            self._edge_weights,
            self.n_nodes,
        )

    def __call__(
        self,
        enc_params: Dict,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """
        Encode query points into GNN features.

        Parameters
        ----------
        enc_params : dict
            GNN encoder parameter tree from :meth:`init`.
        x : jnp.ndarray, shape ``(n_query, spatial_dims[+1])``
            Query coordinates.  If the domain has a time interval, the last
            column must be ``t``.

        Returns
        -------
        jnp.ndarray, shape ``(n_query, output_dim)``
        """
        nodes_jnp = jnp.array(self._nodes_np)
        faces_jnp = jnp.array(self._faces_np)
        x_spatial  = x[:, : self.spatial_dims]

        # Build per-node feature matrix: spatial coordinates (fixed) +
        # context fields (interpolated from query points to mesh nodes).
        if self.n_context > 0:
            ctx_query = x[:, -self.n_context:]  # (n_query, n_context)
            ctx_node_feats = _query_context_to_nodes(
                ctx_query, x_spatial, nodes_jnp, faces_jnp, self.n_nodes
            )  # (n_nodes, n_context)
            node_feats = jnp.concatenate(
                [self._nodes_norm, ctx_node_feats], axis=-1
            )  # (n_nodes, spatial_dims + n_context)
        else:
            node_feats = self._nodes_norm  # (n_nodes, spatial_dims)

        node_embeds = self._module.apply(
            enc_params,
            node_feats,
            self._edge_src,
            self._edge_dst,
            self._edge_weights,
            self.n_nodes,
        )  # (n_nodes, hidden_dim)

        query_embeds = _interpolate_mesh(node_embeds, nodes_jnp, faces_jnp, x_spatial)
        # (n_query, hidden_dim)

        parts = [query_embeds]
        if self.has_time:
            parts.append(x[:, self.spatial_dims : self.spatial_dims + 1])
        if self.n_context > 0:
            parts.append(x[:, -self.n_context:])
        return jnp.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]

    def transform(
        self,
        enc_params: Dict,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """Alias for ``__call__``."""
        return self.__call__(enc_params, x, params_dict)

    def __repr__(self) -> str:
        mode = f"hidden_dim={self.hidden_dim}" + (" + t" if self.has_time else "")
        ctx = f", n_context={self.n_context}" if self.n_context > 0 else ""
        return f"GNNFeatures(output_dim={self.output_dim}, {mode}{ctx})"


__all__ = ["GNNFeatures"]
