"""
Graph Neural Network for mesh-based PINNs (JAX/Flax).

GNNMeshNetwork maps arbitrary query points to PDE solution values by:
  1. Running message-passing on the fixed spatial mesh to predict nodal
     coefficients (function values at every mesh vertex).
  2. Interpolating the nodal coefficients at the query points via
     barycentric (Lagrange) shape functions on the triangular elements.

The only inputs are (x, t, …) query coordinates; the mesh topology and
vertex positions are stored as constant arrays inside the object.  All
operations are JIT-compatible and fully differentiable w.r.t. both
network parameters *and* query locations (needed for PDE residuals).

Public API mirrors ``FNN``::

    net = GNNMeshNetwork(domain, hidden_dim=64, depth=3, message_steps=4)
    params = net.init(jax.random.PRNGKey(0))
    y = net.apply(params, x)          # (n_query, n_outputs)
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Dict, Optional, Sequence, Callable, Any

from .networks import get_activation


# ---------------------------------------------------------------------------
# Flax sub-modules
# ---------------------------------------------------------------------------

class _ChebConvLayer(nn.Module):
    """
    One Chebyshev spectral graph convolution layer (Eqs. 3–5).

    Computes::

        X^l = act( sum_{k=1}^{K} Z^{l-1,k} @ Theta^{l-1,k} + b^{l-1} )

    where the Chebyshev basis vectors are computed recursively::

        Z^{.,1} = X
        Z^{.,2} = L_hat @ X
        Z^{.,k} = 2 * L_hat @ Z^{.,k-1} - Z^{.,k-2}

    and the rescaled Laplacian is::

        L      = I - D^{-1/2} A D^{-1/2}   (normalised graph Laplacian)
        L_hat  = L - I = -D^{-1/2} A D^{-1/2}

    The matrix–vector product ``L_hat @ V`` is implemented via scatter
    operations over the precomputed edge weight list:

        edge_weights[e] = -1 / sqrt(d_{src_e} * d_{dst_e})
    """
    hidden_dim: int
    poly_order: int    # K
    activation: str = 'relu'

    @nn.compact
    def __call__(
        self,
        X: jnp.ndarray,              # (n_nodes, in_dim)
        edge_src: jnp.ndarray,       # (n_edges,)  int
        edge_dst: jnp.ndarray,       # (n_edges,)  int
        edge_weights: jnp.ndarray,   # (n_edges,)  float  entries of L_hat
        n_nodes: int,
    ) -> jnp.ndarray:                # (n_nodes, hidden_dim)

        act = get_activation(self.activation)

        def lhat_matvec(V: jnp.ndarray) -> jnp.ndarray:
            """Sparse matrix–vector product: L_hat @ V.

            L_hat is symmetric, stored as undirected edges with
            ``edge_weights[e] = -1/sqrt(d_src * d_dst)``.
            Both directions are accumulated so each edge contributes
            to both endpoint rows.
            """
            # (n_edges, in_dim) * (n_edges, 1)
            w = edge_weights[:, None]               # (n_edges, 1)
            out = jnp.zeros_like(V)
            out = out.at[edge_dst].add(w * V[edge_src])
            out = out.at[edge_src].add(w * V[edge_dst])
            return out

        # --- Build Chebyshev basis Z^{1} … Z^{K} ---------------------------
        Z = [X]                                          # Z^{.,1}
        if self.poly_order >= 2:
            Z.append(lhat_matvec(X))                     # Z^{.,2}
        for _ in range(3, self.poly_order + 1):
            Zk = 2.0 * lhat_matvec(Z[-1]) - Z[-2]       # Z^{.,k}
            Z.append(Zk)

        # --- Weighted sum: sum_k Z^k @ Theta^k ------------------------------
        out = jnp.zeros((n_nodes, self.hidden_dim), dtype=X.dtype)
        for k, Zk in enumerate(Z):
            out = out + nn.Dense(
                self.hidden_dim,
                use_bias=False,
                name=f'theta_{k}',
            )(Zk)

        # Bias (shared across all k, per the paper's b^{l-1})
        b = self.param('bias', nn.initializers.zeros_init(), (self.hidden_dim,))
        out = out + b

        return act(out)


class _GNNModule(nn.Module):
    """
    Full GNN: encoder → ChebConv layers → decoder.

    Attributes
    ----------
    spatial_dims   : number of spatial dimensions of mesh nodes
    hidden_dim     : width of every hidden layer / output of each ChebConv
    poly_order     : Chebyshev polynomial order K (default 10, as in paper)
    message_steps  : number of stacked ChebConv layers
    n_outputs      : number of output channels per node
    activation     : activation inside ChebConv and input encoder (paper uses ReLU)
    """
    spatial_dims: int
    hidden_dim: int
    poly_order: int
    message_steps: int
    n_outputs: int
    activation: str = 'relu'

    @nn.compact
    def __call__(
        self,
        node_coords: jnp.ndarray,    # (n_nodes, spatial_dims) – normalised
        edge_src: jnp.ndarray,       # (n_edges,)  int
        edge_dst: jnp.ndarray,       # (n_edges,)  int
        edge_weights: jnp.ndarray,   # (n_edges,)  float  L_hat entries
        n_nodes: int,
    ) -> jnp.ndarray:                # (n_nodes, n_outputs)

        enc_act = get_activation(self.activation)

        # --- Node encoder: position → hidden space --------------------------
        h = nn.Dense(self.hidden_dim, name='encoder')(node_coords)
        h = enc_act(h)

        # --- Stacked Chebyshev GCN layers ------------------------------------
        for step in range(self.message_steps):
            h = _ChebConvLayer(
                hidden_dim  = self.hidden_dim,
                poly_order  = self.poly_order,
                activation  = self.activation,
                name        = f'cheb_layer_{step}',
            )(h, edge_src, edge_dst, edge_weights, n_nodes)

        # --- Node decoder ---------------------------------------------------
        coeffs = nn.Dense(self.n_outputs, name='decoder')(h)   # (n_nodes, n_outputs)
        return coeffs


class _GNNEncoderModule(nn.Module):
    """
    GNN encoder without output projection: encoder Dense → ChebConv layers.

    Returns node embeddings of shape ``(n_nodes, hidden_dim)`` rather than
    the final ``(n_nodes, n_outputs)`` produced by :class:`_GNNModule`.
    Used by :class:`GNNFeatures` and the refactored :class:`GNNMeshNetwork`.
    """
    encoder_in_dim: int   # actual encoder input dimension (may include t, u^n)
    hidden_dim: int
    poly_order: int
    message_steps: int
    activation: str = 'relu'

    @nn.compact
    def __call__(
        self,
        node_feats: jnp.ndarray,     # (n_nodes, encoder_in_dim)
        edge_src: jnp.ndarray,       # (n_edges,)
        edge_dst: jnp.ndarray,       # (n_edges,)
        edge_weights: jnp.ndarray,   # (n_edges,)
        n_nodes: int,
    ) -> jnp.ndarray:                # (n_nodes, hidden_dim)
        enc_act = get_activation(self.activation)
        h = nn.Dense(self.hidden_dim, name='encoder')(node_feats)
        h = enc_act(h)
        for step in range(self.message_steps):
            h = _ChebConvLayer(
                hidden_dim  = self.hidden_dim,
                poly_order  = self.poly_order,
                activation  = self.activation,
                name        = f'cheb_layer_{step}',
            )(h, edge_src, edge_dst, edge_weights, n_nodes)
        return h   # (n_nodes, hidden_dim)


class _MLPDecoder(nn.Module):
    """
    Per-point MLP decoder: ``(batch, in_dim) → (batch, n_outputs)``.

    When ``hidden_dims`` is empty this is a single linear projection.
    """
    hidden_dims: Sequence[int]
    n_outputs: int
    activation: str = 'tanh'
    normalize_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = get_activation(self.activation)
        if self.normalize_input:
            x = nn.LayerNorm()(x)
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = act(x)
        return nn.Dense(self.n_outputs)(x)


# ---------------------------------------------------------------------------
# Interpolation utilities (JAX-jittable, differentiable w.r.t. query_x)
# ---------------------------------------------------------------------------

def _barycentric_coords_all_triangles(
    nodes: jnp.ndarray,    # (n_nodes, 2)
    faces: jnp.ndarray,    # (n_faces, 3)  int
    query_x: jnp.ndarray,  # (n_query, 2)
) -> tuple:
    """
    Compute barycentric coordinates of every query point
    w.r.t. every triangle.

    Returns
    -------
    u, v, w : each of shape (n_query, n_faces)
        Barycentric coords (v1, v2, v0 weight respectively).
    """
    v0 = nodes[faces[:, 0]]   # (n_faces, 2)
    v1 = nodes[faces[:, 1]]
    v2 = nodes[faces[:, 2]]

    # Broadcast: (n_query, n_faces, 2)
    q  = query_x[:, None, :]   # (n_query, 1, 2)
    A  = v0[None, :, :]        # (1, n_faces, 2)
    B  = v1[None, :, :]
    C  = v2[None, :, :]

    # Vectors from A
    T0 = B - A   # (1, n_faces, 2)  edge AB
    T1 = C - A   # (1, n_faces, 2)  edge AC
    dX = q - A   # (n_query, n_faces, 2)

    # 2x2 linear system: [T0, T1] [u; v] = dX
    det = T0[..., 0] * T1[..., 1] - T0[..., 1] * T1[..., 0]  # (1, n_faces)
    safe_det = jnp.where(jnp.abs(det) < 1e-15, 1e-15, det)

    u = (dX[..., 0] * T1[..., 1] - dX[..., 1] * T1[..., 0]) / safe_det
    v = (T0[..., 0] * dX[..., 1] - T0[..., 1] * dX[..., 0]) / safe_det
    w = 1.0 - u - v

    return u, v, w   # (n_query, n_faces) each


def _interpolate_mesh(
    node_coeffs: jnp.ndarray,   # (n_nodes, n_outputs)
    nodes: jnp.ndarray,          # (n_nodes, 2)
    faces: jnp.ndarray,          # (n_faces, 3)
    query_x: jnp.ndarray,        # (n_query, 2)  [spatial only]
) -> jnp.ndarray:                # (n_query, n_outputs)
    """
    Barycentric interpolation of nodal values at arbitrary query points.

    The containing triangle is found by argmax of the minimum barycentric
    coordinate (all-positive ↔ inside).  Triangle selection is discrete
    (no gradient), but the barycentric weights *are* differentiable w.r.t.
    ``query_x``, so spatial derivatives of the output are well-defined.
    """
    u, v, w = _barycentric_coords_all_triangles(nodes, faces, query_x)
    # (n_query, n_faces)

    # "Insideness" score: higher when all coords are positive
    min_coord = jnp.minimum(jnp.minimum(u, v), w)    # (n_query, n_faces)

    # Containing triangle: argmax of min_coord  (discrete, no grad needed)
    tri_idx = jnp.argmax(min_coord, axis=-1)           # (n_query,)

    n_query = query_x.shape[0]
    row = jnp.arange(n_query)

    # Barycentric coords for the selected triangle (differentiable w.r.t. query_x)
    u_sel = u[row, tri_idx]   # (n_query,)
    v_sel = v[row, tri_idx]
    w_sel = w[row, tri_idx]

    # Node indices for selected triangles
    n0 = faces[tri_idx, 0]   # (n_query,)
    n1 = faces[tri_idx, 1]
    n2 = faces[tri_idx, 2]

    # Interpolate: w*c(v0) + u*c(v1) + v*c(v2)
    c0 = node_coeffs[n0]     # (n_query, n_outputs)
    c1 = node_coeffs[n1]
    c2 = node_coeffs[n2]

    return w_sel[:, None] * c0 + u_sel[:, None] * c1 + v_sel[:, None] * c2


# ---------------------------------------------------------------------------
# Shared mesh-array builder (used by GNNFeatures and GNNMeshNetwork)
# ---------------------------------------------------------------------------

def _build_mesh_arrays(verts: np.ndarray, faces: np.ndarray):
    """Precompute edge lists, L_hat weights, and normalised node coords."""
    n_nodes = len(verts)
    edges_set: dict = {}
    edges_list = []
    for face in faces:
        for j in range(3):
            v0, v1 = int(face[j]), int(face[(j + 1) % 3])
            key = (min(v0, v1), max(v0, v1))
            if key not in edges_set:
                edges_set[key] = len(edges_list)
                edges_list.append((v0, v1))

    src = np.array([e[0] for e in edges_list], dtype=np.int32)
    dst = np.array([e[1] for e in edges_list], dtype=np.int32)
    edge_src = jnp.array(src)
    edge_dst = jnp.array(dst)

    degree = np.zeros(n_nodes, dtype=np.float32)
    for s, d in zip(src, dst):
        degree[s] += 1
        degree[d] += 1
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    edge_w = -d_inv_sqrt[src] * d_inv_sqrt[dst]
    edge_weights = jnp.array(edge_w, dtype=jnp.float32)

    node_min = verts.min(axis=0)
    node_max = verts.max(axis=0)
    nodes_norm = (2.0 * (verts - node_min) / (node_max - node_min + 1e-8) - 1.0)
    nodes_norm_jnp = jnp.array(nodes_norm, dtype=jnp.float32)

    return edge_src, edge_dst, edge_weights, nodes_norm_jnp


def _build_bc_arrays(domain, n_nodes: int, n_outputs: int, verts: np.ndarray):
    """Build hard-BC mask and value arrays from domain Dirichlet BCs."""
    bc_mask   = np.zeros((n_nodes, n_outputs), dtype=np.float32)
    bc_values = np.zeros((n_nodes, n_outputs), dtype=np.float32)
    from pinns.boundary import MeshNodeBC
    for bc in getattr(domain, 'boundary_conditions', []):
        if not (isinstance(bc, MeshNodeBC) and bc.bc_type == 'dirichlet'):
            continue
        t_mode = getattr(bc, 't_mode', None)
        if t_mode not in (None, 'all'):
            continue
        comp = bc.component
        if comp >= n_outputs:
            continue
        node_idx = (np.unique(bc.node_indices) if bc.node_indices is not None
                    else np.unique(bc.edges))
        node_pos = verts[node_idx]
        vals = bc.get_value(node_pos)
        bc_mask[node_idx, comp]   = 1.0
        bc_values[node_idx, comp] = vals
    return jnp.array(bc_mask, dtype=jnp.float32), jnp.array(bc_values, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# GNNFeatures — trainable GNN-based feature encoder (analogous to LaplacianFeatures)
# ---------------------------------------------------------------------------

class GNNFeatures:
    """
    Trainable GNN-based spatial feature encoder for mesh PINNs.

    Analogous to :class:`~pinns.LaplacianFeatures` but uses a **trainable**
    ChebConv GNN instead of fixed Laplace eigenvectors.  The GNN runs on the
    fixed spatial mesh and produces a ``hidden_dim``-dimensional embedding at
    every mesh node; embeddings are then barycentric-interpolated at the query
    points.

    **Time handling** — if *domain* carries a ``t_interval``:

    * The GNN receives only spatial node features ``[x_norm, y_norm]``.
    * After interpolation at query points, the raw time coordinate ``t`` is
      appended as an extra feature column.
    * ``output_dim = hidden_dim + 1``  (so FNN input width = ``hidden_dim + 1``).

    Parameters
    ----------
    domain : DomainMesh
        2-D triangular mesh domain.
    hidden_dim : int
        Width of GNN hidden layers and output embedding dimension.
    poly_order : int
        Chebyshev polynomial order (higher = more spectral range).
    message_steps : int
        Number of stacked ChebConv layers.
    activation : str
        Activation used in both the input encoder and ChebConv layers.

    Usage
    -----
    ::

        enc = GNNFeatures(domain, hidden_dim=64)
        enc_params = enc.init(jax.random.PRNGKey(0))
        features = enc(enc_params, x_query)          # (n_query, output_dim)

    To compose with a custom MLP decoder::

        enc_p = enc.init(rng1)
        mlp   = flax.linen.Dense(1)
        mlp_p = mlp.init(rng2, jnp.zeros((1, enc.output_dim)))
        y     = mlp.apply(mlp_p, enc(enc_p, x))
    """

    def __init__(
        self,
        domain,
        hidden_dim: int = 64,
        poly_order: int = 10,
        message_steps: int = 4,
        activation: str = 'relu',
    ):
        t_interval = getattr(domain, 't_interval', None)
        self.has_time     = t_interval is not None
        self.spatial_dims = domain._spatial_dims
        self.hidden_dim   = hidden_dim
        self.output_dim   = hidden_dim + (1 if self.has_time else 0)
        if self.has_time:
            self.t_min = float(t_interval[0])
            self.t_max = float(t_interval[1])

        verts = domain._vertices[:, :self.spatial_dims].astype(np.float32)
        faces = domain._faces.astype(np.int32)
        self._nodes_np = verts
        self._faces_np = faces
        self.n_nodes = len(verts)

        (self._edge_src, self._edge_dst,
         self._edge_weights, self._nodes_norm) = _build_mesh_arrays(verts, faces)

        # Encoder input: [x_norm, y_norm]  (purely spatial — t handled externally)
        self._module = _GNNEncoderModule(
            encoder_in_dim = self.spatial_dims,
            hidden_dim     = hidden_dim,
            poly_order     = poly_order,
            message_steps  = message_steps,
            activation     = activation,
        )

    def init(self, rng: 'jax.random.PRNGKey') -> Dict:
        """Initialise GNN encoder parameters."""
        return self._module.init(
            rng,
            self._nodes_norm,          # (n_nodes, spatial_dims)
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
            Query coordinates.  If the domain has a time interval the last
            column must be ``t``.
        params_dict : ignored (API compatibility).

        Returns
        -------
        jnp.ndarray, shape ``(n_query, output_dim)``
        """
        # Run GNN on purely-spatial node features
        node_embeds = self._module.apply(
            enc_params,
            self._nodes_norm,
            self._edge_src,
            self._edge_dst,
            self._edge_weights,
            self.n_nodes,
        )   # (n_nodes, hidden_dim)

        # Interpolate embeddings at query spatial coords
        nodes_jnp = jnp.array(self._nodes_np)
        faces_jnp = jnp.array(self._faces_np)
        x_spatial = x[:, :self.spatial_dims]
        query_embeds = _interpolate_mesh(node_embeds, nodes_jnp, faces_jnp, x_spatial)
        # (n_query, hidden_dim)

        if self.has_time:
            t_col = x[:, self.spatial_dims : self.spatial_dims + 1]   # (n_query, 1)
            return jnp.concatenate([query_embeds, t_col], axis=-1)    # (n_query, hidden_dim+1)
        return query_embeds

    def __repr__(self) -> str:
        mode = f"hidden_dim={self.hidden_dim}" + (" + t" if self.has_time else "")
        return f"GNNFeatures(output_dim={self.output_dim}, {mode})"


# ---------------------------------------------------------------------------
# Public GNNMeshNetwork class
# ---------------------------------------------------------------------------

class GNNMeshNetwork:
    """
    Graph Neural Network on a triangular mesh with two operating modes,
    mirroring the :class:`~pinns.AlphaPINN` API split.

    **Continuous-time mode** (``use_state=False``, default for :class:`~pinns.DomainMeshContinuous`)
        Use with :class:`~pinns.DomainMeshContinuous`.

        Pipeline::

            node_feats [x_norm, y_norm]          (n_nodes, 2)
              → GNN encoder (ChebConv × message_steps)  (n_nodes, hidden_dim)
              → barycentric interp at query pts          (n_query, hidden_dim)
              → [embeds, t]                              (n_query, hidden_dim+1)
              → MLP decoder (decoder_dims)               (n_query, n_outputs)

        Time derivative ``∂u/∂t`` is obtained by autodiff through the MLP
        w.r.t. the appended ``t`` feature — no broadcast tricks needed.

    **State-integrator / rollout mode** (``use_state=True``, auto for :class:`~pinns.DomainMeshDiscrete`)
        Use with :class:`~pinns.DomainMeshDiscrete`.

        The GNN operates over **all mesh nodes** at each rollout step.
        ``u^n`` at mesh nodes is read from ``params_dict["fixed"]["u_prev_nodes"]``
        and appended as an extra node feature::

            node_feats [x_norm, y_norm, u^n]     (n_nodes, 2+n_outputs)
              → GNN encoder (ChebConv × message_steps)  (n_nodes, hidden_dim)
              → MLP decoder (decoder_dims) per node      (n_nodes, n_outputs) ← u^{n+1}
              → hard Dirichlet BC enforcement
              → barycentric interp at query pts          (n_query, n_outputs)

    Parameters
    ----------
    domain : DomainMesh
        2-D triangular mesh domain.
    hidden_dim : int
        GNN hidden / embedding dimension (ChebConv output width).
    poly_order : int
        Chebyshev polynomial order K.
    message_steps : int
        Number of stacked ChebConv layers.
    n_outputs : int
        Number of PDE unknowns per node.
    decoder_dims : sequence of int
        Hidden-layer widths for the MLP decoder that maps GNN embeddings
        to outputs.  ``()`` (default) = single linear projection — identical
        behaviour to the original ``GNNMeshNetwork``.
    activation : str
        Activation used in both the input encoder and ChebConv layers (default ``'relu'``).
    decoder_act : str
        Activation inside the MLP decoder (default ``'tanh'``).
    use_state : bool or None
        ``None`` (default): auto-detect — ``True`` when domain is a
        :class:`~pinns.DomainMeshDiscrete`, ``False`` otherwise.
    input_transform : callable, optional
        Applied to query coords before evaluation.
    output_transform : callable, optional
        Hard-constraint transform applied after the final output.
    normalize_input : bool
        Normalise query coordinates to ``[-1, 1]`` (for output un-normalisation
        path only; node coordinates are always normalised internally).
    """

    def __init__(
        self,
        domain,
        hidden_dim: int = 64,
        poly_order: int = 10,
        message_steps: int = 4,
        n_outputs: int = 1,
        decoder_dims: Sequence[int] = (),
        activation: str = 'relu',
        decoder_act: str = 'tanh',
        use_state: Optional[bool] = None,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        normalize_input: bool = True,
        residual: bool = False,
    ):
        if domain._spatial_dims != 2:
            raise NotImplementedError(
                "GNNMeshNetwork currently supports 2D spatial meshes only "
                f"(got spatial_dims={domain._spatial_dims})."
            )

        # Auto-detect mode from domain type
        if use_state is None:
            use_state = getattr(domain, '_time_mode', None) == 'discrete'

        # Time interval — only relevant in continuous mode
        t_interval = getattr(domain, 't_interval', None)
        self.has_time     = (t_interval is not None) and (not use_state)
        self.use_state    = use_state
        self.spatial_dims = domain._spatial_dims
        self.hidden_dim   = hidden_dim
        self.n_outputs    = n_outputs
        self.poly_order   = poly_order
        self.message_steps = message_steps
        self.decoder_dims  = list(decoder_dims)
        self.input_transform    = input_transform
        self.output_transform   = output_transform
        self.normalize_input    = normalize_input
        self.residual           = residual

        if self.has_time:
            self.t_min = float(t_interval[0])
            self.t_max = float(t_interval[1])
        else:
            self.t_min = None
            self.t_max = None

        # ── Fixed mesh arrays ────────────────────────────────────────────────
        verts = domain._vertices[:, :self.spatial_dims].astype(np.float32)
        faces = domain._faces.astype(np.int32)
        self._nodes_np = verts
        self._faces_np = faces
        self.n_nodes = len(verts)
        self.n_faces = faces.shape[0]

        (self._edge_src, self._edge_dst,
         self._edge_weights, self._nodes_norm) = _build_mesh_arrays(verts, faces)

        # Input/output normalisation bounds (set by trainer)
        self.input_min  = None
        self.input_max  = None
        self.output_min = None
        self.output_max = None

        # ── GNN encoder Flax module ──────────────────────────────────────────
        # Continuous: encoder receives [x_norm, y_norm]  → hidden_dim embeddings
        # State:      encoder receives [x_norm, y_norm, u^n_i]
        encoder_in_dim = self.spatial_dims + (n_outputs if use_state else 0)
        self._encoder_module = _GNNEncoderModule(
            encoder_in_dim = encoder_in_dim,
            hidden_dim     = hidden_dim,
            poly_order     = poly_order,
            message_steps  = message_steps,
            activation     = activation,
        )

        # ── MLP decoder Flax module ──────────────────────────────────────────
        # Continuous: input = (hidden_dim + 1)  [+1 for t appended after interp]
        # State:      input = hidden_dim          [decoder applied per-node]
        decoder_in_dim = hidden_dim + (1 if self.has_time else 0)
        self._decoder_module = _MLPDecoder(
            hidden_dims     = list(decoder_dims),
            n_outputs       = n_outputs,
            activation      = decoder_act,
            normalize_input = False,
        )
        self._decoder_in_dim = decoder_in_dim

        # ── Hard Dirichlet BC arrays ─────────────────────────────────────────
        # Always built in state mode so BC nodes are clamped before rollout state is stored.
        if use_state:
            self._bc_mask_jnp, self._bc_values_jnp = _build_bc_arrays(
                domain, self.n_nodes, n_outputs, verts
            )
        else:
            self._bc_mask_jnp   = jnp.zeros((self.n_nodes, n_outputs), dtype=jnp.float32)
            self._bc_values_jnp = jnp.zeros((self.n_nodes, n_outputs), dtype=jnp.float32)

    # ── Trainer-compatible API ───────────────────────────────────────────── #

    def set_input_range(self, xmin: np.ndarray, xmax: np.ndarray):
        self.input_min = jnp.array(xmin, dtype=jnp.float32)
        self.input_max = jnp.array(xmax, dtype=jnp.float32)

    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        self.output_min = jnp.array(ymin, dtype=jnp.float32)
        self.output_max = jnp.array(ymax, dtype=jnp.float32)

    def init(self, rng: jax.random.PRNGKey, dummy_input=None) -> Dict:
        """
        Initialise network parameters.

        Returns
        -------
        dict
            ``{"encoder": encoder_params, "decoder": decoder_params}``
        """
        rng_enc, rng_dec = jax.random.split(rng)

        # Encoder init — use purely-spatial node features (u^n = 0 for state mode)
        init_feats = self._nodes_norm   # (n_nodes, spatial_dims)
        if self.use_state:
            dummy_state = jnp.zeros((self.n_nodes, self.n_outputs), dtype=jnp.float32)
            init_feats = jnp.concatenate([init_feats, dummy_state], axis=-1)

        enc_params = self._encoder_module.init(
            rng_enc, init_feats,
            self._edge_src, self._edge_dst, self._edge_weights, self.n_nodes,
        )

        # Decoder init — dummy input has the correct decoder input width
        dummy_dec = jnp.zeros((1, self._decoder_in_dim), dtype=jnp.float32)
        dec_params = self._decoder_module.init(rng_dec, dummy_dec)

        return {"encoder": enc_params, "decoder": dec_params}

    def apply(
        self,
        params: Dict,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """
        Evaluate the network at query points *x*.

        **Continuous mode** — forward pass at query points:

        1. Run GNN encoder on spatial node features → ``(n_nodes, hidden_dim)``
        2. Barycentric interpolation at *x* → ``(n_query, hidden_dim)``
        3. Append ``t`` column → ``(n_query, hidden_dim+1)``
        4. MLP decoder → ``(n_query, n_outputs)``
        5. Optional ``output_transform``

        **State-integrator mode** — one-step forward:

        1. Append ``u^n`` (from ``params_dict["fixed"]["u_prev_nodes"]``) to node feats
        2. Run GNN encoder → ``(n_nodes, hidden_dim)``
        3. MLP decoder per node → ``(n_nodes, n_outputs)``  ← ``u^{n+1}``
        4. Hard Dirichlet BC enforcement
        5. Barycentric interpolation at *x* → ``(n_query, n_outputs)``

        Parameters
        ----------
        params : dict  ``{"encoder": ..., "decoder": ...}``
        x : jnp.ndarray  ``(n_query, spatial_dims[+1])``
        params_dict : dict, optional

        Returns
        -------
        (n_query, n_outputs)
        """
        x_original = x

        if self.input_transform is not None:
            x = self.input_transform(x, params_dict)

        enc_params = params["encoder"]
        dec_params = params["decoder"]

        nodes_jnp = jnp.array(self._nodes_np)
        faces_jnp = jnp.array(self._faces_np)
        x_spatial = x_original[:, :self.spatial_dims]

        if not self.use_state:
            # ── Continuous mode ──────────────────────────────────────────────
            # GNN encoder on purely-spatial node features [x_norm, y_norm]
            node_embeds = self._encoder_module.apply(
                enc_params,
                self._nodes_norm,
                self._edge_src, self._edge_dst, self._edge_weights, self.n_nodes,
            )   # (n_nodes, hidden_dim)

            # Interpolate embeddings at query spatial coords
            query_embeds = _interpolate_mesh(node_embeds, nodes_jnp, faces_jnp, x_spatial)
            # (n_query, hidden_dim)

            # Append raw t so the decoder can differentiate w.r.t. time
            if self.has_time:
                t_col = x_original[:, self.spatial_dims : self.spatial_dims + 1]  # (n_query, 1)
                query_embeds = jnp.concatenate([query_embeds, t_col], axis=-1)

            # MLP decoder at query points
            y = self._decoder_module.apply(dec_params, query_embeds)  # (n_query, n_outputs)

            # Output un-normalisation (if set by trainer)
            if self.output_min is not None:
                y = (y + 1.0) / 2.0 * (self.output_max - self.output_min) + self.output_min

            if self.output_transform is not None:
                y = self.output_transform(x_original, y, params_dict)

            return y

        else:
            # ── State-integrator mode ────────────────────────────────────────
            # Read u^n nodal values from params_dict
            _fixed = (params_dict or {}).get("fixed") or {}
            u_prev_raw = _fixed.get("u_prev_nodes", None)
            if u_prev_raw is None:
                u_prev_nodes = jnp.zeros((self.n_nodes, self.n_outputs), dtype=jnp.float32)
            else:
                u_prev_nodes = jnp.asarray(u_prev_raw, dtype=jnp.float32)
            if u_prev_nodes.ndim == 1:
                u_prev_nodes = u_prev_nodes[:, None]   # (n_nodes, n_outputs)

            # GNN encoder on [x_norm, y_norm, u^n]
            node_feats = jnp.concatenate([self._nodes_norm, u_prev_nodes], axis=-1)
            node_embeds = self._encoder_module.apply(
                enc_params,
                node_feats,
                self._edge_src, self._edge_dst, self._edge_weights, self.n_nodes,
            )   # (n_nodes, hidden_dim)

            # Per-node MLP decoder → Δu^n (residual mode) or u^{n+1} directly
            node_raw = self._decoder_module.apply(dec_params, node_embeds)
            # (n_nodes, n_outputs)
            if self.residual:
                node_out = u_prev_nodes + node_raw   # u^{n+1} = u^n + Δu^n
            else:
                node_out = node_raw

            # Hard Dirichlet BC enforcement (static condensation) — always applied in state mode
            node_out = (node_out * (1.0 - self._bc_mask_jnp)
                        + self._bc_values_jnp * self._bc_mask_jnp)

            # Barycentric interpolation at query points
            y = _interpolate_mesh(node_out, nodes_jnp, faces_jnp, x_spatial)
            # (n_query, n_outputs)

            if self.output_transform is not None:
                y = self.output_transform(x_original, y, params_dict)

            return y

    def forward(self, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        return self.apply(self.params, x, params_dict)

    def predict(self, x_np: np.ndarray, params_dict=None) -> np.ndarray:
        return np.array(self.apply(self.params, jnp.array(x_np), params_dict))

    # ── Introspection ────────────────────────────────────────────────────── #

    @property
    def mesh_nodes(self) -> np.ndarray:
        return self._nodes_np.copy()

    @property
    def mesh_faces(self) -> np.ndarray:
        return self._faces_np.copy()

    def to(self, device: str = None, dtype=None, seed: int = 0) -> 'GNNMeshNetwork':
        """PyTorch-compatible device/dtype migration (no-op for JAX)."""
        if device is None:
            device = jax.devices()[0].platform
        if dtype is None:
            dtype = jnp.float32
        self.device = device
        self.dtype = dtype
        if not hasattr(self, 'params') or self.params is None:
            self.params = self.init(jax.random.PRNGKey(seed))
        return self

    def node_embeddings(self, params: Dict, u_prev_nodes=None) -> np.ndarray:
        """
        Return GNN node embeddings ``(n_nodes, hidden_dim)``.

        In state mode pass the current nodal state as *u_prev_nodes*.
        """
        enc_params = params["encoder"]
        node_feats = self._nodes_norm
        if self.use_state and u_prev_nodes is not None:
            u = jnp.asarray(u_prev_nodes, dtype=jnp.float32)
            if u.ndim == 1:
                u = u[:, None]
            node_feats = jnp.concatenate([node_feats, u], axis=-1)
        embeds = self._encoder_module.apply(
            enc_params,
            node_feats,
            self._edge_src, self._edge_dst, self._edge_weights, self.n_nodes,
        )
        return np.array(embeds)

    def predict_rollout(
        self,
        n_steps: int = None,
        u0: np.ndarray = None,
        dt: float = None,
    ) -> np.ndarray:
        """Run the state-integrator forward from *u0* for *n_steps* steps.

        Returns
        -------
        u_all : (n_steps+1, n_nodes) float32 numpy array
            Row 0 is the initial condition; rows 1‒n_steps are the predicted
            states after each time step.
        """
        if not self.use_state:
            raise RuntimeError(
                "predict_rollout is only available in state-integrator mode."
            )
        if u0 is None:
            # Seed from BC values (hot=1, cold=0) and IC=0 elsewhere
            bc_mask_1d   = np.array(self._bc_mask_jnp[:, 0])
            bc_values_1d = np.array(self._bc_values_jnp[:, 0])
            u0 = bc_values_1d * bc_mask_1d  # IC=0, BCs at their prescribed values
        if n_steps is None:
            n_steps = 1
        _dt    = jnp.float32(dt if dt is not None else 0.0)
        _mesh  = jnp.array(self._nodes_np)
        _params = self.params

        def _step(u_nodes):
            pdict = {"fixed": {
                "u_prev_nodes": u_nodes,
                "u_prev":       u_nodes,
                "dt":           _dt,
                "kappa":        jnp.float32(0.0),
            }}
            return self.apply(_params, _mesh, pdict)[:, 0]

        # Apply BC mask to ensure initial state honours hard constraints
        _mask_1d   = self._bc_mask_jnp[:, 0]
        _values_1d = self._bc_values_jnp[:, 0]
        u_cur = jnp.array(u0, dtype=jnp.float32)
        u_cur = u_cur * (1.0 - _mask_1d) + _values_1d * _mask_1d

        u_all = [np.array(u_cur)]
        for _ in range(n_steps):
            u_cur = _step(u_cur)
            u_all.append(np.array(u_cur))
        return np.stack(u_all, axis=0)  # (n_steps+1, n_nodes)

    def __repr__(self) -> str:
        mode = "state-integrator" if self.use_state else "continuous"
        dec = f"+MLP{self.decoder_dims}" if self.decoder_dims else "+Linear"
        res = ", residual=True" if self.residual else ""
        return (
            f"GNNMeshNetwork({mode}, n_nodes={self.n_nodes}, "
            f"hidden_dim={self.hidden_dim}, poly_order={self.poly_order}, "
            f"message_steps={self.message_steps}{dec}, "
            f"n_outputs={self.n_outputs}{res})"
        )
