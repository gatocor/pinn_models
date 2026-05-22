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

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Dict, Optional, Sequence, Callable, Any

from .fnn import get_activation


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
    nodes: jnp.ndarray,    # (n_nodes, D)  D=2 or D=3
    faces: jnp.ndarray,    # (n_faces, 3)  int
    query_x: jnp.ndarray,  # (n_query, D)
) -> tuple:
    """
    Compute barycentric coordinates of every query point w.r.t. every triangle.

    Works for both 2-D flat meshes (D=2) and 3-D surface meshes (D=3).
    For 3-D meshes the query point is first projected onto the triangle plane
    using the Gram-matrix formulation, which is equivalent to a least-squares
    solution of the overdetermined system  [T0 | T1] [u; v] = (q - A).

    Returns
    -------
    u, v, w : each of shape (n_query, n_faces)
        Barycentric coords (v1, v2, v0 weight respectively).
    """
    v0 = nodes[faces[:, 0]]   # (n_faces, D)
    v1 = nodes[faces[:, 1]]
    v2 = nodes[faces[:, 2]]

    # Broadcast: shapes become (n_query, n_faces, D)
    q  = query_x[:, None, :]   # (n_query, 1, D)
    A  = v0[None, :, :]        # (1, n_faces, D)
    B  = v1[None, :, :]
    C  = v2[None, :, :]

    T0 = B - A   # (1, n_faces, D)  edge AB
    T1 = C - A   # (1, n_faces, D)  edge AC
    dX = q - A   # (n_query, n_faces, D)

    # Gram matrix entries: Gij = Ti · Tj  (dot along last axis)
    G00 = jnp.sum(T0 * T0, axis=-1)   # (1, n_faces)
    G01 = jnp.sum(T0 * T1, axis=-1)
    G11 = jnp.sum(T1 * T1, axis=-1)

    # RHS: bi = Ti · dX
    b0  = jnp.sum(T0 * dX, axis=-1)   # (n_query, n_faces)
    b1  = jnp.sum(T1 * dX, axis=-1)

    # Cramer's rule on the 2×2 Gram system
    det = G00 * G11 - G01 * G01        # (1, n_faces)
    safe_det = jnp.where(jnp.abs(det) < 1e-15, 1e-15, det)

    u = (G11 * b0 - G01 * b1) / safe_det   # (n_query, n_faces)
    v = (G00 * b1 - G01 * b0) / safe_det
    w = 1.0 - u - v

    return u, v, w   # (n_query, n_faces) each


def _interpolate_mesh(
    node_coeffs: jnp.ndarray,   # (n_nodes, n_outputs)
    nodes: jnp.ndarray,          # (n_nodes, D)
    faces: jnp.ndarray,          # (n_faces, 3)
    query_x: jnp.ndarray,        # (n_query, D)
    face_tree=None,              # unused, kept for API compat
    node_to_face=None,           # legacy (n_nodes,) int32 — ignored if node_to_faces given
    node_to_faces=None,          # (n_nodes, K) int32 — all adjacent faces, -1 padded
) -> jnp.ndarray:                # (n_query, n_outputs)
    """
    Barycentric interpolation of nodal values at arbitrary query points.

    For 2-D meshes, all triangle candidates are tested and the best is
    selected by argmax of the minimum barycentric coordinate.

    For 3-D surface meshes with ``node_to_faces`` (n_nodes, K):
      1. Find the nearest mesh vertex by L2 distance (stop_gradient).
      2. Gather all K adjacent faces of that vertex.
      3. Compute differentiable barycentric weights for each candidate face.
      4. Select the face with the highest minimum barycentric coordinate
         (i.e., the face that best contains the query point).

    This correctly handles off-vertex query points where a single fixed
    face-per-vertex would give wrong extrapolated barycentric coordinates.
    """
    D = nodes.shape[1]

    if D == 3 and node_to_faces is not None:
        # Step 1: nearest vertex — discrete, stop_gradient so JVP sees zero tangent
        diff  = nodes[None, :, :] - query_x[:, None, :]   # (n_q, n_nodes, D)
        dist2 = jnp.sum(diff ** 2, axis=-1)               # (n_q, n_nodes)
        nearest = jax.lax.stop_gradient(jnp.argmin(dist2, axis=-1))  # (n_q,)

        # Step 2: all K adjacent faces of nearest vertex — shape (n_q, K)
        K = node_to_faces.shape[1]
        cand_faces = jax.lax.stop_gradient(node_to_faces[nearest])  # (n_q, K)
        # Clamp -1 padding to 0 (will be masked out by score below)
        cand_faces_safe = jnp.where(cand_faces >= 0, cand_faces, 0)

        # Step 3: compute barycentric weights for each candidate face
        n0 = faces[cand_faces_safe, 0]   # (n_q, K)
        n1 = faces[cand_faces_safe, 1]
        n2 = faces[cand_faces_safe, 2]
        v0 = nodes[n0]   # (n_q, K, D)
        v1 = nodes[n1]
        v2 = nodes[n2]

        T0 = v1 - v0
        T1 = v2 - v0
        dX = query_x[:, None, :] - v0   # (n_q, K, D)

        G00 = jnp.sum(T0 * T0, axis=-1)   # (n_q, K)
        G01 = jnp.sum(T0 * T1, axis=-1)
        G11 = jnp.sum(T1 * T1, axis=-1)
        b0  = jnp.sum(T0 * dX, axis=-1)
        b1  = jnp.sum(T1 * dX, axis=-1)

        det = G00 * G11 - G01 * G01
        safe_det = jnp.where(jnp.abs(det) < 1e-15, 1e-15, det)
        u_cand = (G11 * b0 - G01 * b1) / safe_det   # (n_q, K)
        v_cand = (G00 * b1 - G01 * b0) / safe_det
        w_cand = 1.0 - u_cand - v_cand

        # Step 4: pick the face with highest min barycentric coord
        # (i.e., the face best containing the query point).
        # Mask padded slots with -inf so they are never selected.
        min_coord = jnp.minimum(jnp.minimum(u_cand, v_cand), w_cand)  # (n_q, K)
        min_coord = jnp.where(cand_faces >= 0, min_coord, -jnp.inf)
        best_k = jax.lax.stop_gradient(jnp.argmax(min_coord, axis=-1))  # (n_q,)

        n_q = query_x.shape[0]
        row = jnp.arange(n_q)
        u_sel = u_cand[row, best_k]
        v_sel = v_cand[row, best_k]
        w_sel = w_cand[row, best_k]
        tri_global = cand_faces_safe[row, best_k]
        n0 = faces[tri_global, 0]
        n1 = faces[tri_global, 1]
        n2 = faces[tri_global, 2]

    elif D == 3 and node_to_face is not None:
        # Legacy path: single face per vertex (less accurate for off-vertex points)
        diff  = nodes[None, :, :] - query_x[:, None, :]   # (n_q, n_nodes, D)
        dist2 = jnp.sum(diff ** 2, axis=-1)               # (n_q, n_nodes)
        nearest = jax.lax.stop_gradient(jnp.argmin(dist2, axis=-1))  # (n_q,)

        # Step 2: face from precomputed map
        tri_global = node_to_face[nearest]   # (n_q,)

        # Step 3: differentiable barycentric weights for the selected face
        n0 = faces[tri_global, 0]
        n1 = faces[tri_global, 1]
        n2 = faces[tri_global, 2]
        v0 = nodes[n0]    # (n_q, D)
        v1 = nodes[n1]
        v2 = nodes[n2]

        T0 = v1 - v0
        T1 = v2 - v0
        dX = query_x - v0

        G00 = jnp.sum(T0 * T0, axis=-1)
        G01 = jnp.sum(T0 * T1, axis=-1)
        G11 = jnp.sum(T1 * T1, axis=-1)
        b0  = jnp.sum(T0 * dX, axis=-1)
        b1  = jnp.sum(T1 * dX, axis=-1)

        det      = G00 * G11 - G01 * G01
        safe_det = jnp.where(jnp.abs(det) < 1e-15, 1e-15, det)
        u_sel = (G11 * b0 - G01 * b1) / safe_det
        v_sel = (G00 * b1 - G01 * b0) / safe_det
        w_sel = 1.0 - u_sel - v_sel

    else:
        # 2-D or fallback: test all triangles
        u, v, w = _barycentric_coords_all_triangles(nodes, faces, query_x)
        min_coord  = jnp.minimum(jnp.minimum(u, v), w)
        tri_global = jnp.argmax(min_coord, axis=-1)

        n_query = query_x.shape[0]
        row    = jnp.arange(n_query)
        u_sel  = u[row, tri_global]
        v_sel  = v[row, tri_global]
        w_sel  = w[row, tri_global]
        n0 = faces[tri_global, 0]
        n1 = faces[tri_global, 1]
        n2 = faces[tri_global, 2]

    c0 = node_coeffs[n0]
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


def _query_context_to_nodes(
    ctx_query: jnp.ndarray,   # (n_query, n_ctx)
    x_spatial: jnp.ndarray,   # (n_query, 2)
    nodes: jnp.ndarray,        # (n_nodes, 2)
    faces: jnp.ndarray,        # (n_faces, 3) — kept for API consistency
    n_nodes: int,
) -> jnp.ndarray:
    """
    Scatter context values from query points to mesh nodes via nearest-neighbour.

    For each mesh node finds the closest query point (L2) and takes its context
    value.  This is the inverse of :func:`_interpolate_mesh`.

    Returns
    -------
    jnp.ndarray, shape ``(n_nodes, n_context)``
    """
    # Pairwise squared distances: (n_nodes, n_query)
    diff  = nodes[:, None, :] - x_spatial[None, :, :]   # (n_nodes, n_query, 2)
    dist2 = jnp.sum(diff ** 2, axis=-1)                  # (n_nodes, n_query)
    nearest = jnp.argmin(dist2, axis=-1)                 # (n_nodes,)
    return ctx_query[nearest]                            # (n_nodes, n_ctx)


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

    # ── ModelBase composable protocol ───────────────────────────────────── #

    def _configure(self, network, input_dim: int) -> int:
        """Called by ModelBase.add().  Builds mesh using network's domain and
        any accumulated coordinate transforms from preceding Normalize layers."""
        self._build_from_domain(
            network.domain,
            coord_transform=network._apply_coord_transforms
            if network._coord_transforms else None,
        )
        return self.output_dim

    def apply(
        self,
        x,
        params: Dict = None,
        params_dict=None,
    ):
        """ModelBase-protocol alias: apply(x, params) → forward pass."""
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
