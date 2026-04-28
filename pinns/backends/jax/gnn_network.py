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
    activation     : activation inside ChebConv (paper uses ReLU)
    encoder_act    : activation for the input encoder Dense layer
    """
    spatial_dims: int
    hidden_dim: int
    poly_order: int
    message_steps: int
    n_outputs: int
    activation: str = 'relu'
    encoder_act: str = 'tanh'

    @nn.compact
    def __call__(
        self,
        node_coords: jnp.ndarray,    # (n_nodes, spatial_dims) – normalised
        edge_src: jnp.ndarray,       # (n_edges,)  int
        edge_dst: jnp.ndarray,       # (n_edges,)  int
        edge_weights: jnp.ndarray,   # (n_edges,)  float  L_hat entries
        n_nodes: int,
    ) -> jnp.ndarray:                # (n_nodes, n_outputs)

        enc_act = get_activation(self.encoder_act)

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
# Public GNNMeshNetwork class
# ---------------------------------------------------------------------------

class GNNMeshNetwork:
    """
    Graph Neural Network on a triangular mesh, compatible with the
    ``FNN`` API used by the ``pinns`` trainer.

    The network performs message passing on the **fixed** spatial mesh
    to produce nodal coefficient vectors, then evaluates at arbitrary
    query points by barycentric (Lagrange-P1) interpolation.

    Parameters
    ----------
    domain : DomainMesh
        Mesh domain.  Must contain a 2D triangular mesh
        (``domain._spatial_dims == 2``).
    hidden_dim : int
        Width of every hidden layer (encoder, ChebConv layers, decoder).
    poly_order : int
        Chebyshev polynomial order K (default 10, as recommended in the
        paper).  Higher K captures longer-range spectral information.
    message_steps : int
        Number of stacked ChebConv layers (depth of the GCN).
    n_outputs : int
        Number of PDE unknowns / output channels (default 1).
    activation : str
        Activation used inside ChebConv layers (paper uses ``'relu'``).
    encoder_act : str
        Activation for the input encoder (default ``'tanh'``).
    normalize_input : bool
        Normalise query coordinates to ``[-1, 1]`` before interpolation
        (the node coordinates fed to the GNN are always normalised).
    input_transform : callable, optional
        Applied to query coordinates before normalisation.
    output_transform : callable, optional
        Hard-constraint transform applied after interpolation.
        Signature: ``output_transform(x_original, y, params_dict) -> y``
    hard_constraints : bool
        If ``True`` (default), Dirichlet boundary nodes are **hardwired** to
        their prescribed values by static condensation: the GNN output at
        those nodes is replaced by the known BC values *before* barycentric
        interpolation.  This is the approach of Eq. (13) in the paper —
        essential BCs are satisfied *exactly* by construction and their loss
        terms become identically zero.  Set to ``False`` to fall back to the
        soft (penalty / Lagrange) enforcement.

    Example
    -------
    ::

        import trimesh, pinns, jax

        mesh   = trimesh.load("mesh.obj")
        domain = pinns.DomainMesh(mesh)
        net    = GNNMeshNetwork(domain, hidden_dim=64, depth=3, message_steps=4)
        params = net.init(jax.random.PRNGKey(0))
        y      = net.apply(params, x_query)   # (n_query, 1)
    """

    def __init__(
        self,
        domain,                            # DomainMesh
        hidden_dim: int = 64,
        poly_order: int = 10,
        message_steps: int = 4,
        n_outputs: int = 1,
        activation: str = 'relu',
        encoder_act: str = 'tanh',
        normalize_input: bool = True,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        hard_constraints: bool = True,
    ):
        if domain._spatial_dims != 2:
            raise NotImplementedError(
                "GNNMeshNetwork currently supports 2D spatial meshes only "
                f"(got spatial_dims={domain._spatial_dims})."
            )

        self.hidden_dim     = hidden_dim
        self.poly_order     = poly_order
        self.message_steps  = message_steps
        self.n_outputs      = n_outputs
        self.activation     = activation
        self.encoder_act    = encoder_act
        self.normalize_input = normalize_input
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.hard_constraints = hard_constraints

        # ---- Fixed mesh arrays (JAX) ---------------------------------------
        verts = domain._vertices[:, :domain._spatial_dims]   # (n_nodes, 2)
        faces = domain._faces                                  # (n_faces, 3)

        self._nodes_np = verts.astype(np.float32)
        self._faces_np = faces.astype(np.int32)

        self.n_nodes = verts.shape[0]
        self.n_faces = faces.shape[0]
        self.spatial_dims = domain._spatial_dims

        # Normalise node coords to [-1, 1] for encoder input
        node_min = verts.min(axis=0)
        node_max = verts.max(axis=0)
        self._node_min = jnp.array(node_min, dtype=jnp.float32)
        self._node_max = jnp.array(node_max, dtype=jnp.float32)
        nodes_norm = (2.0 * (verts - node_min) / (node_max - node_min + 1e-8) - 1.0)
        self._nodes_norm = jnp.array(nodes_norm, dtype=jnp.float32)

        # ---- Build undirected edge list from faces --------------------------
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
        self._edge_src = jnp.array(src)
        self._edge_dst = jnp.array(dst)

        # ---- Precompute L_hat edge weights ---------------------------------
        # L      = I - D^{-1/2} A D^{-1/2}   (normalised Laplacian)
        # L_hat  = L - I = -D^{-1/2} A D^{-1/2}
        # For undirected edge (i,j): L_hat[i,j] = -1 / sqrt(d_i * d_j)
        degree = np.zeros(self.n_nodes, dtype=np.float32)
        for s, d in zip(src, dst):
            degree[s] += 1
            degree[d] += 1
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1.0))
        # weight = -1 / sqrt(deg_src * deg_dst)  (off-diagonal entry of L_hat)
        edge_w = -d_inv_sqrt[src] * d_inv_sqrt[dst]
        self._edge_weights = jnp.array(edge_w, dtype=jnp.float32)

        # ---- Flax GNN module ------------------------------------------------
        self._module = _GNNModule(
            spatial_dims  = self.spatial_dims,
            hidden_dim    = hidden_dim,
            poly_order    = poly_order,
            message_steps = message_steps,
            n_outputs     = n_outputs,
            activation    = activation,
            encoder_act   = encoder_act,
        )

        # Input/output normalisation bounds (set by trainer via set_*_range)
        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None

        # ---- Hard Dirichlet BC arrays --------------------------------------
        # Build a (n_nodes, n_outputs) mask and values array from the domain's
        # Dirichlet BCs.  At forward time we overwrite constrained node outputs
        # with their prescribed values before barycentric interpolation.
        bc_mask   = np.zeros((self.n_nodes, n_outputs), dtype=np.float32)
        bc_values = np.zeros((self.n_nodes, n_outputs), dtype=np.float32)
        if hard_constraints:
            from pinns.boundary import MeshNodeBC
            for bc in getattr(domain, 'boundary_conditions', []):
                if not (isinstance(bc, MeshNodeBC) and bc.bc_type == 'dirichlet'):
                    continue
                comp = bc.component
                if comp >= n_outputs:
                    continue
                # Resolve node indices from stored edge pairs
                node_idx = np.unique(bc.edges)           # (n_bc_nodes,)
                node_pos = verts[node_idx]               # (n_bc_nodes, 2)
                # Evaluate prescribed value (scalar or callable)
                vals = bc.get_value(node_pos)            # (n_bc_nodes,)
                bc_mask[node_idx, comp]   = 1.0
                bc_values[node_idx, comp] = vals
        self._bc_mask_jnp   = jnp.array(bc_mask,   dtype=jnp.float32)
        self._bc_values_jnp = jnp.array(bc_values, dtype=jnp.float32)

    # ---------------------------------------------------------------------- #
    #  Trainer-compatible API                                                 #
    # ---------------------------------------------------------------------- #

    def set_input_range(self, xmin: np.ndarray, xmax: np.ndarray):
        """Set input normalisation bounds (called by the trainer)."""
        self.input_min = jnp.array(xmin, dtype=jnp.float32)
        self.input_max = jnp.array(xmax, dtype=jnp.float32)

    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        """Set output un-normalisation bounds (called by the trainer)."""
        self.output_min = jnp.array(ymin, dtype=jnp.float32)
        self.output_max = jnp.array(ymax, dtype=jnp.float32)

    def init(self, rng: jax.random.PRNGKey, dummy_input=None) -> Dict:
        """
        Initialise network parameters.

        Parameters
        ----------
        rng : PRNGKey
        dummy_input : ignored (kept for API compatibility with FNN)

        Returns
        -------
        dict : Flax parameter tree
        """
        return self._module.init(
            rng,
            self._nodes_norm,
            self._edge_src,
            self._edge_dst,
            self._edge_weights,
            self.n_nodes,
        )

    def apply(
        self,
        params: Dict,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """
        Evaluate the network at query points *x*.

        Steps
        -----
        1. Apply optional ``input_transform(x, params_dict)``.
        2. Optionally normalise *x* to ``[-1, 1]``.
        3. Run GNN on fixed mesh to obtain nodal coefficients.
        4. Barycentric interpolation at the *spatial* part of *x*.
        5. Optionally un-normalise output.
        6. Apply optional ``output_transform(x_original, y, params_dict)``.

        Parameters
        ----------
        params : dict
            Flax parameter tree from :meth:`init`.
        x : jnp.ndarray, shape ``(n_query, n_dims)``
            Query coordinates.  For space-time problems the spatial
            dimensions must come first (``x[:, :spatial_dims]``).
        params_dict : dict, optional
            Passed to ``input_transform`` / ``output_transform``.

        Returns
        -------
        jnp.ndarray, shape ``(n_query, n_outputs)``
        """
        x_original = x

        # Optional input transform (e.g. symmetry encoding)
        if self.input_transform is not None:
            x = self.input_transform(x, params_dict)

        # Normalise query coords to [-1, 1]
        if self.normalize_input and self.input_min is not None:
            x_norm = (2.0 * (x - self.input_min)
                      / (self.input_max - self.input_min + 1e-8) - 1.0)
        else:
            x_norm = x   # use as-is for spatial interpolation below

        # --- GNN forward pass on fixed mesh ---------------------------------
        # Node features are always the normalised *mesh* node coordinates,
        # independent of the query batch.
        node_coeffs = self._module.apply(
            params,
            self._nodes_norm,
            self._edge_src,
            self._edge_dst,
            self._edge_weights,
            self.n_nodes,
        )   # (n_nodes, n_outputs)

        # --- Hard Dirichlet BC enforcement (static condensation) ------------
        # Replace constrained node outputs with their prescribed BC values.
        # bc_mask is 1 at constrained DOFs, 0 elsewhere.
        if self.hard_constraints:
            node_coeffs = (node_coeffs * (1.0 - self._bc_mask_jnp)
                           + self._bc_values_jnp * self._bc_mask_jnp)

        # --- Spatial interpolation at query points --------------------------
        # Use the *original* (un-normalised) spatial coords of the mesh nodes
        # together with *original* spatial coords of the query points,
        # so that barycentric coordinates remain correct.
        nodes_orig = jnp.array(self._nodes_np)
        faces_arr  = jnp.array(self._faces_np)
        x_spatial  = x_original[:, :self.spatial_dims]

        y = _interpolate_mesh(node_coeffs, nodes_orig, faces_arr, x_spatial)
        # y: (n_query, n_outputs)

        # Optional output un-normalisation
        if self.output_min is not None:
            y = (y + 1.0) / 2.0 * (self.output_max - self.output_min) + self.output_min

        # Optional hard-constraint transform
        if self.output_transform is not None:
            y = self.output_transform(x_original, y, params_dict)

        return y

    def forward(
        self,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """Forward pass using stored ``self.params``."""
        return self.apply(self.params, x, params_dict)

    def predict(
        self,
        x_np: np.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> np.ndarray:
        """Predict with NumPy I/O."""
        return np.array(self.forward(jnp.array(x_np), params_dict))

    # ---------------------------------------------------------------------- #
    #  Introspection / utilities                                              #
    # ---------------------------------------------------------------------- #

    @property
    def mesh_nodes(self) -> np.ndarray:
        """Mesh vertex positions, shape ``(n_nodes, spatial_dims)``."""
        return self._nodes_np.copy()

    @property
    def mesh_faces(self) -> np.ndarray:
        """Triangular element connectivity, shape ``(n_faces, 3)``."""
        return self._faces_np.copy()

    def get_node_coefficients(self, params: Dict) -> np.ndarray:
        """
        Return the nodal coefficients produced by the GNN.

        Useful for post-processing / visualisation.

        Parameters
        ----------
        params : dict  – network parameters

        Returns
        -------
        np.ndarray, shape ``(n_nodes, n_outputs)``
        """
        coeffs = self._module.apply(
            params,
            self._nodes_norm,
            self._edge_src,
            self._edge_dst,
            self._edge_weights,
            self.n_nodes,
        )
        return np.array(coeffs)

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

    def __repr__(self) -> str:
        return (
            f"GNNMeshNetwork("
            f"n_nodes={self.n_nodes}, n_faces={self.n_faces}, "
            f"hidden_dim={self.hidden_dim}, poly_order={self.poly_order}, "
            f"message_steps={self.message_steps}, n_outputs={self.n_outputs}, "
            f"hard_constraints={self.hard_constraints})"
        )
