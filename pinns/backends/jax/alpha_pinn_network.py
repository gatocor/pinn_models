"""
AlphaPINNNetwork — Laplace-spectral state integrator for mesh-based PINNs (JAX/Flax).

Architecture
------------
1. **Precompute** the K smallest eigenvectors Φ of the mesh graph Laplacian.
   Φ has shape (n_nodes, K) and forms an orthonormal spectral basis on the mesh.

2. **Encode** the current nodal state u^n into spectral coefficients:
       α^n = Φᵀ u^n   ∈ ℝᴷ

3. **FNN** maps the low-dimensional spectral code to the *next* spectral code:
       α^{n+1} = MLP(α^n)   ∈ ℝᴷ

4. **Decode** back to nodal values:
       u^{n+1} = Φ α^{n+1}   ∈ ℝ^{n_nodes}

5. Hard Dirichlet BCs are enforced by static condensation (same as GNNMeshNetwork).
6. Barycentric interpolation evaluates u at arbitrary query points.

The spectral bottleneck (only K modes) acts as a built-in low-pass filter and
regulariser — only smooth functions can be represented, which is physically
motivated for diffusion-type PDEs.

Public API mirrors ``GNNMeshNetwork`` / ``FNN`` exactly so it can be swapped
in place with no other code changes::

    net = AlphaPINNNetwork(domain, n_eigenvectors=32, hidden_dim=128,
                           n_layers=4, use_state=True)
    params = net.init(jax.random.PRNGKey(0))
    y      = net.apply(params, x_query, params_dict)
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from flax import linen as nn
from typing import Dict, Optional, Sequence, Callable, Any

from .gnn_network import _interpolate_mesh
from .networks import get_activation


# ─────────────────────────────────────────────────────────────────────────────
# Flax MLP
# ─────────────────────────────────────────────────────────────────────────────

class _MLPModule(nn.Module):
    """Standard multi-layer perceptron."""
    features: Sequence[int]
    activation: str = 'tanh'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = get_activation(self.activation)
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = act(x)
        x = nn.Dense(self.features[-1])(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Laplace eigenvector computation (scipy, done once at construction)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_laplace_eigenvectors(
    verts: np.ndarray,   # (n_nodes, 2)
    faces: np.ndarray,   # (n_faces, 3)  int
    K: int,
) -> np.ndarray:
    """
    Compute the K eigenvectors of the normalised symmetric graph Laplacian
    L_sym = D^{-1/2} (D - A) D^{-1/2} with the smallest eigenvalues.

    Returns
    -------
    Phi : (n_nodes, K) float32, columns are L2-normalised eigenvectors.
        The first column corresponds to the constant (λ=0) mode.
    """
    n = len(verts)

    # Build adjacency from triangle edges
    rows, cols = [], []
    for face in faces:
        for j in range(3):
            v0, v1 = int(face[j]), int(face[(j + 1) % 3])
            rows.extend([v0, v1])
            cols.extend([v1, v0])
    data = np.ones(len(rows), dtype=np.float64)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # Cap at 1 (remove duplicate entries)
    A.data = np.ones_like(A.data)

    # Degree matrix
    degree = np.array(A.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    # Normalised Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    L_sym = sp.eye(n, format='csr') - D_inv_sqrt @ A @ D_inv_sqrt

    # K smallest eigenpairs (including λ≈0 constant mode)
    K_req = min(K, n - 2)
    eigenvalues, eigenvectors = spla.eigsh(L_sym, k=K_req, which='SM')

    # Sort by eigenvalue (ascending)
    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]   # (n_nodes, K_req)

    # Pad with zeros if fewer eigenvectors were found
    if K_req < K:
        pad = np.zeros((n, K - K_req), dtype=np.float64)
        eigenvectors = np.concatenate([eigenvectors, pad], axis=1)

    return eigenvectors.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public AlphaPINNNetwork class
# ─────────────────────────────────────────────────────────────────────────────

class AlphaPINNNetwork:
    """
    Laplace-spectral state integrator for mesh-based PINNs.

    The network projects the current state u^n (at mesh nodes) onto the K
    smallest Laplace eigenvectors, applies a standard FNN in that spectral
    space, and decodes back to nodal values u^{n+1}.

    Identical public API to ``GNNMeshNetwork`` — can be swapped in with no
    other code changes.

    Parameters
    ----------
    domain : DomainMesh
        2-D triangular mesh domain.
    n_eigenvectors : int
        Number of Laplace eigenvectors K.  Controls the spectral resolution
        of the state representation.  Larger values = richer modes but a
        higher-dimensional FNN input.  Typical range: 16–128.
    hidden_dim : int
        Width of each hidden layer in the FNN.
    n_layers : int
        Number of hidden layers.  The MLP is:
        K  →  [hidden_dim] × n_layers  →  K
    activation : str
        Activation function inside the FNN (default ``'tanh'``).
    n_outputs : int
        Number of PDE unknowns / output channels per node (default 1).
    hard_constraints : bool
        Enforce Dirichlet BCs by static condensation (same as GNNMeshNetwork).
    use_state : bool
        Must be ``True`` for the time-stepping (state-integrator) role.
        When ``False``, the FNN maps a **learned** spectral code (stored in
        the network parameters) to nodal values — useful for static PDEs.
    normalize_input : bool
        Normalise query coordinates to ``[-1, 1]`` before interpolation.
    input_transform : callable, optional
        Applied to query coordinates before normalisation.
    output_transform : callable, optional
        Hard-constraint transform applied after interpolation.

    Notes
    -----
    *Eigenvector computation* (scipy, CPU, done once at construction) can take
    a few seconds for large meshes.  The result is stored as a fixed JAX array.
    """

    def __init__(
        self,
        domain,
        n_eigenvectors: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 4,
        activation: str = 'tanh',
        n_outputs: int = 1,
        hard_constraints: bool = True,
        use_state: bool = True,
        normalize_input: bool = True,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
    ):
        if domain._spatial_dims != 2:
            raise NotImplementedError(
                "AlphaPINNNetwork supports 2-D spatial meshes only "
                f"(got spatial_dims={domain._spatial_dims})."
            )

        self.n_eigenvectors  = n_eigenvectors
        self.hidden_dim      = hidden_dim
        self.n_layers        = n_layers
        self.activation      = activation
        self.n_outputs       = n_outputs
        self.hard_constraints = hard_constraints
        self.use_state       = use_state
        self.normalize_input = normalize_input
        self.input_transform = input_transform
        self.output_transform = output_transform

        # Time interval (optional — same handling as GNNMeshNetwork)
        t_interval = getattr(domain, 't_interval', None)
        self.has_time = t_interval is not None
        if self.has_time:
            self.t_min = float(t_interval[0])
            self.t_max = float(t_interval[1])

        # ── Fixed mesh arrays ──────────────────────────────────────────────
        verts = domain._vertices[:, :domain._spatial_dims].astype(np.float32)
        faces = domain._faces.astype(np.int32)

        self._nodes_np = verts
        self._faces_np = faces
        self.n_nodes   = verts.shape[0]
        self.n_faces   = faces.shape[0]
        self.spatial_dims = domain._spatial_dims

        node_min = verts.min(axis=0)
        node_max = verts.max(axis=0)
        self._node_min = jnp.array(node_min, dtype=jnp.float32)
        self._node_max = jnp.array(node_max, dtype=jnp.float32)

        # ── Laplace eigenvectors ──────────────────────────────────────────
        print(f"AlphaPINNNetwork: computing {n_eigenvectors} Laplace eigenvectors "
              f"for mesh with {self.n_nodes} nodes … ", end="", flush=True)
        _Phi = _compute_laplace_eigenvectors(verts, faces, n_eigenvectors)  # (n_nodes, K)
        self._Phi     = jnp.array(_Phi,     dtype=jnp.float32)   # (n_nodes, K)
        self._Phi_T   = jnp.array(_Phi.T,   dtype=jnp.float32)   # (K, n_nodes)
        print("done.")

        # ── FNN: K*n_outputs  →  hidden ×n_layers  →  K*n_outputs ──────────
        # (We flatten multi-output spectral codes: α has shape K*n_outputs)
        alpha_dim = n_eigenvectors * n_outputs
        # Input: spectral code of u^n (state mode) or zeros (static mode)
        # Output: spectral code of u^{n+1}
        fnn_features = [hidden_dim] * n_layers + [alpha_dim]
        self._module = _MLPModule(features=fnn_features, activation=activation)

        # ── Input/output normalisation (set by trainer) ──────────────────
        self.input_min  = None
        self.input_max  = None
        self.output_min = None
        self.output_max = None

        # ── Hard Dirichlet BCs (static condensation) ─────────────────────
        bc_mask   = np.zeros((self.n_nodes, n_outputs), dtype=np.float32)
        bc_values = np.zeros((self.n_nodes, n_outputs), dtype=np.float32)
        if hard_constraints:
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
                node_idx = np.unique(bc.edges)
                node_pos = verts[node_idx]
                vals = bc.get_value(node_pos)
                bc_mask[node_idx, comp]   = 1.0
                bc_values[node_idx, comp] = vals
        self._bc_mask_jnp   = jnp.array(bc_mask,   dtype=jnp.float32)
        self._bc_values_jnp = jnp.array(bc_values, dtype=jnp.float32)

    # ──────────────────────────────────────────────────────────────────────── #
    #  Trainer-compatible API                                                  #
    # ──────────────────────────────────────────────────────────────────────── #

    def set_input_range(self, xmin: np.ndarray, xmax: np.ndarray):
        self.input_min = jnp.array(xmin, dtype=jnp.float32)
        self.input_max = jnp.array(xmax, dtype=jnp.float32)

    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        self.output_min = jnp.array(ymin, dtype=jnp.float32)
        self.output_max = jnp.array(ymax, dtype=jnp.float32)

    def init(self, rng: jax.random.PRNGKey, dummy_input=None) -> Dict:
        """
        Initialise network parameters.

        The FNN is initialised with a dummy zero spectral code as input so
        that Flax can infer all layer shapes.
        """
        alpha_dim = self.n_eigenvectors * self.n_outputs
        dummy_alpha = jnp.zeros((alpha_dim,), dtype=jnp.float32)
        return self._module.init(rng, dummy_alpha)

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
        1. Encode current state u^n to spectral coefficients α^n = Φᵀ u^n
           (only in ``use_state=True`` mode; otherwise uses a zero code that
           the FNN maps to a learned static solution).
        2. FNN: α^n → α^{n+1}  (spectral evolution).
        3. Decode: u^{n+1} = Φ α^{n+1}  (n_nodes, n_outputs).
        4. Hard BC enforcement (static condensation).
        5. Barycentric interpolation at spatial coordinates of *x*.

        Parameters
        ----------
        params : dict           – Flax parameter tree from :meth:`init`.
        x : (n_query, n_dims)   – query coordinates (spatial first).
        params_dict : dict      – must contain ``params_dict["fixed"]["u_prev_nodes"]``
                                  when ``use_state=True``.

        Returns
        -------
        (n_query, n_outputs)
        """
        x_original = x

        if self.input_transform is not None:
            x = self.input_transform(x, params_dict)

        # ── 1. Build spectral code from current state ─────────────────────
        if self.use_state and params_dict is not None:
            u_prev = params_dict["fixed"]["u_prev_nodes"]   # (n_nodes,) or (n_nodes, n_out)
            u_prev = jnp.asarray(u_prev, dtype=jnp.float32)
            if u_prev.ndim == 1:
                u_prev = u_prev[:, None]                    # (n_nodes, 1)
            # Project each output channel independently, then flatten
            # alpha_n: (K * n_outputs,)
            alpha_n = (self._Phi_T @ u_prev).ravel()        # (K*n_out,)
        else:
            # Static mode: zero input → network learns a constant spectral code
            alpha_n = jnp.zeros(self.n_eigenvectors * self.n_outputs,
                                 dtype=jnp.float32)

        # ── 2. FNN: spectral space evolution ─────────────────────────────
        alpha_next = self._module.apply(params, alpha_n)    # (K*n_out,)

        # ── 3. Decode to nodal values ─────────────────────────────────────
        # Reshape alpha to (K, n_outputs) so Φ @ alpha → (n_nodes, n_outputs)
        alpha_next_2d = alpha_next.reshape(self.n_eigenvectors, self.n_outputs)
        node_coeffs   = self._Phi @ alpha_next_2d           # (n_nodes, n_outputs)

        # ── 4. Hard Dirichlet BC enforcement ─────────────────────────────
        if self.hard_constraints:
            node_coeffs = (node_coeffs * (1.0 - self._bc_mask_jnp)
                           + self._bc_values_jnp * self._bc_mask_jnp)

        # ── 5. Barycentric interpolation at query points ──────────────────
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

    def forward(self, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        """Forward pass using stored ``self.params``."""
        return self.apply(self.params, x, params_dict)

    def predict(self, x_np: np.ndarray, params_dict=None) -> np.ndarray:
        """Predict with NumPy I/O."""
        return np.array(self.forward(jnp.array(x_np), params_dict))

    def get_node_coefficients(self, params: Dict) -> np.ndarray:
        """
        Return the decoded nodal values for the *zero* spectral input (static).

        Useful for inspecting the learned static solution (``use_state=False``).
        """
        alpha_zero = jnp.zeros(self.n_eigenvectors * self.n_outputs, dtype=jnp.float32)
        alpha_out  = self._module.apply(params, alpha_zero)
        alpha_2d   = alpha_out.reshape(self.n_eigenvectors, self.n_outputs)
        return np.array(self._Phi @ alpha_2d)

    def to(self, device=None, dtype=None, seed: int = 0) -> 'AlphaPINNNetwork':
        """PyTorch-compatible no-op for JAX."""
        if not hasattr(self, 'params') or self.params is None:
            self.params = self.init(jax.random.PRNGKey(seed))
        return self

    @property
    def mesh_nodes(self) -> np.ndarray:
        return self._nodes_np.copy()

    @property
    def mesh_faces(self) -> np.ndarray:
        return self._faces_np.copy()

    @property
    def eigenvectors(self) -> np.ndarray:
        """Laplace eigenvectors Φ, shape ``(n_nodes, K)``."""
        return np.array(self._Phi)

    def __repr__(self) -> str:
        return (
            f"AlphaPINNNetwork("
            f"n_nodes={self.n_nodes}, K={self.n_eigenvectors}, "
            f"hidden_dim={self.hidden_dim}, n_layers={self.n_layers}, "
            f"n_outputs={self.n_outputs}, "
            f"hard_constraints={self.hard_constraints}, "
            f"use_state={self.use_state})"
        )
