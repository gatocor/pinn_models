"""
AlphaPINN — Laplace-spectral state integrator for mesh-based PINNs (JAX/Flax).

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

    net = AlphaPINN(domain, n_eigenvectors=32, hidden_dims=[128, 128, 128, 128], use_state=True)
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
from .networks import get_activation, FNN


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
# LaplacianFeatures — mesh-based spectral input encoder
# ─────────────────────────────────────────────────────────────────────────────

class LaplacianFeatures:
    """
    Laplacian spectral feature encoding for mesh-based PINNs.

    Maps spatial query points ``(x, y)`` [or ``(x, y, t)``] to the K smallest
    Laplace–Beltrami eigenvectors of the mesh graph, evaluated via barycentric
    interpolation, producing a geometry-aware feature vector.

    Drop-in replacement for :class:`FourierFeatures` — identical callable API::

        enc = LaplacianFeatures(domain, n_features=32)
        net = AlphaPINN(domain, n_features=32, hidden_dims=[128, 128, 128, 1])

    **Time handling** — if *domain* carries a ``t_interval``:

    * ``n_features - 1`` eigenvectors are computed (spatial modes).
    * The last feature column is the raw time coordinate ``t``.
    * ``output_dim == n_features`` in both cases.

    So ``n_features`` always controls network input width, regardless of whether
    the domain is purely spatial or space-time.

    Parameters
    ----------
    domain : DomainMesh
        2-D triangular mesh domain.  Must expose ``_vertices``, ``_faces``,
        ``_spatial_dims``, and optionally ``t_interval``.
    n_features : int
        Total output dimension.  Spatial eigenvectors = ``n_features`` (no time)
        or ``n_features - 1`` (with time).
    """

    def __init__(self, domain, n_features: int = 32):
        t_interval = getattr(domain, 't_interval', None)
        self.has_time     = t_interval is not None
        self.spatial_dims = domain._spatial_dims
        self.n_features   = n_features
        self.n_eig        = n_features - 1 if self.has_time else n_features
        self.output_dim   = n_features

        if self.has_time:
            self.t_min = float(t_interval[0])
            self.t_max = float(t_interval[1])

        verts = domain._vertices[:, :self.spatial_dims].astype(np.float32)
        faces = domain._faces.astype(np.int32)
        self._nodes_np = verts
        self._faces_np = faces

        print(
            f"LaplacianFeatures: computing {self.n_eig} Laplace eigenvectors "
            f"for mesh with {len(verts)} nodes … ", end="", flush=True
        )
        _Phi = _compute_laplace_eigenvectors(verts, faces, self.n_eig)
        self._Phi = jnp.array(_Phi, dtype=jnp.float32)   # (n_nodes, n_eig)
        print("done.")

    def __call__(
        self,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """
        Encode query coordinates into Laplacian spectral features.

        Parameters
        ----------
        x : (n_pts, spatial_dims) or (n_pts, spatial_dims+1)
            Query coordinates.  The time column (if present) is appended as
            the last output feature.
        params_dict : ignored (API compatibility with ``input_transform``).

        Returns
        -------
        (n_pts, n_features)
        """
        nodes = jnp.array(self._nodes_np)
        faces = jnp.array(self._faces_np)
        x_spatial = x[:, :self.spatial_dims]

        phi_x = _interpolate_mesh(self._Phi, nodes, faces, x_spatial)  # (n_pts, n_eig)

        if self.has_time:
            t_col = x[:, self.spatial_dims : self.spatial_dims + 1]    # (n_pts, 1)
            return jnp.concatenate([phi_x, t_col], axis=-1)            # (n_pts, n_features)
        return phi_x

    def __repr__(self) -> str:
        mode = f"{self.n_eig} eigenvectors + t" if self.has_time else f"{self.n_eig} eigenvectors"
        return f"LaplacianFeatures(output_dim={self.output_dim}, mode='{mode}')"


# ─────────────────────────────────────────────────────────────────────────────
# Public AlphaPINN class
# ─────────────────────────────────────────────────────────────────────────────

class AlphaPINN:
    """
    Laplace-spectral PINN — a standard FNN with a :class:`LaplacianFeatures`
    input encoding.

    The :class:`LaplacianFeatures` encoder maps each query point ``(x, y[, t])``
    to the K smallest Laplace eigenvectors of the mesh graph (evaluated via
    barycentric interpolation), plus the raw ``t`` coordinate when the domain
    is space-time.  The result is fed into a plain fully-connected network.

    **This is equivalent to** ``FNN([n_features]+hidden_dims+[n_outputs],
    normalize_input=False, feature_encoding=LaplacianFeatures(domain,
    n_features))`` — and in fact that is exactly the internal implementation.

    Hard boundary constraints (output clamping) should be specified via
    ``output_transform`` — typically obtained from
    ``ProblemWeak(..., hard_constraints=True).output_transform`` and passed
    in here.

    Parameters
    ----------
    domain : DomainMesh
        2-D triangular mesh domain.  Time mode is auto-detected from
        ``domain.t_interval``.
    n_features : int
        Spectral feature dimension (= FNN input width).  If domain has a time
        interval: ``n_features - 1`` eigenvectors + 1 time column.
    hidden_dims : sequence of int
        Width of each hidden layer, e.g. ``[128, 128, 128, 128]``.
    activation : str
        Activation function (default ``'tanh'``).
    n_outputs : int
        Number of PDE unknowns per query point (default 1).
    output_transform : callable, optional
        Hard-constraint or other post-processing transform applied *after*
        the FNN.  Signature: ``f(x_original, y, params_dict) → y``.
        Pass ``problem.output_transform`` here when hard BCs are needed.
    input_transform : callable, optional
        Applied to raw query coordinates *before* the feature encoding.
    """

    def __init__(
        self,
        domain,
        n_features: int = 32,
        hidden_dims: Sequence[int] = (128, 128, 128, 128),
        activation: str = 'tanh',
        n_outputs: int = 1,
        output_transform: Optional[Callable] = None,
        input_transform: Optional[Callable] = None,
    ):
        if domain._spatial_dims != 2:
            raise NotImplementedError(
                "AlphaPINN supports 2-D spatial meshes only "
                f"(got spatial_dims={domain._spatial_dims})."
            )

        self.n_features  = n_features
        self.hidden_dims = list(hidden_dims)
        self.n_outputs   = n_outputs
        self.activation  = activation

        # ── Spectral encoder ──────────────────────────────────────────────
        self.laplacian_features = LaplacianFeatures(domain, n_features)

        # ── FNN backbone ──────────────────────────────────────────────────
        # normalize_input=False: LaplacianFeatures operates on raw coords.
        # unnormalize_output=False: no output rescaling (PINN outputs are raw).
        layer_sizes = [n_features] + list(hidden_dims) + [n_outputs]
        self._fnn = FNN(
            layer_sizes       = layer_sizes,
            activation        = activation,
            normalize_input   = False,
            unnormalize_output= False,
            feature_encoding  = self.laplacian_features,
            input_transform   = input_transform,
            output_transform  = output_transform,
        )

        # Expose for repr / trainer introspection
        self.n_nodes      = self.laplacian_features._nodes_np.shape[0]
        self.spatial_dims = domain._spatial_dims
        self.has_time     = self.laplacian_features.has_time

    # ── Trainer-compatible API (delegate to _fnn) ─────────────────────── #

    def set_input_range(self, xmin: np.ndarray, xmax: np.ndarray):
        self._fnn.set_input_range(xmin, xmax)

    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        self._fnn.set_output_range(ymin, ymax)

    def init(self, rng: jax.random.PRNGKey, dummy_input=None) -> Dict:
        """Initialise FNN parameters."""
        if dummy_input is None:
            dummy_input = jnp.zeros((1, self.n_features), dtype=jnp.float32)
        return self._fnn._module.init(rng, dummy_input)

    def apply(
        self,
        params: Dict,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """Evaluate network at query points *x* — delegates to inner FNN."""
        return self._fnn.apply(params, x, params_dict)

    def forward(self, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        return self.apply(self.params, x, params_dict)

    def predict(self, x_np: np.ndarray, params_dict=None) -> np.ndarray:
        return np.array(self.apply(self.params, jnp.array(x_np), params_dict))

    def to(self, device=None, dtype=None, seed: int = 0) -> 'AlphaPINN':
        if not hasattr(self, 'params') or self.params is None:
            self.params = self.init(jax.random.PRNGKey(seed))
        return self

    @property
    def mesh_nodes(self) -> np.ndarray:
        return self.laplacian_features._nodes_np.copy()

    @property
    def eigenvectors(self) -> np.ndarray:
        """Laplace eigenvectors Φ, shape ``(n_nodes, n_eig)``."""
        return np.array(self.laplacian_features._Phi)

    def __repr__(self) -> str:
        mode = "space-time" if self.has_time else "spatial"
        return (
            f"AlphaPINN("
            f"n_nodes={self.n_nodes}, "
            f"encoding={self.laplacian_features!r}, "
            f"hidden_dims={self.hidden_dims}, "
            f"n_outputs={self.n_outputs}, "
            f"mode={mode!r})"
        )


# Backward-compatibility alias
AlphaPINNNetwork = AlphaPINN
