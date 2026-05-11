"""
Laplacian spectral feature encoder for mesh-based PINNs (JAX).

LaplacianFeatures maps spatial query points to the K smallest Laplace–Beltrami
eigenvectors of the mesh graph, evaluated via barycentric interpolation.
This provides a geometry-aware, low-pass feature encoding ("AlphaTransform")
that regularises PDE solutions toward smooth functions.
"""
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, Optional

from .gnn import _interpolate_mesh


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


class LaplacianFeatures:
    """
    Laplacian spectral feature encoding for mesh-based PINNs.

    Maps spatial query points ``(x, y)`` or ``(x, y, t)`` to the K smallest
    Laplace–Beltrami eigenvectors of the mesh graph evaluated via barycentric
    interpolation.  The resulting feature vector is geometry-aware and acts as
    a built-in low-pass filter — only smooth (physically motivated) functions
    can be represented.

    Drop-in replacement for :class:`FourierFeatures` — identical callable API.

    **Time handling** — if *domain* carries a ``t_interval``:

    * ``n_features - 1`` spatial eigenvectors are computed.
    * The last feature column is the raw ``t`` coordinate.
    * ``output_dim == n_features`` in both cases.

    **Context encoding** — if ``n_context > 0`` and ``encode_context=True``
    (the default), each context field ``u_j(x)`` is projected onto the
    eigenbasis by pointwise multiplication:

    .. math::

        \\text{output} = [\\phi(x),\\; \\phi(x)\\cdot u_1(x),\\; \\ldots,\\;
                          \\phi(x)\\cdot u_{n_c}(x)]

    giving ``n_eig * (1 + n_context)`` spectral features.  Set
    ``encode_context=False`` to fall back to raw concatenation
    (``n_eig + n_context`` features).

    Parameters
    ----------
    domain : DomainMesh
        2-D triangular mesh domain.  Must expose ``_vertices``, ``_faces``,
        ``_spatial_dims``, and optionally ``t_interval``.
    n_features : int
        Number of spatial Laplace eigenvectors to compute.  When the domain
        has a time axis, ``n_eig = n_features`` and ``t`` is appended
        separately, so ``output_dim = n_features * (1 + n_context) + 1``.
    n_context : int
        Number of extra context columns in the input *after* spatial (+time)
        coordinates (e.g. solution values U from the previous time step).
    encode_context : bool
        If ``True`` (default), context columns are spectrally encoded via
        ``φ(x) * u_j(x)``.  If ``False``, they are appended raw.

    Example::

        enc = LaplacianFeatures(domain, n_features=32)
        net = FNN(layer_sizes=[enc.output_dim, 128, 128, 1],
                  feature_encoding=enc)
    """

    def __init__(
        self,
        domain=None,
        n_features: int = 32,
        n_context: int = 0,
        encode_context: bool = True,
    ):
        self.n_features     = n_features
        self.n_context      = n_context
        self.encode_context = encode_context
        self.n_eig          = n_features

        # Set by _build_from_domain / _configure
        self.has_time     = None
        self.spatial_dims = None
        self.output_dim   = None
        self._nodes_np    = None
        self._faces_np    = None
        self._Phi         = None

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
        """Compute Laplace eigenvectors from (optionally transformed) mesh vertices."""
        self._nodes_np = verts
        self._faces_np = faces

        if self.encode_context:
            self.output_dim = (
                self.n_features * (1 + self.n_context)
                + (1 if self.has_time else 0)
            )
        else:
            self.output_dim = (
                self.n_features
                + (1 if self.has_time else 0)
                + self.n_context
            )

        print(
            f"LaplacianFeatures: computing {self.n_eig} Laplace eigenvectors "
            f"for mesh with {len(verts)} nodes … ",
            end="",
            flush=True,
        )
        _Phi = _compute_laplace_eigenvectors(verts, faces, self.n_eig)
        self._Phi = jnp.array(_Phi, dtype=jnp.float32)  # (n_nodes, n_eig)
        print("done.")

    # ── ModelBase composable protocol ───────────────────────────────────── #

    def _configure(self, network, input_dim: int) -> int:
        """Called by ModelBase.add().  Builds eigenvectors using network's domain
        and any accumulated coordinate transforms from preceding Normalize layers."""
        self._build_from_domain(
            network.domain,
            coord_transform=network._apply_coord_transforms
            if network._coord_transforms else None,
        )
        return self.output_dim

    def init(self, rng) -> Dict:
        """No trainable parameters."""
        return {}

    def apply(
        self,
        params: Dict,
        x,
        params_dict=None,
    ):
        """ModelBase-protocol alias: apply(params, x) → forward pass."""
        return self.__call__(x, params_dict)

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
            Query coordinates.  Time column (if present) is appended as the
            last output feature.
        params_dict : ignored (API compatibility with ``input_transform``).

        Returns
        -------
        (n_pts, n_features)
        """
        nodes = jnp.array(self._nodes_np)
        faces = jnp.array(self._faces_np)
        x_spatial = x[:, : self.spatial_dims]
        phi_x = _interpolate_mesh(self._Phi, nodes, faces, x_spatial)  # (n_pts, n_eig)

        if self.n_context > 0:
            ctx = x[:, -self.n_context:]   # (n_pts, n_context)
            if self.encode_context:
                # φᵀu: for each context field u_j, compute φ(x) * u_j(x)
                # Block layout: [φ*u_1 | φ*u_2 | … | φ*u_nc]
                # (n_pts, n_eig, n_context) → transpose (0,2,1) → (n_pts, n_context, n_eig) → reshape
                n_pts = x.shape[0]
                ctx_encoded = phi_x[:, :, None] * ctx[:, None, :]   # (n_pts, n_eig, n_context)
                ctx_encoded = ctx_encoded.transpose(0, 2, 1).reshape(n_pts, -1)  # (n_pts, n_context*n_eig)
                parts = [phi_x, ctx_encoded]
            else:
                parts = [phi_x, ctx]
        else:
            parts = [phi_x]

        if self.has_time:
            parts.append(x[:, self.spatial_dims : self.spatial_dims + 1])

        return jnp.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]

    def transform(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """Alias for ``__call__``."""
        return self.__call__(x, params_dict)

    def __repr__(self) -> str:
        mode = f"{self.n_eig} eigenvectors" + (" + t" if self.has_time else "")
        ctx = ""
        if self.n_context > 0:
            enc = "φᵀu" if self.encode_context else "raw"
            ctx = f", n_context={self.n_context}({enc})"
        return f"LaplacianFeatures(output_dim={self.output_dim}, mode={mode!r}{ctx})"


# Alias: AlphaTransform is the name used by AlphaPINN for this encoder
AlphaTransform = LaplacianFeatures

__all__ = ["LaplacianFeatures", "AlphaTransform"]