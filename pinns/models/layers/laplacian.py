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
    verts: np.ndarray,   # (n_nodes, D)  — 2-D or 3-D vertex positions
    faces: np.ndarray,   # (n_faces, 3)  int
    K: int,
) -> np.ndarray:
    """
    Compute the K eigenvectors of the Laplace–Beltrami operator via the FEM
    cotangent discretisation, solving the generalised eigenproblem:

        A u = λ M u

    where A is the cotangent stiffness matrix and M is the lumped mass matrix.
    This is geometry-aware and mesh-size-independent, unlike the graph Laplacian.

    Returns
    -------
    Phi : (n_nodes, K) float32, columns are mass-orthonormal eigenvectors
        (φ_i^T M φ_j = δ_ij).  The first column is the constant (λ≈0) mode.
    """
    n = len(verts)
    v = verts.astype(np.float64)

    # Triangle vertex indices
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]

    # Edge vectors within each triangle
    e0 = v[i2] - v[i1]   # edge opposite vertex 0
    e1 = v[i0] - v[i2]   # edge opposite vertex 1
    e2 = v[i1] - v[i0]   # edge opposite vertex 2

    # Triangle areas (half the cross-product norm)
    if v.shape[1] == 3:
        cross = np.cross(e0, -e2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)  # (n_faces,)
    else:
        areas = 0.5 * np.abs(e0[:, 0] * (-e2[:, 1]) - e0[:, 1] * (-e2[:, 0]))
    areas = np.maximum(areas, 1e-14)

    # Cotangent weights: cot(angle at vertex k) = (a · b) / (2 * area)
    # where a, b are the two edge vectors FROM that vertex.
    # At i0: edges from i0 are e2 (i0→i1) and -e1 (i0→i2)
    # At i1: edges from i1 are e0 (i1→i2) and -e2 (i1→i0)
    # At i2: edges from i2 are e1 (i2→i0) and -e0 (i2→i1)
    cot0 = 0.5 * np.einsum('ij,ij->i',  e2, -e1) / areas   # cot(angle at i0)
    cot1 = 0.5 * np.einsum('ij,ij->i',  e0, -e2) / areas   # cot(angle at i1)
    cot2 = 0.5 * np.einsum('ij,ij->i',  e1, -e0) / areas   # cot(angle at i2)

    # Clamp to avoid negative weights from obtuse triangles
    cot0 = np.maximum(cot0, 0.0)
    cot1 = np.maximum(cot1, 0.0)
    cot2 = np.maximum(cot2, 0.0)

    # Assemble stiffness matrix A (cotangent Laplacian).
    # Edge (i1,i2) opposite i0: weight cot0/2  →  off-diag -cot0/2, diag +cot0/2
    # Edge (i0,i2) opposite i1: weight cot1/2  →  off-diag -cot1/2, diag +cot1/2
    # Edge (i0,i1) opposite i2: weight cot2/2  →  off-diag -cot2/2, diag +cot2/2
    # 6 off-diagonal + 6 diagonal = 12 triplets per triangle
    rows = np.concatenate([i1, i2, i0, i2, i0, i1,      # off-diag
                            i1, i2, i0, i2, i0, i1])     # diagonal (same row)
    cols = np.concatenate([i2, i1, i2, i0, i1, i0,      # off-diag (swapped)
                            i1, i2, i0, i2, i0, i1])     # diagonal (same col)
    data = np.concatenate([-0.5*cot0, -0.5*cot0,
                            -0.5*cot1, -0.5*cot1,
                            -0.5*cot2, -0.5*cot2,
                             0.5*cot0,  0.5*cot0,
                             0.5*cot1,  0.5*cot1,
                             0.5*cot2,  0.5*cot2])
    A_coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A_mat = A_coo.tocsr()

    # Assemble lumped mass matrix M (one-third of surrounding triangle areas)
    mass = np.zeros(n, dtype=np.float64)
    np.add.at(mass, i0, areas / 3.0)
    np.add.at(mass, i1, areas / 3.0)
    np.add.at(mass, i2, areas / 3.0)
    mass = np.maximum(mass, 1e-14)
    M_mat = sp.diags(mass, format='csr')

    # Solve generalised eigenproblem A u = λ M u.
    # Since M is diagonal (lumped), transform to the equivalent standard form:
    #   L_sym = M^{-1/2} A M^{-1/2},  u = M^{-1/2} v
    # This avoids the slow shift-invert LU factorisation needed when passing M
    # to eigsh.  We add a tiny diagonal shift so the null-space mode (λ≈0)
    # doesn't cause convergence issues in ARPACK.
    m_invsqrt = 1.0 / np.sqrt(mass)
    M_invsqrt = sp.diags(m_invsqrt, format='csr')
    L_sym = M_invsqrt @ A_mat @ M_invsqrt          # standard symmetric problem
    L_reg = L_sym + 1e-8 * sp.eye(n, format='csr') # shift null-space away from 0

    K_req = min(K, n - 2)
    eigenvalues, vecs = spla.eigsh(L_reg, k=K_req, which='LM', sigma=0.0)
    eigenvalues -= 1e-8   # undo shift

    # Transform eigenvectors back: u = M^{-1/2} v, then mass-orthonormalise
    eigenvectors = M_invsqrt @ vecs   # (n, K_req)

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

    Drop-in replacement for :class:`RandomFourierFeatures` — identical callable API.

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
        self._face_tree    = None
        self._node_to_face = None
        self._node_to_faces = None  # (n_verts, max_valence) all adjacent faces

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
            f"LaplacianFeatures: computing {self.n_eig + 1} Laplace eigenvectors "
            f"for mesh with {len(verts)} nodes (dropping constant mode 0) … ",
            end="",
            flush=True,
        )
        _Phi = _compute_laplace_eigenvectors(verts, faces, self.n_eig + 1)
        phi = jnp.array(_Phi[:, 1:], dtype=jnp.float32)  # drop mode 0, keep (n_nodes, n_eig)
        phi = phi / jnp.abs(phi).max(axis=0, keepdims=True)  # normalize each mode to [-1, 1]
        self._Phi = phi

        # Precompute vertex→all-adjacent-faces map for 3-D meshes so
        # _interpolate_mesh can find the BEST containing face (highest
        # min barycentric coord) rather than one arbitrary adjacent face.
        self._face_tree    = None   # kept for API compat, no longer used
        self._node_to_face = None
        self._node_to_faces = None
        if verts.shape[1] == 3:
            _v2f_lists = [[] for _ in range(len(verts))]
            for _fi, _face in enumerate(faces):
                for _vi in _face:
                    _v2f_lists[_vi].append(_fi)
            _max_val = max(len(fl) for fl in _v2f_lists)
            _n2fs = np.full((len(verts), _max_val), -1, dtype=np.int32)
            for _vi, _fl in enumerate(_v2f_lists):
                _n2fs[_vi, :len(_fl)] = _fl
            self._node_to_faces = jnp.array(_n2fs)  # (n_verts, max_valence)
            self._node_to_face  = self._node_to_faces[:, 0]  # legacy compat
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
        phi_x = _interpolate_mesh(self._Phi, nodes, faces, x_spatial,
                                   node_to_faces=self._node_to_faces)  # (n_pts, n_eig)

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