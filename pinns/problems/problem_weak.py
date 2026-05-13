"""
Weak-form problem for Physics-Informed Neural Networks on meshes.

``ProblemWeak`` casts the PDE into its Galerkin weak form:

    Find u_θ such that for all test functions v_j in V_h:
        a(u_θ, v_j) = ℓ(v_j)

where *a* is the bilinear/volume form and *ℓ* encodes body forces and
Neumann boundary data.  Dirichlet conditions are enforced strongly via
the domain's ``add_dirichlet`` method and are *excluded* from the test
space at training time.

Only :class:`~pinns.domain.DomainMesh` is accepted as domain.

Test functions
--------------
Piecewise Lagrange basis functions of arbitrary polynomial order N ≥ 1
(P1, P2, P3, …) are supported via the ``lagrange_order`` parameter of
:class:`ProblemWeak` (default 1 → P1).

The global DOFs are enumerated as:
  - Corner DOFs (same as the mesh vertices).
  - Interior-edge DOFs: N-1 evenly-spaced nodes per unique mesh edge
    (shared between adjacent elements).
  - Interior-element DOFs: (N-1)·(N-2)/2 nodes inside each triangle
    (not shared).

For each element the local DOF ordering is:
  corners 0, 1, 2
  → edge 0→1 (s=1..N-1)
  → edge 1→2 (s=1..N-1)
  → edge 2→0 (s=1..N-1)
  → interior nodes

Cubature
--------
Triangle integrals are evaluated with Dunavant quadrature rules of a
user-specified polynomial exactness *order* (1–5).  Edge (boundary)
integrals use 1-D Gauss–Legendre rules of the same order.  For accurate
integration when using high-order test functions you should raise
``cubature_order`` (a rule of thumb: ``cubature_order ≥ 2*lagrange_order``).

Precomputed data
----------------
All quadrature data are assembled during ``__post_init__`` and stored as
numpy arrays on the problem object.  The trainer converts them to the
chosen backend tensors at compile time.

Volume data (stored as ``cubature_data``, a dict):
    ``pts``        (n_faces, n_qpts, 2)          – physical quadrature points
    ``weights``    (n_faces, n_qpts)              – weights (ref_weight × 2·area)
    ``phi``        (n_faces, n_qpts, n_local)     – basis values
    ``grad_phi``   (n_faces, n_qpts, n_local, 2) – physical gradients
    ``node_ids``   (n_faces, n_local)             – global DOF indices per face
    ``free_mask``  (n_dofs,) bool                 – True for non-Dirichlet DOFs

Neumann boundary data for each Neumann BC (stored in
``neumann_data``, a list of dicts per BC):
    ``pts``        (n_edges, n_eq, 2)    – physical quadrature points
    ``weights``    (n_edges, n_eq)        – weights (ref_weight × edge_len)
    ``phi``        (n_edges, n_eq, 2)    – P1 basis values at the 2 endpoints
    ``normals``    (n_edges, 2)          – outward unit normals
    ``edge_ids``   (n_edges, 2)          – global node indices of edge endpoints
    ``bc``                                – the ``TermMeshNodeBC`` object
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Any, List, Optional, Union

import numpy as np

from .base_problem import BaseProblem


# ---------------------------------------------------------------------------
# Dunavant quadrature rules on the reference triangle (ξ,η) ∈ {ξ+η≤1, ξ,η≥0}
# Exact for polynomials up to degree *order*.
# Each rule is a list of (ξ, η, weight) tuples; weights sum to 1/2 (area of
# the reference triangle).
# ---------------------------------------------------------------------------

def _triangle_cubature(order: int):
    """Return (pts, weights) with pts (n,2) and weights (n,); order 1-5."""
    if order <= 1:
        # 1-point centroid rule, exact degree 1
        pts = np.array([[1/3, 1/3]])
        w   = np.array([0.5])
    elif order == 2:
        # 3-point midpoint rule, exact degree 2
        pts = np.array([[1/6, 1/6],
                        [2/3, 1/6],
                        [1/6, 2/3]])
        w   = np.full(3, 1/6)
    elif order == 3:
        # 6-point degree-4 Dunavant rule (all positive weights); also integrates
        # degree-3 polynomials exactly – avoids the well-known negative-weight
        # 4-point degree-3 rule.
        a1, b1 = 0.108103018168070, 0.445948490915965
        a2, b2 = 0.816847572980459, 0.091576213509771
        pts = np.array([[a1, b1], [b1, a1], [b1, b1],
                        [a2, b2], [b2, a2], [b2, b2]])
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        w   = np.array([w1, w1, w1, w2, w2, w2])
    elif order == 4:
        # 6-point Dunavant, exact degree 4
        a1, b1 = 0.108103018168070, 0.445948490915965
        a2, b2 = 0.816847572980459, 0.091576213509771
        pts = np.array([[a1, b1], [b1, a1], [b1, b1],
                        [a2, b2], [b2, a2], [b2, b2]])
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        w   = np.array([w1, w1, w1, w2, w2, w2])
    else:
        # 7-point Dunavant, exact degree 5
        a1, b1 = 0.47014206410511509, 0.47014206410511509
        # Use symmetry class
        pts = np.array([[1/3, 1/3],
                        [a1,  b1],
                        [b1,  a1],
                        [1-2*a1, a1],
                        [0.10128650732345633, 0.10128650732345633],
                        [0.10128650732345633, 0.79742698535308720],
                        [0.79742698535308720, 0.10128650732345633]])
        w   = np.array([0.22500000000000000/2,
                        0.13239415278850618/2,
                        0.13239415278850618/2,
                        0.13239415278850618/2,
                        0.12593918054482717/2,
                        0.12593918054482717/2,
                        0.12593918054482717/2])

    return pts, w


def _edge_cubature_1d(order: int):
    """Gauss–Legendre on [0,1], (pts_1d, weights_1d); weights sum to 1."""
    n = max(1, (order + 1 + 1) // 2)   # ceil((order+1)/2) points
    pts, w = np.polynomial.legendre.leggauss(n)
    pts = 0.5 * pts + 0.5   # map [-1,1] → [0,1]
    w   = 0.5 * w
    return pts, w


# ---------------------------------------------------------------------------
# Generic Lagrange basis on the reference triangle for order N
# ---------------------------------------------------------------------------

def _ref_nodes_pqr(N: int):
    """
    Return the local DOF multi-indices as a list of (p, q, r) with p+q+r=N.

    Ordering:
      - Corners:      (N,0,0), (0,N,0), (0,0,N)
      - Edge 0→1:     (N-s, s, 0) for s=1..N-1
      - Edge 1→2:     (0, N-s, s) for s=1..N-1
      - Edge 2→0:     (s, 0, N-s) for s=1..N-1
      - Interior:     remaining (p,q,r) with p,q,r≥1

    The reference position of node (p,q,r) is (xi,eta) = (q/N, r/N).
    """
    nodes = []
    # corners
    nodes += [(N, 0, 0), (0, N, 0), (0, 0, N)]
    # edge 0→1 (r=0, away from corner 0 toward corner 1)
    for s in range(1, N):
        nodes.append((N - s, s, 0))
    # edge 1→2 (p=0)
    for s in range(1, N):
        nodes.append((0, N - s, s))
    # edge 2→0 (q=0)
    for s in range(1, N):
        nodes.append((s, 0, N - s))
    # interior
    for r in range(1, N):
        for q in range(1, N - r):
            p = N - q - r
            if p >= 1:
                nodes.append((p, q, r))
    return nodes


def _local_edge_dof_indices(le: int, N: int) -> list:
    """
    Local DOF indices within an element's ``elem_dofs`` row for the N+1 DOFs
    on local edge *le*, ordered from element corner *la* toward *lb*.

    le=0: corners (0→1) + interior edge-0 DOFs
    le=1: corners (1→2) + interior edge-1 DOFs
    le=2: corners (2→0) + interior edge-2 DOFs
    """
    n_ei      = N - 1
    offsets   = [3, 3 + n_ei, 3 + 2 * n_ei]
    endpoints = [(0, 1), (1, 2), (2, 0)]
    ea, eb    = endpoints[le]
    interior  = list(range(offsets[le], offsets[le] + n_ei))
    return [ea] + interior + [eb]   # length N+1


def _lagrange_basis_and_grad(pts_ref: np.ndarray, N: int):
    """
    Evaluate order-N Lagrange basis functions and their reference gradients
    at a set of reference-triangle points.

    Parameters
    ----------
    pts_ref : (n_pts, 2)  — (ξ, η) coordinates on the reference triangle
    N       : polynomial order (≥ 1)

    Returns
    -------
    phi          : (n_pts, n_local)       — basis values
    grad_phi_ref : (n_pts, n_local, 2)   — gradients w.r.t. (ξ, η)

    Notes
    -----
    The Lagrange basis function for multi-index (p,q,r) with p+q+r=N is

        φ_{p,q,r}(λ) = ∏_{s=0}^{p-1} (N λ₁-s)/(s+1)
                      × ∏_{s=0}^{q-1} (N λ₂-s)/(s+1)
                      × ∏_{s=0}^{r-1} (N λ₃-s)/(s+1)

    with barycentric coordinates  λ₁=1-ξ-η,  λ₂=ξ,  λ₃=η.
    Gradients w.r.t. (ξ,η) follow from the chain rule.
    """
    n_pts  = len(pts_ref)
    xi, eta = pts_ref[:, 0], pts_ref[:, 1]
    lam    = np.stack([1.0 - xi - eta, xi, eta], axis=1)  # (n_pts, 3)

    nodes  = _ref_nodes_pqr(N)
    n_dofs = len(nodes)

    phi          = np.ones((n_pts, n_dofs),    dtype=np.float64)
    dphi_dlam    = np.zeros((n_pts, n_dofs, 3), dtype=np.float64)

    for a, pqr in enumerate(nodes):
        # For each barycentric direction k, compute
        #   phi_k  = ∏_{s=0}^{m-1} (N*λ_k - s)/(s+1)
        #   dphi_k = d(phi_k)/d(λ_k)
        phi_k  = np.ones((n_pts, 3), dtype=np.float64)
        dphi_k = np.zeros((n_pts, 3), dtype=np.float64)

        for k in range(3):
            m = pqr[k]
            if m == 0:
                continue  # phi_k = 1, dphi_k = 0 (already initialised)

            # phi_k = ∏_{s=0}^{m-1} (N*λ_k - s) / (s+1)
            pk = np.ones(n_pts, dtype=np.float64)
            for s in range(m):
                pk *= (N * lam[:, k] - s) / (s + 1)
            phi_k[:, k] = pk

            # d(phi_k)/d(λ_k) via product rule
            dpk = np.zeros(n_pts, dtype=np.float64)
            for s in range(m):
                term = np.full(n_pts, N / (s + 1), dtype=np.float64)
                for t in range(m):
                    if t != s:
                        term *= (N * lam[:, k] - t) / (t + 1)
                dpk += term
            dphi_k[:, k] = dpk

        # φ_a = φ_k[:, 0] * φ_k[:, 1] * φ_k[:, 2]
        phi[:, a] = phi_k[:, 0] * phi_k[:, 1] * phi_k[:, 2]

        # dφ_a/dλ_k = dφ_k[:, k] * ∏_{j≠k} φ_k[:, j]
        for k in range(3):
            j, l = [jj for jj in range(3) if jj != k]
            dphi_dlam[:, a, k] = dphi_k[:, k] * phi_k[:, j] * phi_k[:, l]

    # Chain rule from (λ) to (ξ, η):
    #   dφ/dξ  = dφ/dλ₁·(-1) + dφ/dλ₂·(1) + dφ/dλ₃·(0)
    #   dφ/dη  = dφ/dλ₁·(-1) + dφ/dλ₂·(0) + dφ/dλ₃·(1)
    grad_phi_ref = np.empty((n_pts, n_dofs, 2), dtype=np.float64)
    grad_phi_ref[:, :, 0] = -dphi_dlam[:, :, 0] + dphi_dlam[:, :, 1]
    grad_phi_ref[:, :, 1] = -dphi_dlam[:, :, 0] + dphi_dlam[:, :, 2]

    return phi, grad_phi_ref


# ---------------------------------------------------------------------------
# Higher-order DOF generation
# ---------------------------------------------------------------------------

def _build_higher_order_mesh(vertices: np.ndarray,
                              faces:    np.ndarray,
                              N:        int):
    """
    Generate all global DOF positions and element connectivity for order-N
    Lagrange elements on a triangular mesh.

    Parameters
    ----------
    vertices : (n_verts, 2)
    faces    : (n_faces, 3) — corner vertex indices (P1 connectivity)
    N        : Lagrange order (≥ 1)

    Returns
    -------
    dof_coords : (n_dofs, 2)   — physical coordinates of every DOF
    elem_dofs  : (n_faces, n_local)  — global DOF indices per element
    edge_to_dofs : dict (i_min, i_max) → list of global-DOF indices
        Interior edge DOFs in canonical direction (i_min → i_max), length N-1.
    """
    if N == 1:
        return vertices.copy(), faces.copy(), {}

    dof_coords = list(vertices)          # start with vertex DOFs

    # ── Interior edge DOFs ─────────────────────────────────────────────────
    edge_to_dofs: dict = {}   # canonical (i_min, i_max) → [dof_idx, ...]
    for face in faces:
        for a, b in [(face[0], face[1]),
                     (face[1], face[2]),
                     (face[2], face[0])]:
            key = (min(a, b), max(a, b))
            if key not in edge_to_dofs:
                x0, x1 = vertices[a], vertices[b]
                dofs = []
                for s in range(1, N):
                    t = s / N
                    dof_coords.append((1 - t) * x0 + t * x1)
                    dofs.append(len(dof_coords) - 1)
                # stored in canonical direction (min-index → max-index)
                edge_to_dofs[key] = dofs

    # ── Interior element DOFs ──────────────────────────────────────────────
    face_interior_dofs = []
    for face in faces:
        i0, i1, i2 = face
        x0, x1, x2 = vertices[i0], vertices[i1], vertices[i2]
        interior = []
        for r in range(1, N):
            for q in range(1, N - r):
                p = N - q - r
                if p >= 1:
                    xi, eta = q / N, r / N
                    pos = (1 - xi - eta) * x0 + xi * x1 + eta * x2
                    dof_coords.append(pos)
                    interior.append(len(dof_coords) - 1)
        face_interior_dofs.append(interior)

    dof_coords = np.array(dof_coords, dtype=np.float64)

    # ── Assemble elem_dofs ────────────────────────────────────────────────
    def _edge_dofs_ordered(a, b):
        """Return edge-interior DOF indices from a toward b (s=1..N-1)."""
        key = (min(a, b), max(a, b))
        dofs = edge_to_dofs[key]
        if a > b:               # reverse canonical direction
            dofs = list(reversed(dofs))
        return dofs

    elem_dofs = []
    for f, face in enumerate(faces):
        i0, i1, i2 = face
        dofs = [i0, i1, i2]
        dofs.extend(_edge_dofs_ordered(i0, i1))
        dofs.extend(_edge_dofs_ordered(i1, i2))
        dofs.extend(_edge_dofs_ordered(i2, i0))
        dofs.extend(face_interior_dofs[f])
        elem_dofs.append(dofs)

    return dof_coords, np.array(elem_dofs, dtype=np.int64), edge_to_dofs


# ---------------------------------------------------------------------------
# Main precomputation
# ---------------------------------------------------------------------------

def _precompute_volume(vertices:   np.ndarray,
                       faces:      np.ndarray,
                       cub_order:  int,
                       lag_order:  int = 1):
    """
    Precompute volume cubature data for all triangular elements.

    Parameters
    ----------
    vertices  : (n_verts, 2)  — corner vertex positions
    faces     : (n_faces, 3)  — corner indices (P1 connectivity)
    cub_order : quadrature exactness order (1–5)
    lag_order : Lagrange polynomial order N (≥ 1)

    Returns a dict with keys:
        pts        (n_faces, n_qpts, 2)
        weights    (n_faces, n_qpts)
        phi        (n_faces, n_qpts, n_local)
        grad_phi   (n_faces, n_qpts, n_local, 2)  — physical gradients
        node_ids   (n_faces, n_local)
        dof_coords (n_dofs, 2)
        edge_to_dofs  — passed through from _build_higher_order_mesh
    """
    N = lag_order
    ref_pts, ref_w   = _triangle_cubature(cub_order)
    phi_ref, gphi_ref = _lagrange_basis_and_grad(ref_pts, N)  # (Q,L), (Q,L,2)
    n_qpts  = len(ref_w)
    n_local = phi_ref.shape[1]

    dof_coords, elem_dofs, edge_to_dofs = _build_higher_order_mesh(vertices, faces, N)
    n_faces = len(faces)

    phys_pts  = np.empty((n_faces, n_qpts, 2),         dtype=np.float64)
    weights   = np.empty((n_faces, n_qpts),             dtype=np.float64)
    phi_all   = np.empty((n_faces, n_qpts, n_local),    dtype=np.float64)
    grad_all  = np.empty((n_faces, n_qpts, n_local, 2), dtype=np.float64)

    for k, (i0, i1, i2) in enumerate(faces):
        x0, x1, x2 = vertices[i0], vertices[i1], vertices[i2]
        # Jacobian J = [x1-x0 | x2-x0]  (2×2, columns)
        J     = np.stack([x1 - x0, x2 - x0], axis=1)
        det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        J_inv = np.linalg.inv(J)

        phys_pts[k] = x0 + ref_pts @ J.T           # (Q, 2)
        weights[k]  = ref_w * abs(det_J)            # (Q,)
        phi_all[k]  = phi_ref                       # (Q, L)

        # grad_phi_phys[q, a] = grad_phi_ref[q, a] @ J_inv  →  (Q, L, 2)
        # gphi_ref is (Q, L, 2); J_inv is (2, 2)
        grad_all[k] = gphi_ref @ J_inv              # broadcast: (Q,L,2)@(2,2) → (Q,L,2)

    return {
        'pts':          phys_pts.astype(np.float32),
        'weights':      weights.astype(np.float32),
        'phi':          phi_all.astype(np.float32),
        'grad_phi':     grad_all.astype(np.float32),
        'node_ids':     elem_dofs.astype(np.int64),
        'dof_coords':   dof_coords.astype(np.float32),
        'edge_to_dofs': edge_to_dofs,
    }


def _precompute_boundary_edges(vertices:     np.ndarray,
                                edges:        np.ndarray,       # (n_edges, 2)
                                edge_normals: np.ndarray,       # (n_edges, 2)
                                order:        int):
    """
    Precompute boundary cubature data for Neumann edges.

    Uses P1 interpolation along the edge (endpoint nodes only); this is
    consistent with how the Neumann contribution is assembled in the trainer.

    Returns a dict with keys:
        pts       (n_edges, n_eq, 2)
        weights   (n_edges, n_eq)
        phi       (n_edges, n_eq, 2)  – P1 values at the 2 endpoint nodes
        normals   (n_edges, 2)
        edge_ids  (n_edges, 2)
    """
    ref_t, ref_w = _edge_cubature_1d(order)
    n_eq    = len(ref_t)
    n_edges = len(edges)

    pts_all    = np.empty((n_edges, n_eq, 2), dtype=np.float64)
    weights_all = np.empty((n_edges, n_eq),   dtype=np.float64)
    phi_all    = np.empty((n_edges, n_eq, 2), dtype=np.float64)

    for e, (i0, i1) in enumerate(edges):
        x0, x1 = vertices[i0], vertices[i1]
        length  = np.linalg.norm(x1 - x0)
        pts_all[e]     = x0 + ref_t[:, None] * (x1 - x0)
        weights_all[e] = ref_w * length
        # P1 values: φ_i0 = 1-t, φ_i1 = t  (linear along the edge)
        phi_all[e, :, 0] = 1.0 - ref_t
        phi_all[e, :, 1] = ref_t

    return {
        'pts':      pts_all.astype(np.float32),
        'weights':  weights_all.astype(np.float32),
        'phi':      phi_all.astype(np.float32),
        'normals':  edge_normals.astype(np.float32),
        'edge_ids': edges.astype(np.int64),
    }


def _precompute_lm_boundary(vertices:          np.ndarray,
                             faces:             np.ndarray,
                             elem_dofs:         np.ndarray,
                             boundary_edges:    np.ndarray,
                             bc_value,
                             bc_component:      int,
                             cub_order:         int,
                             lag_order:         int,
                             lm_global_dof_ids: np.ndarray):
    """
    Precompute boundary cubature data for one Lagrange-multiplier (LM) BC.

    The LM space uses the same order-N Lagrange trace basis as the primal
    space, so ``phi`` in the returned dict plays the role of *both* the
    primal-trace and LM-basis matrices (they coincide when both spaces are
    order-N on the boundary).

    Returns
    -------
    dict with keys:
        pts            (n_bdy, n_eq, 2)    – physical quadrature points
        weights        (n_bdy, n_eq)       – Gauss weights × edge length
        phi            (n_bdy, n_eq, N+1)  – trace Lagrange basis
        global_dof_ids (n_bdy, N+1) int32  – global DOF indices (scatter → R)
        lm_local_ids   (n_bdy, N+1) int32  – LM index (scatter/gather lm_params)
        g_vals         (n_bdy, n_eq)       – prescribed BC values at quad pts
    """
    N = lag_order

    # Build edge → (face_idx, local_edge_idx) lookup
    edge_to_face: dict = {}
    local_edge_pairs = [(0, 1), (1, 2), (2, 0)]
    for f, face in enumerate(faces):
        for le, (la, lb) in enumerate(local_edge_pairs):
            key = (min(int(face[la]), int(face[lb])), max(int(face[la]), int(face[lb])))
            edge_to_face[key] = (f, le)

    ref_t, ref_w = _edge_cubature_1d(cub_order)
    n_eq      = len(ref_t)
    n_bdy     = len(boundary_edges)
    n_local_e = N + 1

    lm_g2l         = {int(g): i for i, g in enumerate(lm_global_dof_ids)}
    pts_all        = np.empty((n_bdy, n_eq, 2),        dtype=np.float64)
    weights_all    = np.empty((n_bdy, n_eq),            dtype=np.float64)
    phi_all        = np.empty((n_bdy, n_eq, n_local_e), dtype=np.float64)
    global_ids_all = np.empty((n_bdy, n_local_e),      dtype=np.int64)
    lm_loc_all     = np.empty((n_bdy, n_local_e),      dtype=np.int64)
    g_vals_all     = np.empty((n_bdy, n_eq),            dtype=np.float64)

    def _ref_edge_pts(le, t):
        """Map t ∈ [0,1] to reference-triangle coords along local edge le."""
        if le == 0:
            return np.stack([t, np.zeros_like(t)], axis=1)
        elif le == 1:
            return np.stack([1.0 - t, t], axis=1)
        else:
            return np.stack([np.zeros_like(t), 1.0 - t], axis=1)

    for e, (va, vb) in enumerate(boundary_edges):
        va, vb    = int(va), int(vb)
        key       = (min(va, vb), max(va, vb))
        f_idx, le = edge_to_face[key]
        face      = faces[f_idx]
        la, lb    = local_edge_pairs[le]
        fa_node   = int(face[la])
        # direction_matches: physical va==face[la], so t goes la→lb in ref
        direction_matches = (va == fa_node)

        xa, xb = vertices[va], vertices[vb]
        length = np.linalg.norm(xb - xa)
        pts_all[e]    = xa + ref_t[:, None] * (xb - xa)
        weights_all[e] = ref_w * length

        # Reference coords of the quadrature points on this edge
        t_ref        = ref_t if direction_matches else (1.0 - ref_t)
        ref_pts_edge = _ref_edge_pts(le, t_ref)                # (n_eq, 2)
        phi_full, _  = _lagrange_basis_and_grad(ref_pts_edge, N)  # (n_eq, n_local)
        local_ids    = _local_edge_dof_indices(le, N)           # N+1 in la→lb order

        phi_all[e]        = phi_full[:, local_ids]              # (n_eq, N+1)
        glob_ids          = np.array(elem_dofs[f_idx])[local_ids]
        global_ids_all[e] = glob_ids
        lm_loc_all[e]     = np.array([lm_g2l[int(g)] for g in glob_ids])

        # Prescribed BC value at quad points
        if callable(bc_value):
            for q in range(n_eq):
                val = bc_value(pts_all[e, q])
                g_vals_all[e, q] = (val[bc_component]
                                    if hasattr(val, '__len__') else float(val))
        else:
            g_vals_all[e, :] = float(bc_value)

    return {
        'pts':            pts_all.astype(np.float32),
        'weights':        weights_all.astype(np.float32),
        'phi':            phi_all.astype(np.float32),
        'global_dof_ids': global_ids_all.astype(np.int32),
        'lm_local_ids':   lm_loc_all.astype(np.int32),
        'g_vals':         g_vals_all.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# ProblemWeak
# ---------------------------------------------------------------------------

class ProblemWeak(BaseProblem):
    """
    Weak-form (Galerkin) problem on a mesh with order-N Lagrange test functions.

    Interface mirrors :class:`~pinns.problems.ProblemStrong`: construct with
    ``domain`` and ``output_names``, then register terms with
    :meth:`add_inner`, :meth:`add_dirichlet`, :meth:`add_neumann`, attach
    observables with :meth:`add_observable`, and set the reference solution
    with :meth:`add_solution`.

    Parameters
    ----------
    domain : DomainMesh
        Mesh domain.  Named boundary regions for :meth:`add_dirichlet` and
        :meth:`add_neumann` are registered on the domain beforehand via
        ``domain.add_boundary(select, name=...)``.
    output_names : list[str]
        Names for the network output components.
    cubature_order : int
        Polynomial exactness order for the cubature rules (1–5, default 3).
    lagrange_order : int
        Polynomial order of the Lagrange test-function space (default 1 → P1).
    basis : str
        Test function basis — currently only ``"lagrange"`` is supported.
    params : dict or None
        Fixed problem parameters passed as ``params["fixed"]``.
    solution : callable or None
        Reference solution for error tracking (can also be set via
        :meth:`add_solution`).

    Examples
    --------
    ::

        problem = ProblemWeak(domain, output_names=["u"], cubature_order=3)
        problem.add_inner(volume_fn, name="pde")
        problem.add_dirichlet(0.0, name="bottom", region="bottom")
        problem.add_solution(lambda xy: np.sin(np.pi * xy[:, 0]))
    """

    def __init__(
        self,
        domain,
        output_names,
        *,
        cubature_order: int = 3,
        lagrange_order: int = 1,
        basis: str = "lagrange",
        params: Optional[Dict[str, Any]] = None,
        solution: Optional[Callable] = None,
    ):
        from ..domain import DomainMesh

        if not isinstance(domain, DomainMesh):
            raise TypeError(
                "ProblemWeak requires a DomainMesh domain; "
                f"got {type(domain).__name__}."
            )

        super().__init__(domain, output_names)

        self.cubature_order     = cubature_order
        self.lagrange_order     = lagrange_order
        self.basis              = basis
        # Allow construction-time params via the 'params' alias → fixed_params
        if params:
            self.fixed_params.update(params)
        if solution is not None:
            self.solution = solution

        # ── runtime-filled ───────────────────────────────────────────────
        self.cubature_data:    Dict       = {}
        self.neumann_data:     List       = []
        self.boundary_fn_data: List       = []
        self.free_nodes:       np.ndarray = None
        self.dirichlet_nodes:  np.ndarray = None

        self._init_body()

    # ── Backward-compat properties ────────────────────────────────────────

    @property
    def volume_fn(self):
        """First (or only) volume integrand.  Access via :meth:`add_inner`."""
        return self.inner_terms[0].fn if self.inner_terms else None

    @volume_fn.setter
    def volume_fn(self, fn):
        if fn is None:
            return
        from .terms import TermInner as _TermInner
        if self.inner_terms:
            self.inner_terms[0].fn = fn
        else:
            self._terms.append(_TermInner(fn=fn, name='pde'))

    @property
    def _volume_name(self):
        """Name of the first (or only) inner volume term."""
        return self.inner_terms[0].name if self.inner_terms else 'pde'

    @_volume_name.setter
    def _volume_name(self, name):
        from .terms import TermInner as _TermInner
        if self.inner_terms:
            self.inner_terms[0].name = name
        else:
            self._terms.append(_TermInner(fn=None, name=name))

    # ---------------------------------------------------------------------- #
    #  Term registration (mirrors ProblemStrong)                             #
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    #  Internal initialisation body (formerly __post_init__)                 #
    # ---------------------------------------------------------------------- #

    def _init_body(self):
        if self.basis != "lagrange":
            raise ValueError(
                f"Only 'lagrange' basis is currently supported; "
                f"got '{self.basis}'."
            )
        if not (1 <= self.cubature_order <= 5):
            raise ValueError(
                f"cubature_order must be between 1 and 5; "
                f"got {self.cubature_order}."
            )
        if self.lagrange_order < 1:
            raise ValueError(
                f"lagrange_order must be ≥ 1; got {self.lagrange_order}."
            )

        verts = self.domain._vertices   # (n_verts, 2)  — corner vertices only
        faces = self.domain._faces      # (n_faces, 3)

        # ── Volume cubature (includes HO DOF generation) ────────────────
        self.cubature_data = _precompute_volume(
            verts, faces, self.cubature_order, self.lagrange_order
        )

        dof_coords   = self.cubature_data['dof_coords']   # (n_dofs, 2)
        edge_to_dofs = self.cubature_data['edge_to_dofs'] # canonical edge → dofs
        n_dofs       = len(dof_coords)

        # ── Classify Dirichlet vs free DOFs ─────────────────────────────
        #
        # Strategy:
        #   1. Vertex DOFs: match each bc.node_position to a vertex index.
        #   2. Edge DOFs:   if both endpoints of a mesh-edge are Dirichlet
        #                   (i.e., found in the sets above), mark all interior
        #                   edge DOFs as Dirichlet.
        #   3. Interior element DOFs: never Dirichlet for standard BCs.
        #
        # For the common case where the mesh provides bc.edges (vertex-index
        # pairs), we use that directly; otherwise we fall back to distance
        # matching of the supplied node_positions.

        # All nodes are free — Dirichlet BCs are enforced via soft loss or Lifting layer.
        all_dofs = np.arange(n_dofs, dtype=np.int64)
        self.dirichlet_nodes = np.array([], dtype=np.int64)
        free_mask = np.ones(n_dofs, dtype=bool)
        self.free_nodes = all_dofs

        # ── Split free nodes into inner and boundary ──────────────────────────
        # Boundary free nodes: any free node appearing on a non-Dirichlet BC edge.
        # All BCs now store a region key; edges are looked up from the domain.
        from .terms import TermDirichletBC as _TDB
        _boundary_node_set = set()
        for bc in self.boundary_conditions:
            if isinstance(bc, _TDB):
                continue
            _region = getattr(bc, 'region', None)
            if _region == 'all':
                _edges = self.domain._bnd_edges
            elif _region and _region in self.domain._boundary_regions:
                _edges = self.domain._boundary_regions[_region].get('edges')
            else:
                _edges = None
            if _edges is not None:
                for _i0, _i1 in _edges:
                    _boundary_node_set.add(int(_i0))
                    _boundary_node_set.add(int(_i1))
        _free_set = set(self.free_nodes.tolist())
        _boundary_free = sorted(_free_set & _boundary_node_set)
        _inner_free    = sorted(_free_set - _boundary_node_set)
        self.boundary_free_nodes = np.array(_boundary_free, dtype=np.int64)
        self.inner_free_nodes    = np.array(_inner_free,    dtype=np.int64)

        # Store free_mask in cubature_data for easy access
        self.cubature_data['free_mask'] = free_mask

        # ── Nodal support volumes + boundary edge lengths ─────────────────────
        # For each node j we build a normaliser that makes R̂_j = R_j / norm_j
        # independent of mesh size h:
        #
        #   Volume term:   ∫_Ω σ:∇φ_j dΩ  ~ σ·h²   → normalise by V_j  ~ h²
        #   Boundary term: ∫_Γ t·φ_j dS   ~ t·h    → normalise by L_j  ~ h
        #
        # Boundary nodes receive BOTH contributions, so norm_j = V_j + L_j
        # keeps both at O(1).  Interior nodes have L_j=0, so norm_j = V_j.
        _elem_areas  = self.cubature_data['weights'].sum(axis=1)   # (n_faces,)
        _node_ids_np = self.cubature_data['node_ids']              # (n_faces, n_local)
        _support_vol = np.zeros(n_dofs, dtype=np.float64)
        for _k in range(len(_elem_areas)):
            for _a in range(_node_ids_np.shape[1]):
                _support_vol[_node_ids_np[_k, _a]] += _elem_areas[_k]
        # L_j will be filled in after boundary_fn_data is built (below)
        self._support_vol_tmp = _support_vol   # hold temporarily
        from .terms import TermNeumannBC as _TermNeumannBC
        self.neumann_data = []
        for bc in self.boundary_conditions:
            if not isinstance(bc, _TermNeumannBC):
                continue
            # Look up edges and normals from domain._boundary_regions
            _region = getattr(bc, 'region', None)
            if _region == 'all':
                _bc_edges = self.domain._bnd_edges
            elif _region and _region in self.domain._boundary_regions:
                _bc_edges = self.domain._boundary_regions[_region].get('edges')
            else:
                _bc_edges = None
            if _bc_edges is None:
                continue
            _bc_normals = (self.domain._infer_edge_outward_normals(_bc_edges)
                           if self.domain._spatial_dims == 2 else None)
            data = _precompute_boundary_edges(
                verts, _bc_edges, _bc_normals, self.cubature_order
            )
            data['bc'] = bc
            self.neumann_data.append(data)

        # ── boundary_fn cubature (weak-form traction RHS terms) ──────────────
        # TermCustomBC (add_boundary) and TermNeumannBC-with-fn (add_neumann)
        # both contribute weak-form integrands assembled via boundary_fn_data.
        self.boundary_fn_data = []
        for bc in self.boundary_conditions:
            if bc.kind not in ('boundary', 'neumann', 'robin'):
                continue
            if getattr(bc, 'fn', None) is None:
                continue
            _fn = bc.fn
            _region = getattr(bc, 'region', None)
            if _region == 'all':
                _bc_edges = self.domain._bnd_edges
            elif _region and _region in self.domain._boundary_regions:
                _bc_edges = self.domain._boundary_regions[_region].get('edges')
            else:
                _bc_edges = None
            if _bc_edges is None:
                raise ValueError(
                    f"Weak BC '{bc.name}' has no edges registered on the domain; "
                    "call domain.add_boundary(select, name=...) before assembling."
                )
            edge_normals = self.domain._infer_edge_outward_normals(_bc_edges)
            data = _precompute_boundary_edges(
                verts, _bc_edges, edge_normals, self.cubature_order
            )
            data['fn']   = _fn
            data['name'] = bc.name
            self.boundary_fn_data.append(data)

        # ── Finalise node normaliser: norm_j = V_j + L_j ────────────────────────
        # L_j = sum of weak-BC edge lengths touching node j (1-D boundary support).
        _support_len = np.zeros(n_dofs, dtype=np.float64)
        for _bd in self.boundary_fn_data:
            _edge_lens = _bd['weights'].sum(axis=1)   # (n_edges,)  = edge lengths
            for _e, (_i0, _i1) in enumerate(_bd['edge_ids']):
                _support_len[int(_i0)] += _edge_lens[_e]
                _support_len[int(_i1)] += _edge_lens[_e]
        _node_norm = self._support_vol_tmp + _support_len
        _node_norm = np.where(_node_norm > 0, _node_norm, 1.0)
        self.node_norm = _node_norm.astype(np.float32)
        del self._support_vol_tmp   # no longer needed

    # ── Convenience properties ───────────────────────────────────────────

    @property
    def n_free_nodes(self) -> int:
        """Number of free (non-Dirichlet) DOFs = number of test functions."""
        return len(self.free_nodes)

    @property
    def n_dofs(self) -> int:
        """Total number of global DOFs."""
        return len(self.cubature_data['dof_coords'])

    # ------------------------------------------------------------------ #
    #  Boundary-condition builders                                       #
    # ------------------------------------------------------------------ #

    def add_neumann(
        self,
        value,
        name: str,
        region: str = 'all',
        outputs=None,
    ) -> 'ProblemWeak':
        """Add a **Neumann** BC: ``du/dn = value`` on a named boundary region.

        Mirrors :meth:`~pinns.problems.ProblemStrong.add_neumann`.

        The Neumann contribution is assembled as a weak-form RHS term::

            ∫_Γ g(x) · φ_j(x) ds  (subtracted from the Galerkin residual)

        where ``g = value``.

        Args:
            value: Prescribed normal-flux value.  Scalar or callable
                ``g(x) -> array`` or ``g(x, pars) -> array``.
            name: Unique label for this term.
            region: Named boundary region, or ``'all'`` (default).
            outputs: Which output(s) this term applies to.  ``None`` (default)
                is only allowed when ``n_outputs == 1``.

        Returns:
            ``self`` for method chaining.
        """
        from .terms import TermNeumannBC
        for out_idx, suffix in self._resolve_outputs(outputs):
            _v = value

            def _neumann_fn(x, u, params, phi, deriv, _val=_v):
                """Weak Neumann integrand: g(x) * φ."""
                if callable(_val):
                    import inspect as _inspect
                    g = (_val(x, params)
                         if len(_inspect.signature(_val).parameters) >= 2
                         else _val(x))
                else:
                    import jax.numpy as _jnp
                    g = _jnp.full(phi.shape, float(_val))
                return g * phi

            self._terms.append(
                TermNeumannBC(region=region, value=value,
                              component=out_idx, name=name + suffix,
                              fn=_neumann_fn)
            )
        return self

    def add_robin(
        self,
        alpha,
        beta,
        g,
        name: str,
        region: str = 'all',
        outputs=None,
    ) -> 'ProblemWeak':
        """Add a **Robin** BC: ``α·u + β·n·∇u = g`` on a named boundary region.

        Mirrors :meth:`~pinns.problems.ProblemStrong.add_robin`.

        The contribution is assembled as a weak-form boundary integrand::

            (α·u·φ + β·(n·∇u)·φ - g·φ)  integrated over Γ

        Args:
            alpha: Coefficient on ``u``.  Scalar or callable ``alpha(x, pars)``.
            beta: Coefficient on ``n·∇u``.  Scalar or callable.
            g: Right-hand side.  Scalar or callable ``g(x, pars)``.
            name: Unique label for this term.
            region: Named boundary region, or ``'all'`` (default).
            outputs: Which output(s) this term applies to.

        Returns:
            ``self`` for method chaining.
        """
        from .terms import TermRobinBC
        _domain = self.domain
        for out_idx, suffix in self._resolve_outputs(outputs):
            _comp  = out_idx or 0
            _alpha = alpha
            _beta  = beta
            _g     = g

            def _robin_fn(x, u, params, phi, deriv,
                          _d=_domain, _c=_comp,
                          _a=_alpha, _b=_beta, _gv=_g, _r=region):
                import jax.numpy as jnp
                import numpy as _np
                import inspect as _inspect

                def _eval(v):
                    if callable(v):
                        return (v(x, params)
                                if len(_inspect.signature(v).parameters) >= 2
                                else v(x))
                    return float(v)

                normals = _np.asarray(_d.get_boundary_normals(_np.asarray(x), _r))
                n_spatial = normals.shape[1]
                du_dn = sum(
                    normals[:, i] * deriv(u, x, _c, (i,))
                    for i in range(n_spatial)
                )
                residual = _eval(_a) * u[:, _c] + _eval(_b) * du_dn - _eval(_gv)
                return residual * phi

            self._terms.append(
                TermRobinBC(region=region, alpha=alpha, beta=beta, value=g,
                            component=_comp, name=name + suffix,
                            fn=_robin_fn)
            )
        return self

    def make_residual_fn(self, network):
        """Return ``fn(params) -> dict[str, jnp.ndarray]``.

        Assembles the per-free-node Galerkin residual for every registered inner
        term, normalised by the nodal measure (support volume + edge length).
        The trainer aggregates these as ``loss = mean(r**2)``.

        Parameters
        ----------
        network :
            JAX network supporting ``network.apply(params, x, params_dict)``.

        Returns
        -------
        residual_fn : callable
            ``residual_fn(params) -> dict[str, jnp.ndarray]``
        """
        import jax
        import jax.numpy as jnp
        import numpy as _np
        from .terms import TermDirichletBC as _TermDirichletBC
        from .terms import TermInitial as _TermInitial

        _weak_params_dict = self._build_params()
        _n_out = self.n_outputs

        if _n_out == 1:
            def _u_and_grad(params, xy):
                def u_single(z):
                    return network.apply(params, z[None], _weak_params_dict)[0, 0]
                return jax.value_and_grad(u_single)(xy)
        else:
            def _u_and_grad(params, xy):
                def u_vec(z):
                    return network.apply(params, z[None], _weak_params_dict)[0]
                u = u_vec(xy)
                jac = jax.jacobian(u_vec)(xy)
                return u, jac

        cd            = self.cubature_data
        pts_jax       = jnp.asarray(cd['pts'],      dtype=jnp.float32)
        weights_jax   = jnp.asarray(cd['weights'],  dtype=jnp.float32)
        phi_jax       = jnp.asarray(cd['phi'],      dtype=jnp.float32)
        grad_phi_jax  = jnp.asarray(cd['grad_phi'], dtype=jnp.float32)
        node_ids_jax  = jnp.asarray(cd['node_ids'], dtype=jnp.int32)
        node_norm_jax = jnp.asarray(self.node_norm, dtype=jnp.float32)
        dof_coords_jax = jnp.asarray(cd['dof_coords'], dtype=jnp.float32)

        # True free nodes: exclude Dirichlet-constrained DOFs from the test space
        _dirichlet_set: set = set()
        for _bc in self.boundary_conditions:
            if isinstance(_bc, _TermDirichletBC):
                _region = getattr(_bc, 'region', None)
                if _region and _region in self.domain._boundary_regions:
                    _ni = self.domain._boundary_regions[_region].get('node_indices')
                    if _ni is not None:
                        _dirichlet_set.update(int(i) for i in _ni)
                    _edges = self.domain._boundary_regions[_region].get('edges')
                    if _edges is not None:
                        for _i0, _i1 in _edges:
                            _dirichlet_set.add(int(_i0))
                            _dirichlet_set.add(int(_i1))
        _all_free  = set(self.free_nodes.tolist())
        _true_free = sorted(_all_free - _dirichlet_set)
        free_nodes_jax = jnp.asarray(
            _np.array(_true_free, dtype=_np.int64), dtype=jnp.int32)

        n_dofs   = self.n_dofs
        # ── Precompute IC node data (TermInitial soft penalty) ─────────────────
        _ic_data: list = []
        _t_min_ic = getattr(self.domain, '_t_min', None)
        for _ic_bc in self.boundary_conditions:
            if not isinstance(_ic_bc, _TermInitial):
                continue
            _ic_ni = _np.array(
                self.domain._boundary_regions[_ic_bc.region]['node_indices'],
                dtype=_np.int64)
            _ic_xy = dof_coords_jax[jnp.asarray(_ic_ni, dtype=jnp.int32)]  # (n_ic, n_spatial)
            if _t_min_ic is not None:
                _t_col = jnp.full((_ic_xy.shape[0], 1), float(_t_min_ic), dtype=jnp.float32)
                _ic_pts = jnp.concatenate([_ic_xy, _t_col], axis=-1)
            else:
                _ic_pts = _ic_xy
            _ic_data.append((_ic_bc, _ic_pts))

        # Precompute the list of soft-penalty BC terms (Dirichlet, Custom)
        # These are evaluated on sampled points from sample_data inside residual_fn.
        _soft_bcs = [
            bc for bc in self.boundary_conditions
            if bc.kind not in ('initial', 'periodic', 'inner')
        ]

        n_faces  = pts_jax.shape[0]
        n_qpts   = pts_jax.shape[1]
        n_local  = phi_jax.shape[2]
        from .terms import TermInner as _TermInner
        from .terms import TermPeriodicBC as _TermPeriodicBC
        _terms   = list(self._inner_terms) or [_TermInner(fn=self.volume_fn, name='pde')]
        _periodic_terms = [bc for bc in self.boundary_conditions
                           if isinstance(bc, _TermPeriodicBC)]
        params_dict = self._build_params()

        _bdata_jax = []
        for _bd in self.boundary_fn_data:
            _bdata_jax.append({
                'pts':      jnp.asarray(_bd['pts'],      dtype=jnp.float32),
                'weights':  jnp.asarray(_bd['weights'],  dtype=jnp.float32),
                'phi':      jnp.asarray(_bd['phi'],      dtype=jnp.float32),
                'edge_ids': jnp.asarray(_bd['edge_ids'], dtype=jnp.int32),
                'fn':       _bd['fn'],
            })

        _n_pts_dims = pts_jax.shape[-1]

        def residual_fn(params, sample_data=None):
            pts_flat = pts_jax.reshape(-1, _n_pts_dims)
            u_flat, grad_u_flat = jax.vmap(
                lambda xy: _u_and_grad(params, xy))(pts_flat)

            if u_flat.ndim == 1:
                y_flat = u_flat.reshape(-1, 1)
                def make_deriv(gu, gphi):
                    def deriv_fn(Y, X, component, order):
                        dim = order[0] if isinstance(order, (list, tuple)) else order
                        if Y.ndim == 1: return gphi[:, dim]
                        return gu[:, dim]
                    return deriv_fn
            else:
                y_flat = u_flat
                def make_deriv(jac, gphi):
                    def deriv_fn(Y, X, component, order):
                        dim = order[0] if isinstance(order, (list, tuple)) else order
                        if Y.ndim == 1: return gphi[:, dim]
                        return jac[:, component, dim]
                    return deriv_fn

            # Probe first term to detect multi-component output
            _gphi0   = grad_phi_jax[:, :, 0, :].reshape(-1, 2)
            R_sample = _terms[0].fn(
                pts_flat, y_flat, params_dict,
                phi_jax[:, :, 0].reshape(-1),
                make_deriv(grad_u_flat, _gphi0),
            )
            _multi  = isinstance(R_sample, (tuple, list))
            _n_comp = len(R_sample) if _multi else 1

            result = {}
            for _iterm, _term in enumerate(_terms):
                _tfn = _term.fn
                Rs = [jnp.zeros(n_dofs, dtype=jnp.float32) for _ in range(_n_comp)]

                for a in range(n_local):
                    phi_a       = phi_jax[:, :, a]
                    gphi_a_flat = grad_phi_jax[:, :, a, :].reshape(-1, 2)
                    integrand   = _tfn(
                        pts_flat, y_flat, params_dict,
                        phi_a.reshape(-1),
                        make_deriv(grad_u_flat, gphi_a_flat),
                    )
                    if _multi:
                        for k, ig in enumerate(integrand):
                            elem_int = jnp.einsum(
                                'fq,fq->f', weights_jax,
                                ig.reshape(n_faces, n_qpts))
                            Rs[k] = Rs[k].at[node_ids_jax[:, a]].add(elem_int)
                    else:
                        elem_int = jnp.einsum(
                            'fq,fq->f', weights_jax,
                            integrand.reshape(n_faces, n_qpts))
                        Rs[0] = Rs[0].at[node_ids_jax[:, a]].add(elem_int)

                # Subtract boundary traction / Robin RHS (only from the first inner term)
                if _iterm == 0:
                    for _bj in _bdata_jax:
                        _bp_ndims  = _bj['pts'].shape[-1]
                        _bpts_flat = _bj['pts'].reshape(-1, _bp_ndims)
                        _bw        = _bj['weights']
                        _bphi_mat  = _bj['phi']
                        _beid      = _bj['edge_ids']
                        _bfn       = _bj['fn']
                        _n_bedges  = _bj['pts'].shape[0]
                        _n_bqpts   = _bj['pts'].shape[1]

                        _bu_flat, _bgu_flat = jax.vmap(
                            lambda xy: _u_and_grad(params, xy))(_bpts_flat)
                        # Normalise shapes: u always (N, n_out), grad always (N, n_out, n_dim) or (N, n_dim)
                        _by_flat = (_bu_flat.reshape(-1, 1)
                                    if _bu_flat.ndim == 1 else _bu_flat)

                        def _mk_bderiv(bgu):
                            def _bderiv(Y, X, c, o):
                                dim = o[0] if isinstance(o, (list, tuple)) else o
                                return bgu[:, dim] if bgu.ndim == 2 else bgu[:, c, dim]
                            return _bderiv

                        _bderiv = _mk_bderiv(_bgu_flat)

                        for _p in range(2):
                            _bphi_p = _bphi_mat[:, :, _p]
                            _b_intg = _bfn(
                                _bpts_flat, _by_flat, params_dict,
                                _bphi_p.reshape(-1), _bderiv)
                            if isinstance(_b_intg, (tuple, list)):
                                for k, ig in enumerate(_b_intg):
                                    _belem = jnp.einsum(
                                        'eq,eq->e', _bw,
                                        ig.reshape(_n_bedges, _n_bqpts))
                                    Rs[k] = Rs[k].at[_beid[:, _p]].add(-_belem)
                            else:
                                _belem = jnp.einsum(
                                    'eq,eq->e', _bw,
                                    _b_intg.reshape(_n_bedges, _n_bqpts))
                                Rs[0] = Rs[0].at[_beid[:, _p]].add(-_belem)

                # Normalise and slice to free nodes (components concatenated)
                _nodes = (
                    jnp.asarray(sample_data['free_nodes'], dtype=jnp.int32)
                    if sample_data is not None and 'free_nodes' in sample_data
                    else free_nodes_jax
                )
                norm = node_norm_jax[_nodes]
                result[_term.name] = jnp.concatenate([
                    R[_nodes] / norm for R in Rs
                ])

            # ── IC soft penalties (TermInitial) ──────────────────────────────
            for _ic_bc_r, _ic_pts_r in _ic_data:
                _ic_u_r, _ = jax.vmap(
                    lambda xy: _u_and_grad(params, xy))(_ic_pts_r)
                _ic_y_r = _ic_u_r if _ic_u_r.ndim == 2 else _ic_u_r.reshape(-1, 1)
                result[_ic_bc_r.name] = _ic_bc_r.fn(_ic_pts_r, _ic_y_r, params_dict)

            # ── Periodic BC soft penalties ────────────────────────────────────
            if sample_data:
                for _pbc_t in _periodic_terms:
                    if _pbc_t.name not in sample_data:
                        continue
                    _pbc_pts = jnp.asarray(sample_data[_pbc_t.name], dtype=jnp.float32)
                    _pbc_half = _pbc_pts.shape[0] // 2
                    _pbc_xa   = _pbc_pts[:_pbc_half]
                    _pbc_xb   = _pbc_pts[_pbc_half:]
                    _pbc_ua, _ = jax.vmap(lambda xy: _u_and_grad(params, xy))(_pbc_xa)
                    _pbc_ub, _ = jax.vmap(lambda xy: _u_and_grad(params, xy))(_pbc_xb)
                    _pbc_ya = _pbc_ua if _pbc_ua.ndim == 2 else _pbc_ua.reshape(-1, 1)
                    _pbc_yb = _pbc_ub if _pbc_ub.ndim == 2 else _pbc_ub.reshape(-1, 1)
                    _pbc_c  = _pbc_t.component
                    if _pbc_c is not None:
                        result[_pbc_t.name] = _pbc_ya[:, _pbc_c] - _pbc_yb[:, _pbc_c]
                    else:
                        result[_pbc_t.name] = (_pbc_ya - _pbc_yb).reshape(-1)

            # ── Soft Dirichlet / Custom BC penalties ─────────────────────────
            # These are the sampled-point penalties whose gradients must be
            # included in the JIT training step (not just the logging path).
            if sample_data is not None:
                for _sbc in _soft_bcs:
                    if _sbc.name not in sample_data:
                        continue
                    _sbc_pts = jnp.asarray(sample_data[_sbc.name], dtype=jnp.float32)
                    _sbc_u, _sbc_grad = jax.vmap(
                        lambda xy: _u_and_grad(params, xy))(_sbc_pts)
                    _sbc_y = _sbc_u if _sbc_u.ndim == 2 else _sbc_u.reshape(-1, 1)

                    if _sbc.kind == 'dirichlet':
                        col = _sbc.component
                        if callable(_sbc.value):
                            # Assume the value fn accepts JAX arrays
                            _t = jnp.asarray(_sbc.value(_sbc_pts), dtype=jnp.float32)
                        else:
                            _t = jnp.full(_sbc_y.shape[0], float(_sbc.value),
                                          dtype=jnp.float32)
                        result[_sbc.name] = _sbc_y[:, col] - _t

                    elif _sbc.kind == 'boundary':  # TermCustomBC
                        _captured_grad = _sbc_grad
                        def _cbc_deriv(Y, X, comp, order,
                                       _g=_captured_grad):
                            dim = order[0] if isinstance(order, (list, tuple)) else order
                            return (_g[:, dim] if _g.ndim == 2
                                    else _g[:, comp, dim])
                        raw = _sbc.fn(_sbc_pts, _sbc_y, params_dict, _cbc_deriv)
                        result[_sbc.name] = (
                            raw if not isinstance(raw, (list, tuple))
                            else jnp.concatenate([r.reshape(-1) for r in raw])
                        )

            return result

        return residual_fn

    # ------------------------------------------------------------------
    # Sampling interface
    # ------------------------------------------------------------------

    def sample(self, train_samples=None, rng=None):
        """Return a dict with periodic BC collocation points, or ``None``.

        For ProblemWeak the volume/Neumann cubature data are fixed at
        construction time.  Periodic BC points are the only data that need
        to be (re-)sampled per training step.

        Returns:
            ``dict`` with one entry per :class:`~pinns.terms.TermPeriodicBC`
            (key = term name, value = stacked ``(2N, n_dims)`` array), or
            ``None`` when no periodic terms are registered.
        """
        import numpy as _np
        from .terms import TermPeriodicBC as _TermPeriodicBC
        if rng is None:
            rng = _np.random.default_rng()
        if train_samples is None:
            train_samples = {}
        data = {}
        for bc in self.boundary_conditions:
            if not isinstance(bc, _TermPeriodicBC):
                continue
            n = train_samples.get(bc.name, getattr(bc, 'n_pairs', None) or 200)
            data[bc.name] = self.domain.sample_boundary(n, region=bc.region, rng=rng)
        return data or None

    def assemble(self, term, r, cubature_data=None):
        """Return the normalised per-free-DOF Galerkin residual.

        This is the post-processing step that turns the raw per-DOF scatter-sum
        ``r`` (produced by the Galerkin assembly loop) into a loss-ready vector
        of length ``n_free_nodes``.  The trainer then applies
        ``loss = mean(w * R**2)`` uniformly across problem types.

        Steps
        -----
        1. Normalise: ``R_norm = r / node_norm``  — makes all entries O(1)
           regardless of mesh size (``node_norm[j] = support_vol_j + edge_len_j``).
        2. Select: return ``R_norm[free_nodes]``,  dropping Dirichlet DOFs
           from the test space.

        Args:
            term: The term whose residual was assembled (not used here; present
                  for API symmetry with
                  :class:`~pinns.problems.ProblemStrong`).
            r: Per-DOF accumulated residual vector, shape ``(n_dofs,)``.
               Typically produced by the Galerkin scatter-add loop inside
               :meth:`make_loss_fn`.
            cubature_data: Ignored; present for API symmetry.

        Returns:
            ``R_norm[free_nodes]``, shape ``(n_free_nodes,)``.
        """
        import jax.numpy as jnp
        node_norm = jnp.asarray(self.node_norm, dtype=jnp.float32)
        free      = jnp.asarray(self.free_nodes, dtype=jnp.int32)
        return (r / node_norm)[free]

    # ---------------------------------------------------------------------- #

    def __repr__(self):
        N = self.lagrange_order
        n_local = (N + 1) * (N + 2) // 2
        return (
            f"ProblemWeak("
            f"n_verts={len(self.domain._vertices)}, "
            f"n_dofs={self.n_dofs}, "
            f"n_faces={len(self.domain._faces)}, "
            f"n_free={self.n_free_nodes}, "
            f"n_dirichlet={len(self.dirichlet_nodes)}, "
            f"n_neumann_bcs={len(self.neumann_data)}, "
            f"cubature_order={self.cubature_order}, "
            f"lagrange_order={N} (P{N}, {n_local} dofs/elem), "
            f"basis='{self.basis}')"
        )
