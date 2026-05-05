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
    ``bc``                                – the ``MeshNodeBC`` object
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Union

import numpy as np


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
        # 4-point Dunavant, exact degree 3
        a1, b1 = 1/3, 1/3
        a2, b2 = 0.6, 0.2
        pts = np.array([[a1, b1],
                        [a2, b2],
                        [b2, a2],
                        [b2, b2]])
        w   = np.array([-9/32, 25/96, 25/96, 25/96])
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

@dataclass
class ProblemWeak:
    """
    Weak-form (Galerkin) problem on a mesh with order-N Lagrange test functions.

    Parameters
    ----------
    domain : DomainMesh
        Mesh domain.  Boundary conditions are attached via
        ``domain.add_dirichlet(...)`` / ``domain.add_neumann(...)``.
    volume_fn : callable
        Weak-form volume integrand.  Mirrors the strong-form
        ``pde_fn(x, y, params, derivative=None)`` with two extra trailing
        arguments for the test function::

            volume_fn(x, y, params, phi, derivative=None) -> (n_pts,)

        Arguments (all JAX arrays):
          - ``x``          (n_pts, 2)       – quadrature point positions
          - ``y``          (n_pts, n_out)   – network output (same layout as strong form)
          - ``params``     dict             – ``{fixed, infer, internal}`` (same as strong form)
          - ``phi``        (n_pts,)         – test function :math:`\varphi_j` values
          - ``grad_phi``   (n_pts, 2)       – :math:`\nabla\varphi_j` in physical coords
          - ``derivative`` callable or None – ``derivative(y, x, comp, order)`` (same API as
            strong form; provided by the assembler)
        Returns: ``(n_pts,)`` per-quadrature-point integrand values.
    boundary_fn : dict[str, callable] or None
        Weak-form Neumann (traction) boundary integrands, one per boundary
        name.  Each callable is evaluated at the boundary quadrature points
        and its result is **subtracted** from the corresponding residual
        vector(s) as the RHS traction integral::

            boundary_fn = {
                "right": lambda x, y, params, phi, derivative: (t1*phi, t2*phi)
            }

        Signature: ``f(x, y, params, phi, derivative) -> array or tuple``

          - ``x``          (n_pts, 2)         – boundary quadrature coords
          - ``y``          (n_pts, n_out)     – network output at those pts
          - ``params``     dict               – same as ``volume_fn``
          - ``phi``        (n_pts,)           – test-function values
          - ``derivative`` callable           – same derivative API

        The return must match the number of components returned by
        ``volume_fn`` (scalar or tuple of length n_comp).  Each returned
        array is ``∫ f_k φ_j ds`` and is subtracted from ``R_k``.  The key
        must match the name used in ``domain.add_bc``.
    params : dict
        Fixed problem parameters passed as ``params["fixed"]``.
    input_names : list[str]
        Names for input dimensions.
    output_names : list[str]
        Names for output components.
    output_range : tuple or list[tuple] or None
        Per-output unnormalization range.
    cubature_order : int
        Polynomial exactness order for the cubature rules (1–5, default 3).
        For accurate weak-form integration with order-N test functions use at
        least ``cubature_order ≥ 2*lagrange_order``.
    lagrange_order : int
        Polynomial order of the Lagrange test-function space (default 1 → P1).
        N=2 gives P2 (quadratic), N=3 gives P3 (cubic), etc.
    basis : str
        Test function basis — currently only ``"lagrange"`` is supported.
    solution : callable or None
        Reference solution for error tracking.

    Attributes (set during ``__post_init__``)
    -----------------------------------------
    cubature_data : dict
        Precomputed volume cubature arrays (see module docstring).
    neumann_data : list[dict]
        Precomputed edge cubature arrays for each Neumann BC.
    free_nodes : np.ndarray  (n_free,)
        Global DOF indices not constrained by Dirichlet conditions.
    dirichlet_nodes : np.ndarray  (n_dir,)
        Global DOF indices constrained by Dirichlet conditions.
    """

    domain: Any                                           # DomainMesh
    volume_fn: Callable
    boundary_fn: Optional[Union[Callable, Dict[str, Callable]]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    output_range: Optional[Union[tuple, List[Optional[tuple]]]] = None
    cubature_order: int = 3
    lagrange_order: int = 1
    basis: str = "lagrange"
    solution: Optional[Callable] = None
    lagrange_multipliers: List[str] = field(default_factory=list)
    obs_fn: Optional[Callable] = field(default=None)
    obs_names: Optional[List[str]] = field(default=None)
    obs_spatial: Optional[List[str]] = field(default=None)
    n_time_points: Optional[int] = None
    n_time_steps: Optional[int] = None
    hard_constraints: Union[bool, List[str]] = True
    ic_transition: Union[str, 'Callable'] = "tanh"

    # ── filled by __post_init__ ──────────────────────────────────────────
    cubature_data:    Dict       = field(init=False, default_factory=dict)
    neumann_data:     List       = field(init=False, default_factory=list)
    boundary_fn_data: List       = field(init=False, default_factory=list)
    free_nodes:       np.ndarray = field(init=False, default=None)
    dirichlet_nodes:  np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        from .domain import DomainMesh, DomainMeshContinuous, DomainMeshDiscrete
        from .boundary import MeshNodeBC

        if not isinstance(self.domain, DomainMesh):
            raise TypeError(
                "ProblemWeak requires a DomainMesh domain; "
                f"got {type(self.domain).__name__}."
            )

        # ── Auto-populate from typed subclasses ──────────────────────────────
        # DomainMeshDiscrete: read n_steps → n_time_steps
        if isinstance(self.domain, DomainMeshDiscrete) and self.n_time_steps is None:
            self.n_time_steps = self.domain.n_steps
        # DomainMeshContinuous: read n_time_points from domain if not set
        if isinstance(self.domain, DomainMeshContinuous) and self.n_time_points is None:
            self.n_time_points = self.domain.n_time_points
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

        self.n_dims    = self.domain.n_dims
        self.n_outputs = len(self.output_names)

        if not self.input_names:
            raise ValueError("input_names is required.")
        if len(self.input_names) != self.n_dims:
            raise ValueError(
                f"input_names has {len(self.input_names)} elements "
                f"but domain has {self.n_dims} dimensions."
            )
        if not self.output_names:
            raise ValueError("output_names is required.")

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

        dirichlet_vertex_set: set = set()

        # Pass 1 — vertex DOFs
        # Use hard_spatial_bc_names: groups collectively covering [t_min, t_max].
        # IC BCs (hard via t-factorization, not FEM DOF removal) are excluded.
        _hard_names_d = self.hard_spatial_bc_names
        for bc in self.domain.boundary_conditions:
            if isinstance(bc, MeshNodeBC) and bc.bc_type == "dirichlet":
                if getattr(bc, 'name', None) not in _hard_names_d:
                    continue   # not a hard-constrained spatial BC — keep as soft
                if bc.node_indices is not None:
                    for ni in bc.node_indices:
                        dirichlet_vertex_set.add(int(ni))
                elif bc.edges is not None:
                    for i0, i1 in bc.edges:
                        dirichlet_vertex_set.add(int(i0))
                        dirichlet_vertex_set.add(int(i1))
                else:
                    for xy in bc.node_positions:
                        dists = np.linalg.norm(verts - xy, axis=1)
                        dirichlet_vertex_set.add(int(np.argmin(dists)))

        dirichlet_set = set(dirichlet_vertex_set)

        # Pass 2 — edge interior DOFs for strong Dirichlet (N ≥ 2)
        if self.lagrange_order >= 2:
            for bc in self.domain.boundary_conditions:
                if isinstance(bc, MeshNodeBC) and bc.bc_type == "dirichlet":
                    if getattr(bc, 'name', None) not in _hard_names_d:
                        continue   # not a hard-constrained spatial BC — soft only
                    if bc.node_indices is not None:
                        for ni in bc.node_indices:
                            # find edge DOFs adjacent to this vertex
                            for key, dofs in edge_to_dofs.items():
                                if int(ni) in key:
                                    for idx in dofs:
                                        dirichlet_set.add(idx)
                    elif bc.edges is not None:
                        for i0, i1 in bc.edges:
                            key = (min(int(i0), int(i1)), max(int(i0), int(i1)))
                            if key in edge_to_dofs:
                                for idx in edge_to_dofs[key]:
                                    dirichlet_set.add(idx)
                    else:
                        for dof_idx in range(len(verts), n_dofs):
                            pos = dof_coords[dof_idx]
                            dists = np.linalg.norm(verts - pos, axis=1)
                            nn_v = int(np.argmin(dists))
                            if nn_v in dirichlet_vertex_set:
                                dirichlet_set.add(dof_idx)

        all_dofs = np.arange(n_dofs, dtype=np.int64)
        self.dirichlet_nodes = np.array(sorted(dirichlet_set), dtype=np.int64)
        free_mask = np.ones(n_dofs, dtype=bool)
        free_mask[self.dirichlet_nodes] = False
        self.free_nodes = all_dofs[free_mask]

        # ── Split free nodes into inner and boundary ──────────────────────────
        # Boundary free nodes: any free node appearing on a non-Dirichlet BC edge
        # (custom add_bc edges for top/bottom/right).  Inner nodes are the rest.
        _boundary_node_set = set()
        for bc in self.domain.boundary_conditions:
            if getattr(bc, 'bc_type', 'custom') != 'dirichlet':
                _edges = getattr(bc, 'edges', None)
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

        # ── Hard-constraint BC mask ───────────────────────────────────────────
        # Build nodal bc_mask / bc_values from 'all'-time Dirichlet BCs.
        # Exposed via self.output_transform as a callable for use as
        # output_transform in any network (FNN, AlphaPINN, …).
        n_out = len(self.output_names)
        _bc_mask   = np.zeros((len(verts), n_out), dtype=np.float32)
        _bc_values = np.zeros((len(verts), n_out), dtype=np.float32)
        _hard_spatial = self.hard_spatial_bc_names
        _hard_ic      = self.hard_ic_names
        _ic_mask   = np.zeros((len(verts), n_out), dtype=np.float32)
        _ic_values = np.zeros((len(verts), n_out), dtype=np.float32)
        if _hard_spatial or _hard_ic:
            for bc in self.domain.boundary_conditions:
                if not (isinstance(bc, MeshNodeBC) and bc.bc_type == 'dirichlet'):
                    continue
                bc_name = getattr(bc, 'name', None)
                comp = getattr(bc, 'component', 0)
                if comp >= n_out:
                    continue
                if bc.node_indices is not None:
                    node_idx = bc.node_indices
                elif bc.edges is not None:
                    node_idx = np.unique(bc.edges)
                else:
                    continue
                node_pos = verts[node_idx]
                vals = bc.get_value(node_pos)
                if bc_name in _hard_spatial:
                    _bc_mask[node_idx, comp]   = 1.0
                    _bc_values[node_idx, comp] = vals
                elif bc_name in _hard_ic:
                    _ic_mask[node_idx, comp]   = 1.0
                    _ic_values[node_idx, comp] = vals
        self._bc_mask_np   = _bc_mask
        self._bc_values_np = _bc_values
        self._ic_mask_np   = _ic_mask
        self._ic_values_np = _ic_values

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
        self.neumann_data = []
        for bc in self.domain.boundary_conditions:
            if isinstance(bc, MeshNodeBC) and bc.bc_type == "neumann" and bc.edges is not None:
                data = _precompute_boundary_edges(
                    verts, bc.edges, bc.edge_normals, self.cubature_order
                )
                data['bc'] = bc
                self.neumann_data.append(data)

        # ── boundary_fn cubature (weak-form traction RHS terms) ──────────────
        # Sources:
        #   1. Explicit boundary_fn dict passed to ProblemWeak (legacy / override)
        #   2. Any domain BC with is_weak=True (auto-detected via phi in signature)
        self.boundary_fn_data = []
        _seen_weak_fns: set = set()   # deduplicate by function identity

        # Legacy boundary_fn argument
        if self.boundary_fn is not None:
            import warnings
            warnings.warn(
                "The 'boundary_fn' argument to ProblemWeak is deprecated. "
                "Define weak BCs directly via domain.add_bc() using a function "
                "that accepts 'phi' in its signature.",
                DeprecationWarning,
                stacklevel=2,
            )
            bfn_dict = (
                self.boundary_fn if isinstance(self.boundary_fn, dict)
                else {'__default__': self.boundary_fn}
            )
            # Build name → edges lookup from domain BCs (any type with .edges)
            name_to_edges: dict = {}
            for bc in self.domain.boundary_conditions:
                if getattr(bc, 'edges', None) is not None and bc.name is not None:
                    name_to_edges[bc.name] = bc.edges
            for bc_name, fn in bfn_dict.items():
                if bc_name not in name_to_edges:
                    raise ValueError(
                        f"boundary_fn key '{bc_name}' does not match any BC name "
                        f"in the domain.  Available: {list(name_to_edges)}"
                    )
                edges = name_to_edges[bc_name]
                edge_normals = self.domain._infer_edge_outward_normals(edges)
                data = _precompute_boundary_edges(
                    verts, edges, edge_normals, self.cubature_order
                )
                data['fn']   = fn
                data['name'] = bc_name
                self.boundary_fn_data.append(data)
                _seen_weak_fns.add(id(fn))

        # Auto-detect weak BCs from domain
        # Group by function identity so list-named BCs (which share the same
        # weak_fn object) are stored as a single entry with all names as a list.
        _fn_id_to_names: dict = {}
        _fn_id_to_bc:    dict = {}
        for bc in self.domain.boundary_conditions:
            if not getattr(bc, 'is_weak', False):
                continue
            fn = bc.weak_fn
            if fn is None or id(fn) in _seen_weak_fns:
                continue
            fid = id(fn)
            if fid not in _fn_id_to_names:
                _fn_id_to_names[fid] = []
                _fn_id_to_bc[fid]    = bc
            _fn_id_to_names[fid].append(bc.name)

        for fid, bc in _fn_id_to_bc.items():
            _seen_weak_fns.add(fid)
            if getattr(bc, 'edges', None) is None:
                raise ValueError(
                    f"Weak BC '{bc.name}' has no edges; edge information is required "
                    "for boundary cubature integration."
                )
            edge_normals = self.domain._infer_edge_outward_normals(bc.edges)
            data = _precompute_boundary_edges(
                verts, bc.edges, edge_normals, self.cubature_order
            )
            names = _fn_id_to_names[fid]        # list of per-component weight keys
            data['fn']   = bc.weak_fn
            data['name'] = names if len(names) > 1 else names[0]
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

        # ── Space-time extension: tile spatial cubature with time samples ─────
        # When domain has a t_interval the cubature points are extended from
        # (F, Q, 2) to (n_t*F, Q, n_dims) by tiling faces for each sampled
        # time value.  test functions (φ, ∇φ) are purely spatial and simply
        # repeated.  Weights are scaled by Δt so the einsum integrates over
        # [t_min, t_max] correctly.
        _t_min = getattr(self.domain, '_t_min', None)
        _t_max = getattr(self.domain, '_t_max', None)
        self._t_min = _t_min
        self._t_max = _t_max
        _t_sampling = getattr(self.domain, '_t_sampling_method', 'midpoint')
        self._t_sampling_method = _t_sampling
        # "midpoint" → fixed tiling at construction; anything else → dynamic
        # per-epoch random/quasi-random sampling inside make_loss_fn.
        _random_time_sampling = (_t_sampling != 'midpoint')
        self._random_time_sampling = _random_time_sampling
        if _t_min is not None and _t_max is not None:
            n_t = self.n_time_points if (self.n_time_points is not None) else 10
            self._n_t = n_t
            if _random_time_sampling:
                # Keep spatial-only cubature; dynamic tiling done inside make_loss_fn
                pass
            else:
                _dt = (_t_max - _t_min) / n_t
                # Mid-point rule: t_i = t_min + (i + 0.5)*dt
                t_vals = _t_min + (np.arange(n_t) + 0.5) * _dt  # (n_t,)
                cd = self.cubature_data
                F, Q, _ = cd['pts'].shape
                # pts: (F, Q, 2) → (n_t, F, Q, 3) → (n_t*F, Q, 3)
                pts_xy   = cd['pts']                                    # (F, Q, 2)
                t_col    = np.broadcast_to(
                    t_vals[:, None, None, None], (n_t, F, Q, 1)
                ).copy()                                                 # (n_t, F, Q, 1)
                pts_xy4d = np.broadcast_to(
                    pts_xy[None], (n_t, F, Q, 2)
                ).copy()                                                 # (n_t, F, Q, 2)
                pts_st   = np.concatenate([pts_xy4d, t_col], axis=-1)   # (n_t, F, Q, 3)
                pts_st   = pts_st.reshape(n_t * F, Q, 3).astype(np.float32)
                # weights scaled by dt (time integration weight)
                weights_st  = np.tile(cd['weights'],   (n_t, 1)) * _dt  # (n_t*F, Q)
                phi_st      = np.tile(cd['phi'],       (n_t, 1, 1))     # (n_t*F, Q, L)
                gphi_st     = np.tile(cd['grad_phi'],  (n_t, 1, 1, 1))  # (n_t*F, Q, L, 2)
                node_ids_st = np.tile(cd['node_ids'],  (n_t, 1))        # (n_t*F, L)
                self.cubature_data = {
                    'pts':       pts_st,
                    'weights':   weights_st.astype(np.float32),
                    'phi':       phi_st.astype(np.float32),
                    'grad_phi':  gphi_st.astype(np.float32),
                    'node_ids':  node_ids_st,
                    'dof_coords':   cd['dof_coords'],
                    'edge_to_dofs': cd['edge_to_dofs'],
                    'free_mask':    cd['free_mask'],
                }

        # ── output_range ────────────────────────────────────────────────
        if self.output_range is not None:
            if (isinstance(self.output_range, tuple)
                    and len(self.output_range) == 2
                    and not isinstance(self.output_range[0], (list, tuple))):
                self.output_range = [self.output_range] * self.n_outputs

        # ── BC coverage check ────────────────────────────────────────────
        self._check_bc_coverage()

    def _check_bc_coverage(self):
        """Warn if boundary nodes or interior nodes appear uncovered.

        For each output component this verifies:

        1. **Spatial boundary nodes** are in at least one BC
           (Dirichlet or Neumann) that has full or partial time coverage.
        2. **Interior nodes** are covered at ``t = t_min`` by at least one BC
           (typically the initial condition).

        Issues a ``UserWarning`` for each uncovered component rather than
        raising an exception, because partial coverage can be intentional.
        """
        import warnings
        from .boundary import MeshNodeBC

        has_time = self.domain._t_min is not None
        n_outputs = len(self.output_names)
        bnd_mask = self.domain.boundary_node_mask      # (n_verts,)
        int_mask = self.domain.interior_node_mask      # (n_verts,)
        bnd_node_ids = set(np.where(bnd_mask)[0].tolist())
        int_node_ids = set(np.where(int_mask)[0].tolist())

        for comp in range(n_outputs):
            comp_name = self.output_names[comp]

            bnd_covered: set = set()
            ic_covered:  set = set()

            for bc in self.domain.boundary_conditions:
                if not isinstance(bc, MeshNodeBC):
                    continue
                if bc.component != comp:
                    continue
                if bc.edges is None:
                    node_ids = set()
                else:
                    node_ids = set(bc.edges.ravel().tolist())
                if bc.node_indices is not None:
                    node_ids = node_ids | set(bc.node_indices.tolist())

                if node_ids & bnd_node_ids:
                    bnd_covered |= (node_ids & bnd_node_ids)

                if has_time:
                    tw = getattr(bc, 'time_window', None)
                    t_min_dom = self.domain._t_min
                    if tw is not None:
                        pts_tw = [float(v) for v in tw]
                        if abs(min(pts_tw) - t_min_dom) < 1e-10:
                            ic_covered |= node_ids
                    else:
                        ic_covered |= node_ids

            uncovered_bnd = bnd_node_ids - bnd_covered
            if uncovered_bnd:
                warnings.warn(
                    f"ProblemWeak: component '{comp_name}' has "
                    f"{len(uncovered_bnd)} uncovered spatial boundary node(s). "
                    "Add a Dirichlet or Neumann BC that selects these nodes.",
                    UserWarning, stacklevel=3,
                )

            if has_time:
                uncovered_ic = (bnd_node_ids | int_node_ids) - ic_covered
                if uncovered_ic:
                    warnings.warn(
                        f"ProblemWeak: component '{comp_name}' has "
                        f"{len(uncovered_ic)} node(s) not covered at "
                        f"t = {self.domain._t_min} (no initial condition). "
                        "Call add_initial_condition() or add a Dirichlet BC "
                        "whose time_window includes t_min.",
                        UserWarning, stacklevel=3,
                    )

    # ── Convenience properties ───────────────────────────────────────────

    @property
    def n_free_nodes(self) -> int:
        """Number of free (non-Dirichlet) DOFs = number of test functions."""
        return len(self.free_nodes)

    @staticmethod
    def _make_ic_transition_fn(ic_transition):
        """
        Return a JAX-compatible callable ``f(t_shifted) -> factor`` where
        ``t_shifted = t - t_min >= 0``.

        Built-in options
        ----------------
        ``"tanh"`` *(default)*
            ``jnp.tanh(t_shifted)`` — smooth ramp from 0 at ``t_min``,
            quickly saturates.  Gradient is non-zero everywhere; preferred
            for training stability.
        ``"linear"``
            ``t_shifted`` — plain linear envelope.  Equivalent to the
            classic  ``u = u_IC + (t-t_min) * NN`` factorization.
        Callable
            Any differentiable function ``f(t_shifted: jnp.ndarray) -> jnp.ndarray``
            of the same shape.  Must satisfy ``f(0) == 0`` for the IC to be
            exactly satisfied at ``t = t_min``.

        Example::

            # Quadratic ramp
            problem = ProblemWeak(
                ...,
                ic_transition=lambda t: t**2,
            )
        """
        if callable(ic_transition):
            return ic_transition
        if ic_transition == "tanh":
            import jax.numpy as _jnp
            return lambda t: _jnp.tanh(t)
        if ic_transition == "linear":
            return lambda t: t
        raise ValueError(
            f"ic_transition must be 'tanh', 'linear', or a callable; "
            f"got {ic_transition!r}."
        )

    @staticmethod
    def _bc_intervals_cover(windows, t0, t1, tol=1e-10) -> bool:
        """True when the union of *windows* (list of time_window lists) covers [t0, t1]."""
        if not windows:
            return False
        ivs = []
        for tw in windows:
            pts = sorted(float(v) for v in tw)
            lo, hi = pts[0], pts[-1]
            ivs.append((lo, hi))
        ivs.sort()
        cover = t0
        for lo, hi in ivs:
            if lo > cover + tol:
                return False
            cover = max(cover, hi)
        return cover >= t1 - tol

    @property
    def hard_spatial_bc_names(self) -> set:
        """Names of **spatial** Dirichlet BCs that are hard-constrained.

        These are full-time-coverage BCs (individually or collectively
        covering ``[t_min, t_max]``) that are enforced via the network's
        output mask-and-clamp transform.  They do NOT include IC BCs
        (see :attr:`hard_ic_names`).
        """
        hc = self.hard_constraints
        if hc is False or hc == []:
            return set()

        from .boundary import MeshNodeBC
        from collections import defaultdict

        t_min = getattr(self.domain, '_t_min', None)
        t_max = getattr(self.domain, '_t_max', None)

        # Group by (node-fingerprint, component), but only non-IC BCs
        groups: dict = defaultdict(list)
        for bc in self.domain.boundary_conditions:
            if not isinstance(bc, MeshNodeBC) or bc.bc_type != 'dirichlet':
                continue
            if getattr(bc, 'name', None) is None:
                continue
            # Skip point-time BCs (IC candidates)
            tw = bc.time_window
            if tw is not None:
                pts = [float(v) for v in tw]
                if len(pts) <= 2 and abs(pts[0] - pts[-1]) < 1e-10:
                    continue   # point BC — belongs to IC category
            ni = bc.node_indices
            if ni is None:
                continue
            key = (tuple(sorted(ni.tolist())), bc.component)
            groups[key].append(bc)

        eligible: set = set()
        for key, bcs in groups.items():
            for bc in bcs:
                if bc.is_full_time_coverage():
                    eligible.add(bc.name)
                    continue
                if t_min is None or t_max is None:
                    continue
                windows = [b.time_window for b in bcs if b.time_window is not None]
                if self._bc_intervals_cover(windows, t_min, t_max):
                    eligible.add(bc.name)

        if hc is True:
            return eligible
        return eligible & set(hc)

    @property
    def hard_ic_names(self) -> set:
        """Names of **initial-condition** Dirichlet BCs that are hard-constrained.

        These are point-time BCs at ``t = t_min`` enforced via the
        ``u = u_IC(x) + (t - t_min) \\cdot \\text{NN}(x, t)`` factorization
        in the network's output transform.  They are automatically detected
        when ``hard_constraints=True`` or when their name is listed in
        ``hard_constraints``.
        """
        hc = self.hard_constraints
        if hc is False or hc == []:
            return set()

        t_min = getattr(self.domain, '_t_min', None)
        if t_min is None:
            return set()   # no time axis — ICs don't apply

        from .boundary import MeshNodeBC
        eligible: set = set()
        for bc in self.domain.boundary_conditions:
            if not isinstance(bc, MeshNodeBC) or bc.bc_type != 'dirichlet':
                continue
            name = getattr(bc, 'name', None)
            if name is None:
                continue
            tw = bc.time_window
            if tw is None:
                continue
            pts = [float(v) for v in tw]
            # IC = point BC at t_min
            if len(pts) <= 2 and abs(pts[0] - t_min) < 1e-10 and abs(pts[-1] - t_min) < 1e-10:
                eligible.add(name)

        if hc is True:
            return eligible
        return eligible & set(hc)

    @property
    def hard_bc_names(self) -> set:
        """Union of :attr:`hard_spatial_bc_names` and :attr:`hard_ic_names`.

        Used by the trainer and ``show_problem()`` to exclude all
        hard-constrained BCs from soft losses and display.
        """
        return self.hard_spatial_bc_names | self.hard_ic_names

    @property
    def _rollout_ic_bc_names(self) -> set:
        """All IC-type BCs (point-time at t_min) when in rollout mode.

        In rollout / BPTT mode these BCs are consumed as the initial state
        ``u0`` and must NOT appear as soft loss terms — regardless of what is
        listed in ``hard_constraints``.  Returns an empty set when the problem
        is not in rollout mode (``n_time_steps is None``).
        """
        if self.n_time_steps is None:
            return set()
        t_min = getattr(self.domain, '_t_min', None)
        if t_min is None:
            return set()
        from .boundary import MeshNodeBC
        eligible: set = set()
        for bc in self.domain.boundary_conditions:
            if not isinstance(bc, MeshNodeBC) or bc.bc_type != 'dirichlet':
                continue
            name = getattr(bc, 'name', None)
            if name is None:
                continue
            tw = bc.time_window
            if tw is None:
                continue
            pts = [float(v) for v in tw]
            if len(pts) <= 2 and abs(pts[0] - t_min) < 1e-10 and abs(pts[-1] - t_min) < 1e-10:
                eligible.add(name)
        return eligible

    @property
    def n_dofs(self) -> int:
        """Total number of global DOFs."""
        return len(self.cubature_data['dof_coords'])

    @property
    def boundary_conditions(self):
        """Pass-through to domain boundary conditions."""
        return self.domain.boundary_conditions

    @property
    def xmin(self):
        return self.domain.xmin

    @property
    def xmax(self):
        return self.domain.xmax

    def _build_params(self, internal=None):
        return {
            "fixed":    self.params,
            "infer":    {},
            "internal": internal or {'global_step': 0, 'step': 0},
        }

    @property
    def output_transform(self):
        """
        Hard-constraint output transform derived from Dirichlet BCs on the domain.

        Returns a callable ``f(x_original, y, params_dict) -> y`` that clamps
        network outputs to their prescribed Dirichlet values at constrained
        boundary nodes (only ``t_mode='all'`` BCs).

        Pass directly to a network::

            net = AlphaPINN(domain, n_features=32, hidden_dims=[128]*4,
                            output_transform=problem.output_transform)

        Returns ``None`` when ``hard_constraints=False`` or when there are no
        eligible Dirichlet BCs on the domain.
        """
        has_spatial = bool(self.hard_spatial_bc_names) and np.any(self._bc_mask_np)
        has_ic      = bool(self.hard_ic_names)          and np.any(self._ic_mask_np)
        if not has_spatial and not has_ic:
            return None

        _verts = self.domain._vertices[:, :self.domain._spatial_dims].astype(np.float32)
        _faces = self.domain._faces.astype(np.int32)
        _bc_mask   = np.array(self._bc_mask_np,   dtype=np.float32)
        _bc_values = np.array(self._bc_values_np, dtype=np.float32)
        _ic_mask   = np.array(self._ic_mask_np,   dtype=np.float32)
        _ic_values = np.array(self._ic_values_np, dtype=np.float32)
        _t_min     = float(self.domain._t_min) if self.domain._t_min is not None else 0.0
        _t_dim     = self.domain._spatial_dims   # index of the time column in x
        _has_spatial = has_spatial
        _has_ic      = has_ic
        _transition_fn = self._make_ic_transition_fn(self.ic_transition)

        def _transform(x_original, y, params_dict=None):
            from .backends.jax.gnn_network import _interpolate_mesh
            import jax.numpy as jnp
            nodes = jnp.array(_verts)
            faces = jnp.array(_faces)
            spatial_dims = _verts.shape[1]
            x_spatial = x_original[:, :spatial_dims]

            # ── Step 1: IC hard constraint — u = u_IC(x) + f(t−t_min)·NN ───
            if _has_ic:
                ic_m = _interpolate_mesh(jnp.array(_ic_mask),   nodes, faces, x_spatial)
                ic_v = _interpolate_mesh(jnp.array(_ic_values), nodes, faces, x_spatial)
                t_shifted = x_original[:, _t_dim:_t_dim+1] - _t_min   # (n, 1)
                factor = _transition_fn(t_shifted)                      # (n, 1)
                # At IC nodes: y = ic_v + factor * y
                # At other nodes: y unchanged
                y = ic_v * ic_m + y * (1.0 - ic_m + ic_m * factor)

            # ── Step 2: Spatial Dirichlet mask-and-clamp ────────────────────
            if _has_spatial:
                bc_m = _interpolate_mesh(jnp.array(_bc_mask),   nodes, faces, x_spatial)
                bc_v = _interpolate_mesh(jnp.array(_bc_values), nodes, faces, x_spatial)
                y = (1.0 - bc_m) * y + bc_m * bc_v

            return y

        return _transform

    def make_rollout_loss_fn(self, network, n_steps=None, face_batch_size=None, step_weights=None):
        """
        Return a JIT-able loss function that unrolls the GNN state-integrator
        over ``n_steps`` time steps via ``jax.lax.scan`` (BPTT).

        Parameters
        ----------
        network : GNNMeshNetwork
        n_steps : int, optional
            Number of time steps to unroll (defaults to ``self.n_time_steps``).
        face_batch_size : int, optional
            If set, ``loss_fn`` expects a second argument ``face_idx`` of shape
            ``(face_batch_size,)`` containing the face indices to use for the
            residual in each training step.  The GNN forward pass always runs on
            **all** mesh nodes; only the weak-form assembly is restricted to the
            selected faces.  Pass ``None`` (default) to use all faces.
        step_weights : array-like of shape (n_steps,), optional
            Per-step scalar weights applied to each step's MSE loss before
            averaging.  Useful for exponential weighting to counteract vanishing
            gradients in long rollouts.  If ``None`` all steps are weighted
            equally (uniform mean).

        Returns
        -------
        loss_fn : callable
            * No batching: ``loss_fn(params) -> scalar``
            * With batching: ``loss_fn(params, face_idx) -> scalar``
        """
        if self.n_time_steps is None:
            raise ValueError(
                "make_rollout_loss_fn requires n_time_steps to be set on ProblemWeak."
            )
        import jax
        import jax.numpy as jnp

        cd   = self.cubature_data
        # Keep face-level arrays (F, Q, L) so we can index by face_idx
        _phi = jnp.asarray(cd['phi'],      dtype=jnp.float32)  # (F, Q, L)
        _gph = jnp.asarray(cd['grad_phi'], dtype=jnp.float32)  # (F, Q, L, 2)
        _wts = jnp.asarray(cd['weights'],  dtype=jnp.float32)  # (F, Q)
        _nid = jnp.asarray(cd['node_ids'], dtype=jnp.int32)    # (F, L)
        _pts = jnp.asarray(cd['pts'],      dtype=jnp.float32)  # (F, Q, 2)

        F, Q, L = _phi.shape
        n_dofs  = self.n_dofs

        _free      = jnp.asarray(self.free_nodes, dtype=jnp.int32)
        _dt        = jnp.float32(self.domain.dt)
        _n_time    = n_steps if n_steps is not None else self.n_time_steps
        _net       = network
        _mesh      = jnp.asarray(self.domain._vertices, dtype=jnp.float32)
        _fixed_par = self.params          # user-supplied fixed params (kappa, …)
        _volume_fn = self.volume_fn       # the user's weak-form integrand

        # Per-step weights: normalised so they sum to n_time (same scale as uniform mean)
        if step_weights is not None:
            import numpy as _np_sw
            _sw = _np_sw.asarray(step_weights, dtype=_np_sw.float32)
            _sw = _sw / _sw.mean()   # keep average weight = 1
            _step_weights_jax = jnp.asarray(_sw, dtype=jnp.float32)  # (n_time,)
        else:
            _step_weights_jax = None

        # ── Full-batch flattened arrays ──────────────────────────────────────
        _phi_f_full = _phi.reshape(F * Q, L)
        _gph_f_full = _gph.reshape(F * Q, L, 2)
        _wts_f_full = _wts.reshape(F * Q)
        _nid_f_full = jnp.tile(_nid[:, None, :], (1, Q, 1)).reshape(F * Q, L)
        _pts_f_full = _pts.reshape(F * Q, _pts.shape[-1])  # (FQ, 2)

        # ── Lumped-mass normaliser: m[i] = ∫φ_i dΩ ─────────────────────────
        _lumped_mass = jnp.zeros(n_dofs, dtype=jnp.float32)
        _lumped_mass = _lumped_mass.at[_nid_f_full.reshape(-1)].add(
            (_wts_f_full[:, None] * _phi_f_full).reshape(-1)
        )
        _lumped_mass_free = jnp.maximum(_lumped_mass[_free], 1e-12)

        def _build_step_fn(phi_f, gph_f, wts_f, nid_f, pts_f, lumped_mass_free=None):
            """Build a scan step that calls volume_fn for each local test function.

            pdict layout seen by volume_fn:
              pdict["fixed"]    — user params + u_prev_nodes (read by AlphaPINN)
              pdict["internal"] — per-step assembler state: u_prev, dt, …
            """
            _lm_free = _lumped_mass_free if lumped_mass_free is None else lumped_mass_free

            def step_fn(carry, w):
                params, u_nodes = carry
                # Interpolate u^n and its FEM gradient at quadrature points
                u_prev_cub  = jnp.sum(phi_f * u_nodes[nid_f], axis=-1)           # (FQ,)
                grad_u_prev = jnp.einsum('kl,kli->ki', u_nodes[nid_f], gph_f)    # (FQ, 2)

                # pdict["fixed"] carries u_prev_nodes so AlphaPINN can encode α^n
                # pdict["internal"] carries per-step state for volume_fn
                pdict = {
                    "fixed":    {**_fixed_par, "u_prev_nodes": u_nodes},
                    "internal": {"u_prev": u_prev_cub, "grad_u_prev": grad_u_prev, "dt": _dt},
                }

                # Network forward: u^{n+1} at all mesh nodes
                u_next_nodes = _net.apply(params, _mesh, pdict)[:, 0]  # (n_dofs,)

                # Interpolate u^{n+1} and its FEM gradient at quadrature points
                u_next_cub = jnp.sum(phi_f * u_next_nodes[nid_f], axis=-1)  # (FQ,)
                y_flat     = u_next_cub[:, None]                              # (FQ, 1)
                grad_u     = jnp.einsum('kl,kli->ki', u_next_nodes[nid_f], gph_f)  # (FQ, 2)

                # Assemble residual by calling volume_fn for each local basis fn
                # L is small (3 for P1) so the Python loop unrolls at trace time.
                R = jnp.zeros(n_dofs, dtype=jnp.float32)
                for la in range(L):
                    phi_a  = phi_f[:, la]       # (FQ,)  test-fn values
                    gphi_a = gph_f[:, la, :]    # (FQ, 2) test-fn gradients

                    # derivative wrapper: phi is 1-D, y is 2-D
                    def _deriv(Y, X, comp, order, _gu=grad_u, _gp=gphi_a):
                        dim = order[0] if isinstance(order, (tuple, list)) else order
                        return _gp[:, dim] if Y.ndim == 1 else _gu[:, dim]

                    integrand = _volume_fn(pts_f, y_flat, pdict, phi_a, _deriv)  # (FQ,)
                    R = R.at[nid_f[:, la]].add(integrand * wts_f)

                step_loss = w * jnp.mean((R[_free] / _lm_free) ** 2)
                return (params, u_next_nodes), step_loss
            return step_fn

        # xs fed into scan: per-step weights (or ones if not provided)
        if _step_weights_jax is not None:
            _xs_full = _step_weights_jax  # (n_time,)
        else:
            _xs_full = jnp.ones(_n_time, dtype=jnp.float32)

        # ── Initial nodal state u0 ──────────────────────────────────────────
        # Seed BC nodes with their Dirichlet value (e.g. hot_top=1.0); all
        # other nodes take the IC value (typically 0.0).  Using zeros here
        # means the first rollout step sees hot boundary nodes as u_prev=0,
        # which pollutes the weak-form residual and prevents convergence.
        _u0 = jnp.where(
            jnp.asarray(self._bc_mask_np[:, 0], dtype=bool),
            jnp.asarray(self._bc_values_np[:, 0], dtype=jnp.float32),
            jnp.asarray(self._ic_values_np[:, 0], dtype=jnp.float32),
        )

        if face_batch_size is None:
            # ── Full-batch: loss_fn(params) ──────────────────────────────
            _step = _build_step_fn(
                _phi_f_full, _gph_f_full, _wts_f_full, _nid_f_full, _pts_f_full
            )

            def loss_fn(params):
                (_, _), step_losses = jax.lax.scan(_step, (params, _u0), _xs_full)
                return jnp.mean(step_losses)

        else:
            # ── Mini-batch: loss_fn(params, face_idx) ────────────────────
            def loss_fn(params, face_idx):
                phi_b = _phi[face_idx]   # (B, Q, L)
                gph_b = _gph[face_idx]   # (B, Q, L, 2)
                wts_b = _wts[face_idx]   # (B, Q)
                nid_b = _nid[face_idx]   # (B, L)
                pts_b = _pts[face_idx]   # (B, Q, 2)
                B = face_batch_size
                phi_f = phi_b.reshape(B * Q, L)
                gph_f = gph_b.reshape(B * Q, L, 2)
                wts_f = wts_b.reshape(B * Q)
                nid_f = jnp.tile(nid_b[:, None, :], (1, Q, 1)).reshape(B * Q, L)
                pts_f = pts_b.reshape(B * Q, pts_b.shape[-1])
                lm_batch = jnp.zeros(n_dofs, dtype=jnp.float32)
                lm_batch = lm_batch.at[nid_f.reshape(-1)].add(
                    (wts_f[:, None] * phi_f).reshape(-1)
                )
                lm_batch_free = jnp.maximum(lm_batch[_free], 1e-12)
                _step = _build_step_fn(phi_f, gph_f, wts_f, nid_f, pts_f, lm_batch_free)
                (_, _), step_losses = jax.lax.scan(_step, (params, _u0), _xs_full)
                return jnp.mean(step_losses)

        return loss_fn

    def make_rollout_al_loss_fn(self, network, n_steps=None):
        """
        Augmented-Lagrangian variant of ``make_rollout_loss_fn``.

        Each time step gets its own Lagrange-multiplier vector λ_n ∈ ℝ^{n_free}.
        The per-step loss is the linearised AL term:

            ℒ_n = λ_n · R_n / n_free  +  0.5 · mean(R_n²)

        where R_n = R[free_nodes] is the free-node residual vector at step n.
        The total loss is mean(ℒ_1, …, ℒ_N).

        Parameters
        ----------
        network : AlphaPINN (or any network with .apply(params, mesh, pdict))
        n_steps : int, optional
            Defaults to ``self.n_time_steps``.

        Returns
        -------
        loss_fn : callable
            ``loss_fn(params, lambdas) -> (scalar, step_residuals)``

            *   ``lambdas``        shape ``(n_time, n_free)``
            *   ``step_residuals`` shape ``(n_time, n_free)``  — use for dual ascent:
                ``lambdas += lr * step_residuals``
        """
        if self.n_time_steps is None:
            raise ValueError(
                "make_rollout_al_loss_fn requires n_time_steps to be set on ProblemWeak."
            )
        import jax
        import jax.numpy as jnp

        cd   = self.cubature_data
        _phi = jnp.asarray(cd['phi'],      dtype=jnp.float32)  # (F, Q, L)
        _gph = jnp.asarray(cd['grad_phi'], dtype=jnp.float32)  # (F, Q, L, 2)
        _wts = jnp.asarray(cd['weights'],  dtype=jnp.float32)  # (F, Q)
        _nid = jnp.asarray(cd['node_ids'], dtype=jnp.int32)    # (F, L)

        F, Q, L = _phi.shape
        n_dofs  = self.n_dofs
        _free   = jnp.asarray(self.free_nodes, dtype=jnp.int32)
        n_free  = len(self.free_nodes)

        _dt         = jnp.float32(self.domain.dt)
        _n_time     = n_steps if n_steps is not None else self.n_time_steps
        _net        = network
        _mesh       = jnp.asarray(self.domain._vertices, dtype=jnp.float32)
        _fixed_par  = self.params
        _volume_fn  = self.volume_fn
        # node_norm[a] = ∫ φ_a dΩ — scale-free normaliser
        _node_norm_free = jnp.asarray(self.node_norm[self.free_nodes], dtype=jnp.float32)

        phi_f = _phi.reshape(F * Q, L)
        gph_f = _gph.reshape(F * Q, L, 2)
        wts_f = _wts.reshape(F * Q)
        nid_f = jnp.tile(_nid[:, None, :], (1, Q, 1)).reshape(F * Q, L)
        pts_f = jnp.asarray(cd['pts'], dtype=jnp.float32).reshape(F * Q, -1)  # (FQ, 2)

        def step_fn(carry, lam_n):
            params, u_nodes = carry
            u_prev_cub  = jnp.sum(phi_f * u_nodes[nid_f], axis=-1)           # (FQ,)
            grad_u_prev = jnp.einsum('kl,kli->ki', u_nodes[nid_f], gph_f)    # (FQ, 2)

            pdict = {
                "fixed":    {**_fixed_par, "u_prev_nodes": u_nodes},
                "internal": {"u_prev": u_prev_cub, "grad_u_prev": grad_u_prev, "dt": _dt},
            }

            u_next_nodes = _net.apply(params, _mesh, pdict)[:, 0]   # (n_dofs,)

            u_next_cub = jnp.sum(phi_f * u_next_nodes[nid_f], axis=-1)  # (FQ,)
            y_flat     = u_next_cub[:, None]                              # (FQ, 1)
            grad_u     = jnp.einsum('kl,kli->ki', u_next_nodes[nid_f], gph_f)  # (FQ, 2)

            R = jnp.zeros(n_dofs, dtype=jnp.float32)
            for la in range(L):
                phi_a  = phi_f[:, la]
                gphi_a = gph_f[:, la, :]

                def _deriv(Y, X, comp, order, _gu=grad_u, _gp=gphi_a):
                    dim = order[0] if isinstance(order, (tuple, list)) else order
                    return _gp[:, dim] if Y.ndim == 1 else _gu[:, dim]

                integrand = _volume_fn(pts_f, y_flat, pdict, phi_a, _deriv)  # (FQ,)
                R = R.at[nid_f[:, la]].add(integrand * wts_f)

            # Normalise by support volume: R̂[a] = R[a] / ∫ φ_a dΩ
            R_free = R[_free] / _node_norm_free
            # Augmented Lagrangian: linear term + quadratic penalty
            step_loss = jnp.dot(lam_n, R_free) / n_free + 0.5 * jnp.mean(R_free ** 2)
            return (params, u_next_nodes), (step_loss, R_free)

        def loss_fn(params, lambdas):
            # lambdas: (n_time, n_free) — xs fed one row per scan step
            _u0_al = jnp.where(
                jnp.asarray(self._bc_mask_np[:, 0], dtype=bool),
                jnp.asarray(self._bc_values_np[:, 0], dtype=jnp.float32),
                jnp.asarray(self._ic_values_np[:, 0], dtype=jnp.float32),
            )
            (_, _), (step_losses, step_residuals) = jax.lax.scan(
                step_fn, (params, _u0_al), lambdas
            )
            return jnp.mean(step_losses), step_residuals

        return loss_fn

    def make_loss_fn(self, u_and_grad_fn, bc_weights: dict = None, node_batch_size: int = None):
        """
        Return a JAX-jittable loss function that assembles the full weak-form
        residual and returns the MSE over free nodes.

        When ``random_time_sampling=False`` the returned function has signature
        ``loss_fn(params) -> scalar``.

        When ``random_time_sampling=True`` the returned function has signature
        ``loss_fn(params, t_vals) -> scalar``.

        When ``node_batch_size`` is set (only valid with ``random_time_sampling=True``)
        the returned function has signature
        ``loss_fn(params, node_idx, t_per_node) -> scalar``.
        ``node_idx`` selects which free nodes (test functions) to include;
        ``t_per_node`` assigns one random time level per selected node.
        This gives an unbiased estimate of the full loss and only evaluates
        the small patch of faces that support each selected test function.
        """
        import jax
        import jax.numpy as jnp

        _random_time = self._random_time_sampling

        cd             = self.cubature_data
        pts_jax        = jnp.asarray(cd['pts'],      dtype=jnp.float32)   # (F, Q, 2) or (n_t*F, Q, 3)
        weights_jax    = jnp.asarray(cd['weights'],  dtype=jnp.float32)   # (F, Q) or (n_t*F, Q)
        phi_jax        = jnp.asarray(cd['phi'],      dtype=jnp.float32)   # (F, Q, L)
        grad_phi_jax   = jnp.asarray(cd['grad_phi'], dtype=jnp.float32)   # (F, Q, L, 2)
        node_ids_jax   = jnp.asarray(cd['node_ids'], dtype=jnp.int32)     # (F, L)
        free_nodes_jax        = jnp.asarray(self.free_nodes,         dtype=jnp.int32)
        inner_free_nodes_jax  = jnp.asarray(self.inner_free_nodes,   dtype=jnp.int32)
        boundary_free_nodes_jax = jnp.asarray(self.boundary_free_nodes, dtype=jnp.int32)
        node_norm_jax         = jnp.asarray(self.node_norm,          dtype=jnp.float32)
        _has_inner    = len(self.inner_free_nodes)    > 0
        _has_boundary = len(self.boundary_free_nodes) > 0

        n_dofs  = self.n_dofs
        n_faces = pts_jax.shape[0]
        n_qpts  = pts_jax.shape[1]
        n_local = phi_jax.shape[2]          # (N+1)(N+2)/2
        volume_fn   = self.volume_fn
        params_dict = self._build_params()

        _bc_weights = bc_weights or {}

        # Spatial-only dimensions for random-time tiling
        _sp_F = n_faces   # spatial faces (same as n_faces when random_time_sampling)
        _sp_Q = n_qpts
        _t_interval_f = float((self._t_max or 1.0) - (self._t_min or 0.0)) if _random_time else 1.0

        # Pre-convert boundary data to JAX arrays (outside closure for efficiency)
        # Also precompute the set of free nodes belonging to each weak-BC boundary
        # so their residuals can be weighted independently.
        free_nodes_set = set(self.free_nodes.tolist())
        _bdata_jax = []
        for _bd in self.boundary_fn_data:
            raw_ids = _bd['edge_ids'].flatten().tolist()
            bc_free = sorted(set(raw_ids) & free_nodes_set)
            _bdata_jax.append({
                'pts':       jnp.asarray(_bd['pts'],      dtype=jnp.float32),  # (E, Q, 2)
                'weights':   jnp.asarray(_bd['weights'],  dtype=jnp.float32),  # (E, Q)
                'phi':       jnp.asarray(_bd['phi'],      dtype=jnp.float32),  # (E, Q, 2)
                'edge_ids':  jnp.asarray(_bd['edge_ids'], dtype=jnp.int32),    # (E, 2)
                'fn':        _bd['fn'],
                'name':      _bd['name'],   # str or list[str], one entry per component
                'free_nodes': jnp.asarray(bc_free, dtype=jnp.int32) if bc_free else None,
            })

        _n_pts_dims = pts_jax.shape[-1]  # 2 for spatial-only, 3 for space-time

        if _random_time:
            if node_batch_size is not None:
                # ── Node-level mini-batch ────────────────────────────────────────
                # For each free node j, precompute the list of (face_idx, local_a)
                # pairs that contribute to its residual R_j.
                import numpy as _np
                _cd_nids_np = np.array(cd['node_ids'])   # (F, L)
                _F, _L = _cd_nids_np.shape
                _free_set = {int(v): i for i, v in enumerate(np.array(self.free_nodes))}
                _N_free = len(self.free_nodes)
                _patch_list = [[] for _ in range(_N_free)]
                for _f in range(_F):
                    for _a in range(_L):
                        _gn = int(_cd_nids_np[_f, _a])
                        if _gn in _free_set:
                            _patch_list[_free_set[_gn]].append((_f, _a))
                _max_K = max(len(p) for p in _patch_list) if _patch_list else 1
                _pf_np  = np.zeros((_N_free, _max_K), dtype=np.int32)
                _pa_np  = np.zeros((_N_free, _max_K), dtype=np.int32)
                _pm_np  = np.zeros((_N_free, _max_K), dtype=np.float32)
                for _fi, _patches in enumerate(_patch_list):
                    for _k, (_f, _a) in enumerate(_patches):
                        _pf_np[_fi, _k] = _f
                        _pa_np[_fi, _k] = _a
                        _pm_np[_fi, _k] = 1.0
                _patch_faces_j = jnp.asarray(_pf_np)    # (N_free, max_K)
                _patch_local_j = jnp.asarray(_pa_np)    # (N_free, max_K)
                _patch_mask_j  = jnp.asarray(_pm_np)    # (N_free, max_K)
                _Q_val  = n_qpts
                _K_val  = _max_K

                def loss_fn(params, node_idx, t_per_node):
                    # node_idx: (B,)  t_per_node: (B,)
                    B    = node_idx.shape[0]
                    pf   = _patch_faces_j[node_idx]   # (B, K)
                    pa   = _patch_local_j[node_idx]   # (B, K)
                    pm   = _patch_mask_j[node_idx]    # (B, K)
                    # Build space-time quad points: one t per node, broadcast to all its patches
                    pts_xy  = pts_jax[pf]             # (B, K, Q, 2)
                    t_bkq   = jnp.reshape(t_per_node, (B, 1, 1, 1)) * jnp.ones((1, _K_val, _Q_val, 1))
                    pts_st  = jnp.concatenate([pts_xy, t_bkq], axis=-1)  # (B, K, Q, 3)
                    BK      = B * _K_val
                    # Evaluate network at all (B*K*Q) quadrature points
                    u_f, gu_f = jax.vmap(
                        lambda xy: u_and_grad_fn(params, xy))(
                        pts_st.reshape(BK * _Q_val, 3))
                    _scalar = (u_f.ndim == 1)
                    if _scalar:
                        y_bkq  = u_f.reshape(BK, _Q_val, 1)          # (BK, Q, 1)
                        gu_bkq = gu_f.reshape(BK, _Q_val, gu_f.shape[-1])  # (BK, Q, n_dims)
                    else:
                        y_bkq  = u_f.reshape(BK, _Q_val, u_f.shape[-1])
                        gu_bkq = gu_f.reshape(BK, _Q_val, u_f.shape[-1], gu_f.shape[-1])
                    # Gather test-function arrays for each (b, k)
                    phi_bk   = phi_jax[pf, :, pa]          # (B, K, Q)
                    gphi_bk  = grad_phi_jax[pf, :, pa, :]  # (B, K, Q, 2)
                    w_bk     = weights_jax[pf]              # (B, K, Q)
                    # Per-element scalar integral via vmap
                    def single_elem(y_q, gu_q, phi_q, gphi_q, w_q, pts_q):
                        # shapes: (Q,1), (Q,n), (Q,), (Q,2), (Q,), (Q,3)
                        if _scalar:
                            def _d(Y, X, comp, order):
                                dim = order[0] if isinstance(order, (tuple, list)) else order
                                if Y.ndim == 1: return gphi_q[:, dim]  # derivative(phi,...)
                                return gu_q[:, dim]
                        else:
                            def _d(Y, X, comp, order):
                                dim = order[0] if isinstance(order, (tuple, list)) else order
                                if Y.ndim == 1: return gphi_q[:, dim]  # derivative(phi,...)
                                return gu_q[:, comp, dim]
                        ig = volume_fn(pts_q, y_q, params_dict, phi_q, _d)
                        return jnp.sum(w_q * ig)
                    elem_ints = jax.vmap(single_elem)(
                        y_bkq,
                        gu_bkq,
                        phi_bk.reshape(BK, _Q_val),
                        gphi_bk.reshape(BK, _Q_val, 2),
                        w_bk.reshape(BK, _Q_val),
                        pts_st.reshape(BK, _Q_val, 3),
                    )  # (BK,)
                    # Scale by time-interval length (MC weight for unbiased time integral)
                    elem_ints = (elem_ints * _t_interval_f).reshape(B, _K_val)
                    # Sum patches per node (masked)
                    R = jnp.sum(elem_ints * pm, axis=1)   # (B,)
                    # Normalise by node support volume (same as full _assemble path)
                    norm_b = node_norm_jax[free_nodes_jax[node_idx]]  # (B,)
                    R_norm = R / norm_b
                    return jnp.mean(R_norm ** 2)
            else:
                def loss_fn(params, t_vals):
                    # Dynamically tile spatial cubature with sampled time levels
                    n_t = t_vals.shape[0]
                    dt_w = _t_interval_f / n_t                           # MC time weight
                    t4d  = jnp.reshape(t_vals, (n_t, 1, 1, 1)) * jnp.ones((_sp_F, _sp_Q, 1))
                    pts_xy4d = jnp.broadcast_to(pts_jax[None], (n_t, _sp_F, _sp_Q, 2))
                    pts_st   = jnp.concatenate([pts_xy4d, t4d], axis=-1).reshape(n_t * _sp_F, _sp_Q, 3)
                    _eff_pts   = pts_st
                    _eff_w     = jnp.tile(weights_jax, (n_t, 1)) * dt_w  # (n_t*F, Q)
                    _eff_phi   = jnp.tile(phi_jax,     (n_t, 1, 1))      # (n_t*F, Q, L)
                    _eff_gphi  = jnp.tile(grad_phi_jax,(n_t, 1, 1, 1))   # (n_t*F, Q, L, 2)
                    _eff_nids  = jnp.tile(node_ids_jax,(n_t, 1))         # (n_t*F, L)
                    _eff_nf    = n_t * _sp_F
                    _eff_nq    = _sp_Q
                    return _assemble(params, _eff_pts, _eff_w, _eff_phi, _eff_gphi, _eff_nids, _eff_nf, _eff_nq)
        else:
            def loss_fn(params):
                return _assemble(params, pts_jax, weights_jax, phi_jax, grad_phi_jax, node_ids_jax, n_faces, n_qpts)

        def _assemble(params, _eff_pts, _eff_w, _eff_phi, _eff_gphi, _eff_nids, _eff_nf, _eff_nq):
            pts_flat = _eff_pts.reshape(-1, _eff_pts.shape[-1])               # (F*Q, n_dims)

            # Evaluate u and ∇u / full Jacobian at all quadrature points
            u_flat, grad_u_flat = jax.vmap(
                lambda xy: u_and_grad_fn(params, xy))(pts_flat)

            # Support both scalar (n,) → y_flat (n,1) and multi-output (n, n_out)
            # grad_u_flat: scalar → (n, n_dims),  multi-output → (n, n_out, n_dims)
            if u_flat.ndim == 1:
                y_flat = u_flat.reshape(-1, 1)                        # (F*Q, 1)
                def make_deriv(gu, gphi):
                    def deriv_fn(Y, X, component, order):
                        dim = order[0] if isinstance(order, (list, tuple)) else order
                        if Y.ndim == 1: return gphi[:, dim]  # derivative(phi,...)
                        return gu[:, dim]
                    return deriv_fn
            else:
                y_flat = u_flat                                        # (F*Q, n_out)
                def make_deriv(jac, gphi):
                    def deriv_fn(Y, X, component, order):
                        dim = order[0] if isinstance(order, (list, tuple)) else order
                        if Y.ndim == 1: return gphi[:, dim]  # derivative(phi,...)
                        return jac[:, component, dim]
                    return deriv_fn

            # Assemble global residual(s): loop over n_local local DOFs per element.
            # volume_fn may return (F*Q,) for a scalar equation or a tuple/list of
            # n_out arrays each of shape (F*Q,) for a vector equation.  In the
            # multi-component case each R_k is assembled from the k-th integrand.
            # The loss is the mean of MSE across all component residual vectors.
            _gphi0 = _eff_gphi[:, :, 0, :].reshape(-1, 2)
            R_sample = volume_fn(
                pts_flat, y_flat, params_dict,
                _eff_phi[:, :, 0].reshape(-1),
                make_deriv(grad_u_flat, _gphi0),
            )
            _multi = isinstance(R_sample, (tuple, list))
            _n_comp = len(R_sample) if _multi else 1

            Rs = [jnp.zeros(n_dofs, dtype=jnp.float32) for _ in range(_n_comp)]
            for a in range(n_local):
                phi_a  = _eff_phi[:, :, a]                            # (F, Q)
                gphi_a = _eff_gphi[:, :, a, :]                        # (F, Q, 2)
                gphi_a_flat = gphi_a.reshape(-1, 2)

                integrand = volume_fn(
                    pts_flat,
                    y_flat,
                    params_dict,
                    phi_a.reshape(-1),
                    make_deriv(grad_u_flat, gphi_a_flat),
                )                                        # (F*Q,) or tuple of (F*Q,)

                if _multi:
                    for k, ig in enumerate(integrand):
                        elem_int = jnp.einsum(
                            'fq,fq->f', _eff_w,
                            ig.reshape(_eff_nf, _eff_nq))
                        Rs[k] = Rs[k].at[_eff_nids[:, a]].add(elem_int)
                else:
                    elem_int = jnp.einsum(
                        'fq,fq->f', _eff_w,
                        integrand.reshape(_eff_nf, _eff_nq))
                    Rs[0] = Rs[0].at[_eff_nids[:, a]].add(elem_int)

            # ── Subtract boundary traction RHS  ∫_Γ t_k · φ_j ds ──────────────
            for _bj in _bdata_jax:
                _bp_ndims   = _bj['pts'].shape[-1]
                _bpts_flat  = _bj['pts'].reshape(-1, _bp_ndims)   # (E*Q, n_dims)
                _bw         = _bj['weights']                     # (E, Q)
                _bphi_mat   = _bj['phi']                         # (E, Q, 2)
                _beid       = _bj['edge_ids']                    # (E, 2)
                _bfn        = _bj['fn']
                _n_bedges   = _bj['pts'].shape[0]
                _n_bqpts    = _bj['pts'].shape[1]

                # Evaluate network at boundary quadrature points
                _bu_flat, _bgu_flat = jax.vmap(
                    lambda xy: u_and_grad_fn(params, xy))(_bpts_flat)
                if _bu_flat.ndim == 1:
                    _by_flat = _bu_flat.reshape(-1, 1)
                    def _make_bderiv(bgu):
                        def _bderiv(Y, X, component, order):
                            dim = order[0] if isinstance(order, (list, tuple)) else order
                            return bgu[:, dim]
                        return _bderiv
                    _bderiv = _make_bderiv(_bgu_flat)
                else:
                    _by_flat = _bu_flat
                    def _make_bderiv(bjac):
                        def _bderiv(Y, X, component, order):
                            dim = order[0] if isinstance(order, (list, tuple)) else order
                            return bjac[:, component, dim]
                        return _bderiv
                    _bderiv = _make_bderiv(_bgu_flat)

                for _p in range(2):   # 2 endpoint nodes per boundary edge
                    _bphi_p = _bphi_mat[:, :, _p]               # (E, Q)
                    _b_intg = _bfn(
                        _bpts_flat, _by_flat, params_dict,
                        _bphi_p.reshape(-1), _bderiv,
                    )
                    if _multi:
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

            # Loss = inner free-node MSE  +  per-BC weighted boundary MSEs.
            # Each weak-BC entry in _bdata_jax carries its own weight key(s)
            # so the user can tune each boundary independently via the weights dict.
            # R is normalised by norm_j = V_j + L_j:
            #   V_j = nodal support volume (2-D, ~ h²)  handles ∫_Ω σ:∇φ dΩ ~ h²
            #   L_j = boundary edge length support (1-D, ~ h) handles ∫_Γ t·φ dS ~ h
            # Together they keep R̂_j = R_j / norm_j ~ O(σ) for all node types.
            def _node_loss(R, comp_idx):
                R_norm = R / node_norm_jax   # R̂_j = R_j / (V_j + L_j)
                # Interior nodes
                if _has_inner:
                    loss = jnp.mean(R_norm[inner_free_nodes_jax] ** 2)
                elif not _has_boundary:
                    return jnp.mean(R_norm[free_nodes_jax] ** 2)
                else:
                    loss = 0.0
                # Per-BC boundary contributions
                for _bj in _bdata_jax:
                    if _bj['free_nodes'] is None:
                        continue
                    _name = _bj['name']
                    if isinstance(_name, (list, tuple)):
                        key = _name[comp_idx] if comp_idx < len(_name) else None
                    else:
                        key = _name if comp_idx == 0 else None
                    w = float(_bc_weights.get(key, 1.0)) if key is not None else 1.0
                    loss = loss + w * jnp.mean(R_norm[_bj['free_nodes']] ** 2)
                return loss
            return sum(_node_loss(R, k) for k, R in enumerate(Rs)) / _n_comp

        return loss_fn

    def make_residual_vector_fn(self, u_and_grad_fn):
        """
        Return a JAX-jittable function that assembles and returns the full
        per-DOF residual vector  R  (shape ``(n_dofs,)``).

        Useful for diagnostics and plotting: the nodal weak residual is

            R_j = \\sum_{k∋j} \\int_{T_k} volume_fn(x, u, params, φ_j, ∇φ_j)  dΩ

        Free-node entries encode how well the weak form is satisfied;
        Dirichlet-node entries are zero.

        Parameters
        ----------
        u_and_grad_fn : same as in :meth:`make_loss_fn`.

        Returns
        -------
        residual_fn : callable
            ``residual_fn(params) -> jnp.ndarray``  shape ``(n_dofs,)``
        """
        import jax
        import jax.numpy as jnp

        cd             = self.cubature_data
        _pts_sp        = cd['pts']   # (F, Q, 2) spatial — may need time tiling

        # ── If random-time-sampling, tile spatial cubature with midpoint times ─
        # cd['pts'] is spatial-only (F,Q,2) in random-time mode.  Append the
        # midpoints of n_t equal sub-intervals as representative time levels so
        # the residual vector covers the full time domain rather than crashing
        # with a 2-D input when LaplacianFeatures expects a 3-D (x,y,t) point.
        _random_time = getattr(self, '_random_time_sampling', False)
        if _random_time:
            import numpy as _np_rvf
            _t0  = float(getattr(self, '_t_min', 0.0) or 0.0)
            _t1  = float(getattr(self, '_t_max', 1.0) or 1.0)
            _n_t = int(getattr(self, '_n_t', 10) or 10)
            _dt  = (_t1 - _t0) / _n_t
            _t_vals = _t0 + (_np_rvf.arange(_n_t) + 0.5) * _dt   # midpoints
            _F_sp, _Q_sp = _pts_sp.shape[:2]
            _pts_xy4d = _np_rvf.broadcast_to(
                _pts_sp[None], (_n_t, _F_sp, _Q_sp, 2)).copy()
            _t_col = _np_rvf.broadcast_to(
                _t_vals[:, None, None, None], (_n_t, _F_sp, _Q_sp, 1)).copy()
            _pts_st = _np_rvf.concatenate([_pts_xy4d, _t_col], axis=-1)  # (n_t,F,Q,3)
            _pts_st = _pts_st.reshape(_n_t * _F_sp, _Q_sp, 3).astype(_np_rvf.float32)
            _weights_np = _np_rvf.tile(cd['weights'], (_n_t, 1)) * _dt
            _phi_np     = _np_rvf.tile(cd['phi'],     (_n_t, 1, 1))
            _gphi_np    = _np_rvf.tile(cd['grad_phi'],(_n_t, 1, 1, 1))
            _node_ids_np= _np_rvf.tile(cd['node_ids'],(_n_t, 1))
            pts_jax      = jnp.asarray(_pts_st,       dtype=jnp.float32)
            weights_jax  = jnp.asarray(_weights_np,   dtype=jnp.float32)
            phi_jax      = jnp.asarray(_phi_np,       dtype=jnp.float32)
            grad_phi_jax = jnp.asarray(_gphi_np,      dtype=jnp.float32)
            node_ids_jax = jnp.asarray(_node_ids_np,  dtype=jnp.int32)
        else:
            pts_jax      = jnp.asarray(_pts_sp,          dtype=jnp.float32)
            weights_jax  = jnp.asarray(cd['weights'],    dtype=jnp.float32)
            phi_jax      = jnp.asarray(cd['phi'],        dtype=jnp.float32)
            grad_phi_jax = jnp.asarray(cd['grad_phi'],   dtype=jnp.float32)
            node_ids_jax = jnp.asarray(cd['node_ids'],   dtype=jnp.int32)

        n_dofs  = self.n_dofs
        n_faces = pts_jax.shape[0]
        n_qpts  = pts_jax.shape[1]
        n_local = phi_jax.shape[2]
        volume_fn   = self.volume_fn
        params_dict = self._build_params()

        # Pre-convert boundary data to JAX arrays (outside closure for efficiency)
        _bdata_jax = []
        for _bd in self.boundary_fn_data:
            _bdata_jax.append({
                'pts':      jnp.asarray(_bd['pts'],      dtype=jnp.float32),
                'weights':  jnp.asarray(_bd['weights'],  dtype=jnp.float32),
                'phi':      jnp.asarray(_bd['phi'],      dtype=jnp.float32),
                'edge_ids': jnp.asarray(_bd['edge_ids'], dtype=jnp.int32),
                'fn':       _bd['fn'],
            })

        _n_pts_dims_r = pts_jax.shape[-1]

        def residual_fn(params):
            pts_flat = pts_jax.reshape(-1, _n_pts_dims_r)
            u_flat, grad_u_flat = jax.vmap(
                lambda xy: u_and_grad_fn(params, xy))(pts_flat)

            if u_flat.ndim == 1:
                y_flat = u_flat.reshape(-1, 1)
                def make_deriv(gu, gphi):
                    def deriv_fn(Y, X, component, order):
                        dim = order[0] if isinstance(order, (list, tuple)) else order
                        if Y.ndim == 1: return gphi[:, dim]  # derivative(phi,...)
                        return gu[:, dim]
                    return deriv_fn
            else:
                y_flat = u_flat
                def make_deriv(jac, gphi):
                    def deriv_fn(Y, X, component, order):
                        dim = order[0] if isinstance(order, (list, tuple)) else order
                        if Y.ndim == 1: return gphi[:, dim]  # derivative(phi,...)
                        return jac[:, component, dim]
                    return deriv_fn

            _gphi0 = grad_phi_jax[:, :, 0, :].reshape(-1, 2)
            R_sample = volume_fn(
                pts_flat, y_flat, params_dict,
                phi_jax[:, :, 0].reshape(-1),
                make_deriv(grad_u_flat, _gphi0),
            )
            _multi = isinstance(R_sample, (tuple, list))
            _n_comp = len(R_sample) if _multi else 1

            Rs = [jnp.zeros(n_dofs, dtype=jnp.float32) for _ in range(_n_comp)]
            for a in range(n_local):
                phi_a  = phi_jax[:, :, a]
                gphi_a = grad_phi_jax[:, :, a, :]
                gphi_a_flat = gphi_a.reshape(-1, 2)
                integrand = volume_fn(
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

            # ── Subtract boundary traction RHS  ∫_Γ t_k · φ_j ds ──────────────
            for _bj in _bdata_jax:
                _bp_ndims_r = _bj['pts'].shape[-1]
                _bpts_flat  = _bj['pts'].reshape(-1, _bp_ndims_r)
                _bw         = _bj['weights']
                _bphi_mat   = _bj['phi']
                _beid       = _bj['edge_ids']
                _bfn        = _bj['fn']
                _n_bedges   = _bj['pts'].shape[0]
                _n_bqpts    = _bj['pts'].shape[1]

                _bu_flat, _bgu_flat = jax.vmap(
                    lambda xy: u_and_grad_fn(params, xy))(_bpts_flat)
                if _bu_flat.ndim == 1:
                    _by_flat = _bu_flat.reshape(-1, 1)
                    def _make_bderiv(bgu):
                        def _bderiv(Y, X, component, order):
                            dim = order[0] if isinstance(order, (list, tuple)) else order
                            return bgu[:, dim]
                        return _bderiv
                    _bderiv = _make_bderiv(_bgu_flat)
                else:
                    _by_flat = _bu_flat
                    def _make_bderiv(bjac):
                        def _bderiv(Y, X, component, order):
                            dim = order[0] if isinstance(order, (list, tuple)) else order
                            return bjac[:, component, dim]
                        return _bderiv
                    _bderiv = _make_bderiv(_bgu_flat)

                for _p in range(2):
                    _bphi_p = _bphi_mat[:, :, _p]
                    _b_intg = _bfn(
                        _bpts_flat, _by_flat, params_dict,
                        _bphi_p.reshape(-1), _bderiv,
                    )
                    if _multi:
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

            # Return the full stacked residual vector for diagnostics
            return jnp.concatenate([R for R in Rs])

        return residual_fn

    def make_residual_vector_fn_at_t(self, u_and_grad_fn, t_val: float):
        """Like :meth:`make_residual_vector_fn` but for a fixed time ``t_val``.

        For continuous-time (transient) problems the spatial cubature points
        ``cd['pts']`` have shape ``(F, Q, 2)``.  This method broadcasts them
        to ``(F, Q, 3)`` by appending the requested time level before calling
        the same assembly loop as :meth:`make_residual_vector_fn`.

        For static (non-random-time) problems the ``t_val`` argument is simply
        ignored and the function reduces to :meth:`make_residual_vector_fn`.

        Returns
        -------
        residual_fn : callable
            ``residual_fn(params) -> jnp.ndarray``  shape ``(n_dofs,)``
        """
        import jax
        import jax.numpy as jnp

        cd           = self.cubature_data
        pts_sp       = cd['pts']                       # (F, Q, 2)  spatial
        weights_jax  = jnp.asarray(cd['weights'],  dtype=jnp.float32)
        phi_jax      = jnp.asarray(cd['phi'],      dtype=jnp.float32)
        grad_phi_jax = jnp.asarray(cd['grad_phi'], dtype=jnp.float32)
        node_ids_jax = jnp.asarray(cd['node_ids'], dtype=jnp.int32)

        # Append time dimension for continuous-time domains
        _is_transient = getattr(self, '_random_time_sampling', False)
        if _is_transient:
            import numpy as _np
            t_col  = _np.full(pts_sp.shape[:2] + (1,), float(t_val), dtype=_np.float32)
            pts_jax = jnp.asarray(_np.concatenate([pts_sp, t_col], axis=-1), dtype=jnp.float32)
        else:
            pts_jax = jnp.asarray(pts_sp, dtype=jnp.float32)

        n_dofs  = self.n_dofs
        n_faces = pts_jax.shape[0]
        n_qpts  = pts_jax.shape[1]
        n_local = phi_jax.shape[2]
        volume_fn   = self.volume_fn
        params_dict = self._build_params()
        _n_pts_dims = pts_jax.shape[-1]

        def residual_fn(params):
            pts_flat = pts_jax.reshape(-1, _n_pts_dims)
            u_flat, grad_u_flat = jax.vmap(
                lambda xy: u_and_grad_fn(params, xy))(pts_flat)

            if u_flat.ndim == 1:
                y_flat = u_flat.reshape(-1, 1)
                def make_deriv(gu, gphi):
                    def deriv_fn(Y, X, component, order):
                        dim = order[0] if isinstance(order, (list, tuple)) else order
                        if Y.ndim == 1: return gphi[:, dim]  # derivative(phi,...)
                        return gu[:, dim]
                    return deriv_fn
            else:
                y_flat = u_flat
                def make_deriv(jac, gphi):
                    def deriv_fn(Y, X, component, order):
                        dim = order[0] if isinstance(order, (list, tuple)) else order
                        if Y.ndim == 1: return gphi[:, dim]  # derivative(phi,...)
                        return jac[:, component, dim]
                    return deriv_fn

            _gphi0 = grad_phi_jax[:, :, 0, :].reshape(-1, 2)
            R_sample = volume_fn(
                pts_flat, y_flat, params_dict,
                phi_jax[:, :, 0].reshape(-1),
                make_deriv(grad_u_flat, _gphi0),
            )
            _multi = isinstance(R_sample, (tuple, list))
            _n_comp = len(R_sample) if _multi else 1

            Rs = [jnp.zeros(n_dofs, dtype=jnp.float32) for _ in range(_n_comp)]
            for a in range(n_local):
                phi_a  = phi_jax[:, :, a]
                gphi_a = grad_phi_jax[:, :, a, :]
                gphi_a_flat = gphi_a.reshape(-1, 2)
                integrand = volume_fn(
                    pts_flat, y_flat, params_dict,
                    phi_a.reshape(-1),
                    make_deriv(grad_u_flat, gphi_a_flat),
                )
                if _multi:
                    for k, ig in enumerate(integrand):
                        elem_int = jnp.einsum('fq,fq->f', weights_jax, ig.reshape(n_faces, n_qpts))
                        Rs[k] = Rs[k].at[node_ids_jax[:, a]].add(elem_int)
                else:
                    elem_int = jnp.einsum('fq,fq->f', weights_jax, integrand.reshape(n_faces, n_qpts))
                    Rs[0] = Rs[0].at[node_ids_jax[:, a]].add(elem_int)

            return jnp.concatenate([R for R in Rs])

        return residual_fn

    # ---------------------------------------------------------------------- #
    #  LaTeX / display helpers                                               #
    # ---------------------------------------------------------------------- #

    def _latex_name(self, name: str) -> str:
        """Convert a user-facing name into a LaTeX-safe label."""
        if name is None:
            return "unnamed"
        name = str(name)
        name = name.replace("\\", r"\backslash ")
        name = name.replace("_", r"\_")
        name = name.replace(" ", r"\ ")
        return name

    def _is_lagrange(self, name: str) -> bool:
        """Check if a term name is in the lagrange_multipliers list."""
        if not self.lagrange_multipliers:
            return False
        return name in self.lagrange_multipliers

    def get_problem_latex(self, include_legend: bool = True) -> str:
        """
        Build a LaTeX representation of the weak-form optimization problem.

        The loss has the structure::

            min_theta  [max_lambda]  L(theta, lambda)

        where each term is either a weighted quadratic ``w/N ||R||^2``
        and/or an augmented-Lagrangian inner product
        ``(1/N) <lambda, R>`` depending on ``lagrange_multipliers``.

        The PDE term uses weak-form (integral) notation:

            (w_pde / N_e) sum_e (∫_Ke ∇u_h · ∇φ dK)^2

        Dirichlet BC terms use pointwise nodal residual notation.
        Neumann BC terms are annotated as natural (line-integral) conditions.

        Args:
            include_legend: If True, append a legend block describing each symbol.

        Returns:
            A LaTeX string suitable for ``IPython.display.Math``.
        """
        terms = []
        legend_terms = []
        has_any_al = False
        all_lambda_syms = []

        # ── PDE (volume) term ────────────────────────────────────────────
        pde_sym = self._latex_name("pde")
        is_al_pde = self._is_lagrange("pde")

        terms.append(
            rf"\frac{{w_{{{pde_sym}}}}}{{N_e}}"
            rf"\sum_e\!\left(\int_{{K_e}}\nabla u_h\cdot\nabla\phi\,dK\right)^2"
        )
        if is_al_pde:
            has_any_al = True
            all_lambda_syms.append(rf"\boldsymbol{{\lambda}}_{{{pde_sym}}}")
            terms.append(
                rf"\frac{{1}}{{N_e}}\langle\boldsymbol{{\lambda}}_{{{pde_sym}}},\,"
                rf"\mathbf{{R}}_{{{pde_sym}}}\rangle"
            )

        if include_legend:
            legend_terms.append(rf"K_e:\text{{ mesh element (triangle)}}")
            legend_terms.append(rf"N_e:\text{{ number of elements}}")
            legend_terms.append(rf"w_{{{pde_sym}}}:\text{{ PDE weight}}")
            if is_al_pde:
                legend_terms.append(
                    rf"\boldsymbol{{\lambda}}_{{{pde_sym}}}:\text{{ Lagrange multipliers for PDE}}"
                )

        # ── Boundary condition terms ─────────────────────────────────────
        # The quadratic ||B||^2 always appears (it is always in the loss).
        # The Lagrange inner product <λ, B> is added only when the BC name
        # is in lagrange_multipliers.
        # Hard-constrained BCs are satisfied by the network exactly and are
        # not part of the soft optimisation problem, so we skip them here.
        _hard_names = self.hard_bc_names | self._rollout_ic_bc_names
        for bc in self.domain.boundary_conditions:
            if getattr(bc, 'name', None) in _hard_names:
                continue
            raw_name = getattr(bc, 'name', None) or "bc"
            sym = self._latex_name(raw_name)
            is_al_bc = self._is_lagrange(raw_name)
            bc_type = getattr(bc, 'bc_type', 'dirichlet')

            if bc_type == 'neumann':
                # Natural BC — enters as a line integral on the RHS;
                # shown as a boundary residual term ||R_N||^2
                terms.append(
                    rf"\frac{{w_{{{sym}}}}}{{N_{{{sym}}}}}"
                    rf"\left\|\oint_{{\partial\Omega_{{{sym}}}}}"
                    rf"\frac{{\partial u_h}}{{\partial n}}\phi\,dS\right\|_2^2"
                )
            else:
                # Essential (Dirichlet) BC — pointwise nodal residual
                terms.append(
                    rf"\frac{{w_{{{sym}}}}}{{N_{{{sym}}}}}"
                    rf"\left\|\mathcal{{B}}_{{{sym}}}\right\|_2^2"
                )

            if is_al_bc:
                has_any_al = True
                all_lambda_syms.append(rf"\boldsymbol{{\lambda}}_{{{sym}}}")
                terms.append(
                    rf"\frac{{1}}{{N_{{{sym}}}}}\langle\boldsymbol{{\lambda}}_{{{sym}}},\,"
                    rf"\mathcal{{B}}_{{{sym}}}\rangle"
                )

            if include_legend:
                if bc_type == 'neumann':
                    legend_terms.append(
                        rf"\partial\Omega_{{{sym}}}:\text{{ Neumann boundary segment }}{sym}"
                    )
                else:
                    legend_terms.append(
                        rf"\mathcal{{B}}_{{{sym}}}:\text{{ Dirichlet residual on }}{sym}"
                    )
                legend_terms.append(rf"N_{{{sym}}}:\text{{ BC nodes for }}{sym}")
                legend_terms.append(rf"w_{{{sym}}}:\text{{ weight for }}{sym}")
                if is_al_bc:
                    legend_terms.append(
                        rf"\boldsymbol{{\lambda}}_{{{sym}}}:\text{{ Lagrange multipliers for }}{sym}"
                    )

        # ── Assemble objective ───────────────────────────────────────────
        obj_body = " + ".join(terms)
        if has_any_al:
            lam_vars = ",".join(all_lambda_syms)
            operator = rf"\min_{{\theta}}\max_{{{lam_vars}}}\;"
            objective = rf"\mathcal{{L}}(\theta,\boldsymbol{{\lambda}})={obj_body}"
        else:
            operator = rf"\min_{{\theta}}\;"
            objective = rf"\mathcal{{L}}(\theta)={obj_body}"

        lines = [rf"{operator}{objective}"]
        if include_legend and legend_terms:
            legend_block = (
                r" \\[4pt] \begin{array}{l} "
                + r" \\ ".join(legend_terms)
                + r" \end{array}"
            )
            lines.append(legend_block)

        return "".join(lines)

    def show_problem(self, include_legend: bool = True) -> str:
        """
        Display the weak-form optimization problem in LaTeX.

        Renders as a formatted math block in notebooks; falls back to
        printing the LaTeX string in plain Python sessions.

        Args:
            include_legend: If True, append a legend for all symbols.

        Returns:
            The generated LaTeX string.
        """
        latex = self.get_problem_latex(include_legend=include_legend)
        try:
            from IPython.display import Math, display
            display(Math(latex))
        except Exception:
            print(latex)
        return latex

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
