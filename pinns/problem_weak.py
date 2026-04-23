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

            volume_fn(x, y, params, phi, grad_phi, derivative=None) -> (n_pts,)

        Arguments (all JAX arrays):
          - ``x``          (n_pts, 2)       – quadrature point positions
          - ``y``          (n_pts, n_out)   – network output (same layout as strong form)
          - ``params``     dict             – ``{fixed, infer, internal}`` (same as strong form)
          - ``phi``        (n_pts,)         – test function :math:`\varphi_j` values
          - ``grad_phi``   (n_pts, 2)       – :math:`\nabla\varphi_j` in physical coords
          - ``derivative`` callable or None – ``derivative(y, x, comp, order)`` (same API as
            strong form; provided by the assembler)
        Returns: ``(n_pts,)`` per-quadrature-point integrand values.
    boundary_fn : callable or None
        Neumann boundary integrand.  If ``None`` all boundaries with
        ``add_neumann`` are treated as natural (zero-flux) — you still need
        to call ``add_neumann`` on the domain to mark the Neumann boundary
        segments.  If provided, same signature as ``volume_fn`` but with
        per-edge quadrature::

            boundary_fn(x, V, phi, normals, params) -> scalar_per_sample

          - ``normals``   (n_pts, 2)          – outward unit normals
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
    boundary_fn: Optional[Callable] = None
    params: Dict[str, Any] = field(default_factory=dict)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    output_range: Optional[Union[tuple, List[Optional[tuple]]]] = None
    cubature_order: int = 3
    lagrange_order: int = 1
    basis: str = "lagrange"
    solution: Optional[Callable] = None
    lagrange_multipliers: List[str] = field(default_factory=list)

    # ── filled by __post_init__ ──────────────────────────────────────────
    cubature_data:    Dict       = field(init=False, default_factory=dict)
    neumann_data:     List       = field(init=False, default_factory=list)
    free_nodes:       np.ndarray = field(init=False, default=None)
    dirichlet_nodes:  np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        from .domain import DomainMesh
        from .boundary import MeshNodeBC

        if not isinstance(self.domain, DomainMesh):
            raise TypeError(
                "ProblemWeak requires a DomainMesh domain; "
                f"got {type(self.domain).__name__}."
            )
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
        for bc in self.domain.boundary_conditions:
            if isinstance(bc, MeshNodeBC) and bc.bc_type == "dirichlet":
                if bc.edges is not None:
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
                    if bc.edges is not None:
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

        # Store free_mask in cubature_data for easy access
        self.cubature_data['free_mask'] = free_mask

        # ── Neumann boundary cubature ────────────────────────────────────
        self.neumann_data = []
        for bc in self.domain.boundary_conditions:
            if isinstance(bc, MeshNodeBC) and bc.bc_type == "neumann" and bc.edges is not None:
                data = _precompute_boundary_edges(
                    verts, bc.edges, bc.edge_normals, self.cubature_order
                )
                data['bc'] = bc
                self.neumann_data.append(data)

        # ── output_range ────────────────────────────────────────────────
        if self.output_range is not None:
            if (isinstance(self.output_range, tuple)
                    and len(self.output_range) == 2
                    and not isinstance(self.output_range[0], (list, tuple))):
                self.output_range = [self.output_range] * self.n_outputs

    # ── Convenience properties ───────────────────────────────────────────

    @property
    def n_free_nodes(self) -> int:
        """Number of free (non-Dirichlet) DOFs = number of test functions."""
        return len(self.free_nodes)

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

    def make_loss_fn(self, u_and_grad_fn):
        """
        Return a JAX-jittable ``loss_fn(params) -> scalar`` that assembles
        the full weak-form residual and returns the MSE over free nodes.

        Parameters
        ----------
        u_and_grad_fn : callable
            Single-point evaluator with signature::

                u_and_grad_fn(params, xy) -> (u_scalar, grad_u_2d)

            where ``xy`` is a 1-D array of shape ``(n_dims,)``,
            ``u_scalar`` is a scalar, and ``grad_u_2d`` has shape ``(n_dims,)``.
            Typically built with ``jax.value_and_grad``.

        Returns
        -------
        loss_fn : callable
            ``loss_fn(params) -> scalar`` — suitable for ``jax.jit`` and
            ``jax.value_and_grad``.
        """
        import jax
        import jax.numpy as jnp

        cd             = self.cubature_data
        pts_jax        = jnp.asarray(cd['pts'],      dtype=jnp.float32)   # (F, Q, 2)
        weights_jax    = jnp.asarray(cd['weights'],  dtype=jnp.float32)   # (F, Q)
        phi_jax        = jnp.asarray(cd['phi'],      dtype=jnp.float32)   # (F, Q, L)
        grad_phi_jax   = jnp.asarray(cd['grad_phi'], dtype=jnp.float32)   # (F, Q, L, 2)
        node_ids_jax   = jnp.asarray(cd['node_ids'], dtype=jnp.int32)     # (F, L)
        free_nodes_jax = jnp.asarray(self.free_nodes, dtype=jnp.int32)    # (n_free,)

        n_dofs  = self.n_dofs
        n_faces = pts_jax.shape[0]
        n_qpts  = pts_jax.shape[1]
        n_local = phi_jax.shape[2]          # (N+1)(N+2)/2
        volume_fn   = self.volume_fn
        params_dict = self._build_params()

        def loss_fn(params):
            pts_flat = pts_jax.reshape(-1, 2)                         # (F*Q, 2)

            # Evaluate u and ∇u at all quadrature points (one vmapped pass)
            u_flat, grad_u_flat = jax.vmap(
                lambda xy: u_and_grad_fn(params, xy))(pts_flat)
            y_flat = u_flat.reshape(-1, 1)                            # (F*Q, 1)

            # Derivative closure — same API as the strong-form
            def make_deriv(gu):
                def deriv_fn(Y, X, component, order):
                    dim = order[0] if isinstance(order, (list, tuple)) else order
                    return gu[:, dim]
                return deriv_fn

            deriv = make_deriv(grad_u_flat)

            # Assemble global residual: loop over n_local local DOFs per element
            R = jnp.zeros(n_dofs, dtype=jnp.float32)
            for a in range(n_local):
                phi_a  = phi_jax[:, :, a]                             # (F, Q)
                gphi_a = grad_phi_jax[:, :, a, :]                     # (F, Q, 2)

                integrand = volume_fn(
                    pts_flat,
                    y_flat,
                    params_dict,
                    phi_a.reshape(-1),
                    gphi_a.reshape(-1, 2),
                    deriv,
                )                                                     # (F*Q,)

                elem_int = jnp.einsum(
                    'fq,fq->f', weights_jax,
                    integrand.reshape(n_faces, n_qpts))
                R = R.at[node_ids_jax[:, a]].add(elem_int)

            return jnp.mean(R[free_nodes_jax] ** 2)

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
        pts_jax        = jnp.asarray(cd['pts'],      dtype=jnp.float32)
        weights_jax    = jnp.asarray(cd['weights'],  dtype=jnp.float32)
        phi_jax        = jnp.asarray(cd['phi'],      dtype=jnp.float32)
        grad_phi_jax   = jnp.asarray(cd['grad_phi'], dtype=jnp.float32)
        node_ids_jax   = jnp.asarray(cd['node_ids'], dtype=jnp.int32)

        n_dofs  = self.n_dofs
        n_faces = pts_jax.shape[0]
        n_qpts  = pts_jax.shape[1]
        n_local = phi_jax.shape[2]
        volume_fn   = self.volume_fn
        params_dict = self._build_params()

        def residual_fn(params):
            pts_flat = pts_jax.reshape(-1, 2)
            u_flat, grad_u_flat = jax.vmap(
                lambda xy: u_and_grad_fn(params, xy))(pts_flat)
            y_flat = u_flat.reshape(-1, 1)

            def make_deriv(gu):
                def deriv_fn(Y, X, component, order):
                    dim = order[0] if isinstance(order, (list, tuple)) else order
                    return gu[:, dim]
                return deriv_fn

            deriv = make_deriv(grad_u_flat)

            R = jnp.zeros(n_dofs, dtype=jnp.float32)
            for a in range(n_local):
                phi_a  = phi_jax[:, :, a]
                gphi_a = grad_phi_jax[:, :, a, :]
                integrand = volume_fn(
                    pts_flat, y_flat, params_dict,
                    phi_a.reshape(-1), gphi_a.reshape(-1, 2), deriv,
                )
                elem_int = jnp.einsum(
                    'fq,fq->f', weights_jax,
                    integrand.reshape(n_faces, n_qpts))
                R = R.at[node_ids_jax[:, a]].add(elem_int)
            return R

        return residual_fn

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
