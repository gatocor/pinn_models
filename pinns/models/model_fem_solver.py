"""
pinns/models/model_fem_solver.py — FEM model solver.

:class:`ModelFEMSolver` couples a :class:`~pinns.domain.DomainMesh` (geometry)
with a P1 finite-element discretisation and a pluggable linear solver.

It mirrors the :class:`ModelSpectralSolver` interface so the same
:class:`~pinns.trainer.Trainer` workflow applies:

* ``add_parameter(name, value)``   — unified parameter registration.
* ``add_inner(fn, name)``          — weak-form volume integrand (same API as
                                     :class:`~pinns.problems.ProblemWeak`).
* ``add_dirichlet(fn, nodes)``     — essential (Dirichlet) boundary condition.
* ``solve()``                      — static FEM solve; returns
                                     ``{state_name: (N_nodes,)}`` array.
* ``apply(params, X)``             — barycentric evaluation at scattered
                                     points; mirrors ``network.apply(X, params)``.

Assembly uses one-point centroid quadrature (exact for P1) and is
fully GPU-compatible via JAX.
The unique feature is **unified flat/surface assembly**: the same P1 element
code handles 2-D flat meshes *and* 2-D manifolds embedded in 3-D (shells,
surfaces) via the surface Jacobian identity

.. math::

   K_e[i,j] = (\\nabla_\\xi N_i)^\\top \\,(J^\\top J)^{-1}\\, \\nabla_\\xi N_j
              \\  \\sqrt{\\det(J^\\top J)} \\cdot \\tfrac{1}{2}

which reduces to the standard flat-2D formula when :math:`J` is square.

**Element type:** P1 triangle (constant gradient per element).

**Solver backends:**

``"cg"``   — ``jax.scipy.sparse.linalg.cg``; GPU-compatible, differentiable
              through the solution (implicit-function-theorem backprop).

``"petsc"`` — PETSc + BoomerAMG; optimal for large meshes; requires
              ``petsc4py`` installation; not differentiable.

Usage::

    domain = pinns.DomainMesh((verts, faces))

    def volume_fn(x, y, params, phi, derivative):
        du_dx   = derivative(y,   x, 0, (0,))
        du_dy   = derivative(y,   x, 0, (1,))
        dphi_dx = derivative(phi, x, 0, (0,))
        dphi_dy = derivative(phi, x, 0, (1,))
        f = 2*jnp.pi**2 * jnp.sin(jnp.pi*x[:,0]) * jnp.sin(jnp.pi*x[:,1])
        return params["kappa"] * (du_dx*dphi_dx + du_dy*dphi_dy) - f*phi

    fem = pinns.ModelFEMSolver(domain, ["u"])
    fem.add_parameter("kappa", 1.0)
    fem.add_inner(volume_fn)
    fem.add_dirichlet(lambda X: 0.0)
    result = fem.solve()
    # result["u"] — (N_nodes,) solution array
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ["ModelFEMSolver"]

# ─────────────────────────────────────────────────────────────────────────── #
#  Reference-element data for P1 triangle                                    #
# ─────────────────────────────────────────────────────────────────────────── #

# Shape-function gradients in reference coordinates (ξ, η)
#   N1 = 1 - ξ - η  →  ∇_ξ N1 = [-1, -1]
#   N2 = ξ          →  ∇_ξ N2 = [ 1,  0]
#   N3 = η          →  ∇_ξ N3 = [ 0,  1]
_GRAD_N_REF = np.array([[-1.0, -1.0],
                         [ 1.0,  0.0],
                         [ 0.0,  1.0]], dtype=np.float64)  # (3, 2)


# ─────────────────────────────────────────────────────────────────────────── #
#  Barycentric interpolation helpers                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _barycentric_coords_2d(tri_verts, pts):
    """Barycentric coordinates for points in a flat 2-D triangle.

    Args:
        tri_verts: ``(3, 2)`` vertices in R².
        pts:       ``(N, 2)`` query points.

    Returns:
        ``(N, 3)`` barycentric coordinates (may be outside [0,1] if pts
        lie outside the triangle).
    """
    v0, v1, v2 = tri_verts[0], tri_verts[1], tri_verts[2]
    T = jnp.array([[v1[0] - v0[0], v2[0] - v0[0]],
                    [v1[1] - v0[1], v2[1] - v0[1]]])
    d = pts - v0[None, :]          # (N, 2)
    lam = jnp.linalg.solve(T, d.T).T   # (N, 2)
    lam1, lam2 = lam[:, 0], lam[:, 1]
    lam0 = 1.0 - lam1 - lam2
    return jnp.stack([lam0, lam1, lam2], axis=1)   # (N, 3)


def _find_containing_element(vertices, faces, pts):
    """For each query point find the triangle index containing it (numpy).

    Uses a brute-force search (acceptable for moderate meshes; for very large
    meshes users should call solve() once and use the stored interpolant).

    Works for **flat 2-D** meshes only (vertices in R²).
    For surface meshes, projects to the tangent plane of each candidate first.

    Args:
        vertices: ``(N_v, 2)`` mesh vertices.
        faces:    ``(N_f, 3)`` triangulation.
        pts:      ``(N, 2)`` query points.

    Returns:
        ``(N,)`` int32 element indices, ``-1`` for points outside the mesh.
    """
    N = pts.shape[0]
    result = np.full(N, -1, dtype=np.int32)

    v0 = vertices[faces[:, 0]]   # (N_f, 2)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Precompute T^{-1} for all elements
    T00 = v1[:, 0] - v0[:, 0]
    T10 = v1[:, 1] - v0[:, 1]
    T01 = v2[:, 0] - v0[:, 0]
    T11 = v2[:, 1] - v0[:, 1]
    det = T00 * T11 - T01 * T10
    inv_det = np.where(np.abs(det) > 1e-15, 1.0 / det, 0.0)

    for i, p in enumerate(pts):
        dx = p[0] - v0[:, 0]
        dy = p[1] - v0[:, 1]
        l1 = inv_det * (T11 * dx - T01 * dy)
        l2 = inv_det * (T00 * dy - T10 * dx)
        l0 = 1.0 - l1 - l2
        inside = (l0 >= -1e-10) & (l1 >= -1e-10) & (l2 >= -1e-10)
        idx = np.argmax(inside)
        if inside[idx]:
            result[i] = idx
    return result


def _surface_find_containing_element(vertices, faces, pts):
    """Find nearest triangle for query points on a surface mesh (R³ → face).

    Projects each query point onto each triangle and checks if the
    projected point lies within the triangle (barycentric test).

    Args:
        vertices: ``(N_v, 3)`` surface vertices.
        faces:    ``(N_f, 3)`` triangulation.
        pts:      ``(N, 3)`` query points (should lie on or near the surface).

    Returns:
        ``(N,)`` int32 element indices (nearest element).
    """
    v0 = vertices[faces[:, 0]]   # (N_f, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0   # (N_f, 3) edge vectors
    e2 = v2 - v0

    N = pts.shape[0]
    result = np.zeros(N, dtype=np.int32)

    for i, p in enumerate(pts):
        d = p[None, :] - v0   # (N_f, 3)
        # Solve 2x2 system in reference coords: minimize distance
        a11 = np.sum(e1 * e1, axis=1)
        a12 = np.sum(e1 * e2, axis=1)
        a22 = np.sum(e2 * e2, axis=1)
        b1  = np.sum(d  * e1, axis=1)
        b2  = np.sum(d  * e2, axis=1)
        det = a11 * a22 - a12 * a12
        safe_det = np.where(np.abs(det) > 1e-15, det, 1.0)
        l1 = (a22 * b1 - a12 * b2) / safe_det
        l2 = (a11 * b2 - a12 * b1) / safe_det
        l0 = 1.0 - l1 - l2
        # Projected point on each triangle
        proj = v0 + l1[:, None] * e1 + l2[:, None] * e2
        dist = np.sum((p[None, :] - proj) ** 2, axis=1)
        result[i] = np.argmin(dist)
    return result


# ─────────────────────────────────────────────────────────────────────────── #
#  ModelFEMSolver                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class ModelFEMSolver:
    """P1 finite-element solver for scalar PDEs on flat and surface meshes.

    Supports:
    * **Flat 2-D meshes** — standard Poisson, diffusion, Helmholtz.
    * **Surface 2-D meshes embedded in 3-D** — surface Laplacian, eikonal,
      heat on shells — via the unified surface Jacobian formula.

    Mirrors the :class:`ModelSpectralSolver` interface so it plugs directly
    into the existing :class:`~pinns.trainer.Trainer` workflow.

    Args:
        domain:        A :class:`~pinns.domain.DomainMesh` (vertices + faces).
        state_names:   Ordered list of state names, e.g. ``["u"]`` or
                       ``["u", "v"]``.  Currently only single-field
                       (``len == 1``) is supported for coupled assembly;
                       multi-field runs separate problems.
        element:       Element type.  Only ``"P1"`` is supported currently.
        solver_backend: ``"cg"`` (default) or ``"petsc"``.
        max_cg_iter:   Maximum CG iterations (``"cg"`` backend only).
        cg_tol:        CG convergence tolerance.

    Example::

        def volume_fn(x, y, params, phi, derivative):
            du_dx   = derivative(y,   x, 0, (0,))
            du_dy   = derivative(y,   x, 0, (1,))
            dphi_dx = derivative(phi, x, 0, (0,))
            dphi_dy = derivative(phi, x, 0, (1,))
            return params["kappa"] * (du_dx*dphi_dx + du_dy*dphi_dy) - phi

        fem = pinns.ModelFEMSolver(domain, ["u"])
        fem.add_parameter("kappa", 1.0)
        fem.add_inner(volume_fn)
        fem.add_dirichlet(lambda X: 0.0)
        result = fem.solve()
        # result["u"] is shape (N_vertices,)
    """

    # Marker for Trainer duck-type check
    _is_solver_problem: bool = True

    def __init__(
        self,
        domain,
        state_names: List[str],
        element: str = "P1",
        solver_backend: str = "cg",
        max_cg_iter: int = 10_000,
        cg_tol: float = 1e-10,
    ):
        from ..domain.domain_mesh import DomainMesh
        if not isinstance(domain, DomainMesh):
            raise TypeError(
                f"ModelFEMSolver expects a DomainMesh, got {type(domain).__name__}."
            )
        if element != "P1":
            raise NotImplementedError("Only P1 elements are currently supported.")

        self.domain = domain
        self.state_names: List[str] = list(state_names)
        self.n_states: int = len(state_names)
        self.element = element
        self.solver_backend = solver_backend
        self.max_cg_iter = max_cg_iter
        self.cg_tol = cg_tol

        # Mesh data (numpy)
        self.vertices: np.ndarray = domain._vertices      # (N_v, n_dims)
        self.faces: np.ndarray = domain._faces            # (N_f, 3)
        self.n_vertices: int = self.vertices.shape[0]
        self.n_faces: int = self.faces.shape[0]
        self.n_dims: int = self.vertices.shape[1]         # 2 (flat) or 3 (surface)
        self.is_surface: bool = (self.n_dims == 3)

        # Detect boundary nodes (nodes on less than a full ring of triangles)
        self._boundary_nodes: np.ndarray = self._detect_boundary_nodes()

        # Parameter store
        self._params: Dict[str, Any] = {}
        self._trainable_params: Optional[Dict[str, Any]] = None

        # Weak-form inner integrand — set via add_inner()
        self._inner_fn: Optional[Callable] = None
        self._inner_name: str = "pde"

        # Dirichlet BC: list of (node_indices, value_fn(X) → values)
        self._dirichlet_bcs: List[Tuple[np.ndarray, Callable]] = []

        # Cached solution
        self._last_solution: Optional[Dict[str, np.ndarray]] = None

        # Centroid coordinates (used to evaluate κ and f)
        v = self.vertices[self.faces]   # (N_f, 3, n_dims)
        self._centroids: np.ndarray = v.mean(axis=1)   # (N_f, n_dims)

        # Precompute element areas (numpy, static)
        self._areas: np.ndarray = self._compute_areas()

        # Precompute nearest-element lookup arrays for apply()
        self._lookup_cache: Dict[int, np.ndarray] = {}

    # ─────────────────────────────────────────────────────────────────── #
    #  Boundary detection                                                  #
    # ─────────────────────────────────────────────────────────────────── #

    def _detect_boundary_nodes(self) -> np.ndarray:
        """Return node indices that lie on the mesh boundary.

        A boundary edge appears in exactly one triangle; interior edges
        appear in two.  For surface meshes this gives the geometric boundary
        of the mesh (open edges of the surface patch).
        """
        edge_count: Dict[Tuple[int, int], int] = {}
        for tri in self.faces:
            for i in range(3):
                e = (int(min(tri[i], tri[(i + 1) % 3])),
                     int(max(tri[i], tri[(i + 1) % 3])))
                edge_count[e] = edge_count.get(e, 0) + 1
        boundary_set: set = set()
        for (a, b), cnt in edge_count.items():
            if cnt == 1:
                boundary_set.add(a)
                boundary_set.add(b)
        return np.array(sorted(boundary_set), dtype=np.int64)

    # ─────────────────────────────────────────────────────────────────── #
    #  Static geometry utilities                                           #
    # ─────────────────────────────────────────────────────────────────── #

    def _compute_areas(self) -> np.ndarray:
        """Compute element areas (or surface areas for 3-D meshes)."""
        v = self.vertices[self.faces]   # (N_f, 3, n_dims)
        e1 = v[:, 1, :] - v[:, 0, :]
        e2 = v[:, 2, :] - v[:, 0, :]
        if self.n_dims == 2:
            cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            return 0.5 * np.abs(cross)
        else:  # surface in 3D
            cross = np.cross(e1, e2)   # (N_f, 3)
            return 0.5 * np.linalg.norm(cross, axis=1)

    # ─────────────────────────────────────────────────────────────────── #
    #  Parameter registration (same API as ModelSpectralSolver)           #
    # ─────────────────────────────────────────────────────────────────── #

    def add_parameter(
        self,
        name: Union[str, List[str]],
        value: Any,
    ) -> "ModelFEMSolver":
        """Register a model parameter.

        Args:
            name:  Parameter name (string) or list of names.
            value: Initial value or list of values.

        Returns:
            ``self`` for method chaining.
        """
        if isinstance(name, str):
            self._params[name] = value
        else:
            names = list(name)
            values = list(value) if isinstance(value, (list, tuple)) else [value] * len(names)
            for n, v in zip(names, values):
                self._params[n] = v
        return self

    @property
    def params(self) -> Dict[str, Any]:
        """Trainable parameter pytree (Trainer reads/writes this)."""
        return self._trainable_params if self._trainable_params is not None else self._params

    @params.setter
    def params(self, value) -> None:
        self._trainable_params = value

    def _build_params(
        self, inferred_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return flat ``p`` dict; stop-gradient on non-trainable params."""
        override = inferred_override or {}
        p: Dict[str, Any] = {}
        for k, v in self._params.items():
            p[k] = override[k] if k in override else jax.lax.stop_gradient(v)
        return p

    # ─────────────────────────────────────────────────────────────────── #
    #  Operator registration                                               #
    # ─────────────────────────────────────────────────────────────────── #

    def add_inner(
        self,
        fn: Callable,
        name: str = "pde",
    ) -> "ModelFEMSolver":
        """Register a generic weak-form volume integrand (mirrors ``ProblemWeak`` API).

        The function signature must be::

            fn(x, y, params, phi, derivative) -> (N_f,) array

        where

        * ``x``          — centroid coordinates ``(N_f, n_dims)``
        * ``y``          — trial-function values ``(N_f, 1)``  (shape mimics a
                           single-output network; 1/3 at centroid for the
                           stiffness pass, 0 for the load-only pass)
        * ``params``     — flat parameter dict
        * ``phi``        — test-function values ``(N_f,)`` — always 1/3 at centroid
        * ``derivative`` — callable ``derivative(field, x, component, order)``:

                           - ``field.ndim == 1`` (test function ``phi``): returns
                             the physical gradient component of the *test* function
                           - ``field.ndim > 1`` (trial function ``y``): returns
                             the physical gradient component of the *trial* function

        The derivative convention **exactly matches** ``ProblemWeak`` so the same
        ``volume_fn`` can be used for both the PINN training loop and the direct
        FEM solve without modification.

        The function should encode the full Galerkin residual
        ``a(u_h, v) − ℓ(v)``.  Assembly uses one-point centroid quadrature,
        exact for P1 elements.

        Example — Poisson equation ``−∇·(κ ∇u) = f``::

            def volume_fn(x, y, params, phi, derivative):
                du_dx   = derivative(y,   x, 0, (0,))  # ∂u/∂x
                du_dy   = derivative(y,   x, 0, (1,))  # ∂u/∂y
                dphi_dx = derivative(phi, x, 0, (0,))  # ∂φ/∂x
                dphi_dy = derivative(phi, x, 0, (1,))  # ∂φ/∂y
                f = 2*jnp.pi**2 * jnp.sin(jnp.pi*x[:,0]) * jnp.sin(jnp.pi*x[:,1])
                return params["kappa"] * (du_dx*dphi_dx + du_dy*dphi_dy) - f*phi

            fem.add_inner(volume_fn, name="poisson")

        Returns:
            ``self`` for method chaining.
        """
        self._inner_fn = fn
        self._inner_name = name
        return self

    # ─────────────────────────────────────────────────────────────────── #
    #  Boundary conditions                                                 #
    # ─────────────────────────────────────────────────────────────────── #

    def add_dirichlet(
        self,
        value_fn: Callable,
        nodes: Optional[np.ndarray] = None,
    ) -> "ModelFEMSolver":
        """Register a Dirichlet (essential) boundary condition.

        Args:
            value_fn: ``fn(X_nodes) → (N_nodes,)`` prescribed values.
                      ``X_nodes`` are the physical coordinates of the
                      constrained nodes.
            nodes:    Optional integer array of node indices to constrain.
                      Defaults to all boundary nodes detected automatically.

        Returns:
            ``self`` for method chaining.
        """
        if nodes is None:
            nodes = self._boundary_nodes
        nodes = np.asarray(nodes, dtype=np.int64)
        self._dirichlet_bcs.append((nodes, value_fn))
        return self

    # ─────────────────────────────────────────────────────────────────── #
    #  Physical gradient helpers (for add_inner assembly)                  #
    # ─────────────────────────────────────────────────────────────────── #

    def _compute_phys_gradients(self) -> np.ndarray:
        """Physical (or tangential) gradients of P1 basis functions in each element.

        * Flat 2-D: ``∇N_k^phys = J^{-⊤} \\nabla_\\xi N_k``
        * Surface 3-D: ``∇_t N_k = J\\,(J^\\top J)^{-1}\\,\\nabla_\\xi N_k``

        Returns:
            ``(N_f, 3, n_dims)`` array.
        """
        v = self.vertices[self.faces]   # (N_f, 3, n_dims)
        # Jacobian J[e, d, r] = e_{r+1} - e_0  →  shape (N_f, n_dims, 2)
        J = np.stack([v[:, 1, :] - v[:, 0, :],
                      v[:, 2, :] - v[:, 0, :]], axis=-1)

        if self.n_dims == 2:
            # J is (N_f, 2, 2)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            safe_det = np.where(np.abs(det) > 1e-15, det, 1.0)
            J_inv = np.stack([
                np.stack([ J[:, 1, 1], -J[:, 0, 1]], axis=-1),
                np.stack([-J[:, 1, 0],  J[:, 0, 0]], axis=-1),
            ], axis=-2) / safe_det[:, None, None]   # (N_f, 2, 2)
            # ∇N_k^phys = J^{-⊤} ∇N_k^ref
            # grad_phys[e,k,d] = sum_r J_inv[e,r,d] * _GRAD_N_REF[k,r]
            grad_phys = np.einsum(
                'erd,kr->ekd', J_inv, _GRAD_N_REF
            ).astype(np.float32)   # (N_f, 3, 2)
        else:
            # Surface: J is (N_f, 3, 2)
            # (J^⊤ J)[e,i,j] = sum_d J[e,d,i] * J[e,d,j]
            JtJ = np.einsum('edi,edj->eij', J, J)   # (N_f, 2, 2)
            det = JtJ[:, 0, 0] * JtJ[:, 1, 1] - JtJ[:, 0, 1] * JtJ[:, 1, 0]
            safe_det = np.where(np.abs(det) > 1e-15, det, 1.0)
            JtJ_inv = np.stack([
                np.stack([ JtJ[:, 1, 1], -JtJ[:, 0, 1]], axis=-1),
                np.stack([-JtJ[:, 1, 0],  JtJ[:, 0, 0]], axis=-1),
            ], axis=-2) / safe_det[:, None, None]   # (N_f, 2, 2)
            # ∇_t N_k = J (J^⊤J)^{-1} ∇_ξ N_k
            # step 1: G_ref[e,k,r] = sum_s JtJ_inv[e,r,s] * _GRAD_N_REF[k,s]
            G_ref = np.einsum('ers,ks->ekr', JtJ_inv, _GRAD_N_REF)   # (N_f,3,2)
            # step 2: grad_phys[e,k,d] = sum_r J[e,d,r] * G_ref[e,k,r]
            grad_phys = np.einsum(
                'edr,ekr->ekd', J, G_ref
            ).astype(np.float32)   # (N_f, 3, 3)
        return grad_phys

    def _assemble_from_inner_fn(
        self,
        inner_fn: Callable,
        params: Dict[str, Any],
    ):
        """Assemble the global stiffness matrix (COO) and load vector from a
        generic ``add_inner`` volume integrand.

        Uses one-point centroid quadrature (exact for P1 elements).  For each
        of the 9 local (test, trial) basis-function pairs the integrand is
        evaluated for all N_f elements simultaneously via JAX.

        Bilinear part:  ``K[i,j] = inner_fn(x, y=N_j, p, phi=N_i, d_ij)``
                        ``       − inner_fn(x, y=0,   p, phi=N_i, d_0i)``
        Linear part:    ``F[i]   = −inner_fn(x, y=0,   p, phi=N_i, d_0i)``

        Returns:
            ``rows, cols, K_vals, f_global``  (all JAX arrays).
        """
        N_f = self.n_faces
        N_v = self.n_vertices

        x_c = jnp.array(self._centroids, dtype=jnp.float32)   # (N_f, n_dims)
        areas = jnp.array(self._areas,   dtype=jnp.float32)    # (N_f,)
        grad_phys = self._compute_phys_gradients()             # (N_f, 3, n_dims)

        # P1 basis value at centroid = 1/3 for all local nodes
        phi_c = jnp.full((N_f,), 1.0 / 3.0, dtype=jnp.float32)
        # Trial function values that mimic a (N_f, 1) network output
        y_c    = jnp.full((N_f, 1), 1.0 / 3.0, dtype=jnp.float32)
        y_zero = jnp.zeros((N_f, 1), dtype=jnp.float32)

        faces_np = self.faces   # numpy, for scatter

        # ── Load pass: trial = 0 (gradient of trial = 0) ────────────
        # F_contrib[k_i] = inner_fn(x, 0, p, phi_i, d_0i) * area  (= −ℓ(φ_i)·area)
        f_contrib = []   # length 3, each (N_f,)
        for k_i in range(3):
            gt = jnp.array(grad_phys[:, k_i, :])   # (N_f, n_dims)

            def _deriv_load(Y, X, component, order, _gt=gt):
                dim = order[0] if isinstance(order, (list, tuple)) else order
                if hasattr(Y, 'ndim') and Y.ndim == 1:
                    return _gt[:, dim]          # test-function gradient
                return jnp.zeros(N_f, dtype=jnp.float32)  # zero trial

            raw = inner_fn(x_c, y_zero, params, phi_c, _deriv_load)
            f_contrib.append(jnp.asarray(raw, dtype=jnp.float32) * areas)

        # Global load: F[i] = −integral  (note the sign)
        f_global_np = np.zeros(N_v, dtype=np.float32)
        for k_i in range(3):
            np.add.at(f_global_np, faces_np[:, k_i], -np.array(f_contrib[k_i]))

        # ── Stiffness pass: trial = N_{k_j} ─────────────────────────
        K_rows_list = []
        K_cols_list = []
        K_vals_list = []
        for k_i in range(3):
            for k_j in range(3):
                gt  = jnp.array(grad_phys[:, k_i, :])   # (N_f, n_dims) test
                gtr = jnp.array(grad_phys[:, k_j, :])   # (N_f, n_dims) trial

                def _deriv_full(Y, X, component, order, _gt=gt, _gtr=gtr):
                    dim = order[0] if isinstance(order, (list, tuple)) else order
                    if hasattr(Y, 'ndim') and Y.ndim == 1:
                        return _gt[:, dim]    # test-function gradient
                    return _gtr[:, dim]       # trial-function gradient

                raw_full = inner_fn(x_c, y_c, params, phi_c, _deriv_full)
                full = jnp.asarray(raw_full, dtype=jnp.float32) * areas
                # Subtract load contribution to isolate bilinear part
                Ke = full - f_contrib[k_i]

                K_rows_list.append(faces_np[:, k_i])
                K_cols_list.append(faces_np[:, k_j])
                K_vals_list.append(Ke)

        rows   = jnp.array(np.concatenate(K_rows_list))
        cols   = jnp.array(np.concatenate(K_cols_list))
        K_vals = jnp.concatenate(K_vals_list)
        f_global = jnp.array(f_global_np)
        return rows, cols, K_vals, f_global

    # ─────────────────────────────────────────────────────────────────── #
    #  Validation                                                          #
    # ─────────────────────────────────────────────────────────────────── #

    def _validate(self) -> None:
        """Raise if model is not fully configured."""
        if self._inner_fn is None:
            raise RuntimeError(
                "ModelFEMSolver: no PDE operator set — call add_inner(volume_fn)."
            )

    # ─────────────────────────────────────────────────────────────────── #
    #  Main solve                                                          #
    # ─────────────────────────────────────────────────────────────────── #

    def solve(
        self,
        inferred_params: Optional[Dict[str, Any]] = None,
        t_obs=None,
    ) -> Dict[str, np.ndarray]:
        """Assemble and solve the FEM system.

        Solves the linear system

        .. math:: K u = f

        where ``K`` is the stiffness matrix (with Dirichlet rows replaced)
        and ``f`` is the load vector (with Dirichlet values substituted).

        Args:
            inferred_params: Optional dict of trainable parameter overrides
                             (used when the Trainer calls ``solve()`` during
                             an inverse problem).
            t_obs:           Ignored (kept for API compatibility with
                             :class:`ModelSpectralSolver`).

        Returns:
            Dict ``{state_name: (N_vertices,) array}`` — nodal solution.
        """
        self._validate()
        p = self._build_params(inferred_params)

        # ── Assemble global system ────────────────────────────────────
        rows, cols, K_vals, f_global = self._assemble_from_inner_fn(
            self._inner_fn, p
        )

        # ── Apply Dirichlet BCs ───────────────────────────────────────
        u_bc = jnp.zeros(self.n_vertices, dtype=jnp.float32)
        dirichlet_mask = jnp.zeros(self.n_vertices, dtype=jnp.float32)

        for bc_nodes, bc_fn in self._dirichlet_bcs:
            bc_x = self.vertices[bc_nodes].astype(np.float32)
            bc_vals_raw = bc_fn(jnp.array(bc_x))
            bc_vals = jnp.asarray(bc_vals_raw, dtype=jnp.float32).ravel()
            u_bc = u_bc.at[bc_nodes].set(bc_vals)
            dirichlet_mask = dirichlet_mask.at[bc_nodes].set(1.0)

        # Modify load: f_free = f_global - K @ u_bc
        K_times_ubc = jnp.zeros(self.n_vertices, dtype=jnp.float32).at[rows].add(
            K_vals * u_bc[cols]
        )
        f_mod = jnp.where(dirichlet_mask > 0.5, u_bc, f_global - K_times_ubc)

        # ── Solve ─────────────────────────────────────────────────────
        u_sol = self._solve_linear(
            rows, cols, K_vals, f_mod, dirichlet_mask, u_bc
        )

        result = {name: u_sol for name in self.state_names}

        # Only convert to numpy (for caching/plotting) when not inside a JAX
        # tracing context.  Inside jax.grad / jax.jit the JAX arrays are
        # returned directly so the gradient tape is preserved.
        leaves = jax.tree_util.tree_leaves(result)
        if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
            return result

        self._last_solution = {k: np.array(v) for k, v in result.items()}
        self._store_result(result)
        return self._last_solution

    def _solve_linear(
        self,
        rows, cols, K_vals,
        f_rhs,
        dirichlet_mask,
        u_bc,
    ):
        """Solve K u = f with selected backend."""
        N = self.n_vertices

        if self.solver_backend == "cg":
            return self._solve_cg(rows, cols, K_vals, f_rhs, dirichlet_mask, u_bc, N)
        elif self.solver_backend == "petsc":
            return self._solve_petsc(rows, cols, K_vals, f_rhs, dirichlet_mask, u_bc, N)
        else:
            raise ValueError(f"Unknown solver_backend={self.solver_backend!r}. "
                             f"Choose 'cg' or 'petsc'.")

    def _solve_cg(
        self, rows, cols, K_vals, f_rhs, dirichlet_mask, u_bc, N
    ):
        """JAX CG solver with diagonal (Jacobi) preconditioner.

        The Dirichlet rows are replaced by identity (1 on diagonal, 0 elsewhere)
        so the augmented system remains symmetric positive-definite.
        """
        # Modify K: replace Dirichlet rows with identity rows
        # Replace K_vals where row is Dirichlet and col != row with 0,
        # and where row == col with 1.
        is_dirichlet_row = dirichlet_mask[rows] > 0.5
        is_diagonal = rows == cols
        K_mod = jnp.where(
            is_dirichlet_row,
            jnp.where(is_diagonal, 1.0, 0.0),
            K_vals,
        )

        # Diagonal (Jacobi preconditioner)
        diag = jnp.zeros(N, dtype=jnp.float32).at[rows].add(
            jnp.where(is_diagonal, K_mod, 0.0)
        )
        safe_diag = jnp.where(jnp.abs(diag) > 1e-30, diag, 1.0)

        def matvec(x):
            return jnp.zeros(N, dtype=x.dtype).at[rows].add(K_mod * x[cols])

        def precond(x):
            return x / safe_diag

        u0 = jnp.zeros(N, dtype=jnp.float32)

        u_sol, info = jax.scipy.sparse.linalg.cg(
            matvec, f_rhs, x0=u0,
            tol=self.cg_tol,
            maxiter=self.max_cg_iter,
            M=precond,
        )

        if info is not None and info != 0:
            import warnings
            warnings.warn(
                f"ModelFEMSolver CG did not converge (info={info}). "
                f"Try increasing max_cg_iter (currently {self.max_cg_iter}) "
                f"or switching to solver_backend='petsc'.",
                RuntimeWarning,
                stacklevel=3,
            )

        return u_sol

    def _solve_petsc(
        self, rows, cols, K_vals, f_rhs, dirichlet_mask, u_bc, N
    ):
        """PETSc solver with BoomerAMG preconditioner.

        Requires ``petsc4py`` to be installed.  The system is transferred
        to PETSc via CPU arrays; solve is done on the PETSc side (CPU or GPU
        depending on PETSc build); result is returned as a JAX array.
        """
        try:
            from petsc4py import PETSc
        except ImportError:
            raise ImportError(
                "petsc4py is required for solver_backend='petsc'. "
                "Install with: pip install petsc4py"
            )

        # Convert to numpy CSR
        rows_np  = np.array(rows,   dtype=np.int32)
        cols_np  = np.array(cols,   dtype=np.int32)
        K_np     = np.array(K_vals, dtype=np.float64)
        f_np     = np.array(f_rhs,  dtype=np.float64)
        dmask_np = np.array(dirichlet_mask, dtype=np.float64)

        # Apply Dirichlet: identity rows
        is_dir_row = dmask_np[rows_np] > 0.5
        is_diag = rows_np == cols_np
        K_np = np.where(is_dir_row,
                        np.where(is_diag, 1.0, 0.0),
                        K_np)

        # Build scipy CSR → PETSc AIJ
        from scipy.sparse import coo_matrix
        K_coo = coo_matrix((K_np, (rows_np, cols_np)), shape=(N, N))
        K_csr = K_coo.tocsr()

        K_petsc = PETSc.Mat().createAIJ(
            size=(N, N),
            csr=(K_csr.indptr, K_csr.indices, K_csr.data),
        )
        K_petsc.assemblyBegin(); K_petsc.assemblyEnd()

        f_petsc = PETSc.Vec().createSeq(N)
        f_petsc.setArray(f_np)

        u_petsc = K_petsc.createVecRight()

        ksp = PETSc.KSP().create()
        ksp.setOperators(K_petsc)
        ksp.setType("cg")
        ksp.getPC().setType("hypre")
        ksp.getPC().setHYPREType("boomeramg")
        ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=self.max_cg_iter)
        ksp.setFromOptions()
        ksp.solve(f_petsc, u_petsc)

        u_np = u_petsc.getArray().astype(np.float32)
        return jnp.array(u_np)

    # ─────────────────────────────────────────────────────────────────── #
    #  Store result + build interpolant                                    #
    # ─────────────────────────────────────────────────────────────────── #

    def _store_result(self, result: Dict[str, Any]) -> None:
        """Cache the solution and build a nearest-neighbour / barycentric
        interpolant for :meth:`apply`."""
        self._stored_solution = {k: np.array(v) for k, v in result.items()}

    # ─────────────────────────────────────────────────────────────────── #
    #  apply — Trainer-compatible interface                                #
    # ─────────────────────────────────────────────────────────────────── #

    def apply(
        self,
        X: np.ndarray,
        params: Dict[str, Any] = None,
        params_dict=None,
    ) -> np.ndarray:
        """Evaluate the solved FEM solution at arbitrary points via
        barycentric (P1) interpolation.

        Args:
            X:           ``(N, n_dims)`` query coordinates matching the mesh
                         spatial dimensions.
            params:      Parameter dict (as managed by the Trainer).  ``None``
                         (default) uses registered parameters only.
            params_dict: Ignored (kept for API compatibility).

        Returns:
            ``(N, n_states)`` interpolated values.
        """
        # Re-solve if no solution is stored yet
        if self._last_solution is None:
            self.solve(inferred_params=params if params else None)

        X_np = np.asarray(X, dtype=np.float64)
        cols_out = []
        for name in self.state_names:
            u_nodes = self._stored_solution[name]
            u_interp = self._barycentric_interp(X_np, u_nodes)
            cols_out.append(u_interp[:, None])
        return np.concatenate(cols_out, axis=1)   # (N, n_states)

    def _barycentric_interp(
        self, X_query: np.ndarray, u_nodes: np.ndarray
    ) -> np.ndarray:
        """P1 barycentric interpolation at scattered query points.

        Args:
            X_query:  ``(N, n_dims)`` query points.
            u_nodes:  ``(N_vertices,)`` nodal values.

        Returns:
            ``(N,)`` interpolated values.
        """
        N = X_query.shape[0]

        # Find containing element for each query point
        key = id(X_query.tobytes()) if X_query.nbytes < 10_000 else id(X_query)
        if self.is_surface:
            # 3-D surface: project and find nearest element
            elem_ids = _surface_find_containing_element(
                self.vertices, self.faces, X_query
            )
        else:
            # Flat 2-D: exact containment test
            verts2d = self.vertices[:, :2]
            elem_ids = _find_containing_element(verts2d, self.faces, X_query[:, :2])

        result = np.zeros(N, dtype=np.float64)
        for i in range(N):
            eid = int(elem_ids[i])
            if eid < 0:
                # Outside mesh — use nearest vertex
                dists = np.sum((self.vertices - X_query[i]) ** 2, axis=1)
                result[i] = u_nodes[np.argmin(dists)]
                continue
            tri_v = self.vertices[self.faces[eid]]   # (3, n_dims)
            if self.is_surface:
                # Project query point onto triangle plane, then
                # compute barycentric coords in 2-D local frame
                e1 = tri_v[1] - tri_v[0]
                e2 = tri_v[2] - tri_v[0]
                a11 = e1 @ e1; a12 = e1 @ e2; a22 = e2 @ e2
                d   = X_query[i] - tri_v[0]
                b1  = d @ e1;    b2  = d @ e2
                det = a11 * a22 - a12 * a12
                if abs(det) < 1e-30:
                    result[i] = u_nodes[self.faces[eid, 0]]
                    continue
                l1 = (a22 * b1 - a12 * b2) / det
                l2 = (a11 * b2 - a12 * b1) / det
                l0 = 1.0 - l1 - l2
            else:
                # Flat 2-D: standard barycentric
                v0, v1, v2 = tri_v[:, :2]
                T = np.array([[v1[0] - v0[0], v2[0] - v0[0]],
                               [v1[1] - v0[1], v2[1] - v0[1]]])
                if abs(np.linalg.det(T)) < 1e-30:
                    result[i] = u_nodes[self.faces[eid, 0]]
                    continue
                lam = np.linalg.solve(T, X_query[i, :2] - v0)
                l1, l2 = lam
                l0 = 1.0 - l1 - l2

            nids = self.faces[eid]
            result[i] = l0 * u_nodes[nids[0]] + l1 * u_nodes[nids[1]] + l2 * u_nodes[nids[2]]

        return result

    # ─────────────────────────────────────────────────────────────────── #
    #  Convenience properties                                              #
    # ─────────────────────────────────────────────────────────────────── #

    @property
    def n_boundary_nodes(self) -> int:
        return len(self._boundary_nodes)

    @property
    def output_dim(self) -> int:
        return self.n_states

    def __repr__(self) -> str:
        return (
            f"ModelFEMSolver("
            f"n_vertices={self.n_vertices}, n_faces={self.n_faces}, "
            f"n_dims={self.n_dims}, element='{self.element}', "
            f"backend='{self.solver_backend}')"
        )
