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
    """Standard multi-layer perceptron with optional input LayerNorm."""
    features: Sequence[int]
    activation: str = 'tanh'
    normalize_input: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = get_activation(self.activation)
        if self.normalize_input:
            x = nn.LayerNorm()(x)
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
    Laplace-spectral PINN with two operating modes selected by ``use_state``.

    **Continuous-time mode** (``use_state=False``, default)
        Use with :class:`~pinns.DomainMeshContinuous`.  The encoder maps each
        query point ``(x, y, t)`` to ``[Φ(x,y), t]`` via barycentric
        interpolation of mesh eigenvectors, and a plain FNN predicts
        ``u(x, y, t)`` globally.  Time derivative follows by autodiff.

    **State-integrator / rollout mode** (``use_state=True``)
        Use with :class:`~pinns.DomainMeshDiscrete`.  The network is a learned
        one-step time integrator:

        1. **Spectral encode** the current nodal state:
           :math:`\\alpha^n = \\Phi^\\top u^n \\in \\mathbb{R}^K`
        2. **MLP** advances the coefficients:
           :math:`\\alpha^{n+1} = \\text{MLP}(\\alpha^n) \\in \\mathbb{R}^K`
        3. **Spectral decode** back to nodal values:
           :math:`u^{n+1} = \\Phi\\,\\alpha^{n+1} \\in \\mathbb{R}^{n_\\text{nodes}}`

        Trained end-to-end via BPTT (``jax.lax.scan`` over all time steps).
        The previous state :math:`u^n` is read from
        ``params_dict["fixed"]["u_prev_nodes"]`` at each rollout step.

    Parameters
    ----------
    domain : DomainMesh
        2-D triangular mesh domain.
    n_features : int
        Number of Laplace eigenvectors (= spectral bottleneck dimension).
        In continuous mode with a time interval: ``n_features - 1``
        eigenvectors + 1 time column = ``n_features`` total input features.
        In state mode: exactly ``n_features`` eigenvectors.
    hidden_dims : sequence of int
        Width of each hidden layer, e.g. ``[128, 128, 128, 128]``.
    activation : str
        Activation function (default ``'tanh'``).
    n_outputs : int
        Number of PDE unknowns per query point (default 1).
    output_transform : callable, optional
        Hard-constraint transform applied *after* network evaluation.
        Signature: ``f(x_original, y, params_dict) → y``.
        Obtain from ``ProblemWeak(..., hard_constraints=[...]).output_transform``.
        For ``use_state=True``, prefer ``hard_constraints=True`` instead.
    input_transform : callable, optional
        Applied to raw query coordinates *before* the feature encoding
        (continuous mode only).
    use_state : bool or ``None``
        ``None`` (default): auto-detect — ``True`` when *domain* is a
        :class:`~pinns.DomainMeshDiscrete`, ``False`` otherwise.
        Pass explicitly to override.
    hard_constraints : bool
        When ``True`` (default) and in state-integrator mode, Dirichlet BCs
        registered on *domain* are enforced after spectral decode via static
        condensation.
    residual : bool
        When ``True`` (default) and in state-integrator mode, use a residual
        (skip) connection: ``α^{n+1} = α^n + MLP(α^n)``.
        The MLP learns only the *increment* Δα, which keeps each Jacobian
        factor close to the identity and prevents gradient vanishing in BPTT.
    normalize_input : bool
        When ``True`` (default) and in state-integrator mode, apply
        ``LayerNorm`` to ``α^n`` before the first Dense layer.  The spectral
        coefficients span several orders of magnitude across modes, so
        normalisation gives the first layer uniform gradient scale.
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
        use_state: Optional[bool] = None,
        hard_constraints: bool = True,
        residual: bool = True,
        normalize_input: bool = True,
    ):
        if domain._spatial_dims != 2:
            raise NotImplementedError(
                "AlphaPINN supports 2-D spatial meshes only "
                f"(got spatial_dims={domain._spatial_dims})."
            )

        # Auto-detect mode from domain type
        if use_state is None:
            from pinns.domain import DomainMeshDiscrete
            use_state = isinstance(domain, DomainMeshDiscrete)

        self.n_features       = n_features
        self.hidden_dims      = list(hidden_dims)
        self.n_outputs        = n_outputs
        self.activation       = activation
        self.use_state        = use_state
        self.hard_constraints = hard_constraints
        self.residual         = residual
        self.normalize_input  = normalize_input
        self.spatial_dims     = domain._spatial_dims
        self.output_transform = output_transform

        if use_state:
            # ── State-integrator mode ─────────────────────────────────────
            # Purely spatial eigenvectors — no time column.
            verts = domain._vertices[:, :domain._spatial_dims].astype(np.float32)
            faces = domain._faces.astype(np.int32)
            self._nodes_np = verts
            self._faces_np = faces
            self.n_nodes   = len(verts)
            self.has_time  = False

            print(
                f"AlphaPINN (state-integrator): computing {n_features} "
                f"Laplace eigenvectors for mesh with {len(verts)} nodes … ",
                end="", flush=True,
            )
            _Phi = _compute_laplace_eigenvectors(verts, faces, n_features)
            self._Phi = jnp.array(_Phi, dtype=jnp.float32)   # (n_nodes, K)
            print("done.")

            # MLP: α^n ∈ R^{K*n_out} → α^{n+1} ∈ R^{K*n_out}
            self._state_module = _MLPModule(
                features        = list(hidden_dims) + [n_features * n_outputs],
                activation      = activation,
                normalize_input = normalize_input,
            )
            self._fnn               = None
            self.laplacian_features = None

            # Build BC masks (same logic as GNNMeshNetwork)
            bc_mask_np   = np.zeros((self.n_nodes, n_outputs), dtype=np.float32)
            bc_values_np = np.zeros((self.n_nodes, n_outputs), dtype=np.float32)
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
                    if bc.node_indices is not None:
                        node_idx = np.asarray(bc.node_indices).astype(np.int64)
                    elif bc.edges is not None:
                        node_idx = np.unique(bc.edges).astype(np.int64)
                    else:
                        continue
                    node_pos = verts[node_idx]
                    vals = bc.get_value(node_pos)
                    bc_mask_np[node_idx, comp]   = 1.0
                    bc_values_np[node_idx, comp] = vals
            self._bc_mask_jnp   = jnp.array(bc_mask_np,   dtype=jnp.float32)
            self._bc_values_jnp = jnp.array(bc_values_np, dtype=jnp.float32)

        else:
            # ── Continuous space-time mode ────────────────────────────────
            self.laplacian_features = LaplacianFeatures(domain, n_features)

            layer_sizes = [n_features] + list(hidden_dims) + [n_outputs]
            self._fnn = FNN(
                layer_sizes        = layer_sizes,
                activation         = activation,
                normalize_input    = False,
                unnormalize_output = False,
                feature_encoding   = self.laplacian_features,
                input_transform    = input_transform,
                output_transform   = output_transform,
            )
            self._state_module = None
            self.n_nodes       = self.laplacian_features._nodes_np.shape[0]
            self._nodes_np     = self.laplacian_features._nodes_np
            self._faces_np     = self.laplacian_features._faces_np
            self.has_time      = self.laplacian_features.has_time

    # ── Trainer-compatible API ────────────────────────────────────────── #

    def set_input_range(self, xmin: np.ndarray, xmax: np.ndarray):
        if self._fnn is not None:
            self._fnn.set_input_range(xmin, xmax)

    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        if self._fnn is not None:
            self._fnn.set_output_range(ymin, ymax)

    def init(self, rng: jax.random.PRNGKey, dummy_input=None) -> Dict:
        """Initialise network parameters."""
        if self.use_state:
            # MLP input dimension = n_features * n_outputs (flattened spectral vector)
            dummy = jnp.zeros((1, self.n_features * self.n_outputs), dtype=jnp.float32)
            return self._state_module.init(rng, dummy)
        if dummy_input is None:
            dummy_input = jnp.zeros((1, self.n_features), dtype=jnp.float32)
        return self._fnn._module.init(rng, dummy_input)

    def apply(
        self,
        params: Dict,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """
        Evaluate network at query points *x*.

        **Continuous mode** (``use_state=False``): delegates to inner FNN.

        **State-integrator mode** (``use_state=True``):

        1. Encode  ``α^n = Φᵀ u^n``  from ``params_dict["fixed"]["u_prev_nodes"]``.
        2. Advance ``α^{n+1} = MLP(α^n)``.
        3. Decode  ``u^{n+1} = Φ α^{n+1}``  → nodal values.
        4. Apply hard Dirichlet BCs (if any).
        5. Barycentric-interpolation at query points ``x``.
        """
        if not self.use_state:
            return self._fnn.apply(params, x, params_dict)

        # ── 1. Read previous nodal state u^n ─────────────────────────────
        # params_dict may be None when called from residual diagnostics (e.g.
        # Lagrange-mode initial residual check) — use zeros as a safe fallback.
        _fixed = (params_dict or {}).get("fixed") or {}
        _u_prev_raw = _fixed.get("u_prev_nodes", None)
        if _u_prev_raw is None:
            u_prev = jnp.zeros((self.n_nodes, self.n_outputs), dtype=jnp.float32)
        else:
            u_prev = jnp.asarray(_u_prev_raw, dtype=jnp.float32)   # (n_nodes,) or (n_nodes, n_out)
        if u_prev.ndim == 1:
            u_prev = u_prev[:, None]                            # (n_nodes, n_out)

        # ── 2. Spectral encode: α^n = Φᵀ u^n  →  (n_out, K) ─────────────
        # self._Phi: (n_nodes, K)
        alpha_prev = self._Phi.T @ u_prev                       # (K, n_out)
        alpha_flat = alpha_prev.ravel()[None, :]                # (1, K*n_out)

        # ── 3. MLP advance: α^{n+1} = MLP(α^n)  [+ residual skip] ───────
        delta = self._state_module.apply(params, alpha_flat)            # (1, K*n_out)
        alpha_next_flat = alpha_flat + delta if self.residual else delta
        alpha_next = alpha_next_flat[0].reshape(self.n_features, self.n_outputs)  # (K, n_out)

        # ── 4. Spectral decode: u^{n+1} = Φ α^{n+1}  →  (n_nodes, n_out) ─
        node_coeffs = self._Phi @ alpha_next                    # (n_nodes, n_out)

        # ── 5. Hard Dirichlet BCs (static condensation) ─────────────────────
        if self.hard_constraints:
            node_coeffs = (node_coeffs * (1.0 - self._bc_mask_jnp)
                           + self._bc_values_jnp * self._bc_mask_jnp)

        # ── 6. Barycentric interpolation at query points ──────────────────
        nodes_jnp = jnp.array(self._nodes_np)
        faces_jnp = jnp.array(self._faces_np)
        x_spatial = x[:, :self.spatial_dims]                   # (n_query, 2)
        y = _interpolate_mesh(node_coeffs, nodes_jnp, faces_jnp, x_spatial)
        # y: (n_query, n_outputs)

        # Optional output_transform (e.g. soft constraint lifts, symmetry)
        if self.output_transform is not None:
            y = self.output_transform(x, y, params_dict)

        return y

    def forward(self, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        return self.apply(self.params, x, params_dict)

    def predict(self, x_np: np.ndarray, params_dict=None) -> np.ndarray:
        return np.array(self.apply(self.params, jnp.array(x_np), params_dict))

    def predict_rollout(
        self,
        n_steps: int = None,
        u0: np.ndarray = None,
        dt: float = None,
    ) -> np.ndarray:
        """Run the state-integrator forward from *u0* for *n_steps* steps.

        Returns
        -------
        u_all : (n_steps+1, n_nodes) float32 numpy array
            Row 0 is the initial condition; rows 1‥n_steps are the predicted
            states after each time step.
        """
        if not self.use_state:
            raise RuntimeError(
                "predict_rollout is only available in state-integrator mode."
            )
        n_nodes = self.n_nodes
        if u0 is None:
            u0 = np.zeros(n_nodes, dtype=np.float32)
        if n_steps is None:
            n_steps = 1
        _dt = jnp.float32(dt if dt is not None else 0.0)
        _mesh = jnp.array(self._nodes_np)
        _params = self.params

        # Use a plain Python loop — predict_rollout is inference-only so
        # there is no need for jax.lax.scan, and a Python loop avoids
        # retracing shape errors when called at different curriculum stages.
        def _step(u_nodes):
            pdict = {"fixed": {
                "u_prev_nodes": u_nodes,
                "u_prev":       u_nodes,
                "dt":           _dt,
                "kappa":        jnp.float32(0.0),
                "t_cur":        jnp.float32(0.0),
            }}
            return self.apply(_params, _mesh, pdict)[:, 0]

        u_cur = jnp.array(u0, dtype=jnp.float32)
        # Apply hard BC mask to the initial condition so boundary nodes
        # already have their prescribed values at t=0 in the output trajectory.
        # Squeeze masks to (n_nodes,) to avoid broadcasting (n_nodes,)*(n_nodes,1)→(n_nodes,n_nodes).
        _mask_1d   = self._bc_mask_jnp[:, 0]    # (n_nodes,)
        _values_1d = self._bc_values_jnp[:, 0]  # (n_nodes,)
        u_cur = u_cur * (1.0 - _mask_1d) + _values_1d * _mask_1d
        u_all = [np.array(u_cur)]
        for _ in range(n_steps):
            u_cur = _step(u_cur)
            u_all.append(np.array(u_cur))
        return np.stack(u_all, axis=0)  # (n_steps+1, n_nodes)

    def to(self, device=None, dtype=None, seed: int = 0) -> 'AlphaPINN':
        if not hasattr(self, 'params') or self.params is None:
            self.params = self.init(jax.random.PRNGKey(seed))
        return self

    @property
    def mesh_nodes(self) -> np.ndarray:
        if self.use_state:
            return self._nodes_np.copy()
        return self.laplacian_features._nodes_np.copy()

    @property
    def eigenvectors(self) -> np.ndarray:
        """Laplace eigenvectors Φ, shape ``(n_nodes, K)``."""
        if self.use_state:
            return np.array(self._Phi)
        return np.array(self.laplacian_features._Phi)

    def __repr__(self) -> str:
        if self.use_state:
            return (
                f"AlphaPINN("
                f"n_nodes={self.n_nodes}, "
                f"n_eigenvectors={self.n_features}, "
                f"hidden_dims={self.hidden_dims}, "
                f"n_outputs={self.n_outputs}, "
                f"mode='state-integrator')"
            )
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
