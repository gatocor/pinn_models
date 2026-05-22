"""
pinns/models/model_solver.py — Spectral model solver.

:class:`ModelSpectralSolver` couples a :class:`~pinns.domain.DomainCubic` (geometry)
with a spectral discretisation (``shape``, ``bc``) and an owned integrator.
The spectral infrastructure — grid nodes, wavenumbers, K², and forward/inverse
transforms — lives directly on the model.

* ``add_parameter(name, value)``  — unified parameter registration.
* ``solve()``                     — direct forward solve.
* ``apply(params, X)``           — differentiable evaluation at scattered
                                   ``(x, t)`` points; mirrors ``network.apply``.

Operator function signatures use a **flat** ``p`` dict::

    model.set_linear_op(
        lambda X, p: {"u": 1j * p["mu"]**2 * X * (X**2)}
    )
    def nonlinear(X, U, p):
        u = model.inverse(U["u"])
        return {"u": -1j * p["eta"] / 2 * X * model.forward(u*u)}

Usage::

    domain = pinns.DomainCubic(space=[(-1.0, 1.0)], time=(0.0, 1.0))
    integrator = pinns.IntegratorETD2RK(dt=5e-4, checkpoint=True)
    model = pinns.ModelSpectralSolver(domain, ["u"], integrator, shape=64)
    model.set_linear_op(lambda X, p: {"u": 1j * p["mu"]**2 * X * (X**2)})
    model.set_nonlinear_op(nonlinear)
    model.add_parameter("eta", eta_val)
    model.add_parameter("mu",  0.02)
    model.add_initial(jnp.cos(jnp.pi * model.x))

    # Direct solve (observation times from domain + dt)
    U = model.solve()

    # Differentiable apply (for Dataset-based inverse)
    X_obs = ...  # (N, 2) columns [x, t]
    V = model.apply(X_obs, {"mu": jnp.array(0.02)})  # (N, 1)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

__all__ = ["ModelSpectralSolver"]

# ─────────────────────────────────────────────────────────────────────────── #
#  JAX bilinear interpolation (differentiable, works inside jax.grad)        #
# ─────────────────────────────────────────────────────────────────────────── #

def _jax_interp_1d(U, t_grid, x_grid, t_pts, x_pts):
    """Differentiable bilinear interpolation on a regular (t, x) grid.

    Gradients flow through ``U`` (the model output), making the full chain
    ``params → solve → U → interp → loss`` differentiable.

    Args:
        U:       ``(Nt, Nx)`` JAX array (physical-space state).
        t_grid:  ``(Nt,)`` observation times (JAX or numpy).
        x_grid:  ``(Nx,)`` spatial nodes (JAX or numpy).
        t_pts:   ``(N,)`` query times.
        x_pts:   ``(N,)`` query spatial coords.

    Returns:
        ``(N,)`` interpolated values.
    """
    import jax.numpy as jnp

    Nt, Nx = U.shape

    # ── t fractional index ────────────────────────────────────────────────
    i0 = jnp.clip(
        jnp.searchsorted(t_grid, t_pts, side="right") - 1, 0, Nt - 2
    )
    dt_i = t_grid[i0 + 1] - t_grid[i0]
    ft = jnp.clip((t_pts - t_grid[i0]) / jnp.where(dt_i > 0, dt_i, 1.0), 0.0, 1.0)

    # ── x fractional index ────────────────────────────────────────────────
    j0 = jnp.clip(
        jnp.searchsorted(x_grid, x_pts, side="right") - 1, 0, Nx - 2
    )
    dx_j = x_grid[j0 + 1] - x_grid[j0]
    fx = jnp.clip((x_pts - x_grid[j0]) / jnp.where(dx_j > 0, dx_j, 1.0), 0.0, 1.0)

    # ── bilinear weights ──────────────────────────────────────────────────
    return (
        (1 - ft) * (1 - fx) * U[i0,     j0    ] +
        (1 - ft) *      fx  * U[i0,     j0 + 1] +
             ft  * (1 - fx) * U[i0 + 1, j0    ] +
             ft  *      fx  * U[i0 + 1, j0 + 1]
    )


def _compute_stencil_1d(t_grid, x_grid, t_pts, x_pts):
    """Precompute bilinear interpolation stencil as numpy arrays.

    Returns integer indices and float weights that can be passed to
    :func:`_jax_apply_stencil_1d`.  Computing the stencil once in numpy
    (outside the JAX/autodiff trace) eliminates ``searchsorted`` overhead
    from every gradient step.

    Args:
        t_grid: ``(Nt,)`` numpy array of observation times.
        x_grid: ``(Nx,)`` numpy array of spatial nodes.
        t_pts:  ``(N,)`` numpy array of query times.
        x_pts:  ``(N,)`` numpy array of query spatial coords.

    Returns:
        Tuple ``(i0, j0, w00, w01, w10, w11)`` — all ``(N,)`` numpy arrays.
        ``i0, j0`` are int32 lower-left grid indices; ``w*`` are float32 weights.
    """
    import numpy as np
    Nt = len(t_grid)
    Nx = len(x_grid)

    i0 = np.clip(np.searchsorted(t_grid, t_pts, side="right") - 1, 0, Nt - 2).astype(np.int32)
    dt_i = t_grid[i0 + 1] - t_grid[i0]
    ft = np.clip((t_pts - t_grid[i0]) / np.where(dt_i > 0, dt_i, 1.0), 0.0, 1.0).astype(np.float32)

    j0 = np.clip(np.searchsorted(x_grid, x_pts, side="right") - 1, 0, Nx - 2).astype(np.int32)
    dx_j = x_grid[j0 + 1] - x_grid[j0]
    fx = np.clip((x_pts - x_grid[j0]) / np.where(dx_j > 0, dx_j, 1.0), 0.0, 1.0).astype(np.float32)

    w00 = ((1 - ft) * (1 - fx)).astype(np.float32)
    w01 = ((1 - ft) *      fx ).astype(np.float32)
    w10 = (     ft  * (1 - fx)).astype(np.float32)
    w11 = (     ft  *      fx ).astype(np.float32)
    return i0, j0, w00, w01, w10, w11


def _jax_apply_stencil_1d(U, i0, j0, w00, w01, w10, w11):
    """Apply a precomputed 1-D bilinear stencil to a ``(Nt, Nx)`` JAX array.

    Gradients flow through ``U`` only — ``i0, j0, w*`` are treated as static
    constants (no ``searchsorted`` inside the autodiff trace).

    Args:
        U:   ``(Nt, Nx)`` JAX array.
        i0:  ``(N,)`` int32 time-axis lower indices.
        j0:  ``(N,)`` int32 space-axis lower indices.
        w00, w01, w10, w11: ``(N,)`` float32 bilinear weights.

    Returns:
        ``(N,)`` interpolated values.
    """
    return (
        w00 * U[i0,     j0    ] +
        w01 * U[i0,     j0 + 1] +
        w10 * U[i0 + 1, j0    ] +
        w11 * U[i0 + 1, j0 + 1]
    )


# ─────────────────────────────────────────────────────────────────────────── #
#  Chebyshev differentiation matrix                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def _cheb_diff_matrix(N: int, L: float) -> np.ndarray:
    """Chebyshev pseudospectral differentiation matrix on ``N`` Gauss-Lobatto nodes.

    Constructs ``D`` such that ``D @ v ≈ dv/dx`` for values ``v`` sampled at
    the Gauss-Lobatto nodes ``x_j = xmin + (L/2)(1 - cos(j π/(N-1)))``,
    ``j = 0, …, N-1`` (the nodes built by :meth:`_build_grids` for
    ``bc="chebyshev"``).

    The matrix is computed on the reference interval ``[-1, 1]`` and then
    scaled by ``2/L`` to account for the physical domain length ``L``.

    Args:
        N: Number of Gauss-Lobatto nodes (includes both endpoints).
        L: Physical domain length ``xmax - xmin``.

    Returns:
        ``(N, N)`` float64 NumPy array.

    References:
        Trefethen (2000) *Spectral Methods in MATLAB*, Program 6.
    """
    if N == 1:
        return np.zeros((1, 1))
    j = np.arange(N)
    # Gauss-Lobatto nodes on [-1, 1]
    x = -np.cos(j * np.pi / (N - 1))
    # Barycentric weights
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** j
    # Off-diagonal entries
    X_mat = np.tile(x[:, None], (1, N))   # X_mat[i, j] = x[i]
    dX = X_mat - X_mat.T                   # dX[i, j] = x[i] - x[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(
            np.eye(N, dtype=bool),
            0.0,
            (c[:, None] / c[None, :]) / dX,
        )
    # Diagonal by negative row-sum (ensures D·1 = 0)
    np.fill_diagonal(D, -D.sum(axis=1))
    # Scale from [-1, 1] to [xmin, xmax] (length L)
    D *= 2.0 / L
    return D


# ─────────────────────────────────────────────────────────────────────────── #
#  ModelSpectralSolver                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class ModelSpectralSolver:
    """Spectral PDE model that owns a numerical integrator.

    Analogous to a neural-network model: exposes :meth:`apply` with the
    signature ``apply(params, X) → (N, n_states)`` so that the
    :class:`~pinns.trainer.Trainer` can treat it identically to a network.

    All parameters are registered via :meth:`add_parameter` into a flat dict.
    Trainability is **not** stored here — it is decided externally by the
    Trainer (``fit_model_parameters=[...]``) or by which keys appear in the
    ``params`` dict passed to :meth:`apply`.  Parameters absent from the
    override dict have ``jax.lax.stop_gradient`` applied in
    :meth:`_build_params`.

    Args:
        domain:      A :class:`~pinns.domain.DomainCubic` specifying geometry
                     and time interval.
        state_names: Ordered list of state variable names, e.g. ``["u"]``.
        integrator:  An owned :class:`~pinns.integrators.Integrator` instance
                     (e.g. ``IntegratorETD2RK(dt=5e-4, checkpoint=True)``).
        shape:       Number of spectral grid points per spatial dimension.
                     An ``int`` is broadcast to all dimensions.
        bc:          Spectral basis / boundary-condition type.
                     ``"periodic"`` (default) — Fourier / FFT.
                     ``"dirichlet"``          — Sine basis (DST-1).
                     ``"neumann"``             — Cosine basis (DCT-2).
                     ``"chebyshev"``           — Chebyshev pseudospectral collocation
                                                on Gauss-Lobatto nodes.  Supports
                                                arbitrary boundary conditions.  The
                                                linear operator receives the Chebyshev
                                                differentiation matrix ``D`` as ``X``
                                                and must return a dense matrix per
                                                state; only compatible with ETD2RK.

    After construction the model exposes the spectral grid as attributes:
        :attr:`x`, :attr:`y`  — physical coordinate arrays.
        :attr:`k`             — wavenumber array (1-D) or list of 1-D arrays.
                                For ``bc="chebyshev"`` this is the Chebyshev
                                differentiation matrix ``D`` (shape ``(N, N)``).
        :attr:`D`             — alias for ``k`` when ``bc="chebyshev"``.
        :attr:`K2`            — eigenvalue array of ``-Δ``, shape ``(*shape)``.
                                For ``bc="chebyshev"`` this is the matrix ``D²``.
        :meth:`forward`       — physical → working-basis transform.
                                For ``bc="chebyshev"`` this is the identity.
        :meth:`inverse`       — working-basis → physical transform.
                                For ``bc="chebyshev"`` this is the identity.
    """

    _VALID_BC = ("periodic", "dirichlet", "neumann", "chebyshev")

    # Marker for integrator duck-type check
    _is_solver_problem: bool = True

    def __init__(
        self,
        domain,
        state_names: Sequence[str],
        integrator=None,
        shape: Union[int, Sequence[int]] = 64,
        bc: str = "periodic",
        nonlinear=None,
        linear=None,
    ):
        from ..domain.domain_cubic import DomainCubic
        from .integrators.integrator_base import Integrator

        if not isinstance(domain, DomainCubic):
            raise TypeError(
                f"ModelSpectralSolver expects a DomainCubic, got {type(domain).__name__}."
            )

        # Detect stationary vs time-dependent from the domain
        self.stationary: bool = (domain._t_min is None)

        if not self.stationary:
            if integrator is None:
                raise ValueError(
                    "A time-dependent domain requires an integrator.  "
                    "Pass an Integrator instance or use DomainCubic(time=None) "
                    "for a stationary problem."
                )
            if not isinstance(integrator, Integrator):
                raise TypeError(
                    f"ModelSpectralSolver expects an Integrator, got {type(integrator).__name__}."
                )

        bc = bc.lower()
        if bc not in self._VALID_BC:
            raise ValueError(
                f"bc={bc!r} not supported.  Choose from {self._VALID_BC}."
            )

        self.domain = domain   # DomainCubic — kept for sampling / BC registration
        self.bc = bc

        # ── Spatial grid parameters ───────────────────────────────────────
        n_spatial = domain._spatial_dims
        self.n_dims = n_spatial
        self.space: List[Tuple[float, float]] = [
            (float(domain.xmin[i]), float(domain.xmax[i])) for i in range(n_spatial)
        ]
        self._lengths: List[float] = [xmax - xmin for xmin, xmax in self.space]
        self._t_min = domain._t_min
        self._t_max = domain._t_max

        # Normalise shape
        if isinstance(shape, int):
            self.shape: Tuple[int, ...] = tuple([shape] * n_spatial)
        else:
            self.shape = tuple(int(s) for s in shape)
        if len(self.shape) != n_spatial:
            raise ValueError(
                f"len(shape)={len(self.shape)} != n_spatial={n_spatial}."
            )

        # ── Build spectral infrastructure ─────────────────────────────────
        self._grids: List[np.ndarray] = self._build_grids()
        self._axes:  List[np.ndarray] = self._build_axes()

        import jax.numpy as jnp
        if n_spatial == 1:
            self.x = jnp.array(self._grids[0])
        elif n_spatial == 2:
            X, Y = np.meshgrid(self._grids[0], self._grids[1], indexing="ij")
            self.x = jnp.array(X)
            self.y = jnp.array(Y)
        else:
            mesh = np.meshgrid(*self._grids, indexing="ij")
            for i, m in enumerate(mesh):
                setattr(self, f"x{i+1}" if i > 0 else "x", jnp.array(m))

        self.K2 = self._build_K2()

        if n_spatial == 1:
            self.k = jnp.array(self._axes[0])
        else:
            self.k = [jnp.array(ax) for ax in self._axes]

        # For Chebyshev, also expose 'D' as a convenient alias
        if self.bc == "chebyshev":
            self.D = self.k

        # ── Other model state ─────────────────────────────────────────────
        self.state_names: List[str] = list(state_names)
        self.n_states: int = len(self.state_names)
        self._integrator = integrator

        # ── flat parameter storage (no fixed/inferred split) ─────────────
        self._params: Dict[str, Any] = {}
        # Trainable subset set by the Trainer (None = expose all _params).
        self._trainable_params: Optional[Dict[str, Any]] = None

        # ── nonlinear / linear solvers (stationary + implicit stepping) ──
        self._nonlinear_solver = nonlinear
        self._linear_solver    = linear

        # ── spectral operators ────────────────────────────────────────────
        self._linear_op:  Optional[Callable] = None
        self._linear_fn:  Optional[Callable] = None
        self._source_op:  Optional[Callable] = None
        self._source_fn:  Optional[Callable] = None

        # ── initial conditions (physical space) ──────────────────────────
        self._initial: Optional[Dict[str, np.ndarray]] = None

        # ── precomputed interpolation stencil (set via precompute_interp_stencil) ──
        self._interp_stencil: Optional[Dict[str, Any]] = None

    # ──────────────────────────────────────────────────────────────────── #
    #  Spectral grid construction                                          #
    # ──────────────────────────────────────────────────────────────────── #

    def _build_grids(self) -> List[np.ndarray]:
        """Physical node positions per dimension."""
        grids = []
        for i, (xmin, xmax) in enumerate(self.space):
            N = self.shape[i]
            if self.bc == "periodic":
                grids.append(np.linspace(xmin, xmax, N, endpoint=False))
            elif self.bc == "chebyshev":
                # Gauss-Lobatto nodes on [-1, 1] then scaled to [xmin, xmax]
                j = np.arange(N)
                nodes_m1p1 = -np.cos(j * np.pi / (N - 1))
                grids.append(xmin + (xmax - xmin) * (nodes_m1p1 + 1.0) / 2.0)
            else:
                grids.append(np.linspace(xmin, xmax, N))
        return grids

    def _build_axes(self) -> List[np.ndarray]:
        """1-D spectral wavenumber/eigenvalue axis per dimension.

        For ``bc="chebyshev"`` returns the Chebyshev differentiation matrix
        ``D`` of shape ``(N, N)`` rather than a 1-D wavenumber vector.
        """
        axes = []
        for i, (L, N) in enumerate(zip(self._lengths, self.shape)):
            if self.bc == "periodic":
                k = np.fft.fftfreq(N, d=L / N) * (2.0 * np.pi)
            elif self.bc == "dirichlet":
                k = np.arange(1, N + 1) * np.pi / L
            elif self.bc == "chebyshev":
                k = _cheb_diff_matrix(N, L)
            else:  # neumann
                k = np.arange(N) * np.pi / L
            axes.append(k)
        return axes

    def _build_K2(self):
        """N-D eigenvalue / operator array of ``-Δ``.

        For Fourier / DST / DCT: returns a diagonal eigenvalue array.
        For Chebyshev: returns the matrix ``-D²`` (shape ``(N, N)`` for 1-D),
        computed in NumPy to avoid JAX BLAS initialisation overhead at model
        build time.
        """
        import jax.numpy as jnp
        if self.bc == "chebyshev":
            if self.n_dims == 1:
                # 1-D: single (N, N) matrix  −D²
                D_np = self._axes[0]
                return jnp.array(-(D_np @ D_np))
            # N-D: list of per-dimension (N_i, N_i) matrices  [−D₀², −D₁², …]
            # The Laplacian is the Kronecker sum ∑_i  I ⊗ … ⊗ (D_i²) ⊗ … ⊗ I.
            # Returning a list signals the Kronecker path to all integrators.
            return [jnp.array(-(D_np @ D_np)) for D_np in self._axes]
        k2_1d = [jnp.array(k ** 2) for k in self._axes]
        K2 = k2_1d[0]
        for kk in k2_1d[1:]:
            K2 = K2[..., None] + kk[None, :]
        return K2

    # ──────────────────────────────────────────────────────────────────── #
    #  Forward / inverse transforms                                        #
    # ──────────────────────────────────────────────────────────────────── #

    def forward(self, u):
        """Physical space → working-basis representation.

        For ``bc="chebyshev"`` the working basis **is** physical space
        (collocation values at Gauss-Lobatto nodes), so this is the identity.
        """
        if self.bc == "periodic":
            import jax.numpy as jnp
            return jnp.fft.fftn(u)
        elif self.bc == "dirichlet":
            return self._dst1n(u)
        elif self.bc == "chebyshev":
            return u  # collocation — working basis = physical space
        else:
            return self._dctn(u)

    def inverse(self, u_hat):
        """Working-basis representation → physical space.

        For ``bc="chebyshev"`` this is the identity (see :meth:`forward`).
        """
        if self.bc == "periodic":
            import jax.numpy as jnp
            return jnp.fft.ifftn(u_hat).real
        elif self.bc == "dirichlet":
            return self._idst1n(u_hat)
        elif self.bc == "chebyshev":
            return u_hat  # collocation — physical space = working basis
        else:
            return self._idctn(u_hat)

    @staticmethod
    def _dst1_axis(u, axis: int):
        import jax.numpy as jnp
        N = u.shape[axis]
        ndim = u.ndim
        def sl(idx_or_slice, ax):
            s = [slice(None)] * ndim
            s[ax] = idx_or_slice
            return tuple(s)
        zeros = jnp.zeros_like(u[sl(slice(0, 1), axis)])
        flipped = jnp.flip(-u, axis=axis)
        extended = jnp.concatenate([zeros, u, zeros, flipped], axis=axis)
        ft = jnp.fft.fft(extended, axis=axis)
        return jnp.take(-ft.imag, jnp.arange(1, N + 1), axis=axis)

    def _dst1n(self, u):
        result = u
        for ax in range(self.n_dims):
            result = self._dst1_axis(result, ax)
        return result

    def _idst1n(self, u_hat):
        result = u_hat
        for ax in range(self.n_dims):
            N = self.shape[ax]
            result = self._dst1_axis(result, ax) / (2.0 * (N + 1))
        return result

    def _dctn(self, u):
        import jax.scipy.fft
        result = u
        for ax in range(self.n_dims):
            result = jax.scipy.fft.dct(result, type=2, axis=ax, norm="ortho")
        return result

    def _idctn(self, u_hat):
        import jax.scipy.fft
        result = u_hat
        for ax in range(self.n_dims):
            result = jax.scipy.fft.idct(result, type=2, axis=ax, norm="ortho")
        return result

    # ──────────────────────────────────────────────────────────────────── #
    #  Parameter registration                                              #
    # ──────────────────────────────────────────────────────────────────── #

    def add_parameter(
        self,
        name: Union[str, List[str]],
        value: Any,
    ) -> "ModelSpectralSolver":
        """Register a model parameter.

        Trainability is decided at Trainer compile time (``fit_model_parameters``).
        Call this for **all** scalar constants and to-be-inferred parameters.

        Three call styles::

            model.add_parameter("mu",  0.022)
            model.add_parameter(["b1","b2"], [40., 100.])

        Returns:
            ``self`` for method chaining.
        """
        if isinstance(name, str):
            self._params[name] = value
        else:
            names = list(name)
            values = list(value) if isinstance(value, (list, tuple)) else [value] * len(names)
            if len(values) != len(names):
                raise ValueError(
                    f"add_parameter: {len(names)} names but {len(values)} values."
                )
            for n, v in zip(names, values):
                self._params[n] = v
        return self

    @property
    def params(self) -> Dict[str, Any]:
        """Trainable parameter pytree visible to the Trainer / optimizer.

        Settable — the Trainer assigns the optimizer-managed subset here
        (e.g. only the keys listed in ``fit_model_parameters``).
        """
        return self._trainable_params if self._trainable_params is not None else self._params

    @params.setter
    def params(self, value) -> None:
        self._trainable_params = value


    # ──────────────────────────────────────────────────────────────────── #
    #  Operator registration                                               #
    # ──────────────────────────────────────────────────────────────────── #

    def set_linear_op(self, fn: Callable) -> "ModelSpectralSolver":
        """Register the linear operator as an explicit object.

        ``fn(X, p) → {state_name: L}``

        ``X`` is the collocation-point array (physical space).  ``L`` can be:

        * A **diagonal eigenvalue array** (same shape as state) — Fourier /
          DST / DCT where ``L û = λ û``.
        * A **dense matrix** ``(N, N)`` — 1-D Chebyshev where ``L = D²``.
        * A **list of matrices** — N-D Chebyshev Kronecker path.

        The integrator / solver uses ``L`` directly (e.g. ``expm(L dt)``,
        ``linalg.solve(L, f)``).  Use :meth:`set_linear_fn` for the
        matrix-free path instead.

        ``p["parameter"]`` holds user scalars; ``p["model"]["D"]`` holds the
        differentiation matrix / list of matrices for Chebyshev.

        Example (1-D Chebyshev heat equation)::

            model.set_linear_op(lambda X, p: {"u": p["parameter"]["nu"] * p["model"]["D"] @ p["model"]["D"]})

        Example (Fourier KdV dispersion)::

            model.set_linear_op(lambda X, p: {"u": 1j * p["parameter"]["mu2"] * X**3})
        """
        self._linear_op = fn
        return self

    def set_linear_fn(self, fn: Callable) -> "ModelSpectralSolver":
        """Register the linear operator as a matrix-free action.

        ``fn(X, U, p) → {state_name: L(u)}``

        ``X`` is the collocation-point array (or list of 1-D arrays for N-D).
        ``U`` is the current state dict ``{name: array}`` in physical space.
        The return value is ``L(u)`` — the operator applied to the current
        state — same shape as ``U[name]``.

        Used by:

        * Explicit integrators (RK4, Diffrax) — called at each stage.
        * Stiff integrators (ETD2RK, IMEX) — Krylov ``expm`` approximation.
        * Stationary / implicit solvers — GMRES ``matvec``.

        Preferred over :meth:`set_linear_op` for Chebyshev 2D+ (avoids
        forming the Kronecker matrix) and for problems with cross-derivatives.

        ``p["parameter"]`` holds user scalars; ``p["model"]["D"]`` holds the
        differentiation matrix / list for Chebyshev.

        Example (2-D Chebyshev Laplacian)::

            def linear(X, U, p):
                Dx, Dy = p["model"]["D"]
                nu = p["parameter"]["nu"]
                u  = U["u"]
                return {"u": nu * (Dx@Dx @ u + u @ (Dy@Dy).T)}

            model.set_linear_fn(linear)
        """
        self._linear_fn = fn
        return self

    def set_source_op(self, fn: Callable) -> "ModelSpectralSolver":
        """Register the source / RHS as an explicit operator or array.

        ``fn(X, p) → {state_name: f}``

        ``f`` does **not** depend on the current state ``U`` — use this for
        known forcing terms, initial data transforms, etc.
        For state-dependent (nonlinear) sources use :meth:`set_source_fn`.

        ``X`` is the collocation-point array.
        ``p["parameter"]`` holds user scalars.

        Example (Poisson forcing)::

            model.set_source_op(lambda X, p: {"u": 2.0*(2.0 - X[0]**2 - X[1]**2)})
        """
        self._source_op = fn
        return self

    def set_source_fn(self, fn: Callable) -> "ModelSpectralSolver":
        """Register the source / RHS as a state-dependent (nonlinear) function.

        ``fn(X, U, p) → {state_name: f(u)}``

        ``X`` is the collocation-point array. ``U`` is the current state dict.
        Use for nonlinear reaction terms, advection, etc.

        For time-dependent split problems this is the **nonlinear part** ``N(u)``
        in ``∂_t u = L u + N(u)``.

        For stationary problems this is evaluated at ``U = 0`` as the RHS
        ``f`` when using :meth:`set_source_op` would also suffice.

        ``p["parameter"]`` holds user scalars; ``p["model"]["D"]`` the operator.

        Example (KdV nonlinear term, Fourier)::

            def kdv_nonlinear(X, U, p):
                u = model.inverse(U["u"])
                return {"u": -1j * p["parameter"]["eta"] / 2.0 * X * model.forward(u * u)}

            model.set_source_fn(kdv_nonlinear)
        """
        self._source_fn = fn
        return self

    # ──────────────────────────────────────────────────────────────────── #
    #  Initial conditions and observations                                  #
    # ──────────────────────────────────────────────────────────────────── #

    def add_initial(self, *arrays, **kwargs) -> "ModelSpectralSolver":
        """Set initial-condition arrays (physical space).

        Three call styles::

            model.add_initial(u0)
            model.add_initial(u0, v0)              # positional, state_names order
            model.add_initial({"u": u0, "v": v0})  # dict
            model.add_initial(u=u0, v=v0)           # kwargs
        """
        if kwargs:
            self._initial = {k: np.asarray(v) for k, v in kwargs.items()}
            return self
        if len(arrays) == 1 and isinstance(arrays[0], dict):
            self._initial = {k: np.asarray(v) for k, v in arrays[0].items()}
            return self
        if len(arrays) != self.n_states:
            raise ValueError(
                f"add_initial: expected {self.n_states} array(s) "
                f"(one per state), got {len(arrays)}."
            )
        self._initial = {
            name: np.asarray(arr)
            for name, arr in zip(self.state_names, arrays)
        }
        return self

    # ──────────────────────────────────────────────────────────────────── #
    #  Internal helpers required by integrators                            #
    # ──────────────────────────────────────────────────────────────────── #

    def _build_params(
        self, inferred_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return a structured ``p`` dict for operator functions.

        Returns a dict with two top-level keys:

        * ``p["parameter"]`` — user-registered scalars / arrays.  Each value
          is either the trainable override (gradient flows) or
          ``stop_gradient`` (frozen).
        * ``p["model"]``     — model internals exposed to operator functions:

          - ``p["model"]["D"]`` — differentiation matrix for ``bc="chebyshev"``.
            For 1-D: ``(N, N)`` matrix.  For N-D: list of ``(N_i, N_i)`` matrices.
            ``None`` for non-Chebyshev BCs.

        Args:
            inferred_override: Dict of trainable parameter values coming from
                               the optimiser.  Keys that do not appear here
                               are frozen.

        Returns:
            Dict ``{"parameter": {...}, "model": {...}}``.
        """
        import jax
        override = inferred_override if inferred_override is not None else {}
        parameter: Dict[str, Any] = {}
        for k, v in self._params.items():
            if k in override:
                parameter[k] = override[k]
            else:
                parameter[k] = jax.lax.stop_gradient(v)

        # Model internals — stop_gradient: these are static infrastructure,
        # not trainable.  For inverse problems where D is differentiable
        # the user can retrieve it directly from model.D.
        model_internals: Dict[str, Any] = {
            "D": self.D if self.bc == "chebyshev" else None,
        }

        return {"parameter": parameter, "model": model_internals}

    def get_initial_state(self) -> Dict[str, Any]:
        """Return initial state in the model's working basis (called by integrators).

        For spectral models this FFTs the physical-space IC into the spectral
        basis.  Override in subclasses that operate in physical space.
        """
        if self._initial is None:
            raise RuntimeError("Call add_initial() before solving.")
        import jax.numpy as jnp
        return {
            name: self.forward(jnp.array(arr))
            for name, arr in self._initial.items()
        }

    # keep old name as alias for backward compatibility
    def get_initial_hat(self) -> Dict[str, Any]:
        """Alias for :meth:`get_initial_state` (kept for backward compatibility)."""
        return self.get_initial_state()

    # ──────────────────────────────────────────────────────────────────── #
    #  Generic operator dispatch (called by integrators)                   #
    # ──────────────────────────────────────────────────────────────────── #

    def _call_linear_op(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call the linear operator registration and return the operator dict.

        Dispatches to :attr:`_linear_op` (``fn(X, p)``) if registered.
        Returns zero arrays if neither ``set_linear_op`` nor ``set_linear_fn``
        was called (pure explicit ODE).

        Integrators that need the operator matrix (ETD2RK direct path, direct
        solve) call this method.  Integrators that only need the action
        ``L(u)`` call :meth:`_call_linear_fn` instead.
        """
        import jax.numpy as jnp
        if self._linear_op is not None:
            return self._linear_op(self.k, params)
        # No operator registered — return zeros (used as fallback by integrators)
        return {name: jnp.zeros(self.shape) for name in self.state_names}

    def _call_linear_fn(
        self, state: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate the linear action ``L(u)`` on the current state.

        Dispatches to :attr:`_linear_fn` (``fn(X, U, p)``) if registered,
        else falls back to :attr:`_linear_op` applied to the state.
        Returns zeros if nothing was registered.
        """
        import jax.numpy as jnp
        X = self._get_X()
        if self._linear_fn is not None:
            return self._linear_fn(X, state, params)
        if self._linear_op is not None:
            # Apply operator matrix / eigenvalues to state
            L_dict = self._linear_op(self.k, params)
            result = {}
            for name in self.state_names:
                L = L_dict[name]
                u = state[name]
                import numpy as np
                L_arr = jnp.asarray(L)
                if L_arr.ndim == 2:
                    result[name] = L_arr @ u.ravel() if u.ndim > 1 else L_arr @ u
                else:
                    result[name] = L_arr * u
            return result
        return {name: jnp.zeros_like(state[name]) for name in self.state_names}

    def _call_nonlinear_op(
        self, state: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate the source / RHS at the current state.

        Dispatches to :attr:`_source_fn` (``fn(X, U, p)``) first, then
        :attr:`_source_op` (``fn(X, p)``).  Returns zeros if nothing
        was registered.

        Kept as ``_call_nonlinear_op`` so existing integrators need no changes.
        """
        import jax.numpy as jnp
        X = self._get_X()
        if self._source_fn is not None:
            return self._source_fn(X, state, params)
        if self._source_op is not None:
            return self._source_op(X, params)
        return {name: jnp.zeros_like(state[name]) for name in self.state_names}

    def _get_X(self):
        """Return the collocation-point array passed as ``X`` to operator fns.

        * 1-D: ``self.x``  — shape ``(N,)``
        * N-D Chebyshev: ``[self._grids[0], self._grids[1], ...]`` — list of
          1-D numpy arrays, one per dimension.
        * N-D Fourier: ``self.k`` — list of 1-D wavenumber arrays.
        """
        if self.n_dims == 1:
            return self.x
        if self.bc == "chebyshev":
            import jax.numpy as jnp
            return [jnp.array(g) for g in self._grids]
        return self.k

    def _to_physical_batch(
        self, state_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert a batch of basis-space snapshots to physical space.

        Args:
            state_dict: ``{name: (N_obs, *shape)}`` basis-space arrays.

        Returns:
            ``{name: (N_obs, *shape)}`` physical-space arrays.
        """
        import jax
        return {
            name: jax.vmap(self.inverse)(state_dict[name])
            for name in self.state_names
        }

    def _detect_linear_diagonal(self, params: Dict[str, Any]) -> bool:
        """Return ``True`` if the linear op is diagonal (returns eigenvalue arrays).

        Probes ``_linear_op`` once with dummy params and checks whether the
        returned arrays have the same shape as the state (diagonal / spectral)
        or a larger shape (matrix).  Called once at solve time; result is
        cached on ``self._linear_is_diagonal``.
        """
        import numpy as np
        result = self._call_linear_op(params)
        for name in self.state_names:
            v = np.asarray(result[name])
            if v.shape == self.shape:
                self._linear_is_diagonal = True
                return True
            else:
                self._linear_is_diagonal = False
                return False
        self._linear_is_diagonal = True
        return True

    def diff(self, u_hat, order: int = 1):
        """Spectral derivative of order ``n`` applied to a basis-space array.

        For Fourier computes ``(ik)^n · û``; for sine/cosine ``k^n · û``.
        For Chebyshev applies ``D^order`` via repeated matrix-vector products.
        Returns the result in the **same basis** (not physical space).

        Args:
            u_hat: Basis-space array with the same shape as the model grid.
            order: Derivative order (default 1).

        Returns:
            Basis-space array.
        """
        import jax.numpy as jnp
        if self.bc == "chebyshev":
            D = self.k  # (N, N) differentiation matrix
            result = u_hat
            for _ in range(order):
                result = D @ result
            return result
        k = self.k
        if self.bc == "periodic":
            factor = (1j * k) ** order
        else:
            # sine/cosine: eigenvalue of d/dx is k (positive real)
            factor = k ** order
        return factor * u_hat

    def _validate(self) -> None:
        """Raise if the model is not fully configured (called by integrators)."""
        errors = []
        if not self.stationary and self._initial is None:
            errors.append("initial condition not set — call add_initial().")
        if errors:
            raise RuntimeError(
                "ModelSpectralSolver is not fully configured:\n  " + "\n  ".join(errors)
            )

    # ──────────────────────────────────────────────────────────────────── #
    #  Main interface                                                       #
    # ──────────────────────────────────────────────────────────────────── #

    def solve(
        self,
        inferred_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Solve and return the solution.

        * **Stationary domain** (``DomainCubic(time=None)``) — directly solves
          the linear system ``L u = -f`` where ``f`` is the source from the
          nonlinear operator.  Returns ``{state_name: (*shape,) array}``.
        * **Time-dependent domain** — time-marches with the owned integrator.
          Observation times are derived from the domain bounds and ``integrator.dt``.
          Returns ``{state_name: (Nt, *shape) array}``.

        Args:
            inferred_params: Trainable parameter override dict.

        Returns:
            Dict ``{state_name: jax.Array}``.
        """
        if self.stationary:
            return self.solve_stationary(inferred_params=inferred_params)
        U_dict = self._integrator.solve(self, inferred_params=inferred_params)
        # Cache for predict() — physical-space snapshots + observation times
        import numpy as np
        self._cached_U = {k: np.array(v) for k, v in U_dict.items()}
        self._cached_t_obs = np.array(self._integrator._get_obs_times(self))
        return U_dict

    def solve_stationary(
        self,
        inferred_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Solve the stationary (time-independent) system ``L u = f``.

        The pipeline is:

        1. Build ``p`` from registered parameters.
        2. Build RHS ``f`` from :meth:`set_source_op` / :meth:`set_source_fn`
           (evaluated at the zero state for the operator path).
        3. If a :class:`~pinns.models.nonlinear.NonlinearSolverBase` was
           passed at construction, use it to solve ``R(u) = L(u) - f(u) = 0``
           via the registered :class:`~pinns.models.linear.LinearSolverBase`.
        4. Otherwise fall back to the direct path:

           * **set_linear_op** registered: ``jnp.linalg.solve(L, f)`` (matrix)
             or element-wise ``f / λ`` (diagonal Fourier).
           * **set_linear_fn** registered: GMRES via
             ``jax.scipy.sparse.linalg.gmres``.

        Returns:
            Dict ``{state_name: (*shape,) jax.Array}`` in physical space.
        """
        import jax.numpy as jnp
        import jax.scipy.sparse.linalg as jssl

        self._validate()
        p = self._build_params(inferred_params)

        zero_state = {name: jnp.zeros(self.shape) for name in self.state_names}
        F_dict = self._call_nonlinear_op(zero_state, p)

        # ── Pipeline path: NonlinearSolver + LinearSolver ────────────────
        if self._nonlinear_solver is not None and self._linear_solver is not None:
            result = {}
            for name in self.state_names:
                shape = self.shape
                f_flat = F_dict[name].ravel()

                def residual(u_flat, _name=name, _shape=shape):
                    U = {n: jnp.zeros(_shape) for n in self.state_names}
                    U[_name] = u_flat.reshape(_shape)
                    Lu = self._call_linear_fn(U, p)[_name].ravel()
                    fu = self._call_nonlinear_op(U, p)[_name].ravel()
                    return Lu - fu

                u_flat = self._nonlinear_solver.solve(
                    residual,
                    jnp.zeros_like(f_flat),
                    self._linear_solver.solve,
                )
                result[name] = u_flat.reshape(shape)
            return result

        # ── Direct / fallback path ────────────────────────────────────────
        result = {}
        for name in self.state_names:
            f = F_dict[name]

            if self._linear_fn is not None:
                # Matrix-free: use GMRES
                shape = self.shape
                f_flat = f.ravel()

                def matvec(v, _name=name, _shape=shape):
                    U = {n: jnp.zeros(_shape) for n in self.state_names}
                    U[_name] = v.reshape(_shape)
                    return self._call_linear_fn(U, p)[_name].ravel()

                u_flat, _ = jssl.gmres(matvec, f_flat)
                result[name] = u_flat.reshape(shape)

            elif self._linear_op is not None:
                L_dict = self._call_linear_op(p)
                L = L_dict[name]
                L_arr = jnp.asarray(L)
                if L_arr.ndim == 2:
                    result[name] = jnp.linalg.solve(L_arr, f.ravel()).reshape(self.shape)
                else:
                    # Diagonal (Fourier / sine / cosine)
                    u_hat = f / L_arr
                    result[name] = self.inverse(u_hat)
            else:
                raise RuntimeError(
                    f"State '{name}': no linear operator registered.  "
                    "Call set_linear_op() or set_linear_fn() before solve_stationary()."
                )

        return result

    def precompute_interp_stencil(self, X) -> "ModelSpectralSolver":
        """Precompute the bilinear interpolation stencil for query points ``X``.

        Call this **once** before training whenever ``X`` (the observation
        coordinates) is fixed.  The stencil (grid indices and weights) is
        stored on the model and reused by every subsequent :meth:`apply` call,
        eliminating ``searchsorted`` from the JIT/autodiff trace and giving a
        significant speedup (~5–10×) for large datasets.

        Args:
            X: ``(N, n_spatial + 1)`` array with columns ``[x, [y,], t]``.
               Must match the ``X`` that will be passed to :meth:`apply`.

        Returns:
            ``self`` for method chaining.
        """
        import numpy as np
        X = np.asarray(X)
        n_spatial = self.n_dims
        x_pts = X[:, :n_spatial]
        t_pts = X[:,  n_spatial]

        t_grid = self._integrator._get_obs_times(self)
        grids  = self._grids

        if n_spatial == 1:
            x_grid = np.asarray(grids[0])
            stencils = {}
            for state in self.state_names:
                i0, j0, w00, w01, w10, w11 = _compute_stencil_1d(
                    t_grid, x_grid, t_pts.astype(np.float64), x_pts[:, 0].astype(np.float64)
                )
                stencils[state] = (i0, j0, w00, w01, w10, w11)
            self._interp_stencil = {"type": "1d", "stencils": stencils, "X_shape": X.shape}
        elif n_spatial == 2:
            x_grid = np.asarray(grids[0])
            y_grid = np.asarray(grids[1])
            # 2-D case: store raw points for fallback (can be extended later)
            self._interp_stencil = {"type": "2d_fallback",
                                     "t_pts": t_pts, "x_pts": x_pts,
                                     "X_shape": X.shape}
        else:
            raise NotImplementedError(f"n_spatial={n_spatial} not supported")
        return self

    def apply(
        self,
        X,
        params: Dict[str, Any] = None,
        params_dict=None,
    ):
        """Differentiable evaluation at scattered ``(x[, y, ...][, t])`` points.

        The full chain ``params → solve → U → bilinear_interp`` is
        differentiable end-to-end (gradients flow through U via JAX).

        Args:
            X:      For **stationary** models: ``(N, n_spatial)`` array with
                    columns ``[x, [y, ...]]``.
                    For **time-dependent** models: ``(N, n_spatial + 1)`` array
                    with columns ``[x, [y, ...], t]`` (time last).
            params: Trainable parameter values, e.g.
                    ``{"mu": jnp.array(0.02)}``.  ``None`` (default) uses
                    registered parameters only.
            params_dict: Ignored (kept for API compatibility).

        Returns:
            ``(N, n_states)`` JAX array.  Column order matches
            :attr:`state_names`.
        """
        import jax.numpy as jnp

        # ── Stationary branch (no integrator / no time axis) ─────────────────
        if self.stationary:
            U_dict = self.solve_stationary(inferred_params=params)
            n_spatial = self.n_dims
            x_pts = X[:, 0]  # (N,)
            grids = self._grids
            results = []
            if n_spatial == 1:
                x_grid = jnp.array(grids[0])
                for state in self.state_names:
                    vals = _jax_interp_stationary_1d(U_dict[state].ravel(), x_grid, x_pts)
                    results.append(vals[:, None])
            elif n_spatial == 2:
                x_grid = jnp.array(grids[0])
                y_grid = jnp.array(grids[1])
                y_pts  = X[:, 1]
                for state in self.state_names:
                    vals = _jax_interp_stationary_2d(U_dict[state], x_grid, y_grid, x_pts, y_pts)
                    results.append(vals[:, None])
            else:
                raise NotImplementedError(
                    f"apply() stationary path is implemented for 1D and 2D; "
                    f"got n_spatial={n_spatial}."
                )
            return jnp.concatenate(results, axis=-1)

        # Run the full forward solve; observation times come from domain + dt
        t_obs = jnp.array(self._integrator._get_obs_times(self))
        U_dict = self._integrator.solve(self, inferred_params=params)

        # ── Fast path: use precomputed stencil (no searchsorted in autodiff) ──
        if self._interp_stencil is not None:
            stype = self._interp_stencil["type"]
            if stype == "1d":
                results = []
                for state in self.state_names:
                    U = U_dict[state]  # (Nt, Nx)
                    i0, j0, w00, w01, w10, w11 = self._interp_stencil["stencils"][state]
                    vals = _jax_apply_stencil_1d(U, i0, j0, w00, w01, w10, w11)
                    results.append(vals[:, None])
                return jnp.concatenate(results, axis=-1)
            # 2-D stencil fast path not yet implemented; fall through to slow path

        # ── Slow path: compute stencil inside the JIT (searchsorted per step) ──
        n_spatial = self.n_dims
        x_pts = X[:, :n_spatial]   # (N, n_spatial)
        t_pts = X[:,  n_spatial]   # (N,)

        grids = self._grids  # list of 1-D coord arrays

        if n_spatial == 1:
            x_grid = jnp.array(grids[0])
            results = []
            for state in self.state_names:
                U = U_dict[state]  # (Nt, Nx)
                vals = _jax_interp_1d(U, t_obs, x_grid, t_pts, x_pts[:, 0])
                results.append(vals[:, None])
            return jnp.concatenate(results, axis=-1)  # (N, n_states)

        elif n_spatial == 2:
            x_grid = jnp.array(grids[0])
            y_grid = jnp.array(grids[1])
            results = []
            for state in self.state_names:
                U = U_dict[state]  # (Nt, Nx, Ny)
                vals = _jax_interp_2d(U, t_obs, x_grid, y_grid,
                                       t_pts, x_pts[:, 0], x_pts[:, 1])
                results.append(vals[:, None])
            return jnp.concatenate(results, axis=-1)

        else:
            raise NotImplementedError(
                f"apply() is implemented for 1D and 2D spatial domains; "
                f"got n_spatial={n_spatial}."
            )

    def predict(self, X):
        """Evaluate the cached solution at arbitrary space-time points.

        Unlike :meth:`apply`, this method **does not re-run the integrator**.
        It reuses the ``U`` arrays stored by the last :meth:`solve` call and
        applies the same JAX bilinear interpolation used by :meth:`apply`
        (fast stencil path if :meth:`precompute_interp_stencil` was called,
        otherwise the standard ``searchsorted`` path).  Gradients **do not**
        flow back through the cached ``U`` — this is purely a cheap reference
        evaluation.

        Args:
            X: ``(N, n_spatial + 1)`` array with columns ``[x, [y, ...], t]``
               (time last), matching the convention of
               :class:`~pinns.domain.DomainCubic`.

        Returns:
            ``(N, n_states)`` JAX array.  For single-state problems this is
            ``(N, 1)``.

        Raises:
            RuntimeError: If :meth:`solve` has not been called yet.
        """
        import jax.numpy as jnp

        if not hasattr(self, "_cached_U"):
            raise RuntimeError(
                "Call solve() before predict().  No solution is cached yet."
            )

        X = jnp.asarray(X)
        t_obs  = jnp.array(self._cached_t_obs)
        n_spatial = self.n_dims
        x_pts  = X[:, :n_spatial]
        t_pts  = X[:,  n_spatial]
        grids  = self._grids

        # ── Fast stencil path (precomputed) ──────────────────────────────────
        if self._interp_stencil is not None and self._interp_stencil["type"] == "1d":
            results = []
            for state in self.state_names:
                U = jnp.array(self._cached_U[state])
                i0, j0, w00, w01, w10, w11 = self._interp_stencil["stencils"][state]
                vals = _jax_apply_stencil_1d(U, i0, j0, w00, w01, w10, w11)
                results.append(vals[:, None])
            return jnp.concatenate(results, axis=-1)

        # ── Standard path: same bilinear interp as apply() ───────────────────
        if n_spatial == 1:
            x_grid = jnp.array(grids[0])
            results = []
            for state in self.state_names:
                U = jnp.array(self._cached_U[state])
                vals = _jax_interp_1d(U, t_obs, x_grid, t_pts, x_pts[:, 0])
                results.append(vals[:, None])
            return jnp.concatenate(results, axis=-1)

        if n_spatial == 2:
            x_grid = jnp.array(grids[0])
            y_grid = jnp.array(grids[1])
            results = []
            for state in self.state_names:
                U = jnp.array(self._cached_U[state])
                vals = _jax_interp_2d(U, t_obs, x_grid, y_grid,
                                      t_pts, x_pts[:, 0], x_pts[:, 1])
                results.append(vals[:, None])
            return jnp.concatenate(results, axis=-1)

        raise NotImplementedError(
            f"predict() is implemented for 1D and 2D spatial domains; "
            f"got n_spatial={n_spatial}."
        )

    # ──────────────────────────────────────────────────────────────────── #
    #  Repr                                                                #
    # ──────────────────────────────────────────────────────────────────── #

    def __repr__(self) -> str:
        pnames = list(self._params.keys())
        if self.stationary:
            solver_info = "stationary"
        else:
            solver_info = f"integrator={type(self._integrator).__name__}"
        return (
            f"ModelSpectralSolver(states={self.state_names}, shape={self.shape}, "
            f"bc={self.bc!r}, params={pnames}, {solver_info})"
        )


# ─────────────────────────────────────────────────────────────────────────── #
#  2-D spatial bilinear interpolation (trilinear in t, x, y)                #
# ─────────────────────────────────────────────────────────────────────────── #

def _jax_interp_2d(U, t_grid, x_grid, y_grid, t_pts, x_pts, y_pts):
    """Differentiable trilinear interpolation on a (t, x, y) grid.

    Args:
        U:       ``(Nt, Nx, Ny)`` JAX array.
        t_grid:  ``(Nt,)``
        x_grid:  ``(Nx,)``
        y_grid:  ``(Ny,)``
        t_pts, x_pts, y_pts: ``(N,)`` query coords.

    Returns:
        ``(N,)`` interpolated values.
    """
    import jax.numpy as jnp

    Nt, Nx, Ny = U.shape

    def _frac(grid, pts, N):
        i0 = jnp.clip(jnp.searchsorted(grid, pts, side="right") - 1, 0, N - 2)
        dg = grid[i0 + 1] - grid[i0]
        f  = jnp.clip((pts - grid[i0]) / jnp.where(dg > 0, dg, 1.0), 0.0, 1.0)
        return i0, f

    i0, ft = _frac(t_grid, t_pts, Nt)
    j0, fx = _frac(x_grid, x_pts, Nx)
    k0, fy = _frac(y_grid, y_pts, Ny)

    return (
        (1-ft)*(1-fx)*(1-fy) * U[i0,   j0,   k0  ] +
        (1-ft)*(1-fx)*   fy  * U[i0,   j0,   k0+1] +
        (1-ft)*   fx *(1-fy) * U[i0,   j0+1, k0  ] +
        (1-ft)*   fx *   fy  * U[i0,   j0+1, k0+1] +
           ft *(1-fx)*(1-fy) * U[i0+1, j0,   k0  ] +
           ft *(1-fx)*   fy  * U[i0+1, j0,   k0+1] +
           ft *   fx *(1-fy) * U[i0+1, j0+1, k0  ] +
           ft *   fx *   fy  * U[i0+1, j0+1, k0+1]
    )


# ─────────────────────────────────────────────────────────────────────────── #
#  Stationary (spatial-only) bilinear interpolation                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _jax_interp_stationary_1d(U, x_grid, x_pts):
    """Differentiable linear interpolation on a stationary (x,) grid.

    Args:
        U:      ``(Nx,)`` JAX array.
        x_grid: ``(Nx,)`` spatial nodes.
        x_pts:  ``(N,)`` query coords.

    Returns:
        ``(N,)`` interpolated values.
    """
    import jax.numpy as jnp
    Nx = U.shape[0]
    j0 = jnp.clip(jnp.searchsorted(x_grid, x_pts, side="right") - 1, 0, Nx - 2)
    dx = x_grid[j0 + 1] - x_grid[j0]
    fx = jnp.clip((x_pts - x_grid[j0]) / jnp.where(dx > 0, dx, 1.0), 0.0, 1.0)
    return (1 - fx) * U[j0] + fx * U[j0 + 1]


def _jax_interp_stationary_2d(U, x_grid, y_grid, x_pts, y_pts):
    """Differentiable bilinear interpolation on a stationary (x, y) grid.

    Args:
        U:      ``(Nx, Ny)`` JAX array.
        x_grid: ``(Nx,)`` spatial nodes in x.
        y_grid: ``(Ny,)`` spatial nodes in y.
        x_pts, y_pts: ``(N,)`` query coords.

    Returns:
        ``(N,)`` interpolated values.
    """
    import jax.numpy as jnp
    Nx, Ny = U.shape

    def _frac(grid, pts, N):
        i0 = jnp.clip(jnp.searchsorted(grid, pts, side="right") - 1, 0, N - 2)
        dg = grid[i0 + 1] - grid[i0]
        f  = jnp.clip((pts - grid[i0]) / jnp.where(dg > 0, dg, 1.0), 0.0, 1.0)
        return i0, f

    j0, fx = _frac(x_grid, x_pts, Nx)
    k0, fy = _frac(y_grid, y_pts, Ny)

    return (
        (1 - fx) * (1 - fy) * U[j0,     k0    ] +
        (1 - fx) *      fy  * U[j0,     k0 + 1] +
             fx  * (1 - fy) * U[j0 + 1, k0    ] +
             fx  *      fy  * U[j0 + 1, k0 + 1]
    )
