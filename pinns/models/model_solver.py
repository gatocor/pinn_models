"""
pinns/models/model_solver.py — Spectral model solver.

:class:`ModelSolver` couples a :class:`~pinns.domain.DomainCubic` (geometry)
with a spectral discretisation (``shape``, ``bc``) and an owned integrator.
The spectral infrastructure — grid nodes, wavenumbers, K², and forward/inverse
transforms — lives directly on the model.

* ``add_parameter(name, value)``  — unified parameter registration.
* ``solve(t_obs)``                — direct forward solve.
* ``apply(params, X)``           — differentiable evaluation at scattered
                                   ``(x, t)`` points; mirrors ``network.apply``.

Operator function signatures use a **flat** ``p`` dict::

    model.set_linear_op(
        lambda K2, p: {"u": 1j * p["mu"]**2 * model.k * K2}
    )
    def nonlinear(state_hat, p):
        u = model.inverse(state_hat["u"])
        return {"u": -1j * p["eta"] / 2 * model.k * model.forward(u*u)}

Usage::

    domain = pinns.DomainCubic(space=[(-1.0, 1.0)], time=(0.0, 1.0))
    integrator = pinns.IntegratorETD2RK(dt=5e-4, checkpoint=True)
    model = pinns.ModelSolver(domain, ["u"], integrator, shape=64)
    model.set_linear_op(lambda K2, p: {"u": 1j * p["mu"]**2 * model.k * K2})
    model.set_nonlinear_op(nonlinear)
    model.add_parameter("eta", eta_val)
    model.add_parameter("mu",  0.02)
    model.add_initial(jnp.cos(jnp.pi * model.x))

    # Direct solve
    U = model.solve(t_obs=t_ref)

    # Differentiable apply (for Dataset-based inverse)
    X_obs = ...  # (N, 2) columns [x, t]
    V = model.apply({"mu": jnp.array(0.02)}, X_obs)  # (N, 1)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

__all__ = ["ModelSolver"]

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
#  ModelSolver                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class ModelSolver:
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

    After construction the model exposes the spectral grid as attributes:
        :attr:`x`, :attr:`y`  — physical coordinate arrays.
        :attr:`k`             — wavenumber array (1-D) or list of 1-D arrays.
        :attr:`K2`            — eigenvalue array of ``-Δ``, shape ``(*shape)``.
        :meth:`forward`       — physical → spectral transform.
        :meth:`inverse`       — spectral → physical transform.
    """

    _VALID_BC = ("periodic", "dirichlet", "neumann")

    # Marker for integrator duck-type check
    _is_solver_problem: bool = True

    def __init__(
        self,
        domain,
        state_names: Sequence[str],
        integrator,
        shape: Union[int, Sequence[int]],
        bc: str = "periodic",
    ):
        from ..domain.domain_cubic import DomainCubic
        from .integrators.integrator_base import Integrator

        if not isinstance(domain, DomainCubic):
            raise TypeError(
                f"ModelSolver expects a DomainCubic, got {type(domain).__name__}."
            )
        if not isinstance(integrator, Integrator):
            raise TypeError(
                f"ModelSolver expects an Integrator, got {type(integrator).__name__}."
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

        # ── Other model state ─────────────────────────────────────────────
        self.state_names: List[str] = list(state_names)
        self.n_states: int = len(self.state_names)
        self._integrator = integrator

        # ── flat parameter storage (no fixed/inferred split) ─────────────
        self._params: Dict[str, Any] = {}
        # Trainable subset set by the Trainer (None = expose all _params).
        self._trainable_params: Optional[Dict[str, Any]] = None

        # ── spectral operators ───────────────────────────────────────────
        self._linear_op:    Optional[Callable] = None
        self._nonlinear_op: Optional[Callable] = None

        # ── initial conditions (physical space) ──────────────────────────
        self._initial: Optional[Dict[str, np.ndarray]] = None

        # ── observations (for direct solve without apply) ─────────────────
        self._obs_times: Optional[np.ndarray] = None
        self._obs_data:  Optional[Dict[str, np.ndarray]] = None
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
            else:
                grids.append(np.linspace(xmin, xmax, N))
        return grids

    def _build_axes(self) -> List[np.ndarray]:
        """1-D spectral wavenumber/eigenvalue axis per dimension."""
        axes = []
        for i, (L, N) in enumerate(zip(self._lengths, self.shape)):
            if self.bc == "periodic":
                k = np.fft.fftfreq(N, d=L / N) * (2.0 * np.pi)
            elif self.bc == "dirichlet":
                k = np.arange(1, N + 1) * np.pi / L
            else:  # neumann
                k = np.arange(N) * np.pi / L
            axes.append(k)
        return axes

    def _build_K2(self):
        """N-D eigenvalue array of ``-Δ`` via Kronecker sum."""
        import jax.numpy as jnp
        k2_1d = [jnp.array(k ** 2) for k in self._axes]
        K2 = k2_1d[0]
        for kk in k2_1d[1:]:
            K2 = K2[..., None] + kk[None, :]
        return K2

    # ──────────────────────────────────────────────────────────────────── #
    #  Forward / inverse transforms                                        #
    # ──────────────────────────────────────────────────────────────────── #

    def forward(self, u):
        """Physical space → spectral coefficients."""
        if self.bc == "periodic":
            import jax.numpy as jnp
            return jnp.fft.fftn(u)
        elif self.bc == "dirichlet":
            return self._dst1n(u)
        else:
            return self._dctn(u)

    def inverse(self, u_hat):
        """Spectral coefficients → physical space."""
        if self.bc == "periodic":
            import jax.numpy as jnp
            return jnp.fft.ifftn(u_hat).real
        elif self.bc == "dirichlet":
            return self._idst1n(u_hat)
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
    ) -> "ModelSolver":
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

    def set_linear_op(self, fn: Callable) -> "ModelSolver":
        """Register the linear spectral operator.

        ``fn(K2, p) → {state_name: eigenvalue_array}``

        ``p`` is a **flat** dict — use ``p["mu"]`` not ``p["infer"]["mu"]``.
        """
        self._linear_op = fn
        return self

    def set_nonlinear_op(self, fn: Callable) -> "ModelSolver":
        """Register the nonlinear spectral operator.

        ``fn(state_hat, p) → {state_name: spectral_array}``

        ``p`` is a **flat** dict — use ``p["eta"]`` directly.
        """
        self._nonlinear_op = fn
        return self

    # ──────────────────────────────────────────────────────────────────── #
    #  Initial conditions and observations                                  #
    # ──────────────────────────────────────────────────────────────────── #

    def add_initial(self, *arrays, **kwargs) -> "ModelSolver":
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

    def add_observations(
        self,
        t_obs,
        data: Union[Dict[str, np.ndarray], List[np.ndarray]],
    ) -> "ModelSolver":
        """Set observation data.  Required only for direct :meth:`solve` without
        explicit ``t_obs`` argument; :meth:`apply` uses stored ``_obs_times``.
        """
        self._obs_times = np.asarray(t_obs)
        if isinstance(data, dict):
            self._obs_data = {k: np.asarray(v) for k, v in data.items()}
        else:
            self._obs_data = {
                n: np.asarray(a) for n, a in zip(self.state_names, data)
            }
        return self

    # ──────────────────────────────────────────────────────────────────── #
    #  Internal helpers required by integrators                            #
    # ──────────────────────────────────────────────────────────────────── #

    def _build_params(
        self, inferred_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return a **flat** ``p`` dict for operator functions.

        Parameters in ``inferred_override`` receive their override value
        (gradient flows through them).  All other parameters are wrapped with
        ``jax.lax.stop_gradient`` so they are treated as constants.

        Args:
            inferred_override: Dict of trainable parameter values coming from
                               the optimiser.  Keys that do not appear here
                               are frozen.

        Returns:
            Flat dict ``{name: value_or_stop_gradient(value)}``.
        """
        import jax
        override = inferred_override or {}
        p: Dict[str, Any] = {}
        for k, v in self._params.items():
            if k in override:
                p[k] = override[k]
            else:
                p[k] = jax.lax.stop_gradient(v)
        return p

    def get_initial_hat(self) -> Dict[str, Any]:
        """Return initial state in spectral space (called by integrators)."""
        if self._initial is None:
            raise RuntimeError("Call add_initial() before solving.")
        import jax.numpy as jnp
        return {
            name: self.forward(jnp.array(arr))
            for name, arr in self._initial.items()
        }

    def _validate(self) -> None:
        """Raise if the model is not fully configured (called by integrators)."""
        errors = []
        if self._linear_op is None:
            errors.append("linear_op not set — call set_linear_op().")
        if self._nonlinear_op is None:
            errors.append("nonlinear_op not set — call set_nonlinear_op().")
        if self._initial is None:
            errors.append("initial condition not set — call add_initial().")
        if errors:
            raise RuntimeError(
                "ModelSolver is not fully configured:\n  " + "\n  ".join(errors)
            )

    # ──────────────────────────────────────────────────────────────────── #
    #  Main interface                                                       #
    # ──────────────────────────────────────────────────────────────────── #

    def solve(
        self,
        t_obs=None,
        inferred_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Forward-solve and return ``{state_name: (Nt, *shape)}`` (physical space).

        Args:
            t_obs:           Observation times.  Falls back to stored
                             ``_obs_times`` (from :meth:`add_observations`).
            inferred_params: Trainable parameter override dict.

        Returns:
            Dict ``{state_name: jax.Array of shape (Nt, *shape)}``.
        """
        return self._integrator.solve(
            self,
            inferred_params=inferred_params,
            t_obs=t_obs if t_obs is not None else self._obs_times,
        )

    def precompute_interp_stencil(self, X) -> "ModelSolver":
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

        t_grid = np.asarray(self._obs_times)
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
        params: Dict[str, Any],
        X,
        params_dict=None,
    ):
        """Differentiable evaluation at scattered ``(x[, y, ...], t)`` points.

        Mirrors the ``network.apply(params, X[, params_dict])`` interface so
        that Trainer can use ModelSolver and a neural network interchangeably.
        The optional ``params_dict`` argument is accepted for API compatibility
        but ignored (ModelSolver handles its own parameter dict internally).

        The full chain ``params → integrator.solve → U → bilinear_interp``
        is differentiable end-to-end (gradients flow through U via JAX).

        Args:
            params: Trainable parameter values, e.g. ``{"mu": jnp.array(0.02)}``.
                    Parameters absent from this dict are frozen via
                    ``stop_gradient``.
            X:      ``(N, n_spatial + 1)`` array.  Columns are
                    ``[x, [y, ...], t]`` — spatial coords first, time last.
                    Matches the :class:`~pinns.domain.DomainCubic` convention.

        Returns:
            ``(N, n_states)`` JAX array.  Column order matches
            :attr:`state_names`.
        """
        import jax.numpy as jnp

        if self._obs_times is None:
            raise RuntimeError(
                "apply() requires observation times.  Either call "
                "model.add_observations(t_obs, ...) or set _obs_times manually."
            )

        # Run the full forward solve at the stored t_obs grid
        t_obs = jnp.array(self._obs_times)
        U_dict = self._integrator.solve(self, inferred_params=params, t_obs=self._obs_times)

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

    # ──────────────────────────────────────────────────────────────────── #
    #  Repr                                                                #
    # ──────────────────────────────────────────────────────────────────── #

    def __repr__(self) -> str:
        pnames = list(self._params.keys())
        return (
            f"ModelSolver(states={self.state_names}, shape={self.shape}, "
            f"bc={self.bc!r}, params={pnames}, "
            f"integrator={type(self._integrator).__name__})"
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
