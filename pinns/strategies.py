"""
Training strategies for Physics-Informed Neural Networks.

Each strategy is a lightweight configuration object held by a
:class:`~pinns.model.Model` instance.  The Trainer inspects
``problem.model.spatial`` / ``problem.model.temporal`` at compile time to
determine how to build the loss function and training loop.

Available strategies
--------------------
* :class:`StrategyUnique` — standard single-network PINN (default spatial strategy).
* :class:`StrategyFB`     — Finite-Basis PINN (domain decomposition + window functions).
* :class:`StrategyX`      — Extended PINN (non-overlapping subdomains + interface residuals).
* :class:`StrategyStep`   — discrete time-stepping (MoL / BPTT).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import jax.numpy as jnp


__all__ = [
    "StrategyUnique",
    "StrategyFB",
    "StrategyX",
    "StrategyStep",
    "register_interface_loss",
]


class StrategyUnique:
    """Standard single-network PINN strategy.

    The default spatial strategy: one network trained on the entire domain
    using a weighted sum of interior, boundary, and initial-condition
    residuals.  Used automatically when no explicit spatial strategy is passed
    to :class:`~pinns.model.Model`.

    Example::

        model = Model(network)          # StrategyUnique is the default
        model = Model(network, spatial=StrategyUnique())  # explicit
    """

    def setup(self, domain) -> None:
        """Called by the Problem when a model is attached.  No-op."""

    def predict(self, apply_fn, params, x, params_dict=None):
        """Forward pass: standard sequential evaluation."""
        return apply_fn(params, x, params_dict)

    def __repr__(self) -> str:
        return "StrategyUnique()"


class StrategyFB:
    """Finite-Basis PINN (FB-PINN) strategy.

    Each network covers one subdomain and its output is weighted by a smooth
    **window function** that peaks at the subdomain centre and decays to zero
    at the boundary, with the width of the transition zone controlled by
    *overlap*.  The global solution is the sum of all windowed predictions
    (partition of unity).

    Subdomain bounds are read automatically from the domain during
    :meth:`setup`, or can be specified explicitly via *xmin* / *xmax* when a
    single network covers only part of a larger domain (multi-network
    decomposition).

    Parameters
    ----------
    overlap : float
        Controls the width of the tanh transition zone as a fraction of each
        subdomain's spatial extent (0 = no smooth overlap; 0.5 = default).
    continuity_weight : float
        Weight on the interface-continuity penalty used by the Trainer.
        Default ``1.0``.
    xmin, xmax : array-like or None
        Explicit spatial bounds of this network's subdomain.  When ``None``
        (default), bounds are taken from the domain passed to :meth:`setup`.

    Example::

        # Single network for the whole domain:
        net = Network(domain, output_dim=1, spatial=StrategyFB(overlap=0.3))

        # Multi-network decomposition (each network gets its own bounds):
        net_left  = Network(domain, output_dim=1,
                            spatial=StrategyFB(overlap=0.3, xmin=[0], xmax=[0.5]))
        net_right = Network(domain, output_dim=1,
                            spatial=StrategyFB(overlap=0.3, xmin=[0.5], xmax=[1.0]))
    """

    def __init__(
        self,
        overlap: float = 0.5,
        continuity_weight: float = 1.0,
        xmin=None,
        xmax=None,
    ):
        if not (0.0 <= overlap < 1.0):
            raise ValueError("StrategyFB: overlap must be in [0, 1).")
        if continuity_weight < 0.0:
            raise ValueError("StrategyFB: continuity_weight must be >= 0.")
        self.overlap = float(overlap)
        self.continuity_weight = float(continuity_weight)
        self._xmin: Optional[np.ndarray] = (
            np.asarray(xmin, dtype=np.float64) if xmin is not None else None
        )
        self._xmax: Optional[np.ndarray] = (
            np.asarray(xmax, dtype=np.float64) if xmax is not None else None
        )

    def setup(self, domain) -> None:
        """Read spatial bounds from *domain* (if not already set explicitly)."""
        if self._xmin is not None:
            return  # user provided explicit bounds
        n_s = domain._spatial_dims
        self._xmin = np.asarray(domain.xmin[:n_s], dtype=np.float64)
        self._xmax = np.asarray(domain.xmax[:n_s], dtype=np.float64)

    # ------------------------------------------------------------------
    # Window function
    # ------------------------------------------------------------------

    def _window(self, x_spatial):
        """Smooth bump window: product of tanh ramps over each spatial dim.

        ``w(x) = prod_d  tanh((x_d - xmin_d)/σ_d) * tanh((xmax_d - x_d)/σ_d)``

        where ``σ_d = max(overlap * (xmax_d - xmin_d), 1e-8)``.
        The result is in ``(0, 1]`` (maximum 1 at the domain centre).
        Shape: ``(batch, 1)``.
        """
        xmin = jnp.array(self._xmin, dtype=x_spatial.dtype)
        xmax = jnp.array(self._xmax, dtype=x_spatial.dtype)
        sigma = jnp.maximum(
            self.overlap * (xmax - xmin),
            jnp.full_like(xmax, 1e-8),
        )
        w = jnp.prod(
            jnp.tanh((x_spatial - xmin) / sigma)
            * jnp.tanh((xmax - x_spatial) / sigma),
            axis=-1,
            keepdims=True,
        )
        return w

    def predict(self, apply_fn, params, x, params_dict=None):
        """Forward pass: sequential apply weighted by the subdomain window.

        Parameters
        ----------
        apply_fn :
            The network's sequential forward pass
            (``params, x, params_dict → jnp.ndarray``).
        params : dict
            Network parameter dict.
        x : jnp.ndarray  shape ``(batch, n_dims)``
            Collocation points (spatial + optional time).
        params_dict : dict or None
            Optional auxiliary dict forwarded to every layer.

        Returns
        -------
        jnp.ndarray  shape ``(batch, output_dim)``
            Network output multiplied element-wise by the window function
            evaluated on the spatial coordinates.
        """
        if self._xmin is None:
            raise RuntimeError(
                "StrategyFB.predict() called before setup(). "
                "Either call network._setup(domain) or use problem.set_model(network)."
            )
        n_s = self._xmin.shape[0]
        x_spatial = x[:, :n_s]
        y = apply_fn(params, x, params_dict)
        w = self._window(x_spatial)
        return w * y

    def __repr__(self) -> str:
        parts = [f"overlap={self.overlap}", f"continuity_weight={self.continuity_weight}"]
        if self._xmin is not None:
            parts.append(f"xmin={self._xmin.tolist()}, xmax={self._xmax.tolist()}")
        return f"StrategyFB({', '.join(parts)})"


class StrategyX:
    """Extended PINN (X-PINN) strategy.

    Decomposes the domain into **non-overlapping** subdomains.  Each network
    covers exactly one subdomain; its output is **hard-masked to zero** for
    points that lie outside those bounds.  Interface residuals (continuity of
    the solution and optional normal-flux) are added to the loss by the
    Trainer.

    Subdomain bounds are read automatically from the domain during
    :meth:`setup`, or can be specified explicitly via *xmin* / *xmax*.

    Parameters
    ----------
    interface_weight : float
        Weight on the interface-continuity residuals.  Default ``1.0``.
    flux_weight : float
        Weight on the normal-flux-continuity residuals.  Default ``0.0``.
    xmin, xmax : array-like or None
        Explicit spatial bounds of this network's subdomain.  When ``None``
        (default), bounds are taken from the domain passed to :meth:`setup`.

    Example::

        net_left  = Network(domain, output_dim=1,
                            spatial=StrategyX(interface_weight=10.0,
                                             xmin=[0], xmax=[0.5]))
        net_right = Network(domain, output_dim=1,
                            spatial=StrategyX(interface_weight=10.0,
                                             xmin=[0.5], xmax=[1.0]))
    """

    def __init__(
        self,
        interface_weight: float = 1.0,
        flux_weight: float = 0.0,
        xmin=None,
        xmax=None,
    ):
        if interface_weight < 0.0:
            raise ValueError("StrategyX: interface_weight must be >= 0.")
        if flux_weight < 0.0:
            raise ValueError("StrategyX: flux_weight must be >= 0.")
        self.interface_weight = float(interface_weight)
        self.flux_weight = float(flux_weight)
        self._xmin: Optional[np.ndarray] = (
            np.asarray(xmin, dtype=np.float64) if xmin is not None else None
        )
        self._xmax: Optional[np.ndarray] = (
            np.asarray(xmax, dtype=np.float64) if xmax is not None else None
        )

    def setup(self, domain) -> None:
        """Read spatial bounds from *domain* (if not already set explicitly)."""
        if self._xmin is not None:
            return  # user provided explicit bounds
        n_s = domain._spatial_dims
        self._xmin = np.asarray(domain.xmin[:n_s], dtype=np.float64)
        self._xmax = np.asarray(domain.xmax[:n_s], dtype=np.float64)

    def predict(self, apply_fn, params, x, params_dict=None):
        """Forward pass: output zeroed for points outside the subdomain.

        Points inside the subdomain bounds receive the normal network
        prediction; points outside receive ``0.0``.  This lets the global
        loss accumulate contributions only from the responsible network at
        each point.

        Parameters
        ----------
        apply_fn :
            The network's sequential forward pass.
        params : dict
            Network parameter dict.
        x : jnp.ndarray  shape ``(batch, n_dims)``
            Collocation points (spatial + optional time).
        params_dict : dict or None
            Optional auxiliary dict forwarded to every layer.

        Returns
        -------
        jnp.ndarray  shape ``(batch, output_dim)``
            Network output, zeroed where ``x`` lies outside ``[xmin, xmax]``.
        """
        if self._xmin is None:
            raise RuntimeError(
                "StrategyX.predict() called before setup(). "
                "Either call network._setup(domain) or use problem.set_model(network)."
            )
        n_s = self._xmin.shape[0]
        x_spatial = x[:, :n_s]
        xmin = jnp.array(self._xmin, dtype=x_spatial.dtype)
        xmax = jnp.array(self._xmax, dtype=x_spatial.dtype)
        inside = jnp.all(
            (x_spatial >= xmin) & (x_spatial <= xmax),
            axis=-1,
            keepdims=True,
        )  # (batch, 1) bool
        y = apply_fn(params, x, params_dict)
        return jnp.where(inside, y, jnp.zeros_like(y))

    def __repr__(self) -> str:
        parts = [
            f"interface_weight={self.interface_weight}",
            f"flux_weight={self.flux_weight}",
        ]
        if self._xmin is not None:
            parts.append(f"xmin={self._xmin.tolist()}, xmax={self._xmax.tolist()}")
        return f"StrategyX({', '.join(parts)})"


class StrategyStep:
    """Discrete time-stepping (MoL / BPTT) strategy.

    Replaces the old ``Stepper`` class.  Pass an *uninitialized* instance to
    the problem constructor; the time breakpoints are read from the domain's
    ``time_grid_positions`` array inside ``__init__`` / ``__post_init__``.

    The Trainer injects ``dt`` for the current step into
    ``params["dependencies"]["dt"]`` and all quantities registered with
    :meth:`~pinns.problem.ProblemStrong.add_dependency` into
    ``params["dependencies"]`` before each residual evaluation.

    Parameters
    ----------
    bptt : bool
        If ``True``, gradients flow through the entire rollout (full BPTT).
        If ``False`` (default), each step is optimised independently
        (truncated / detached stepping).
    n_steps : int or None
        Override the number of steps taken from the domain's time grid.
        ``None`` (default) uses all steps defined by the time partition.

    Time-grid properties (available after the strategy is attached to a problem)
    -----------------------------------------------------------------------------
    ts : np.ndarray
        Strictly increasing time breakpoints, shape ``(n_steps + 1,)``.
    t_min, t_max : float
        First and last breakpoints.
    n_steps : int
        Number of time steps (``len(ts) - 1``), or the overridden value.
    dts : np.ndarray
        Per-step sizes, shape ``(n_steps,)``.
    dt : float
        Uniform step size; raises if steps are non-uniform.
    is_uniform : bool
        ``True`` if all steps have the same size.

    Example::

        domain = DomainCubic(...)  # must have a partitioned time axis
        problem = ProblemStrong(domain, ['u'], strategy=StrategyStep())
        problem.add_dependency('u_prev', component=0, order=())

        # --- inside a volume / residual function ---
        def residual(x, u, params, deriv):
            dt     = params["dependencies"]["dt"]
            u_prev = params["dependencies"]["u_prev"]
            ...
    """

    def __init__(self, bptt: bool = False, n_steps: Optional[int] = None):
        if n_steps is not None and (not isinstance(n_steps, int) or n_steps < 1):
            raise ValueError("StrategyStep: n_steps must be a positive int or None.")
        self.bptt = bool(bptt)
        self._n_steps_override = n_steps
        self._ts: Optional[np.ndarray] = None  # populated by _init_from_domain

    def setup(self, domain) -> None:
        """Read ``time_grid_positions`` from *domain* and initialise the time grid.

        Called automatically by the Problem constructor.  Idempotent — calling
        it a second time on the same domain is safe.

        Raises ``ValueError`` if the domain has no partitioned time axis.
        """
        if self._ts is not None:
            return  # already initialised
        self._init_from_domain(domain)

    def predict(self, apply_fn, params, x, params_dict=None):
        """Forward pass for one time slice.

        The time dimension is already embedded in ``x`` (last spatial+time
        column).  Stepping and ``dt`` injection are handled by the Trainer;
        this method simply runs the sequential network forward pass.

        Parameters
        ----------
        apply_fn :
            The network's sequential forward pass.
        params : dict
            Network parameter dict (may include ``params['dependencies']['dt']``
            and ``params['dependencies']['u_prev']`` set by the Trainer).
        x : jnp.ndarray  shape ``(batch, n_dims)``
            Collocation points at the current time step.
        params_dict : dict or None
            Auxiliary dict passed through to every layer.

        Returns
        -------
        jnp.ndarray  shape ``(batch, output_dim)``
        """
        return apply_fn(params, x, params_dict)

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _init_from_domain(self, domain) -> None:
        """Read time breakpoints from *domain* and store as ``ts``.

        Supports both :class:`~pinns.domain.DomainCubic` (which exposes
        ``time_grid_positions``) and :class:`~pinns.domain.DomainMesh`
        (which exposes ``_time_points`` when ``time=`` is given as an array).

        Raises ``ValueError`` if the domain has no partitioned time axis.
        """
        ts = getattr(domain, "time_grid_positions", None)
        if ts is None:
            # DomainMesh discrete-time path
            _tpts = getattr(domain, "_time_points", None)
            if _tpts is not None:
                ts = np.asarray(_tpts, dtype=float).ravel()
        if ts is None:
            raise ValueError(
                "StrategyStep requires a domain with a partitioned time axis. "
                "Pass a time breakpoints array when constructing the domain."
            )
        ts = np.asarray(ts, dtype=float).ravel()
        if len(ts) < 2:
            raise ValueError(
                "StrategyStep: domain time_grid_positions must have at least 2 breakpoints."
            )
        if not np.all(np.diff(ts) > 0):
            raise ValueError(
                "StrategyStep: domain time_grid_positions must be strictly increasing."
            )
        self._ts = ts

    # ------------------------------------------------------------------
    # Time-grid properties
    # ------------------------------------------------------------------

    @property
    def ts(self) -> Optional[np.ndarray]:
        """Full breakpoints array, or ``None`` if not yet initialized."""
        return self._ts

    @property
    def t_min(self) -> float:
        """First time point (initial-condition time)."""
        return float(self._ts[0])

    @property
    def t_max(self) -> float:
        """Last time point."""
        return float(self._ts[-1])

    @property
    def n_steps(self) -> int:
        """Number of time steps to train.

        Returns ``_n_steps_override`` if set, otherwise ``len(ts) - 1``.
        """
        if self._n_steps_override is not None:
            return self._n_steps_override
        return len(self._ts) - 1

    @property
    def dts(self) -> np.ndarray:
        """Per-step sizes, shape ``(len(ts) - 1,)``."""
        return np.diff(self._ts)

    @property
    def dt(self) -> float:
        """Uniform step size.  Raises ``ValueError`` if steps are non-uniform."""
        dts = np.diff(self._ts)
        if not np.allclose(dts, dts[0]):
            raise ValueError(
                "StrategyStep.dt: steps are not uniform — use .dts instead."
            )
        return float(dts[0])

    @property
    def is_uniform(self) -> bool:
        """``True`` if all time steps have the same size."""
        dts = np.diff(self._ts)
        return bool(np.allclose(dts, dts[0]))

    def __repr__(self) -> str:
        if self._ts is None:
            return f"StrategyStep(bptt={self.bptt}, n_steps={self._n_steps_override}, uninitialized)"
        return (
            f"StrategyStep(bptt={self.bptt}, "
            f"t_min={self.t_min}, t_max={self.t_max}, "
            f"n_steps={self.n_steps}, uniform={self.is_uniform})"
        )


# Tuple of all valid strategy types — used for isinstance checks.
_STRATEGIES = (StrategyUnique, StrategyFB, StrategyX, StrategyStep)
_SPATIAL_STRATEGIES = (StrategyUnique, StrategyFB, StrategyX)
_TEMPORAL_STRATEGIES = (StrategyStep,)


def register_interface_loss(
    net_a,
    net_b,
    x_interface=None,
    name: str = "interface",
    weight: Optional[float] = None,
) -> None:
    """
    Register an X-PINN interface-continuity loss on two :class:`~pinns.network.Network`
    instances that use :class:`StrategyX`.

    Adds a :class:`~pinns.network.NetworkLoss` to **both** networks so that
    each one tries to match the other's prediction at the interface.  The
    gradient flows only through the *owning* network’s parameters;
    the other network’s output is treated as a fixed target via
    :func:`jax.lax.stop_gradient` (i.e. alternating-optimisation style).

    Parameters
    ----------
    net_a, net_b : Network
        The two networks sharing an interface.  Both must have a
        :class:`StrategyX` spatial strategy.
    x_interface : array-like, shape ``(n_pts, n_dims)``
        Collocation points on the shared interface.  These are stored as
        a fixed constant in the :class:`~pinns.network.NetworkLoss` and
        reused at every training iteration.
        When ``None``, uses the PDE collocation batch (less precise—not
        recommended).
    name : str
        Name prefix for the two loss terms (suffixed with ``'_a'`` and
        ``'_b'``).
    weight : float or None
        Loss weight; defaults to ``net_a.spatial.interface_weight``.

    Example
    -------
    ::

        net_left  = Network(domain, output_dim=1,
                            spatial=StrategyX(xmin=[0.0], xmax=[0.5]))
        net_right = Network(domain, output_dim=1,
                            spatial=StrategyX(xmin=[0.5], xmax=[1.0]))
        # Both networks must be set up (i.e. attached to a problem) before calling.
        x_iface = np.array([[0.5, t] for t in np.linspace(0, 1, 200)])
        register_interface_loss(net_left, net_right, x_interface=x_iface)

    Notes
    -----
    * This function imports
      :class:`~pinns.network.NetworkLoss` at call time to avoid a
      circular import at module load.
    * Both networks are treated symmetrically: ``net_a`` minimises
      ``||u_a(x) - stop_grad(u_b(x))||^2`` and vice-versa.
    """
    from pinns.network import NetworkLoss

    import jax
    import jax.numpy as _jnp

    if not isinstance(getattr(net_a, 'spatial', None), StrategyX):
        raise TypeError(
            "register_interface_loss: net_a.spatial must be a StrategyX instance."
        )
    if not isinstance(getattr(net_b, 'spatial', None), StrategyX):
        raise TypeError(
            "register_interface_loss: net_b.spatial must be a StrategyX instance."
        )

    _weight = float(weight) if weight is not None else net_a.spatial.interface_weight
    _x = _jnp.asarray(x_interface, dtype=_jnp.float32) if x_interface is not None else None

    # Keep references to the strategies and sequential apply functions so the
    # closures stay self-contained.
    _spatial_a = net_a.spatial
    _spatial_b = net_b.spatial
    _seq_a = net_a._sequential_apply
    _seq_b = net_b._sequential_apply

    def _fn_a(params_a, x):
        """net_a tries to match net_b (stop-grad on net_b params)."""
        u_a = _spatial_a.predict(_seq_a, params_a, x)
        u_b = jax.lax.stop_gradient(
            _spatial_b.predict(_seq_b, net_b.params, x)
        )
        return _jnp.mean((u_a - u_b) ** 2)

    def _fn_b(params_b, x):
        """net_b tries to match net_a (stop-grad on net_a params)."""
        u_a = jax.lax.stop_gradient(
            _spatial_a.predict(_seq_a, net_a.params, x)
        )
        u_b = _spatial_b.predict(_seq_b, params_b, x)
        return _jnp.mean((u_a - u_b) ** 2)

    net_a.add_network_loss(NetworkLoss(name=f"{name}_a", fn=_fn_a, weight=_weight, x=_x))
    net_b.add_network_loss(NetworkLoss(name=f"{name}_b", fn=_fn_b, weight=_weight, x=_x))
