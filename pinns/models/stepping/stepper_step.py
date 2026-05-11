"""Discrete time-stepping (MoL / BPTT) strategy for the Trainer / Problem."""

from __future__ import annotations

from typing import Optional
import numpy as np

__all__ = ["StepperStep"]


class StepperStep:
    """Discrete time-stepping (MoL / BPTT) strategy for the Trainer / Problem.

    Pass an *uninitialized* instance to the problem constructor; the time
    breakpoints are read from the domain's ``time_grid_positions`` array
    inside ``__init__`` / ``__post_init__``.

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
        problem = ProblemStrong(domain, ['u'], strategy=StepperStep())
        problem.add_dependency('u_prev', component=0, order=())

        # --- inside a volume / residual function ---
        def residual(x, u, params, deriv):
            dt     = params["dependencies"]["dt"]
            u_prev = params["dependencies"]["u_prev"]
            ...
    """

    def __init__(self, bptt: bool = False, n_steps: Optional[int] = None):
        if n_steps is not None and (not isinstance(n_steps, int) or n_steps < 1):
            raise ValueError("StepperStep: n_steps must be a positive int or None.")
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
            ModelBase parameter dict (may include ``params['dependencies']['dt']``
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
        """Read time breakpoints from *domain* and store as ``ts``."""
        ts = getattr(domain, "time_grid_positions", None)
        if ts is None:
            _tpts = getattr(domain, "_time_points", None)
            if _tpts is not None:
                ts = np.asarray(_tpts, dtype=float).ravel()
        if ts is None:
            raise ValueError(
                "StepperStep requires a domain with a partitioned time axis. "
                "Pass a time breakpoints array when constructing the domain."
            )
        ts = np.asarray(ts, dtype=float).ravel()
        if len(ts) < 2:
            raise ValueError(
                "StepperStep: domain time_grid_positions must have at least 2 breakpoints."
            )
        if not np.all(np.diff(ts) > 0):
            raise ValueError(
                "StepperStep: domain time_grid_positions must be strictly increasing."
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
        """Number of time steps to train."""
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
                "StepperStep.dt: steps are not uniform — use .dts instead."
            )
        return float(dts[0])

    @property
    def is_uniform(self) -> bool:
        """``True`` if all time steps have the same size."""
        dts = np.diff(self._ts)
        return bool(np.allclose(dts, dts[0]))

    def __repr__(self) -> str:
        if self._ts is None:
            return f"StepperStep(bptt={self.bptt}, n_steps={self._n_steps_override}, uninitialized)"
        return (
            f"StepperStep(bptt={self.bptt}, "
            f"t_min={self.t_min}, t_max={self.t_max}, "
            f"n_steps={self.n_steps}, uniform={self.is_uniform})"
        )
