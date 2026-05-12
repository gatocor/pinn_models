"""
ProblemStrong — Physics-Informed Neural ModelBase problem definition.

Supports both :class:`~pinns.domain.DomainCubic` and
:class:`~pinns.domain.DomainMesh` domains.  Backend: JAX only.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Any, List, Optional, Union

from ..domain import DomainCubic, DomainMesh
from ..models.partition import PartitionFB, PartitionX
from ..models.stepping  import StepperStep
_STRATEGIES = (PartitionFB, PartitionX, StepperStep)
from ..models.model_base import ModelBase

# ---------------------------------------------------------------------------
# Internal container for a single residual term
# ---------------------------------------------------------------------------

class _ResidualTerm:
    """Stores one interior / boundary / initial residual specification."""

    def __init__(self, fn, name, kind, region, lagrange,
                 points=None, output_idx=None, eq_idx=None):
        self.fn = fn
        self.name = name
        self.kind = kind
        self.region = region
        self.lagrange = lagrange
        self.x_data = points          # fixed input coordinates (points terms only)
        self.u_data = None            # fixed target values   (points terms only)
        self.output_idx = output_idx  # int index of the output this term applies to, or None (all)
        self.eq_idx = eq_idx          # int column of fn's return array this term corresponds to, or None
        self.rhs = None               # RHS value/callable for Dirichlet/Neumann/Robin
        self.alpha = None             # α coefficient for Robin BC
        self.beta = None              # β coefficient for Robin BC

    def __repr__(self):
        extra = ''
        if self.output_idx is not None:
            extra += f', output_idx={self.output_idx}'
        if self.eq_idx is not None:
            extra += f', eq_idx={self.eq_idx}'
        return (
            f"_ResidualTerm(name={self.name!r}, kind={self.kind!r}, "
            f"region={self.region!r}, lagrange={self.lagrange}{extra})"
        )

# ---------------------------------------------------------------------------
# Dependency descriptor for discrete time-stepping
# ---------------------------------------------------------------------------

class _Dependency:
    """Declares a quantity the Strategy must evaluate from the previous step's
    network and store in ``pars['fixed']`` before the next step.

    Mirrors the ``derivative(u, x, component, order)`` API:

    * ``component`` — output index.
    * ``order``     — tuple of input-dimension indices; ``()`` means plain
      field evaluation (no derivative).
    """

    def __init__(self, name: str, component: int, order: tuple):
        self.name = name
        self.component = component
        self.order = tuple(order)

    def __repr__(self):
        return (
            f"_Dependency(name={self.name!r}, component={self.component}, "
            f"order={self.order})"
        )

# ---------------------------------------------------------------------------
# ProblemStrong
# ---------------------------------------------------------------------------

class ProblemStrong:
    """
    A strong-form PINN problem definition.

    Combines a domain (:class:`~pinns.domain.DomainCubic` or
    :class:`~pinns.domain.DomainMesh`) with physics residuals, boundary
    conditions, and optional initial conditions.  Backend: **JAX only**.

    All residual functions receive **four** positional arguments::

        def residual(x, u, pars, derivative):
            ...

    * **x** — collocation points ``(n_points, n_dims)``.
    * **u** — network output ``(n_points, n_outputs)``.
    * **pars** — ``{"fixed": {}, "inferred": {}, "internal": {…}}``.
    * **derivative** — autodiff helper (typically ``pinns.derivative``).
    * Returns a scalar residual per point, or an array ``(n_points, n_eqs)``.

    Terms are registered with :meth:`add_inner`, :meth:`add_boundary`, and
    :meth:`add_initial`.  Each term references a **named region** that was
    previously registered on the domain with ``domain.add_inner(…)`` /
    ``domain.add_boundary(…)``.

    Input dimension names are derived automatically from the domain:
    spatial axes are labelled ``x``, ``y``, ``z`` (in order) and the time
    axis (if present) is labelled ``t``.

    Fixed problem parameters and an analytical solution can be registered
    with :meth:`add_fixed`, :meth:`add_inferred`, and :meth:`add_solution`.

    Args:
        domain: :class:`~pinns.domain.DomainCubic` or
            :class:`~pinns.domain.DomainMesh` instance.
        output_names: Names for the network output components.  Length
            determines ``n_outputs``.

    Example::

        from pinns.domain import DomainMesh
        from pinns import meshes
        from pinns.problem import ProblemStrong
        import jax.numpy as jnp

        domain = DomainMesh(meshes.disk(), time=(0.0, 1.0))
        domain.add_boundary(lambda v: v[:, 0] < -0.99, name='left_arc')

        problem = ProblemStrong(domain=domain, output_names=['u'])
        problem.add_fixed('alpha', 0.01)

        def heat(x, u, pars, diff):
            a = pars['fixed']['alpha']
            return diff(u, x, 0, (2,)) - a * (diff(u, x, 0, (0, 0)) + diff(u, x, 0, (1, 1)))

        problem.add_inner(heat, name='pde')

        def dirichlet(x, u, pars, diff):
            return u[:, 0]

        problem.add_dirichlet(dirichlet, name='left_bc', region='left_arc')

        def ic(x, u, pars, diff):
            return u[:, 0] - jnp.sin(jnp.pi * x[:, 0])

        problem.add_initial(ic, name='ic')
    """

    # Default labels for spatial axes (time is always 't').
    _SPATIAL_LABELS = ('x', 'y', 'z', 'x4', 'x5', 'x6', 'x7', 'x8')

    def __init__(
        self,
        domain,
        output_names,
        strategy=None,
        stepper=None,
    ):
        if not isinstance(domain, (DomainCubic, DomainMesh)):
            raise TypeError(
                "domain must be a DomainCubic or DomainMesh instance, "
                f"got {type(domain).__name__!r}."
            )

        self.domain = domain
        self.n_dims = domain.n_dims
        self.fixed_params: Dict[str, Any] = {}
        self.inferred_params: Dict[str, Any] = {}
        self.solution = None

        # ── input_names (auto-derived) ───────────────────────────────────
        has_time = getattr(domain, '_t_min', None) is not None
        n_spatial = getattr(domain, '_spatial_dims', self.n_dims - int(has_time))
        spatial_labels = list(self._SPATIAL_LABELS[:n_spatial])
        self.input_names: List[str] = spatial_labels + (['t'] if has_time else [])

        # ── output_names ─────────────────────────────────────────────────
        if not output_names:
            raise ValueError("output_names must not be empty.")
        self.output_names = list(output_names)
        self.n_outputs = len(self.output_names)

        # ── term registry ────────────────────────────────────────────────
        self._terms: List[_ResidualTerm] = []
        self._model: Optional['ModelBase'] = None

        # ── strategy ──────────────────────────────────────────────────────
        if strategy is not None and not isinstance(strategy, _STRATEGIES):
            raise TypeError(
                f"strategy must be a StrategyBase, PartitionFB, PartitionX, or "
                f"StepperStep instance, got {type(strategy).__name__!r}."
            )
        self._strategy: Optional[_STRATEGIES] = strategy

        # ── step dependencies (discrete schemes) ─────────────────────────
        self.dependencies: List[_Dependency] = []

        # ── stepper (StepperStep acts as the stepper) ─────────────────────
        # For backward compat, a bare StepperStep instance passed as
        # ``stepper=`` is also accepted and silently moved to ``strategy``.
        if stepper is not None:
            if not isinstance(stepper, StepperStep):
                raise TypeError(
                    f"stepper must be a StepperStep instance, got {type(stepper).__name__!r}."
                )
            if strategy is None:
                strategy = stepper
                self._strategy = strategy
        # Call setup(domain) on the strategy — every strategy implements this.
        if self._strategy is not None:
            self._strategy.setup(domain)
        self._stepper: Optional[StepperStep] = (
            self._strategy if isinstance(self._strategy, StepperStep) else None
        )

    # ------------------------------------------------------------------ #
    #  Model attachment                                                   #
    # ------------------------------------------------------------------ #

    def set_model(self, model: 'ModelBase') -> 'ProblemStrong':
        """Attach a :class:`~pinns.modelbase.ModelBase` to this problem.

        This wires the network’s spatial and temporal strategies to the domain
        and stores the network reference for the Trainer to inspect.

        Architecture-driven extra losses (e.g. X-PINN interface terms)
        are registered directly on the network via
        :meth:`~pinns.modelbase.ModelBase.add_network_loss` and are independent
        of the problem.  The problem only holds physics residuals.

        Args:
            model: A :class:`~pinns.modelbase.ModelBase` instance.

        Returns:
            ``self`` for method chaining.
        """
        if not isinstance(model, ModelBase):
            raise TypeError(
                f"model must be a ModelBase instance, got {type(model).__name__!r}."
            )
        self._model = model
        return self

    @property
    def model(self) -> Optional['Model']:
        """The currently attached :class:`~pinns.model.Model`, or ``None``."""
        return self._model

    # ── Trainer-facing properties derived from model ──────────────────────

    @property
    def strategy(self):
        """Strategy passed to the constructor, or ``None``."""
        return self._strategy

    @property
    def stepper(self) -> Optional[StepperStep]:
        """The :class:`~pinns.strategies.StepperStep`, or ``None``."""
        return self._stepper

    # ------------------------------------------------------------------ #
    #  Output resolution helper                                          #
    # ------------------------------------------------------------------ #

    def _resolve_outputs(self, outputs):
        """Return list of (output_idx, suffix) pairs for the given outputs spec.

        * ``None``           → [(None, '')]           — only allowed when n_outputs == 1
        * ``'u'``            → [(0, '_u')]            — single named output
        * ``['u', 'v']``     → [(0, '_u'), (1, '_v')] — one term per output
        * ``0`` / ``[0, 1]`` — integer indices also accepted
        """
        if outputs is None:
            if self.n_outputs > 1:
                raise ValueError(
                    f"This problem has {self.n_outputs} outputs "
                    f"({self.output_names}). You must specify 'outputs' "
                    f"explicitly (e.g. outputs='u' or outputs={self.output_names})."
                )
            return [(None, '')]
        if isinstance(outputs, (str, int)):
            outputs = [outputs]
        result = []
        for o in outputs:
            if isinstance(o, int):
                if not (0 <= o < self.n_outputs):
                    raise ValueError(
                        f"Output index {o} out of range (n_outputs={self.n_outputs})."
                    )
                result.append((o, f'_{self.output_names[o]}'))
            else:
                if o not in self.output_names:
                    raise ValueError(
                        f"Output {o!r} not found. Available: {self.output_names}."
                    )
                idx = self.output_names.index(o)
                result.append((idx, f'_{o}'))
        # If only one output was requested suppress the suffix
        if len(result) == 1:
            result = [(result[0][0], '')]
        return result

    def add_inner(
        self,
        fn,
        name,
        region: str = 'all',
        lagrange: bool = False,
    ) -> 'ProblemStrong':
        """Register an **interior** physics residual.

        Collocation points are sampled from the domain interior or from a
        named interior region (registered beforehand with
        ``domain.add_inner(…, name=region)``).

        Args:
            fn: Residual callable ``fn(x, u, pars, derivative) -> residual``.

                * **x** — ``(n_points, n_dims)`` collocation coordinates.
                * **u** — ``(n_points, n_outputs)`` network predictions.
                * **pars** — ``{"fixed": …, "infer": {}, "internal": {…}}``.
                * **derivative** — autodiff helper (e.g. ``pinns.derivative``).
                * Returns a scalar / ``(n_points,)`` array for a single
                  equation, or ``(n_points, n_eqs)`` for multiple equations.

            name: Label(s) for this term.

                * ``str`` — single equation; one term is registered.
                * ``list[str]`` — multiple equations; one term per equation
                  is registered, each tracking a different column of the
                  function's output (``eq_idx``).

            region: Named interior region of the domain, or ``'all'``
                (default) to use the entire interior.
            lagrange: If ``True`` the trainer maintains per-point Lagrange
                multipliers for this term.  Default ``False``.

        Returns:
            ``self`` for method chaining.
        """
        if region != 'all' and isinstance(self.domain, DomainMesh):
            if region not in self.domain._inner_regions:
                raise ValueError(
                    f"Interior region {region!r} is not registered on the domain. "
                    f"Available: {list(self.domain._inner_regions.keys())}"
                )
        # Resolve equation names
        if isinstance(name, str):
            eq_pairs = [(name, None)]   # (term_name, eq_idx)
        else:
            names = list(name)
            if len(names) < 2:
                raise ValueError(
                    "When name is a list it must contain at least 2 entries "
                    "(one per equation column returned by fn)."
                )
            eq_pairs = [(n, i) for i, n in enumerate(names)]
        for term_name, eq_idx in eq_pairs:
            self._terms.append(
                _ResidualTerm(fn=fn, name=term_name, kind='inner',
                              region=region, lagrange=lagrange,
                              eq_idx=eq_idx)
            )
        return self

    def add_boundary(
        self,
        fn,
        name,
        region: str = 'all',
        lagrange: bool = False,
    ) -> 'ProblemStrong':
        """Register a generic **boundary** residual (custom BC).

        Use this when the boundary condition does not fit neatly into
        Dirichlet / Neumann / Robin categories.  For standard BCs prefer
        :meth:`add_dirichlet`, :meth:`add_neumann`, or :meth:`add_robin`.

        Behaviour mirrors :meth:`add_inner`:

        * ``name`` can be a ``str`` (single equation) or a ``list[str]``
          (one name per equation column returned by ``fn``).
        * Collocation points are sampled from the boundary or from a named
          boundary region registered with ``domain.add_boundary(…, name=region)``.

        Args:
            fn: Residual callable ``fn(x, u, pars, derivative) -> residual``.
            name: ``str`` or ``list[str]`` — label(s) for this term.
            region: Named boundary region, or ``'all'`` (default).
            lagrange: Enable per-point Lagrange multipliers.

        Returns:
            ``self`` for method chaining.
        """
        if region != 'all' and isinstance(self.domain, DomainMesh):
            if region not in self.domain._boundary_regions:
                raise ValueError(
                    f"Boundary region {region!r} is not registered on the domain. "
                    f"Available: {list(self.domain._boundary_regions.keys())}"
                )
        if isinstance(name, str):
            eq_pairs = [(name, None)]
        else:
            names = list(name)
            if len(names) < 2:
                raise ValueError(
                    "When name is a list it must contain at least 2 entries "
                    "(one per equation column returned by fn)."
                )
            eq_pairs = [(n, i) for i, n in enumerate(names)]
        for term_name, eq_idx in eq_pairs:
            self._terms.append(
                _ResidualTerm(fn=fn, name=term_name, kind='boundary',
                              region=region, lagrange=lagrange,
                              eq_idx=eq_idx)
            )
        return self

    _BC_KINDS = ('boundary', 'dirichlet', 'neumann', 'robin', 'periodic')

    def _add_bc(self, kind, fn, name, region, lagrange, out_idx=None):
        """Internal helper — validate region and append a BC term."""
        if kind != 'periodic' and region != 'all' and isinstance(self.domain, DomainMesh):
            if region not in self.domain._boundary_regions:
                raise ValueError(
                    f"Boundary region {region!r} is not registered on the domain. "
                    f"Available: {list(self.domain._boundary_regions.keys())}"
                )
        self._terms.append(
            _ResidualTerm(fn=fn, name=name, kind=kind,
                          region=region, lagrange=lagrange,
                          output_idx=out_idx)
        )
        return self

    def add_dirichlet(
        self,
        value,
        name: str,
        region: str = 'all',
        lagrange: bool = False,
        outputs=None,
    ) -> 'ProblemStrong':
        """Register a **Dirichlet** boundary condition  ``u = g`` (soft enforcement).

        The trainer adds a squared residual term ``||u - g||²`` to the loss.
        For **hard** enforcement (exact satisfaction by construction) use a
        :class:`~pinns.models.layers.Lifting` layer on the network instead.

        Args:
            value: The prescribed boundary value ``g``.  Two forms:

                * **scalar / array** — constant; broadcast to all collocation points.
                * **callable** ``g(x, pars) -> array`` — spatially varying RHS.

            name: Unique label for this term.
            region: Named boundary region from ``domain.add_boundary(…)``,
                or ``'all'`` (default) for the entire boundary.
            lagrange: Enable per-point Lagrange multipliers.
            outputs: Which output(s) this term applies to.  ``None`` (default)
                is only allowed when ``n_outputs == 1``.

        Returns:
            ``self`` for method chaining.
        """
        for out_idx, suffix in self._resolve_outputs(outputs):
            term = _ResidualTerm(fn=None, name=name + suffix,
                                 kind='dirichlet', region=region,
                                 lagrange=lagrange, output_idx=out_idx)
            term.rhs = value
            self._terms.append(term)
        return self

    def add_neumann(
        self,
        value,
        name: str,
        region: str = 'all',
        lagrange: bool = False,
        outputs=None,
    ) -> 'ProblemStrong':
        """Register a **Neumann** boundary condition  ``n·∇u = g``.

        Args:
            value: The prescribed normal-flux value ``g``.  Accepted forms:

                * **scalar / array** — constant flux.
                * **callable** ``g(x, pars) -> array`` — spatially varying
                  flux; receives boundary coordinates and params dict.

                The trainer automatically forms the residual
                ``n·∇u - g``.

            name: Unique label.
            region: Named boundary region, or ``'all'``.
            lagrange: Enable per-point Lagrange multipliers.
            outputs: Which output(s) this term applies to (see
                :meth:`add_dirichlet` for details).

        Returns:
            ``self``.
        """
        for out_idx, suffix in self._resolve_outputs(outputs):
            term = _ResidualTerm(fn=None, name=name + suffix,
                                 kind='neumann', region=region,
                                 lagrange=lagrange, output_idx=out_idx)
            term.rhs = value
            self._terms.append(term)
        return self

    def add_robin(
        self,
        alpha,
        beta,
        g,
        name: str,
        region: str = 'all',
        lagrange: bool = False,
        outputs=None,
    ) -> 'ProblemStrong':
        """Register a **Robin** boundary condition  ``α·u + β·n·∇u = g``.

        The trainer automatically forms the residual
        ``α·u + β·(n·∇u) - g``.

        Args:
            alpha: Coefficient on ``u``.  Scalar, array, or callable
                ``alpha(x, pars) -> array``.
            beta: Coefficient on the normal derivative ``n·∇u``.  Same
                accepted forms as ``alpha``.
            g: Right-hand side value.  Scalar, array, or callable
                ``g(x, pars) -> array``.
            name: Unique label.
            region: Named boundary region, or ``'all'``.
            lagrange: Enable per-point Lagrange multipliers.
            outputs: Which output(s) this term applies to (see
                :meth:`add_dirichlet` for details).

        Returns:
            ``self``.
        """
        for out_idx, suffix in self._resolve_outputs(outputs):
            term = _ResidualTerm(fn=None, name=name + suffix,
                                 kind='robin', region=region,
                                 lagrange=lagrange, output_idx=out_idx)
            term.rhs = g
            term.alpha = alpha
            term.beta = beta
            self._terms.append(term)
        return self

    _PERIODIC_AXES = ('x', 'y', 'z', 't')

    def add_periodic(
        self,
        fn,
        name: str,
        axis: str,
        lagrange: bool = False,
        outputs=None,
    ) -> 'ProblemStrong':
        """Register a **periodic** boundary condition along a coordinate axis.

        Args:
            fn: Residual callable returning the periodicity mismatch, e.g.
                ``u(x_min) - u(x_max)``.  Same signature as
                :meth:`add_dirichlet`.
            name: Unique label.
            axis: Coordinate axis to enforce periodicity on.  Must be one of
                ``'x'``, ``'y'``, ``'z'``, or ``'t'``.
            lagrange: Enable per-point Lagrange multipliers.
            outputs: Which output(s) this term applies to (see
                :meth:`add_dirichlet` for details).

        Returns:
            ``self``.

        Raises:
            TypeError: If the domain is a :class:`~pinns.domain.DomainMesh`
                (only :class:`~pinns.domain.DomainCubic` is supported).
            ValueError: If ``axis`` is not one of the allowed values.
        """
        if isinstance(self.domain, DomainMesh):
            raise TypeError(
                "add_periodic is only supported for DomainCubic domains. "
                "DomainMesh does not support axis-aligned periodicity."
            )
        if axis not in self._PERIODIC_AXES:
            raise ValueError(
                f"axis {axis!r} is not valid for add_periodic. "
                f"Choose from {self._PERIODIC_AXES}."
            )
        for out_idx, suffix in self._resolve_outputs(outputs):
            self._add_bc('periodic', fn, name + suffix, region=axis,
                         lagrange=lagrange, out_idx=out_idx)
        return self

    def add_initial(
        self,
        fn,
        name: str = 'ic',
        lagrange: bool = False,
        outputs=None,
    ) -> 'ProblemStrong':
        """Register an **initial condition** residual.

        Collocation points are sampled at ``t = t_min`` from the domain
        interior.  Requires the domain to have a time axis.

        Args:
            fn: Residual callable — same signature as :meth:`add_inner`.
            name: Unique label.  Default ``'ic'``.
            lagrange: Enable per-point Lagrange multipliers.
            outputs: Which output(s) this term applies to (see
                :meth:`add_dirichlet` for details).

        Returns:
            ``self``.

        Raises:
            ValueError: If the domain has no time axis.
        """
        has_time = (
            self.domain._t_min is not None
            if isinstance(self.domain, DomainMesh)
            else getattr(self.domain, 'has_time', False)
        )
        if not has_time:
            raise ValueError(
                "add_initial requires a time-dependent domain. "
                "Pass time=(t_min, t_max) when constructing the domain."
            )
        for out_idx, suffix in self._resolve_outputs(outputs):
            self._terms.append(
                _ResidualTerm(fn=fn, name=name + suffix, kind='initial',
                              region='initial', lagrange=lagrange,
                              output_idx=out_idx)
            )
        return self

    def add_points(
        self,
        x,
        u,
        name: str,
        lagrange: bool = False,
        outputs=None,
    ) -> 'ProblemStrong':
        """Register a **data / observation** term at fixed coordinates.

        Unlike interior or boundary terms, the collocation coordinates are
        provided explicitly and are **not** sampled from the domain.  The
        trainer minimises the mismatch between the network prediction and the
        supplied target values ``u``.  Useful for data assimilation, inverse
        problems, or fitting to measurements.

        Args:
            x: Array-like of shape ``(n_points, n_dims)`` with the fixed
                observation coordinates.
            u: Array-like of shape ``(n_points,)`` or
                ``(n_points, n_outputs)`` with the target (observed) values.
            name: Unique label used in loss weighting and logging.
            lagrange: Enable per-point Lagrange multipliers.
            outputs: Which output(s) this term applies to (see
                :meth:`add_dirichlet` for details).  When multiple outputs are
                given, ``u`` must have at least as many columns; the
                corresponding column is stored per term.

        Returns:
            ``self`` for method chaining.
        """
        import numpy as np
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if x_arr.shape[1] != self.n_dims:
            raise ValueError(
                f"x has {x_arr.shape[1]} column(s) but domain has "
                f"{self.n_dims} dimension(s)."
            )
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1:
            u_arr = u_arr[:, None]
        if u_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(
                f"x has {x_arr.shape[0]} row(s) but u has {u_arr.shape[0]}."
            )
        resolved = self._resolve_outputs(outputs)
        for out_idx, suffix in resolved:
            term = _ResidualTerm(fn=None, name=name + suffix, kind='points',
                                 region=None, lagrange=lagrange,
                                 output_idx=out_idx)
            term.x_data = x_arr
            # store only the relevant column when an output is selected
            if out_idx is not None:
                term.u_data = u_arr[:, [out_idx]]
            else:
                term.u_data = u_arr
            self._terms.append(term)
        return self

    # ------------------------------------------------------------------ #
    #  Accessors                                                          #
    # ------------------------------------------------------------------ #

    @property
    def xmin(self):
        """Domain lower bounds."""
        return self.domain.xmin

    @property
    def xmax(self):
        """Domain upper bounds."""
        return self.domain.xmax

    @property
    def inner_terms(self):
        """All registered interior terms."""
        return [t for t in self._terms if t.kind == 'inner']

    @property
    def boundary_terms(self):
        """All registered boundary-condition terms (Dirichlet, Neumann, Robin, periodic)."""
        return [t for t in self._terms if t.kind in self._BC_KINDS]

    @property
    def initial_terms(self):
        """All registered initial-condition terms."""
        return [t for t in self._terms if t.kind == 'initial']

    @property
    def points_terms(self):
        """All registered data/observation terms."""
        return [t for t in self._terms if t.kind == 'points']

    @property
    def lagrange_terms(self):
        """All terms that use Lagrange multipliers."""
        return [t for t in self._terms if t.lagrange]

    @property
    def is_stepping(self) -> bool:
        """``True`` if this problem uses a time-stepping model."""
        return self.stepper is not None or len(self.dependencies) > 0

    def validate(self) -> 'ProblemStrong':
        """Check the problem definition for consistency.

        Raises :exc:`ValueError` for any invalid configuration.  Called
        automatically by the Strategy before training begins; can also be
        called manually.

        Current checks:

        * A stepping problem (one with step dependencies) **must** declare at
          least one initial-condition term via :meth:`add_initial`.

        Returns:
            ``self`` for method chaining.
        """
        errors = []

        if self.is_stepping and not self.initial_terms:
            errors.append(
                "A stepping problem (add_dependency was called) requires at "
                "least one initial condition registered with add_initial()."
            )

        if errors:
            raise ValueError(
                "ProblemStrong validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
        return self

    def update_params(self, **kwargs) -> 'ProblemStrong':
        """Update or add **fixed** parameters in-place."""
        self.fixed_params.update(kwargs)
        return self

    def add_solution(self, fn) -> 'ProblemStrong':
        """Register a known analytical / reference solution.

        The solution is used by the trainer and evaluation utilities to
        compute errors (e.g. L² relative error) and to generate comparison
        plots.

        Args:
            fn: Callable ``fn(x, pars) -> u`` where

                * **x** — ``(n_points, n_dims)`` query coordinates.
                * **pars** — ``{"fixed": …, "inferred": {…}}`` parameter dict.
                * Returns ``(n_points,)`` or ``(n_points, n_outputs)`` values.

        Returns:
            ``self`` for method chaining.
        """
        self.solution = fn
        return self

    def add_fixed(
        self,
        name=None,
        value=None,
        **kwargs,
    ) -> 'ProblemStrong':
        """Register one or more **fixed** (constant) parameters.

        Fixed parameters are passed to every residual function as
        ``pars['fixed']`` and are **not** modified during training.

        Can be called in three ways::

            problem.add_fixed('omega', 1.0)             # single param
            problem.add_fixed(['omega', 'beta'], [1.0, 0.5])  # list
            problem.add_fixed(omega=1.0, beta=0.5)      # keyword style

        Args:
            name: Parameter name (``str``) or list of names.  Omit when
                using keyword-style (``**kwargs``).
            value: Scalar value or list matching ``name``.
            **kwargs: Key-value pairs to register (keyword-style call).

        Returns:
            ``self`` for method chaining.
        """
        # keyword style: add_fixed(omega=1.0, beta=0.5)
        if kwargs:
            for k, v in kwargs.items():
                self.fixed_params[k] = v
            return self
        if isinstance(name, str):
            names = [name]
            values = [value]
        else:
            names = list(name)
            if value is None:
                values = [None] * len(names)
            elif isinstance(value, (list, tuple)):
                values = list(value)
                if len(values) != len(names):
                    raise ValueError(
                        f"add_fixed: {len(names)} name(s) but {len(values)} value(s)."
                    )
            else:
                raise ValueError(
                    "add_fixed: when name is a list, value must be a "
                    "list/tuple of the same length, or None."
                )
        for n, v in zip(names, values):
            self.fixed_params[n] = v
        return self

    def add_inferred(
        self,
        name,
        init=0.0,
    ) -> 'ProblemStrong':
        """Register one or more **inferred** (learnable) parameters.

        Inferred parameters are optimised together with the network weights
        during training.  They are accessible in residual functions as
        ``pars['inferred']``.

        Args:
            name: Parameter name (``str``) or list of names
                (``list[str]``).
            init: Initial value for the parameter(s).  A scalar is
                broadcast to all names in a list; pass a list of the
                same length to set individual initialisations.
                Default ``0.0``.

        Returns:
            ``self`` for method chaining.
        """
        if isinstance(name, str):
            names = [name]
            inits = [init]
        else:
            names = list(name)
            if isinstance(init, (list, tuple)):
                inits = list(init)
                if len(inits) != len(names):
                    raise ValueError(
                        f"add_inferred: {len(names)} name(s) but {len(inits)} init(s)."
                    )
            else:
                inits = [init] * len(names)
        for n, v in zip(names, inits):
            self.inferred_params[n] = v
        return self

    def add_dependency(
        self,
        name: str,
        component: int = 0,
        order: tuple = (),
    ) -> 'ProblemStrong':
        """Declare a **step dependency** for discrete time-integration schemes.

        Tells the Trainer which quantities to evaluate from the previous
        step's trained network and store under ``params["dependencies"][name]``
        before optimising the next step.  ``dt`` for the current step is also
        automatically injected into ``params["dependencies"]["dt"]`` when a
        :class:`~pinns.domain.Stepper` is attached to this problem.

        The ``component`` and ``order`` arguments follow exactly the same
        convention as ``derivative(u, x, component, order)``:

        * ``component`` — index of the network output to differentiate.
        * ``order``     — tuple of input-dimension indices specifying the
          mixed partial derivative.  ``()`` (default) means plain field
          evaluation — no derivative is taken.

        Examples::

            # u_{n}(x)  — plain field, output 0
            problem.add_dependency('u_prev', component=0, order=())

            # ∂u/∂x₀ at step n
            problem.add_dependency('du_dx_prev', component=0, order=(0,))

            # ∂²u/∂x₀²  (one term of the Laplacian)
            problem.add_dependency('d2u_dx_prev', component=0, order=(0, 0))

            # v-component gradient in y for a 2-output problem
            problem.add_dependency('dv_dy_prev', component=1, order=(1,))

        The registered name must match the key used inside the residual
        function, e.g. ``params["dependencies"]["u_prev"]``.

        Args:
            name: Key under which the evaluated quantity will be stored in
                ``params["dependencies"]`` during stepping.
            component: ModelBase output index.  Default ``0``.
            order: Derivative order tuple (input-dimension indices).
                ``()`` means no derivative — evaluate the field directly.
                Default ``()``.

        Returns:
            ``self`` for method chaining.
        """
        if not isinstance(self._strategy, StepperStep):
            raise TypeError(
                "add_dependency requires a model with a StepperStep temporal strategy. "
                "Use a ModelStep/ModelXStep/ModelFBStep, or Model(net, temporal=StepperStep())."
            )
        if not isinstance(name, str) or not name:
            raise ValueError("add_dependency: name must be a non-empty string.")
        if not (0 <= component < self.n_outputs):
            raise ValueError(
                f"add_dependency: component {component} out of range "
                f"(n_outputs={self.n_outputs})."
            )
        order = tuple(order)
        for d in order:
            if not (0 <= d < self.n_dims):
                raise ValueError(
                    f"add_dependency: dimension index {d} in order out of "
                    f"range (n_dims={self.n_dims})."
                )
        if any(dep.name == name for dep in self.dependencies):
            raise ValueError(
                f"add_dependency: a dependency named {name!r} is already registered."
            )
        self.dependencies.append(_Dependency(name=name, component=component, order=order))
        return self

    # ------------------------------------------------------------------ #
    #  LaTeX  / display                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _latex_name(name: str) -> str:
        name = str(name)
        name = name.replace("\\", r"\backslash ")
        name = name.replace("_", r"\_")
        name = name.replace(" ", r"\ ")
        return name

    def get_problem_latex(self, include_legend: bool = True) -> str:
        """Return a LaTeX string for the loss functional."""
        terms = []
        legend = []
        has_lagrange = any(t.lagrange for t in self._terms)

        kind_label = {
            'inner':     'PDE',
            'boundary':  'BC',
            'dirichlet': 'Dirichlet BC',
            'neumann':   'Neumann BC',
            'robin':     'Robin BC',
            'periodic':  'Periodic BC',
            'initial':   'IC',
            'points':    'Data',
        }

        for t in self._terms:
            sym = self._latex_name(t.name)
            terms.append(
                rf"\frac{{w_{{{sym}}}}}{{N_{{{sym}}}}}"
                rf"\left\|\mathcal{{R}}_{{{sym}}}\right\|_2^2"
            )
            if t.lagrange:
                terms.append(
                    rf"\frac{{1}}{{N_{{{sym}}}}}"
                    rf"\langle\boldsymbol{{\lambda}}_{{{sym}}},\,"
                    rf"\mathcal{{R}}_{{{sym}}}\rangle"
                )
            if include_legend:
                kl = kind_label[t.kind]
                legend.append(
                    rf"\mathcal{{R}}_{{{sym}}}:\text{{ {kl} residual "
                    rf"— {t.name} (region: {t.region})}}"
                )
                legend.append(rf"N_{{{sym}}}:\text{{ \# samples for {t.name}}}")
                legend.append(rf"w_{{{sym}}}:\text{{ weight for {t.name}}}")
                if t.lagrange:
                    legend.append(
                        rf"\boldsymbol{{\lambda}}_{{{sym}}}:"
                        rf"\text{{ Lagrange multipliers for {t.name}}}"
                    )

        if not terms:
            return r"\mathcal{L}(\theta) = 0"

        if has_lagrange:
            lam = ",".join(
                rf"\boldsymbol{{\lambda}}_{{{self._latex_name(t.name)}}}"
                for t in self._terms if t.lagrange
            )
            operator = rf"\min_\theta\,\max_{{{lam}}}\;"
            lhs = r"\mathcal{L}(\theta,\boldsymbol{\lambda})"
        else:
            operator = r"\min_\theta\;"
            lhs = r"\mathcal{L}(\theta)"

        body = f"{operator}{lhs}=" + " + ".join(terms)
        if include_legend and legend:
            body += (
                r" \\[4pt]\begin{array}{l} "
                + r" \\ ".join(legend)
                + r" \end{array}"
            )
        return body

    def show_problem(self, include_legend: bool = True) -> str:
        """Render the loss functional as LaTeX (notebook) or print it."""
        latex = self.get_problem_latex(include_legend=include_legend)
        try:
            from IPython.display import Math, display
            display(Math(latex))
        except Exception:
            print(latex)
        return latex

    def __repr__(self) -> str:
        n_in  = len(self.inner_terms)
        n_bnd = len(self.boundary_terms)
        n_ic  = len(self.initial_terms)
        stepping = (
            f", stepping=True, dependencies={[d.name for d in self.dependencies]}"
        ) if self.dependencies else ""
        model_str = f", model={self._model!r}" if self._model is not None else ""
        return (
            f"ProblemStrong("
            f"domain={type(self.domain).__name__}, "
            f"n_dims={self.n_dims}, n_outputs={self.n_outputs}, "
            f"inner={n_in}, boundary={n_bnd}, initial={n_ic}, "
            f"fixed={list(self.fixed_params.keys())}, "
            f"inferred={list(self.inferred_params.keys())}"
            f"{model_str}{stepping})"
        )


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: Deprecated — use :class:`ProblemStrong` directly.