"""
pinns/problems/base_problem.py — Shared base class for ProblemStrong and ProblemWeak.

Both problem forms share:
* A flat ``_terms`` list as the single source of truth for all registered terms.
* ``boundary_conditions`` and ``_inner_terms`` as filter *properties* over ``_terms``.
* Parameter management (``fixed_params``, ``inferred_params``, ``_build_params``).
* Observable / solution attachment.
* Unified ``add_periodic`` via ``domain._periodic_regions``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

class _Dependency:
    """Declares a quantity to evaluate from the previous step's network
    and store in ``pars['fixed']`` before the next step."""

    def __init__(self, name: str, component: int, order: tuple):
        self.name = name
        self.component = component
        self.order = tuple(order)

    def __repr__(self):
        return (
            f"_Dependency(name={self.name!r}, component={self.component}, "
            f"order={self.order})"
        )


class BaseProblem:
    """Abstract base class shared by :class:`~pinns.problems.ProblemStrong`
    and :class:`~pinns.problems.ProblemWeak`.

    Subclasses must implement ``make_residual_fn``, ``sample``, and
    ``__repr__``.  They may also override ``add_boundary``, ``add_dirichlet``,
    ``add_neumann``, and ``add_initial`` to add form-specific logic.
    """

    _SPATIAL_LABELS = ('x', 'y', 'z', 'x4', 'x5', 'x6', 'x7', 'x8')
    _BC_KINDS = ('boundary', 'dirichlet', 'neumann', 'robin', 'periodic')

    def __init__(self, domain, output_names):
        self.domain = domain
        self.n_dims = domain.n_dims

        # ── parameter storage ────────────────────────────────────────────
        self.fixed_params: Dict[str, Any] = {}
        self.inferred_params: Dict[str, Any] = {}

        # ── solution / observable bookkeeping ────────────────────────────
        self.solution = None
        self.obs_fn:      Optional[Callable] = None
        self.obs_names:   Optional[List[str]] = None
        self.obs_spatial: List[str] = []

        # ── Auto-derive input_names from domain ───────────────────────────
        has_time = getattr(domain, '_t_min', None) is not None
        n_spatial = getattr(domain, '_spatial_dims', self.n_dims - int(has_time))
        self.input_names: List[str] = (
            list(self._SPATIAL_LABELS[:n_spatial]) + (['t'] if has_time else [])
        )

        # ── output names ─────────────────────────────────────────────────
        if not output_names:
            raise ValueError("output_names must not be empty.")
        self.output_names = list(output_names)
        self.n_outputs = len(self.output_names)

        # ── unified term registry ─────────────────────────────────────────
        self._terms: list = []

        # ── step dependencies (discrete schemes) ─────────────────────────
        self.dependencies: List[_Dependency] = []

    # ──────────────────────────────────────────────────────────────────── #
    #  Backward-compat alias                                              #
    # ──────────────────────────────────────────────────────────────────── #

    @property
    def params(self) -> Dict[str, Any]:
        """Alias for :attr:`fixed_params` (backward compat with ProblemWeak API)."""
        return self.fixed_params

    @property
    def xmin(self):
        """Lower bounds of the domain — proxy for ``domain.xmin``."""
        return self.domain.xmin

    @property
    def xmax(self):
        """Upper bounds of the domain — proxy for ``domain.xmax``."""
        return self.domain.xmax

    @params.setter
    def params(self, value: Dict[str, Any]) -> None:
        self.fixed_params = value

    # ──────────────────────────────────────────────────────────────────── #
    #  Term filter properties                                             #
    # ──────────────────────────────────────────────────────────────────── #

    @property
    def inner_terms(self) -> list:
        """All registered interior PDE terms."""
        return [t for t in self._terms if t.kind == 'inner']

    @property
    def _inner_terms(self) -> list:
        """Alias for :attr:`inner_terms` (backward compat)."""
        return self.inner_terms

    @property
    def boundary_terms(self) -> list:
        """All registered boundary-condition terms (excluding initial)."""
        return [t for t in self._terms if t.kind in self._BC_KINDS]

    @property
    def initial_terms(self) -> list:
        """All registered initial-condition terms."""
        return [t for t in self._terms if t.kind == 'initial']

    @property
    def boundary_conditions(self) -> list:
        """All non-inner terms (BCs + initial): used by the trainer."""
        return [t for t in self._terms if t.kind != 'inner']

    # ──────────────────────────────────────────────────────────────────── #
    #  Output resolution                                                  #
    # ──────────────────────────────────────────────────────────────────── #

    def _resolve_outputs(self, outputs):
        """Return ``list[(output_idx, suffix)]`` for the given *outputs* spec.

        * ``None``           → ``[(0, '')]``    — only when ``n_outputs == 1``
        * ``'u'``            → ``[(0, '')]``    — single named output
        * ``['u', 'v']``     → ``[(0, '_u'), (1, '_v')]``
        * ``0`` / ``[0, 1]`` — integer indices also accepted
        """
        if outputs is None:
            if self.n_outputs > 1:
                raise ValueError(
                    f"This problem has {self.n_outputs} outputs "
                    f"({self.output_names}). You must specify 'outputs' "
                    f"explicitly (e.g. outputs='u' or outputs={self.output_names})."
                )
            return [(0, '')]
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
        if len(result) == 1:
            result = [(result[0][0], '')]
        return result

    # ──────────────────────────────────────────────────────────────────── #
    #  Interior term registration                                         #
    # ──────────────────────────────────────────────────────────────────── #

    def add_inner(self, fn, name=None, region: str = 'all') -> 'BaseProblem':
        """Register an **interior** physics residual.

        Args:
            fn: Residual callable ``fn(x, u, pars, derivative) -> residual``.
            name: Label(s) for this term.  A ``str`` registers a single term;
                a ``list[str]`` registers one term per equation column.
                Defaults to ``'pde'``.
            region: Named interior region, or ``'all'`` (default).

        Returns:
            ``self`` for method chaining.
        """
        if name is None:
            name = 'pde'
        from .terms import TermInner
        if isinstance(name, str):
            eq_pairs = [(name, None)]
        else:
            names = list(name)
            if len(names) < 2:
                raise ValueError(
                    "When name is a list it must contain at least 2 entries."
                )
            eq_pairs = [(n, i) for i, n in enumerate(names)]
        for term_name, eq_idx in eq_pairs:
            # Idempotent: if a term with this name already exists, replace its fn.
            _region = None if region == 'all' else region
            for existing in self._terms:
                if existing.kind == 'inner' and existing.name == term_name:
                    existing.fn = fn
                    break
            else:
                self._terms.append(
                    TermInner(fn=fn, name=term_name, region=_region, eq_idx=eq_idx)
                )
        return self

    # ──────────────────────────────────────────────────────────────────── #
    #  Boundary, initial, periodic registration                          #
    # ──────────────────────────────────────────────────────────────────── #

    def add_boundary(
        self,
        fn,
        name,
        region: str = 'all',
    ) -> 'BaseProblem':
        """Register a generic **boundary** residual (custom BC).

        Use this when the boundary condition does not fit neatly into
        Dirichlet / Neumann / Robin categories.  For standard BCs prefer
        :meth:`add_dirichlet`, :meth:`add_neumann`, or :meth:`add_robin`.

        * ``name`` can be a ``str`` (single equation) or a ``list[str]``
          (one name per equation column returned by ``fn``).
        * For :class:`~pinns.domain.DomainMesh` the *region* must be
          pre-registered via ``domain.add_boundary(select, name=region)``
          before this call.

        Args:
            fn: Residual callable.  Strong form: ``fn(x, u, pars, derivative)``.
                Weak form: ``fn(x, y, pars, phi, derivative)``.
            name: ``str`` or ``list[str]`` — label(s) for this term.
            region: Named boundary region, or ``'all'`` (default).

        Returns:
            ``self`` for method chaining.
        """
        from .terms import TermCustomBC
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
                TermCustomBC(region=region, fn=fn, name=term_name, eq_idx=eq_idx)
            )
        return self

    def add_dirichlet(
        self,
        value,
        name: str,
        region: str = 'all',
        outputs=None,
    ) -> 'BaseProblem':
        """Register a **Dirichlet** boundary condition ``u = g``.

        Args:
            value: Prescribed boundary value.  Scalar, array, or callable
                ``g(x, pars) -> array``.
            name: Unique label for this term.
            region: Named boundary region, or ``'all'`` (default).
            outputs: Which output(s) this term applies to.  ``None`` is only
                allowed when ``n_outputs == 1``.

        Returns:
            ``self`` for method chaining.
        """
        from .terms import TermDirichletBC
        for out_idx, suffix in self._resolve_outputs(outputs):
            self._terms.append(
                TermDirichletBC(region=region, value=value,
                                component=out_idx or 0, name=name + suffix)
            )
        return self

    def add_initial(
        self,
        fn_or_value=0.0,
        name: str = 'ic',
        outputs=None,
    ) -> 'BaseProblem':
        """Register an **initial condition** residual.

        Collocation points are sampled at ``t = t_min`` from the domain
        interior.  Requires the domain to have a time axis.

        For multi-output problems *outputs* must be specified to select which
        components to enforce and determine the per-component term names.
        Passing ``outputs=['u', 'v']`` registers terms ``'ic_u'`` and
        ``'ic_v'``.  In the compile dict the **base** name ``'ic'`` may be
        used as a shorthand and will automatically expand to every term whose
        name starts with ``'ic_'``, applying the same train/test/weight config
        to all of them.

        Args:
            fn_or_value: Callable ``fn(x, u, pars, *args) -> residual`` **or**
                a scalar/array constant ``u0`` (creates residual
                ``u[:, out] - u0`` automatically).
            name: Base label.  Default ``'ic'``.
            outputs: Which output(s) this term applies to.  ``None`` is only
                allowed when ``n_outputs == 1``.

        Returns:
            ``self``.

        Raises:
            ValueError: If the domain has no time axis.
        """
        from .terms import TermInitial
        domain = self.domain
        has_time = (
            domain._t_min is not None
            if hasattr(domain, '_t_min')
            else getattr(domain, 'has_time', False)
        )
        if not has_time:
            raise ValueError(
                "add_initial requires a time-dependent domain. "
                "Pass time=(t_min, t_max) when constructing the domain."
            )
        for out_idx, suffix in self._resolve_outputs(outputs):
            if callable(fn_or_value):
                fn = fn_or_value
            else:
                def _make_ic(v, i):
                    def _ic(x, u, pars, *args):
                        return u[:, i] - v
                    return _ic
                fn = _make_ic(fn_or_value, out_idx)
            # For DomainMesh: register IC node region so _init_body can use it
            region = 'initial'
            if hasattr(domain, 'interior_node_mask'):
                import numpy as _np
                _ic_name = f'__ic_c{out_idx}__'
                if _ic_name not in domain._boundary_regions:
                    ni = _np.where(domain.interior_node_mask)[0].astype(_np.intp)
                    domain._boundary_regions[_ic_name] = {
                        'node_indices': ni,
                        'edges':        None,
                        'edge_lengths': None,
                        'edge_probs':   None,
                        'normals':      None,
                    }
                region = _ic_name
            self._terms.append(
                TermInitial(fn=fn, name=name + suffix,
                            output_idx=out_idx, region=region)
            )
        return self

    # ──────────────────────────────────────────────────────────────────── #
    #  Periodic BC registration (unified new-style)                      #
    # ──────────────────────────────────────────────────────────────────── #

    def add_periodic(
        self,
        region: str,
        name: str,
        component=None,
        match_x_derivative: int = 1,
    ) -> 'BaseProblem':
        """Register a **periodic** BC using a pre-registered domain pairing.

        The domain must have a ``_periodic_regions`` dict with an entry for
        *region*, created by :meth:`~pinns.domain.DomainMesh.add_periodic` or
        :meth:`~pinns.domain.DomainCubic.add_periodic`.

        Args:
            region: Key in ``domain._periodic_regions`` identifying the pairing.
            name: Unique label for this term.
            component: Network output component to enforce, or ``None`` for all.
            match_x_derivative: Number of spatial-derivative orders to match.
                ``0`` — field values only; ``1`` — also ``du/dx``;
                ``2`` — also ``d²u/dx²``; etc.  Default ``1``.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If *region* is not in ``domain._periodic_regions``.
        """
        from .terms import TermPeriodicBC
        periodic_regions = getattr(self.domain, '_periodic_regions', {})
        if region not in periodic_regions:
            raise ValueError(
                f"Periodic region {region!r} is not registered on the domain. "
                f"Call domain.add_periodic(..., name={region!r}) first. "
                f"Registered: {list(periodic_regions.keys())}"
            )
        self._terms.append(
            TermPeriodicBC(
                region=region,
                component=component,
                name=name,
                match_x_derivative=int(match_x_derivative),
            )
        )
        return self

    # ──────────────────────────────────────────────────────────────────── #
    #  Step dependency registration                                       #
    # ──────────────────────────────────────────────────────────────────── #

    def add_dependency(
        self,
        name: str,
        component: int = 0,
        order: tuple = (),
    ) -> 'BaseProblem':
        """Declare a **step dependency** for discrete time-integration schemes.

        Registers a quantity to be evaluated from the previous step's trained
        network and stored under ``params["dependencies"][name]`` before
        optimising the next step.

        Args:
            name: Key under which the evaluated quantity will be stored.
            component: Network output index.  Default ``0``.
            order: Derivative order tuple (input-dimension indices).
                ``()`` means plain field evaluation.  Default ``()``.

        Returns:
            ``self`` for method chaining.
        """
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

    # ──────────────────────────────────────────────────────────────────── #
    #  Solution / observable                                              #
    # ──────────────────────────────────────────────────────────────────── #

    def add_solution(self, fn) -> 'BaseProblem':
        """Attach a reference / analytical solution for error tracking.

        Args:
            fn: Callable ``fn(xy, params=None) -> array``, shape ``(n, n_dims)``.

        Returns:
            ``self`` for method chaining.
        """
        self.solution = fn
        return self

    def add_observable(
        self,
        fn: Callable,
        names: List[str],
        *,
        spatial: bool = False,
    ) -> 'BaseProblem':
        """Register a derived observable quantity for plotting/logging.

        Args:
            fn: Callable ``fn(x, y, params[, derivative]) -> array``.
            names: Output names for each component.
            spatial: If ``True``, treated as a displacement field for mesh
                visualisation.

        Returns:
            ``self`` for method chaining.
        """
        self.obs_fn    = fn
        self.obs_names = list(names)
        if spatial:
            self.obs_spatial = list(names)
        return self

    # ──────────────────────────────────────────────────────────────────── #
    #  Parameter management                                               #
    # ──────────────────────────────────────────────────────────────────── #

    def add_fixed(self, name=None, value=None, **kwargs) -> 'BaseProblem':
        """Register fixed (constant) parameters.

        Supports three call styles::

            problem.add_fixed('alpha', 1e-3)                   # single
            problem.add_fixed(['alpha', 'beta'], [1e-3, 0.5])  # list
            problem.add_fixed(alpha=1e-3, beta=0.5)            # kwargs

        Returns:
            ``self`` for method chaining.
        """
        if kwargs:
            self.fixed_params.update(kwargs)
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

    def add_inferred(self, name, init=0.0) -> 'BaseProblem':
        """Register learnable (inferred) parameters.

        Args:
            name: Parameter name (``str``) or list of names.
            init: Initial value(s).  A scalar is broadcast over all names.

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

    def update_params(self, **kwargs) -> 'BaseProblem':
        """Update existing fixed or inferred parameters.

        Returns:
            ``self`` for method chaining.

        Raises:
            KeyError: If a key is not found in either ``fixed_params`` or
                ``inferred_params``.
        """
        for k, v in kwargs.items():
            if k in self.fixed_params:
                self.fixed_params[k] = v
            elif k in self.inferred_params:
                self.inferred_params[k] = v
            else:
                raise KeyError(f"update_params: unknown parameter '{k}'.")
        return self

    def _build_params(self, internal=None) -> Dict[str, Any]:
        """Return the params dict passed to ``term.fn(x, u, params, deriv)``.

        Returns:
            ``{"fixed": fixed_params, "infer": {}, "internal": {...},
               "domain": domain}``
        """
        if internal is None:
            internal = {'global_step': 0, 'step': 0}
        return {
            "fixed":    self.fixed_params,
            "infer":    {},
            "internal": internal,
            "domain":   self.domain,
        }

    # ──────────────────────────────────────────────────────────────────── #
    #  Sampling interface (subclass must override)                        #
    # ──────────────────────────────────────────────────────────────────── #

    def sample(self, train_samples=None, rng=None):
        """Return a ``sample_data`` dict suitable for passing to the
        ``residual_fn`` returned by :meth:`make_residual_fn`.

        For :class:`~pinns.problems.ProblemStrong`: returns
        ``{term_name: x_collocation}`` — one (N, n_dims) array per term.

        For :class:`~pinns.problems.ProblemWeak`: returns ``None`` (fixed
        cubature, no per-step sampling required).  Optionally returns
        ``{'free_nodes': subsampled_indices}`` when *n_free* and *key* are
        supplied.

        Args:
            train_samples: ``dict[str, int]`` mapping term names to the number
                of collocation points to draw per step (ProblemStrong only).
            rng: Random-number generator (``numpy.random.Generator``) used
                to draw collocation points.

        Returns:
            ``dict`` of collocation data, or ``None`` for ProblemWeak.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement sample()"
        )
