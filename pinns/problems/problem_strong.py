"""
ProblemStrong — Physics-Informed Neural ModelBase problem definition.

Supports both :class:`~pinns.domain.DomainCubic` and
:class:`~pinns.domain.DomainMesh` domains.  Backend: JAX only.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Any, List, Optional, Union

from ..domain import DomainCubic, DomainMesh
# ---------------------------------------------------------------------------
# Term classes (unified with weak-form terms)
# ---------------------------------------------------------------------------

from .terms import (TermInner, TermCustomBC, TermDirichletBC, TermNeumannBC,
                     TermRobinBC, TermPeriodicBC, TermInitial)
from .base_problem import BaseProblem

# ---------------------------------------------------------------------------
# ProblemStrong
# ---------------------------------------------------------------------------

class ProblemStrong(BaseProblem):
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
    * **pars** — flat dict of all registered parameters plus ``"internal"`` and ``"domain"``.
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
    with :meth:`add_parameter`.

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
        problem.add_parameter('alpha', 0.01)

        def heat(x, u, pars, diff):
            a = pars['parameter']['alpha']
            return diff(u, x, 0, (2,)) - a * (diff(u, x, 0, (0, 0)) + diff(u, x, 0, (1, 1)))

        problem.add_inner(heat, name='pde')

        def dirichlet(x, u, pars, diff):
            return u[:, 0]

        problem.add_dirichlet(dirichlet, name='left_bc', region='left_arc')

        def ic(x, u, pars, diff):
            return u[:, 0] - jnp.sin(jnp.pi * x[:, 0])

        problem.add_initial(ic, name='ic')
    """

    def __init__(
        self,
        domain,
        output_names,
    ):
        if not isinstance(domain, (DomainCubic, DomainMesh)):
            raise TypeError(
                "domain must be a DomainCubic or DomainMesh instance, "
                f"got {type(domain).__name__!r}."
            )

        super().__init__(domain, output_names)

    def add_neumann(
        self,
        value,
        name: str,
        region: str = 'all',
        outputs=None,
    ) -> 'ProblemStrong':
        """Register a **Neumann** boundary condition  ``n·∇u = g``.

        The outward unit normal at each collocation point is obtained from
        the domain via :meth:`~pinns.domain.DomainCubic.get_boundary_normals`
        and the condition is synthesised as a custom residual::

            n·∇u(x) - g(x) = 0

        Args:
            value: Prescribed normal-flux ``g``.  Scalar, or callable
                ``g(x, pars) -> array``.
            name: Unique label.
            region: Named boundary region, or ``'all'``.
            outputs: Which output(s) to apply the BC to.

        Returns:
            ``self``.
        """
        _domain = self.domain
        for out_idx, suffix in self._resolve_outputs(outputs):
            _comp = out_idx or 0
            _value = value

            def _neumann_fn(x, u, pars, derivative,
                            _d=_domain, _c=_comp, _v=_value, _r=region):
                import jax.numpy as jnp
                import numpy as _np
                normals = _np.asarray(_d.get_boundary_normals(
                    _np.asarray(x), _r))           # (n, n_spatial)
                n_spatial = normals.shape[1]
                # n·∇u = sum_i  n_i * du/dx_i
                du_dn = sum(
                    normals[:, i] * derivative(u, x, _c, (i,))
                    for i in range(n_spatial)
                )
                if callable(_v):
                    import inspect as _inspect
                    g = _v(x, pars) if len(_inspect.signature(_v).parameters) >= 2 else _v(x)
                else:
                    g = float(_v)
                return du_dn - g

            self._terms.append(
                TermNeumannBC(region=region, value=value,
                              component=_comp, name=name + suffix,
                              fn=_neumann_fn)
            )
        return self

    def add_robin(
        self,
        alpha,
        beta,
        g,
        name: str,
        region: str = 'all',
        outputs=None,
    ) -> 'ProblemStrong':
        """Register a **Robin** boundary condition  ``α·u + β·n·∇u = g``.

        The outward unit normal is obtained from the domain via
        :meth:`~pinns.domain.DomainCubic.get_boundary_normals` and the
        condition is synthesised as a custom residual::

            α·u(x) + β·(n·∇u)(x) - g(x) = 0

        Args:
            alpha: Coefficient on ``u``.  Scalar or callable
                ``alpha(x, pars) -> array``.
            beta: Coefficient on ``n·∇u``.  Same accepted forms.
            g: Right-hand side.  Scalar or callable ``g(x, pars) -> array``.
            name: Unique label.
            region: Named boundary region, or ``'all'``.
            outputs: Which output(s) to apply the BC to.

        Returns:
            ``self``.
        """
        _domain = self.domain
        for out_idx, suffix in self._resolve_outputs(outputs):
            _comp  = out_idx or 0
            _alpha = alpha
            _beta  = beta
            _g     = g

            def _robin_fn(x, u, pars, derivative,
                          _d=_domain, _c=_comp,
                          _a=_alpha, _b=_beta, _gv=_g, _r=region):
                import jax.numpy as jnp
                import numpy as _np
                import inspect as _inspect
                normals = _np.asarray(_d.get_boundary_normals(
                    _np.asarray(x), _r))
                n_spatial = normals.shape[1]
                du_dn = sum(
                    normals[:, i] * derivative(u, x, _c, (i,))
                    for i in range(n_spatial)
                )
                def _eval(v):
                    if callable(v):
                        return v(x, pars) if len(_inspect.signature(v).parameters) >= 2 else v(x)
                    return float(v)
                return _eval(_a) * u[:, _c] + _eval(_b) * du_dn - _eval(_gv)

            self._terms.append(
                TermRobinBC(region=region, alpha=alpha, beta=beta, value=g,
                            component=_comp, name=name + suffix,
                            fn=_robin_fn)
            )
        return self


    # ------------------------------------------------------------------
    # Sampling interface
    # ------------------------------------------------------------------

    def sample(self, train_samples=None, t_interval=None, rng=None):
        """Sample collocation points for every registered term.

        Args:
            train_samples: ``dict[str, int]`` mapping term name → number of
                points.  Terms not present in the dict keep the defaults
                (1000 for inner/initial, 10 for boundary terms).
            rng: ``numpy.random.Generator`` instance.  When ``None`` a fresh
                generator is created.

        Returns:
            ``dict[str, np.ndarray]`` — ``{term_name: x_pts(N, n_dims)}``.
        """
        import numpy as _np
        if rng is None:
            rng = _np.random.default_rng()
        if train_samples is None:
            raise ValueError("train_samples dict is required for ProblemStrong sampling.")
        domain = self.domain
        params = self._build_params()
        data = {}
        for term in self._terms:
            n = train_samples.get(term.name)
            if n is None:
                raise ValueError(f"Number of samples for term '{term.name}' is not specified in train_samples.")
            if n <= 0:
                continue
            region = None if term.region == 'all' else term.region
            if term.kind == 'inner':
                pts = domain.sample_interior(n, region=region, t_interval=t_interval, rng=rng, params=params)
            elif term.kind == 'initial':
                pts = domain.sample_initial(n, region=region, t_interval=t_interval, rng=rng, params=params)
            else:
                pts = domain.sample_boundary(n, region=region, t_interval=t_interval, rng=rng, params=params)
            data[term.name] = pts
        return data

    # ------------------------------------------------------------------
    # Unified assemble interface (strong form = identity)
    # ------------------------------------------------------------------

    def assemble(self, term, r, cubature_data=None):
        """Return the per-point residual array unchanged.

        For strong-form problems the "assembly" step is just the identity:
        the loss is ``mean(r**2)`` evaluated directly at collocation points.

        Args:
            term: The term whose residual was computed (not used here).
            r: Raw per-point residual, shape ``(N,)`` or ``(N, n_eqs)``.
            cubature_data: Ignored for strong form (present for API symmetry
                with :class:`~pinns.problems.ProblemWeak`).

        Returns:
            ``r`` unchanged.
        """
        return r

    def make_residual_fn(self, network, fit_problem_parameters=None):
        """Return ``fn(params, data) -> dict[str, jnp.ndarray]``.

        Each key is a term name; each value is the per-point (or per-sample)
        residual vector for that term.  Calling :meth:`assemble` on the result
        is a no-op for strong-form problems (identity).

        Parameters
        ----------
        network :
            The JAX network.  Must support ``network.apply(x, params, params_dict)``.
        fit_problem_parameters : list[str], optional
            Names of problem parameters to treat as trainable (pulled from
            ``params["__problem_params__"]`` during JAX tracing so that
            gradients flow to the optimizer).

        Returns
        -------
        residual_fn : callable
            ``residual_fn(params, data) -> dict[str, jnp.ndarray]``
        """
        import inspect as _inspect
        import jax
        import jax.numpy as jnp
        from pinns.functional import make_derivative_fn

        _terms = list(self._terms)
        _fit_prob_params = fit_problem_parameters
        _static_pd = self._build_params()
        _problem = self

        def _model_apply(params, x, pd=None):
            return network.apply(x, params, pd if pd is not None else _static_pd)

        def _call_fn(fn, x, u, deriv_fn, _p):
            """Dispatch fn(x,u[,params_dict[,deriv_fn]]) → tuple of residuals.

            Single-argument callables ``fn(x)`` are treated as value functions
            returning target arrays; the caller is responsible for forming
            ``u - target``.  A None sentinel is returned in that case so the
            caller can detect it via ``raw is None``.
            """
            if not callable(fn):
                # Scalar constant target — caller uses output_idx to pick component
                return (fn,)
            n_p = len(_inspect.signature(fn).parameters)
            if n_p >= 4:
                raw = fn(x, u, _p, deriv_fn)
            elif n_p == 3:
                raw = fn(x, u, _p)
            elif n_p == 2:
                raw = fn(x, u)
            else:
                # fn(x) -> target values (value function, not residual)
                return None, fn(x)
            return raw if isinstance(raw, (list, tuple)) else (raw,)

        def _split_residual(name, x, residual):
            """Validate shape and split multi-column residuals into named sub-terms.

            Returns
            -------
            dict[str, jnp.ndarray]
                * Single-column ``(N,)`` or ``(N,1)``  → ``{name: (N,1)}``
                * Multi-column  ``(N, K)`` with K > 1  → ``{name_0: (N,1), ..., name_{K-1}: (N,1)}``

            Raises
            ------
            ValueError
                If the residual is square ``(N, N)`` — the classic broadcast bug
                from mixing ``(N,)`` with ``(N,1)`` derivative outputs.
            """
            if not hasattr(residual, 'shape'):
                return {name: residual}
            N = x.shape[0]
            r = residual
            if r.ndim == 1:
                r = r[:, None]
            # Catch (N, N) — the classic broadcast from mixing (N,) and (N,1)
            if r.ndim == 2 and r.shape[0] == N and r.shape[1] == N and N > 1:
                raise ValueError(
                    f"Term '{name}': residual has square shape {r.shape} matching "
                    f"N_samples={N}. This is almost certainly a broadcasting error. "
                    "Make sure arrays in your residual function use consistent shapes: "
                    "use U[:, 0:1] (shape (N,1)) instead of U[:, 0] (shape (N,)) "
                    "so that arithmetic with derivative outputs (N,1) does not broadcast "
                    "to (N,N)."
                )
            K = r.shape[1]
            if K == 1:
                return {name: r}
            # Multi-column: split into name_0, name_1, ..., name_{K-1}
            return {f"{name}_{i}": r[:, i:i+1] for i in range(K)}

        def residual_fn(params, data):
            # Dynamic problem parameter override (fit_problem_parameters)
            if _fit_prob_params:
                _prob_vals = params.get('__problem_params__', {})
                _base_pd = {
                    **_static_pd,
                    'parameter': {**_static_pd['parameter'], **_prob_vals},
                }
            else:
                _base_pd = _static_pd

            # Merge any dynamic fixed-param overrides passed through train_data.
            # Keys prefixed '__fixed__' carry per-sample arrays that must be
            # explicit JIT arguments so JAX sees updated values after every
            # SchedulerResample refresh.
            _dyn_fixed = {k[len('__fixed__'):]: data[k]
                          for k in data if k.startswith('__fixed__')
                          and not k.startswith('__fixed_domain__')}
            if _dyn_fixed:
                _p = {**_base_pd, 'parameter': {**_base_pd['parameter'], **_dyn_fixed}}
            else:
                _p = _base_pd

            _maf = lambda p, x: _model_apply(p, x, _p)
            result = {}
            deriv_fn = make_derivative_fn(_maf, params)
            for term in _terms:
                if term.name not in data:
                    continue
                x = data[term.name]
                u = _maf(params, x)
                kind = term.kind

                # Per-term domain data (e.g. surface normals for 3-D mesh domains).
                # Stored in train_data as '__fixed_domain__normals_{term_name}';
                # injected into params['domain']['normals'] for this term only.
                _normals_key = f'__fixed_domain__normals_{term.name}'
                if _normals_key in data:
                    _tp = {**_p, 'domain': {**_p['domain'], 'normals': data[_normals_key]}}
                else:
                    _tp = _p

                if kind in ('inner', 'initial', 'boundary'):
                    fn = term.fn
                    raw = _call_fn(fn, x, u, deriv_fn, _tp)
                    if isinstance(raw, tuple) and len(raw) == 2 and raw[0] is None:
                        # fn(x) -> target values
                        col = getattr(term, 'output_idx', 0) or 0
                        target = jnp.asarray(raw[1])
                        if target.ndim == 1:
                            target = target[:, None]
                        # If the target function returns multiple columns (e.g. a
                        # multi-output IC returning all species at once), slice to
                        # only the column this term is responsible for.
                        if target.ndim == 2 and target.shape[1] > 1:
                            target = target[:, col:col + 1]
                        residual = u[:, col:col + 1] - target
                    elif len(raw) == 1 and not callable(fn):
                        # Scalar constant target
                        col = getattr(term, 'output_idx', 0) or 0
                        residual = u[:, col:col + 1] - float(raw[0])
                    else:
                        residual = raw[0] if len(raw) == 1 else jnp.stack(raw, axis=-1)
                        eq_idx = getattr(term, 'eq_idx', None)
                        if eq_idx is not None and hasattr(residual, 'ndim') and residual.ndim == 2:
                            residual = residual[:, eq_idx]

                elif kind == 'dirichlet':
                    col = term.component
                    target = jnp.asarray(term.get_value(x, _tp))
                    if hasattr(target, 'ndim'):
                        if target.ndim == 1:
                            target = target[:, None]
                        elif target.ndim == 2:
                            target = target[:, 0:1]
                    residual = u[:, col:col + 1] - target

                elif kind in ('neumann', 'robin'):
                    if term.fn is None:
                        raise NotImplementedError(
                            f"{kind.capitalize()} BC '{term.name}' has no pre-built fn. "
                            "Use add_neumann / add_robin to register it."
                        )
                    raw = _call_fn(term.fn, x, u, deriv_fn, _tp)
                    residual = raw[0] if len(raw) == 1 else jnp.stack(raw, axis=-1)

                elif kind == 'periodic':
                    _pts_p = jnp.asarray(x, dtype=jnp.float32)
                    _half  = _pts_p.shape[0] // 2
                    x_a, x_b = _pts_p[:_half], _pts_p[_half:]
                    # One forward pass for both halves — halves model evals per term.
                    _u_ab = _maf(params, _pts_p)
                    u_a, u_b = _u_ab[:_half], _u_ab[_half:]
                    _comps_p = ([term.component] if term.component is not None
                                else list(range(u_a.shape[1])))
                    residual = jnp.concatenate(
                        [u_a[:, c:c+1] - u_b[:, c:c+1] for c in _comps_p], axis=1)
                    _n_derivs = int(getattr(term, 'match_x_derivative', 0))
                    if _n_derivs > 0:
                        _pinfo = getattr(_problem.domain, '_periodic_regions', {}).get(term.region, {})
                        _axis  = _pinfo.get('axis', 'x')
                        _ALBL  = getattr(_problem.domain, '_BOUNDARY_LABEL_MAP', {})
                        _albl_key = f'{_axis}min'
                        _dim_p = (_ALBL[_albl_key][0] if _albl_key in _ALBL
                                  else {'x': 0, 'y': 1, 'z': 2, 't': -1}.get(_axis, 0))
                        _tangent = jnp.zeros_like(x_a).at[:, _dim_p].set(1.0)

                        def _nth_deriv(c, order, xi, tangent=_tangent):
                            """Compute the *order*-th JVP of model[:,c] at xi."""
                            f = lambda xin: _maf(params, xin)[:, c]
                            for _ in range(order):
                                f = lambda xin, _f=f: jax.jvp(_f, (xin,), (tangent,))[1]
                            return f(xi)

                        for _c in _comps_p:
                            for _ord in range(1, _n_derivs + 1):
                                _diff = _nth_deriv(_c, _ord, x_a) - _nth_deriv(_c, _ord, x_b)
                                residual = jnp.concatenate(
                                    [residual, _diff.reshape(-1, 1)], axis=1)
                    result.update(_split_residual(term.name, x, residual))
                    continue

                else:
                    continue

                for _key, _r in _split_residual(term.name, x, residual).items():
                    result[_key] = _problem.assemble(term, _r)
            return result

        return residual_fn

    def __repr__(self) -> str:
        n_in  = len(self.inner_terms)
        n_bnd = len(self.boundary_terms)
        n_ic  = len(self.initial_terms)

        deps = (f", dependencies={[d.name for d in self.dependencies]}"
                ) if self.dependencies else ""
        return (
            f"ProblemStrong("
            f"domain={type(self.domain).__name__}, "
            f"n_dims={self.n_dims}, n_outputs={self.n_outputs}, "
            f"inner={n_in}, boundary={n_bnd}, initial={n_ic}, "
            f"params={list(self._params.keys())}, "
            f"trainable={list(self._trainable)}"
            f"{deps})"
        )