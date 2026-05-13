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

    def sample(self, train_samples=None, rng=None):
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
            train_samples = {}
        domain = self.domain
        params = self._build_params()
        data = {}
        for term in self._terms:
            n = train_samples.get(term.name,
                                  1000 if term.kind in ('inner', 'initial') else 10)
            if n <= 0:
                continue
            region = None if term.region == 'all' else term.region
            if term.kind == 'inner':
                pts = domain.sample_interior(n, region=region, rng=rng, params=params)
            elif term.kind == 'initial':
                pts = domain.sample_interior(n, rng=rng, params=params)
                t_dim = getattr(domain, '_spatial_dims', 0)
                if t_dim < pts.shape[1]:
                    pts[:, t_dim] = domain.xmin[t_dim]
            else:
                pts = domain.sample_boundary(n, region=region, rng=rng, params=params)
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

    def make_residual_fn(self, network):
        """Return ``fn(params, data) -> dict[str, jnp.ndarray]``.

        Each key is a term name; each value is the per-point (or per-sample)
        residual vector for that term.  Calling :meth:`assemble` on the result
        is a no-op for strong-form problems (identity).

        Parameters
        ----------
        network :
            The JAX network.  Must support ``network.apply(params, x, params_dict)``.

        Returns
        -------
        residual_fn : callable
            ``residual_fn(params, data) -> dict[str, jnp.ndarray]``
        """
        import inspect as _inspect
        import jax.numpy as jnp
        from pinns.functional import make_derivative_fn

        _terms = list(self._terms)
        _params_dict = self._build_params()
        _problem = self

        def _model_apply(params, x):
            return network.apply(params, x, _params_dict)

        def _call_fn(fn, x, u, deriv_fn):
            """Dispatch fn(x,u[,params_dict[,deriv_fn]]) → tuple of residuals."""
            if not callable(fn):
                # Scalar constant target — caller uses output_idx to pick component
                return (fn,)
            n_p = len(_inspect.signature(fn).parameters)
            if n_p >= 4:
                raw = fn(x, u, _params_dict, deriv_fn)
            elif n_p == 3:
                raw = fn(x, u, _params_dict)
            else:
                raw = fn(x, u)
            return raw if isinstance(raw, (list, tuple)) else (raw,)

        def residual_fn(params, data):
            result = {}
            deriv_fn = make_derivative_fn(_model_apply, params)
            for term in _terms:
                if term.name not in data:
                    continue
                x = data[term.name]
                u = _model_apply(params, x)
                kind = term.kind

                if kind in ('inner', 'initial', 'boundary'):
                    fn = term.fn
                    raw = _call_fn(fn, x, u, deriv_fn)
                    if len(raw) == 1 and not callable(fn):
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
                    target = jnp.asarray(term.get_value(x, _params_dict))
                    if hasattr(target, 'ndim') and target.ndim == 2:
                        target = target[:, 0:1]
                    residual = u[:, col:col + 1] - target

                elif kind in ('neumann', 'robin'):
                    if term.fn is None:
                        raise NotImplementedError(
                            f"{kind.capitalize()} BC '{term.name}' has no pre-built fn. "
                            "Use add_neumann / add_robin to register it."
                        )
                    raw = _call_fn(term.fn, x, u, deriv_fn)
                    residual = raw[0] if len(raw) == 1 else jnp.stack(raw, axis=-1)

                elif kind == 'periodic':
                    _pts_p = jnp.asarray(x, dtype=jnp.float32)
                    _half  = _pts_p.shape[0] // 2
                    x_a, x_b = _pts_p[:_half], _pts_p[_half:]
                    u_a = _model_apply(params, x_a)
                    u_b = _model_apply(params, x_b)
                    _n_out_p = u_a.shape[1]
                    _comps_p = ([term.component] if term.component is not None
                                else list(range(_n_out_p)))
                    residual = jnp.concatenate(
                        [u_a[:, c:c+1] - u_b[:, c:c+1] for c in _comps_p], axis=1)
                    if getattr(term, 'match_x_derivative', False):
                        _dim_p = 1
                        _ta = jnp.zeros_like(u_a[:, :1]).at[:, 0].set(0.0)  # placeholder
                        _ta = jnp.zeros((u_a.shape[0], x_a.shape[-1] if hasattr(x_a, '__len__') else jnp.asarray(x_a).shape[-1])).at[:, _dim_p].set(1.0)
                        _tb = jnp.zeros_like(_ta).at[:, _dim_p].set(1.0)
                        _c0 = _comps_p[0]
                        _xa_j = jnp.asarray(x_a, dtype=jnp.float32)
                        _xb_j = jnp.asarray(x_b, dtype=jnp.float32)
                        def _fa(xin): return _model_apply(params, xin)[:, _c0]
                        def _fb(xin): return _model_apply(params, xin)[:, _c0]
                        _, _ua_d = jax.jvp(_fa, (_xa_j,), (_ta,))
                        _, _ub_d = jax.jvp(_fb, (_xb_j,), (_tb,))
                        residual = jnp.concatenate(
                            [residual, (_ua_d - _ub_d).reshape(-1, 1)], axis=1)
                    result[term.name] = residual
                    continue

                else:
                    continue

                result[term.name] = _problem.assemble(term, residual)
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
            f"fixed={list(self.fixed_params.keys())}, "
            f"inferred={list(self.inferred_params.keys())}"
            f"{deps})"
        )