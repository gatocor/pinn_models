"""
pinns/terms.py — Unified residual-term definitions.

Every term (boundary condition, interior PDE, initial condition, data)
is a plain dataclass that:

* Stores **only the loss specification** — what to enforce (value / residual
  function) and **where** (a region name string).
* Carries **no resolved coordinates**, node arrays, edge lists, or problem-form
  specific evaluation logic — those live on the problem and trainer.

The trainer (or problem's ``make_residual_fn``) reads ``term.kind`` and the
relevant data fields (``value``, ``component``, ``fn``/``f``, ``components``,
``outputs``, ``eq_idx``) to compute residuals appropriate for the problem form.

Boundary region names
---------------------
``DomainCubic`` recognizes the built-in face labels
``'xmin'``, ``'xmax'``, ``'ymin'``, ``'ymax'``, ``'zmin'``, ``'zmax'``,
``'tmin'``, ``'tmax'`` (and common aliases ``'left'``/``'right'``,
``'bottom'``/``'top'``, ``'front'``/``'back'``, ``'inlet'``/``'outlet'``)
without any prior registration.

``DomainMesh`` requires explicit registration via
``domain.add_boundary(name, select, ...)`` before a term can reference it
by name.

Custom interior sub-regions (e.g. for ``TermInner``) are registered with
``domain.add_inner(name, ...)``.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Union, Callable, Optional, List

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _call_value_function(value_fn, x) -> np.ndarray:
    """Call a scalar value function on *x*, returning a float32 ``(n,)`` array."""
    if hasattr(x, 'detach'):          # torch tensor (legacy path)
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)
    result = value_fn(x_np)
    result = np.asarray(result, dtype=np.float32)
    if result.ndim > 1:
        result = result.squeeze(-1)
    return result


# ---------------------------------------------------------------------------
# Boundary / constraint terms
# ---------------------------------------------------------------------------

@dataclass
class TermDirichletBC:
    """Dirichlet condition: ``u_component(x) = value`` on a named boundary region.

    Args:
        region: Name of the boundary region on the domain (e.g. ``'xmin'``,
                ``'left_wall'``).  Built-in face labels are resolved
                automatically for :class:`~pinns.domain.DomainCubic`.
        value: Target value — scalar float, callable
               ``(x: np.ndarray) -> np.ndarray`` returning ``(n,)``, or
               callable ``(x, params_dict) -> array`` (strong-form usage).
        component: Output component index (default 0).
        name: Label used in loss / weight dicts.

    Example::

        TermDirichletBC(region='xmin', value=0.0, component=0, name='left')
        TermDirichletBC(region='tmin', value=lambda x: np.sin(np.pi*x[:,0]),
                        component=0, name='ic')
    """
    region:    str
    value:     Union[float, Callable]
    component: int           = 0
    name:      Optional[str] = None

    # class-level kind string — used by the trainer for routing
    kind:      'ClassVar[str]'  = 'dirichlet'
    has_value: 'ClassVar[bool]' = True

    def get_value(self, x, params_dict=None) -> np.ndarray:
        """Evaluate the prescribed value at *x*.

        Supports three callable forms:
        * ``(x) -> array``
        * ``(x, params_dict) -> array``
        * scalar constant
        """
        if callable(self.value):
            import inspect as _inspect
            n_p = len(_inspect.signature(self.value).parameters)
            if n_p >= 2:
                result = self.value(x, params_dict)
            else:
                result = _call_value_function(self.value, x)
            return np.asarray(result, dtype=np.float32)
        return np.full(x.shape[0], float(self.value), dtype=np.float32)

@dataclass
class TermNeumannBC:
    """Neumann condition: ``du_component/dn = value`` on a named boundary region.

    The outward unit normal **n** is injected by the trainer via
    ``ops.normals`` (shape ``(n_pts, n_spatial_dims)``), so
    ``compute_loss_dict`` is domain-agnostic.

    Args:
        region: Name of the boundary region.
        value: Normal-derivative target — scalar or callable.
        component: Output component index (default 0).
        name: Label used in loss / weight dicts.

    Example::

        TermNeumannBC(region='xmax', value=0.0, component=0, name='flux')
    """
    region:    str
    value:     Union[float, Callable]
    component: int                = 0
    name:      Optional[str]      = None
    fn:        Optional[Callable] = None   # pre-built residual closure (strong form)

    kind:      'ClassVar[str]'  = 'neumann'
    has_value: 'ClassVar[bool]' = True

    def get_value(self, x, params_dict=None) -> np.ndarray:
        if callable(self.value):
            import inspect as _inspect
            n_p = len(_inspect.signature(self.value).parameters)
            if n_p >= 2:
                result = self.value(x, params_dict)
            else:
                result = _call_value_function(self.value, x)
            return np.asarray(result, dtype=np.float32)
        return np.full(x.shape[0], float(self.value), dtype=np.float32)


@dataclass
class TermRobinBC:
    """Robin (mixed) condition: ``alpha*u + beta*du/dn = value`` on a boundary region.

    Args:
        region: Name of the boundary region.
        alpha: Coefficient of *u*.
        beta: Coefficient of *du/dn*.
        value: RHS value — scalar or callable.
        component: Output component index (default 0).
        name: Label used in loss / weight dicts.

    Example::

        # Convective BC: h*u + k*du/dn = h*u_inf
        TermRobinBC(region='xmax', alpha=10.0, beta=1.0, value=100.0)
    """
    region:    str
    alpha:     float
    beta:      float
    value:     Union[float, Callable]
    component: int                = 0
    name:      Optional[str]      = None
    fn:        Optional[Callable] = None   # pre-built residual closure (strong form)

    kind:      'ClassVar[str]'  = 'robin'
    has_value: 'ClassVar[bool]' = True

    def get_value(self, x, params_dict=None) -> np.ndarray:
        if callable(self.value):
            import inspect as _inspect
            n_p = len(_inspect.signature(self.value).parameters)
            if n_p >= 2:
                result = self.value(x, params_dict)
            else:
                result = _call_value_function(self.value, x)
            return np.asarray(result, dtype=np.float32)
        return np.full(x.shape[0], float(self.value), dtype=np.float32)


@dataclass
class TermCustomBC:
    """Custom residual condition on a named boundary or interior region.

    The residual callable *f* has the same signature as interior PDE
    residuals::

        f(x, y)                           -> residual
        f(x, y, params_dict)              -> residual
        f(x, y, params_dict, derivative)  -> residual  # or tuple

    When *f* returns a tuple each element becomes a separate sub-loss.

    Args:
        region: Name of the boundary (or interior) region to sample from.
        f: Residual callable.
        name: Base label used in weight dicts and plots.
        output_names: Optional per-output labels (when *f* returns a tuple).

    Example::

        def traction(x, y):
            # sigma_xx*nx + sigma_xy*ny = tx
            ...
        TermCustomBC(region='right', fn=traction, name='traction')
    """
    region:       str
    fn:           Callable
    name:         Optional[str]       = None
    output_names: Optional[List[str]] = None
    eq_idx:       Optional[int]       = None

    kind:      'ClassVar[str]'  = 'boundary'
    has_value: 'ClassVar[bool]' = False

    def _output_names(self, residuals):
        base  = self.name or 'bc'
        return (self.output_names
                or ([base] if len(residuals) == 1
                    else [f'{base}_{k}' for k in range(len(residuals))]))


@dataclass
class TermPeriodicBC:
    """Periodic condition: ``u(x_a) = u(x_b)`` for matched points on two regions.

    The trainer samples ``n_pairs`` matched point pairs from ``region_a`` and
    ``region_b`` (either by random node pairing on a mesh or by sampling the
    lower / upper face of the same spatial dimension on a cubic domain).

    For strong-form problems (via :meth:`~ProblemStrong.add_periodic`) the
    boundary residual is supplied by ``fn`` — in that case ``region_a`` holds
    the axis name (``'x'``, ``'y'``, …) and ``region_b`` may be left empty.

    Args:
        region_a: Name of the first boundary region (e.g. ``'xmin'``), or
                  the axis label for strong-form periodic BCs.
        region_b: Name of the matching boundary region (e.g. ``'xmax'``).  
                  May be empty (``''``) for strong-form usage.
        n_pairs: Number of collocation pairs.  ``None`` → use all available
                 node pairs (DomainMesh) or a default count (DomainCubic).
        component: Output component to enforce, or ``None`` for all components.
        name: Base label used in weight dicts.
        match_x_derivative: Number of spatial-derivative orders to match at the
                             periodic boundary.  ``0`` matches only field values.
                             ``1`` additionally matches ``du/dx``, ``2`` matches
                             ``du/dx`` *and* ``d²u/dx²``, etc.  ``True``/``False``
                             are accepted for backward compatibility (mapped to
                             1 and 0 respectively).  Default is ``1``.
        fn: Residual callable for strong-form usage.  ``None`` for weak form.

    Example::

        TermPeriodicBC(region_a='xmin', region_b='xmax',
                       component=0, name='periodic_x')
    """
    region:             str           = ''
    region_a:           str           = ''
    region_b:           str           = ''
    n_pairs:            Optional[int] = None
    component:          Optional[int] = None
    name:               Optional[str] = None
    match_x_derivative: int           = 1
    fn:                 Optional[Callable] = None

    kind:      'ClassVar[str]'  = 'periodic'
    has_value: 'ClassVar[bool]' = False
    eq_idx = None
    rhs    = None


# ---------------------------------------------------------------------------
# Interior / initial PDE residual terms
# ---------------------------------------------------------------------------

@dataclass
class TermInner:
    """Interior PDE residual term.

    Evaluated at interior collocation points sampled from *region* (or the
    full domain interior when ``region`` is ``None``).

    Args:
        fn: Residual callable — same signature as accepted by
            :meth:`~pinns.problems.ProblemStrong.add_inner`.
        name: Base label used in loss / weight dicts.
        region: Named sub-domain (registered via ``domain.add_inner(name, ...)``).
                ``None`` → full interior.
        output_names: Per-residual labels when *fn* returns a tuple.

        eq_idx: When set, select only this column from a multi-equation *fn*
                return. Used internally when ``add_inner(fn, name=[...])`` creates
                one term per equation.

    Example::

        TermInner(fn=heat_residual, name='heat')
        TermInner(fn=heat_residual, name='heat_left', region='left_half')
    """
    fn:           Callable
    name:         str              = 'pde'
    region:       Optional[str]    = None
    output_names: Optional[List[str]] = None
    eq_idx:       Optional[int]    = None

    kind:      'ClassVar[str]'  = 'inner'
    has_value: 'ClassVar[bool]' = False


@dataclass
class TermInitial:
    """Initial-condition residual term, sampled from the ``t = t_min`` slice.

    Example::

        TermInitial(fn=lambda x, y: y[:, 0] - u0(x), name='ic')
    """
    fn:           Callable
    name:         str              = 'ic'
    output_names: Optional[List[str]] = None
    region:       str              = 'initial'
    output_idx:   Optional[int]    = None

    kind:      'ClassVar[str]'  = 'initial'
    has_value: 'ClassVar[bool]' = False


# ---------------------------------------------------------------------------
# Term collection helper
# ---------------------------------------------------------------------------

class TermCollection:
    """Simple ordered container for heterogeneous term objects.

    Terms are added with :meth:`add` and iterated in insertion order.

    Example::

        terms = TermCollection()
        terms.add(TermDirichletBC(region='xmin', value=0.0))
        terms.add(TermNeumannBC(region='xmax', value=0.0))
        terms.add(TermInner(fn=pde_residual))
    """

    def __init__(self):
        self._terms: list = []

    def add(self, term) -> 'TermCollection':
        """Append *term* and return ``self`` for chaining."""
        self._terms.append(term)
        return self

    def __iter__(self):
        return iter(self._terms)

    def __len__(self):
        return len(self._terms)

    def __repr__(self):
        counts: dict = {}
        for t in self._terms:
            k = type(t).__name__
            counts[k] = counts.get(k, 0) + 1
        parts = ', '.join(f'{k}x{v}' for k, v in counts.items())
        return f'TermCollection([{parts}])'
