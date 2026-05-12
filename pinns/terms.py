"""
pinns/terms.py — Unified residual-term definitions.

Every term (boundary condition, interior PDE, initial condition, data)
is a plain dataclass that:

* Stores **only the loss specification** — what to enforce (value / residual
  function) and **where** (a region name string).
* Exposes ``compute_loss_dict(x, y, ops)`` → ``{name: scalar_loss}``.
* Carries **no resolved coordinates**, node arrays, or edge lists — those
  live on the domain.

The trainer calls ``domain.sample_boundary(n, region=term.region)`` to
obtain collocation points, then sets ``ops.normals`` before calling
``term.compute_loss_dict(x, y, ops)`` for Neumann / Robin terms.

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

def _call_residual_fn(fn, x, y, ops):
    """Call *fn* with the right number of arguments and return a tuple of residuals."""
    import inspect as _inspect
    n_p = len(_inspect.signature(fn).parameters)
    if n_p >= 4:
        residual = fn(x, y, ops.params_dict, None)
    elif n_p == 3:
        residual = fn(x, y, ops.params_dict)
    else:
        residual = fn(x, y)
    return residual if isinstance(residual, (list, tuple)) else (residual,)


# ---------------------------------------------------------------------------
# TermOps — backend operation bundle
# ---------------------------------------------------------------------------

class TermOps:
    """Bundle of backend tensor operations passed to every ``compute_loss_dict``.

    Created by ``Trainer._make_bc_ops(params_dict)`` so that term classes can
    compute losses without importing or depending on any backend.

    Attributes:
        to_tensor: ``(np.ndarray) -> tensor`` — convert numpy to backend type.
        mean_sq: ``(tensor) -> scalar`` — mean of squared residual values.
        directional_derivative: ``(x, component, dim) -> du_component/dx_dim``.
        params_dict: Structured parameters dict (fixed + inferred params).
        normals: Per-point outward unit normal vectors as a backend tensor,
                 shape ``(n, n_spatial_dims)``.  Set by the trainer before
                 calling ``compute_loss_dict`` on Neumann / Robin terms.
                 ``None`` for terms that do not need normals.
    """

    def __init__(self, to_tensor, mean_sq, directional_derivative,
                 params_dict=None, normals=None):
        self.to_tensor = to_tensor
        self.mean_sq = mean_sq
        self.directional_derivative = directional_derivative
        self.params_dict = params_dict
        self.normals = normals          # (n, n_spatial_dims) tensor or None


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
        value: Target value — scalar float or callable
               ``(x: np.ndarray) -> np.ndarray`` returning ``(n,)``.
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

    def get_value(self, x) -> np.ndarray:
        if callable(self.value):
            return _call_value_function(self.value, x)
        return np.full(x.shape[0], float(self.value), dtype=np.float32)

    def compute_loss_dict(self, x, y, ops) -> dict:
        target = ops.to_tensor(self.get_value(x))
        return {self.name or 'bc': ops.mean_sq(y[:, self.component] - target)}


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
    component: int           = 0
    name:      Optional[str] = None

    def get_value(self, x) -> np.ndarray:
        if callable(self.value):
            return _call_value_function(self.value, x)
        return np.full(x.shape[0], float(self.value), dtype=np.float32)

    def compute_loss_dict(self, x, y, ops) -> dict:
        """Requires ``ops.normals`` to be set by the trainer."""
        normals = ops.normals            # (n, n_spatial_dims) tensor
        n_dims  = normals.shape[1] if hasattr(normals, 'shape') else len(normals[0])
        du_dn   = sum(
            ops.directional_derivative(x, self.component, d) * normals[:, d]
            for d in range(n_dims)
        )
        target = ops.to_tensor(self.get_value(x))
        return {self.name or 'bc': ops.mean_sq(du_dn - target)}


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
    component: int           = 0
    name:      Optional[str] = None

    def get_value(self, x) -> np.ndarray:
        if callable(self.value):
            return _call_value_function(self.value, x)
        return np.full(x.shape[0], float(self.value), dtype=np.float32)

    def compute_loss_dict(self, x, y, ops) -> dict:
        """Requires ``ops.normals`` to be set by the trainer."""
        normals = ops.normals
        n_dims  = normals.shape[1] if hasattr(normals, 'shape') else len(normals[0])
        du_dn   = sum(
            ops.directional_derivative(x, self.component, d) * normals[:, d]
            for d in range(n_dims)
        )
        gamma    = ops.to_tensor(self.get_value(x))
        residual = self.alpha * y[:, self.component] + self.beta * du_dn - gamma
        return {self.name or 'bc': ops.mean_sq(residual)}


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
        TermCustomBC(region='right', f=traction, name='traction')
    """
    region:       str
    f:            Callable
    name:         Optional[str]       = None
    output_names: Optional[List[str]] = None

    def compute_loss_dict(self, x, y, ops) -> dict:
        residuals = _call_residual_fn(self.f, x, y, ops)
        base  = self.name or 'bc'
        names = (self.output_names
                 or ([base] if len(residuals) == 1
                     else [f'{base}_{k}' for k in range(len(residuals))]))
        return {n: ops.mean_sq(r) for n, r in zip(names, residuals)}


@dataclass
class TermPeriodicBC:
    """Periodic condition: ``u(x_a) = u(x_b)`` for matched points on two regions.

    The trainer samples ``n_pairs`` matched point pairs from ``region_a`` and
    ``region_b`` (either by random node pairing on a mesh or by sampling the
    lower / upper face of the same spatial dimension on a cubic domain).

    This term has **no** ``compute_loss_dict`` — the trainer handles it
    specially because it needs two forward passes at *different* coordinates.

    Args:
        region_a: Name of the first boundary region (e.g. ``'xmin'``).
        region_b: Name of the matching boundary region (e.g. ``'xmax'``).
        n_pairs: Number of collocation pairs.  ``None`` → use all available
                 node pairs (DomainMesh) or a default count (DomainCubic).
        component: Output component to enforce, or ``None`` for all components.
        name: Base label used in weight dicts.
        match_x_derivative: If ``True``, also penalise the tangential derivative
                             mismatch ``|du/ds(x_a) - du/ds(x_b)|^2``.

    Example::

        TermPeriodicBC(region_a='xmin', region_b='xmax',
                       component=0, name='periodic_x')
    """
    region_a:           str
    region_b:           str
    n_pairs:            Optional[int] = None
    component:          Optional[int] = None
    name:               Optional[str] = None
    match_x_derivative: bool          = True


# ---------------------------------------------------------------------------
# Data / observation term
# ---------------------------------------------------------------------------

@dataclass
class TermPoints:
    """Data-assimilation term at arbitrary fixed coordinates.

    Points may lie anywhere — on the boundary, in the interior, or scattered
    measurement data.  The trainer always uses **all** points in a single
    batch (no domain-based sampling).

    When multiple output components are supervised, provide an ``(N, K)``
    target array and a matching ``components`` list.  Each column produces
    one independent sub-loss.

    Args:
        inputs: ``(N, n_dims)`` observation coordinates.
        outputs: ``(N, K)`` target values.  A 1-D array of length ``N`` is
                 treated as ``(N, 1)``.
        components: Network output index for each column of *outputs*.
        name: Base label used in weight dicts and plots.
        output_names: Optional per-column labels.

    Example::

        obs = TermPoints(inputs=x_meas, outputs=u_meas, components=0, name='obs')
    """
    inputs:       'np.ndarray'
    outputs:      'np.ndarray'
    components:   Union[int, List[int]] = 0
    name:         Optional[str]         = None
    output_names: Optional[List[str]]   = None

    def __post_init__(self):
        self.inputs  = np.asarray(self.inputs,  dtype=np.float32)
        self.outputs = np.asarray(self.outputs, dtype=np.float32)
        if self.inputs.ndim  == 1: self.inputs  = self.inputs[:, None]
        if self.outputs.ndim == 1: self.outputs = self.outputs[:, None]
        if isinstance(self.components, int):
            self.components = [self.components]
        n_cols = self.outputs.shape[1]
        if n_cols != len(self.components):
            raise ValueError(
                f"TermPoints '{self.name}': outputs has {n_cols} column(s) "
                f"but components has {len(self.components)} element(s)."
            )

    def get_input_names(self) -> List[str]:
        base = self.name or 'pts'
        if self.output_names is not None:
            return list(self.output_names)
        return [base] if len(self.components) == 1 else \
               [f'{base}_{k}' for k in range(len(self.components))]

    def compute_loss_dict(self, x, y, ops) -> dict:
        out_names = self.get_input_names()
        return {
            oname: ops.mean_sq(y[:, comp] - ops.to_tensor(self.outputs[:, k]))
            for k, (comp, oname) in enumerate(zip(self.components, out_names))
        }


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

    Example::

        TermInner(fn=heat_residual, name='heat')
        TermInner(fn=heat_residual, name='heat_left', region='left_half')
    """
    fn:           Callable
    name:         str              = 'pde'
    region:       Optional[str]    = None
    output_names: Optional[List[str]] = None

    def compute_loss_dict(self, x, y, ops) -> dict:
        residuals = _call_residual_fn(self.fn, x, y, ops)
        base  = self.name
        names = (self.output_names
                 or ([base] if len(residuals) == 1
                     else [f'{base}_{k}' for k in range(len(residuals))]))
        return {n: ops.mean_sq(r) for n, r in zip(names, residuals)}


@dataclass
class TermInitial:
    """Initial-condition residual term, sampled from the ``t = t_min`` slice.

    Structurally identical to :class:`TermInner` but marks the term as an
    initial condition so the trainer samples from the initial-time surface
    rather than the full interior.

    Args:
        fn: Residual callable ``f(x, y[, params[, derivative]])``.
        name: Base label used in loss / weight dicts.
        output_names: Per-residual labels when *fn* returns a tuple.

    Example::

        TermInitial(fn=lambda x, y: y[:, 0] - u0(x), name='ic')
    """
    fn:           Callable
    name:         str              = 'ic'
    output_names: Optional[List[str]] = None

    def compute_loss_dict(self, x, y, ops) -> dict:
        residuals = _call_residual_fn(self.fn, x, y, ops)
        base  = self.name
        names = (self.output_names
                 or ([base] if len(residuals) == 1
                     else [f'{base}_{k}' for k in range(len(residuals))]))
        return {n: ops.mean_sq(r) for n, r in zip(names, residuals)}


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
