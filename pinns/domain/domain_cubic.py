import warnings
import numpy as np
from itertools import product
from typing import TYPE_CHECKING, Callable, Optional, Union, Literal, Tuple, List, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..problems.terms import TermDirichletBC, TermNeumannBC, TermRobinBC, TermPoints

# ============================================================================
# Subdomain info class for easy filtering
# ============================================================================

@dataclass
class SubdomainInfo:
    """
    Information about a subdomain, used for filtering active subdomains.
    
    Provides convenient access to subdomain properties like bounds, center, etc.
    
    Attributes:
        index (int): Flat index of the subdomain
        multi_index (tuple): Per-dimension index tuple, e.g., (i, j) for 2D
        xmin (np.ndarray): Lower bounds of the subdomain (shape: n_dims)
        xmax (np.ndarray): Upper bounds of the subdomain (shape: n_dims)
        center (np.ndarray): Center of the subdomain (shape: n_dims)
        
    Example:
        # Filter by subdomain position in FBPINN
        fbpinn = FBPINN(
            partition, network,
            active_subdomains=lambda sub: sub.xmin[0] >= 0  # Only x >= 0
        )
    """
    index: int
    multi_index: tuple
    xmin: np.ndarray
    xmax: np.ndarray
    
    @property
    def center(self) -> np.ndarray:
        """Center of the subdomain."""
        return (self.xmin + self.xmax) / 2
    
    @property
    def size(self) -> np.ndarray:
        """Size of the subdomain in each dimension."""
        return self.xmax - self.xmin
    
    def __repr__(self):
        return f"SubdomainInfo(index={self.index}, xmin={self.xmin.tolist()}, xmax={self.xmax.tolist()})"

# ============================================================================
# Sampling utilities
# ============================================================================

def _uniform_samples(n_points: int, n_dims: int, rng) -> np.ndarray:
    """Generate uniform random samples in [0, 1]^n_dims."""
    return rng.uniform(0, 1, size=(n_points, n_dims))


def _latin_hypercube_samples(n_points: int, n_dims: int, rng) -> np.ndarray:
    """Generate Latin Hypercube samples in [0, 1]^n_dims."""
    samples = np.zeros((n_points, n_dims))
    for d in range(n_dims):
        # Create n_points intervals and sample one point per interval
        intervals = np.linspace(0, 1, n_points + 1)
        points = rng.uniform(intervals[:-1], intervals[1:])
        # Shuffle the points
        rng.shuffle(points)
        samples[:, d] = points
    return samples


def _sobol_samples(n_points: int, n_dims: int, rng) -> np.ndarray:
    """Generate Sobol sequence samples in [0, 1]^n_dims."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=rng)
        # Round up to next power of 2 for Sobol
        n_pow2 = 2 ** int(np.ceil(np.log2(max(n_points, 2))))
        samples = sampler.random(n_pow2)
        return samples[:n_points]
    except ImportError:
        # Fallback to uniform if scipy not available
        return _uniform_samples(n_points, n_dims, rng)


def _halton_samples(n_points: int, n_dims: int, rng) -> np.ndarray:
    """Generate Halton sequence samples in [0, 1]^n_dims."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Halton(d=n_dims, scramble=True, seed=rng)
        return sampler.random(n_points)
    except ImportError:
        # Fallback to uniform if scipy not available
        return _uniform_samples(n_points, n_dims, rng)


SamplingMethod = Literal["uniform", "latin_hypercube", "lhs", "sobol", "halton"]

_SAMPLING_METHODS = {
    "uniform": _uniform_samples,
    "latin_hypercube": _latin_hypercube_samples,
    "lhs": _latin_hypercube_samples,
    "sobol": _sobol_samples,
    "halton": _halton_samples,
}


def sample_unit_hypercube(
    n_points: int, 
    n_dims: int, 
    method: Union[SamplingMethod, Callable] = "uniform",
    rng=None
) -> np.ndarray:
    """
    Generate samples in [0, 1]^n_dims using the specified method.
    
    Args:
        n_points: Number of points to sample.
        n_dims: Number of dimensions.
        method: Sampling method. Can be:
            - "uniform": Standard uniform random sampling (default)
            - "latin_hypercube" or "lhs": Latin Hypercube Sampling
            - "sobol": Sobol quasi-random sequence
            - "halton": Halton quasi-random sequence
            - Callable: Custom function with signature (n_points, n_dims, rng) -> ndarray
        rng: Random number generator (np.random.Generator).
        
    Returns:
        np.ndarray of shape (n_points, n_dims) with values in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if callable(method):
        return method(n_points, n_dims, rng)
    elif method in _SAMPLING_METHODS:
        return _SAMPLING_METHODS[method](n_points, n_dims, rng)
    else:
        raise ValueError(
            f"Unknown sampling method: {method}. "
            f"Available: {list(_SAMPLING_METHODS.keys())} or a callable."
        )


def transform_samples(
    samples: np.ndarray,
    xmin: np.ndarray,
    xmax: np.ndarray,
    transform: Optional[Callable] = None,
    reject_outside: bool = True,
    rng=None,
    method: Union[SamplingMethod, Callable] = "uniform",
    max_attempts: int = 100,
    params: Optional[dict] = None
) -> np.ndarray:
    """
    Transform samples from [0,1]^n to the target domain.
    
    Args:
        samples: Samples in [0, 1]^n_dims.
        xmin: Lower bounds of target domain.
        xmax: Upper bounds of target domain.
        transform: Optional custom transform function. Takes samples in [0,1]^n
                  and params dict, returns transformed samples. This is like an inverse CDF.
                  Signature: transform(X, params) -> X_transformed
                  If None, uses linear scaling to [xmin, xmax].
        reject_outside: If True and transform is provided, reject points outside
                       [xmin, xmax] and resample until we have enough points.
        rng: Random number generator for resampling.
        method: Sampling method for generating new samples during rejection.
        max_attempts: Maximum resampling attempts to avoid infinite loops.
        params: Optional params dict passed to transform function.
        
    Returns:
        np.ndarray of shape (n_points, n_dims) in [xmin, xmax].
    """
    n_points = samples.shape[0]
    n_dims = samples.shape[1]
    
    if transform is None:
        # Linear transform from [0,1] to [xmin, xmax]
        return xmin + samples * (xmax - xmin)
    
    # Apply custom transform with params
    transformed = transform(samples, params)
    
    if not reject_outside:
        return transformed
    
    # Rejection sampling: keep only points inside domain
    if rng is None:
        rng = np.random.default_rng()
    
    inside_mask = np.all((transformed >= xmin) & (transformed <= xmax), axis=1)
    result = transformed[inside_mask]
    
    attempts = 0
    while len(result) < n_points and attempts < max_attempts:
        # Need more points
        n_needed = n_points - len(result)
        # Sample extra to account for rejection rate
        rejection_rate = 1.0 - len(result) / max(len(transformed), 1)
        n_extra = int(n_needed / max(1 - rejection_rate, 0.1)) + 10
        
        new_samples = sample_unit_hypercube(n_extra, n_dims, method=method, rng=rng)
        new_transformed = transform(new_samples, params)
        
        inside_mask = np.all((new_transformed >= xmin) & (new_transformed <= xmax), axis=1)
        result = np.vstack([result, new_transformed[inside_mask]])
        attempts += 1
    
    # Trim to exact number
    return result[:n_points]

def bump(x, x_min, x_max, sigma_lower, sigma_upper=None):
    """
    Smooth bump function using sigmoid products (NumPy).

    Args:
        x: Array of shape (..., n_dims)
        x_min: Lower bounds, array of shape (n_dims,) or broadcastable
        x_max: Upper bounds, array of shape (n_dims,) or broadcastable
        sigma_lower: Smoothing width for the lower (xmin) boundary.
        sigma_upper: Smoothing width for upper (xmax) boundary.
                    If None, uses sigma_lower.

    Returns:
        ndarray of shape (...,) with bump values in [0, 1]
    """
    if sigma_upper is None:
        sigma_upper = sigma_lower
    lower_arg = np.clip((x - x_min) / sigma_lower, -10, 10)
    upper_arg = np.clip((x - x_max) / sigma_upper, -10, 10)
    lower_sigmoid = 1.0 / (1 + np.exp(-lower_arg))
    upper_sigmoid = 1.0 / (1 + np.exp(upper_arg))
    return np.prod(lower_sigmoid * upper_sigmoid, axis=-1)


def bump_vectorized(x, x_min, x_max, sigma_lower, sigma_upper=None):
    """
    Vectorized bump function for multiple subdomains at once (NumPy).

    Args:
        x: Array of shape (batch_size, n_dims)
        x_min: Shape (n_subdomains, n_dims)
        x_max: Shape (n_subdomains, n_dims)
        sigma_lower: Shape (n_subdomains, n_dims)
        sigma_upper: Shape (n_subdomains, n_dims) or None

    Returns:
        ndarray of shape (batch_size, n_subdomains)
    """
    if sigma_upper is None:
        sigma_upper = sigma_lower
    # x: (batch, 1, dims)  |  x_min etc.: (1, subdomains, dims)
    x_e = x[:, np.newaxis, :]
    x_min = x_min[np.newaxis, :, :]
    x_max = x_max[np.newaxis, :, :]
    sl = sigma_lower[np.newaxis, :, :]
    su = sigma_upper[np.newaxis, :, :]
    lower_arg = np.clip((x_e - x_min) / sl, -10, 10)
    upper_arg = np.clip((x_e - x_max) / su, -10, 10)
    lower_sigmoid = 1.0 / (1 + np.exp(-lower_arg))
    upper_sigmoid = 1.0 / (1 + np.exp(upper_arg))
    return np.prod(lower_sigmoid * upper_sigmoid, axis=-1)


class DomainCubic:
    """
    A rectangular (hyper-cubic) spatial domain, with an optional time axis.

    The ``space`` argument specifies the spatial extent one dimension at a time.
    Each element of the list selects the construction mode for that dimension:

    * **Plain bound** — a 2-tuple ``(x_min, x_max)``::

        domain = DomainCubic(
            space=[(0, 1), (0, 1)],
        )

    * **Partition (FBPINN)** — a strictly-increasing 1-D array of breakpoints::

        domain = DomainCubic(
            space=[np.array([0, 0.1, 0.5, 1]), np.array([0, 0.1, 0.5, 1])],
        )

      ``n`` breakpoints create ``n-1`` subdomains per dimension.  All dimensions
      must either all be plain tuples or all be arrays — mixing is not allowed.

    An optional ``time`` argument adds a time dimension:

    * ``time=None`` (default) — **stationary** domain (no time coordinate).
    * ``time=(t_min, t_max)`` or ``time=[t_min, t_max]`` — **continuous**
      time interval.
    * ``time=array`` with >2 breakpoints — **partitioned** time axis;
      ``n`` breakpoints create ``n-1`` time subdomains.

    Args:
        space: List of per-dimension specifications.  Each element is either a
            ``(min, max)`` tuple (plain mode) or a strictly-increasing 1-D array
            of breakpoints (partition mode).  May be ``None`` (or omitted) when
            ``time`` is provided and no spatial dimensions are needed (e.g. pure
            ODE problems).
        sampling_method: Default interior sampling method.  One of
            ``"uniform"`` (default), ``"latin_hypercube"``/``"lhs"``,
            ``"sobol"``, ``"halton"``, or a callable
            ``(n_points, n_dims, rng) -> ndarray``.
        sampling_transform: Optional inverse-CDF transform for interior
            sampling.  Points outside the domain are rejected and resampled.
        time: Time specification — ``None``, a 2-element sequence
            ``(t_min, t_max)`` for a continuous interval, or a strictly-increasing
            array/list with >2 values for a partitioned time axis.
    """

    def __init__(self, space=None, sampling_method="uniform",
                 sampling_transform=None, time=None):

        if space is None and time is None:
            raise ValueError(
                "DomainCubic requires at least one of 'space' or 'time'.")

        _partition_mode = False

        if space is not None:
            if not space:
                raise ValueError("'space' must be a non-empty list.")

            # ── Detect mode from the first element ───────────────────────────
            # tuple → plain bound (min, max)
            # array / list → partition breakpoints
            _is_partition_elem = lambda e: not isinstance(e, tuple)
            _modes = [_is_partition_elem(e) for e in space]
            if any(_modes) and not all(_modes):
                raise ValueError(
                    "All elements of 'space' must be the same type: either all "
                    "2-tuples (plain bounds) or all arrays (partition breakpoints)."
                )
            _partition_mode = all(_modes)

            # ── Partition mode ────────────────────────────────────────────────
            if _partition_mode:
                self.grid_positions = [np.asarray(e, dtype=np.float64) for e in space]
                for i, p in enumerate(self.grid_positions):
                    if len(p) < 2:
                        raise ValueError(
                            f"space[{i}] must have at least 2 breakpoints "
                            f"to define at least 1 subdomain."
                        )
                    if not np.all(np.diff(p) > 0):
                        raise ValueError(
                            f"space[{i}] must be strictly increasing."
                        )
                xmin = [p[0] for p in self.grid_positions]
                xmax = [p[-1] for p in self.grid_positions]
            else:
                # ── Plain mode ────────────────────────────────────────────────
                self.grid_positions = None
                for i, e in enumerate(space):
                    if len(e) != 2:
                        raise ValueError(
                            f"space[{i}] must be a 2-tuple (min, max), got length {len(e)}."
                        )
                xmin = [float(e[0]) for e in space]
                xmax = [float(e[1]) for e in space]
        else:
            # ── No spatial dimensions (pure-time domain) ──────────────────
            self.grid_positions = None
            xmin = []
            xmax = []

        # ── Common bounds setup ───────────────────────────────────────────
        self.xmin = np.asarray(xmin, dtype=np.float64)
        self.xmax = np.asarray(xmax, dtype=np.float64)

        if len(self.xmin) > 0 and np.any(self.xmin >= self.xmax):
            raise ValueError("min must be strictly less than max in all dimensions.")

        self.n_dims = len(self.xmin)
        self.sampling_method = sampling_method
        self.sampling_transform = sampling_transform

        # Storage for boundary conditions
        self.boundary_conditions: 'List[Union[TermDirichletBC, TermNeumannBC, TermRobinBC, TermPoints]]' = []

        # _spatial_dims records the number of spatial-only dimensions
        self._spatial_dims = self.n_dims

        # ── Partition-specific setup ──────────────────────────────────────
        if _partition_mode:
            self.n_subdomains_per_dim = [len(p) - 1 for p in self.grid_positions]
            self.n_subdomains = int(np.prod(self.n_subdomains_per_dim))
            self._subdomain_centers = None
            self._compute_subdomains()
        else:
            self.n_subdomains = None
            self.n_subdomains_per_dim = None

        # ── Time axis ────────────────────────────────────────────────────
        self._t_min            = None
        self._t_max            = None
        self.t_interval        = None
        self.time_grid_positions = None   # 1-D array of breakpoints (partitioned time)
        self.n_time_subdomains   = None   # number of time subdomains (partitioned time)

        # derived read-only property: has_time, is_time_partitioned
        if time is None:
            pass  # stationary
        else:
            time_arr = np.asarray(time, dtype=float).ravel()
            if len(time_arr) < 2:
                raise ValueError(
                    "DomainCubic: 'time' must be None, a 2-element sequence "
                    "(t_min, t_max), or a strictly-increasing array with >2 "
                    "breakpoints for a partitioned time axis."
                )
            if not np.all(np.diff(time_arr) > 0):
                raise ValueError(
                    "DomainCubic: time values must be strictly increasing.")
            self._t_min = float(time_arr[0])
            self._t_max = float(time_arr[-1])
            if len(time_arr) > 2:
                # partitioned time axis
                self.time_grid_positions = time_arr
                self.n_time_subdomains   = len(time_arr) - 1
            self.t_interval = [self._t_min, self._t_max]
            # Extend the domain bounds to include time as the last dimension.
            self.xmin  = np.append(self.xmin, self._t_min)
            self.xmax  = np.append(self.xmax, self._t_max)
            self.n_dims = len(self.xmin)

        # ── Custom sampling regions ───────────────────────────────────────
        # _inner_regions:    name → (lo, hi) arrays shape (n_dims,)
        # _boundary_regions: name → {'fixed_dim', 'fixed_val', 'lo', 'hi'}
        self._inner_regions: dict = {}
        self._boundary_regions: dict = {}
        self._periodic_regions: dict = {}

    @property
    def has_time(self) -> bool:
        """``True`` if the domain was constructed with a time axis."""
        return self._t_min is not None

    @property
    def is_time_partitioned(self) -> bool:
        """``True`` if the time axis is partitioned (>2 breakpoints given)."""
        return self.time_grid_positions is not None

    @property
    def bounds(self):
        """``(xmin, xmax)`` arrays covering the full domain (including time if present)."""
        return self.xmin, self.xmax

    @property
    def volume(self):
        """Hypervolume of the domain (product of all axis extents)."""
        return np.prod(self.xmax - self.xmin)

    @property
    def extents(self):
        """Per-dimension lengths: ``xmax - xmin``."""
        return self.xmax - self.xmin
    
    def sample_interior(self, n_points, region=None, size='equal', rng=None,
                        method=None, transform=None, params=None, mode='uniform',
                        t_interval=None):
        """
        Sample points from the interior of the domain.

        Args:
            n_points (int): Total number of points to return.
            region: Selects which part of the interior to sample from.

                * ``None`` / ``'all'`` — uniform sampling over the full domain
                  (default).
                * ``'subdomains'`` — equally distribute *n_points* across every
                  spatial subdomain (requires a partitioned domain).
                * ``str`` — sample from a named region registered with
                  :meth:`add_inner`.
                * ``tuple`` — sample from a specific subdomain by multi-index.
                  For a plain-partitioned domain pass ``(i, j, ...)``, one
                  integer per spatial dimension.  When the time axis is also
                  partitioned append the time index: ``(i, j, t)``.
                * ``list`` — concatenate samples from each element; elements
                  may be any of the forms above.  Use *size* to control the
                  split.
            size (str | list[float]): How to distribute *n_points* when
                *region* is a list:

                * ``'equal'`` (default) — equal split.
                * ``'size'`` — proportional to region volume.
                * list of non-negative floats — explicit weights (normalised).
            rng (np.random.Generator | None): Random-number generator.  A new
                default generator is created when ``None``.
            method (str | None): Sampling method override.  Recognised values:
                ``'uniform'``, ``'lhs'`` / ``'latin_hypercube'``, ``'sobol'``,
                ``'halton'``.  Defaults to the value set at construction time.
            transform: Optional inverse-CDF transform applied to each sample.
                Points that fall outside the region after transformation are
                rejected and resampled.
            params (dict | None): Extra keyword arguments forwarded to
                *transform*.
            mode (str): ``'uniform'`` (default) or ``'per_partition'``.  Only
                relevant when *region* is ``None`` / ``'all'`` and the domain
                is partitioned; ``'per_partition'`` is equivalent to
                ``region='subdomains'``.

        Returns:
            np.ndarray: Shape ``(n_points, n_dims)`` where *n_dims* is the
            number of spatial dimensions plus 1 if the domain has a time axis.
        """
        if rng is None:
            rng = np.random.default_rng()
        if method is None:
            method = self.sampling_method

        _BUILTIN_INNER = ('all', 'partition', 'subdomains')
        _partition_size = size  # may be overridden below when region='partition'/'subdomains'
        # ── list-of-regions branch ────────────────────────────────────────
        if isinstance(region, list):
            for r in region:
                if (not isinstance(r, tuple) and
                        r not in _BUILTIN_INNER and
                        r not in self._inner_regions):
                    raise KeyError(
                        f"Unknown inner region: {r!r}. "
                        f"Built-in: {list(_BUILTIN_INNER)}  "
                        f"Custom: {list(self._inner_regions)}")
            counts = self._resolve_counts(n_points, region, size,
                                          kind='inner')
            parts = []
            for r, n in zip(region, counts):
                if n > 0:
                    parts.append(self.sample_interior(
                        n, region=r, rng=rng, method=method,
                        transform=transform, params=params,
                        t_interval=t_interval))
            return np.vstack(parts)

        # ── tuple multi-index: specific sub-partition ─────────────────────
        if isinstance(region, tuple):
            if self.grid_positions is None:
                raise ValueError(
                    "Tuple region requires a partitioned domain.")
            has_time_grid = self.time_grid_positions is not None
            expected_len = self._spatial_dims + (1 if has_time_grid else 0)
            if len(region) != expected_len:
                if has_time_grid:
                    raise ValueError(
                        f"Tuple region must have {expected_len} elements: "
                        f"{self._spatial_dims} spatial + 1 time index "
                        f"(got {len(region)}).")
                else:
                    raise ValueError(
                        f"Tuple region must have {self._spatial_dims} elements, "
                        f"one per spatial dimension (got {len(region)}).")
            mi = region[:self._spatial_dims]
            for d, idx in enumerate(mi):
                n_cells = len(self.grid_positions[d]) - 1
                if not (0 <= idx < n_cells):
                    raise IndexError(
                        f"Sub-partition index {idx} out of range for "
                        f"dimension {d} (0..{n_cells - 1}).")
            sub_lo = np.array([self.grid_positions[d][mi[d]]
                                for d in range(self._spatial_dims)])
            sub_hi = np.array([self.grid_positions[d][mi[d] + 1]
                                for d in range(self._spatial_dims)])
            s = sample_unit_hypercube(n_points, self._spatial_dims,
                                      method=method, rng=rng)
            pts = transform_samples(s, sub_lo, sub_hi, transform=transform,
                                    reject_outside=True, rng=rng,
                                    method=method, params=params)
            if self._t_min is not None:
                if has_time_grid:
                    t_idx = region[self._spatial_dims]
                    n_t_cells = self.n_time_subdomains
                    if not (0 <= t_idx < n_t_cells):
                        raise IndexError(
                            f"Time partition index {t_idx} out of range "
                            f"(0..{n_t_cells - 1}).")
                    t_lo = float(self.time_grid_positions[t_idx])
                    t_hi = float(self.time_grid_positions[t_idx + 1])
                else:
                    t_lo, t_hi = self._t_min, self._t_max
                if t_interval is not None:
                    _ti_lo, _ti_hi = float(t_interval[0]), float(t_interval[1])
                    if _ti_lo < self._t_min or _ti_hi > self._t_max or _ti_lo >= _ti_hi:
                        raise ValueError(
                            f"t_interval [{_ti_lo}, {_ti_hi}] is invalid: must satisfy "
                            f"[{self._t_min}, {self._t_max}] and lo < hi.")
                    t_lo, t_hi = _ti_lo, _ti_hi
                t = rng.uniform(t_lo, t_hi, (len(pts), 1))
                pts = np.hstack([pts, t])
            return pts

        # ── built-in region names ─────────────────────────────────────────
        if region == 'all' or region is None:
            region = None  # fall through to full-domain logic below
        elif region in ('partition', 'subdomains'):
            if self.grid_positions is None:
                raise ValueError(
                    f"region={region!r} requires a partitioned domain. "
                    "Use set_partition() or construct DomainCubic with breakpoint arrays.")
            mode = 'per_partition'
            region = None  # fall through to per-partition logic below
            _partition_size = size  # carry 'size' into per-partition branch

        # ── single named custom region ────────────────────────────────────
        if region is not None:
            if region not in self._inner_regions:
                raise KeyError(
                    f"Unknown inner region: {region!r}. "
                    f"Built-in: {list(_BUILTIN_INNER)}  "
                    f"Custom: {list(self._inner_regions)}")
            lo, hi = self._inner_regions[region]
            samples = sample_unit_hypercube(n_points, self.n_dims, method=method, rng=rng)
            return transform_samples(samples, lo, hi, transform=None,
                                     reject_outside=True, rng=rng,
                                     method=method, params=params)

        # Spatial xmin/xmax (exclude time dimension if present)
        sp_min = self.xmin[:self._spatial_dims]
        sp_max = self.xmax[:self._spatial_dims]

        if mode == 'uniform' or self.grid_positions is None:
            samples = sample_unit_hypercube(n_points, self._spatial_dims, method=method, rng=rng)
            pts = transform_samples(
                samples, sp_min, sp_max,
                transform=transform, reject_outside=True, rng=rng, method=method, params=params
            )
        elif mode == 'per_partition':
            # Compute per-cell counts respecting _partition_size
            all_sub_min, all_sub_max = [], []
            for dim_idx in range(self.n_subdomains):
                mi = self.get_multi_index(dim_idx)
                all_sub_min.append([self.grid_positions[d][mi[d]] for d in range(self._spatial_dims)])
                all_sub_max.append([self.grid_positions[d][mi[d] + 1] for d in range(self._spatial_dims)])
            all_sub_min = np.array(all_sub_min)
            all_sub_max = np.array(all_sub_max)

            if isinstance(_partition_size, (list, tuple, np.ndarray)):
                weights = np.asarray(_partition_size, dtype=float)
                if len(weights) != self.n_subdomains:
                    raise ValueError(
                        f"size list length ({len(weights)}) must match n_subdomains "
                        f"({self.n_subdomains}).")
                weights = weights / weights.sum()
            elif _partition_size == 'size':
                vols = np.prod(all_sub_max - all_sub_min, axis=1).astype(float)
                weights = vols / vols.sum()
            else:  # 'equal'
                weights = np.ones(self.n_subdomains) / self.n_subdomains

            # Largest-remainder rounding so counts sum exactly to n_points
            raw = weights * n_points
            counts = np.floor(raw).astype(int)
            fracs = raw - counts
            for idx in np.argsort(-fracs)[:n_points - counts.sum()]:
                counts[idx] += 1

            all_points = []
            for dim_idx, (sub_min, sub_max, n_pts) in enumerate(
                    zip(all_sub_min, all_sub_max, counts)):
                if n_pts > 0:
                    s = sample_unit_hypercube(int(n_pts), self._spatial_dims, method=method, rng=rng)
                    all_points.append(transform_samples(
                        s, sub_min, sub_max,
                        transform=transform, reject_outside=True, rng=rng, method=method, params=params
                    ))
            pts = np.vstack(all_points)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'uniform' or 'per_partition'.")

        # Append time column if domain is time-dependent
        if t_interval is not None and self._t_min is None:
            raise ValueError(
                "t_interval was provided but this domain has no time axis.")
        if self._t_min is not None:
            if rng is None:
                rng = np.random.default_rng()
            if t_interval is not None:
                _ti_lo, _ti_hi = float(t_interval[0]), float(t_interval[1])
                if _ti_lo < self._t_min or _ti_hi > self._t_max or _ti_lo >= _ti_hi:
                    raise ValueError(
                        f"t_interval [{_ti_lo}, {_ti_hi}] is invalid: must satisfy "
                        f"[{self._t_min}, {self._t_max}] and lo < hi.")
                _t_lo, _t_hi = _ti_lo, _ti_hi
            else:
                _t_lo, _t_hi = self._t_min, self._t_max
            t = rng.uniform(_t_lo, _t_hi, (len(pts), 1))
            pts = np.hstack([pts, t])
        return pts

    def sample_initial(self, n_points, region=None, size='equal', rng=None,
                       method=None, transform=None, params=None, mode='uniform'):
        """Sample points on the initial time slice (``t == t_min``).

        Convenience wrapper around :meth:`sample_interior` that fixes the time
        column to ``_t_min`` after sampling spatial coordinates.

        Args:
            n_points (int): Number of points to return.
            region: Forwarded to :meth:`sample_interior`.
            size: Forwarded to :meth:`sample_interior`.
            rng: Random-number generator.
            method: Sampling method override.
            transform: Optional inverse-CDF transform.
            params: Extra keyword arguments forwarded to *transform*.
            mode: Forwarded to :meth:`sample_interior`.

        Returns:
            np.ndarray: Shape ``(n_points, n_spatial + 1)`` with the time
            column set to ``_t_min``.

        Raises:
            ValueError: If the domain has no time axis.
        """
        if self._t_min is None:
            raise ValueError(
                "sample_initial requires a time axis. "
                "Construct DomainCubic with the 'time' argument.")
        # Sample spatial coords with a degenerate t_interval so no time is
        # appended by sample_interior, then attach the fixed t_min column.
        pts = self.sample_interior(
            n_points, region=region, size=size, rng=rng,
            method=method, transform=transform, params=params, mode=mode,
            t_interval=None,
        )
        # sample_interior already appended t uniformly; replace it with t_min
        pts[:, self._spatial_dims] = self._t_min
        return pts

    def sample_boundary(self, n_points, region=None, size='equal', rng=None,
                        method=None, transform=None, params=None,
                        t_interval=None):
        """
        Sample points on the boundary of the domain.

        Args:
            n_points (int): Total number of points to return.
            region: Selects which boundary faces or regions to sample from.

                * ``None`` / ``'all'`` — distribute *n_points* equally across
                  all spatial faces (default).
                * ``'subdomains'`` — distribute equally across the outer faces
                  of every boundary subdomain (requires a partitioned domain).
                * ``str`` — sample from a single named region registered with
                  :meth:`add_boundary`.
                * ``list`` — concatenate samples from each element.  Use *size*
                  to control the split.
            size (str | list[float]): How to distribute *n_points* when
                *region* is a list:

                * ``'equal'`` (default) — equal split.
                * ``'size'`` — proportional to face area.
                * list of non-negative floats — explicit weights (normalised).
            rng (np.random.Generator | None): Random-number generator.
            method (str | None): Sampling method override.
            transform: Optional inverse-CDF transform.
            params (dict | None): Extra keyword arguments forwarded to
                *transform*.

        Returns:
            np.ndarray: Shape ``(n_points, n_dims)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        if method is None:
            method = self.sampling_method
        if transform is None:
            transform = self.sampling_transform

        _BUILTIN_BOUNDARY = ('all', 'partition', 'partition_outer',
                              'partition_inner')

        # ── list-of-regions branch ────────────────────────────────────────
        if isinstance(region, list):
            for r in region:
                if r not in _BUILTIN_BOUNDARY and r not in self._boundary_regions:
                    raise KeyError(
                        f"Unknown boundary region: {r!r}. "
                        f"Built-in: {list(_BUILTIN_BOUNDARY)}  "
                        f"Custom: {list(self._boundary_regions)}")
            counts = self._resolve_counts(n_points, region, size,
                                          kind='boundary')
            parts = []
            for r, n in zip(region, counts):
                if n > 0:
                    parts.append(self.sample_boundary(
                        n, region=r, rng=rng, method=method,
                        transform=transform, params=params,
                        t_interval=t_interval))
            return np.vstack(parts)

        # ── built-in region names ─────────────────────────────────────────
        if region == 'all' or region is None:
            # all faces, equally distributed
            faces = [(d, s) for d in range(self._spatial_dims) for s in (0, 1)]
            n_faces = len(faces)
            ppp = n_points // n_faces
            rem = n_points % n_faces
            all_pts = []
            for i, (dim, side) in enumerate(faces):
                n_pts = ppp + (1 if i < rem else 0)
                if n_pts > 0:
                    all_pts.append(self._sample_face(
                        n_pts, dim, side, rng, method, transform, params,
                        t_interval=t_interval))
            return np.vstack(all_pts)

        if region == 'partition_outer':
            if self.grid_positions is None:
                raise ValueError(
                    "region='partition_outer' requires a partitioned domain. "
                    "Use set_partition() or construct DomainCubic with breakpoint arrays.")
            return self._sample_partition_outer_boundary(
                n_points, size, rng, method, transform, params)

        if region == 'partition_inner':
            if self.grid_positions is None:
                raise ValueError(
                    "region='partition_inner' requires a partitioned domain. "
                    "Use set_partition() or construct DomainCubic with breakpoint arrays.")
            return self._sample_partition_inner_boundaries(
                n_points, size, rng, method, params)

        if region == 'subdomains':
            region = 'partition'

        if region == 'partition':
            if self.grid_positions is None:
                raise ValueError(
                    "region='partition' (boundary) requires a partitioned domain. "
                    "Use set_partition() or construct DomainCubic with breakpoint arrays.")
            if isinstance(size, (list, tuple, np.ndarray)):
                w = np.asarray(size, dtype=float)
                if len(w) != 2:
                    raise ValueError(
                        "size for region='partition' (boundary) must have 2 weights: "
                        "[outer_weight, inner_weight].")
                w = w / w.sum()
                n_outer = int(round(w[0] * n_points))
                n_inner = n_points - n_outer
            elif size == 'size':
                # Weight by total area of outer vs inner faces
                outer_area = self._partition_outer_area()
                inner_area = self._partition_inner_area()
                total = outer_area + inner_area
                n_outer = int(round(outer_area / total * n_points))
                n_inner = n_points - n_outer
            else:  # 'equal'
                n_outer = n_points // 2
                n_inner = n_points - n_outer
            outer = self._sample_partition_outer_boundary(
                n_outer, size, rng, method, transform, params)
            inner = self._sample_partition_inner_boundaries(
                n_inner, size, rng, method, params)
            return np.vstack([outer, inner])

        # ── built-in face labels: 'xmin', 'xmax', 'ymin', etc. ─────────────
        if isinstance(region, str) and region.strip().lower() in self._BOUNDARY_LABEL_MAP:
            tup = self._parse_boundary_str(region)
            for dim, side in enumerate(tup):
                if side is not None:
                    return self._sample_face(
                        n_points, dim, side, rng, method, transform, params,
                        t_interval=t_interval)
            raise ValueError(f"_parse_boundary_str returned no fixed dim for {region!r}")

        # ── periodic region: return stacked pair (side A then side B) ─────
        if isinstance(region, str) and region in self._periodic_regions:
            _pr = self._periodic_regions[region]
            x_a = self.sample_boundary(n_points, region=_pr['region_a'],
                                       rng=rng, method=method,
                                       transform=transform, params=params,
                                       t_interval=t_interval)
            x_b = x_a.copy()
            # Replace the periodic axis column with the B-side value
            _blm = self._BOUNDARY_LABEL_MAP
            _rb = _pr['region_b']
            if _rb in _blm:
                _dim, _side = _blm[_rb]
                _val = self.xmax[_dim] if _side == 1 else self.xmin[_dim]
                x_b[:, _dim] = _val
            return np.concatenate([x_a, x_b], axis=0)

        # ── single named custom region ────────────────────────────────────
        if isinstance(region, str):
            if region not in self._boundary_regions:
                raise KeyError(
                    f"Unknown boundary region: {region!r}. "
                    f"Built-in: {list(_BUILTIN_BOUNDARY)}  "
                    f"Custom: {list(self._boundary_regions)}")
            return self._sample_boundary_region(
                n_points, self._boundary_regions[region], rng, method, transform, params)

        raise TypeError("region must be None, a string, or a list of strings.")

    # ------------------------------------------------------------------ #
    #  Internal sampling helpers                                          #
    # ------------------------------------------------------------------ #

    def _resolve_counts(self, n_points, regions, size, kind):
        """
        Convert the *size* argument into a list of integer counts that sum
        exactly to *n_points* (largest-remainder rounding).

        Args:
            n_points (int): Total budget.
            regions (list): Region identifiers (used only to determine length
                and, for ``size='size'``, to look up volumes).
            size (str | list): ``'equal'``, ``'size'``, or an explicit weight
                list.
            kind (str): ``'inner'`` or ``'boundary'`` — selects which registry
                to query when ``size='size'``.

        Returns:
            list[int]: Per-region point counts.
        """
        n = len(regions)
        if isinstance(size, (list, tuple, np.ndarray)):
            weights = np.asarray(size, dtype=float)
            if len(weights) != n:
                raise ValueError(
                    f"size list length ({len(weights)}) must match number of "
                    f"regions ({n}).")
            if np.any(weights < 0):
                raise ValueError("size weights must be non-negative.")
            weights = weights / weights.sum()
        elif size == 'equal':
            weights = np.ones(n) / n
        elif size == 'size':
            if kind == 'inner':
                vols = np.array([
                    float(np.prod(self._inner_regions[r][1] -
                                  self._inner_regions[r][0]))
                    for r in regions], dtype=float)
            else:  # boundary
                vols = np.array([
                    float(np.prod(
                        self._boundary_regions[r]['hi'] -
                        self._boundary_regions[r]['lo']
                    )) for r in regions], dtype=float)
            s = vols.sum()
            weights = vols / s if s > 0 else np.ones(n) / n
        else:
            raise ValueError(
                f"size must be 'equal', 'size', or a list of weights, got {size!r}.")
        # Distribute n_points proportionally, handling rounding via largest-remainder
        raw = weights * n_points
        counts = np.floor(raw).astype(int)
        remainder = n_points - counts.sum()
        fracs = raw - counts
        for idx in np.argsort(-fracs)[:remainder]:
            counts[idx] += 1
        return counts.tolist()

    def _sample_face(self, n_points, dim, side, rng, method, transform, params,
                     t_interval=None):
        """
        Sample *n_points* on the axis-aligned face ``space[dim] == bound``.

        Args:
            n_points (int): Number of points.
            dim (int): Spatial dimension index (0-based).
            side (int): ``0`` for the lower face, ``1`` for the upper face.
            rng: Random-number generator.
            method: Sampling method.
            transform: Optional transform.
            params: Optional transform kwargs.

        Returns:
            np.ndarray: Shape ``(n_points, n_dims)``.
        """
        boundary_value = self.xmin[dim] if side == 0 else self.xmax[dim]
        sp_min = self.xmin[:self._spatial_dims]
        sp_max = self.xmax[:self._spatial_dims]
        samples = sample_unit_hypercube(n_points, self._spatial_dims,
                                         method=method, rng=rng)
        points = transform_samples(samples, sp_min, sp_max, transform=transform,
                                   reject_outside=True, rng=rng, method=method,
                                   params=params)
        points[:, dim] = boundary_value
        if t_interval is not None and self._t_min is None:
            raise ValueError(
                "t_interval was provided but this domain has no time axis.")
        if self._t_min is not None:
            if t_interval is not None:
                _ti_lo, _ti_hi = float(t_interval[0]), float(t_interval[1])
                if _ti_lo < self._t_min or _ti_hi > self._t_max or _ti_lo >= _ti_hi:
                    raise ValueError(
                        f"t_interval [{_ti_lo}, {_ti_hi}] is invalid: must satisfy "
                        f"[{self._t_min}, {self._t_max}] and lo < hi.")
                _t_lo, _t_hi = _ti_lo, _ti_hi
            else:
                _t_lo, _t_hi = self._t_min, self._t_max
            t = rng.uniform(_t_lo, _t_hi, (n_points, 1))
            points = np.hstack([points, t])
        return points

    def _sample_boundary_region(self, n_points, reg, rng, method, transform, params):
        """
        Sample *n_points* from a registered boundary-region dict.

        The region fixes one spatial dimension to a constant value (*fixed_dim*
        equals *fixed_val*) and samples all remaining dimensions uniformly
        within the per-dimension ``[lo, hi]`` bounds stored in *reg*.

        Args:
            n_points (int): Number of points.
            reg (dict): Region dict with keys ``fixed_dim``, ``fixed_val``,
                ``lo``, ``hi``.
            rng, method, transform, params: Forwarded to the sampler.

        Returns:
            np.ndarray: Shape ``(n_points, n_dims)``.
        """
        fixed_dim = reg['fixed_dim']
        fixed_val = reg['fixed_val']
        lo = reg['lo']
        hi = reg['hi']
        free_dims = [d for d in range(self.n_dims) if d != fixed_dim]
        n_free = len(free_dims)
        if n_free == 0:
            return np.tile(lo, (n_points, 1))
        free_lo = lo[free_dims]
        free_hi = hi[free_dims]
        samples = sample_unit_hypercube(n_points, n_free, method=method, rng=rng)
        free_pts = transform_samples(samples, free_lo, free_hi, transform=None,
                                     reject_outside=True, rng=rng, method=method,
                                     params=params)
        pts = np.empty((n_points, self.n_dims), dtype=np.float64)
        for j, d in enumerate(free_dims):
            pts[:, d] = free_pts[:, j]
        pts[:, fixed_dim] = fixed_val
        return pts

    # ------------------------------------------------------------------ #
    #  Custom region registration                                         #
    # ------------------------------------------------------------------ #

    def _partition_outer_area(self):
        """Total area of all outer boundary faces of the partition."""
        total = 0.0
        sp_ext = self.xmax[:self._spatial_dims] - self.xmin[:self._spatial_dims]
        for d in range(self._spatial_dims):
            face_area = np.prod(sp_ext[[dd for dd in range(self._spatial_dims) if dd != d]])
            # Two sides × number of cells on each boundary face
            n_boundary_cells_per_side = int(np.prod(
                [self.n_subdomains_per_dim[dd] for dd in range(self._spatial_dims) if dd != d]))
            total += 2 * n_boundary_cells_per_side * face_area
        return total

    def _partition_inner_area(self):
        """Total area of all internal interfaces of the partition."""
        total = 0.0
        sp_ext = self.xmax[:self._spatial_dims] - self.xmin[:self._spatial_dims]
        for d in range(self._spatial_dims):
            n_ifaces = len(self.grid_positions[d]) - 2  # internal breakpoints
            if n_ifaces <= 0:
                continue
            iface_area = np.prod(sp_ext[[dd for dd in range(self._spatial_dims) if dd != d]])
            total += n_ifaces * iface_area
        return total

    def _sample_partition_outer_boundary(self, n_points, size, rng, method, transform, params):
        """Sample from the outer boundary faces of partition cells that lie on the domain boundary."""
        boundary_subs = []
        for dim in range(self._spatial_dims):
            for side in (0, 1):
                for s_idx in range(self.n_subdomains):
                    mi = self.get_multi_index(s_idx)
                    pos = mi[dim]
                    if side == 0 and pos != 0:
                        continue
                    if side == 1 and pos != self.n_subdomains_per_dim[dim] - 1:
                        continue
                    boundary_subs.append((dim, side, mi))

        # Compute face areas for each boundary sub-face
        face_areas = np.array([
            np.prod([
                self.grid_positions[dd][mi[dd] + 1] - self.grid_positions[dd][mi[dd]]
                for dd in range(self._spatial_dims) if dd != dim
            ])
            for (dim, side, mi) in boundary_subs
        ], dtype=float)

        n_bs = len(boundary_subs)
        if isinstance(size, (list, tuple, np.ndarray)):
            weights = np.asarray(size, dtype=float)
            if len(weights) != n_bs:
                raise ValueError(
                    f"size list length ({len(weights)}) must match the number of "
                    f"boundary sub-faces ({n_bs}).")
            weights = weights / weights.sum()
        elif size == 'size':
            weights = face_areas / face_areas.sum()
        else:  # 'equal'
            weights = np.ones(n_bs) / n_bs

        raw = weights * n_points
        counts = np.floor(raw).astype(int)
        for idx in np.argsort(-(raw - counts))[:n_points - counts.sum()]:
            counts[idx] += 1

        all_pts = []
        for n_pts, (dim, side, mi) in zip(counts, boundary_subs):
            if n_pts == 0:
                continue
            sub_lo = np.array([self.grid_positions[d][mi[d]]
                               for d in range(self._spatial_dims)])
            sub_hi = np.array([self.grid_positions[d][mi[d] + 1]
                               for d in range(self._spatial_dims)])
            bval = self.xmin[dim] if side == 0 else self.xmax[dim]
            samples = sample_unit_hypercube(n_pts, self._spatial_dims,
                                            method=method, rng=rng)
            pts = transform_samples(samples, sub_lo, sub_hi, transform=None,
                                    reject_outside=True, rng=rng,
                                    method=method, params=params)
            pts[:, dim] = bval
            if self._t_min is not None:
                t = rng.uniform(self._t_min, self._t_max, (n_pts, 1))
                pts = np.hstack([pts, t])
            all_pts.append(pts)
        return np.vstack(all_pts)

    def _sample_partition_inner_boundaries(self, n_points, size, rng, method, params):
        """
        Sample from the internal interfaces between adjacent partition cells.

        For each spatial dimension *d* and each internal breakpoint *p* (i.e.
        ``grid_positions[d][1:-1]``), one interface is a co-dimension-1
        hyperplane ``x_d = p`` spanning the full domain in all other dimensions.
        Time is sampled uniformly when a time axis is present.

        Args:
            n_points: Total number of points.
            size:    ``'equal'`` (default), ``'size'`` (weight by interface area),
                     or an explicit list of weights per interface.
            rng:      Random-number generator.
            method:   Sampling method.
            params:   Extra params dict (unused, kept for API consistency).

        Returns:
            np.ndarray: Shape ``(n_points, n_dims)``.
        """
        # Collect all internal interfaces: (dim, position)
        interfaces = []
        for d in range(self._spatial_dims):
            for pos in self.grid_positions[d][1:-1]:
                interfaces.append((d, float(pos)))

        if not interfaces:
            raise ValueError(
                "Partition has no internal interfaces. "
                "Each spatial dimension needs at least 3 breakpoints to create "
                "at least one internal interface.")

        # Interface areas: product of domain extents in all free dimensions
        sp_min = self.xmin[:self._spatial_dims]
        sp_max = self.xmax[:self._spatial_dims]
        sp_ext = sp_max - sp_min
        iface_areas = np.array([
            np.prod(sp_ext[[dd for dd in range(self._spatial_dims) if dd != d]])
            for (d, _) in interfaces
        ], dtype=float)

        n_ifaces = len(interfaces)
        if isinstance(size, (list, tuple, np.ndarray)):
            weights = np.asarray(size, dtype=float)
            if len(weights) != n_ifaces:
                raise ValueError(
                    f"size list length ({len(weights)}) must match the number of "
                    f"internal interfaces ({n_ifaces}).")
            weights = weights / weights.sum()
        elif size == 'size':
            weights = iface_areas / iface_areas.sum()
        else:  # 'equal'
            weights = np.ones(n_ifaces) / n_ifaces

        raw = weights * n_points
        counts = np.floor(raw).astype(int)
        for idx in np.argsort(-(raw - counts))[:n_points - counts.sum()]:
            counts[idx] += 1

        all_pts = []
        for n_pts, (d, pos) in zip(counts, interfaces):
            if n_pts == 0:
                continue
            # Sample free (non-fixed) spatial dimensions
            free_dims = [dd for dd in range(self._spatial_dims) if dd != d]
            n_free = len(free_dims)
            if n_free == 0:
                pts_sp = np.full((n_pts, 1), pos)
            else:
                free_lo = sp_min[free_dims]
                free_hi = sp_max[free_dims]
                s = sample_unit_hypercube(n_pts, n_free, method=method, rng=rng)
                free_pts = transform_samples(
                    s, free_lo, free_hi, transform=None,
                    reject_outside=True, rng=rng, method=method, params=params)
                pts_sp = np.empty((n_pts, self._spatial_dims))
                for j, dd in enumerate(free_dims):
                    pts_sp[:, dd] = free_pts[:, j]
                pts_sp[:, d] = pos
            if self._t_min is not None:
                t = rng.uniform(self._t_min, self._t_max, (n_pts, 1))
                pts_sp = np.hstack([pts_sp, t])
            all_pts.append(pts_sp)
        return np.vstack(all_pts)

    def _resolve_time_arg(self, time):
        """Convert a ``time`` argument to a ``(t_lo, t_hi)`` float pair.

        *time* may be:
        * ``(lo, hi)`` tuple — used directly.
        * ``int`` — index into ``time_grid_positions`` (partitioned time).
        """
        if self._t_min is None:
            raise ValueError(
                "Cannot restrict time: domain was not constructed with a time axis.")
        if isinstance(time, (int, np.integer)):
            if self.time_grid_positions is None:
                raise ValueError(
                    "time=<int> requires a domain with time breakpoints "
                    "(pass e.g. time=[0, 0.1, 1] to DomainCubic).")
            n_t = self.n_time_subdomains
            if not (0 <= time < n_t):
                raise IndexError(
                    f"Time partition index {time} out of range (0..{n_t - 1}).")
            return (float(self.time_grid_positions[time]),
                    float(self.time_grid_positions[time + 1]))
        # assume (lo, hi)
        t_lo, t_hi = float(time[0]), float(time[1])
        if t_lo < self._t_min or t_hi > self._t_max:
            raise ValueError(
                f"Time range ({t_lo}, {t_hi}) exceeds domain time bounds "
                f"[{self._t_min}, {self._t_max}].")
        return t_lo, t_hi

    def _parse_inner_spec(self, domain_spec):
        """
        Parse a per-dimension ``(lo, hi)`` specification into
        ``(lo_arr, hi_arr)`` NumPy arrays of length *n_dims*.

        If *domain_spec* covers only the spatial dimensions and the domain has
        a time axis, the full time range is appended automatically.

        Args:
            domain_spec (list[tuple]): One ``(lo, hi)`` tuple per dimension
                (spatial only, or spatial + time).

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(lo, hi)`` arrays.

        Raises:
            ValueError: Incorrect length, out-of-bounds values, or ``lo >= hi``.
        """
        n = len(domain_spec)
        valid = (self._spatial_dims, self.n_dims)
        if n not in valid:
            raise ValueError(
                f"domain_spec must have {self._spatial_dims} (spatial) or "
                f"{self.n_dims} (full with time) elements, got {n}.")
        lo = np.array([float(e[0]) for e in domain_spec], dtype=np.float64)
        hi = np.array([float(e[1]) for e in domain_spec], dtype=np.float64)
        # Auto-extend with full time range if only spatial dims given
        if n == self._spatial_dims and self._t_min is not None:
            lo = np.append(lo, self._t_min)
            hi = np.append(hi, self._t_max)
        if np.any(lo < self.xmin) or np.any(hi > self.xmax):
            raise ValueError("Region extends outside domain bounds.")
        if np.any(lo >= hi):
            raise ValueError("Region lo must be strictly less than hi in all dimensions.")
        return lo, hi

    def _parse_boundary_spec(self, domain_spec):
        """
        Parse a boundary spec list into ``(fixed_dim, fixed_val, lo, hi)``.

        Exactly one element of *domain_spec* must be the string ``'min'`` or
        ``'max'``, which identifies the boundary face.  All other elements are
        ``(lo, hi)`` tuples restricting the range within that face.

        If *domain_spec* covers only the spatial dimensions and the domain has
        a time axis, the full time range is appended to *lo* / *hi*
        automatically.

        Args:
            domain_spec (list): One element per dimension; one must be
                ``'min'`` or ``'max'``, the rest are ``(lo, hi)`` tuples.

        Returns:
            tuple: ``(fixed_dim, fixed_val, lo_arr, hi_arr)``.

        Raises:
            ValueError: If the face selector is missing/ambiguous or
                a range exceeds domain bounds.
        """
        n = len(domain_spec)
        valid = (self._spatial_dims, self.n_dims)
        if n not in valid:
            raise ValueError(
                f"domain_spec must have {self._spatial_dims} (spatial) or "
                f"{self.n_dims} (full with time) elements, got {n}.")
        fixed_dims = [i for i, e in enumerate(domain_spec) if isinstance(e, str)]
        if len(fixed_dims) != 1:
            raise ValueError(
                "Exactly one element of domain_spec must be the string 'min' or 'max' "
                f"to specify the boundary face, got {len(fixed_dims)}.")
        fixed_dim = fixed_dims[0]
        side = domain_spec[fixed_dim]
        if side not in ('min', 'max'):
            raise ValueError(f"Fixed element must be 'min' or 'max', got {side!r}.")
        fixed_val = float(self.xmin[fixed_dim] if side == 'min' else self.xmax[fixed_dim])
        lo, hi = [], []
        for i, e in enumerate(domain_spec):
            if isinstance(e, str):
                lo.append(fixed_val)
                hi.append(fixed_val)
            else:
                lo.append(float(e[0]))
                hi.append(float(e[1]))
        lo = np.array(lo, dtype=np.float64)
        hi = np.array(hi, dtype=np.float64)
        # Auto-extend with full time range if only spatial dims given
        if n == self._spatial_dims and self._t_min is not None:
            lo = np.append(lo, self._t_min)
            hi = np.append(hi, self._t_max)
        # Validate free (non-fixed) dimensions are within bounds
        for d in range(len(lo)):
            if d == fixed_dim:
                continue
            if lo[d] < self.xmin[d] or hi[d] > self.xmax[d]:
                raise ValueError(
                    f"Dimension {d} range [{lo[d]}, {hi[d]}] exceeds domain "
                    f"bounds [{self.xmin[d]}, {self.xmax[d]}].")
        return fixed_dim, fixed_val, lo, hi

    def add_inner(self, domain_spec, name: str, time=None, strict: bool = True):
        """
        Register a named interior sampling region.

        Args:
            domain_spec: A list with one element per **spatial** dimension (or
                per **all** dimensions including time).  Each element is a
                ``(lo, hi)`` tuple.  If only spatial dimensions are given and
                the domain has a time axis, the full time range is used
                automatically.
            name (str): Identifier used when calling
                ``sample_interior(n, region=name)``.
            time: Optional time restriction.  Can be:

                * ``(lo, hi)`` — explicit time range.
                * ``int`` — index of a time partition (requires the domain to
                  have been constructed with time breakpoints).
                * ``None`` (default) — full time range.

            strict (bool): Accepted for API parity with :class:`DomainMesh`.
                Has no effect for box domains where regions are defined by
                axis-aligned bounding boxes.

        Example::

            domain.add_inner([(0.2, 0.8), (0.2, 0.8)], name='center')
            domain.add_inner([(0.2, 0.8), (0.2, 0.8)], name='center_t0', time=0)
            domain.add_inner([(0.2, 0.8), (0.2, 0.8)], name='center_early', time=(0, 0.1))
            pts = domain.sample_interior(500, region='center')
        """
        lo, hi = self._parse_inner_spec(domain_spec)
        if time is not None:
            t_lo, t_hi = self._resolve_time_arg(time)
            lo[-1] = t_lo
            hi[-1] = t_hi
        self._inner_regions[name] = (lo, hi)

    def add_boundary(self, domain_spec, name: str, time=None, strict: bool = True):
        """
        Register a named boundary sampling region.

        Exactly **one** element of *domain_spec* must be the string ``'min'``
        or ``'max'``, which selects the face (the minimum or maximum value of
        that dimension).  All other elements are ``(lo, hi)`` tuples that
        restrict the sampling range within the face.

        Args:
            domain_spec: A list with one element per **spatial** dimension (or
                all dimensions including time).  One element must be ``'min'``
                or ``'max'``; the rest are ``(lo, hi)`` range tuples.
            name (str): Identifier used when calling
                ``sample_boundary(n, region=name)``.
            time: Optional time restriction.  Can be:

                * ``(lo, hi)`` — explicit time range.
                * ``int`` — index of a time partition (requires the domain to
                  have been constructed with time breakpoints).
                * ``None`` (default) — full time range.

            strict (bool): Accepted for API parity with :class:`DomainMesh`.
                Has no effect for box domains where faces are exact.

        Example::

            # Full x=0 face
            domain.add_boundary(['min', (0, 1), (0, 1)], name='x_left')
            # x=0 face, restricted to first time partition
            domain.add_boundary(['min', (0, 1), (0, 1)], name='x_left_t0', time=0)
            # x=0 face, explicit time window
            domain.add_boundary(['min', (0, 1), (0, 1)], name='x_left_early', time=(0, 0.5))

            pts = domain.sample_boundary(500, region='x_left')
        """
        fixed_dim, fixed_val, lo, hi = self._parse_boundary_spec(domain_spec)
        if time is not None:
            t_lo, t_hi = self._resolve_time_arg(time)
            lo[-1] = t_lo
            hi[-1] = t_hi
        self._boundary_regions[name] = {
            'fixed_dim': fixed_dim,
            'fixed_val': fixed_val,
            'lo': lo,
            'hi': hi,
        }

    # Mapping from axis name to spatial dimension index
    _AXIS_DIM: dict = {'x': 0, 'y': 1, 'z': 2, 'x4': 3, 'x5': 4}

    def add_periodic(self, axis: str, name: str) -> None:
        """Register a **periodic** boundary pairing along a coordinate axis.

        Samples along ``axis_min`` and ``axis_max`` faces are paired for
        periodicity enforcement.  The pairing is stored in
        ``domain._periodic_regions[name]`` and can be referenced from
        :meth:`~pinns.problems.BaseProblem.add_periodic`.

        Args:
            axis: Spatial axis to enforce periodicity on.  One of
                ``'x'``, ``'y'``, ``'z'``, or ``'t'``.
            name: String label stored in ``_periodic_regions``.

        Raises:
            ValueError: If *axis* is not recognised.

        Example::

            domain.add_periodic('x', name='per_x')
        """
        _AXIS_LABELS = {'x': 0, 'y': 1, 'z': 2, 'x4': 3, 'x5': 4, 't': -1}
        if axis not in _AXIS_LABELS:
            raise ValueError(
                f"add_periodic: axis {axis!r} not recognised. "
                f"Choose from {list(_AXIS_LABELS)}.")
        self._periodic_regions[name] = {
            'region_a': f'{axis}min',
            'region_b': f'{axis}max',
            'axis':     axis,
        }

    def contains(self, x):
        """
        Test whether points lie inside the domain (inclusive bounds).

        Args:
            x (array-like): Shape ``(n_points, n_dims)`` or ``(n_dims,)``.

        Returns:
            bool | np.ndarray[bool]: Scalar boolean for a single point;
            1-D boolean array for a batch.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return np.all((x >= self.xmin) & (x <= self.xmax))
        return np.all((x >= self.xmin) & (x <= self.xmax), axis=1)
    
    # =========================================================================
    # Boundary Condition Methods
    # =========================================================================

    # Human-readable boundary label → (dim_offset, side)
    # dim_offset is relative to spatial dims; 't' means the time dimension.
    _BOUNDARY_LABEL_MAP: dict = {
        # Cartesian names
        'xmin': (0, 0), 'xmax': (0, 1),
        'ymin': (1, 0), 'ymax': (1, 1),
        'zmin': (2, 0), 'zmax': (2, 1),
        'tmin': ('t', 0), 'tmax': ('t', 1),
        # Common aliases
        'left':   (0, 0), 'right':  (0, 1),
        'bottom': (1, 0), 'top':    (1, 1),
        'front':  (2, 0), 'back':   (2, 1),
        'inlet':  (0, 0), 'outlet': (0, 1),
    }

    def _parse_boundary_str(self, boundary) -> Tuple:
        """Convert a boundary label string to the legacy tuple representation.

        Accepts either the existing tuple form (pass-through for backward
        compatibility) or one of the human-readable strings:
        ``'xmin'``, ``'xmax'``, ``'ymin'``, ``'ymax'``, ``'zmin'``,
        ``'zmax'``, ``'tmin'``, ``'tmax'``
        (and common aliases: ``'left'``/``'right'``, ``'bottom'``/``'top'``,
        ``'front'``/``'back'``).

        Returns a tuple of length ``n_dims`` where the fixed dimension is 0
        (lower) or 1 (upper) and all free dimensions are ``None``.
        """
        if isinstance(boundary, tuple):
            return boundary  # backward-compatible pass-through

        if not isinstance(boundary, str):
            raise TypeError(
                f"'boundary' must be a str (e.g. 'xmin') or tuple, "
                f"got {type(boundary).__name__!r}"
            )

        key = boundary.strip().lower()
        if key not in self._BOUNDARY_LABEL_MAP:
            valid = "', '".join(sorted(self._BOUNDARY_LABEL_MAP))
            raise ValueError(
                f"Unknown boundary label {boundary!r}. "
                f"Valid options are: '{valid}'"
            )

        dim_or_t, side = self._BOUNDARY_LABEL_MAP[key]
        if dim_or_t == 't':
            dim = self._spatial_dims  # time dimension is the last one
        else:
            dim = int(dim_or_t)
            if dim >= self._spatial_dims:
                raise ValueError(
                    f"Boundary label {boundary!r} refers to spatial dimension "
                    f"{dim}, but this domain only has {self._spatial_dims} "
                    f"spatial dimension(s)."
                )

        tup = [None] * self.n_dims
        tup[dim] = side
        return tuple(tup)

    def get_face_normal_direction(self, region: str):
        """Return ``(dim, sign)`` for a named boundary face.

        Returns ``None`` if `region` is not a built-in face label.
        `sign` is ``+1`` for the upper face and ``-1`` for the lower face.
        """
        key = region.strip().lower()
        if key not in self._BOUNDARY_LABEL_MAP:
            return None
        dim_or_t, side = self._BOUNDARY_LABEL_MAP[key]
        if dim_or_t == 't':
            dim = self._spatial_dims
        else:
            dim = int(dim_or_t)
        return (dim, 1 if side == 1 else -1)

    def get_boundary_normals(self, x: 'np.ndarray', region: str) -> 'np.ndarray':
        """Return outward unit normals at *x* for the given boundary *region*.

        For :class:`DomainCubic` every registered boundary region is an
        axis-aligned face, so the normal is **constant** and is simply
        broadcast to all *n* points.

        Args:
            x:      Query coordinates, shape ``(n, n_dims)``.
            region: Built-in face label (e.g. ``'xmin'``, ``'left'``) or a
                    custom region name registered with :meth:`add_boundary`.

        Returns:
            ``np.ndarray`` of shape ``(n, n_spatial_dims)`` with the outward
            unit normal replicated for every point.

        Raises:
            KeyError: If *region* is not recognised.
        """
        n = len(x)
        n_spatial = self._spatial_dims

        # 1) Built-in face label
        face_info = self.get_face_normal_direction(region)
        if face_info is not None:
            dim, sign = face_info
            dim = min(dim, n_spatial - 1)   # clamp time dim index
            normal = np.zeros(n_spatial, dtype=np.float32)
            normal[dim] = float(sign)
            return np.broadcast_to(normal, (n, n_spatial)).copy()

        # 2) Custom registered region (also axis-aligned)
        if region in self._boundary_regions:
            reg = self._boundary_regions[region]
            fixed_dim = reg['fixed_dim']
            fixed_val = float(reg['fixed_val'])
            sign = (+1.0 if abs(fixed_val - float(self.xmax[fixed_dim]))
                         < abs(fixed_val - float(self.xmin[fixed_dim]))
                    else -1.0)
            normal = np.zeros(n_spatial, dtype=np.float32)
            normal[fixed_dim] = sign
            return np.broadcast_to(normal, (n, n_spatial)).copy()

        raise KeyError(
            f"Unknown boundary region '{region}'. "
            f"Built-in: {list(self._BOUNDARY_LABEL_MAP)}  "
            f"Custom: {list(self._boundary_regions)}"
        )

    def _validate_subspace(self, boundary: Tuple, subspace, name: str) -> None:
        """Validate that all (lo, hi) ranges in *subspace* lie within the domain.

        Parameters
        ----------
        boundary:
            Parsed boundary tuple (elements are 0, 1, or None).
        subspace:
            List of ``(lo, hi)`` tuples, one per free dimension in order.
            If ``None``, validation is skipped.
        name:
            BC name used in error messages.
        """
        if subspace is None:
            return
        free_dims = [d for d, s in enumerate(boundary) if s is None]
        if len(subspace) > len(free_dims):
            raise ValueError(
                f"BC '{name}': subspace has {len(subspace)} tuple(s) but the "
                f"boundary face only has {len(free_dims)} free dimension(s)."
            )
        coord_labels = ['x', 'y', 'z', 't']
        for i, (lo, hi) in enumerate(subspace):
            d = free_dims[i]
            dom_lo = float(self.xmin[d])
            dom_hi = float(self.xmax[d])
            label = coord_labels[d] if d < len(coord_labels) else f'dim{d}'
            if lo < dom_lo or lo > dom_hi:
                raise ValueError(
                    f"BC '{name}': subspace[{i}] lower bound {lo} is out of "
                    f"domain range [{dom_lo}, {dom_hi}] for dimension '{label}'."
                )
            if hi < dom_lo or hi > dom_hi:
                raise ValueError(
                    f"BC '{name}': subspace[{i}] upper bound {hi} is out of "
                    f"domain range [{dom_lo}, {dom_hi}] for dimension '{label}'."
                )
            if lo >= hi:
                raise ValueError(
                    f"BC '{name}': subspace[{i}] lower bound {lo} must be "
                    f"strictly less than upper bound {hi}."
                )

    def _validate_time_subspace(self, time_subspace, name: str) -> None:
        """Validate that ``time_subspace=(t_lo, t_hi)`` lies within the time domain."""
        if time_subspace is None:
            return
        if self._t_min is None:
            raise ValueError(
                f"BC '{name}': time_subspace was specified but this domain "
                f"has no time dimension."
            )
        t_lo, t_hi = time_subspace
        if t_lo < self._t_min or t_lo > self._t_max:
            raise ValueError(
                f"BC '{name}': time_subspace lower bound {t_lo} is out of "
                f"time domain range [{self._t_min}, {self._t_max}]."
            )
        if t_hi < self._t_min or t_hi > self._t_max:
            raise ValueError(
                f"BC '{name}': time_subspace upper bound {t_hi} is out of "
                f"time domain range [{self._t_min}, {self._t_max}]."
            )
        if t_lo >= t_hi:
            raise ValueError(
                f"BC '{name}': time_subspace lower bound {t_lo} must be "
                f"strictly less than upper bound {t_hi}."
            )

    
    # =========================================================================
    # Partition-specific methods (only available when partition is provided)
    # =========================================================================

    def _compute_subdomains(self):
        """Compute subdomain centres for every cell in the partition grid."""
        indices = [range(n) for n in self.n_subdomains_per_dim]
        centers_list = []
        for idx in product(*indices):
            center = [(self.grid_positions[d][idx[d]] + self.grid_positions[d][idx[d] + 1]) / 2
                      for d in range(self._spatial_dims)]
            centers_list.append(center)
        self._subdomain_centers = np.array(centers_list)

    def get_subdomain_centers(self):
        """
        Return the centre point of every subdomain.

        Returns:
            np.ndarray: Shape ``(n_subdomains, spatial_dims)``.
        """
        self._require_partition()
        return self._subdomain_centers.copy()

    def get_internal_boundary_positions(self, dim):
        """
        Return the internal breakpoint positions along dimension *dim*
        (i.e. the grid positions excluding the outer boundary values).

        Args:
            dim (int): Spatial dimension index.

        Returns:
            np.ndarray: 1-D array of interior breakpoints.
        """
        self._require_partition()
        return self.grid_positions[dim][1:-1].copy()

    def get_subdomain_bounds(self):
        """
        Return axis-aligned bounds for every subdomain in the partition.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(lower_bounds, upper_bounds)``,
            each of shape ``(n_subdomains, spatial_dims)``.
        """
        self._require_partition()
        indices = [range(n) for n in self.n_subdomains_per_dim]
        lower_list, upper_list = [], []
        for idx in product(*indices):
            lower_list.append([self.grid_positions[d][idx[d]] for d in range(self._spatial_dims)])
            upper_list.append([self.grid_positions[d][idx[d] + 1] for d in range(self._spatial_dims)])
        return np.array(lower_list), np.array(upper_list)

    @property
    def subdomains(self) -> List['SubdomainInfo']:
        """
        List of :class:`SubdomainInfo` objects, one per subdomain in the
        partition (flat ordering, row-major).
        """
        self._require_partition()
        lower_bounds, upper_bounds = self.get_subdomain_bounds()
        return [
            SubdomainInfo(index=i, multi_index=self.get_multi_index(i),
                          xmin=lower_bounds[i], xmax=upper_bounds[i])
            for i in range(self.n_subdomains)
        ]

    def get_multi_index(self, flat_index):
        """
        Convert a flat subdomain index to a per-dimension tuple of indices.

        Args:
            flat_index (int): Linear index in ``[0, n_subdomains)``.

        Returns:
            tuple[int, ...]: One index per spatial dimension.
        """
        self._require_partition()
        indices = []
        remaining = flat_index
        for dim in reversed(range(self._spatial_dims)):
            n = self.n_subdomains_per_dim[dim]
            indices.append(remaining % n)
            remaining //= n
        return tuple(reversed(indices))

    def to_numpy(self):
        """
        Return subdomain centres as a NumPy array.

        Returns:
            np.ndarray: Shape ``(n_subdomains, spatial_dims)``, dtype float32.
        """
        self._require_partition()
        return np.array(self._subdomain_centers, dtype=np.float32)

    def _require_partition(self):
        """Raise :exc:`AttributeError` if the domain has no partition grid."""
        if self.grid_positions is None:
            raise AttributeError(
                "This method is only available for partitioned domains. "
                "Use set_partition() or construct with "
                "DomainCubic(space=[np.linspace(...), ...])."
            )

    def set_partition(self, space=None, time=None):
        """
        Add (or replace) a partition grid on an existing domain.

        A *partition* divides the domain into a regular grid of rectangular
        cells.  Once set, you can:

        * Sample from individual cells with a tuple region ``(i, j, ...)`` or
          ``(i, j, t_idx)`` (time partition index appended when applicable).
        * Sample all inner cells with ``region='partition'``.
        * Sample outer boundary faces with ``region='partition_outer'``
          in :meth:`sample_boundary`.
        * Sample internal interfaces between cells with
          ``region='partition_inner'``.
        * Sample all boundaries (inner + outer) with ``region='partition'``.

        Args:
            space (list | None): One entry per spatial dimension.  Each entry
                may be:

                * An **integer** ``n`` — auto-generates ``n`` equally-spaced
                  cells via ``np.linspace(xmin[d], xmax[d], n+1)``.
                * A **strictly-increasing 1-D array** of breakpoints.  The
                  first element must equal ``xmin[d]`` and the last must equal
                  ``xmax[d]`` (within floating-point tolerance).  At least 2
                  breakpoints (1 cell) per dimension are required; 3 or more
                  are needed to produce internal interfaces.

                Pass ``None`` to leave any existing spatial partition unchanged.
            time (int | array-like | None): Time partition.  May be:

                * An **integer** ``n`` — auto-generates ``n`` equally-spaced
                  time slabs via ``np.linspace(t_min, t_max, n+1)``.
                * A **strictly-increasing 1-D array** of breakpoints spanning
                  ``[t_min, t_max]``.

                Requires the domain to have a time axis.  Pass ``None`` to
                leave any existing time partition unchanged.

        Raises:
            ValueError: If ``space`` / ``time`` are inconsistent with the
                domain bounds, not strictly increasing, or too short.
            AttributeError: If ``time`` is given but the domain has no time
                axis.

        Example::

            dv = DomainCubic([[0, 1], [0, 1]], time=[0, 1])

            # Integer shorthand: 10 cells in x, custom breakpoints in y, 2 time slabs
            dv.set_partition(space=[10, [0, 0.2, 1]], time=2)

            # Or with explicit breakpoints:
            dv.set_partition(
                space=[np.linspace(0, 1, 11), np.linspace(0, 1, 11)],
                time=[0, 0.5, 1],
            )
            # sample interior of all cells
            pts_inner = dv.sample_interior(1000, region='partition')
            # sample internal interfaces
            pts_iface = dv.sample_boundary(500, region='partition_inner')
            # sample outer boundary faces
            pts_outer = dv.sample_boundary(500, region='partition_outer')
        """
        if space is not None:
            if len(space) != self._spatial_dims:
                raise ValueError(
                    f"'space' must have {self._spatial_dims} elements "
                    f"(one per spatial dimension), got {len(space)}.")
            # Allow integers: auto-generate uniform breakpoints for that dimension
            resolved = []
            for d, p in enumerate(space):
                if isinstance(p, (int, np.integer)):
                    lo = self.xmin[d]
                    hi = self.xmax[d]
                    resolved.append(np.linspace(lo, hi, int(p) + 1))
                else:
                    resolved.append(np.asarray(p, dtype=np.float64).ravel())
            grid_positions = resolved
            for d, (p, lo, hi) in enumerate(
                    zip(grid_positions,
                        self.xmin[:self._spatial_dims],
                        self.xmax[:self._spatial_dims])):
                if len(p) < 2:
                    raise ValueError(
                        f"space[{d}] must have at least 2 breakpoints "
                        f"(got {len(p)}).")
                if not np.all(np.diff(p) > 0):
                    raise ValueError(f"space[{d}] must be strictly increasing.")
                if not np.isclose(p[0], lo) or not np.isclose(p[-1], hi):
                    raise ValueError(
                        f"space[{d}] breakpoints must span the domain bounds "
                        f"[{lo}, {hi}], got [{p[0]}, {p[-1]}].")
            self.grid_positions = grid_positions
            self.n_subdomains_per_dim = [len(p) - 1 for p in grid_positions]
            self.n_subdomains = int(np.prod(self.n_subdomains_per_dim))
            self._subdomain_centers = None
            self._compute_subdomains()

        if time is not None:
            if self._t_min is None:
                raise AttributeError(
                    "Cannot set a time partition: the domain has no time axis. "
                    "Pass time=... to DomainCubic() first.")
            # Allow a single integer: auto-generate uniform time breakpoints
            if isinstance(time, (int, np.integer)):
                time = np.linspace(self._t_min, self._t_max, int(time) + 1)
            time_arr = np.asarray(time, dtype=float).ravel()
            if len(time_arr) < 2:
                raise ValueError(
                    "'time' must have at least 2 breakpoints.")
            if not np.all(np.diff(time_arr) > 0):
                raise ValueError("'time' breakpoints must be strictly increasing.")
            if (not np.isclose(time_arr[0], self._t_min) or
                    not np.isclose(time_arr[-1], self._t_max)):
                raise ValueError(
                    f"'time' breakpoints must span the domain time bounds "
                    f"[{self._t_min}, {self._t_max}], "
                    f"got [{time_arr[0]}, {time_arr[-1]}].")
            self.time_grid_positions = time_arr
            self.n_time_subdomains = len(time_arr) - 1

    def __len__(self):
        """Total number of subdomains (``n_subdomains``); requires a partitioned domain."""
        self._require_partition()
        return self.n_subdomains

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_jupyter() -> bool:
        """Return True when running inside a Jupyter kernel."""
        try:
            from IPython import get_ipython
            shell = get_ipython()
            return shell is not None and 'IPKernelApp' in shell.config
        except ImportError:
            return False

    def plot(
        self,
        show_overlaps: bool = False,
        region='all',
        boundary=None,
        points=None,
        figsize=None,
        backend: str = 'auto',
    ):
        """
        Visualise the domain geometry, partition subdomains, and any registered
        boundary conditions.

        Supports **1-D**, **2-D**, and **3-D** spatial domains.

        Args:
            show_overlaps (bool): When ``True`` and the domain has a time axis,
                split into one subplot per time phase (breakpoints come from
                registered BC ``time_window`` values).  Default ``False``
                (single panel).
            region: Which interior regions to highlight.  Accepts:

                * ``'all'`` (default) — draw the subdomain partition grid and
                  all named inner regions registered with :meth:`add_inner`.
                * ``'subdomains'`` — draw only the partition grid.
                * ``str`` — highlight a single named inner region.
                * ``list`` — highlight those specific regions / ``'subdomains'``.
                * ``None`` — draw nothing.
            boundary: Which boundary regions to highlight.  Accepts the same
                kinds of values as *region* but for regions registered with
                :meth:`add_boundary`.  ``'all'`` highlights every registered
                boundary region.  Default ``None`` (none drawn).  Registered
                boundary conditions are also drawn when this is not ``None``.
            points (array-like | None): An (N, D) array of spatial
                coordinates to scatter on every panel (e.g. training
                samples).  Plotted in red.  Default None.
            figsize (tuple | None): Figure size; auto-computed when ``None``.
            backend (str): Rendering backend — ``'auto'`` (default),
                ``'matplotlib'``, or ``'pyvista'``.  When ``'auto'`` and
                running inside a Jupyter kernel, ``pyvista`` is used for an
                interactive 3-D view; otherwise ``matplotlib`` is used.

        Returns:
            Axes or array of Axes (matplotlib), or a PyVista plotter
            (pyvista backend).
        """
        _backend = backend
        if _backend == 'auto':
            _backend = 'pyvista' if self._is_jupyter() else 'matplotlib'

        if _backend == 'pyvista':
            return self._plot_domain_pyvista()

        if self._spatial_dims > 3:
            raise NotImplementedError(
                "DomainCubic.plot_domain() supports 1-D, 2-D, and 3-D spatial domains only."
            )

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        sp_min = self.xmin[:self._spatial_dims]
        sp_max = self.xmax[:self._spatial_dims]
        _dim_labels = ['x', 'y', 'z']

        # ── Build time phases ─────────────────────────────────────────────
        has_time = self._t_min is not None
        breakpoints: list = []
        if has_time and show_overlaps:
            breakpoints.extend([self._t_min, self._t_max])
            for bc in self.boundary_conditions:
                tw = getattr(bc, 'time_window', None)
                if tw is not None:
                    for v in tw:
                        breakpoints.append(float(v))
        tol = 1e-10
        unique_bps = []
        for v in sorted(set(round(b, 12) for b in breakpoints)):
            if not unique_bps or abs(v - unique_bps[-1]) > tol:
                unique_bps.append(v)

        if show_overlaps and len(unique_bps) >= 2:
            phases = [(unique_bps[i], unique_bps[i + 1])
                      for i in range(len(unique_bps) - 1)]
        else:
            phases = [None]   # single panel

        def _bc_in_phase(bc, phase) -> bool:
            if phase is None:
                return True
            tw = getattr(bc, 'time_window', None)
            p_lo, p_hi = phase
            p_mid = 0.5 * (p_lo + p_hi)
            if tw is None:
                return True
            pts = [float(v) for v in tw]
            if not pts:
                return True
            if len(pts) > 2:
                return any(p_lo - tol <= v <= p_hi + tol for v in pts)
            a, b = min(pts), max(pts)
            if abs(a - b) < tol:
                return p_lo - tol <= a <= p_hi + tol
            return a - tol <= p_mid <= b + tol

        # ── Resolve region / boundary highlight sets ──────────────────────
        _BUILTIN = ('all', 'partition', 'subdomains')

        def _resolve_region_set(arg, registry):
            """Return list of region keys to highlight, or special tokens."""
            if arg is None:
                return []
            if arg == 'all':
                # partition grid + all custom named regions
                return ['partition'] + list(registry.keys())
            if arg in ('partition', 'subdomains',
                       'partition_outer', 'partition_inner'):
                # partition grid only, no custom named regions
                return ['partition']
            if arg == 'custom':
                # custom named regions only, no partition grid
                return list(registry.keys())
            if isinstance(arg, str):
                return [arg]
            return list(arg)  # list of strings

        _highlight_inner = _resolve_region_set(region, self._inner_regions)
        _highlight_boundary = _resolve_region_set(boundary, self._boundary_regions)

        # ── Partition boundary flags ───────────────────────────────────────
        _draw_partition_outer = (boundary in ('partition', 'partition_outer', 'all'))
        _draw_partition_inner = (boundary in ('partition', 'partition_inner', 'all'))
        # Remove partition tokens from named-region list (handled separately)
        _highlight_boundary = [b for b in _highlight_boundary
                               if b not in ('partition', 'partition_outer',
                                            'partition_inner', 'subdomains')]

        # ── Colour palettes ────────────────────────────────────────────────
        _colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        bc_color = {id(bc): _colors[i % len(_colors)]
                    for i, bc in enumerate(self.boundary_conditions)}
        # Inner region colors (skip first few reserved for BCs)
        _inner_color_list = ['tab:orange', 'tab:green', 'tab:purple',
                             'tab:brown', 'tab:pink', 'tab:cyan']
        _inner_colors = {name: _inner_color_list[i % len(_inner_color_list)]
                         for i, name in enumerate(self._inner_regions)}
        _boundary_color_list = ['tab:red', 'tab:blue', 'deeppink',
                                 'darkcyan', 'goldenrod', 'slateblue']
        _boundary_colors = {name: _boundary_color_list[i % len(_boundary_color_list)]
                            for i, name in enumerate(self._boundary_regions)}
        _partition_palette = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd',
                              '#8c564b', '#e377c2', '#bcbd22', '#17becf',
                              '#ff7f0e', '#7f7f7f']

        # ── Figure layout ─────────────────────────────────────────────────
        n_panels = len(phases)
        is_1d = (self._spatial_dims == 1)
        is_3d = (self._spatial_dims == 3)

        if is_1d:
            panel_w, panel_h = 6.0, 2.5
        elif is_3d:
            panel_w, panel_h = 6.0, 6.0
        else:
            panel_w, panel_h = 5.5, 5.0

        if figsize is None:
            _has_legend = (region is not None or boundary is not None)
            _legend_extra = 2.0 if _has_legend else 0.0
            figsize = (panel_w * n_panels + _legend_extra, panel_h)

        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            fig = plt.figure(figsize=figsize)
            _axes3d = [fig.add_subplot(1, n_panels, col + 1, projection='3d')
                       for col in range(n_panels)]
            axes = None
        else:
            fig, _ax2 = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
            axes = _ax2
            _axes3d = None

        # ── Helpers: 3-D geometry ─────────────────────────────────────────
        def _box_edges_3d(lo, hi):
            """12 edges of an axis-aligned box as (xs, ys, zs) tuples."""
            x0, y0, z0 = lo
            x1, y1, z1 = hi
            return [
                ([x0, x1], [y0, y0], [z0, z0]), ([x0, x1], [y1, y1], [z0, z0]),
                ([x0, x1], [y0, y0], [z1, z1]), ([x0, x1], [y1, y1], [z1, z1]),
                ([x0, x0], [y0, y1], [z0, z0]), ([x1, x1], [y0, y1], [z0, z0]),
                ([x0, x0], [y0, y1], [z1, z1]), ([x1, x1], [y0, y1], [z1, z1]),
                ([x0, x0], [y0, y0], [z0, z1]), ([x1, x1], [y0, y0], [z0, z1]),
                ([x0, x0], [y1, y1], [z0, z1]), ([x1, x1], [y1, y1], [z0, z1]),
            ]

        def _face_verts_3d(dim, side, lo, hi):
            """4 corner vertices of a box face for Poly3DCollection."""
            x0, y0, z0 = lo
            x1, y1, z1 = hi
            c = lo[dim] if side == 0 else hi[dim]
            if dim == 0:
                return [[c, y0, z0], [c, y1, z0], [c, y1, z1], [c, y0, z1]]
            elif dim == 1:
                return [[x0, c, z0], [x1, c, z0], [x1, c, z1], [x0, c, z1]]
            else:
                return [[x0, y0, c], [x1, y0, c], [x1, y1, c], [x0, y1, c]]

        # ── Helper: draw one 3-D panel ────────────────────────────────────
        def _draw_3d_panel(ax, phase):
            title = (f"t \u2208 [{phase[0]:.3g}, {phase[1]:.3g}]"
                     if phase is not None else "")
            ax.set_title(title, fontsize=9)
            lo = sp_min.tolist()
            hi = sp_max.tolist()

            for xs, ys, zs in _box_edges_3d(lo, hi):
                ax.plot(xs, ys, zs, color='#555', linewidth=1.5)

            # ── highlighted inner regions ─────────────────────────────
            if any(k in _highlight_inner for k in ('partition', 'subdomains')) and self.grid_positions is not None:
                lo_bounds, up_bounds = self.get_subdomain_bounds()
                for i in range(self.n_subdomains):
                    color = _partition_palette[i % len(_partition_palette)]
                    for dim in range(3):
                        for side in range(2):
                            verts = _face_verts_3d(dim, side,
                                                   lo_bounds[i, :3].tolist(),
                                                   up_bounds[i, :3].tolist())
                            ax.add_collection3d(Poly3DCollection(
                                [verts], alpha=0.10,
                                facecolor=color, edgecolor=color,
                                linewidth=0.4))
                    for xs, ys, zs in _box_edges_3d(lo_bounds[i, :3].tolist(),
                                                     up_bounds[i, :3].tolist()):
                        ax.plot(xs, ys, zs, color='grey', linewidth=0.7,
                                linestyle='--')
            for rname in _highlight_inner:
                if rname in ('partition', 'subdomains') or rname not in self._inner_regions:
                    continue
                r_lo, r_hi = self._inner_regions[rname]
                rcolor = _inner_colors[rname]
                for xs, ys, zs in _box_edges_3d(r_lo[:3].tolist(),
                                                  r_hi[:3].tolist()):
                    ax.plot(xs, ys, zs, color=rcolor, linewidth=1.5,
                            linestyle='-')
            # ── highlighted boundary regions ──────────────────────────
            for bname in _highlight_boundary:
                if bname in ('partition', 'subdomains') or bname not in self._boundary_regions:
                    continue
                reg = self._boundary_regions[bname]
                bcolor = _boundary_colors[bname]
                fdim, fval = reg['fixed_dim'], reg['fixed_val']
                b_lo = reg['lo'][:3].tolist()
                b_hi = reg['hi'][:3].tolist()
                verts = _face_verts_3d(fdim, 0 if fval == reg['lo'][fdim] else 1,
                                       b_lo, b_hi)
                ax.add_collection3d(Poly3DCollection(
                    [verts], alpha=0.35, facecolor=bcolor,
                    edgecolor=bcolor, linewidth=1.5))

            # ── Partition boundary drawing ──────────────────────────────
            lo3 = sp_min[:3].tolist()
            hi3 = sp_max[:3].tolist()
            if _draw_partition_outer:
                for dim in range(min(3, self._spatial_dims)):
                    for side in range(2):
                        verts = _face_verts_3d(dim, side, lo3, hi3)
                        ax.add_collection3d(Poly3DCollection(
                            [verts], alpha=0.15, facecolor='#ff7f0e',
                            edgecolor='#ff7f0e', linewidth=1.5))
            if _draw_partition_inner and self.grid_positions is not None:
                gp = self.grid_positions
                for dim in range(min(3, self._spatial_dims)):
                    for bk in gp[dim][1:-1]:
                        lo_f = lo3.copy(); hi_f = hi3.copy()
                        lo_f[dim] = bk; hi_f[dim] = bk
                        verts = _face_verts_3d(dim, 0, lo_f, hi3)
                        ax.add_collection3d(Poly3DCollection(
                            [verts], alpha=0.20, facecolor='#d62728',
                            edgecolor='#d62728', linewidth=1.0,
                            linestyle='dashed'))

            if points is not None:
                _pts = np.asarray(points)
                ax.scatter(_pts[:, 0], _pts[:, 1], _pts[:, 2],
                           s=8, color='tomato', alpha=0.5)

            margin = 0.04 * max(hi[d] - lo[d] for d in range(3))
            ax.set_xlim(lo[0] - margin, hi[0] + margin)
            ax.set_ylim(lo[1] - margin, hi[1] + margin)
            ax.set_zlim(lo[2] - margin, hi[2] + margin)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

            # Remove background panes and grid
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor('none')
            ax.grid(False)

            legend_handles = []
            # legend entries for highlighted named inner regions
            for rname in _highlight_inner:
                if rname not in ('partition', 'subdomains') and rname in self._inner_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_inner_colors[rname], label=f'inner:{rname}'))
            # legend entries for highlighted boundary regions
            for bname in _highlight_boundary:
                if bname not in ('partition', 'subdomains') and bname in self._boundary_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_boundary_colors[bname], label=f'boundary:{bname}'))

            if boundary is not None:
                for bc in self.boundary_conditions:
                    if not _bc_in_phase(bc, phase):
                        continue
                    color = bc_color[id(bc)]
                    label = getattr(bc, 'name', None) or type(bc).__name__
                    bc_boundary = getattr(bc, 'boundary', None)
                    if hasattr(bc, 'inputs'):
                        pts_in = np.asarray(bc.inputs)
                        ax.scatter(pts_in[:, 0], pts_in[:, 1], pts_in[:, 2],
                                   s=30, color=color, marker='x')
                        legend_handles.append(mpatches.Patch(color=color, label=label))
                        continue
                    if getattr(bc, 'bc_type', None) == 'cubic_periodic':
                        for side in (0, 1):
                            verts = _face_verts_3d(bc.dim, side, lo, hi)
                            ax.add_collection3d(Poly3DCollection(
                                [verts], alpha=0.35,
                                facecolor=color, edgecolor=color, linewidth=1.5))
                        legend_handles.append(mpatches.Patch(color=color, label=label))
                        continue
                    if bc_boundary is None:
                        continue
                    drawn = False
                    for dim_idx, side in enumerate(bc_boundary):
                        if side is None:
                            continue
                        verts = _face_verts_3d(dim_idx, side, lo, hi)
                        ax.add_collection3d(Poly3DCollection(
                            [verts], alpha=0.35,
                            facecolor=color, edgecolor=color, linewidth=1.5))
                        drawn = True
                    if drawn:
                        legend_handles.append(mpatches.Patch(color=color, label=label))
            if legend_handles:
                ax.legend(handles=legend_handles, fontsize=8,
                          loc='upper left', bbox_to_anchor=(1.01, 1),
                          borderaxespad=0, framealpha=0.8)

        # ── Helper: draw one 1-D panel ────────────────────────────────────
        def _draw_1d_panel(ax, phase):
            title = (f"t ∈ [{phase[0]:.3g}, {phase[1]:.3g}]"
                     if phase is not None else "")
            ax.set_title(title, fontsize=9)
            ax.axhline(0, color='lightgrey', linewidth=1)
            ax.plot([sp_min[0], sp_max[0]], [0, 0],
                    color='#555', linewidth=4, solid_capstyle='round')
            ax.set_xlim(sp_min[0] - 0.05 * (sp_max[0] - sp_min[0]),
                        sp_max[0] + 0.05 * (sp_max[0] - sp_min[0]))
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            ax.set_xlabel('x')

            if any(k in _highlight_inner for k in ('partition', 'subdomains')) and self.grid_positions is not None:
                lo_bounds, up_bounds = self.get_subdomain_bounds()
                for i in range(self.n_subdomains):
                    color = _partition_palette[i % len(_partition_palette)]
                    core_lo = lo_bounds[i, 0]
                    core_up = up_bounds[i, 0]
                    ax.axvspan(core_lo, core_up, alpha=0.30, color=color, zorder=2)
                    ax.axvline(core_lo, color='grey', linestyle='--',
                               linewidth=0.8, alpha=0.7)
                    ax.axvline(core_up, color='grey', linestyle='--',
                               linewidth=0.8, alpha=0.7)
            for rname in _highlight_inner:
                if rname in ('partition', 'subdomains') or rname not in self._inner_regions:
                    continue
                r_lo, r_hi = self._inner_regions[rname]
                rcolor = _inner_colors[rname]
                ax.axvspan(r_lo[0], r_hi[0], alpha=0.25, color=rcolor, zorder=2)
            for bname in _highlight_boundary:
                if bname in ('partition', 'subdomains') or bname not in self._boundary_regions:
                    continue
                reg = self._boundary_regions[bname]
                bcolor = _boundary_colors[bname]
                ax.axvline(reg['fixed_val'], color=bcolor, linewidth=2.5, zorder=4)

            # ── Partition boundary drawing ─────────────────────────────────
            if _draw_partition_outer:
                for coord in (sp_min[0], sp_max[0]):
                    ax.axvline(coord, color='#ff7f0e', linewidth=2.5, zorder=5)
            if _draw_partition_inner and self.grid_positions is not None:
                for x in self.grid_positions[0][1:-1]:
                    ax.axvline(x, color='#d62728', linewidth=1.5,
                               linestyle='--', zorder=5)

            if points is not None:
                _pts = np.asarray(points)
                ax.scatter(_pts[:, 0], np.zeros(len(_pts)),
                           s=8, color='tomato', alpha=0.5, zorder=3)

            legend_handles = []
            for rname in _highlight_inner:
                if rname not in ('partition', 'subdomains') and rname in self._inner_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_inner_colors[rname], label=f'inner:{rname}'))
            for bname in _highlight_boundary:
                if bname not in ('partition', 'subdomains') and bname in self._boundary_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_boundary_colors[bname], label=f'boundary:{bname}'))

            if boundary is not None:
                for bc in self.boundary_conditions:
                    if not _bc_in_phase(bc, phase):
                        continue
                    color = bc_color[id(bc)]
                    label = getattr(bc, 'name', None) or type(bc).__name__
                    bc_boundary = getattr(bc, 'boundary', None)
                    if hasattr(bc, 'inputs'):
                        pts_in = np.asarray(bc.inputs)
                        ax.scatter(pts_in[:, 0], np.zeros(len(pts_in)),
                                   s=30, color=color, zorder=5, marker='x')
                        legend_handles.append(mpatches.Patch(color=color, label=label))
                        continue
                    if getattr(bc, 'bc_type', None) == 'cubic_periodic':
                        for side in (0, 1):
                            coord = sp_min[bc.dim] if side == 0 else sp_max[bc.dim]
                            ax.plot([coord], [0], 'o', color=color,
                                    markersize=10, zorder=5)
                        legend_handles.append(mpatches.Patch(color=color, label=label))
                        continue
                    if bc_boundary is None:
                        continue
                    drawn = False
                    for dim_idx, side in enumerate(bc_boundary):
                        if side is None:
                            continue
                        coord = sp_min[dim_idx] if side == 0 else sp_max[dim_idx]
                        ax.plot([coord], [0], 'o', color=color,
                                markersize=10, zorder=5)
                        drawn = True
                    if drawn:
                        legend_handles.append(mpatches.Patch(color=color, label=label))
            if legend_handles:
                ax.legend(handles=legend_handles, fontsize=8,
                          loc='upper left', bbox_to_anchor=(1.01, 1),
                          borderaxespad=0, framealpha=0.8)

        # ── Helper: draw one 2-D projection panel ─────────────────────────
        def _draw_2d_panel(ax, dx, dy, phase, title_prefix=""):
            xlabel = _dim_labels[dx]
            ylabel = _dim_labels[dy]
            title = title_prefix
            if phase is not None:
                sep = " — " if title_prefix else ""
                title = title + sep + f"t ∈ [{phase[0]:.3g}, {phase[1]:.3g}]"
            ax.set_title(title, fontsize=9)

            # Domain bounding rectangle in this projection
            rx, ry = sp_min[dx], sp_min[dy]
            rw, rh = sp_max[dx] - sp_min[dx], sp_max[dy] - sp_min[dy]
            ax.add_patch(mpatches.FancyBboxPatch(
                (rx, ry), rw, rh,
                boxstyle='square,pad=0',
                edgecolor='#555', facecolor='#f5f5f5', linewidth=1.5, zorder=1
            ))

            if any(k in _highlight_inner for k in ('partition', 'subdomains')) and self.grid_positions is not None:
                lo_bounds, up_bounds = self.get_subdomain_bounds()
                for i in range(self.n_subdomains):
                    color = _partition_palette[i % len(_partition_palette)]
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (lo_bounds[i, dx], lo_bounds[i, dy]),
                        up_bounds[i, dx] - lo_bounds[i, dx],
                        up_bounds[i, dy] - lo_bounds[i, dy],
                        boxstyle='square,pad=0',
                        edgecolor=color, facecolor=color,
                        linewidth=0.8, alpha=0.30, zorder=2
                    ))
            for rname in _highlight_inner:
                if rname in ('partition', 'subdomains') or rname not in self._inner_regions:
                    continue
                r_lo, r_hi = self._inner_regions[rname]
                rcolor = _inner_colors[rname]
                ax.add_patch(mpatches.FancyBboxPatch(
                    (r_lo[dx], r_lo[dy]),
                    r_hi[dx] - r_lo[dx], r_hi[dy] - r_lo[dy],
                    boxstyle='square,pad=0',
                    edgecolor=rcolor, facecolor=rcolor,
                    linewidth=1.5, alpha=0.25, zorder=4
                ))
            for bname in _highlight_boundary:
                if bname in ('partition', 'subdomains') or bname not in self._boundary_regions:
                    continue
                reg = self._boundary_regions[bname]
                bcolor = _boundary_colors[bname]
                _draw_face_2d(ax, reg['fixed_dim'], 0 if reg['fixed_val'] == reg['lo'][reg['fixed_dim']] else 1,
                              dx, dy, bcolor, linewidth=3.0)

            # ── Partition boundary drawing ─────────────────────────────────
            if _draw_partition_outer:
                # domain outer faces
                for dim in (dx, dy):
                    for coord in (sp_min[dim], sp_max[dim]):
                        side = 0 if coord == sp_min[dim] else 1
                        _draw_face_2d(ax, dim, side, dx, dy, '#ff7f0e', linewidth=2.5)
            if _draw_partition_inner and self.grid_positions is not None:
                # internal breakpoint lines
                gp = self.grid_positions
                for dim in range(self._spatial_dims):
                    interior = gp[dim][1:-1]  # skip first and last
                    if dim == dx:
                        for x in interior:
                            ax.plot([x, x], [sp_min[dy], sp_max[dy]],
                                    color='#d62728', linewidth=1.5,
                                    linestyle='--', zorder=5)
                    elif dim == dy:
                        for y in interior:
                            ax.plot([sp_min[dx], sp_max[dx]], [y, y],
                                    color='#d62728', linewidth=1.5,
                                    linestyle='--', zorder=5)

            margin = 0.04 * max(rw, rh)
            ax.set_xlim(sp_min[dx] - margin, sp_max[dx] + margin)
            ax.set_ylim(sp_min[dy] - margin, sp_max[dy] + margin)
            ax.set_aspect('equal')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if points is not None:
                _pts = np.asarray(points)
                ax.scatter(_pts[:, dx], _pts[:, dy],
                           s=8, color='tomato', alpha=0.5, zorder=3)

            legend_handles = []
            for rname in _highlight_inner:
                if rname not in ('partition', 'subdomains') and rname in self._inner_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_inner_colors[rname], label=f'inner:{rname}'))
            for bname in _highlight_boundary:
                if bname not in ('partition', 'subdomains') and bname in self._boundary_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_boundary_colors[bname], label=f'boundary:{bname}'))

            if boundary is not None:
                for bc in self.boundary_conditions:
                    if not _bc_in_phase(bc, phase):
                        continue
                    color = bc_color[id(bc)]
                    label = getattr(bc, 'name', None) or type(bc).__name__
                    bc_boundary = getattr(bc, 'boundary', None)

                    if hasattr(bc, 'inputs'):
                        pts_in = np.asarray(bc.inputs)
                        ax.scatter(pts_in[:, dx], pts_in[:, dy],
                                   s=30, color=color, zorder=5, marker='x')
                        legend_handles.append(mpatches.Patch(color=color, label=label))
                        continue

                    if getattr(bc, 'bc_type', None) == 'cubic_periodic':
                        for side in (0, 1):
                            _draw_face_2d(ax, bc.dim, side, dx, dy, color, linewidth=2.5)
                        legend_handles.append(mpatches.Patch(color=color, label=label))
                        continue

                    if bc_boundary is None:
                        continue
                    drawn = False
                    for dim_idx, side in enumerate(bc_boundary):
                        if side is None:
                            continue
                        _draw_face_2d(ax, dim_idx, side, dx, dy, color, linewidth=3)
                        drawn = True
                    if drawn:
                        legend_handles.append(mpatches.Patch(color=color, label=label))

            if legend_handles:
                ax.legend(handles=legend_handles, fontsize=8,
                          loc='upper left', bbox_to_anchor=(1.01, 1),
                          borderaxespad=0, framealpha=0.8)

        # ── Helper: draw a face edge in a 2-D projection ──────────────────
        def _draw_face_2d(ax, dim, side, dx, dy, color, linewidth=3):
            coord = sp_min[dim] if side == 0 else sp_max[dim]
            if dim == dx:
                # vertical line in this projection
                ax.plot([coord, coord], [sp_min[dy], sp_max[dy]],
                        color=color, linewidth=linewidth,
                        solid_capstyle='round', zorder=4)
            elif dim == dy:
                # horizontal line in this projection
                ax.plot([sp_min[dx], sp_max[dx]], [coord, coord],
                        color=color, linewidth=linewidth,
                        solid_capstyle='round', zorder=4)
            # if dim is neither dx nor dy, the face is a full rect — skip for clarity

        # ── Draw all panels ───────────────────────────────────────────────
        for col, phase in enumerate(phases):
            if is_3d:
                _draw_3d_panel(_axes3d[col], phase)
            elif is_1d:
                _draw_1d_panel(axes[0, col], phase)
            else:
                _draw_2d_panel(axes[0, col], 0, 1, phase)

        plt.tight_layout()

        if is_3d:
            return _axes3d[0] if n_panels == 1 else np.array(_axes3d)
        else:
            return axes[0, 0] if n_panels == 1 else axes[0]

    def _plot_domain_pyvista(self):
        """Interactive PyVista visualisation of the domain box."""
        import pyvista as pv
        sp_min = self.xmin[:self._spatial_dims]
        sp_max = self.xmax[:self._spatial_dims]
        if self._spatial_dims == 2:
            box = pv.Box(bounds=(sp_min[0], sp_max[0],
                                  sp_min[1], sp_max[1], 0.0, 0.0))
        elif self._spatial_dims == 3:
            box = pv.Box(bounds=(sp_min[0], sp_max[0],
                                  sp_min[1], sp_max[1],
                                  sp_min[2], sp_max[2]))
        else:
            raise NotImplementedError("PyVista backend supports 2-D and 3-D domains.")
        pl = pv.Plotter(notebook=self._is_jupyter(), off_screen=True)
        pl.add_mesh(box, style='wireframe', color='#555555', line_width=2,
                    label='domain')
        pl.camera_position = 'xy'
        pl.add_axes()
        return pl.show(jupyter_backend='trame' if self._is_jupyter() else None)

    # ------------------------------------------------------------------
    def __repr__(self):
        n_bcs = len(self.boundary_conditions)
        sp_xmin = self.xmin[:self._spatial_dims].tolist()
        sp_xmax = self.xmax[:self._spatial_dims].tolist()
        space_str = [(lo, hi) for lo, hi in zip(sp_xmin, sp_xmax)]
        time_str = ""
        if self._t_min is not None:
            if self.is_time_partitioned:
                time_str = (f", time=[{self._t_min}, {self._t_max}], "
                            f"n_time_subdomains={self.n_time_subdomains}")
            else:
                time_str = f", time=[{self._t_min}, {self._t_max}]"
        if self.grid_positions is not None:
            return (
                f"DomainCubic(space={space_str}, "
                f"n_subdomains_per_dim={self.n_subdomains_per_dim}, "
                f"total_subdomains={self.n_subdomains}{time_str}, n_conditions={n_bcs})"
            )
        return f"DomainCubic(space={space_str}{time_str}, n_conditions={n_bcs})"


# =============================================================================
# Mesh-based domain
# =============================================================================

