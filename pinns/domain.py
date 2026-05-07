import warnings
import numpy as np
from itertools import product
from typing import TYPE_CHECKING, Callable, Optional, Union, Literal, Tuple, List, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    from .boundary import DirichletBC, NeumannBC, RobinBC, PointsetBC

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
            of breakpoints (partition mode).
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

    def __init__(self, space, sampling_method="uniform",
                 sampling_transform=None, time=None):

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

        # ── Common bounds setup ───────────────────────────────────────────
        self.xmin = np.asarray(xmin, dtype=np.float64)
        self.xmax = np.asarray(xmax, dtype=np.float64)

        if np.any(self.xmin >= self.xmax):
            raise ValueError("min must be strictly less than max in all dimensions.")

        self.n_dims = len(self.xmin)
        self.sampling_method = sampling_method
        self.sampling_transform = sampling_transform

        # Storage for boundary conditions
        self.boundary_conditions: 'List[Union[DirichletBC, NeumannBC, RobinBC, PointsetBC]]' = []

        # _spatial_dims must be set before _compute_subdomains
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
                        method=None, transform=None, params=None, mode='uniform'):
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

        _BUILTIN_INNER = ('all', 'subdomains')

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
                        transform=transform, params=params))
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
                t = rng.uniform(t_lo, t_hi, (len(pts), 1))
                pts = np.hstack([pts, t])
            return pts

        # ── built-in region names ─────────────────────────────────────────
        if region == 'all' or region is None:
            region = None  # fall through to full-domain logic below
        elif region == 'subdomains':
            if self.grid_positions is None:
                raise ValueError(
                    "region='subdomains' requires a partitioned domain "
                    "(constructed with breakpoint arrays).")
            mode = 'per_partition'
            region = None  # fall through to per-partition logic below

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
            points_per_subdomain = n_points // self.n_subdomains
            remainder = n_points % self.n_subdomains
            all_points = []
            for dim_idx in range(self.n_subdomains):
                multi_idx = self.get_multi_index(dim_idx)
                sub_min = np.array([self.grid_positions[d][multi_idx[d]] for d in range(self._spatial_dims)])
                sub_max = np.array([self.grid_positions[d][multi_idx[d] + 1] for d in range(self._spatial_dims)])
                n_pts = points_per_subdomain + (1 if dim_idx < remainder else 0)
                if n_pts > 0:
                    s = sample_unit_hypercube(n_pts, self._spatial_dims, method=method, rng=rng)
                    all_points.append(transform_samples(
                        s, sub_min, sub_max,
                        transform=transform, reject_outside=True, rng=rng, method=method, params=params
                    ))
            pts = np.vstack(all_points)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'uniform' or 'per_partition'.")

        # Append time column if domain is time-dependent
        if self._t_min is not None:
            if rng is None:
                rng = np.random.default_rng()
            t = rng.uniform(self._t_min, self._t_max, (len(pts), 1))
            pts = np.hstack([pts, t])
        return pts
    
    def sample_boundary(self, n_points, region=None, size='equal', rng=None,
                        method=None, transform=None, params=None):
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

        _BUILTIN_BOUNDARY = ('all', 'subdomains')

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
                        transform=transform, params=params))
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
                        n_pts, dim, side, rng, method, transform, params))
            return np.vstack(all_pts)

        if region == 'subdomains':
            if self.grid_positions is None:
                raise ValueError(
                    "region='subdomains' requires a partitioned domain.")
            # distribute across all boundary subdomains
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
            n_bs = len(boundary_subs)
            ppp = n_points // n_bs
            rem = n_points % n_bs
            all_pts = []
            for i, (dim, side, mi) in enumerate(boundary_subs):
                n_pts = ppp + (1 if i < rem else 0)
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

    def _sample_face(self, n_points, dim, side, rng, method, transform, params):
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
        if self._t_min is not None:
            t = rng.uniform(self._t_min, self._t_max, (n_points, 1))
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
                "Construct with DomainCubic(space=[np.linspace(...), ...])."
            )

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
        from pinns.boundary import DirichletBC, NeumannBC, RobinBC, PointsetBC, CubicPeriodicBC, InitialConditionBC

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
        _BUILTIN = ('all', 'subdomains')

        def _resolve_region_set(arg, registry):
            """Return list of region keys to highlight, or special tokens."""
            if arg is None:
                return []
            if arg == 'all':
                return ['subdomains'] + list(registry.keys())
            if arg == 'subdomains':
                return ['subdomains']
            if isinstance(arg, str):
                return [arg]
            return list(arg)  # list of strings

        _highlight_inner = _resolve_region_set(region, self._inner_regions)
        _highlight_boundary = _resolve_region_set(boundary, self._boundary_regions)

        # ── Colour cycles ─────────────────────────────────────────────────
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
            figsize = (panel_w * n_panels, panel_h)

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
            if 'subdomains' in _highlight_inner and self.grid_positions is not None:
                lo_bounds, up_bounds = self.get_subdomain_bounds()
                ext_lo = lo_bounds[:, :3]
                ext_up = up_bounds[:, :3]
                for i in range(self.n_subdomains):
                    for dim in range(3):
                        for side in range(2):
                            verts = _face_verts_3d(dim, side,
                                                   ext_lo[i].tolist(),
                                                   ext_up[i].tolist())
                            ax.add_collection3d(Poly3DCollection(
                                [verts], alpha=0.06,
                                facecolor='tab:orange', edgecolor='tab:orange',
                                linewidth=0.4))
                    for xs, ys, zs in _box_edges_3d(lo_bounds[i].tolist(),
                                                     up_bounds[i].tolist()):
                        ax.plot(xs, ys, zs, color='grey', linewidth=0.7,
                                linestyle='--')
            for rname in _highlight_inner:
                if rname == 'subdomains' or rname not in self._inner_regions:
                    continue
                r_lo, r_hi = self._inner_regions[rname]
                rcolor = _inner_colors[rname]
                for xs, ys, zs in _box_edges_3d(r_lo[:3].tolist(),
                                                  r_hi[:3].tolist()):
                    ax.plot(xs, ys, zs, color=rcolor, linewidth=1.5,
                            linestyle='-')
            # ── highlighted boundary regions ──────────────────────────
            for bname in _highlight_boundary:
                if bname == 'subdomains' or bname not in self._boundary_regions:
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
                if rname != 'subdomains' and rname in self._inner_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_inner_colors[rname], label=f'inner:{rname}'))
            # legend entries for highlighted boundary regions
            for bname in _highlight_boundary:
                if bname != 'subdomains' and bname in self._boundary_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_boundary_colors[bname], label=f'boundary:{bname}'))

            if boundary is not None:
                for bc in self.boundary_conditions:
                    if not _bc_in_phase(bc, phase):
                        continue
                    color = bc_color[id(bc)]
                    label = getattr(bc, 'name', None) or type(bc).__name__
                    bc_boundary = getattr(bc, 'boundary', None)
                    if isinstance(bc, PointsetBC):
                        pts_in = np.asarray(bc.inputs)
                        ax.scatter(pts_in[:, 0], pts_in[:, 1], pts_in[:, 2],
                                   s=30, color=color, marker='x')
                        legend_handles.append(mpatches.Patch(color=color, label=label))
                        continue
                    if isinstance(bc, CubicPeriodicBC):
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
                          loc='upper right', framealpha=0.8)

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

            if 'subdomains' in _highlight_inner and self.grid_positions is not None:
                lo_bounds, up_bounds = self.get_subdomain_bounds()
                ext_lo = lo_bounds[:, :1]
                ext_up = up_bounds[:, :1]
                for i in range(self.n_subdomains):
                    core_lo = lo_bounds[i, 0]
                    core_up = up_bounds[i, 0]
                    ex_lo   = ext_lo[i, 0]
                    ex_up   = ext_up[i, 0]
                    if core_lo > ex_lo:
                        ax.axvspan(ex_lo, core_lo, alpha=0.15,
                                   color='tab:orange', zorder=1)
                    if ex_up > core_up:
                        ax.axvspan(core_up, ex_up, alpha=0.15,
                                   color='tab:orange', zorder=1)
                    ax.axvline(core_lo, color='grey', linestyle='--',
                               linewidth=0.8, alpha=0.7)
                    ax.axvline(core_up, color='grey', linestyle='--',
                               linewidth=0.8, alpha=0.7)
            for rname in _highlight_inner:
                if rname == 'subdomains' or rname not in self._inner_regions:
                    continue
                r_lo, r_hi = self._inner_regions[rname]
                rcolor = _inner_colors[rname]
                ax.axvspan(r_lo[0], r_hi[0], alpha=0.25, color=rcolor, zorder=2)
            for bname in _highlight_boundary:
                if bname == 'subdomains' or bname not in self._boundary_regions:
                    continue
                reg = self._boundary_regions[bname]
                bcolor = _boundary_colors[bname]
                ax.axvline(reg['fixed_val'], color=bcolor, linewidth=2.5, zorder=4)

            if points is not None:
                _pts = np.asarray(points)
                ax.scatter(_pts[:, 0], np.zeros(len(_pts)),
                           s=8, color='tomato', alpha=0.5, zorder=3)

            legend_handles = []
            for rname in _highlight_inner:
                if rname != 'subdomains' and rname in self._inner_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_inner_colors[rname], label=f'inner:{rname}'))
            for bname in _highlight_boundary:
                if bname != 'subdomains' and bname in self._boundary_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_boundary_colors[bname], label=f'boundary:{bname}'))

            if boundary is not None:
                for bc in self.boundary_conditions:
                    if not _bc_in_phase(bc, phase):
                        continue
                    color = bc_color[id(bc)]
                    label = getattr(bc, 'name', None) or type(bc).__name__
                    bc_boundary = getattr(bc, 'boundary', None)
                    if isinstance(bc, PointsetBC):
                        pts_in = np.asarray(bc.inputs)
                        ax.scatter(pts_in[:, 0], np.zeros(len(pts_in)),
                                   s=30, color=color, zorder=5, marker='x')
                        legend_handles.append(mpatches.Patch(color=color, label=label))
                        continue
                    if isinstance(bc, CubicPeriodicBC):
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
                          loc='upper right', framealpha=0.8)

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

            if 'subdomains' in _highlight_inner and self.grid_positions is not None:
                lo_bounds, up_bounds = self.get_subdomain_bounds()
                ext_lo = lo_bounds
                ext_up = up_bounds
                for i in range(self.n_subdomains):
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (ext_lo[i, dx], ext_lo[i, dy]),
                        ext_up[i, dx] - ext_lo[i, dx],
                        ext_up[i, dy] - ext_lo[i, dy],
                        boxstyle='square,pad=0',
                        edgecolor='tab:orange', facecolor='tab:orange',
                        linewidth=0.6, alpha=0.12, zorder=2
                    ))
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (lo_bounds[i, dx], lo_bounds[i, dy]),
                        up_bounds[i, dx] - lo_bounds[i, dx],
                        up_bounds[i, dy] - lo_bounds[i, dy],
                        boxstyle='square,pad=0',
                        edgecolor='grey', facecolor='none',
                        linewidth=0.8, linestyle='--', zorder=3
                    ))
            for rname in _highlight_inner:
                if rname == 'subdomains' or rname not in self._inner_regions:
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
                if bname == 'subdomains' or bname not in self._boundary_regions:
                    continue
                reg = self._boundary_regions[bname]
                bcolor = _boundary_colors[bname]
                _draw_face_2d(ax, reg['fixed_dim'], 0 if reg['fixed_val'] == reg['lo'][reg['fixed_dim']] else 1,
                              dx, dy, bcolor, linewidth=3.0)

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
                if rname != 'subdomains' and rname in self._inner_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_inner_colors[rname], label=f'inner:{rname}'))
            for bname in _highlight_boundary:
                if bname != 'subdomains' and bname in self._boundary_regions:
                    legend_handles.append(mpatches.Patch(
                        color=_boundary_colors[bname], label=f'boundary:{bname}'))

            if boundary is not None:
                for bc in self.boundary_conditions:
                    if not _bc_in_phase(bc, phase):
                        continue
                    color = bc_color[id(bc)]
                    label = getattr(bc, 'name', None) or type(bc).__name__
                    bc_boundary = getattr(bc, 'boundary', None)

                    if isinstance(bc, PointsetBC):
                        pts_in = np.asarray(bc.inputs)
                        ax.scatter(pts_in[:, dx], pts_in[:, dy],
                                   s=30, color=color, zorder=5, marker='x')
                        legend_handles.append(mpatches.Patch(color=color, label=label))
                        continue

                    if isinstance(bc, CubicPeriodicBC):
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
                          loc='upper right', framealpha=0.8)

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

class DomainMesh:
    """
    A spatial domain defined by a triangular mesh, with an
    optional time axis controlled by the ``time`` argument.

    The mesh provides vertex positions and face connectivity.  Interior
    sampling uses exact barycentric sampling, which works correctly for both
    **2-D flat meshes** (vertices in R²) and **3-D surface meshes** (vertices
    in R³, triangulated surface embedded in 3-D space).

    **Time modes** — selected automatically from the ``time`` argument:

    * ``time=None`` *(default)* — **stationary** domain.  ``n_dims`` equals
      ``spatial_dims``.
    * ``time=(t_min, t_max)`` — **continuous** time interval.  ``n_dims``
      equals ``spatial_dims + 1``.  ``ProblemWeak`` samples ``n_time_points``
      random levels per epoch.  Time-windows in BCs must be 2-element
      intervals.
    * ``time=array_or_list`` (≥ 2 values) — **discrete** time steps.
      ``domain.dt`` and ``domain.n_steps`` are set automatically.
      ``ProblemWeak`` uses BPTT rollout.  BC ``time_window`` defaults to all
      time steps; pass ``[t0]`` for an initial condition.

    Args:
        mesh: A mesh object with ``.vertices`` and ``.faces`` attributes
              (``trimesh.Trimesh``, ``pymesh.Mesh``, ``meshio`` mesh, or a
              ``(vertices, faces)`` tuple).
        time: Time specification — ``None``, a 2-tuple ``(t_min, t_max)``, or
              a 1-D array/list of time-point values.
        t_sampling_method: How to place time quadrature points (continuous
              mode only).  One of ``"uniform"`` *(default)*, ``"midpoint"``,
              ``"latin_hypercube"``, ``"sobol"``, ``"halton"``, or a callable
              ``(n, rng) -> ndarray`` in ``[0, 1]``.
        n_time_points: Default number of time levels per epoch (continuous
              mode only, default 10).

    Examples::

        # Stationary
        domain = DomainMesh(mesh)

        # Continuous time — BCs are MeshNodeBC objects appended to
        # domain.boundary_conditions directly or via the Problem class
        domain = DomainMesh(mesh, time=(0.0, 1.0))

        # Discrete time steps
        domain = DomainMesh(mesh, time=np.linspace(0, 1, 21))
    """

    @staticmethod
    def _extract_vertices_faces(mesh):
        """
        Extract vertices and triangular faces from heterogeneous mesh objects.

        Supported formats
        -----------------
        * pymesh / trimesh style:  ``mesh.vertices``, ``mesh.faces``
        * meshio style (pygmsh):   ``mesh.points``,   ``mesh.cells_dict["triangle"]``
        * plain tuple/dict:        ``(vertices, faces)``
        """
        # --- tuple / list shortcut ----------------------------------------
        if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
            return np.asarray(mesh[0], dtype=np.float64), np.asarray(mesh[1], dtype=np.int64)

        # --- meshio (pygmsh output) ----------------------------------------
        if hasattr(mesh, "points") and hasattr(mesh, "cells_dict"):
            verts = np.asarray(mesh.points, dtype=np.float64)
            # Drop the z-column when it is all zeros (2-D mesh embedded in R³)
            if verts.shape[1] == 3 and np.allclose(verts[:, 2], 0.0):
                verts = verts[:, :2]
            faces_raw = None
            for key in ("triangle", "triangle6"):
                if key in mesh.cells_dict:
                    faces_raw = mesh.cells_dict[key]
                    break
            if faces_raw is None:
                raise ValueError(
                    "meshio mesh has no 'triangle' cell block. "
                    "Make sure you requested a triangular surface mesh."
                )
            return verts, np.asarray(faces_raw, dtype=np.int64)

        # --- pymesh / trimesh style ----------------------------------------
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            return (np.asarray(mesh.vertices, dtype=np.float64),
                    np.asarray(mesh.faces,    dtype=np.int64))

        raise TypeError(
            f"Unrecognised mesh type {type(mesh)}.  "
            "Provide a pymesh, trimesh, meshio, or (vertices, faces) object."
        )

    def __init__(self, mesh, time=None, t_sampling_method="uniform",
                 n_time_points=10):
        vertices, faces = self._extract_vertices_faces(mesh)
        self._vertices = vertices
        self._spatial_dims = vertices.shape[1]
        self._faces = faces

        # trimesh needed.  For 3D meshes we still try trimesh as a fallback
        # for the _trimesh attribute (used by pyvista plotting).
        self._trimesh = None
        if self._spatial_dims == 3:
            try:
                import trimesh as _trimesh_mod
                if isinstance(mesh, _trimesh_mod.Trimesh):
                    self._trimesh = mesh
                else:
                    self._trimesh = _trimesh_mod.Trimesh(
                        vertices=vertices, faces=self._faces, process=False
                    )
            except ImportError:
                pass

        # Precompute triangle areas for weighted sampling.
        # Works for both 2-D flat meshes (signed-area formula) and 3-D surface
        # meshes embedded in R³ (cross-product magnitude / 2).
        A = vertices[self._faces[:, 0]]
        B = vertices[self._faces[:, 1]]
        C = vertices[self._faces[:, 2]]
        if self._spatial_dims == 2:
            cross = (B - A)[:, 0] * (C - A)[:, 1] - (C - A)[:, 0] * (B - A)[:, 1]
            self._tri_areas = 0.5 * np.abs(cross)          # (n_faces,)
        else:
            # 3-D surface: area = ||(B-A) × (C-A)|| / 2
            cross3 = np.cross(B - A, C - A)                # (n_faces, 3)
            self._tri_areas = 0.5 * np.linalg.norm(cross3, axis=1)  # (n_faces,)
        self._tri_probs = self._tri_areas / self._tri_areas.sum()

        sp_min = vertices.min(axis=0)
        sp_max = vertices.max(axis=0)

        self._t_sampling_method = t_sampling_method
        self.n_time_points = n_time_points

        # ── Interpret the `time` argument ────────────────────────────────
        # None                  → stationary
        # tuple/list of length 2 with numeric scalars → continuous interval
        # array/list with ≥ 2 values (or len>2)       → discrete time steps
        self._time_mode   = None   # 'stationary' | 'continuous' | 'discrete'
        self._time_points = None
        self.dt           = None
        self.n_steps      = None

        if time is None:
            self._time_mode = 'stationary'
            self._t_min = None
            self._t_max = None
            self.t_interval = None
            self.xmin = sp_min
            self.xmax = sp_max
        else:
            time_arr = np.asarray(time, dtype=float).ravel()
            if len(time_arr) < 2:
                raise ValueError(
                    "DomainMesh: 'time' must be None, a 2-tuple (t_min, t_max), "
                    "or an array/list of ≥ 2 time-point values."
                )
            # Distinguish: exactly 2 values that are meant as (t_min, t_max)
            # vs. a discrete array.  A plain 2-tuple always means continuous;
            # a list/array with exactly 2 elements could be either — we treat
            # a Python tuple of length 2 as continuous, everything else as
            # decides by whether the caller passed a sequence longer than 2.
            _is_tuple2 = isinstance(time, tuple) and len(time) == 2
            if _is_tuple2 or len(time_arr) == 2:
                # Check if caller actually passed a plain (t_min, t_max)
                # A tuple → continuous; list/array of length 2 → also continuous
                # (to get discrete with 2 steps, pass np.array([0, 0.5, 1]))
                if _is_tuple2 or (not isinstance(time, np.ndarray) and len(list(time)) == 2):
                    self._time_mode = 'continuous'
                    self._t_min = float(time_arr[0])
                    self._t_max = float(time_arr[1])
                else:
                    self._time_mode = 'discrete'
            else:
                self._time_mode = 'discrete'

            if self._time_mode == 'discrete':
                tp = time_arr
                self._time_points = list(tp)
                dts = tp[1:] - tp[:-1]
                if np.allclose(dts, dts[0]):
                    self.dt = float(dts[0])
                else:
                    self.dt = dts
                self.n_steps = len(tp) - 1
                self._t_min  = float(tp[0])
                self._t_max  = float(tp[-1])
                self._t_sampling_method = 'midpoint'

            self.t_interval = [self._t_min, self._t_max]
            self.xmin = np.append(sp_min, self._t_min)
            self.xmax = np.append(sp_max, self._t_max)

        self.n_dims = len(self.xmin)
        self.boundary_conditions: List = []

        # Precompute all unique mesh edges: (n_edges, 2) vertex index pairs.
        # Used by _resolve_select to let the user address BCs by edge index.
        _seen_edges: dict = {}
        _edges_list: list = []
        for _face in self._faces:
            for _j in range(3):
                _v0, _v1 = int(_face[_j]), int(_face[(_j + 1) % 3])
                _key = (min(_v0, _v1), max(_v0, _v1))
                if _key not in _seen_edges:
                    _seen_edges[_key] = len(_edges_list)
                    _edges_list.append([_v0, _v1])
        self._all_edges = (np.array(_edges_list, dtype=np.int64)
                           if _edges_list else np.empty((0, 2), dtype=np.int64))
        # canonical (min_v, max_v) -> edge_index  (used by helper methods)
        self._edge_lookup: dict = _seen_edges

        # Precompute boundary / interior node masks from edge counts.
        # A mesh edge shared by exactly one face is a boundary edge.
        _edge_face_count: dict = {}
        for _face in self._faces:
            for _j in range(3):
                _v0, _v1 = int(_face[_j]), int(_face[(_j + 1) % 3])
                _key = (min(_v0, _v1), max(_v0, _v1))
                _edge_face_count[_key] = _edge_face_count.get(_key, 0) + 1
        _bnd_mask = np.zeros(len(self._vertices), dtype=bool)
        for (_v0, _v1), _cnt in _edge_face_count.items():
            if _cnt == 1:
                _bnd_mask[_v0] = True
                _bnd_mask[_v1] = True
        self._boundary_node_mask: np.ndarray = _bnd_mask

        # Precompute boundary edge array and per-edge sampling weights.
        # A mesh edge is a boundary edge when both endpoints are boundary nodes.
        # This works for both 2-D flat meshes and 3-D surface meshes — in the
        # 3-D case "boundary" means edges on the open rim of the surface.
        if len(self._all_edges) > 0:
            _be_mask = np.array(
                [bool(_bnd_mask[int(e[0])]) and bool(_bnd_mask[int(e[1])])
                 for e in self._all_edges], dtype=bool)
            self._bnd_edges = self._all_edges[_be_mask]
        else:
            self._bnd_edges = np.empty((0, 2), dtype=np.int64)
        if len(self._bnd_edges) > 0:
            _bv0 = vertices[self._bnd_edges[:, 0]]
            _bv1 = vertices[self._bnd_edges[:, 1]]
            self._bnd_edge_lengths = np.linalg.norm(_bv1 - _bv0, axis=1)
            self._bnd_edge_probs   = self._bnd_edge_lengths / self._bnd_edge_lengths.sum()
        else:
            self._bnd_edge_lengths = None
            self._bnd_edge_probs   = None

        # Named sampling regions (fully independent from the BC system).
        self._inner_regions:    dict = {}
        self._boundary_regions: dict = {}

    # ------------------------------------------------------------------ #
    #  Region registration                                                #
    # ------------------------------------------------------------------ #

    def _resolve_for_region(self, select) -> np.ndarray:
        """Resolve a region selector to node indices.

        Supports all forms accepted by :meth:`_resolve_node_select`, plus a
        **bounding-box** shortcut: a list of ``(lo, hi)`` tuples, one per
        spatial dimension.

        Examples::

            # Callable selector
            domain._resolve_for_region(lambda v: v[:, 0] < 0.5)

            # Bounding box [(x_lo, x_hi), (y_lo, y_hi)]
            domain._resolve_for_region([(0.0, 0.5), (0.0, 1.0)])
        """
        # Bounding-box shortcut: list/array of 2-element numeric tuples
        if (isinstance(select, (list, np.ndarray)) and len(select) > 0):
            first = select[0]
            if (isinstance(first, (tuple, list)) and len(first) == 2
                    and all(isinstance(x, (int, float, np.floating)) for x in first)):
                v = self._vertices
                mask = np.ones(len(v), dtype=bool)
                for dim_i, (lo, hi) in enumerate(select):
                    if dim_i >= v.shape[1]:
                        break
                    mask &= (v[:, dim_i] >= lo) & (v[:, dim_i] <= hi)
                return np.where(mask)[0].astype(np.intp)
        return self._resolve_node_select(select)

    def _resolve_time_region(self, time):
        """Parse the ``time`` argument used in :meth:`add_inner` /
        :meth:`add_boundary`.

        * ``None``        → ``(t_min, t_max)`` (full domain range, or both None
                            for a stationary domain).
        * ``(t_lo, t_hi)``  → explicit sub-interval (requires time axis).

        Raises:
            ValueError: On a stationary domain when *time* is not None, or
                when the requested window lies outside the domain range.
        """
        if time is None:
            return (self._t_min, self._t_max)
        if self._t_min is None:
            raise ValueError(
                "Cannot restrict a region's time window on a stationary "
                "domain (no time axis).  Pass time=None or add a time axis.")
        _t = np.asarray(time, dtype=float).ravel()
        if len(_t) != 2:
            raise ValueError(
                f"'time' for a region must be None or a 2-element (t_lo, t_hi) "
                f"tuple; got {time!r}.")
        t_lo, t_hi = float(_t[0]), float(_t[1])
        if t_lo > t_hi + 1e-12:
            raise ValueError(
                f"Region time window must have t_lo ≤ t_hi; got ({t_lo}, {t_hi}).")
        if t_lo < self._t_min - 1e-10 or t_hi > self._t_max + 1e-10:
            raise ValueError(
                f"Region time window ({t_lo}, {t_hi}) is outside the domain "
                f"time range [{self._t_min}, {self._t_max}].")
        return (t_lo, t_hi)

    def add_inner(self, select, name: str, time=None, strict: bool = True) -> None:
        """Register a named **interior** sampling region.

        The region is defined by *select*, which identifies the mesh vertices
        that belong to it.  By default (*strict=True*) only triangles whose
        **all three** vertices are inside the selected set are included;
        with *strict=False* a triangle is included if **any** of its vertices
        is inside, which avoids gaps at region boundaries.

        Args:
            select: One of:

                * **Callable** ``(v: ndarray) → bool_mask`` — called with all
                  vertex positions ``(n_verts, spatial_dims)``.
                  Example: ``lambda v: v[:, 0] < 0.5``.
                * **Boolean array** of shape ``(n_verts,)`` — direct mask.
                * **1-D integer array** — explicit vertex indices.
                * **List of ``(lo, hi)`` tuples** — axis-aligned bounding box,
                  one tuple per spatial dimension.
                  Example: ``[(0.0, 0.5), (0.0, 1.0)]``.

            name: String label used in :meth:`sample_interior` as the
                ``region=`` key.
            time: Optional time restriction:

                * ``None`` — full domain time range (default).
                * ``(t_lo, t_hi)`` — restrict to this sub-interval.

            strict: If ``True`` (default) a triangle is included only when
                **all three** of its vertices are in the selected set.  If
                ``False`` a triangle is included when **at least one** vertex
                is in the selected set.  Use ``strict=False`` when adjacent
                regions share a boundary threshold (e.g. ``>= 1.0`` and
                ``<= 1.0``) so that straddling triangles are never dropped.

        Raises:
            ValueError: If no triangle matches the criterion.

        Example::

            domain.add_inner(lambda v: v[:, 0] < 0.5, name='left_half')
            domain.add_inner([(0.3, 0.7), (0.3, 0.7)], name='centre')
            domain.add_inner(lambda v: v[:, 0] < 0.5, name='left_t0',
                             time=(0.0, 0.5))
            # Non-strict: no gaps at shared threshold y = 1.0
            domain.add_inner(lambda v: v[:, 1] >= 1.0, name='upper', strict=False)
            domain.add_inner(lambda v: v[:, 1] <= 1.0, name='lower', strict=False)
        """
        node_idx = self._resolve_for_region(select)
        if len(node_idx) == 0:
            raise ValueError(f"Region '{name}': selector returned no vertices.")
        node_set = set(node_idx.tolist())
        if strict:
            face_mask = np.array(
                [int(f[0]) in node_set and int(f[1]) in node_set and
                 int(f[2]) in node_set
                 for f in self._faces], dtype=bool)
        else:
            face_mask = np.array(
                [int(f[0]) in node_set or int(f[1]) in node_set or
                 int(f[2]) in node_set
                 for f in self._faces], dtype=bool)
        face_idx = np.where(face_mask)[0]
        if len(face_idx) == 0:
            raise ValueError(
                f"Region '{name}': no triangle has "
                f"{'all three' if strict else 'any'} vertices inside "
                "the selected node set.  Broaden the selector or the bounding box.")
        # Per-region triangle sampling probabilities (2D and 3D surface).
        # _tri_areas is always populated in __init__ for both dimensionalities.
        reg_areas = self._tri_areas[face_idx]
        reg_probs = reg_areas / reg_areas.sum()
        t_lo, t_hi = self._resolve_time_region(time)
        self._inner_regions[name] = {
            'node_indices': node_idx,
            'face_indices': face_idx,
            'tri_probs':    reg_probs,
            't_lo':         t_lo,
            't_hi':         t_hi,
        }

    def add_boundary(self, select, name: str, time=None, strict: bool = True) -> None:
        """Register a named **boundary** sampling region.

        The region is defined by *select*, which identifies the mesh vertices
        on the boundary of interest.  By default (*strict=True*) only boundary
        **edges** whose **both** endpoints are in the selected set are included;
        with *strict=False* an edge is included if **either** endpoint is in the
        selected set.

        Args:
            select: Same forms as in :meth:`add_inner`.
            name: String label used in :meth:`sample_boundary` as the
                ``region=`` key.
            time: Optional time restriction — ``None`` or ``(t_lo, t_hi)``.
            strict: If ``True`` (default) both endpoints of a boundary edge
                must be in the selected set.  If ``False`` at least one
                endpoint must be in the selected set.  Useful when region
                boundaries are shared.

        Raises:
            ValueError: If no boundary edge is found in the selected set.

        Example::

            # Entire x = 0 face
            domain.add_boundary(lambda v: v[:, 0] < 1e-6, name='x_left')
            # y = 0 face, restricted to x ∈ [0.2, 0.8]
            domain.add_boundary(
                lambda v: (np.abs(v[:, 1]) < 1e-6) & (v[:, 0] > 0.2) & (v[:, 0] < 0.8),
                name='y_bottom_centre')
            # Bounding-box shorthand for the same as above
            domain.add_boundary([(0.0, 0.01), (0.2, 0.8)], name='y_bottom_centre')
        """
        node_idx = self._resolve_for_region(select)
        if len(node_idx) == 0:
            raise ValueError(f"Region '{name}': selector returned no vertices.")
        node_set   = set(node_idx.tolist())
        bnd_nodes  = self._boundary_node_mask
        if strict:
            edge_mask = np.array(
                [bool(bnd_nodes[int(e[0])]) and bool(bnd_nodes[int(e[1])])
                 and int(e[0]) in node_set and int(e[1]) in node_set
                 for e in self._all_edges], dtype=bool)
        else:
            edge_mask = np.array(
                [bool(bnd_nodes[int(e[0])]) and bool(bnd_nodes[int(e[1])])
                 and (int(e[0]) in node_set or int(e[1]) in node_set)
                 for e in self._all_edges], dtype=bool)
        edge_idx = np.where(edge_mask)[0]
        if len(edge_idx) == 0:
            raise ValueError(
                f"Region '{name}': no boundary edge found in the selected node "
                "set.  Make sure the selector covers actual mesh boundary vertices.")
        edges   = self._all_edges[edge_idx]
        v0      = self._vertices[edges[:, 0]]
        v1      = self._vertices[edges[:, 1]]
        lengths = np.linalg.norm(v1 - v0, axis=1)
        t_lo, t_hi = self._resolve_time_region(time)
        self._boundary_regions[name] = {
            'node_indices': node_idx,
            'edge_indices': edge_idx,
            'edges':        edges,
            'edge_lengths': lengths,
            'edge_probs':   lengths / lengths.sum(),
            't_lo':         t_lo,
            't_hi':         t_hi,
        }

    # ------------------------------------------------------------------ #
    #  Private sampling helpers                                           #
    # ------------------------------------------------------------------ #

    def _sample_region_interior_all(self, n_points: int, rng) -> np.ndarray:
        """Sample *n_points* from the entire mesh interior."""
        pts_sp = self._sample_interior_spatial(n_points, rng)
        if self._t_min is not None:
            t = rng.uniform(self._t_min, self._t_max, (n_points, 1))
            return np.hstack([pts_sp, t])
        return pts_sp

    def _sample_region_interior(self, reg_info: dict,
                                 n_points: int, rng) -> np.ndarray:
        """Barycentric sampling over a registered interior region (2D/3D surface)."""
        face_idx  = reg_info['face_indices']
        tri_probs = reg_info['tri_probs']
        f         = self._faces[face_idx]
        v         = self._vertices
        tri_sel   = rng.choice(len(f), n_points, p=tri_probs)
        A = v[f[tri_sel, 0]]
        B = v[f[tri_sel, 1]]
        C = v[f[tri_sel, 2]]
        r1 = rng.uniform(0.0, 1.0, n_points)
        r2 = rng.uniform(0.0, 1.0, n_points)
        swap       = r1 + r2 > 1.0
        r1[swap]   = 1.0 - r1[swap]
        r2[swap]   = 1.0 - r2[swap]
        pts_sp = r1[:, None] * A + r2[:, None] * B + (1 - r1 - r2)[:, None] * C
        t_lo = reg_info.get('t_lo')
        t_hi = reg_info.get('t_hi')
        if t_lo is not None and t_hi is not None:
            return np.hstack([pts_sp, rng.uniform(t_lo, t_hi, (n_points, 1))])
        return pts_sp

    def _sample_multi_interior(self, regions: list, n_points: int,
                                size, rng) -> np.ndarray:
        """Sample *n_points* from a list of named interior regions."""
        n = len(regions)
        if size == 'equal':
            weights = np.ones(n, dtype=float) / n
        elif size == 'area':
            areas = np.array([
                float(self._tri_areas[self._inner_regions[nm]['face_indices']].sum())
                for nm in regions])
            weights = areas / areas.sum()
        else:
            weights = np.asarray(size, dtype=float)
            if len(weights) != n:
                raise ValueError(
                    f"'size' has {len(weights)} elements but region list has {n}.")
            weights = weights / weights.sum()
        counts        = (weights * n_points).astype(int)
        counts[-1]    = n_points - counts[:-1].sum()
        parts = []
        for nm, cnt in zip(regions, counts):
            if nm not in self._inner_regions:
                raise KeyError(
                    f"Unknown interior region '{nm}'. "
                    f"Registered: {list(self._inner_regions.keys())}")
            if cnt > 0:
                parts.append(self._sample_region_interior(
                    self._inner_regions[nm], cnt, rng))
        return np.vstack(parts) if parts else np.empty((0, self.n_dims))

    def _sample_boundary_all_edges(self, n_points: int, rng) -> np.ndarray:
        """Sample *n_points* uniformly along all mesh boundary edges (2D)."""
        if self._bnd_edge_probs is None or len(self._bnd_edges) == 0:
            return self._sample_region_interior_all(n_points, rng)
        idx      = rng.choice(len(self._bnd_edges), n_points,
                               p=self._bnd_edge_probs)
        t_param  = rng.uniform(0.0, 1.0, (n_points, 1))
        v0       = self._vertices[self._bnd_edges[idx, 0]]
        v1       = self._vertices[self._bnd_edges[idx, 1]]
        pts_sp   = v0 + t_param * (v1 - v0)
        if self._t_min is not None:
            return np.hstack([pts_sp, rng.uniform(self._t_min, self._t_max,
                                                   (n_points, 1))])
        return pts_sp

    def _sample_boundary_region(self, reg_info: dict,
                                 n_points: int, rng) -> np.ndarray:
        """Sample *n_points* from a registered boundary region."""
        edges  = reg_info['edges']
        probs  = reg_info['edge_probs']
        idx    = rng.choice(len(edges), n_points, p=probs)
        t_par  = rng.uniform(0.0, 1.0, (n_points, 1))
        v0     = self._vertices[edges[idx, 0]]
        v1     = self._vertices[edges[idx, 1]]
        pts_sp = v0 + t_par * (v1 - v0)
        t_lo = reg_info.get('t_lo')
        t_hi = reg_info.get('t_hi')
        if t_lo is not None and t_hi is not None:
            return np.hstack([pts_sp, rng.uniform(t_lo, t_hi, (n_points, 1))])
        return pts_sp

    def _sample_multi_boundary(self, regions: list, n_points: int,
                                size, rng) -> np.ndarray:
        """Sample *n_points* from a list of named boundary regions."""
        n = len(regions)
        if size == 'equal':
            weights = np.ones(n, dtype=float) / n
        elif size == 'length':
            lengths = np.array([
                float(self._boundary_regions[nm]['edge_lengths'].sum())
                for nm in regions])
            weights = lengths / lengths.sum()
        else:
            weights = np.asarray(size, dtype=float)
            if len(weights) != n:
                raise ValueError(
                    f"'size' has {len(weights)} elements but region list has {n}.")
            weights = weights / weights.sum()
        counts     = (weights * n_points).astype(int)
        counts[-1] = n_points - counts[:-1].sum()
        parts = []
        for nm, cnt in zip(regions, counts):
            if nm not in self._boundary_regions:
                raise KeyError(
                    f"Unknown boundary region '{nm}'. "
                    f"Registered: {list(self._boundary_regions.keys())}")
            if cnt > 0:
                parts.append(self._sample_boundary_region(
                    self._boundary_regions[nm], cnt, rng))
        return np.vstack(parts) if parts else np.empty((0, self.n_dims))

    def _sample_interior_spatial(self, n_points: int, rng) -> np.ndarray:
        """Sample *n_points* spatial points inside the mesh.

        Uses exact barycentric sampling for both 2-D flat meshes and 3-D
        surface meshes.  Points lie exactly on the triangulated surface in
        either case.
        """
        # ---- 2-D flat mesh OR 3-D surface mesh: barycentric sampling ------
        # This is exact and works regardless of spatial dimensionality because
        # we parameterise each triangle with two barycentric coordinates and
        # interpolate vertex positions in R^d.
        v = self._vertices
        f = self._faces
        tri_idx = rng.choice(len(f), n_points, p=self._tri_probs)
        A = v[f[tri_idx, 0]]
        B = v[f[tri_idx, 1]]
        C = v[f[tri_idx, 2]]
        r1 = rng.uniform(0.0, 1.0, n_points)
        r2 = rng.uniform(0.0, 1.0, n_points)
        mask = r1 + r2 > 1.0
        r1[mask] = 1.0 - r1[mask]
        r2[mask] = 1.0 - r2[mask]
        r3 = 1.0 - r1 - r2
        return r1[:, None] * A + r2[:, None] * B + r3[:, None] * C

    def _resolve_node_select(self, select) -> np.ndarray:
        """
        Return a 1-D integer array of **node indices** matching *select*.

        *select* can be one of:

        - **Callable** — called with ``self._vertices`` ``(n_verts, spatial_dims)``;
          must return a boolean mask of shape ``(n_verts,)``.
          Example: ``lambda v: v[:, 0] < 1e-6`` (x ≈ 0 plane).

        - **1-D integer array** — explicit node index list.
          Example: ``np.arange(100)``.

        - **Boolean array** of shape ``(n_verts,)`` — treated as a mask.

        - **2-D ``(n, 2)`` integer array** — interpreted as edge vertex-pair
          table (e.g. ``mesh.cells_dict["line"]``); the unique node indices
          contained in those edges are returned (backward-compatible usage
          for Neumann BCs sourced directly from mesh boundary cells).
        """
        if callable(select):
            mask = np.asarray(select(self._vertices), dtype=bool)
            return np.where(mask)[0].astype(np.intp)
        arr = np.asarray(select)
        if arr.dtype == bool:
            return np.where(arr)[0].astype(np.intp)
        if arr.ndim == 1:
            return arr.astype(np.intp)
        if arr.ndim == 2 and arr.shape[1] == 2:
            # Edge-pair table — extract unique node indices
            return np.unique(arr.ravel()).astype(np.intp)
        raise ValueError(
            "select must be a callable, a boolean mask, a 1-D integer array, "
            f"or a (n, 2) edge-pair array.  Got array with shape {arr.shape}."
        )

    # keep _resolve_select as an alias for internal / backward-compat callers
    def _resolve_select(self, select) -> np.ndarray:
        """Legacy alias — converts node selector to edge indices.

        Calls :meth:`_resolve_node_select` then returns indices of edges whose
        **both** endpoints are in the selected node set.
        """
        node_idx = self._resolve_node_select(select)
        return self.node_indices_to_edge_indices(node_idx)

    def edge_pairs_to_indices(self, edge_pairs: np.ndarray) -> np.ndarray:
        """
        Convert an array of ``(v0, v1)`` vertex-index pairs to edge indices.

        Looks up each pair in ``self._edge_lookup`` (the canonical
        ``(min_v, max_v) -> edge_index`` dict built at construction time).
        This is the most direct way to go from the ``"line"`` cells that Gmsh /
        meshio stores for physical boundaries to the edge arrays expected by
        :class:`~pinns.boundary.MeshNodeBC` (``edges``, ``edge_lengths``,
        ``edge_normals`` fields).

        Args:
            edge_pairs: ``(n, 2)`` integer array of vertex index pairs,
                        e.g. ``mesh.cells_dict["line"]`` for a physical group.

        Returns:
            1-D integer array of indices into ``self._all_edges``.
            Pairs not found in the mesh are silently skipped.

        Example::

            domain = DomainMesh(mesh)
            line_cells = mesh.cells_dict["line"]  # all boundary segments
            eidx = domain.edge_pairs_to_indices(line_cells)
            edges = domain._all_edges[eidx]
            # → pass edges to MeshNodeBC(edges=edges, ...)
        """
        indices = []
        for v0, v1 in edge_pairs:
            key = (min(int(v0), int(v1)), max(int(v0), int(v1)))
            idx = self._edge_lookup.get(key)
            if idx is not None:
                indices.append(idx)
        return np.array(indices, dtype=np.intp)

    def node_indices_to_edge_indices(self, node_indices: np.ndarray) -> np.ndarray:
        """
        Convert vertex indices to edge indices.

        Returns the indices (into ``self._all_edges``) of every mesh edge whose
        **both** endpoints are in *node_indices*.

        Args:
            node_indices: 1-D integer array of vertex indices (e.g. from a
                          physical-group node helper).

        Returns:
            1-D integer array of indices into ``self._all_edges``.
        """
        node_set = set(node_indices.tolist())
        v0_ok = np.array([int(e[0]) in node_set for e in self._all_edges])
        v1_ok = np.array([int(e[1]) in node_set for e in self._all_edges])
        return np.where(v0_ok & v1_ok)[0].astype(np.intp)

    def _infer_edge_outward_normals(self, edges: np.ndarray) -> np.ndarray:
        """
        Compute per-edge outward unit normals for a 2D boundary.

        For each edge the tangent is ``v1 − v0``; rotating 90° CCW gives a
        candidate normal ``(−dy, dx)`` that is then flipped to point away from
        the mesh centroid.

        Args:
            edges: ``(n_edges, 2)`` vertex index pairs (into ``self._vertices``).

        Returns:
            ``(n_edges, 2)`` outward unit normals, one per edge.
        """
        v0 = self._vertices[edges[:, 0]]   # (n_edges, 2)
        v1 = self._vertices[edges[:, 1]]
        tangents = v1 - v0                 # (n_edges, 2)

        # Rotate 90° CCW: (tx, ty) → (−ty, tx)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        norms   = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.where(norms > 0, norms, 1.0)

        # Orient away from mesh centroid
        centroid     = self._vertices.mean(axis=0)
        edge_centers = 0.5 * (v0 + v1)
        outward      = edge_centers - centroid
        flip         = (normals * outward).sum(axis=1) < 0
        normals[flip] *= -1
        return normals

    # ------------------------------------------------------------------ #
    #  Temporal-overlap validation                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tw_overlaps(tw_a, tw_b, t_min: float, t_max: float,
                     tol: float = 1e-10) -> bool:
        """
        Return ``True`` if two time windows share any common time.

        Rules
        -----
        * ``None`` stands for the full domain ``[t_min, t_max]``.
        * A 2-element list/tuple ``[a, b]`` is a closed interval.
        * A list with > 2 elements is a discrete set; each value is compared
          against the other window's discrete set or interval.
        """
        def to_interval(tw):
            """Reduce any time window to (lo, hi); discrete → (min, max)."""
            if tw is None:
                return (t_min, t_max)
            pts = [float(v) for v in tw]
            return (min(pts), max(pts))

        # Both discrete: exact-value intersection
        def is_discrete(tw):
            return tw is not None and len(tw) > 2

        if is_discrete(tw_a) and is_discrete(tw_b):
            # Round to tol grid to handle float noise
            grid = max(tol, 1e-14)
            ka = {round(float(v) / grid) for v in tw_a}
            kb = {round(float(v) / grid) for v in tw_b}
            return bool(ka & kb)

        if is_discrete(tw_a) or is_discrete(tw_b):
            # One discrete, one interval: check whether any discrete value
            # falls inside the interval
            disc, cont = (tw_a, tw_b) if is_discrete(tw_a) else (tw_b, tw_a)
            lo, hi = to_interval(cont)
            return any(lo - tol <= float(v) <= hi + tol for v in disc)

        # Both intervals (or None)
        # Two intervals overlap only if they share more than a single point.
        # Touching at exactly one endpoint ([0,0.5] and [0.5,1]) is NOT
        # an overlap — it is zero-measure and is allowed.
        a_lo, a_hi = to_interval(tw_a)
        b_lo, b_hi = to_interval(tw_b)
        return max(a_lo, b_lo) < min(a_hi, b_hi) - tol

    def _check_bc_time_overlap(self, new_bc) -> None:
        """
        Raise ``ValueError`` if *new_bc* conflicts in time with any already-
        registered BC on the same (bc_type, component, nodes) combination.

        A conflict exists when both BCs have at least one shared **node** *and*
        their **time windows overlap** — meaning the same node would be
        assigned two different conditions at the same time.
        """
        from pinns.boundary import MeshNodeBC
        if not isinstance(new_bc, MeshNodeBC):
            return

        t_min = self._t_min if self._t_min is not None else 0.0
        t_max = self._t_max if self._t_max is not None else 1.0

        for existing in self.boundary_conditions:
            if not isinstance(existing, MeshNodeBC):
                continue
            if existing.bc_type != new_bc.bc_type:
                continue
            if existing.component != new_bc.component:
                continue

            # Check node-set intersection
            n_ex  = existing.node_indices
            n_new = new_bc.node_indices
            if n_ex is None or n_new is None:
                continue   # can't determine without index arrays
            common = np.intersect1d(n_ex, n_new)
            if len(common) == 0:
                continue

            # Check temporal overlap
            if self._tw_overlaps(
                existing.time_window, new_bc.time_window, t_min, t_max
            ):
                raise ValueError(
                    f"BC '{new_bc.name}' "
                    f"(type='{new_bc.bc_type}', component={new_bc.component}, "
                    f"time_window={new_bc.time_window}) overlaps in time with "
                    f"existing BC '{existing.name}' "
                    f"(time_window={existing.time_window}) on "
                    f"{len(common)} shared node(s).  "
                    "Adjust the time_window of one of the two conditions so "
                    "that they do not cover the same time simultaneously."
                )

    # ------------------------------------------------------------------ #
    #  Public sampling API (called by the trainer)                        #
    # ------------------------------------------------------------------ #

    def sample_interior(self, n_points: int, region=None, size='equal',
                        rng=None, **kwargs) -> np.ndarray:
        """Sample interior collocation points of shape ``(n_points, n_dims)``.

        Args:
            n_points: Number of points to return.
            region: Which region to sample from:

                * ``None`` or ``'all'`` — full mesh interior (default).
                * ``'name'`` — a named region registered with
                  :meth:`add_inner`.
                * ``['a', 'b', …]`` — multiple named regions; use *size*
                  to control the distribution of *n_points*.

            size: Distribution strategy when *region* is a list:

                * ``'equal'`` — equal split (default).
                * ``'area'`` — weight each region by its triangle area.
                * List of floats — explicit normalised weights.

            rng: NumPy random generator.

        Returns:
            ``(n_points, n_dims)`` array.
        """
        if rng is None:
            rng = np.random.default_rng()
        if region is None or region == 'all':
            return self._sample_region_interior_all(n_points, rng)
        if isinstance(region, str):
            if region not in self._inner_regions:
                raise KeyError(
                    f"Unknown interior region '{region}'. "
                    f"Registered: {list(self._inner_regions.keys())}")
            return self._sample_region_interior(self._inner_regions[region],
                                                n_points, rng)
        if isinstance(region, list):
            return self._sample_multi_interior(region, n_points, size, rng)
        raise ValueError(f"Invalid region: {region!r}")

    def sample_boundary(self, n_points: int, region=None, size='equal',
                        rng=None, **kwargs) -> np.ndarray:
        """Sample boundary collocation points of shape ``(n_points, n_dims)``.

        This is the **user-facing** method for obtaining spatial (and optional
        temporal) collocation points that lie on the mesh boundary.  It is
        region-aware and mirrors :meth:`sample_interior`.

        .. note::
            This method is **not** the same as :meth:`sample_boundary_bc`.
            Use this method when building custom training loops or exploring
            the domain.  The trainer uses :meth:`sample_boundary_bc` internally
            to draw points for specific registered BCs.

        Args:
            n_points: Number of points to return.
            region: Which boundary region to sample from:

                * ``None`` or ``'all'`` — all mesh boundary edges (default).
                * ``'name'`` — a named region registered with
                  :meth:`add_boundary`.
                * ``['a', 'b', …]`` — multiple named regions.

            size: Distribution strategy when *region* is a list:

                * ``'equal'`` — equal split (default).
                * ``'length'`` — weight each region by its total edge length.
                * List of floats — explicit normalised weights.

            rng: NumPy random generator.

        Returns:
            ``(n_points, n_dims)`` array.

        Note:
            The legacy positional call ``sample_boundary(n, dim, side, rng)``
            used by :class:`ProblemWeak` is still supported.
        """
        if rng is None:
            rng = np.random.default_rng()

        # ── Legacy positional API: sample_boundary(n, dim, side, rng) ───
        # Detected when region is an integer (old dim arg positional)
        # or when 'dim' is passed as a keyword.
        _dim  = kwargs.pop('dim',  None)
        _side = kwargs.pop('side', None)
        if isinstance(region, int) or _dim is not None:
            _d = region if isinstance(region, int) else _dim
            _s = size   if isinstance(region, int) else _side
            pts_sp = self._sample_interior_spatial(n_points, rng)
            if self._t_min is not None and _d == self._spatial_dims:
                t_val = self._t_min if _s == 0 else self._t_max
                return np.hstack([pts_sp, np.full((n_points, 1), t_val)])
            return self._sample_region_interior_all(n_points, rng)

        # ── New region-based API ─────────────────────────────────────────
        if region is None or region == 'all':
            return self._sample_boundary_all_edges(n_points, rng)
        if isinstance(region, str):
            if region not in self._boundary_regions:
                raise KeyError(
                    f"Unknown boundary region '{region}'. "
                    f"Registered: {list(self._boundary_regions.keys())}")
            return self._sample_boundary_region(self._boundary_regions[region],
                                                n_points, rng)
        if isinstance(region, list):
            return self._sample_multi_boundary(region, n_points, size, rng)
        raise ValueError(f"Invalid region: {region!r}")

    def sample_boundary_bc(self, bc, n_points: int, rng=None) -> np.ndarray:
        """Sample *n_points* from a specific :class:`~pinns.boundary.MeshNodeBC`.

        This is the **trainer-internal** method called by
        :class:`~pinns.backends.base_trainer.BaseTrainer` during the BC loss
        computation.  It differs from :meth:`sample_boundary` in two key ways:

        1. It accepts a ``MeshNodeBC`` object directly instead of a region name,
           so the trainer drives sampling from its registered BC list.
        2. It returns a ``(pts, edge_idx)`` *tuple* — the edge indices are
           needed by the trainer to look up per-edge outward normals for Neumann
           conditions (``bc.edge_normals[edge_idx]``).

        When the BC has precomputed edges, points are interpolated *along* the
        boundary edges (weighted by edge length) rather than snapping to mesh
        node positions, giving continuous coverage.  When no edges are stored
        (isolated-node or initial-condition BCs) the method falls back to
        sampling from ``bc.node_positions``.

        A time coordinate is appended according to ``bc.time_window``.

        Args:
            bc: A :class:`~pinns.boundary.MeshNodeBC` that has already been
                appended to ``self.boundary_conditions``.
            n_points: Number of collocation points to draw.
            rng: NumPy random generator.

        Returns:
            ``(pts, edge_idx)`` where *pts* is ``(n_points, n_dims)`` and
            *edge_idx* is a 1-D integer array of length *n_points* holding
            the sampled edge index for each point.
        """
        if rng is None:
            rng = np.random.default_rng()

        if bc.edges is not None:
            # ── Edge-based sampling: uniform along boundary edges ──────────
            probs    = bc.edge_lengths / bc.edge_lengths.sum()
            idx      = rng.choice(len(bc.edges), size=n_points, p=probs)
            t_param  = rng.uniform(0.0, 1.0, (n_points, 1))
            v0       = self._vertices[bc.edges[idx, 0]]
            v1       = self._vertices[bc.edges[idx, 1]]
            pts_sp   = v0 + t_param * (v1 - v0)
        else:
            # ── Fallback: sample from discrete node positions ──────────────
            n_nodes = len(bc.node_positions)
            idx     = rng.integers(0, n_nodes, n_points)
            pts_sp  = bc.node_positions[idx]

        tw = getattr(bc, 'time_window', None)
        if self._t_min is None or tw is None:
            return pts_sp, idx
        pts_tw = [float(v) for v in tw]
        if len(pts_tw) == 0:
            return pts_sp, idx
        if len(pts_tw) == 2 and abs(pts_tw[0] - pts_tw[1]) < 1e-12:
            # degenerate interval — single fixed time (e.g. initial condition)
            t = np.full((n_points, 1), pts_tw[0])
        elif len(pts_tw) == 2:
            # continuous interval [a, b] — uniform sampling within window
            t = rng.uniform(pts_tw[0], pts_tw[1], (n_points, 1))
        else:
            # discrete list of time values — pick randomly
            chosen = rng.choice(pts_tw, size=n_points)
            t = chosen.reshape(-1, 1)
        return np.hstack([pts_sp, t]), idx

    # ------------------------------------------------------------------ #
    #  Boundary-condition builders                                        #
    # ------------------------------------------------------------------ #

    def _default_time_window(self):
        """Return the default time_window for this domain's time mode."""
        if self._time_mode == 'continuous':
            return [self._t_min, self._t_max]
        if self._time_mode == 'discrete':
            return self._time_points
        return None

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bc_label(bc, color_idx: int) -> str:
        """Build a human-readable legend label for one MeshNodeBC."""
        label = bc.name or f"bc_{color_idx}"
        if bc.bc_type:
            label = f"[{bc.bc_type[0].upper()}] {label}"
        return label

    @staticmethod
    def _tw_label(tw) -> str:
        """Short string describing a time_window."""
        if tw is None:
            return "spatial"
        pts = [float(v) for v in tw]
        if len(pts) == 1 or (len(pts) == 2 and abs(pts[0] - pts[-1]) < 1e-12):
            return f"t = {pts[0]:.4g}"
        if len(pts) == 2:
            return f"t ∈ [{pts[0]:.4g}, {pts[1]:.4g}]"
        return f"{len(pts)} time steps"

    def _draw_bc_on_ax(self, ax, bc, color, label,
                       node_size: float, show_normals: bool = False) -> None:
        """Draw a single MeshNodeBC onto *ax*."""
        v = self._vertices
        if bc.node_indices is not None:
            pts = v[bc.node_indices]
        else:
            pts = bc.node_positions
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=node_size, c=color, zorder=3, label=label)
        if bc.bc_type == "neumann" and bc.edges is not None:
            for e in bc.edges:
                p0, p1 = v[e[0]], v[e[1]]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color=color, linewidth=2.5, zorder=4)

    def _draw_background(self, ax, show_mesh: bool) -> None:
        """Draw mesh triangulation edges."""
        import matplotlib.tri as mtri
        v, f = self._vertices, self._faces
        if show_mesh:
            tri = mtri.Triangulation(v[:, 0], v[:, 1], f)
            ax.triplot(tri, color="#cccccc", linewidth=0.5, zorder=1)

    @staticmethod
    def _is_jupyter() -> bool:
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    def plot(
        self,
        show_overlaps: bool = False,
        region=None,
        boundary=None,
        show_mesh: bool = True,
        node_size: float = 30.0,
        figsize=None,
        points=None,
        backend: str = 'auto',
    ):
        """
        Visualise the mesh with named regions and boundary conditions.

        Works for **2-D** and **3-D surface** meshes.
        For 3-D meshes, ``backend='pyvista'`` is recommended.

        Args:
            show_overlaps (bool): When ``True`` and the domain has a time axis,
                split into one subplot per time phase.  Default ``False``.
            region: Highlight named **interior** sampling regions:

                * ``None`` or ``'all'`` — highlight all registered inner
                  regions (default).
                * ``'name'`` — highlight that specific region.
                * ``['a', 'b', …]`` — highlight those regions.
                * ``'none'`` — don't highlight inner regions.

            boundary: Highlight named **boundary** sampling regions:

                * ``None`` — don't highlight boundary regions (default).
                * ``'all'`` — highlight all registered boundary regions.
                * ``'name'`` or ``['a', 'b', …]`` — specific regions.

                Registered boundary conditions are also drawn when this is
                not ``None``.
            show_mesh (bool): Draw triangulation edges (default ``True``).
            node_size (float): Scatter-point size for BC node markers (default 30).
            figsize (tuple | None): Figure size; auto-computed when ``None``.
            points (array-like | None): ``(N, D)`` array of collocation points
                to scatter on every panel.  Plotted in red.  Default ``None``.
            backend (str): ``'auto'``, ``'matplotlib'``, or ``'pyvista'``.

        Returns:
            Axes or array of Axes (matplotlib).

        Example::

            domain.plot()
            domain.plot(region='all', boundary='all')
            domain.plot(region='centre', boundary='x_left', points=pts)
        """
        _backend = backend
        if _backend == 'auto':
            _backend = 'pyvista' if self._is_jupyter() else 'matplotlib'

        if _backend == 'pyvista':
            return self._plot_pyvista(show_mesh=show_mesh,
                                       node_size=node_size)

        if self._spatial_dims not in (2, 3):
            raise NotImplementedError(
                "DomainMesh.plot() requires a 2-D or 3-D spatial mesh."
            )

        import matplotlib.pyplot as plt
        import matplotlib.collections as mcoll
        from pinns.boundary import MeshNodeBC

        mesh_bcs = [bc for bc in self.boundary_conditions
                    if isinstance(bc, MeshNodeBC)]

        # ── Resolve which inner/boundary regions to highlight ─────────────
        # 'all'  → draw the entire mesh / boundary as one unified colour
        # list   → draw each named region in a distinct colour
        # name   → draw only that region
        # None   → draw nothing
        _inner_all = (region == 'all')
        if region is None or region == 'none':
            _inner_highlight = []
        elif region == 'all':
            _inner_highlight = []   # handled separately as full-mesh fill
        elif isinstance(region, str):
            _inner_highlight = [region]
        elif isinstance(region, list):
            _inner_highlight = region
        else:
            _inner_highlight = []

        _bnd_all = (boundary == 'all')
        if boundary is None:
            _bnd_highlight = []
        elif boundary == 'all':
            _bnd_highlight = []     # handled separately as full-boundary draw
        elif isinstance(boundary, str):
            _bnd_highlight = [boundary]
        elif isinstance(boundary, list):
            _bnd_highlight = boundary
        else:
            _bnd_highlight = []

        # ── Build time phases from BC breakpoints ──────────────────────────
        has_time = self._t_min is not None
        breakpoints: list = []

        if has_time and show_overlaps:
            breakpoints.extend([self._t_min, self._t_max])
            for bc in mesh_bcs:
                tw = bc.time_window
                if tw is not None:
                    for bv in tw:
                        breakpoints.append(float(bv))

        tol = 1e-10
        unique_bps: list = []
        for bv in sorted(set(round(b, 12) for b in breakpoints)):
            if not unique_bps or abs(bv - unique_bps[-1]) > tol:
                unique_bps.append(bv)

        if show_overlaps and len(unique_bps) >= 2:
            phases = [(unique_bps[i], unique_bps[i + 1])
                      for i in range(len(unique_bps) - 1)]
        else:
            phases = [None]

        # ── Colour palettes ────────────────────────────────────────────────
        _cyc  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bc_color: dict = {id(bc): _cyc[i % len(_cyc)]
                          for i, bc in enumerate(mesh_bcs)}
        # separate palettes for inner / boundary region highlights
        _inner_palette   = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd',
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        _bnd_palette     = ['#ff7f0e', '#17becf', '#e377c2', '#bcbd22',
                             '#9467bd', '#8c564b', '#7f7f7f', '#1f77b4']
        _inner_color  = {nm: _inner_palette[i % len(_inner_palette)]
                         for i, nm in enumerate(_inner_highlight)}
        _bnd_color    = {nm: _bnd_palette[i % len(_bnd_palette)]
                         for i, nm in enumerate(_bnd_highlight)}

        # ── Helper: does a BC's time_window cover a phase? ─────────────────
        def _bc_in_phase(bc, phase) -> bool:
            if phase is None:
                return True
            tw = bc.time_window
            p_lo, p_hi = phase
            p_mid = 0.5 * (p_lo + p_hi)
            if tw is None:
                return True
            pts_tw = [float(vv) for vv in tw]
            if len(pts_tw) > 2:
                return any(p_lo - tol <= vv <= p_hi + tol for vv in pts_tw)
            a, b = min(pts_tw), max(pts_tw)
            if abs(a - b) < tol:
                return p_lo - tol <= a <= p_hi + tol
            return a - tol <= p_mid <= b + tol

        n_panels = len(phases)
        panel_w, panel_h = 6.0, 5.5
        if figsize is None:
            figsize = (panel_w * n_panels, panel_h)

        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = axes[0]   # (n_panels,)

        v, f = self._vertices, self._faces

        for ax, phase in zip(axes, phases):
            # Background: mesh triangulation
            self._draw_background(ax, show_mesh)

            # ── Highlight inner regions ────────────────────────────────────
            if _inner_all:
                # Fill all triangles as one unified region
                tris = v[f]
                poly = mcoll.PolyCollection(
                    tris, facecolor='#1f77b4', edgecolor='none', alpha=0.30,
                    zorder=2, label='interior')
                ax.add_collection(poly)
            else:
                for nm in _inner_highlight:
                    if nm not in self._inner_regions:
                        continue
                    reg    = self._inner_regions[nm]
                    fidx   = reg['face_indices']
                    color  = _inner_color[nm]
                    tris = v[f[fidx]]           # (n_sel, 3, 2)
                    poly = mcoll.PolyCollection(
                        tris, facecolor=color, edgecolor='none', alpha=0.35,
                        zorder=2, label=f'[I] {nm}')
                    ax.add_collection(poly)

            # ── Highlight boundary regions ─────────────────────────────────
            if _bnd_all:
                # Draw all boundary edges as one unified colour
                all_edges = self._bnd_edges          # (E, 2)
                segs = np.stack([v[all_edges[:, 0]], v[all_edges[:, 1]]], axis=1)
                lcol = mcoll.LineCollection(
                    segs, colors='#ff7f0e', linewidths=3.0, zorder=4,
                    label='boundary')
                ax.add_collection(lcol)
            else:
                for nm in _bnd_highlight:
                    if nm not in self._boundary_regions:
                        continue
                    reg   = self._boundary_regions[nm]
                    edges = reg['edges']      # (n_e, 2)
                    color = _bnd_color[nm]
                    segs  = np.stack([v[edges[:, 0]], v[edges[:, 1]]], axis=1)
                    lcol  = mcoll.LineCollection(
                        segs, colors=color, linewidths=3.0, zorder=4,
                        label=f'[B] {nm}')
                    ax.add_collection(lcol)

            # ── Boundary conditions ────────────────────────────────────────
            if boundary is not None:
                for bc in mesh_bcs:
                    if not _bc_in_phase(bc, phase):
                        continue
                    color = bc_color[id(bc)]
                    label = self._bc_label(
                        bc, list(bc_color.keys()).index(id(bc)))
                    self._draw_bc_on_ax(ax, bc, color, label, node_size)

            ax.set_aspect("equal")
            ax.autoscale_view()
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            if phase is None:
                ax.set_title("spatial", fontsize=10)
            else:
                p_lo, p_hi = phase
                if abs(p_lo - p_hi) < tol:
                    ax.set_title(f"t = {p_lo:.4g}", fontsize=10)
                else:
                    ax.set_title(f"t ∈ [{p_lo:.4g}, {p_hi:.4g}]", fontsize=10)

            handles, labels = ax.get_legend_handles_labels()
            visible = [(h, l) for h, l in zip(handles, labels)
                       if not l.startswith("_")]
            if visible:
                ax.legend(*zip(*visible), loc="upper right",
                          fontsize=8, framealpha=0.85)

        t_info = (f" × t∈[{self._t_min:.3g}, {self._t_max:.3g}]"
                  if self._t_min is not None else "")
        n_inner = len(self._inner_regions)
        n_bnd   = len(self._boundary_regions)
        region_info = (
            f", {n_inner} inner region{'s' if n_inner != 1 else ''}"
            f" + {n_bnd} boundary region{'s' if n_bnd != 1 else ''}"
            if n_inner or n_bnd else ""
        )
        fig.suptitle(
            f"{type(self).__name__}  |  {len(v)} nodes, {len(f)} triangles"
            f"  [{self._spatial_dims}D{t_info}]{region_info}",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()

        if points is not None:
            _pts = np.asarray(points)
            for ax in axes:
                ax.scatter(_pts[:, 0], _pts[:, 1],
                           s=8, color='tomato', alpha=0.5, zorder=5,
                           label='_points')
        return axes[0] if n_panels == 1 else axes

    def _plot_pyvista(
        self,
        show_mesh: bool = True,
        node_size: float = 30.0,
    ):
        """
        Interactive PyVista visualisation of the 2-D mesh.

        The triangulation is rendered as a flat surface (z = 0).  Any
        registered boundary conditions are shown as highlighted point clouds.
        """
        import pyvista as pv
        from pinns.boundary import MeshNodeBC

        v = self._vertices          # (N, 2)
        f = self._faces             # (M, 3)  0-indexed triangles

        # Build pyvista PolyData: vertices as (N, 3) with z = 0
        verts_3d = np.column_stack([v, np.zeros(len(v), dtype=np.float32)])
        # pyvista face format: [3, i0, i1, i2, ...]
        faces_pv = np.column_stack(
            [np.full(len(f), 3, dtype=np.int_), f]
        ).ravel()
        mesh = pv.PolyData(verts_3d, faces_pv)

        pl = pv.Plotter(notebook=self._is_jupyter(), off_screen=True)

        edge_color = '#888888'
        face_color = '#e8e8e8'
        pl.add_mesh(mesh, color=face_color,
                    show_edges=show_mesh, edge_color=edge_color,
                    opacity=0.85, label='mesh')

        # BC point clouds (if any are registered on this domain)
        _colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                   '#911eb4', '#42d4f4', '#f032e6', '#bfef45']
        mesh_bcs = [bc for bc in self.boundary_conditions
                    if isinstance(bc, MeshNodeBC)]
        for i, bc in enumerate(mesh_bcs):
            color = _colors[i % len(_colors)]
            if bc.node_indices is not None:
                pts2 = v[bc.node_indices]
            else:
                pts2 = bc.node_positions
            pts3 = np.column_stack(
                [pts2, np.zeros(len(pts2), dtype=np.float32)]
            )
            cloud = pv.PolyData(pts3)
            label = bc.name or f'bc_{i}'
            pl.add_mesh(cloud, render_points_as_spheres=True,
                        point_size=node_size, color=color, label=label)

        pl.view_xy()
        pl.add_axes()
        if mesh_bcs:
            pl.add_legend(bcolor='white', border=True)
        return pl.show(jupyter_backend='trame' if self._is_jupyter() else None)

    def __repr__(self):
        n_bcs  = len(self.boundary_conditions)
        n_in   = len(self._inner_regions)
        n_bnd  = len(self._boundary_regions)
        sp = f"{self._spatial_dims}D"
        if self._time_mode == 'continuous':
            t_info = f" × t∈[{self._t_min}, {self._t_max}] continuous"
        elif self._time_mode == 'discrete':
            t_info = f" × t∈[{self._t_min}, {self._t_max}] discrete({self.n_steps} steps)"
        else:
            t_info = ""
        region_info = ""
        if n_in or n_bnd:
            region_info = (
                f", inner_regions={list(self._inner_regions.keys())}"
                f", boundary_regions={list(self._boundary_regions.keys())}"
            )
        return (f"DomainMesh({sp}{t_info}, "
                f"n_nodes={len(self._vertices)}, n_conditions={n_bcs}"
                f"{region_info})")


