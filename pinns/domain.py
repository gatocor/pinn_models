import numpy as np
import torch
from itertools import product
from typing import Callable, Optional, Union, Literal, Tuple, List, Any
from dataclasses import dataclass

from .boundary import DirichletBC, NeumannBC, RobinBC, PointsetBC, CubicPeriodicBC


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
    Smooth bump function using sigmoid products, works with PyTorch tensors.
    
    Args:
        x: Input tensor of shape (..., n_dims)
        x_min: Lower bounds, tensor of shape (n_dims,) or broadcastable
        x_max: Upper bounds, tensor of shape (n_dims,) or broadcastable
        sigma_lower: Smoothing parameter for lower (xmin) boundary.
                    Scalar or tensor of shape (n_dims,)
        sigma_upper: Smoothing parameter for upper (xmax) boundary.
                    If None, uses sigma_lower for both. Scalar or tensor of shape (n_dims,)
        
    Returns:
        Tensor of shape (...,) with bump values in [0, 1]
    """
    if sigma_upper is None:
        sigma_upper = sigma_lower
    
    # Clamp arguments to avoid exp overflow (exp(88) is near float32 max)
    lower_arg = torch.clamp((x - x_min) / sigma_lower, min=-10, max=10)
    upper_arg = torch.clamp((x - x_max) / sigma_upper, min=-10, max=10)
    
    lower_sigmoid = 1.0 / (1 + torch.exp(-lower_arg))
    upper_sigmoid = 1.0 / (1 + torch.exp(upper_arg))
    return torch.prod(lower_sigmoid * upper_sigmoid, dim=-1)


def bump_vectorized(x, x_min, x_max, sigma_lower, sigma_upper=None):
    """
    Vectorized bump function for multiple subdomains at once.
    
    Args:
        x: Input tensor of shape (batch_size, n_dims)
        x_min: Lower bounds, tensor of shape (n_subdomains, n_dims)
        x_max: Upper bounds, tensor of shape (n_subdomains, n_dims)
        sigma_lower: Smoothing parameter for lower (xmin) boundaries, tensor of shape (n_subdomains, n_dims)
        sigma_upper: Smoothing parameter for upper (xmax) boundaries, tensor of shape (n_subdomains, n_dims)
                    If None, uses sigma_lower for both.
        
    Returns:
        Tensor of shape (batch_size, n_subdomains) with bump values in [0, 1]
    """
    if sigma_upper is None:
        sigma_upper = sigma_lower
    
    # x: (batch_size, n_dims) -> (batch_size, 1, n_dims)
    x_expanded = x.unsqueeze(1)
    
    # x_min, x_max, sigma: (n_subdomains, n_dims) -> (1, n_subdomains, n_dims)
    x_min = x_min.unsqueeze(0)
    x_max = x_max.unsqueeze(0)
    sigma_lower = sigma_lower.unsqueeze(0)
    sigma_upper = sigma_upper.unsqueeze(0)
    
    # Compute for all subdomains at once: (batch_size, n_subdomains, n_dims)
    lower_arg = torch.clamp((x_expanded - x_min) / sigma_lower, min=-10, max=10)
    upper_arg = torch.clamp((x_expanded - x_max) / sigma_upper, min=-10, max=10)
    
    lower_sigmoid = 1.0 / (1 + torch.exp(-lower_arg))
    upper_sigmoid = 1.0 / (1 + torch.exp(upper_arg))
    
    # Product over dimensions: (batch_size, n_subdomains)
    return torch.prod(lower_sigmoid * upper_sigmoid, dim=-1)


class DomainCubic:
    """
    A simple rectangular domain defined by lower and upper bounds.
    
    Args:
        xmin (array-like): Lower bounds for each dimension.
        xmax (array-like): Upper bounds for each dimension.
        sampling_method: Default sampling method for interior points. Can be:
            - "uniform": Standard uniform random sampling (default)
            - "latin_hypercube" or "lhs": Latin Hypercube Sampling
            - "sobol": Sobol quasi-random sequence
            - "halton": Halton quasi-random sequence
            - Callable: Custom function (n_points, n_dims, rng) -> ndarray in [0,1]^n
        sampling_transform: Optional custom transform function for interior sampling.
            Takes samples in [0,1]^n and returns transformed domain coordinates.
            Like an inverse CDF. Points outside domain are rejected and resampled.
        
    Example:
        # 2D domain [0,1] x [0,2] with Latin Hypercube sampling
        domain = DomainCubic(xmin=[0, 0], xmax=[1, 2], sampling_method="latin_hypercube")
        
        # DomainCubic with custom transform (e.g., denser sampling near origin)
        def my_transform(X):
            return X ** 2  # Denser near 0
        domain = DomainCubic(xmin=[0, 0], xmax=[1, 1], sampling_transform=my_transform)
        
        # Sample interior points
        points = domain.sample_interior(1000)
        
        # Sample boundary points
        boundary = domain.sample_boundary(100, dim=0, side=0)
    """
    
    def __init__(self, xmin, xmax, sampling_method="uniform", sampling_transform=None):
        self.xmin = np.asarray(xmin, dtype=np.float64)
        self.xmax = np.asarray(xmax, dtype=np.float64)
        
        if self.xmin.shape != self.xmax.shape:
            raise ValueError(
                f"xmin shape {self.xmin.shape} must match xmax shape {self.xmax.shape}"
            )
        
        if np.any(self.xmin >= self.xmax):
            raise ValueError("xmin must be strictly less than xmax in all dimensions")
        
        self.n_dims = len(self.xmin)
        self.sampling_method = sampling_method
        self.sampling_transform = sampling_transform
        
        # Storage for boundary conditions
        self.boundary_conditions: List[Union[DirichletBC, NeumannBC, RobinBC, PointsetBC]] = []
    
    @property
    def bounds(self):
        """Return (xmin, xmax) tuple."""
        return self.xmin, self.xmax
    
    @property
    def volume(self):
        """Compute the volume of the domain."""
        return np.prod(self.xmax - self.xmin)
    
    @property
    def extents(self):
        """Return the size in each dimension."""
        return self.xmax - self.xmin
    
    def sample_interior(self, n_points, rng=None, method=None, transform=None, params=None):
        """
        Sample points in the interior of the domain.
        
        Args:
            n_points (int): Number of points to sample.
            rng: Random number generator (np.random.Generator).
            method: Sampling method. If None, uses the domain's default sampling_method.
                Can be:
                - "uniform": Standard uniform random sampling
                - "latin_hypercube" or "lhs": Latin Hypercube Sampling
                - "sobol": Sobol quasi-random sequence
                - "halton": Halton quasi-random sequence
                - Callable: Custom function (n_points, n_dims, rng) -> ndarray in [0,1]^n
            transform: Optional custom transform function. If None, uses the domain's
                      default sampling_transform. Signature: transform(X, params) -> X_transformed.
                      Points outside domain are rejected and resampled.
            params: Optional dict passed to the transform function.
            
        Returns:
            np.ndarray: Array of shape (n_points, n_dims)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Use instance defaults if not specified
        if method is None:
            method = self.sampling_method
        if transform is None:
            transform = self.sampling_transform
        
        samples = sample_unit_hypercube(n_points, self.n_dims, method=method, rng=rng)
        return transform_samples(
            samples, self.xmin, self.xmax, 
            transform=transform, reject_outside=True, rng=rng, method=method, params=params
        )
    
    def sample_boundary(self, n_points, dim, side, rng=None, method="uniform", transform=None, params=None):
        """
        Sample points on a specific boundary.
        
        Args:
            n_points (int): Number of points to sample.
            dim (int): Dimension of the boundary (0 to n_dims-1).
            side: 0 for lower boundary, 1 for upper boundary.
            rng: Random number generator.
            method: Sampling method for the free dimensions. Can be:
                - "uniform": Standard uniform random sampling (default)
                - "latin_hypercube" or "lhs": Latin Hypercube Sampling
                - "sobol": Sobol quasi-random sequence
                - "halton": Halton quasi-random sequence
                - Callable: Custom function (n_points, n_dims, rng) -> ndarray in [0,1]^n
            transform: Optional custom transform function for the free dimensions.
                      Signature: transform(X, params) -> X_transformed.
                      Points outside domain are rejected and resampled.
            params: Optional dict passed to the transform function.
            
        Returns:
            np.ndarray: Array of shape (n_points, n_dims)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if dim < 0 or dim >= self.n_dims:
            raise ValueError(f"dim must be in [0, {self.n_dims - 1}]")
        
        if side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        
        # Sample in [0,1]^n_dims
        samples = sample_unit_hypercube(n_points, self.n_dims, method=method, rng=rng)
        
        # Transform to domain coordinates
        points = transform_samples(
            samples, self.xmin, self.xmax,
            transform=transform, reject_outside=True, rng=rng, method=method, params=params
        )
        
        if side == 0:
            points[:, dim] = self.xmin[dim]
        else:
            points[:, dim] = self.xmax[dim]
        
        return points
    
    def sample_all_boundaries(self, n_points, rng=None):
        """
        Sample points on all boundaries.
        
        Args:
            n_points (int): Number of points per boundary.
            rng: Random number generator.
            
        Returns:
            dict: Dictionary with keys 'dim{i}_{lower|upper}'
        """
        if rng is None:
            rng = np.random.default_rng()
        
        samples = {}
        for dim in range(self.n_dims):
            for side in [0, 1]:
                key = (dim,side)
                samples[key] = self.sample_boundary(n_points, dim, side, rng=rng)
        
        return samples
    
    def sample_interior_torch(self, n_points, rng=None, device='cpu', dtype=torch.float32):
        """Sample interior points and return as PyTorch tensor."""
        points = self.sample_interior(n_points, rng=rng)
        return torch.tensor(points, device=device, dtype=dtype)
    
    def sample_boundary_torch(self, n_points, dim, side, rng=None, 
                               device='cpu', dtype=torch.float32):
        """Sample boundary points and return as PyTorch tensor."""
        points = self.sample_boundary(n_points, dim, side, rng=rng)
        return torch.tensor(points, device=device, dtype=dtype)
    
    def contains(self, x):
        """
        Check if points are inside the domain.
        
        Args:
            x: Array of shape (n_points, n_dims) or (n_dims,)
            
        Returns:
            Boolean array of shape (n_points,) or scalar
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return np.all((x >= self.xmin) & (x <= self.xmax))
        return np.all((x >= self.xmin) & (x <= self.xmax), axis=1)
    
    # =========================================================================
    # Boundary Condition Methods
    # =========================================================================
    
    def add_dirichlet(
        self,
        boundary: Tuple,
        value: Union[float, Callable],
        component: int,
        name: str,
        subdomain: Optional[Tuple] = None,
        sampling_method: Union[str, Callable] = "uniform",
        sampling_transform: Optional[Callable] = None
    ) -> 'DomainCubic':
        """
        Add a Dirichlet boundary condition: u(x) = value on the boundary.
        
        Multiple conditions can be added to the same boundary.
        
        Args:
            boundary: Tuple specifying the boundary location.
                     Each element corresponds to a dimension:
                     - 0: lower boundary (e.g., x_min)
                     - 1: upper boundary (e.g., x_max)
                     - None: not constrained in this dimension
                     Example: (0, None) = x_min plane in 2D
            value: The boundary value. If callable, should take x as numpy array 
                  (batch_size, n_dims) and return values of shape (batch_size,).
            component: Output component index this condition applies to.
            name: Name for this BC (used in plots and compile dict).
            subdomain: Constrain sampling to a subdomain of the boundary.
            sampling_method: Sampling method for generating points.
            sampling_transform: Custom transform function for sampling.
            
        Returns:
            self (for method chaining)
            
        Example:
            domain = DomainCubic([0, 0], [1, 1])
            domain.add_dirichlet((0, None), value=0.0, component=0, name="left")  # u=0 at x_min
            domain.add_dirichlet((1, None), value=1.0, component=0, name="right")  # u=1 at x_max
        """
        bc = DirichletBC(
            boundary=boundary,
            value=value,
            component=component,
            subdomain=subdomain,
            name=name,
            sampling_method=sampling_method,
            sampling_transform=sampling_transform
        )
        self.boundary_conditions.append(bc)
        return self
    
    def add_neumann(
        self,
        boundary: Tuple,
        value: Union[float, Callable],
        component: int,
        name: str,
        subdomain: Optional[Tuple] = None,
        sampling_method: Union[str, Callable] = "uniform",
        sampling_transform: Optional[Callable] = None
    ) -> 'DomainCubic':
        """
        Add a Neumann boundary condition: du/dn = value on the boundary.
        
        The normal derivative is computed as the derivative with respect to
        the dimension perpendicular to the boundary.
        
        Args:
            boundary: Tuple specifying the boundary location.
                     - 0: lower boundary (normal points inward, -x direction)
                     - 1: upper boundary (normal points outward, +x direction)
                     - None: not constrained in this dimension
            value: The normal derivative value. If callable, takes x as numpy array.
            component: Output component index this condition applies to.
            name: Name for this BC (used in plots and compile dict).
            subdomain: Constrain sampling to a subdomain of the boundary.
            sampling_method: Sampling method for generating points.
            sampling_transform: Custom transform function for sampling.
            
        Returns:
            self (for method chaining)
            
        Example:
            domain = DomainCubic([0, 0], [1, 1])
            domain.add_neumann((0, None), value=0.0, component=0, name="left_flux")  # du/dx=0 at x_min
        """
        bc = NeumannBC(
            boundary=boundary,
            value=value,
            component=component,
            subdomain=subdomain,
            name=name,
            sampling_method=sampling_method,
            sampling_transform=sampling_transform
        )
        self.boundary_conditions.append(bc)
        return self
    
    def add_robin(
        self,
        boundary: Tuple,
        alpha: float,
        beta: float,
        value: Union[float, Callable],
        component: int,
        name: str,
        subdomain: Optional[Tuple] = None,
        sampling_method: Union[str, Callable] = "uniform",
        sampling_transform: Optional[Callable] = None
    ) -> 'DomainCubic':
        """
        Add a Robin (mixed) boundary condition: alpha*u + beta*du/dn = value.
        
        Args:
            boundary: Tuple specifying the boundary location.
            alpha: Coefficient for u term.
            beta: Coefficient for du/dn term.
            value: The boundary value. If callable, takes x as numpy array.
            component: Output component index this condition applies to.
            name: Name for this BC (used in plots and compile dict).
            subdomain: Constrain sampling to a subdomain of the boundary.
            sampling_method: Sampling method for generating points.
            sampling_transform: Custom transform function for sampling.
            
        Returns:
            self (for method chaining)
            
        Example:
            domain = DomainCubic([0, 0], [1, 1])
            # Convective BC: h*u + k*du/dn = h*u_inf
            domain.add_robin((1, None), alpha=10.0, beta=1.0, value=100.0, component=0, name="convective")
        """
        bc = RobinBC(
            boundary=boundary,
            alpha=alpha,
            beta=beta,
            value=value,
            component=component,
            subdomain=subdomain,
            name=name,
            sampling_method=sampling_method,
            sampling_transform=sampling_transform
        )
        self.boundary_conditions.append(bc)
        return self
    
    def add_pointset(
        self,
        inputs: Union[torch.Tensor, np.ndarray],
        outputs: Union[torch.Tensor, np.ndarray],
        component: int,
        name: str
    ) -> 'DomainCubic':
        """
        Add a Pointset boundary/data condition: u(x_i) = y_i for given data points.
        
        This condition enforces that the network output matches given values
        at specific input points. Useful for data-driven constraints, initial
        conditions from experimental data, or interior data constraints.
        
        Args:
            inputs: Input points of shape (n_points, n_dims).
            outputs: Target values of shape (n_points,) or (n_points, n_outputs).
            component: Output component index this condition applies to.
            name: Name for this BC (used in plots and compile dict).
            
        Returns:
            self (for method chaining)
            
        Example:
            import numpy as np
            domain = DomainCubic([0, 0], [1, 1])
            x_data = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            y_data = np.array([0.0, 0.5, 1.0])
            domain.add_pointset(x_data, y_data, component=0, name="data")
        """
        bc = PointsetBC(
            inputs=inputs,
            outputs=outputs,
            component=component,
            name=name
        )
        self.boundary_conditions.append(bc)
        return self
    
    def add_periodic(
        self,
        dim: int,
        name: str = 'periodic',
        component: Optional[int] = None,
        n_pairs: int = 200,
        match_x_derivative: bool = True,
    ) -> 'DomainCubic':
        """
        Add a soft periodic boundary condition along a spatial dimension.

        Enforces :math:`u(\\mathbf{x}_l) = u(\\mathbf{x}_r)` (and optionally
        :math:`\\partial_x u(\\mathbf{x}_l) = \\partial_x u(\\mathbf{x}_r)`) by
        adding a penalty term to the loss.

        If *component* is ``None``, one sub-condition is created per output
        component, named ``name_0``, ``name_1``, etc.

        Args:
            dim: Dimension along which the domain is periodic
                 (0 = first dim, 1 = second dim, …).
            name: Base name used in weights dicts.
            component: Output component index, or ``None`` for all components.
            n_pairs: Number of collocation pairs to sample at compile time.
            match_x_derivative: Also penalise the derivative mismatch
                :math:`|\\partial_{x_{dim}} u(x_l) - \\partial_{x_{dim}} u(x_r)|^2`.

        Returns:
            self (for method chaining)

        Example::

            domain = DomainCubic([0.0, -1.0], [1.0, 1.0])
            domain.add_periodic(dim=1, name="periodic")
        """
        bc = CubicPeriodicBC(
            dim=dim,
            n_pairs=n_pairs,
            component=component,
            name=name,
            match_x_derivative=match_x_derivative,
        )
        self.boundary_conditions.append(bc)
        return self

    def get_dirichlet_conditions(self) -> List[DirichletBC]:
        """Get all Dirichlet boundary conditions."""
        return [bc for bc in self.boundary_conditions if isinstance(bc, DirichletBC)]
    
    def get_neumann_conditions(self) -> List[NeumannBC]:
        """Get all Neumann boundary conditions."""
        return [bc for bc in self.boundary_conditions if isinstance(bc, NeumannBC)]
    
    def get_robin_conditions(self) -> List[RobinBC]:
        """Get all Robin boundary conditions."""
        return [bc for bc in self.boundary_conditions if isinstance(bc, RobinBC)]
    
    def get_pointset_conditions(self) -> List[PointsetBC]:
        """Get all Pointset boundary conditions."""
        return [bc for bc in self.boundary_conditions if isinstance(bc, PointsetBC)]
    
    def __repr__(self):
        n_bcs = len(self.boundary_conditions)
        return f"DomainCubic(xmin={self.xmin.tolist()}, xmax={self.xmax.tolist()}, n_conditions={n_bcs})"


class DomainCubicPartition(DomainCubic):
    """
    A domain partition for FBPINNs, inheriting from DomainCubic.
    
    The domain is partitioned into subdomains defined by grid positions in each dimension.
    Each position array defines the boundaries between subdomains, so n positions create
    n-1 subdomains. Domain bounds (xmin, xmax) are automatically extracted from the
    first and last positions.
    
    Inherits all boundary condition methods and properties from DomainCubic.
    
    Args:
        grid_positions (list of arrays): List of 1D arrays, one per dimension.
                                   Each array contains the boundary positions that define
                                   subdomains along that dimension.
                                   n positions create n-1 subdomains.
                                   First position becomes xmin, last becomes xmax.
                                   Example: [np.array([0, 0.5, 1])] creates 2 subdomains: [0,0.5] and [0.5,1]
        overlap (float): A value between 0 and 1 representing the fraction of each cell's
                        width to use as the overlap/sigma for window functions.
                        For example, overlap=0.2 means the window extends by 20% of
                        the cell width on each side.
                        Typical values: 0.1 to 0.5.
        sampling_method: Default sampling method for interior points. Can be:
            - "uniform": Standard uniform random sampling (default)
            - "latin_hypercube" or "lhs": Latin Hypercube Sampling
            - "sobol": Sobol quasi-random sequence
            - "halton": Halton quasi-random sequence
            - Callable: Custom function (n_points, n_dims, rng) -> ndarray in [0,1]^n
        sampling_transform: Optional custom transform function for interior sampling.
            Takes samples in [0,1]^n and returns transformed domain coordinates.
            Like an inverse CDF. Points outside domain are rejected and resampled.
    
    Example:
        # 1D domain [0,1] with 3 subdomains and 20% overlap
        partition = DomainCubicPartition(
            grid_positions=[np.array([0, 0.33, 0.67, 1])],
            overlap=1.0  # 100% of each cell's width
        )
        
        # Add boundary conditions (inherited from DomainCubic)
        partition.add_dirichlet((0,), value=0.0, component=0, name="left")
        
        # 2D domain [0,1] x [0,1] with non-uniform partition and LHS sampling
        partition = DomainCubicPartition(
            grid_positions=[np.array([0, 0.5, 1]), np.array([0, 0.5, 1])],
            overlap=1.0,  # 100% overlap
            sampling_method="latin_hypercube"
        )
    """
    
    def __init__(self, grid_positions, overlap=1.0, sampling_method="uniform", sampling_transform=None):
        # Validate grid_positions
        if not grid_positions:
            raise ValueError("grid_positions cannot be empty")
        
        self.grid_positions = [np.asarray(p, dtype=np.float64) for p in grid_positions]
        
        # Validate each position array
        for i, p in enumerate(self.grid_positions):
            if len(p) < 2:
                raise ValueError(
                    f"Dimension {i}: grid_positions must have at least 2 elements "
                    f"to define at least 1 subdomain"
                )
            if not np.all(np.diff(p) > 0):
                raise ValueError(
                    f"Dimension {i}: grid_positions must be strictly increasing"
                )
        
        # Extract domain bounds from grid_positions
        xmin = [p[0] for p in self.grid_positions]
        xmax = [p[-1] for p in self.grid_positions]
        
        # Initialize parent class (sets xmin, xmax, n_dims, boundary_conditions, sampling_method, etc.)
        super().__init__(xmin, xmax, sampling_method=sampling_method, sampling_transform=sampling_transform)
        
        # Validate overlap is in valid range
        if not (0 < overlap <= 1):
            raise ValueError(
                f"overlap must be between 0 and 1 (exclusive of 0), got {overlap}"
            )
        
        self.overlap = overlap
        
        # Compute lower and upper widths from overlap and cell sizes
        # At each interface, width is based on the minimum of adjacent cell sizes
        # This ensures consistent overlap at shared boundaries
        # Lower = at xmin side of cell, Upper = at xmax side of cell (per dimension)
        self.widths_lower = []
        self.widths_upper = []
        for p in self.grid_positions:
            # Cell sizes: difference between consecutive positions
            cell_sizes = np.diff(p)
            n_cells = len(cell_sizes)
            
            lower_widths = np.zeros(n_cells)
            upper_widths = np.zeros(n_cells)
            
            for i in range(n_cells):
                # Lower width: based on min(this cell, lower neighbor) if exists
                if i > 0:
                    lower_widths[i] = overlap * min(cell_sizes[i], cell_sizes[i - 1])
                else:
                    # First cell: lower boundary is domain edge, use own size
                    lower_widths[i] = overlap * cell_sizes[i]
                
                # Upper width: based on min(this cell, upper neighbor) if exists
                if i < n_cells - 1:
                    upper_widths[i] = overlap * min(cell_sizes[i], cell_sizes[i + 1])
                else:
                    # Last cell: upper boundary is domain edge, use own size
                    upper_widths[i] = overlap * cell_sizes[i]
            
            self.widths_lower.append(lower_widths)
            self.widths_upper.append(upper_widths)
        
        # For backward compatibility, keep self.widths as the average
        self.widths = [(l + u) / 2 for l, u in zip(self.widths_lower, self.widths_upper)]
        
        # Store the number of subdomains per dimension (n_grid_positions - 1)
        self.n_subdomains_per_dim = [len(p) - 1 for p in self.grid_positions]
        self.n_subdomains = int(np.prod(self.n_subdomains_per_dim))
        
        # Precompute all subdomain centers and widths
        self._subdomain_centers = None
        self._subdomain_widths = None
        self._subdomain_widths_lower = None
        self._subdomain_widths_upper = None
        self._compute_subdomains()
    
    def _compute_subdomains(self):
        """Compute all subdomain centers and widths using Cartesian product."""
        # Create indices for all subdomains (n_positions - 1 subdomains per dim)
        indices = [range(n) for n in self.n_subdomains_per_dim]
        
        centers_list = []
        widths_list = []
        widths_lower_list = []
        widths_upper_list = []
        
        for idx in product(*indices):
            # idx is a tuple of indices, one per dimension
            # Subdomain i spans from grid_positions[i] to grid_positions[i+1]
            # Center is midpoint: (grid_positions[i] + grid_positions[i+1]) / 2
            center = [(self.grid_positions[dim][idx[dim]] + self.grid_positions[dim][idx[dim] + 1]) / 2 
                      for dim in range(self.n_dims)]
            width = [self.widths[dim][idx[dim]] for dim in range(self.n_dims)]
            width_lower = [self.widths_lower[dim][idx[dim]] for dim in range(self.n_dims)]
            width_upper = [self.widths_upper[dim][idx[dim]] for dim in range(self.n_dims)]
            centers_list.append(center)
            widths_list.append(width)
            widths_lower_list.append(width_lower)
            widths_upper_list.append(width_upper)
        
        self._subdomain_centers = np.array(centers_list)
        self._subdomain_widths = np.array(widths_list)
        self._subdomain_widths_lower = np.array(widths_lower_list)
        self._subdomain_widths_upper = np.array(widths_upper_list)
    
    def get_subdomain_centers(self):
        """
        Get the centers of all subdomains.
        
        Returns:
            np.ndarray: Array of shape (n_subdomains, n_dims) containing 
                       the center coordinates of each subdomain.
        """
        return self._subdomain_centers.copy()
    
    def get_subdomain_widths(self):
        """
        Get the widths of all subdomains (average of left and right).
        
        Returns:
            np.ndarray: Array of shape (n_subdomains, n_dims) containing 
                       the width in each dimension for each subdomain.
        """
        return self._subdomain_widths.copy()
    
    def get_subdomain_widths_lower(self):
        """
        Get the lower boundary widths of all subdomains.
        
        Returns:
            np.ndarray: Array of shape (n_subdomains, n_dims) containing 
                       the lower (xmin side) width in each dimension for each subdomain.
        """
        return self._subdomain_widths_lower.copy()
    
    def get_subdomain_widths_upper(self):
        """
        Get the upper boundary widths of all subdomains.
        
        Returns:
            np.ndarray: Array of shape (n_subdomains, n_dims) containing 
                       the upper (xmax side) width in each dimension for each subdomain.
        """
        return self._subdomain_widths_upper.copy()
    
    def get_internal_boundary_positions(self, dim):
        """
        Get the internal boundary positions for a specific dimension.
        
        Args:
            dim (int): Dimension index.
            
        Returns:
            np.ndarray: Array of internal boundary positions (grid_positions[1:-1]).
        """
        return self.grid_positions[dim][1:-1].copy()
    
    def get_internal_boundary_widths(self, dim):
        """
        Get the internal boundary widths for a specific dimension.
        
        Args:
            dim (int): Dimension index.
            
        Returns:
            np.ndarray: Array of widths at internal boundaries.
        """
        return self.widths[dim].copy()
    
    def get_subdomain_bounds(self):
        """
        Get the bounding box of each subdomain based on positions.
        
        Returns:
            tuple: (lower_bounds, upper_bounds) where each is an array of 
                  shape (n_subdomains, n_dims)
        """
        indices = [range(n) for n in self.n_subdomains_per_dim]
        
        lower_list = []
        upper_list = []
        
        for idx in product(*indices):
            lower = [self.grid_positions[dim][idx[dim]] for dim in range(self.n_dims)]
            upper = [self.grid_positions[dim][idx[dim] + 1] for dim in range(self.n_dims)]
            lower_list.append(lower)
            upper_list.append(upper)
        
        return np.array(lower_list), np.array(upper_list)
    
    @property
    def subdomains(self) -> List[SubdomainInfo]:
        """
        Get a list of SubdomainInfo objects for all subdomains.
        
        Useful for building boolean masks based on subdomain properties.
        
        Returns:
            List[SubdomainInfo]: List of subdomain info objects.
            
        Example:
            # Create mask for subdomains where x >= 0
            mask = [sub.xmin[0] >= 0 for sub in partition.subdomains]
            
            # Use in FBPINN
            fbpinn = FBPINN(partition, network, active_subdomains=mask)
        """
        lower_bounds, upper_bounds = self.get_subdomain_bounds()
        
        subdomains = []
        for i in range(self.n_subdomains):
            multi_idx = self.get_multi_index(i)
            subdomains.append(SubdomainInfo(
                index=i,
                multi_index=multi_idx,
                xmin=lower_bounds[i],
                xmax=upper_bounds[i]
            ))
        return subdomains
    
    def get_domain_bounds(self):
        """
        Get the overall domain bounds.
        
        Returns:
            tuple: (xmin, xmax) where each is an array of shape (n_dims,)
        """
        return self.xmin.copy(), self.xmax.copy()
    
    def get_subdomain_index(self, *indices):
        """
        Get the flat subdomain index from per-dimension indices.
        
        Args:
            *indices: Index along each dimension.
            
        Returns:
            int: Flat subdomain index.
        """
        if len(indices) != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} indices, got {len(indices)}")
        
        flat_idx = 0
        multiplier = 1
        for dim in reversed(range(self.n_dims)):
            flat_idx += indices[dim] * multiplier
            multiplier *= self.n_subdomains_per_dim[dim]
        return flat_idx
    
    def get_multi_index(self, flat_index):
        """
        Convert a flat subdomain index to per-dimension indices.
        
        Args:
            flat_index: Flat subdomain index.
            
        Returns:
            tuple: Per-dimension indices.
        """
        indices = []
        remaining = flat_index
        for dim in reversed(range(self.n_dims)):
            n = self.n_subdomains_per_dim[dim]
            indices.append(remaining % n)
            remaining //= n
        return tuple(reversed(indices))
    
    def to_numpy(self):
        """
        Get subdomain centers and widths as NumPy arrays.
        
        Returns:
            tuple: (centers, widths_lower, widths_upper)
                   - centers: shape (n_subdomains, n_dims)
                   - widths_lower: shape (n_subdomains, n_dims) - for lower (xmin) boundaries
                   - widths_upper: shape (n_subdomains, n_dims) - for upper (xmax) boundaries
        """
        import numpy as np
        centers = np.array(self._subdomain_centers, dtype=np.float32)
        widths_lower = np.array(self._subdomain_widths_lower, dtype=np.float32)
        widths_upper = np.array(self._subdomain_widths_upper, dtype=np.float32)
        return centers, widths_lower, widths_upper
    
    def to_torch(self, device='cpu', dtype=torch.float32):
        """
        Get subdomain centers and widths as PyTorch tensors.
        
        Args:
            device: PyTorch device.
            dtype: PyTorch dtype.
            
        Returns:
            tuple: (centers_tensor, widths_lower_tensor, widths_upper_tensor)
                   - centers: shape (n_subdomains, n_dims)
                   - widths_lower: shape (n_subdomains, n_dims) - for lower (xmin) boundaries
                   - widths_upper: shape (n_subdomains, n_dims) - for upper (xmax) boundaries
        """
        centers = torch.tensor(self._subdomain_centers, device=device, dtype=dtype)
        widths_lower = torch.tensor(self._subdomain_widths_lower, device=device, dtype=dtype)
        widths_upper = torch.tensor(self._subdomain_widths_upper, device=device, dtype=dtype)
        return centers, widths_lower, widths_upper
    
    def compute_windows(self, x, normalize=True):
        """
        Compute window function values for all subdomains at given points.
        
        Uses smooth bump functions. The widths control the OVERLAP between 
        neighboring subdomains - each window extends beyond its subdomain 
        bounds by the width amount to create smooth blending regions.
        
        Vectorized implementation for efficiency.
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            normalize (bool): Whether to normalize windows to form a partition of unity.
                             Default: True
            
        Returns:
            Tensor of shape (batch_size, n_subdomains) with window values
        """
        device = x.device
        dtype = x.dtype
        
        # Use cached tensors if available and on correct device/dtype
        cache_key = (device, dtype)
        if not hasattr(self, '_window_cache') or self._window_cache.get('key') != cache_key:
            # Create and cache tensors
            lower_bounds_np, upper_bounds_np = self.get_subdomain_bounds()
            lower_bounds = torch.tensor(lower_bounds_np, device=device, dtype=dtype)
            upper_bounds = torch.tensor(upper_bounds_np, device=device, dtype=dtype)
            _, widths_lower, widths_upper = self.to_torch(device=device, dtype=dtype)
            
            # Pre-compute extended bounds
            # Lower widths extend the lower bounds, upper widths extend the upper bounds
            extended_lower = lower_bounds - widths_lower
            extended_upper = upper_bounds + widths_upper
            
            self._window_cache = {
                'key': cache_key,
                'extended_lower': extended_lower,
                'extended_upper': extended_upper,
                'widths_lower': widths_lower,
                'widths_upper': widths_upper
            }
        
        # Use cached values
        extended_lower = self._window_cache['extended_lower']
        extended_upper = self._window_cache['extended_upper']
        widths_lower = self._window_cache['widths_lower']
        widths_upper = self._window_cache['widths_upper']
        
        # Vectorized bump computation for all subdomains at once
        # Uses separate lower/upper widths for asymmetric windows
        # Returns: (batch_size, n_subdomains)
        windows = bump_vectorized(x, extended_lower, extended_upper, widths_lower, widths_upper)
        
        if normalize:
            # Normalize to form partition of unity
            window_sum = windows.sum(dim=-1, keepdim=True)
            window_sum = torch.clamp(window_sum, min=1e-8)
            windows = windows / window_sum
        
        return windows
    
    def _get_subdomain_volumes(self):
        """Compute volumes of all subdomains (based on window widths)."""
        return np.prod(self._subdomain_widths, axis=1)
    
    def sample_interior(self, n_points, rng=None, method=None, transform=None, params=None, mode='uniform'):
        """
        Sample points in the interior of the domain.
        
        Sampling uses the actual domain bounds (xmin, xmax), not subdomain widths.
        
        Args:
            n_points (int): Total number of points to sample.
            rng: Random number generator (np.random.Generator). If None, uses default.
            method: Sampling method. If None, uses the domain's default sampling_method.
                Can be:
                - "uniform": Standard uniform random sampling
                - "latin_hypercube" or "lhs": Latin Hypercube Sampling
                - "sobol": Sobol quasi-random sequence
                - "halton": Halton quasi-random sequence
                - Callable: Custom function (n_points, n_dims, rng) -> ndarray in [0,1]^n
            transform: Optional custom transform function. If None, uses the domain's
                      default sampling_transform. Signature: transform(X, params) -> X_transformed.
            params: Optional dict passed to the transform function.
            mode (str): Partition sampling mode:
                - 'uniform': Sample uniformly across the entire domain (default)
                - 'per_partition': Split n_points evenly across all subdomains
            
        Returns:
            np.ndarray: Array of shape (n_points, n_dims)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if mode == 'uniform':
            # Use parent class implementation for uniform sampling
            return super().sample_interior(n_points, rng=rng, method=method, transform=transform, params=params)
        
        elif mode == 'per_partition':
            # Use instance defaults if not specified
            if method is None:
                method = self.sampling_method
            if transform is None:
                transform = self.sampling_transform
            
            # Split n_points evenly across all subdomains
            points_per_subdomain = n_points // self.n_subdomains
            remainder = n_points % self.n_subdomains
            
            all_points = []
            
            for dim_idx in range(self.n_subdomains):
                multi_idx = self.get_multi_index(dim_idx)
                
                # Compute bounds for this subdomain from boundary positions
                sub_min = np.zeros(self.n_dims)
                sub_max = np.zeros(self.n_dims)
                
                for d in range(self.n_dims):
                    idx = multi_idx[d]
                    # Subdomain i spans from grid_positions[i] to grid_positions[i+1]
                    sub_min[d] = self.grid_positions[d][idx]
                    sub_max[d] = self.grid_positions[d][idx + 1]
                
                # Add extra point to first 'remainder' subdomains
                n_pts = points_per_subdomain + (1 if dim_idx < remainder else 0)
                if n_pts > 0:
                    samples = sample_unit_hypercube(n_pts, self.n_dims, method=method, rng=rng)
                    points = transform_samples(
                        samples, sub_min, sub_max,
                        transform=transform, reject_outside=True, rng=rng, method=method, params=params
                    )
                    all_points.append(points)
            
            return np.vstack(all_points)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'uniform' or 'per_partition'")
    
    def sample_boundary(self, n_points, dim, side, rng=None, method="uniform", transform=None, params=None, mode='uniform'):
        """
        Sample points on a specific boundary of the domain.
        
        For partitioned domains, can sample uniformly or distribute across partitions.
        
        Args:
            n_points (int): Total number of points to sample.
            dim (int): Dimension of the boundary (0 to n_dims-1).
            side (int): 0 for lower boundary, 1 for upper boundary.
            rng: Random number generator.
            method: Sampling method for the free dimensions (passed to parent).
            transform: Optional custom transform function (passed to parent).
            params: Optional dict passed to the transform function.
            mode (str): Partition sampling mode:
                - 'uniform': Sample uniformly on the boundary (default)
                - 'per_partition': Split n_points evenly across boundary partitions
            
        Returns:
            np.ndarray: Array of shape (n_points, n_dims)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if dim < 0 or dim >= self.n_dims:
            raise ValueError(f"dim must be in [0, {self.n_dims - 1}]")
        
        # Get the boundary value
        if side == 0:
            boundary_value = self.xmin[dim]
        else:
            boundary_value = self.xmax[dim]
        
        if mode == 'uniform':
            # Use parent class implementation for uniform sampling
            return super().sample_boundary(n_points, dim, side, rng=rng, method=method, transform=transform, params=params)
        
        elif mode == 'per_partition':
            # Use instance defaults if not specified for per_partition mode
            if method is None:
                method = self.sampling_method
            if transform is None:
                transform = self.sampling_transform
            
            # Find subdomains that touch this boundary
            boundary_subdomains = []
            
            for dim_idx in range(self.n_subdomains):
                multi_idx = self.get_multi_index(dim_idx)
                
                # Check if this subdomain touches the boundary
                pos_idx = multi_idx[dim]
                if side == 0 and pos_idx != 0:
                    continue
                if side == 1 and pos_idx != self.n_subdomains_per_dim[dim] - 1:
                    continue
                
                boundary_subdomains.append((dim_idx, multi_idx))
            
            n_boundary_partitions = len(boundary_subdomains)
            points_per_partition = n_points // n_boundary_partitions
            remainder = n_points % n_boundary_partitions
            
            all_points = []
            
            for i, (dim_idx, multi_idx) in enumerate(boundary_subdomains):
                # Compute bounds for this subdomain from boundary positions
                sub_min = np.zeros(self.n_dims)
                sub_max = np.zeros(self.n_dims)
                
                for d in range(self.n_dims):
                    idx = multi_idx[d]
                    # Subdomain i spans from grid_positions[i] to grid_positions[i+1]
                    sub_min[d] = self.grid_positions[d][idx]
                    sub_max[d] = self.grid_positions[d][idx + 1]
                
                # Add extra point to first 'remainder' partitions
                n_pts = points_per_partition + (1 if i < remainder else 0)
                if n_pts > 0:
                    samples = sample_unit_hypercube(n_pts, self.n_dims, method=method, rng=rng)
                    points = transform_samples(
                        samples, sub_min, sub_max,
                        transform=transform, reject_outside=True, rng=rng, method=method, params=params
                    )
                    points[:, dim] = boundary_value
                    all_points.append(points)
            
            return np.vstack(all_points)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'uniform' or 'per_partition'")
    
    def sample_all_boundaries(self, n_points, mode='uniform', rng=None):
        """
        Sample points on all boundaries of the domain.
        
        Args:
            n_points (int): Number of points per boundary (2 * n_dims boundaries total).
            mode (str): Sampling mode ('uniform' or 'per_partition').
            rng: Random number generator.
            
        Returns:
            dict: Dictionary with keys like 'dim0_lower', 'dim0_upper', etc.
                  Each value is an array of sampled points.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        boundary_samples = {}
        for dim in range(self.n_dims):
            for side in [0,1]:
                key = (dim, side)
                boundary_samples[key] = self.sample_boundary(
                    n_points, dim, side, mode=mode, rng=rng
                )
        
        return boundary_samples
    
    def sample_interior_torch(self, n_points, rng=None, method=None, transform=None, 
                               mode='uniform', device='cpu', dtype=torch.float32):
        """Sample interior points and return as PyTorch tensor."""
        points = self.sample_interior(n_points, rng=rng, method=method, transform=transform, mode=mode)
        return torch.tensor(points, device=device, dtype=dtype)
    
    def sample_boundary_torch(self, n_points, dim, side, rng=None, method="uniform", 
                               transform=None, mode='uniform', device='cpu', dtype=torch.float32):
        """Sample boundary points and return as PyTorch tensor."""
        points = self.sample_boundary(n_points, dim, side, rng=rng, method=method, 
                                       transform=transform, mode=mode)
        return torch.tensor(points, device=device, dtype=dtype)
    
    def __len__(self):
        """Return the total number of subdomains."""
        return self.n_subdomains
    
    def __repr__(self):
        n_bcs = len(self.boundary_conditions)
        return (
            f"DomainCubicPartition(xmin={self.xmin.tolist()}, xmax={self.xmax.tolist()}, "
            f"n_subdomains_per_dim={self.n_subdomains_per_dim}, "
            f"total_subdomains={self.n_subdomains}, n_conditions={n_bcs})"
        )


# =============================================================================
# Mesh-based domain
# =============================================================================

class DomainMesh:
    """
    A spatial domain defined by a triangular/tetrahedral mesh, optionally
    combined with a time interval.

    The mesh provides vertex positions and face connectivity. Interior sampling
    uses ``trimesh`` when available; otherwise falls back to bounding-box uniform
    sampling (acceptable for convex meshes, approximate otherwise).

    Boundary conditions are added via :meth:`add_dirichlet` / :meth:`add_neumann`.
    Nodes are selected either by an integer index array **or** a condition
    callable of the form ``select(vertices) -> bool mask``.

    Args:
        mesh: A mesh object with ``.vertices`` and ``.faces`` attributes
              (``trimesh.Trimesh``, ``pymesh.Mesh``, etc.).
        t_interval: Optional ``[t_min, t_max]``. When given, the domain is
                    ``(spatial mesh) × [t_min, t_max]`` and ``n_dims`` equals
                    ``spatial_dims + 1``.

    Example::

        import trimesh, pinns

        mesh = trimesh.load("geometry.obj")
        domain = pinns.DomainMesh(mesh, t_interval=[0.0, 1.0])

        # u = 0 on left wall (x ≈ 0) at all times
        domain.add_dirichlet(
            select=lambda v: v[:, 0] < 1e-6,
            value=0.0, component=0, name="left_wall"
        )

        # initial condition: u = sin(pi*y) at t=0
        domain.add_dirichlet(
            select=np.arange(500),
            value=lambda x: np.sin(np.pi * x[:, 1]),
            component=0, name="ic", t_mode="t_min"
        )
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

    def __init__(self, mesh, t_interval=None):
        vertices, faces = self._extract_vertices_faces(mesh)
        self._vertices = vertices
        self._spatial_dims = vertices.shape[1]
        self._faces = faces

        # For 2D meshes we use exact barycentric triangle sampling — no
        # trimesh needed.  For 3D meshes we still try trimesh as a fallback.
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

        # Precompute triangle areas (2D) for weighted sampling
        if self._spatial_dims == 2:
            A = vertices[self._faces[:, 0]]
            B = vertices[self._faces[:, 1]]
            C = vertices[self._faces[:, 2]]
            cross = (B - A)[:, 0] * (C - A)[:, 1] - (C - A)[:, 0] * (B - A)[:, 1]
            self._tri_areas = 0.5 * np.abs(cross)   # (n_faces,)
            self._tri_probs  = self._tri_areas / self._tri_areas.sum()
        else:
            self._tri_areas = None
            self._tri_probs  = None

        sp_min = vertices.min(axis=0)
        sp_max = vertices.max(axis=0)

        if t_interval is not None:
            self._t_min = float(t_interval[0])
            self._t_max = float(t_interval[1])
            self.xmin = np.append(sp_min, self._t_min)
            self.xmax = np.append(sp_max, self._t_max)
        else:
            self._t_min = None
            self._t_max = None
            self.xmin = sp_min
            self.xmax = sp_max

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

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _sample_interior_spatial(self, n_points: int, rng) -> np.ndarray:
        """Sample *n_points* spatial points inside the mesh."""
        # ---- 2D: exact barycentric sampling over triangles ---------------
        if self._spatial_dims == 2:
            v = self._vertices
            f = self._faces
            # Pick triangles proportionally to area
            tri_idx = rng.choice(len(f), n_points, p=self._tri_probs)
            A = v[f[tri_idx, 0]]
            B = v[f[tri_idx, 1]]
            C = v[f[tri_idx, 2]]
            # Uniform sampling inside triangle via barycentric coords
            r1 = rng.uniform(0.0, 1.0, n_points)
            r2 = rng.uniform(0.0, 1.0, n_points)
            mask = r1 + r2 > 1.0
            r1[mask] = 1.0 - r1[mask]
            r2[mask] = 1.0 - r2[mask]
            r3 = 1.0 - r1 - r2
            return (r1[:, None] * A + r2[:, None] * B + r3[:, None] * C)

        # ---- 3D: trimesh volume sampling → rejection → bbox fallback -----
        if self._trimesh is not None:
            try:
                import trimesh
                pts = trimesh.sample.volume_mesh(self._trimesh, n_points)
                pts = np.asarray(pts, dtype=float)
                if len(pts) >= n_points:
                    idx = rng.choice(len(pts), n_points, replace=False)
                    return pts[idx]
            except Exception:
                pass

            collected: List[np.ndarray] = []
            while len(collected) < n_points:
                extra = max(n_points * 4, 256)
                cands = rng.uniform(
                    self.xmin[:self._spatial_dims],
                    self.xmax[:self._spatial_dims],
                    (extra, self._spatial_dims)
                )
                inside = self._trimesh.contains(cands)
                for p in cands[inside]:
                    collected.append(p)
                    if len(collected) >= n_points:
                        break
            return np.array(collected[:n_points], dtype=float)

        # Pure bbox fallback (no containment test)
        return rng.uniform(
            self.xmin[:self._spatial_dims],
            self.xmax[:self._spatial_dims],
            (n_points, self._spatial_dims)
        )

    def _resolve_select(self, select) -> np.ndarray:
        """
        Return edge indices (into ``self._all_edges``) matching *select*.

        *select* can be one of:

        - **``(n, 2)`` integer array** — interpreted as ``(v0, v1)`` vertex-pair
          edge pairs (e.g. directly from ``mesh.cells_dict["line"]`` or a
          ``boundary_edges()`` helper).  Each pair is looked up in
          ``_edge_lookup``; unrecognised pairs are silently skipped.
        - **Callable** — called with the full vertex array ``(n_verts, 2)``;
          must return a boolean mask of shape ``(n_verts,)``.  An edge is
          included when **both** of its endpoints satisfy the mask.
        """
        if callable(select):
            mask = np.asarray(select(self._vertices), dtype=bool)
            v0_ok = mask[self._all_edges[:, 0]]
            v1_ok = mask[self._all_edges[:, 1]]
            edge_indices = np.where(v0_ok & v1_ok)[0].astype(np.intp)
        else:
            arr = np.asarray(select)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(
                    "select must be a (n, 2) array of vertex-index pairs "
                    "or a callable returning a boolean vertex mask. "
                    f"Got array with shape {arr.shape}."
                )
            edge_indices = self.edge_pairs_to_indices(arr)
        if len(edge_indices) == 0:
            raise ValueError("Edge selector matched zero edges.")
        return edge_indices

    def edge_pairs_to_indices(self, edge_pairs: np.ndarray) -> np.ndarray:
        """
        Convert an array of ``(v0, v1)`` vertex-index pairs to edge indices.

        Looks up each pair in ``self._edge_lookup`` (the canonical
        ``(min_v, max_v) -> edge_index`` dict built at construction time).
        This is the most direct way to go from the ``"line"`` cells that Gmsh /
        meshio stores for physical boundaries to the edge-index API expected by
        :meth:`add_dirichlet` / :meth:`add_neumann`.

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
            domain.add_neumann(select=eidx, value=0.0, ...)
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
    #  Public sampling API (called by the trainer)                        #
    # ------------------------------------------------------------------ #

    def sample_interior(self, n_points: int, rng=None, **kwargs) -> np.ndarray:
        """Return interior points of shape ``(n_points, n_dims)``."""
        if rng is None:
            rng = np.random.default_rng()
        pts_sp = self._sample_interior_spatial(n_points, rng)
        if self._t_min is not None:
            t = rng.uniform(self._t_min, self._t_max, (n_points, 1))
            return np.hstack([pts_sp, t])
        return pts_sp

    def sample_boundary(self, n_points: int, dim: int, side: int,
                        rng=None, **kwargs) -> np.ndarray:
        """
        Compatibility shim for time boundaries (``dim = spatial_dims``).

        For mesh surface BCs use :meth:`sample_boundary_bc` instead.
        """
        if rng is None:
            rng = np.random.default_rng()
        pts_sp = self._sample_interior_spatial(n_points, rng)
        if self._t_min is not None and dim == self._spatial_dims:
            t_val = self._t_min if side == 0 else self._t_max
            t = np.full((n_points, 1), t_val)
            return np.hstack([pts_sp, t])
        return self.sample_interior(n_points, rng=rng)

    def sample_boundary_bc(self, bc, n_points: int, rng=None) -> np.ndarray:
        """
        Sample *n_points* from the boundary curve stored in a :class:`MeshNodeBC`.

        When the BC has precomputed edges (the normal case), points are drawn
        **uniformly along the boundary edges** weighted by edge length — this
        gives continuous coverage of the boundary rather than being restricted
        to mesh node positions.  The returned index array contains the
        *edge index* of each sampled point; callers use it to look up the
        corresponding per-edge normal (``bc.edge_normals[idx]``).

        When no edges are available (isolated nodes or time BCs), the method
        falls back to drawing from ``bc.node_positions`` as before.

        A time coordinate is appended according to ``bc.t_mode``.
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

        if self._t_min is None or bc.t_mode is None:
            return pts_sp, idx
        elif bc.t_mode == "all":
            t = rng.uniform(self._t_min, self._t_max, (n_points, 1))
            return np.hstack([pts_sp, t]), idx
        elif bc.t_mode == "t_min":
            t = np.full((n_points, 1), self._t_min)
            return np.hstack([pts_sp, t]), idx
        elif bc.t_mode == "t_max":
            t = np.full((n_points, 1), self._t_max)
            return np.hstack([pts_sp, t]), idx
        else:
            raise ValueError(f"Unknown t_mode: {bc.t_mode!r}. "
                             "Choose 'all', 't_min', or 't_max'.")

    # ------------------------------------------------------------------ #
    #  Boundary-condition builders                                        #
    # ------------------------------------------------------------------ #

    def add_dirichlet(
        self,
        select,
        value,
        component: int = 0,
        name: str = None,
        t_mode: str = None,
    ) -> 'DomainMesh':
        """
        Add a Dirichlet BC: ``u = value`` on the selected nodes.

        Args:
            select: Node selector — an ``np.ndarray`` of integer indices **or**
                    a callable ``(vertices: ndarray) -> bool mask``.

                    Examples::

                        select=lambda v: v[:, 0] < 1e-6   # x ≈ 0 plane
                        select=np.array([0, 5, 10, 42])   # explicit indices
                        select=lambda v: np.linalg.norm(v, axis=1) > 0.99

            value: Dirichlet value. Scalar **or** ``(x_np) -> np.ndarray``
                   callable that receives the sampled coordinates (including t
                   if a time interval is set).
            component: Output component index (default 0).
            name: Label used in loss weight dicts and plots.
            t_mode: How to handle the time dimension (ignored for purely
                    spatial domains):

                    - ``None``    — no time appended (pure spatial domain)
                    - ``"all"``   — sample t uniformly in ``[t_min, t_max]``
                    - ``"t_min"`` — fix t = t_min  (initial condition)
                    - ``"t_max"`` — fix t = t_max  (final condition)

        Returns:
            *self* for method chaining.
        """
        from pinns.boundary import MeshNodeBC
        edge_indices  = self._resolve_select(select)
        edges         = self._all_edges[edge_indices]          # (n_sel, 2)
        node_positions = self._vertices[np.unique(edges)]      # unique vertices
        edge_lengths  = np.linalg.norm(
            self._vertices[edges[:, 1]] - self._vertices[edges[:, 0]], axis=1)
        bc = MeshNodeBC(
            node_positions=node_positions,
            value=value,
            bc_type="dirichlet",
            component=component,
            name=name,
            t_mode=t_mode if self._t_min is not None else None,
            t_min=self._t_min or 0.0,
            t_max=self._t_max or 1.0,
            edges=edges,
            edge_lengths=edge_lengths,
        )
        self.boundary_conditions.append(bc)
        return self

    def add_neumann(
        self,
        select,
        value,
        component: int = 0,
        name: str = None,
        t_mode: str = None,
    ) -> 'DomainMesh':
        """
        Add a Neumann BC: ``du/dn = value`` on the selected edges.

        For **time boundaries** (``t_mode="t_min"`` or ``"t_max"``) the normal
        direction is along the time axis.

        For **spatial surface** edges normals are inferred automatically from
        the local tangent of the boundary curve and oriented away from the
        mesh centroid.

        Args:
            select: Edge selector — ``(n, 2)`` vertex-pair array or callable
                    (see :meth:`add_dirichlet`).
            value: Normal-derivative value. Scalar or callable.
            component: Output component index.
            name: Label.
            t_mode: Time sampling mode (see :meth:`add_dirichlet`).

        Returns:
            *self* for method chaining.
        """
        from pinns.boundary import MeshNodeBC
        edge_indices   = self._resolve_select(select)
        edges          = self._all_edges[edge_indices]          # (n_sel, 2)
        node_positions = self._vertices[np.unique(edges)]       # unique vertices
        edge_lengths   = np.linalg.norm(
            self._vertices[edges[:, 1]] - self._vertices[edges[:, 0]], axis=1)
        if t_mode not in ("t_min", "t_max"):
            edge_normals = self._infer_edge_outward_normals(edges)
        else:
            edge_normals = None
        bc = MeshNodeBC(
            node_positions=node_positions,
            value=value,
            bc_type="neumann",
            component=component,
            name=name,
            t_mode=t_mode if self._t_min is not None else None,
            t_min=self._t_min or 0.0,
            t_max=self._t_max or 1.0,
            edges=edges,
            edge_lengths=edge_lengths,
            edge_normals=edge_normals,
        )
        self.boundary_conditions.append(bc)
        return self

    def add_bc(
        self,
        select,
        f: 'Callable',
        name = None,
    ) -> 'DomainMesh':
        """
        Add a **custom** boundary condition defined by a residual function.

        Unlike :meth:`add_dirichlet` / :meth:`add_neumann` which enforce a
        single output component against a scalar target, ``add_bc`` lets you
        write an arbitrary residual that is minimised in the loss.  The
        function signature mirrors ``pde_fn``::

            f(x, y, params, derivative) -> residual   # (n,) array or tuple

        The trainer calls ``f`` on the sampled boundary points and minimises
        ``mean(f(...)²)`` (or the sum of squared terms if a tuple is returned).

        Example — traction BC in elasticity (right edge, normal = (1,0))
        where :math:`\\sigma \\cdot \\mathbf{n} = (0.5, 0)`::

            def traction_right(x, y, params, derivative):
                lam = params['fixed']['lam']
                mu  = params['fixed']['mu']
                u1_x = derivative(y, x, 0, (0,))
                u2_y = derivative(y, x, 1, (1,))
                u1_y = derivative(y, x, 0, (1,))
                u2_x = derivative(y, x, 1, (0,))
                r0 = (lam + 2*mu)*u1_x + lam*u2_y - 0.5   # sigma_11 = 0.5
                r1 = mu*(u1_y + u2_x)                       # sigma_12 = 0
                return r0, r1

            domain.add_bc(edges_right, f=traction_right, name="right_traction")

        Args:
            select: Edge selector — ``(n, 2)`` vertex-pair array (e.g.
                    from a *pygmsh* / *meshio* ``cells_dict``), an
                    ``np.ndarray`` of integer indices into
                    ``domain._all_edges``, or a callable
                    ``(vertices) -> bool mask``.
            f: Residual callable.  Receives ``(x, y, params, derivative)``
               and returns a ``(n,)`` array **or** a tuple of ``(n,)`` arrays.
               The backend supplies the ``derivative`` function automatically;
               it supports the same conventions as ``pde_fn``.
            name: Label for the compile / weight dicts.  Required if you
                  want to assign a weight via ``trainer.compile(weights=...)``.

        Returns:
            *self* for method chaining.
        """
        import inspect as _inspect_mod
        from pinns.boundary import MeshCustomBC
        edge_indices   = self._resolve_select(select)
        edges          = self._all_edges[edge_indices]
        node_positions = self._vertices[np.unique(edges)]
        edge_lengths   = np.linalg.norm(
            self._vertices[edges[:, 1]] - self._vertices[edges[:, 0]], axis=1)
        # Detect weak BC: f accepts 'phi' in its signature
        _f_params  = list(_inspect_mod.signature(f).parameters.keys())
        _is_weak   = 'phi' in _f_params
        _weak_fn   = f if _is_weak else None
        # When name is a list, create one independent MeshCustomBC per output
        if isinstance(name, (list, tuple)):
            for idx, oname in enumerate(name):
                _idx = idx  # capture by value
                def _make_wrapper(fn, i):
                    import inspect as _insp
                    _params = list(_insp.signature(fn).parameters.keys())
                    _n = len(_params)
                    _has_phi = 'phi' in _params
                    def _wrapper_weak(x, y, params, phi, derivative):
                        result = fn(x, y, params, phi, derivative)
                        if isinstance(result, (list, tuple)):
                            return result[i]
                        return result
                    def _wrapper(x, y, params, derivative):
                        result = fn(x, y, params, derivative)
                        if isinstance(result, (list, tuple)):
                            return result[i]
                        return result
                    def _wrapper3(x, y, params):
                        result = fn(x, y, params)
                        if isinstance(result, (list, tuple)):
                            return result[i]
                        return result
                    def _wrapper2(x, y):
                        result = fn(x, y)
                        if isinstance(result, (list, tuple)):
                            return result[i]
                        return result
                    if _has_phi:
                        return _wrapper_weak
                    elif _n >= 4:
                        return _wrapper
                    elif _n == 3:
                        return _wrapper3
                    else:
                        return _wrapper2
                bc = MeshCustomBC(
                    node_positions=node_positions,
                    f=_make_wrapper(f, _idx),
                    name=oname,
                    output_names=[oname],
                    edges=edges,
                    edge_lengths=edge_lengths,
                    is_weak=_is_weak,
                    weak_fn=_weak_fn,
                )
                self.boundary_conditions.append(bc)
            return self
        # Single name — single BC
        bc = MeshCustomBC(
            node_positions=node_positions,
            f=f,
            name=name,
            output_names=[name] if name is not None else None,
            edges=edges,
            edge_lengths=edge_lengths,
            is_weak=_is_weak,
            weak_fn=_weak_fn,
        )
        self.boundary_conditions.append(bc)
        return self

    def add_periodic(
        self,
        select_a,
        select_b,
        component: int = None,
        name: str = 'periodic',
    ) -> 'DomainMesh':
        """
        Add a **periodic boundary condition** pairing nodes on two boundaries.

        For each node on edge set *A* the nearest node on edge set *B* (in the
        transverse direction) is found and the pair is stored.  The trainer
        then minimises

        .. math::
            \\text{mean}\\bigl(u(\\mathbf{x}_A) - u(\\mathbf{x}_B)\\bigr)^2

        Typical usage — periodic in x (left ↔ right)::

            domain.add_periodic(edges_left, edges_right,
                                component=None, name="periodic_x")

        Args:
            select_a: Edge selector for side A (array of vertex-pair rows or
                      callable as in :meth:`add_dirichlet`).
            select_b: Edge selector for side B.
            component: Output component index to enforce, or ``None`` (default)
                       to enforce periodicity for all components jointly.
            name: Label used in the weights dict.

        Returns:
            *self* for method chaining.
        """
        from pinns.boundary import PeriodicBC

        edges_a = self._all_edges[self._resolve_select(select_a)]
        edges_b = self._all_edges[self._resolve_select(select_b)]

        pts_a = self._vertices[np.unique(edges_a)]   # (na, 2)
        pts_b = self._vertices[np.unique(edges_b)]   # (nb, 2)

        # Match each node in pts_a to its periodic partner in pts_b.
        # Strategy: shift pts_a by the mean offset between the two boundaries
        # and then find the nearest neighbour in pts_b.
        shift = pts_b.mean(axis=0) - pts_a.mean(axis=0)   # (2,)
        pts_a_shifted = pts_a + shift                      # expected positions in pts_b

        # Nearest-neighbour matching (euclidean)
        from scipy.spatial import cKDTree
        tree = cKDTree(pts_b)
        dists, idx = tree.query(pts_a_shifted, k=1)

        import warnings
        max_dist = dists.max()
        tol = np.linalg.norm(shift) * 0.1 + 1e-10
        if max_dist > tol:
            warnings.warn(
                f"add_periodic('{name}'): largest pairing distance is {max_dist:.4g}."
                " The two boundaries may not have matching node distributions.",
                UserWarning,
            )

        pts_b_matched = pts_b[idx]   # (na, 2) — matched partners for pts_a

        bc = PeriodicBC(
            node_positions_a=pts_a.astype(np.float32),
            node_positions_b=pts_b_matched.astype(np.float32),
            component=component,
            name=name,
        )
        self.boundary_conditions.append(bc)
        return self

    def __repr__(self):
        n_bcs = len(self.boundary_conditions)
        sp = f"{self._spatial_dims}D"
        t_info = (f" × t∈[{self._t_min}, {self._t_max}]"
                  if self._t_min is not None else "")
        return (f"DomainMesh({sp}{t_info}, "
                f"n_nodes={len(self._vertices)}, n_conditions={n_bcs})")