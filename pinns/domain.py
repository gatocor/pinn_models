import numpy as np
import torch
from itertools import product
from typing import Callable, Optional, Union, Literal, Tuple, List, Any
from dataclasses import dataclass

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