# Domain Module

The domain module provides classes for defining computational domains, sampling strategies, and boundary conditions.

## Classes

### DomainCubic

A simple rectangular domain defined by lower and upper bounds in each dimension.

```python
class DomainCubic(xmin, xmax, sampling_method="uniform", sampling_transform=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `xmin` | array-like | Lower bounds for each dimension |
| `xmax` | array-like | Upper bounds for each dimension |
| `sampling_method` | str or callable | Default sampling method (see below) |
| `sampling_transform` | callable or None | Custom transform function for sampling |

**Sampling Methods:**

- `"uniform"`: Standard uniform random sampling
- `"latin_hypercube"` or `"lhs"`: Latin Hypercube Sampling for better space coverage
- `"sobol"`: Sobol quasi-random sequence
- `"halton"`: Halton quasi-random sequence
- Callable: Custom function `(n_points, n_dims, rng) -> ndarray` in $[0,1]^n$

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `bounds` | tuple | Returns `(xmin, xmax)` |
| `volume` | float | Volume of the domain |
| `extents` | ndarray | Size in each dimension |
| `n_dims` | int | Number of dimensions |
| `boundary_conditions` | list | List of attached boundary conditions |

**Example:**

```python
import pinns
import numpy as np

# 2D domain [0,1] x [0,2]
domain = pinns.DomainCubic(
    xmin=[0.0, 0.0],
    xmax=[1.0, 2.0],
    sampling_method="latin_hypercube"
)

# Sample 1000 interior points
points = domain.sample_interior(1000)
print(points.shape)  # (1000, 2)

# Sample 100 points on the left boundary (x=0)
boundary_points = domain.sample_boundary(100, dim=0, side=0)
```

---

### DomainCubicPartition

A partitioned rectangular domain for use with FBPINN. Inherits from `DomainCubic`.

```python
class DomainCubicPartition(grid_positions, sampling_method="uniform", sampling_transform=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `grid_positions` | list of arrays | Boundary positions for each dimension |
| `sampling_method` | str or callable | Default sampling method |
| `sampling_transform` | callable or None | Custom transform for domain sampling |

**Additional Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `n_subdomains` | int | Total number of subdomains |
| `n_subdomains_per_dim` | list | Number of subdomains per dimension |
| `subdomains` | list[SubdomainInfo] | List of subdomain metadata objects |

**Example:**

```python
# Create a 4x3 partition
x_boundaries = [0.0, 0.25, 0.5, 0.75, 1.0]  # 4 subdomains in x
t_boundaries = [0.0, 0.33, 0.67, 1.0]        # 3 subdomains in t

domain = pinns.DomainCubicPartition([x_boundaries, t_boundaries])

print(f"Total subdomains: {domain.n_subdomains}")  # 12
print(f"Per dimension: {domain.n_subdomains_per_dim}")  # [4, 3]

# Access subdomain info
for sub in domain.subdomains:
    print(f"Subdomain at {sub.center}, size {sub.size}")
```

---

### SubdomainInfo

A dataclass containing metadata about a subdomain.

```python
@dataclass
class SubdomainInfo:
    index: int          # Linear index of the subdomain
    multi_index: tuple  # Multi-dimensional index (i, j, ...)
    xmin: np.ndarray    # Lower bounds
    xmax: np.ndarray    # Upper bounds
```

**Computed Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `center` | ndarray | Center point of the subdomain |
| `size` | ndarray | Size in each dimension |

**Example:**

```python
# Filter subdomains based on position
# (useful for symmetry with active_subdomains in FBPINN)
active_mask = [sub.xmin[0] >= 0 for sub in domain.subdomains]
```

---

## Boundary Condition Methods

These methods are available on both `DomainCubic` and `DomainCubicPartition`.

### add_dirichlet

Add a Dirichlet boundary condition: $u = g$

```python
domain.add_dirichlet(boundary, value, component, name)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `boundary` | tuple | `(side_dim0, side_dim1, ...)` where each is 0/1/None |
| `value` | float or callable | Target value or function `f(X) -> tensor` |
| `component` | int | Output component index |
| `name` | str | Unique identifier for this BC |

**Boundary Specification:**

- `0`: Lower boundary ($x_i = x_{min,i}$)
- `1`: Upper boundary ($x_i = x_{max,i}$)
- `None`: All values along this dimension

**Examples:**

```python
# u = 0 at x = xmin (left boundary), for all y
domain.add_dirichlet((0, None), value=0.0, component=0, name="left")

# u = sin(pi*x) at t = 0 (initial condition)
domain.add_dirichlet(
    (None, 0), 
    value=lambda X: torch.sin(np.pi * X[:, 0:1]),
    component=0,
    name="initial"
)
```

---

### add_neumann

Add a Neumann boundary condition: $\frac{\partial u}{\partial n} = g$

```python
domain.add_neumann(boundary, value, component, name)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `boundary` | tuple | Boundary specification (same as Dirichlet) |
| `value` | float or callable | Target normal derivative value |
| `component` | int | Output component index |
| `name` | str | Unique identifier |

**Example:**

```python
# ∂u/∂x = 1.0 at x = xmax (right boundary)
domain.add_neumann((1, None), value=1.0, component=0, name="flux_right")
```

---

### add_robin

Add a Robin boundary condition: $a \cdot u + b \cdot \frac{\partial u}{\partial n} = g$

```python
domain.add_robin(boundary, a, b, value, component, name)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `boundary` | tuple | Boundary specification |
| `a` | float | Coefficient for $u$ |
| `b` | float | Coefficient for $\partial u / \partial n$ |
| `value` | float or callable | Target value $g$ |
| `component` | int | Output component index |
| `name` | str | Unique identifier |

**Example:**

```python
# u + 0.5 * ∂u/∂n = 0 at bottom boundary (convective BC)
domain.add_robin((None, 0), a=1.0, b=0.5, value=0.0, component=0, name="convective")
```

---

### add_pointset

Add boundary condition on custom points.

```python
domain.add_pointset(points, values, component, name)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | ndarray | Points of shape `(n_points, n_dims)` |
| `values` | ndarray or callable | Target values of shape `(n_points, 1)` |
| `component` | int | Output component index |
| `name` | str | Unique identifier |

**Example:**

```python
# Fit to experimental data
data_points = np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]])
data_values = np.array([[1.0], [0.5], [0.2]])

domain.add_pointset(data_points, data_values, component=0, name="data")
```

---

## Sampling Methods

### sample_interior

Sample points in the interior of the domain.

```python
points = domain.sample_interior(n_points, rng=None, method=None, transform=None, params=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_points` | int | Number of points |
| `rng` | Generator | NumPy random generator |
| `method` | str or callable | Override default sampling method |
| `transform` | callable | Override default transform |
| `params` | dict | Parameters passed to transform function |

**Returns:** `ndarray` of shape `(n_points, n_dims)`

---

### sample_boundary

Sample points on a specific boundary.

```python
points = domain.sample_boundary(n_points, dim, side, rng=None, method="uniform", transform=None, params=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_points` | int | Number of points |
| `dim` | int | Dimension index (0-indexed) |
| `side` | int | 0 for lower, 1 for upper boundary |
| `rng` | Generator | NumPy random generator |
| `method` | str | Sampling method for free dimensions |
| `transform` | callable | Optional transform function |
| `params` | dict | Parameters passed to transform |

**Returns:** `ndarray` of shape `(n_points, n_dims)`

---

## Utility Functions

### bump

Smooth bump function for window functions in FBPINN.

```python
pinns.bump(x, x_min, x_max, sigma_lower, sigma_upper=None)
```

Creates a smooth function that is 1 in $[x_{min}, x_{max}]$ and smoothly transitions to 0 outside.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | Tensor | Input values |
| `x_min` | float | Lower bound of unit region |
| `x_max` | float | Upper bound of unit region |
| `sigma_lower` | float | Transition width at lower boundary |
| `sigma_upper` | float or None | Transition width at upper boundary (defaults to sigma_lower) |

**Returns:** Tensor with values in $[0, 1]$

---

## Custom Sampling Transform

The `sampling_transform` parameter allows non-uniform sampling of the domain. It should be a function that transforms uniform samples in $[0,1]^n$ to points in the domain.

```python
def my_sampler(X, params):
    """
    Custom sampling transform.
    
    Args:
        X: ndarray of shape (n_points, n_dims) with values in [0, 1]
        params: dict with 'fixed', 'infer', 'internal' keys
        
    Returns:
        ndarray of shape (n_points, n_dims) with transformed coordinates
    """
    # Example: denser sampling near the origin
    x = X[:, 0:1]  # Uniform in [0, 1]
    t = X[:, 1:2]
    
    # Transform x to be denser near 0
    x_transformed = x ** 2  # More points near 0
    
    return np.hstack([x_transformed, t])

domain = pinns.DomainCubic(
    xmin=[0, 0], 
    xmax=[1, 1],
    sampling_transform=my_sampler
)
```

**With Problem Parameters:**

```python
import scipy.stats as sp

def gaussian_sampler(X, params):
    """Sample from mixture of uniform and Gaussian."""
    sigma = params["fixed"]["sigma"]
    w = params["fixed"]["weight"]
    
    n = int(X.shape[0] * w)
    
    # Uniform samples
    x_uniform = X[:n, 0:1]
    
    # Gaussian samples (using inverse CDF)
    x_gaussian = sp.norm.ppf(X[n:, 0:1], scale=sigma)
    
    x = np.vstack([x_uniform, x_gaussian])
    t = X[:, 1:2]
    
    return np.hstack([x, t])

domain = pinns.DomainCubicPartition(
    [[−1, 0, 1], [0, 1]],
    sampling_transform=gaussian_sampler
)

problem = pinns.Problem(
    domain=domain,
    pde_fn=my_pde,
    params={"sigma": 0.1, "weight": 0.3},
    ...
)
```
