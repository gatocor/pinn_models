# Problem Module

The Problem class combines a domain, PDE residual function, and parameters into a complete problem definition.

## Problem Class

```python
@dataclass
class Problem:
    domain: Union[DomainCubic, DomainCubicPartition]
    pde_fn: Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor]
    params: Dict[str, Any] = field(default_factory=dict)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    output_range: Optional[Union[tuple, List[Optional[tuple]]]] = None
    solution: Optional[Callable] = None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `domain` | DomainCubic | Computational domain with boundary conditions |
| `pde_fn` | callable | PDE residual function |
| `params` | dict | Fixed problem parameters (coefficients, etc.) |
| `input_names` | list[str] | Names for input dimensions **[required]** |
| `output_names` | list[str] | Names for output components **[required]** |
| `output_range` | list of tuples | Output range for unnormalization |
| `solution` | callable | Analytical solution for error computation |

**Derived Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `n_dims` | int | Number of input dimensions (from domain) |
| `n_outputs` | int | Number of output components (from output_names) |
| `boundary_conditions` | list | Boundary conditions from domain |

---

## PDE Residual Function

The PDE function computes residuals that should be zero when the PDE is satisfied.

### Signature

```python
def pde_fn(X: torch.Tensor, V: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Compute PDE residual.
    
    Args:
        X: Input tensor of shape (batch_size, n_dims)
        V: Network output tensor of shape (batch_size, n_outputs)
        params: Dictionary containing:
            - 'fixed': User-provided parameters
            - 'infer': Parameters to infer (future)
            - 'internal': Training state {global_step, step}
            
    Returns:
        Residual tensor of shape (batch_size,) or (batch_size, n_equations)
    """
    pass
```

### Example: Heat Equation

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

```python
def heat_equation(X, V, params):
    alpha = params["fixed"]["alpha"]
    
    u_t = pinns.derivative(V, X, component=0, order=(1,))    # ∂u/∂t
    u_xx = pinns.derivative(V, X, component=0, order=(0, 0)) # ∂²u/∂x²
    
    return u_t - alpha * u_xx
```

### Example: Navier-Stokes (2D, Steady)

$$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu \nabla^2 u$$
$$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu \nabla^2 v$$
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

```python
def navier_stokes(X, V, params):
    nu = params["fixed"]["nu"]
    rho = params["fixed"]["rho"]
    
    # Outputs: u, v, p
    u = V[:, 0:1]
    v = V[:, 1:2]
    p = V[:, 2:3]
    
    # First derivatives
    u_x = pinns.derivative(V, X, 0, (0,))
    u_y = pinns.derivative(V, X, 0, (1,))
    v_x = pinns.derivative(V, X, 1, (0,))
    v_y = pinns.derivative(V, X, 1, (1,))
    p_x = pinns.derivative(V, X, 2, (0,))
    p_y = pinns.derivative(V, X, 2, (1,))
    
    # Second derivatives
    u_xx = pinns.derivative(V, X, 0, (0, 0))
    u_yy = pinns.derivative(V, X, 0, (1, 1))
    v_xx = pinns.derivative(V, X, 1, (0, 0))
    v_yy = pinns.derivative(V, X, 1, (1, 1))
    
    # Momentum equations
    momentum_x = u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy)
    
    # Continuity equation
    continuity = u_x + v_y
    
    return momentum_x, momentum_y, continuity
```

---

## Output Range

The `output_range` parameter specifies the physical range for each output, enabling unnormalization.

```python
# Single range for all outputs
problem = Problem(..., output_range=(0, 1))

# Different range per output
problem = Problem(
    ...,
    output_names=["h", "vx", "vy"],
    output_range=[
        (0, 1),     # h in [0, 1]
        (-1, 1),    # vx in [-1, 1]
        (-1, 1)     # vy in [-1, 1]
    ]
)

# Some outputs unnormalized, others not
problem = Problem(
    ...,
    output_names=["u", "p"],
    output_range=[
        None,       # u: no unnormalization
        (0, 100)    # p in [0, 100]
    ]
)
```

---

## Analytical Solution

If an analytical solution is known, provide it for error visualization during training.

```python
def analytical_solution(X, params):
    """
    Analytical solution.
    
    Args:
        X: Input array of shape (batch_size, n_dims)
        params: Same dictionary structure as pde_fn
        
    Returns:
        Solution array of shape (batch_size, n_outputs)
    """
    alpha = params["fixed"]["alpha"]
    x = X[:, 0:1]
    t = X[:, 1:2]
    
    u = np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)
    return u

problem = Problem(
    domain=domain,
    pde_fn=heat_equation,
    input_names=["x", "t"],
    output_names=["u"],
    params={"alpha": 0.01},
    solution=analytical_solution
)
```

---

## Curriculum Learning

Use `params["internal"]` for curriculum learning (gradually increasing difficulty):

```python
def pde_with_curriculum(X, V, params):
    step = params["internal"]["global_step"]
    
    # Gradually introduce higher-order terms
    high_order_weight = min(1.0, step / 10000)
    
    u_t = pinns.derivative(V, X, 0, (1,))
    u_xx = pinns.derivative(V, X, 0, (0, 0))
    u_xxxx = pinns.derivative(V, X, 0, (0, 0, 0, 0))  # Fourth derivative
    
    # Start with heat equation, gradually add fourth-order term
    return u_t - u_xx - high_order_weight * u_xxxx
```

---

## Complete Example

```python
import pinns
import torch
import numpy as np

# 1. Define domain
domain = pinns.DomainCubic([0, 0], [1, 1])

# 2. Add boundary conditions
domain.add_dirichlet((0, None), value=0.0, component=0, name="left")
domain.add_dirichlet((1, None), value=0.0, component=0, name="right")
domain.add_dirichlet(
    (None, 0),
    value=lambda x: torch.sin(np.pi * x[:, 0:1]),
    component=0,
    name="initial"
)

# 3. Define PDE
def heat_equation(X, V, params):
    alpha = params["fixed"]["alpha"]
    u_t = pinns.derivative(V, X, 0, (1,))
    u_xx = pinns.derivative(V, X, 0, (0, 0))
    return u_t - alpha * u_xx

# 4. Define analytical solution
def solution(X, params):
    alpha = params["fixed"]["alpha"]
    x, t = X[:, 0:1], X[:, 1:2]
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

# 5. Create problem
problem = pinns.Problem(
    domain=domain,
    pde_fn=heat_equation,
    input_names=["x", "t"],
    output_names=["u"],
    output_range=(0, 1),
    params={"alpha": 0.01},
    solution=solution
)
```
