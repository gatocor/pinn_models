# Functional Module

The functional module provides differential operators for computing derivatives in PDEs using automatic differentiation.

## Functions

### derivative

Compute partial derivatives of any order using automatic differentiation.

```python
pinns.derivative(Y, X, component, order, create_graph=True)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Y` | Tensor | Network output of shape `(batch_size, n_outputs)` |
| `X` | Tensor | Input tensor of shape `(batch_size, n_dims)` with `requires_grad=True` |
| `component` | int | Which output component to differentiate (0-indexed) |
| `order` | tuple[int] | Derivative order specification |
| `create_graph` | bool | Whether to create graph for higher-order derivatives |

**Order Specification:**

The `order` tuple specifies which input dimensions to differentiate with respect to:

| Order | Meaning |
|-------|---------|
| `(0,)` | $\frac{\partial}{\partial x_0}$ |
| `(1,)` | $\frac{\partial}{\partial x_1}$ |
| `(0, 0)` | $\frac{\partial^2}{\partial x_0^2}$ |
| `(1, 1)` | $\frac{\partial^2}{\partial x_1^2}$ |
| `(0, 1)` | $\frac{\partial^2}{\partial x_0 \partial x_1}$ |
| `(0, 0, 0)` | $\frac{\partial^3}{\partial x_0^3}$ |

**Returns:** Tensor of shape `(batch_size, 1)`

**Example:**

```python
import pinns
import torch

# Network with inputs (x, t) and output u
X = torch.randn(100, 2, requires_grad=True)
Y = network(X)  # shape (100, 1)

# First derivatives
u_x = pinns.derivative(Y, X, component=0, order=(0,))  # ∂u/∂x
u_t = pinns.derivative(Y, X, component=0, order=(1,))  # ∂u/∂t

# Second derivatives
u_xx = pinns.derivative(Y, X, component=0, order=(0, 0))  # ∂²u/∂x²
u_tt = pinns.derivative(Y, X, component=0, order=(1, 1))  # ∂²u/∂t²
u_xt = pinns.derivative(Y, X, component=0, order=(0, 1))  # ∂²u/∂x∂t

# Multiple outputs
# If Y has shape (100, 2) with components [u, v]
v_x = pinns.derivative(Y, X, component=1, order=(0,))  # ∂v/∂x
```

---

### gradient

Compute the full gradient vector of an output with respect to all inputs.

```python
pinns.gradient(Y, X, component=0, create_graph=True)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Y` | Tensor | Network output |
| `X` | Tensor | Input tensor with `requires_grad=True` |
| `component` | int | Output component index (default: 0) |
| `create_graph` | bool | Create graph for backprop |

**Returns:** Tensor of shape `(batch_size, n_dims)`

**Example:**

```python
# Get gradient: [∂u/∂x, ∂u/∂y]
grad_u = pinns.gradient(Y, X, component=0)

# Gradient magnitude
grad_magnitude = torch.norm(grad_u, dim=1, keepdim=True)
```

---

### laplacian

Compute the Laplacian (sum of second derivatives) of an output.

$$\nabla^2 u = \sum_{i=1}^{n} \frac{\partial^2 u}{\partial x_i^2}$$

```python
pinns.laplacian(Y, X, component=0, create_graph=True)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Y` | Tensor | Network output |
| `X` | Tensor | Input tensor with `requires_grad=True` |
| `component` | int | Output component index |
| `create_graph` | bool | Create graph for backprop |

**Returns:** Tensor of shape `(batch_size, 1)`

**Example:**

```python
# Laplacian: ∂²u/∂x² + ∂²u/∂y²
lap_u = pinns.laplacian(Y, X, component=0)

# Poisson equation: ∇²u = f
def poisson(X, V, params):
    f = params["fixed"]["source"](X)
    return pinns.laplacian(V, X, 0) - f
```

---

### divergence

Compute the divergence of a vector field.

$$\nabla \cdot \mathbf{v} = \sum_{i=1}^{n} \frac{\partial v_i}{\partial x_i}$$

```python
pinns.divergence(Y, X, components=None, create_graph=True)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Y` | Tensor | Network output |
| `X` | Tensor | Input tensor with `requires_grad=True` |
| `components` | list[int] | Output components forming the vector field |
| `create_graph` | bool | Create graph for backprop |

**Returns:** Tensor of shape `(batch_size, 1)`

**Example:**

```python
# For a 2D velocity field (u, v):
# div = ∂u/∂x + ∂v/∂y

# If output is [u, v, p], divergence of velocity:
div_v = pinns.divergence(Y, X, components=[0, 1])

# Incompressibility constraint: ∇·v = 0
def continuity(X, V, params):
    return pinns.divergence(V, X, components=[0, 1])
```

---

## Common PDE Patterns

### Wave Equation

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

```python
def wave_equation(X, V, params):
    c = params["fixed"]["c"]
    
    u_tt = pinns.derivative(V, X, 0, (1, 1))  # ∂²u/∂t²
    u_xx = pinns.derivative(V, X, 0, (0, 0))  # ∂²u/∂x²
    
    return u_tt - c**2 * u_xx
```

### Burgers' Equation

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

```python
def burgers(X, V, params):
    nu = params["fixed"]["nu"]
    
    u = V[:, 0:1]
    u_t = pinns.derivative(V, X, 0, (1,))
    u_x = pinns.derivative(V, X, 0, (0,))
    u_xx = pinns.derivative(V, X, 0, (0, 0))
    
    return u_t + u * u_x - nu * u_xx
```

### Schrödinger Equation

$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \nabla^2 \psi + V \psi$$

```python
def schrodinger(X, V, params):
    # V has real and imaginary parts: [psi_r, psi_i]
    hbar = params["fixed"]["hbar"]
    m = params["fixed"]["m"]
    potential = params["fixed"]["V"](X)
    
    psi_r = V[:, 0:1]
    psi_i = V[:, 1:2]
    
    # Time derivatives
    psi_r_t = pinns.derivative(V, X, 0, (2,))  # Assuming t is dim 2
    psi_i_t = pinns.derivative(V, X, 1, (2,))
    
    # Laplacians (spatial)
    lap_r = pinns.laplacian(V, X, 0)
    lap_i = pinns.laplacian(V, X, 1)
    
    # Real part of equation
    eq_r = hbar * psi_i_t + (hbar**2 / (2*m)) * lap_r - potential * psi_r
    
    # Imaginary part
    eq_i = -hbar * psi_r_t + (hbar**2 / (2*m)) * lap_i - potential * psi_i
    
    return eq_r, eq_i
```

### Coupled System

For systems with multiple PDEs, return a tuple of residuals:

```python
def coupled_system(X, V, params):
    # Outputs: [u, v]
    u = V[:, 0:1]
    v = V[:, 1:2]
    
    # Derivatives
    u_t = pinns.derivative(V, X, 0, (1,))
    v_t = pinns.derivative(V, X, 1, (1,))
    u_x = pinns.derivative(V, X, 0, (0,))
    v_x = pinns.derivative(V, X, 1, (0,))
    
    # Coupled equations
    eq1 = u_t - v_x  # ∂u/∂t = ∂v/∂x
    eq2 = v_t - u_x  # ∂v/∂t = ∂u/∂x
    
    return eq1, eq2
```

---

## Performance Tips

1. **Reuse derivatives**: If the same derivative appears multiple times, compute it once:
   ```python
   u_x = pinns.derivative(V, X, 0, (0,))
   # Use u_x in multiple places
   ```

2. **Disable graph for inference**: For evaluation (not training), set `create_graph=False`:
   ```python
   with torch.no_grad():
       u_x = pinns.derivative(V, X, 0, (0,), create_graph=False)
   ```

3. **Higher-order derivatives**: Computing very high order derivatives (4th, 5th) can be slow. Consider using finite differences for these if performance is critical.
