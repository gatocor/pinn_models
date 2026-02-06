# Networks Module

The networks module provides neural network architectures for solving PDEs.

## Classes

### FNN

A fully-connected (feedforward) neural network with configurable architecture.

```python
class FNN(layer_sizes, activation='tanh', output_activation=None,
          normalize_input=True, unnormalize_output=True,
          input_transform=None, output_transform=None, seed=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layer_sizes` | list[int] | Size of each layer, e.g., `[2, 64, 64, 1]` |
| `activation` | str or nn.Module | Activation function between layers |
| `output_activation` | str or nn.Module | Optional activation for output layer |
| `normalize_input` | bool | Normalize inputs to [-1, 1] (default: True) |
| `unnormalize_output` | bool | Unnormalize outputs to physical range (default: True) |
| `input_transform` | callable | Pre-normalization transform (e.g., for symmetries) |
| `output_transform` | callable | Post-unnormalization transform (e.g., hard constraints) |
| `seed` | int | Random seed for weight initialization |

**Supported Activations:**

- `'tanh'` (default, recommended for PINNs)
- `'relu'`
- `'gelu'`
- `'silu'` (Swish)
- `'sigmoid'`
- `'leaky_relu'`
- `'elu'`
- `'softplus'`

**Network Pipeline:**

1. **Input Transform** (optional): `x' = input_transform(x, params)`
2. **Normalization** (optional): Maps domain bounds to [-1, 1]
3. **Neural Network**: Forward pass through layers
4. **Unnormalization** (optional): Maps [-1, 1] to output range
5. **Output Transform** (optional): `y' = output_transform(x, y, params)`

**Example:**

```python
import pinns
import torch

# Simple network
net = pinns.FNN([2, 64, 64, 1])

# Network with symmetry (even function in x)
net = pinns.FNN(
    [2, 64, 64, 1],
    activation='tanh',
    input_transform=lambda x, p: torch.hstack([torch.abs(x[:, 0:1]), x[:, 1:2]])
)

# Network with hard constraint (u=0 at t=0)
net = pinns.FNN(
    [2, 64, 64, 1],
    output_transform=lambda x, y, p: x[:, 1:2] * y  # y=0 when t=0
)
```

---

### FBPINN

Finite Basis Physics-Informed Neural Network using domain decomposition.

FBPINN creates a separate neural network for each subdomain and combines them using smooth window functions (partition of unity).

```python
class FBPINN(partition, base_network, window_sigma=0.5,
             normalize_input=True, unnormalize_output=True,
             input_transform=None, output_transform=None,
             active_subdomains=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `partition` | DomainCubicPartition | Partitioned domain |
| `base_network` | FNN | Template network (copied for each subdomain) |
| `window_sigma` | float | Width of window function overlap (default: 0.5) |
| `normalize_input` | bool | Normalize inputs per subdomain |
| `unnormalize_output` | bool | Unnormalize outputs |
| `input_transform` | callable | Pre-transform (applied before windowing) |
| `output_transform` | callable | Post-transform (applied after combination) |
| `active_subdomains` | list[bool] | Mask for active subdomains (for symmetry) |

**How FBPINN Works:**

1. Domain is decomposed into overlapping subdomains
2. Each subdomain has its own neural network
3. Window functions ensure smooth blending at boundaries
4. Final output: $u(x) = \sum_i w_i(x) \cdot u_i(x)$

**Example:**

```python
import pinns

# Create partitioned domain
domain = pinns.DomainCubicPartition([
    [0, 0.25, 0.5, 0.75, 1.0],  # 4 subdomains in x
    [0, 0.5, 1.0]               # 2 subdomains in t
])

# Create base network (template)
base_net = pinns.FNN([2, 32, 32, 1], activation='tanh')

# Create FBPINN
network = pinns.FBPINN(
    partition=domain,
    base_network=base_net,
    normalize_input=True,
    unnormalize_output=True
)

print(f"Number of subdomains: {domain.n_subdomains}")  # 8
print(f"Total parameters: {sum(p.numel() for p in network.parameters())}")
```

---

## Using Symmetries

When your problem has spatial symmetry, you can exploit it to reduce computation and improve training.

### Mirror Symmetry Example

For a problem symmetric about x=0 with $u(-x) = u(x)$:

```python
def input_transform(X, params):
    """Map x -> |x| to exploit symmetry."""
    x = torch.abs(X[:, 0:1])
    t = X[:, 1:2]
    return torch.hstack([x, t])

# Domain spans [-1, 1] x [0, 1]
domain = pinns.DomainCubicPartition([
    [-1, -0.5, 0, 0.5, 1],  # 4 subdomains
    [0, 0.5, 1]
])

# Only use networks for x >= 0
active_mask = [sub.xmin[0] >= 0 for sub in domain.subdomains]

network = pinns.FBPINN(
    partition=domain,
    base_network=pinns.FNN([2, 32, 1]),
    input_transform=input_transform,
    active_subdomains=active_mask  # Disable networks for x < 0
)
```

### Antisymmetric Functions

For antisymmetric functions $v(-x) = -v(x)$:

```python
def output_transform(X, V, params):
    """Restore antisymmetry: v(-x) = -v(x)."""
    x = X[:, 0:1]  # Original x (before input_transform)
    v = V[:, 0:1]
    return torch.sign(x) * v

network = pinns.FBPINN(
    ...,
    input_transform=lambda x, p: torch.hstack([torch.abs(x[:, 0:1]), x[:, 1:]]),
    output_transform=output_transform
)
```

---

## Hard Constraints

Hard constraints encode boundary/initial conditions directly in the network architecture, guaranteeing satisfaction.

### Initial Condition Constraint

For a PDE with initial condition $u(x, 0) = f(x)$:

```python
def output_transform(X, V, params):
    """Enforce u(x, 0) = sin(pi*x)."""
    x = X[:, 0:1]
    t = X[:, 1:2]
    NN = V[:, 0:1]
    
    # u(x, t) = f(x) + t * NN(x, t)
    # At t=0: u = f(x) exactly
    f_x = torch.sin(np.pi * x)
    return f_x + t * NN

network = pinns.FNN(
    [2, 64, 64, 1],
    output_transform=output_transform
)
```

### Boundary Condition Constraint

For $u(0, t) = u(1, t) = 0$:

```python
def output_transform(X, V, params):
    """Enforce u = 0 at x = 0 and x = 1."""
    x = X[:, 0:1]
    NN = V[:, 0:1]
    
    # u(x, t) = x * (1 - x) * NN(x, t)
    # At x=0 or x=1: u = 0 exactly
    return x * (1 - x) * NN

network = pinns.FNN(
    [2, 64, 64, 1],
    output_transform=output_transform
)
```

---

## Transform Function Signatures

### input_transform

```python
def input_transform(X: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Transform inputs before normalization.
    
    Args:
        X: Input tensor of shape (batch_size, n_dims)
        params: Dictionary with 'fixed', 'infer', 'internal' keys
        
    Returns:
        Transformed tensor of shape (batch_size, n_dims)
    """
    pass
```

### output_transform

```python
def output_transform(X: torch.Tensor, V: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Transform outputs after unnormalization.
    
    Args:
        X: Original input tensor (before input_transform)
        V: Network output tensor of shape (batch_size, n_outputs)
        params: Dictionary with 'fixed', 'infer', 'internal' keys
        
    Returns:
        Transformed output tensor of shape (batch_size, n_outputs)
    """
    pass
```

### Accessing Parameters

```python
def output_transform(X, V, params):
    # Access user-defined parameters
    alpha = params["fixed"]["alpha"]
    
    # Access training state (for curriculum learning)
    step = params["internal"]["global_step"]
    
    # Gradually increase constraint strength
    strength = min(1.0, step / 5000)
    
    return V * strength
```
