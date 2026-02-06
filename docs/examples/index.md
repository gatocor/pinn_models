# Examples

This section contains example notebooks demonstrating how to use PINNS for various physics problems.

## Available Examples

### Basic Examples

| Example | Description |
|---------|-------------|
| [1D Overdamped Oscillator (PINN)](pinn_oscillator.md) | Simple ODE with PINN |
| [1D Overdamped Oscillator (FBPINN)](fbpinn_oscillator.md) | Same problem with FBPINN |
| [Heat Equation](heat_equation.md) | 1D heat diffusion |

### Advanced Examples

| Example | Description |
|---------|-------------|
| [Laser Ablation](laser_ablation.md) | 1+1D multi-physics with symmetry |
| [Hard Constraints](hard_constraints.md) | Encoding BCs in the network |
| [Custom Sampling](custom_sampling.md) | Non-uniform domain sampling |

---

## Quick Start Template

Here's a template for creating your own PINN problem:

```python
import pinns
import torch
import numpy as np

# ============================================
# 1. DEFINE PARAMETERS
# ============================================
# Physical parameters
param1 = 1.0
param2 = 0.1

# Domain bounds
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

# ============================================
# 2. DEFINE DOMAIN
# ============================================
domain = pinns.DomainCubic(
    xmin=[x_min, t_min],
    xmax=[x_max, t_max],
    sampling_method="latin_hypercube"  # Better coverage
)

# ============================================
# 3. ADD BOUNDARY CONDITIONS
# ============================================
# Dirichlet: u(0, t) = 0
domain.add_dirichlet(
    boundary=(0, None),
    value=0.0,
    component=0,
    name="left"
)

# Dirichlet: u(1, t) = 0
domain.add_dirichlet(
    boundary=(1, None),
    value=0.0,
    component=0,
    name="right"
)

# Initial condition: u(x, 0) = sin(pi*x)
domain.add_dirichlet(
    boundary=(None, 0),
    value=lambda x: torch.sin(np.pi * x[:, 0:1]),
    component=0,
    name="initial"
)

# ============================================
# 4. DEFINE PDE RESIDUAL
# ============================================
def pde_residual(X, V, params):
    """
    Define your PDE residual here.
    
    Args:
        X: Inputs (batch, n_dims) - e.g., (x, t)
        V: Outputs (batch, n_outputs) - e.g., (u,)
        params: Dictionary with 'fixed', 'infer', 'internal'
    
    Returns:
        Residual that should be zero
    """
    # Get parameters
    alpha = params["fixed"]["alpha"]
    
    # Get output
    u = V[:, 0:1]
    
    # Compute derivatives
    u_t = pinns.derivative(V, X, component=0, order=(1,))
    u_xx = pinns.derivative(V, X, component=0, order=(0, 0))
    
    # Return residual: u_t - alpha * u_xx = 0
    return u_t - alpha * u_xx

# ============================================
# 5. CREATE PROBLEM
# ============================================
problem = pinns.Problem(
    domain=domain,
    pde_fn=pde_residual,
    input_names=["x", "t"],
    output_names=["u"],
    output_range=(0, 1),
    params={"alpha": param1, "beta": param2}
)

# ============================================
# 6. CREATE NETWORK
# ============================================
# Option A: Simple FNN
network = pinns.FNN(
    layer_sizes=[2, 64, 64, 64, 1],
    activation="tanh"
)

# Option B: FBPINN for multi-scale problems
# domain = pinns.DomainCubicPartition([...])
# network = pinns.FBPINN(domain, base_network, ...)

# ============================================
# 7. TRAIN
# ============================================
trainer = pinns.Trainer(problem, network)

trainer.compile(
    train_samples={
        "pde": 5000,
        "left": 200,
        "right": 200,
        "initial": 500
    },
    weights={
        "pde": 1.0,
        "left": 10.0,
        "right": 10.0,
        "initial": 50.0
    },
    optimizer="adam",
    learning_rate=1e-3,
    epochs=10000,
    print_each=500,
    show_plots=True
)

trainer.train()

# ============================================
# 8. EVALUATE
# ============================================
# Create evaluation grid
x = np.linspace(x_min, x_max, 100)
t = np.linspace(t_min, t_max, 50)
X_grid, T_grid = np.meshgrid(x, t)
points = np.column_stack([X_grid.ravel(), T_grid.ravel()])

# Predict
X_tensor = torch.tensor(points, dtype=torch.float32)
with torch.no_grad():
    u_pred = network(X_tensor).numpy()

# Reshape and plot
U = u_pred.reshape(T_grid.shape)
```

---

## Notebook Files

The example notebooks are located in the `examples/` directory:

- `heat_equation.ipynb` - 1D heat diffusion with analytical comparison
- `1D_overdamped_oscillator_pinn.ipynb` - Simple ODE with standard PINN
- `1D_overdamped_oscillator_fbpinn.ipynb` - Same problem with FBPINN
- `1+1_laser_ablations.ipynb` - Advanced: multi-physics with symmetry and hard constraints
