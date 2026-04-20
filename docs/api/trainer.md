# Trainer Module

The Trainer class handles the training loop, sampling, loss computation, and visualization.

## Trainer Class

```python
class Trainer:
    def __init__(self, problem, network, device=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `problem` | Problem | Problem definition (domain, PDE, BCs) |
| `network` | nn.Module | Neural network (FNN or FBPINN) |
| `device` | str | Device: 'cpu', 'cuda', 'mps' (auto-detected if None) |

**Example:**

```python
import pinns

problem = pinns.Problem(...)
network = pinns.FNN([2, 64, 64, 1])

trainer = pinns.Trainer(problem, network)
```

---

## compile()

Configure training parameters before calling `train()`.

```python
trainer.compile(
    train_samples,
    test_samples=None,
    weights=None,
    optimizer="adam",
    learning_rate=1e-3,
    epochs=1000,
    print_each=100,
    show_plots=False,
    save_plots=None,
    show_subdomains=False,
    show_sampling_points=False,
    plot_regions=None,
    plot_n_points=100,
    profile=False
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_samples` | dict or list | Number of training samples per loss term |
| `test_samples` | dict or list | Number of test samples per loss term |
| `weights` | dict or list | Loss weights per term |
| `optimizer` | str | Optimizer: 'adam', 'lbfgs', 'sgd' |
| `learning_rate` | float | Learning rate |
| `epochs` | int | Number of training epochs |
| `print_each` | int | Print/plot every N epochs |
| `show_plots` | bool | Display live training plots |
| `save_plots` | str | Directory to save plots |
| `show_subdomains` | bool | Show subdomain boundaries in plots |
| `show_sampling_points` | bool | Show sampling points in plots |
| `plot_regions` | list | Regions to plot `[((xmin, xmax), (tmin, tmax)), ...]` |
| `plot_n_points` | int | Resolution for plots |
| `profile` | bool | Enable profiling |

### Sample Specification

Samples can be specified as a dictionary with names or a list:

```python
# Dictionary format (recommended)
trainer.compile(
    train_samples={
        "pde": 5000,        # Interior PDE samples
        "left": 200,        # Named BC samples
        "right": 200,
        "initial": 500
    },
    weights={
        "pde": 1.0,
        "left": 10.0,
        "right": 10.0,
        "initial": 50.0
    }
)

# List format (order: pde, bc1, bc2, ...)
trainer.compile(
    train_samples=[5000, 200, 200, 500],
    weights=[1.0, 10.0, 10.0, 50.0]
)
```

### Optimizers

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| `"adam"` | Adaptive moment estimation | Initial training |
| `"lbfgs"` | Limited-memory BFGS | Fine-tuning, final epochs |
| `"sgd"` | Stochastic gradient descent | Simple problems |

**Two-phase training:**

```python
# Phase 1: Adam for exploration
trainer.compile(optimizer="adam", learning_rate=1e-3, epochs=5000)
trainer.train()

# Phase 2: L-BFGS for refinement
trainer.compile(optimizer="lbfgs", epochs=500)
trainer.train()
```

---

## train()

Execute the training loop.

```python
trainer.train()
```

**Returns:** `None` (updates `trainer.history` in-place)

**Features:**
- Automatic resampling each epoch
- Live loss visualization (if `show_plots=True`)
- History tracking
- Progress printing

---

## History

Training history is stored in `trainer.history`:

```python
trainer.history = {
    'epoch': [],        # Epoch numbers
    'loss': [],         # Total loss
    'loss_pde': [],     # PDE loss (list of per-equation losses)
    'loss_bcs': [],     # BC losses (list of per-BC losses)
    'test_loss': [],    # Test loss
    'solution_error': [] # Error vs analytical solution (if provided)
}
```

**Plotting history:**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.semilogy(trainer.history['epoch'], trainer.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

---

## Plot Regions

Focus visualization on specific regions of interest:

```python
trainer.compile(
    ...,
    show_plots=True,
    plot_regions=[
        ((-0.5, 0.5), (0, 0.1)),   # x in [-0.5, 0.5], t in [0, 0.1]
        ((0, 1), None),             # x in [0, 1], full t range
    ],
    plot_n_points=200  # Higher resolution
)
```

---

## Visualization Options

### show_subdomains

When using FBPINN, visualize subdomain boundaries:

```python
trainer.compile(
    ...,
    show_plots=True,
    show_subdomains=True  # Draw subdomain grid
)
```

### show_sampling_points

Overlay training points on the solution plot:

```python
trainer.compile(
    ...,
    show_plots=True,
    show_sampling_points=True
)
```

### save_plots

Save plots to a directory (useful for long training runs):

```python
trainer.compile(
    ...,
    save_plots="./training_plots/",
    print_each=1000  # Save every 1000 epochs
)
```

---

## Complete Training Example

```python
import pinns
import torch
import numpy as np

# Define problem
domain = pinns.DomainCubic([0, 0], [1, 1])
domain.add_dirichlet((0, None), 0.0, 0, "left")
domain.add_dirichlet((1, None), 0.0, 0, "right")
domain.add_dirichlet((None, 0), lambda x: torch.sin(np.pi * x[:, 0:1]), 0, "initial")

def heat_eq(X, V, params):
    return pinns.derivative(V, X, 0, (1,)) - 0.01 * pinns.derivative(V, X, 0, (0, 0))

problem = pinns.Problem(
    domain=domain,
    pde_fn=heat_eq,
    input_names=["x", "t"],
    output_names=["u"]
)

# Create network
network = pinns.FNN([2, 64, 64, 64, 1])

# Create trainer
trainer = pinns.Trainer(problem, network)

# Phase 1: Adam training
trainer.compile(
    train_samples={"pde": 5000, "left": 200, "right": 200, "initial": 500},
    test_samples={"pde": 500, "left": 50, "right": 50, "initial": 100},
    weights={"pde": 1.0, "left": 10.0, "right": 10.0, "initial": 50.0},
    optimizer="adam",
    learning_rate=1e-3,
    epochs=10000,
    print_each=500,
    show_plots=True
)

trainer.train()

# Phase 2: L-BFGS refinement
trainer.compile(
    optimizer="lbfgs",
    epochs=500,
    print_each=50
)

trainer.train()

# Check final loss
print(f"Final loss: {trainer.history['loss'][-1]:.2e}")
```

---

## Tips for Better Training

### 1. Weight Tuning

Balance PDE and BC losses:

```python
# If BCs not satisfied, increase their weight
weights = {
    "pde": 1.0,
    "boundary": 100.0  # High weight for essential BCs
}
```

### 2. Learning Rate Scheduling

Use decreasing learning rates:

```python
# Manual scheduling
for lr in [1e-3, 1e-4, 1e-5]:
    trainer.compile(learning_rate=lr, epochs=3000)
    trainer.train()
```

### 3. Residual-Based Resampling

Focus sampling where residuals are high (built-in for future versions).

### 4. Network Architecture

- Start small, increase if needed
- `tanh` activation works well for smooth solutions
- Use `silu` or `gelu` for sharper features

### 5. Check Gradients

If loss isn't decreasing:
- Check `X.requires_grad = True` in PDE function
- Ensure derivatives are correct
- Verify BC specifications

---

## Debugging

### Check Loss Components

```python
# Print individual losses
for i, (epoch, loss, pde_loss, bc_losses) in enumerate(zip(
    trainer.history['epoch'],
    trainer.history['loss'],
    trainer.history['loss_pde'],
    trainer.history['loss_bcs']
)):
    if i % 100 == 0:
        print(f"Epoch {epoch}: total={loss:.2e}, pde={pde_loss:.2e}, bcs={bc_losses}")
```

### Evaluate at Specific Points

```python
import torch

# Create test points
x_test = torch.tensor([[0.5, 0.0], [0.5, 0.5], [0.5, 1.0]], requires_grad=True)

# Evaluate network
with torch.no_grad():
    u_pred = trainer.network(x_test)
    
print("Predictions:", u_pred)
```

### Check Residuals

```python
# Compute PDE residual at test points
x_test = torch.tensor([[0.5, 0.5]], requires_grad=True)
y_pred = trainer.network(x_test)
params = trainer._build_params()
residual = trainer.problem.pde_fn(x_test, y_pred, params)
print("Residual:", residual.item())
```

---

# ALTrainer (Augmented Lagrangian)

The `ALTrainer` class implements the Augmented Lagrangian method for physics-informed neural networks (AL-PINNs). This approach improves constraint satisfaction by introducing adaptive Lagrange multipliers that automatically balance PDE residuals and boundary conditions.

## When to Use ALTrainer

ALTrainer is particularly beneficial for:

- **Stiff boundary conditions**: When BCs are hard to satisfy with standard weighting
- **Multi-scale problems**: Where different constraints have vastly different magnitudes
- **Long-time simulations**: Where initial conditions must be preserved accurately
- **Inverse problems**: Where data constraints must be balanced with physics

## ALTrainer Class

```python
class ALTrainer:
    def __init__(self, problem, network, device=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `problem` | Problem | Problem definition (domain, PDE, BCs) |
| `network` | nn.Module | Neural network (FNN or FBPINN) |
| `device` | str | Device: 'cpu', 'cuda', 'mps' (auto-detected if None) |

**Example:**

```python
import pinns

problem = pinns.Problem(...)
network = pinns.FNN([2, 64, 64, 1])

trainer = pinns.ALTrainer(problem, network)
```

---

## ALTrainer.compile()

Configure training with Augmented Lagrangian parameters.

```python
trainer.compile(
    train_samples,
    test_samples=None,
    weights=None,
    optimizer="adam",
    learning_rate=1e-3,
    lagrange_constraints=None,
    lagrange_optimizer="adam",
    lagrange_lr=1e-3,
    lagrange_max=1e6,
    epochs=1000,
    print_each=100,
    show_plots=False,
    ...
)
```

### ALTrainer-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lagrange_constraints` | list or None | None | Which constraints get λ multipliers |
| `lagrange_optimizer` | str | "adam" | Optimizer for λ updates: 'adam', 'sgd', 'none' |
| `lagrange_lr` | float | 1e-3 | Learning rate for λ updates |
| `lagrange_max` | float | 1e6 | Maximum magnitude for λ clipping |

### lagrange_constraints

Specify which constraints should have adaptive Lagrange multipliers:

```python
# Apply λ only to boundary conditions (recommended)
trainer.compile(
    ...,
    lagrange_constraints=['initial', 'left', 'right'],
)

# Apply λ to all constraints including PDE
trainer.compile(
    ...,
    lagrange_constraints=None,  # or ['pde', 'initial', 'left', 'right']
)
```

**Best Practice**: Apply Lagrange multipliers to boundary/initial conditions only, keeping the PDE term with fixed weight. This follows the original AL-PINNs paper methodology.

### lagrange_optimizer

Choose how Lagrange multipliers are updated:

| Option | Description |
|--------|-------------|
| `"adam"` | Adaptive updates (recommended, matches AL-PINNs paper) |
| `"sgd"` | Simple gradient ascent |
| `"none"` | Fixed multipliers (no adaptation) |

---

## ALTrainer.train()

Execute the Augmented Lagrangian training loop.

```python
trainer.train()
```

**Training Dynamics:**

Each epoch performs:
1. **Primal step**: Update network parameters to minimize augmented loss
2. **Dual step**: Update Lagrange multipliers via gradient ascent

The augmented Lagrangian loss is:

$$L = \sum_i \left[ \beta_i \cdot \text{MSE}(g_i) + \lambda_i \cdot g_i \right]$$

where:
- $g_i$ = constraint residual
- $\beta_i$ = penalty weight (from `weights`)
- $\lambda_i$ = adaptive Lagrange multiplier

---

## ALTrainer History

ALTrainer tracks additional metrics:

```python
trainer.history = {
    'epoch': [],           # Epoch numbers
    'loss': [],            # MSE loss (for comparison)
    'al_loss': [],         # Augmented Lagrangian loss
    'mse_loss': [],        # Pure MSE loss
    'loss_pde': [],        # PDE MSE loss
    'loss_bcs': [],        # BC MSE losses
    'al_pde_penalty': [],  # PDE penalty term (β·MSE)
    'al_pde_lagrangian': [], # PDE Lagrangian term (λ·g)
    'al_bcs_penalty': [],  # BC penalty terms
    'al_bcs_lagrangian': [], # BC Lagrangian terms
    'test_loss': [],
    'solution_error': []
}
```

---

## Complete ALTrainer Example

```python
import pinns
import numpy as np

# Define problem (viscous Burgers equation)
nu = 0.01 / np.pi
domain = pinns.DomainCubic([0, -1], [1, 1])

def initial_condition(X):
    return -np.sin(np.pi * X[:, 1:2])

domain.add_dirichlet((0, None), value=initial_condition, component=0, name="initial")
domain.add_dirichlet((None, 0), value=0.0, component=0, name="left")
domain.add_dirichlet((None, 1), value=0.0, component=0, name="right")

def burgers_residual(X, U, params, derivative=None):
    if derivative is None:
        derivative = pinns.derivative
    nu = params['fixed']['nu']
    u = U[:, 0:1]
    u_t = derivative(u, X, 0, (0,))
    u_x = derivative(u, X, 0, (1,))
    u_xx = derivative(u, X, 0, (1, 1))
    return u_t + u * u_x - nu * u_xx

problem = pinns.Problem(
    domain,
    burgers_residual,
    input_names=["t", "x"],
    output_names=["u"],
    params={'nu': nu},
)

# Create network and ALTrainer
network = pinns.FNN([2, 256, 256, 1], activation="tanh")
trainer = pinns.ALTrainer(problem, network)

# Compile with AL settings
trainer.compile(
    train_samples={"pde": 2601, "initial": 100, "left": 50, "right": 50},
    test_samples={"pde": 1000},
    weights={"pde": 1.0, "initial": 1.0, "left": 1.0, "right": 1.0},
    optimizer="adam",
    learning_rate=1e-4,
    lagrange_constraints=['initial', 'left', 'right'],  # λ on BCs only
    lagrange_optimizer='adam',
    lagrange_lr=1e-3,
    epochs=50000,
    print_each=1000,
    show_plots=True,
)

trainer.train()
```

---

## ALTrainer vs Trainer Comparison

| Feature | Trainer | ALTrainer |
|---------|---------|-----------|
| Loss function | Weighted MSE | Augmented Lagrangian |
| Constraint balance | Manual weights | Adaptive λ multipliers |
| Convergence | May stall on stiff BCs | Better constraint satisfaction |
| Complexity | Simpler | Additional hyperparameters |
| Best for | Simple problems | Multi-scale, stiff BCs |

**Recommendation**: Start with `Trainer`. Switch to `ALTrainer` if:
- Boundary conditions remain unsatisfied
- Loss plateaus with high BC residuals
- Different constraints have very different scales
