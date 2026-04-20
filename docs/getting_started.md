# Getting Started

This guide will walk you through installing PINNS and solving your first physics-informed neural network problem.

## Installation

Clone the repository and install in development mode:

```bash
git clone <repository-url>
cd pinn_models
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.9
- NumPy
- Matplotlib
- SciPy

## Your First PINN

Let's solve the 1D heat equation with homogeneous boundary conditions:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

with $u(0, t) = u(1, t) = 0$ and $u(x, 0) = \sin(\pi x)$.

### Step 1: Import and Define Parameters

```python
import pinns
import torch
import numpy as np

# Physical parameters
alpha = 0.01
```

### Step 2: Define the Domain

```python
# 2D domain: x in [0, 1], t in [0, 1]
domain = pinns.DomainCubic(
    xmin=[0.0, 0.0],
    xmax=[1.0, 1.0]
)
```

### Step 3: Add Boundary Conditions

```python
# u(0, t) = 0 (left boundary)
domain.add_dirichlet(
    boundary=(0, None),  # x=xmin, all t
    value=0.0,
    component=0,
    name="left"
)

# u(1, t) = 0 (right boundary)
domain.add_dirichlet(
    boundary=(1, None),  # x=xmax, all t
    value=0.0,
    component=0,
    name="right"
)

# u(x, 0) = sin(pi*x) (initial condition)
domain.add_dirichlet(
    boundary=(None, 0),  # all x, t=tmin
    value=lambda x: torch.sin(np.pi * x[:, 0:1]),
    component=0,
    name="initial"
)
```

### Step 4: Define the PDE Residual

```python
def heat_equation(X, V, params):
    """
    Heat equation residual.
    
    Args:
        X: Input tensor (batch, 2) with columns [x, t]
        V: Network output (batch, 1) with column [u]
        params: Dictionary with 'fixed', 'infer', 'internal' keys
    
    Returns:
        Residual tensor (should be zero when PDE is satisfied)
    """
    alpha = params["fixed"]["alpha"]
    
    # Compute derivatives
    u_t = pinns.derivative(V, X, component=0, order=(1,))   # ∂u/∂t
    u_xx = pinns.derivative(V, X, component=0, order=(0, 0)) # ∂²u/∂x²
    
    # Heat equation: u_t = alpha * u_xx
    return u_t - alpha * u_xx
```

### Step 5: Create the Problem

```python
problem = pinns.Problem(
    domain=domain,
    pde_fn=heat_equation,
    input_names=["x", "t"],
    output_names=["u"],
    params={"alpha": alpha}
)
```

### Step 6: Create Network and Trainer

```python
# Create a feedforward network
network = pinns.FNN(
    layer_sizes=[2, 64, 64, 64, 1],
    activation="tanh"
)

# Create trainer
trainer = pinns.Trainer(problem, network)
```

### Step 7: Compile and Train

```python
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
        "initial": 10.0
    },
    optimizer="adam",
    learning_rate=1e-3,
    epochs=10000,
    print_each=500,
    show_plots=True
)

trainer.train()
```

### Step 8: Evaluate the Solution

```python
# Create evaluation grid
x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 50)
X, T = np.meshgrid(x, t)
points = np.column_stack([X.ravel(), T.ravel()])

# Convert to tensor and evaluate
X_tensor = torch.tensor(points, dtype=torch.float32, requires_grad=False)
with torch.no_grad():
    u_pred = network(X_tensor).numpy()

# Reshape for plotting
U = u_pred.reshape(T.shape)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.pcolormesh(X, T, U, shading='auto')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Heat Equation Solution')
plt.show()
```

## Next Steps

- Learn about [FBPINN](examples/fbpinn.md) for complex multi-scale problems
- Try [ALTrainer](api/trainer.md#altrainer-augmented-lagrangian) for better constraint satisfaction
- Explore [domain sampling](api/domain.md) strategies
- Implement [hard constraints](examples/hard_constraints.md) using output transforms
- Check out more [examples](examples/index.md)

## Advanced: Using ALTrainer

If your boundary conditions are not being satisfied well with the standard `Trainer`, consider using `ALTrainer` (Augmented Lagrangian method):

```python
# Replace Trainer with ALTrainer
trainer = pinns.ALTrainer(problem, network)

trainer.compile(
    train_samples={
        "pde": 5000,
        "left": 200,
        "right": 200,
        "initial": 500
    },
    weights={
        "pde": 1.0,
        "left": 1.0,
        "right": 1.0,
        "initial": 1.0
    },
    optimizer="adam",
    learning_rate=1e-4,
    lagrange_constraints=['left', 'right', 'initial'],  # Adaptive λ on BCs
    lagrange_optimizer='adam',
    lagrange_lr=1e-3,
    epochs=50000,
    print_each=1000,
    show_plots=True
)

trainer.train()
```

ALTrainer automatically adjusts constraint weights during training, helping to balance PDE residuals with boundary conditions.
