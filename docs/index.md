# PINNS - Physics-Informed Neural Networks

A PyTorch-based library for Physics-Informed Neural Networks (PINN) and Finite Basis PINN (FBPINN).

## Features

- **PINN and FBPINN**: Standard PINNs and domain-decomposed FBPINNs for improved accuracy
- **Flexible Domain Definition**: Cubic domains with customizable sampling strategies
- **Boundary Conditions**: Dirichlet, Neumann, Robin, and Pointset boundary conditions
- **Automatic Differentiation**: Compute derivatives, gradients, Laplacians using PyTorch autograd
- **Symmetry Support**: Input/output transforms for exploiting problem symmetries
- **Hard Constraints**: Encode initial/boundary conditions directly in the network architecture
- **Visualization**: Built-in training plots with loss curves and solution visualization

## Quick Example

```python
import pinns
import torch

# Define domain
domain = pinns.DomainCubic([0, 0], [1, 1])

# Add boundary conditions
domain.add_dirichlet((0, None), value=0.0, component=0, name="left")
domain.add_dirichlet((1, None), value=1.0, component=0, name="right")

# Define PDE residual (Laplace equation)
def laplace(X, V, params):
    u_xx = pinns.derivative(V, X, 0, (0, 0))
    u_yy = pinns.derivative(V, X, 0, (1, 1))
    return u_xx + u_yy

# Create problem
problem = pinns.Problem(
    domain=domain,
    pde_fn=laplace,
    input_names=['x', 'y'],
    output_names=['u']
)

# Create network and trainer
network = pinns.FNN([2, 64, 64, 1])
trainer = pinns.Trainer(problem, network)

# Train
trainer.compile(
    train_samples={"pde": 1000, "left": 100, "right": 100},
    epochs=5000
)
trainer.train()
```

## Installation

```bash
pip install -e .
```

## Documentation

- [Getting Started](getting_started.md) - Installation and first steps
- [API Reference](api/index.md) - Detailed API documentation
- [Examples](examples/index.md) - Jupyter notebook tutorials

## Package Contents

| Module | Description |
|--------|-------------|
| `pinns.DomainCubic` | Rectangular domain definition |
| `pinns.DomainCubicPartition` | Partitioned domain for FBPINN |
| `pinns.FNN` | Fully-connected neural network |
| `pinns.FBPINN` | Finite Basis PINN with domain decomposition |
| `pinns.Problem` | Problem definition (domain + PDE + BCs) |
| `pinns.Trainer` | Training loop with visualization |
| `pinns.derivative` | Compute partial derivatives |
| `pinns.gradient` | Compute full gradient |
| `pinns.laplacian` | Compute Laplacian |
| `pinns.divergence` | Compute divergence |
