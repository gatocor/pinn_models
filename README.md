# PINNS - Physics-Informed Neural Networks

A PyTorch-based library for Physics-Informed Neural Networks (PINN) and Finite Basis PINN (FBPINN).

## Installation

### From source (development mode)

```bash
cd pinn_models
pip install -e .
```

### With optional dependencies

```bash
# Install with development tools
pip install -e ".[dev]"

# Install with example dependencies (matplotlib, jupyter)
pip install -e ".[examples]"

# Install everything
pip install -e ".[all]"
```

## Quick Start

Solve the 1D heat equation: $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$

```python
import pinns
import numpy as np

# 1. Define the domain
domain = pinns.DomainCubic(xmin=[0, 0], xmax=[1, 1])

# 2. Add boundary conditions
domain.add_dirichlet((0, None), value=0.0, component=0, name="left")      # u(0,t) = 0
domain.add_dirichlet((1, None), value=0.0, component=0, name="right")     # u(1,t) = 0
domain.add_dirichlet((None, 0), value=lambda x: np.sin(np.pi * x[:, 0:1]), 
                     component=0, name="initial")  # u(x,0) = sin(πx)

# 3. Define the PDE residual
def heat_equation(X, V, params):
    alpha = params["fixed"]["alpha"]
    u_t = pinns.derivative(V, X, component=0, order=(1,))     # ∂u/∂t
    u_xx = pinns.derivative(V, X, component=0, order=(0, 0))  # ∂²u/∂x²
    return u_t - alpha * u_xx

# 4. Create the problem
problem = pinns.Problem(
    domain=domain,
    pde_fn=heat_equation,
    input_names=["x", "t"],
    output_names=["u"],
    params={"alpha": 0.01}
)

# 5. Create network and trainer
network = pinns.FNN([2, 64, 64, 64, 1], activation="tanh")
trainer = pinns.Trainer(problem, network)

# 6. Train
trainer.compile(
    train_samples={"pde": 5000, "left": 200, "right": 200, "initial": 500},
    weights={"pde": 1.0, "left": 10.0, "right": 10.0, "initial": 50.0},
    optimizer="adam",
    learning_rate=1e-3,
    epochs=10000,
    print_each=500,
    show_plots=True
)
trainer.train()
```

## Features

### Domain Decomposition
- `DomainCubicPartition`: Define regular grid domain partitions
- Flexible subdomain positioning and widths
- Interior and boundary sampling (uniform or per-partition)

### Networks
- `FNN`: Fully-connected network with configurable layers and activation
- `FBPINN`: Finite Basis PINN with domain decomposition and window functions
- Automatic input scaling and output unnormalization

### Boundary Conditions
- `DirichletBC`: u(x) = value
- `NeumannBC`: du/dn = value
- `RobinBC`: α*u + β*du/dn = value
- `PointsetBC`: Data-driven constraints at specific points

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:
- Solving ODEs and PDEs
- Domain decomposition strategies
- Inverse problems

## License

MIT License
