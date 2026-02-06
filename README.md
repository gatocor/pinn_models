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

```python
import numpy as np
import torch
from pinns.domain import DomainCubicPartition
from pinns.networks import FNN, FBPINN

# 1. Define the domain partition (bounds derived from grid_positions)
partition = DomainCubicPartition(
    grid_positions=[np.linspace(0, 1, 10), np.linspace(0, 1, 5)],
    overlap=0.5  # 50% overlap between subdomains
)

# 2. Add boundary conditions directly to the domain
partition.add_dirichlet(boundary=(0, None), value=0.0, component=0, name="left_bc")
partition.add_dirichlet(boundary=(1, None), value=1.0, component=0, name="right_bc")
partition.add_neumann(boundary=(None, 0), value=0.0, component=0, name="bottom_flux")

# 3. Create the FBPINN model
network = FNN([2, 64, 64, 1], activation='tanh')
model = FBPINN(
    domain=partition,
    networks=network
)

# 4. Sample points
x_interior = partition.sample_interior_torch(10000, mode='uniform')
x_boundary = partition.sample_boundary_torch(1000, boundary_dim=0, boundary_side=0)

# 5. Forward pass
y = model(x_interior)
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
