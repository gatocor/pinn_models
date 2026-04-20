# API Reference

This section provides detailed documentation for all modules in the PINNS package.

## Core Modules

| Module | Description |
|--------|-------------|
| [Domain](domain.md) | Domain definition, sampling, and boundary conditions |
| [Networks](networks.md) | Neural network architectures (FNN, FBPINN) |
| [Problem](problem.md) | Problem definition combining domain, PDE, and parameters |
| [Trainer](trainer.md) | Training loop with visualization (Trainer, ALTrainer) |
| [Functional](functional.md) | Derivative operators for PDEs |

## Quick Reference

### Domain Classes

```python
# Simple rectangular domain
domain = pinns.DomainCubic(xmin=[0, 0], xmax=[1, 1])

# Partitioned domain for FBPINN
partition = pinns.DomainCubicPartition(
    grid_positions=[[0, 0.5, 1], [0, 0.5, 1]],  # 2x2 grid
)
```

### Boundary Conditions

```python
# Dirichlet: u = value
domain.add_dirichlet(boundary=(0, None), value=0.0, component=0, name="left")

# Neumann: ∂u/∂n = value
domain.add_neumann(boundary=(1, None), value=1.0, component=0, name="right")

# Robin: a*u + b*∂u/∂n = value
domain.add_robin(boundary=(None, 0), a=1.0, b=0.5, value=0.0, component=0, name="bottom")

# Pointset: custom points
domain.add_pointset(points, values, component=0, name="data")
```

### Neural Networks

```python
# Feedforward network
net = pinns.FNN([2, 64, 64, 1], activation="tanh")

# FBPINN with domain decomposition
fbpinn = pinns.FBPINN(
    partition=domain,
    base_network=pinns.FNN([2, 32, 1]),
    normalize_input=True,
    unnormalize_output=True
)
```

### Derivatives

```python
# First derivative: ∂u/∂x
u_x = pinns.derivative(V, X, component=0, order=(0,))

# Second derivative: ∂²u/∂x²
u_xx = pinns.derivative(V, X, component=0, order=(0, 0))

# Mixed derivative: ∂²u/(∂x∂y)
u_xy = pinns.derivative(V, X, component=0, order=(0, 1))

# Full gradient: [∂u/∂x, ∂u/∂y]
grad_u = pinns.gradient(V, X, component=0)

# Laplacian: ∂²u/∂x² + ∂²u/∂y²
lap_u = pinns.laplacian(V, X, component=0)
```

### Training

```python
# Standard Trainer
trainer = pinns.Trainer(problem, network)

trainer.compile(
    train_samples={"pde": 1000, "bc1": 100},
    test_samples={"pde": 100, "bc1": 10},
    weights={"pde": 1.0, "bc1": 10.0},
    optimizer="adam",
    learning_rate=1e-3,
    epochs=5000
)

trainer.train()

# ALTrainer (Augmented Lagrangian) for better constraint satisfaction
trainer = pinns.ALTrainer(problem, network)

trainer.compile(
    train_samples={"pde": 1000, "bc1": 100},
    weights={"pde": 1.0, "bc1": 1.0},
    optimizer="adam",
    learning_rate=1e-4,
    lagrange_constraints=['bc1'],  # Adaptive λ on BCs
    lagrange_optimizer='adam',
    lagrange_lr=1e-3,
    epochs=50000
)

trainer.train()
```
