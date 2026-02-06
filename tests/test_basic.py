"""Basic tests for pinns package."""

import numpy as np
import torch
import pytest


def test_import():
    """Test that the package can be imported."""
    import pinns
    assert hasattr(pinns, '__version__')
    assert hasattr(pinns, 'DomainCubicPartition')
    assert hasattr(pinns, 'FNN')
    assert hasattr(pinns, 'FBPINN')


def test_domain_partition():
    """Test DomainCubicPartition creation."""
    from pinns import DomainCubicPartition
    
    domain = DomainCubicPartition(
        positions=[np.linspace(0, 1, 5), np.linspace(0, 2, 3)],
        widths=[np.full(5, 0.3), np.full(3, 0.8)]
    )
    
    assert domain.n_dims == 2
    assert domain.n_subdomains == 15  # 5 * 3
    assert len(domain) == 15


def test_vanilla_network():
    """Test FNN creation and forward pass."""
    from pinns import FNN
    
    net = FNN([2, 32, 32, 1], activation='tanh')
    x = torch.randn(100, 2)
    y = net(x)
    
    assert y.shape == (100, 1)


def test_fbpinn():
    """Test FBPINN creation and forward pass."""
    from pinns import DomainCubicPartition, FNN, FBPINN
    
    domain = DomainCubicPartition(
        positions=[np.linspace(0, 1, 3)],
        widths=[np.full(3, 0.5)]
    )
    
    network = FNN([1, 16, 1])
    fbpinn = FBPINN(domain, network, sigma=0.1)
    
    x = torch.randn(50, 1)
    y = fbpinn(x)
    
    assert y.shape == (50, 1)


def test_boundary_conditions():
    """Test boundary condition creation."""
    from pinns import DirichletBC, NeumannBC, BoundaryConditions
    
    bc1 = DirichletBC(boundary=(0, None), value=0.0, component=0)
    bc2 = NeumannBC(boundary=(None, 1), value=1.0, component=0)
    
    bcs = BoundaryConditions()
    bcs.add(bc1)
    bcs.add(bc2)
    
    assert len(bcs) == 2
    assert len(bcs.dirichlet) == 1
    assert len(bcs.neumann) == 1


def test_sampling():
    """Test domain sampling."""
    from pinns import DomainCubicPartition
    
    domain = DomainCubicPartition(
        positions=[np.linspace(0, 1, 5), np.linspace(0, 1, 5)],
        widths=[np.full(5, 0.3), np.full(5, 0.3)]
    )
    
    # Interior sampling
    points = domain.sample_interior(1000, mode='uniform')
    assert points.shape == (1000, 2)
    
    # Boundary sampling
    boundary_points = domain.sample_boundary(100, boundary_dim=0, boundary_side='lower')
    assert boundary_points.shape == (100, 2)
