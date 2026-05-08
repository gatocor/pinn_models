"""
PINNS - Physics-Informed Neural Networks

A JAX-based library for Physics-Informed Neural Networks (PINN) 
and Finite Basis PINN (FBPINN).

Default backend: 'jax' (JAX/Flax/Optax)

Usage:
    import pinns

Or override via environment variable:
    import os
    os.environ['PINNS_BACKEND'] = 'torch'
    import pinns
"""

__version__ = "0.1.0"

import os
import sys

# Backend selection - default from environment or 'jax'
_BACKEND = os.environ.get('PINNS_BACKEND', 'jax').lower()

# Storage for current backend classes
_backend_classes = {}

# Domain and Problem are backend-agnostic
from .domain import DomainCubic, DomainMesh, SubdomainInfo, bump
from .strategies import StrategyUnique, StrategyFB, StrategyX, StrategyStep, register_interface_loss
from .problem import Problem, ProblemStrong
from .problem_weak import ProblemWeak
from . import meshes
from . import base_models
from . import layers
from . import network
from .network import Network, NetworkLoss

# JAX-only mesh networks (always available if JAX is installed)
try:
    from .backends.jax.gnn_network import GNNMeshNetwork, GNNFeatures
    from .backends.jax.alpha_pinn_network import AlphaPINN, AlphaPINNNetwork, LaplacianFeatures
except ImportError:
    pass

# Learning rate schedulers (backend-agnostic)
from .backends import LRScheduler, ExponentialDecay, ReduceLROnPlateau


def _load_backend(name):
    """Load backend-specific classes."""
    name = name.lower()
    if name not in ('jax', 'torch'):
        raise ValueError(f"Unknown backend: {name}. Choose 'jax' or 'torch'")
    
    if name == 'jax':
        try:
            from .backends.jax import (
                FNN, WFFNN, PirateNet, ResNet, FBPINN, FourierFeatures, DenseRWF,
                derivative, gradient, laplacian, divergence,
                Trainer
            )
        except ImportError as e:
            raise ImportError(
                f"JAX backend requested but JAX/Flax/Optax not installed: {e}\n"
                "Install with: pip install jax jaxlib flax optax"
            )
    else:
        from .backends.torch import (
            FNN, WFFNN, PirateNet, FBPINN, FourierFeatures, LinearRWF,
            derivative, gradient, laplacian, divergence,
            Trainer
        )
        ResNet = None
    
    # Use backend-specific RWF layer
    RWFLayer = DenseRWF if name == 'jax' else LinearRWF
    
    return {
        'FNN': FNN,
        'WFFNN': WFFNN,
        'PirateNet': PirateNet,
        'ResNet': ResNet,
        'FBPINN': FBPINN,
        'FourierFeatures': FourierFeatures,
        'RWFLayer': RWFLayer,
        'derivative': derivative,
        'gradient': gradient,
        'laplacian': laplacian,
        'divergence': divergence,
        'Trainer': Trainer,
    }


def use_backend(name):
    """
    Set the active backend for PINNS.
    
    Args:
        name: Backend name, either 'jax' or 'torch'
    
    Example:
        import pinns
        pinns.use_backend('jax')
        trainer = pinns.Trainer(problem, network)
    """
    global _BACKEND, _backend_classes
    
    name = name.lower()
    _BACKEND = name
    _backend_classes = _load_backend(name)
    
    # Update module namespace
    module = sys.modules[__name__]
    for key, value in _backend_classes.items():
        setattr(module, key, value)
    
    # Also update BACKEND property for backwards compatibility
    setattr(module, 'BACKEND', name)
    
    print(f"pinns: Using {name} backend")


def get_backend():
    """Return the current backend name."""
    return _BACKEND


# Initialize with default backend
_backend_classes = _load_backend(_BACKEND)
BACKEND = _BACKEND

# Export backend-specific classes at module level
FNN = _backend_classes['FNN']
WFFNN = _backend_classes['WFFNN']
PirateNet = _backend_classes['PirateNet']
ResNet = _backend_classes['ResNet']
FBPINN = _backend_classes['FBPINN']
FourierFeatures = _backend_classes['FourierFeatures']
RWFLayer = _backend_classes['RWFLayer']
derivative = _backend_classes['derivative']
gradient = _backend_classes['gradient']
laplacian = _backend_classes['laplacian']
divergence = _backend_classes['divergence']
Trainer = _backend_classes['Trainer']

__all__ = [
    # Version
    "__version__",
    # Backend
    "BACKEND",
    "use_backend",
    "get_backend",
    # Meshes
    "meshes",
    # Base network models
    "base_models",
    # Domain (backend-agnostic)
    "DomainCubic",
    "DomainMesh",
    "SubdomainInfo",
    "bump",
    # Networks
    "FNN",
    "WFFNN",
    "PirateNet",
    "ResNet",
    "FBPINN",
    "FourierFeatures",
    "RWFLayer",
    # Composable network
    "Network",
    "NetworkLoss",
    # Strategies
    "StrategyUnique",
    "StrategyFB",
    "StrategyX",
    "StrategyStep",
    "register_interface_loss",
    # Problems
    "Problem",
    "ProblemStrong",
    "ProblemWeak",
    # JAX-only mesh GNN
    "GNNMeshNetwork",
    "GNNFeatures",
    "AlphaPINN",
    "AlphaPINNNetwork",
    "LaplacianFeatures",
    # Functional
    "derivative",
    "gradient",
    "laplacian",
    "divergence",
    # Training
    "Trainer",
    # Learning rate schedulers
    "LRScheduler",
    "ExponentialDecay",
    "ReduceLROnPlateau",
]