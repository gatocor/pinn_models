"""
PINNS - Physics-Informed Neural Networks

A multi-backend library for Physics-Informed Neural Networks (PINN) 
and Finite Basis PINN (FBPINN).

Supported backends:
- 'torch' (default): PyTorch-based implementation
- 'jax': JAX/Flax/Optax-based implementation

Usage:
    import pinns
    pinns.use_backend('jax')  # or 'torch'
    
Or via environment variable (before import):
    import os
    os.environ['PINNS_BACKEND'] = 'jax'
    import pinns

Or use backend-specific imports:
    from pinns.backends.torch import FNN, FBPINN, Trainer
    from pinns.backends.jax import FNN, FBPINN, Trainer
"""

__version__ = "0.1.0"

import os
import sys

# Backend selection - default from environment or 'torch'
_BACKEND = os.environ.get('PINNS_BACKEND', 'torch').lower()

# Storage for current backend classes
_backend_classes = {}

# Domain and Problem are backend-agnostic
from .domain import DomainCubic, DomainCubicPartition, SubdomainInfo, bump
from .problem import Problem


def _load_backend(name):
    """Load backend-specific classes."""
    name = name.lower()
    if name not in ('jax', 'torch'):
        raise ValueError(f"Unknown backend: {name}. Choose 'jax' or 'torch'")
    
    if name == 'jax':
        try:
            from .backends.jax import (
                FNN, FBPINN, 
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
            FNN, FBPINN,
            derivative, gradient, laplacian, divergence,
            Trainer
        )
    
    return {
        'FNN': FNN,
        'FBPINN': FBPINN,
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
FBPINN = _backend_classes['FBPINN']
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
    # Domain (backend-agnostic)
    "DomainCubic",
    "DomainCubicPartition",
    "SubdomainInfo",
    "bump",
    # Networks
    "FNN",
    "FBPINN",
    # Problems
    "Problem",
    # Functional
    "derivative",
    "gradient",
    "laplacian",
    "divergence",
    # Training
    "Trainer",
]