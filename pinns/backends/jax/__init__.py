"""
JAX backend for PINNS.

API compatible with PyTorch backend - same function signatures.

Usage:
    # Set backend before importing
    import os
    os.environ['PINNS_BACKEND'] = 'jax'
    import pinns
    
    # Then use same API as PyTorch:
    trainer = pinns.Trainer(problem, model)
    trainer.compile(train_samples={'pde': 1000}, epochs=10000)
    trainer.train()
"""

from .functional import (
    derivative, 
    gradient, 
    laplacian,
    divergence,
    set_context,
    clear_context,
    make_derivative_fn,
    DifferentialOperators
)

from .networks import (
    FNN,
    WFFNN,
    PirateNet,
    FBPINN,
    FBPINNModule,
    FourierFeatures,
    DenseRWF,
    create_fnn,
    create_fbpinn,
    get_activation
)

from .trainer import Trainer, ALTrainer

__all__ = [
    # Functional (same API as PyTorch)
    'derivative',
    'gradient', 
    'laplacian',
    'divergence',
    'set_context',
    'clear_context',
    'make_derivative_fn',
    'DifferentialOperators',
    # Networks
    'FNN',
    'WFFNN',
    'PirateNet',
    'FBPINN',
    'FBPINNModule',
    'FourierFeatures',
    'DenseRWF',
    'create_fnn',
    'create_fbpinn',
    'get_activation',
    # Training (same API as PyTorch)
    'Trainer',
    'ALTrainer',
]
