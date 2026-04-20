"""
PyTorch backend for PINNS.
"""

from .functional import derivative, gradient, laplacian, divergence
from .networks import FNN, WFFNN, PirateNet, FBPINN, FourierFeatures, LinearRWF
from .trainer import Trainer, ALTrainer

__all__ = [
    'derivative', 'gradient', 'laplacian', 'divergence',
    'FNN', 'WFFNN', 'PirateNet', 'FBPINN', 'FourierFeatures', 'LinearRWF',
    'Trainer', 'ALTrainer',
]
