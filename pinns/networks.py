"""
PyTorch networks module - re-exports from backends/torch.

For backward compatibility with direct imports.
"""

from .backends.torch.networks import FNN, FBPINN

__all__ = ['FNN', 'FBPINN']
