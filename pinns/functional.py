"""
PyTorch functional module - re-exports from backends/torch.

For backward compatibility with direct imports.
"""

from .backends.torch.functional import derivative, gradient, laplacian, divergence

__all__ = ['derivative', 'gradient', 'laplacian', 'divergence']
