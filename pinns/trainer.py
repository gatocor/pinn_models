"""
PyTorch trainer module - re-exports from backends/torch.

For backward compatibility with direct imports.
"""

from .backends.torch.trainer import Trainer

__all__ = ['Trainer']
