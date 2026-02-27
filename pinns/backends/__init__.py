"""
Backend selection for PINNS.

Set the backend via environment variable PINNS_BACKEND='torch' or 'jax'.
Default is 'torch'.
"""

import os

BACKEND = os.environ.get('PINNS_BACKEND', 'torch').lower()

# Import learning rate schedulers (backend-agnostic)
from .base_trainer import LRScheduler, ExponentialDecay, ReduceLROnPlateau

def get_backend():
    """Return the current backend name."""
    return BACKEND

def set_backend(backend: str):
    """Set the backend (must be called before importing pinns modules)."""
    global BACKEND
    if backend.lower() not in ('torch', 'jax'):
        raise ValueError(f"Unknown backend: {backend}. Use 'torch' or 'jax'.")
    BACKEND = backend.lower()
