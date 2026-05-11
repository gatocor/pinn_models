"""
Shared utilities for layer modules: activation lookup and DenseRWF layer.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable


def get_activation(name: str) -> Callable:
    """Return a Flax/JAX activation function by name."""
    activations = {
        'relu': nn.relu,
        'tanh': nn.tanh,
        'sigmoid': nn.sigmoid,
        'gelu': nn.gelu,
        'silu': nn.silu,
        'leaky_relu': nn.leaky_relu,
        'elu': nn.elu,
        'softplus': nn.softplus,
    }
    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Available: {list(activations.keys())}"
        )
    return activations[name.lower()]


class DenseRWF(nn.Module):
    """
    Dense layer with Random Weight Factorization (RWF).

    Implements W = diag(exp(s)) · V where s and V are trainable.
    Based on Wang et al. "On the eigenvector bias of Fourier feature networks".

    Attributes:
        features:  Number of output features.
        rwf_mu:    Mean for initialising s ~ N(mu, sigma*I). Recommended: 0.5 or 1.0.
        rwf_sigma: Std  for initialising s ~ N(mu, sigma*I). Recommended: 0.1.
    """

    features: int
    rwf_mu: float = 0.5
    rwf_sigma: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: y = (diag(exp(s)) · V) @ x + b"""
        input_features = x.shape[-1]

        V = self.param(
            'V',
            nn.initializers.glorot_normal(),
            (self.features, input_features),
        )
        s = self.param(
            's',
            lambda key, shape: self.rwf_mu + self.rwf_sigma * jax.random.normal(key, shape),
            (self.features,),
        )
        b = self.param('b', nn.initializers.zeros, (self.features,))

        W = jnp.exp(s)[:, None] * V  # (features, input_features)
        return x @ W.T + b


__all__ = ["get_activation", "DenseRWF"]
