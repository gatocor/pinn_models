"""
Residual Network (ResNet) with pre-activation blocks for JAX-based PINNs.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Callable, Optional, Dict

from ._common import get_activation

class ResNetBlock(nn.Module):
    """
    Single pre-activation residual block.

    Architecture::

        x_out = x + Dense2(act(LN(Dense1(act(LN(x))))))

    Layer-norm stabilises training when layer_norm=True.
    """
    hidden_dim: int
    activation: str = "tanh"
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act_fn = get_activation(self.activation)
        h = nn.LayerNorm()(x) if self.layer_norm else x
        h = act_fn(nn.Dense(self.hidden_dim, name="dense1")(h))
        h = nn.LayerNorm()(h) if self.layer_norm else h
        h = nn.Dense(self.hidden_dim, name="dense2")(h)
        return x + h


class ResNetModule(nn.Module):
    """Internal Flax module for ResNet."""
    input_dim: int
    output_dim: int
    hidden_dim: int
    n_blocks: int = 4
    activation: str = "tanh"
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act_fn = get_activation(self.activation)
        x = act_fn(nn.Dense(self.hidden_dim, name="input_proj")(x))
        for i in range(self.n_blocks):
            x = ResNetBlock(
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                layer_norm=self.layer_norm,
                name=f"block_{i}",
            )(x)
        return nn.Dense(self.output_dim, name="output")(x)


class ResNet:
    """
    Residual Network (ResNet) for JAX PINNs.

    Pre-activation ResNet trunk with optional layer-norm inside each block.
    Shares the same API as FNN and PirateNet.

    Architecture::

        input -> Dense(hidden_dim) + act
              -> [Dense->act->Dense + skip] * n_blocks
              -> Dense(output_dim)

    Args:
        input_dim:          Input dimension.
        output_dim:         Output dimension.
        hidden_dim:         Width of every hidden layer.
        n_blocks:           Number of residual blocks (default 4).
        activation:         Activation function name (default tanh).
        layer_norm:         Apply LayerNorm inside each block (default True).
        normalize_input:    Normalise inputs to [-1, 1] (default True).
        unnormalize_output: Reverse-scale outputs (default True).
        input_transform:    Optional (x, params_dict) -> x callable.
        output_transform:   Optional (x_orig, y, params_dict) -> y callable.
        feature_encoding:   Optional feature encoder applied after normalisation.

    Example::

        net = ResNet(input_dim=3, output_dim=1, hidden_dim=128, n_blocks=4)
        params = net.init(jax.random.PRNGKey(0))
        y = net.apply(params, x)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_blocks: int = 4,
        activation: str = "tanh",
        layer_norm: bool = True,
        normalize_input: bool = True,
        unnormalize_output: bool = True,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        feature_encoding: Optional[Callable] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.activation = activation
        self.layer_norm = layer_norm
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.feature_encoding = feature_encoding

        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None
        self.layer_sizes = [input_dim, hidden_dim, output_dim]

        self._module = ResNetModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            activation=activation,
            layer_norm=layer_norm,
        )

    def set_input_range(self, xmin, xmax):
        self.input_min = jnp.array(xmin)
        self.input_max = jnp.array(xmax)

    def set_output_range(self, ymin, ymax):
        self.output_min = jnp.array(ymin)
        self.output_max = jnp.array(ymax)

    def init(self, rng, dummy_input=None):
        if dummy_input is None:
            dummy_input = jnp.ones((1, self.input_dim))
        if self.feature_encoding is not None:
            dummy_input = self.feature_encoding(dummy_input, None)
        return self._module.init(rng, dummy_input)

    def apply(self, params, x, params_dict=None):
        x_original = x
        if self.input_transform is not None:
            x = self.input_transform(x, params_dict)
        if self.normalize_input and self.input_min is not None:
            x = 2.0 * (x - self.input_min) / (self.input_max - self.input_min + 1e-8) - 1.0
        if self.feature_encoding is not None:
            x = self.feature_encoding(x, params_dict)
        y = self._module.apply(params, x)
        if self.unnormalize_output and self.output_min is not None:
            y = (y + 1.0) / 2.0 * (self.output_max - self.output_min) + self.output_min
        if self.output_transform is not None:
            y = self.output_transform(x_original, y, params_dict)
        return y

    def forward(self, x, params_dict=None):
        return self.apply(self.params, x, params_dict)

    def predict(self, x_np, params_dict=None):
        return np.array(self.apply(self.params, jnp.array(x_np), params_dict))

    def to(self, device=None, dtype=None, seed=0):
        if dtype is None:
            dtype = jnp.float32
        self.device = device or jax.devices()[0].platform
        self.dtype = dtype
        if not hasattr(self, "params") or self.params is None:
            dummy = jnp.ones((1, self.input_dim), dtype=dtype)
            if self.feature_encoding is not None:
                dummy = self.feature_encoding(dummy, None)
            self.params = self._module.init(jax.random.PRNGKey(seed), dummy)
        return self

    def __repr__(self):
        ln = ", layer_norm=True" if self.layer_norm else ""
        return (
            f"ResNet(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"n_blocks={self.n_blocks}, output_dim={self.output_dim}{ln})"
        )


__all__ = ["ResNet", "ResNetBlock", "ResNetModule"]
