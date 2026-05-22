"""
Composable ResNet layer for use inside :class:`~pinns.modelbase.ModelBase`.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Optional

from ._common import get_activation


class ResNetBlock(nn.Module):
    """Single pre-activation residual block."""
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
    output_projection: bool = True

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
        if self.output_projection:
            return nn.Dense(self.output_dim, name="output")(x)
        return x


class ResNet:
    """
    Residual ModelBase layer for use inside :class:`~pinns.modelbase.ModelBase`.

    Users specify only the hidden configuration; ``input_dim`` and
    ``output_dim`` are injected by the ModelBase.

    Parameters
    ----------
    hidden_dim : int
        Width of every hidden layer.
    n_blocks : int
        Number of residual blocks (default 4).
    activation : str
        Activation function name (default ``'tanh'``).
    layer_norm : bool
        Apply LayerNorm inside each block (default ``True``).
    output_dim : int or None, optional
        Output width.  When ``None`` (default) the ModelBase's ``output_dim``
        is used.  Pass an explicit ``int`` to override.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_blocks: int = 4,
        activation: str = "tanh",
        layer_norm: bool = True,
        output_dim: Optional[int] = None,
    ):
        self.hidden_dim = hidden_dim
        self.n_blocks   = n_blocks
        self.activation = activation
        self.layer_norm = layer_norm
        self._output_dim_override = output_dim
        self._module: Optional[ResNetModule] = None
        self._input_dim: Optional[int] = None
        self._output_dim: Optional[int] = None

    def _configure(self, network, input_dim: int) -> int:
        self._input_dim  = input_dim
        self._output_dim = self._output_dim_override if self._output_dim_override is not None else network.output_dim
        output_projection = True
        self._module = ResNetModule(
            input_dim=input_dim,
            output_dim=self._output_dim,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            activation=self.activation,
            layer_norm=self.layer_norm,
            output_projection=output_projection,
        )
        return self._output_dim

    def init(self, rng) -> Dict:
        assert self._module is not None, "ResNet not configured — add to a ModelBase first"
        dummy = jnp.ones((1, self._input_dim))
        return self._module.init(rng, dummy)

    def apply(self, x: jnp.ndarray, params: Dict = None, params_dict=None) -> jnp.ndarray:
        return self._module.apply(params, x)

    def __repr__(self) -> str:
        return (
            f"ResNet(input_dim={self._input_dim}, hidden_dim={self.hidden_dim}, "
            f"n_blocks={self.n_blocks}, output_dim={self._output_dim})"
        )


__all__ = ["ResNet"]
