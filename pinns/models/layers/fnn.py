"""
Composable FNN layer for use inside :class:`~pinns.modelbase.ModelBase`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, List, Optional, Sequence

from ._common import get_activation


class FNNModule(nn.Module):
    """Internal Flax module for FNN (no normalisation)."""

    layer_sizes: tuple
    activation: str = 'tanh'
    output_activation: Optional[str] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act_fn = get_activation(self.activation)
        for i, size in enumerate(self.layer_sizes[1:-1]):
            x = nn.Dense(size, name=f'hidden_{i}')(x)
            x = act_fn(x)
        x = nn.Dense(self.layer_sizes[-1], name='output')(x)
        if self.output_activation is not None:
            x = get_activation(self.output_activation)(x)
        return x


class FNN:
    """
    Fully-connected layer for use inside :class:`~pinns.modelbase.ModelBase`.

    Users specify only the *hidden* widths; ``input_dim`` is injected by
    ``ModelBase`` when ``net.add(FNN([...]))`` is called.  The output width
    defaults to ``network.output_dim`` (the final network output), but can be
    overridden via ``output_dim`` to use this layer as an intermediate feature
    extractor.

    Parameters
    ----------
    hidden_dims : sequence of int
        Hidden layer widths.
    activation : str
        Hidden-layer activation (default ``'tanh'``).
    output_activation : str, optional
        Optional final activation name.
    output_dim : int or None, optional
        Output width.  When ``None`` (default) the ModelBase's ``output_dim``
        is used, which is the correct choice for the **last** layer.  Pass an
        explicit ``int`` to override (e.g. when using this as an intermediate
        feature extractor that should not project down to the final output size)::

            net.add(FNN([64, 64, 64], output_dim=64))  # intermediate → 64
            net.add(FNN([64, 64, 64]))                  # final → network.output_dim
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        output_dim: Optional[int] = None,
    ):
        self.hidden_dims      = list(hidden_dims)
        self.activation       = activation
        self.output_activation = output_activation
        self._output_dim_override = output_dim
        self._module: Optional[FNNModule] = None
        self._layer_sizes: Optional[List[int]] = None

    def _configure(self, network, input_dim: int) -> int:
        output_dim = self._output_dim_override if self._output_dim_override is not None else network.output_dim
        self._layer_sizes = [input_dim] + self.hidden_dims + [output_dim]
        self._module = FNNModule(
            layer_sizes=self._layer_sizes,
            activation=self.activation,
            output_activation=self.output_activation,
        )
        return output_dim

    def init(self, rng) -> Dict:
        assert self._module is not None, "FNN not configured — add to a ModelBase first"
        dummy = jnp.ones((1, self._layer_sizes[0]))
        return self._module.init(rng, dummy)

    def apply(self, x: jnp.ndarray, params: Dict = None, params_dict=None) -> jnp.ndarray:
        return self._module.apply(params, x)

    def __repr__(self) -> str:
        return f"FNN(layer_sizes={self._layer_sizes})"


__all__ = ["FNN"]
