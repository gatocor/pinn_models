"""
Composable WFFNN layer for use inside :class:`~pinns.modelbase.ModelBase`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, List, Optional, Sequence

from ._common import get_activation, DenseRWF


class WFFNNModule(nn.Module):
    """Internal Flax module for WFFNN (Weight-Factorised FNN)."""

    layer_sizes: tuple
    activation: str = 'tanh'
    output_activation: Optional[str] = None
    rwf_mu: float = 0.5
    rwf_sigma: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act_fn = get_activation(self.activation)
        for i, size in enumerate(self.layer_sizes[1:-1]):
            x = DenseRWF(size, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma,
                         name=f'hidden_{i}')(x)
            x = act_fn(x)
        x = DenseRWF(self.layer_sizes[-1], rwf_mu=self.rwf_mu,
                     rwf_sigma=self.rwf_sigma, name='output')(x)
        if self.output_activation is not None:
            x = get_activation(self.output_activation)(x)
        return x


class WFFNN:
    """
    Weight-Factorised FNN layer for use inside :class:`~pinns.modelbase.ModelBase`.

    Parameters
    ----------
    hidden_dims : sequence of int
        Hidden layer widths.
    activation : str
        Hidden-layer activation (default ``'tanh'``).
    rwf_mu, rwf_sigma : float
        Parameters for random weight factorisation initialisation.
    output_dim : int or None, optional
        Output width.  When ``None`` (default) the ModelBase's ``output_dim``
        is used.  Pass an explicit ``int`` to override.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        rwf_mu: float = 0.5,
        rwf_sigma: float = 0.1,
        output_dim: Optional[int] = None,
    ):
        self.hidden_dims       = list(hidden_dims)
        self.activation        = activation
        self.output_activation = output_activation
        self.rwf_mu            = rwf_mu
        self.rwf_sigma         = rwf_sigma
        self._output_dim_override = output_dim
        self._module: Optional[WFFNNModule] = None
        self._layer_sizes: Optional[List[int]] = None

    def _configure(self, network, input_dim: int) -> int:
        output_dim = self._output_dim_override if self._output_dim_override is not None else network.output_dim
        self._layer_sizes = [input_dim] + self.hidden_dims + [output_dim]
        self._module = WFFNNModule(
            layer_sizes=tuple(self._layer_sizes),
            activation=self.activation,
            output_activation=self.output_activation,
            rwf_mu=self.rwf_mu,
            rwf_sigma=self.rwf_sigma,
        )
        return output_dim

    def init(self, rng) -> Dict:
        assert self._module is not None, "WFFNN not configured — add to a ModelBase first"
        dummy = jnp.ones((1, self._layer_sizes[0]))
        return self._module.init(rng, dummy)

    def apply(self, params: Dict, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        return self._module.apply(params, x)

    def __repr__(self) -> str:
        return f"WFFNN(layer_sizes={self._layer_sizes})"


__all__ = ["WFFNN"]
