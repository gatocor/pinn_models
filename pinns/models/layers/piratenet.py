"""
Composable PirateNet layer for use inside :class:`~pinns.modelbase.ModelBase`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Optional

from ._common import get_activation, DenseRWF


class PirateNetBlock(nn.Module):
    """Single residual block for PirateNet with Random Weight Factorization."""
    hidden_dim: int
    activation: str = "tanh"
    rwf_mu: float = 0.5
    rwf_sigma: float = 0.1

    @nn.compact
    def __call__(self, x, U, V):
        act_fn = get_activation(self.activation)
        alpha = self.param("alpha", nn.initializers.zeros, ())
        f  = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="dense1")(x))
        z1 = f * U + (1 - f) * V
        g  = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="dense2")(z1))
        z2 = g * U + (1 - g) * V
        h  = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="dense3")(z2))
        return alpha * h + (1 - alpha) * x


class PirateNetModule(nn.Module):
    """Internal Flax module for PirateNet."""
    input_dim: int
    output_dim: int
    hidden_dim: int
    n_blocks: int = 3
    activation: str = "tanh"
    rwf_mu: float = 0.5
    rwf_sigma: float = 0.1
    output_projection: bool = True

    @nn.compact
    def __call__(self, x):
        act_fn = get_activation(self.activation)
        U = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="U_layer")(x))
        V = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="V_layer")(x))
        h = DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="input_projection")(x)
        for i in range(self.n_blocks):
            h = PirateNetBlock(
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                rwf_mu=self.rwf_mu,
                rwf_sigma=self.rwf_sigma,
                name=f"block_{i}",
            )(h, U, V)
        if self.output_projection:
            return DenseRWF(self.output_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="output")(h)
        return h


class PirateNet:
    """
    PirateNet (Physics-Informed Residual AdapTivE Net) layer for use inside
    :class:`~pinns.modelbase.ModelBase`.

    Users specify only the hidden configuration; ``input_dim`` and
    ``output_dim`` are injected by the ModelBase.

    Parameters
    ----------
    hidden_dim : int
        Width of all hidden layers.
    n_blocks : int
        Number of residual blocks (default 3).
    activation : str
        Activation function name (default ``'tanh'``).
    rwf_mu, rwf_sigma : float
        Random Weight Factorisation parameters.
    output_dim : int or None, optional
        Output width.  When ``None`` (default) the ModelBase's ``output_dim``
        is used.  Pass an explicit ``int`` to override.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_blocks: int = 3,
        activation: str = "tanh",
        rwf_mu: float = 0.5,
        rwf_sigma: float = 0.1,
        output_dim: Optional[int] = None,
    ):
        self.hidden_dim = hidden_dim
        self.n_blocks   = n_blocks
        self.activation = activation
        self.rwf_mu     = rwf_mu
        self.rwf_sigma  = rwf_sigma
        self._output_dim_override = output_dim
        self._module: Optional[PirateNetModule] = None
        self._input_dim: Optional[int] = None
        self._output_dim: Optional[int] = None

    def _configure(self, network, input_dim: int) -> int:
        self._input_dim  = input_dim
        self._output_dim = self._output_dim_override if self._output_dim_override is not None else network.output_dim
        output_projection = True
        self._module = PirateNetModule(
            input_dim=input_dim,
            output_dim=self._output_dim,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            activation=self.activation,
            rwf_mu=self.rwf_mu,
            rwf_sigma=self.rwf_sigma,
            output_projection=output_projection,
        )
        return self._output_dim

    def init(self, rng) -> Dict:
        assert self._module is not None, "PirateNet not configured — add to a ModelBase first"
        dummy = jnp.ones((1, self._input_dim))
        return self._module.init(rng, dummy)

    def apply(self, params: Dict, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        return self._module.apply(params, x)

    def __repr__(self) -> str:
        return (
            f"PirateNet(input_dim={self._input_dim}, hidden_dim={self.hidden_dim}, "
            f"n_blocks={self.n_blocks}, output_dim={self._output_dim})"
        )


__all__ = ["PirateNet"]
