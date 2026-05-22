"""
Composable PirateNet layer for use inside :class:`~pinns.modelbase.ModelBase`.
"""

from __future__ import annotations

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
        identity = x
        output_dim = x.shape[-1]  # residual stream keeps embedding dimension

        h = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="dense1")(x))
        h = h * U + (1 - h) * V
        h = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="dense2")(h))
        h = h * U + (1 - h) * V
        h = act_fn(DenseRWF(output_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="dense3")(h))

        alpha = self.param("alpha", nn.initializers.zeros, (1,))
        return alpha * h + (1 - alpha) * identity


class PirateNetModule(nn.Module):
    """Internal Flax module for PirateNet."""
    input_dim: int
    output_dim: int
    hidden_dim: int
    n_blocks: int = 3
    activation: str = "tanh"
    output_activation: Optional[str] = None
    rwf_mu: float = 0.5
    rwf_sigma: float = 0.1
    output_projection: bool = True

    @nn.compact
    def __call__(self, x):
        act_fn = get_activation(self.activation)
        U = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="U_layer")(x))
        V = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="V_layer")(x))
        # No input projection: residual stream starts directly from embedding x
        for i in range(self.n_blocks):
            x = PirateNetBlock(
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                rwf_mu=self.rwf_mu,
                rwf_sigma=self.rwf_sigma,
                name=f"block_{i}",
            )(x, U, V)
        if self.output_projection:
            x = DenseRWF(self.output_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="output")(x)
        if self.output_activation is not None:
            x = get_activation(self.output_activation)(x)
        return x


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
    output_activation : str or None, optional
        Activation applied to the final output (e.g. ``'tanh'`` to bound
        outputs to (-1, 1)).  ``None`` (default) means linear output.
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
        output_activation: Optional[str] = None,
        rwf_mu: float = 0.5,
        rwf_sigma: float = 0.1,
        output_dim: Optional[int] = None,
    ):
        self.hidden_dim        = hidden_dim
        self.n_blocks          = n_blocks
        self.activation        = activation
        self.output_activation = output_activation
        self.rwf_mu            = rwf_mu
        self.rwf_sigma         = rwf_sigma
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
            output_activation=self.output_activation,
            rwf_mu=self.rwf_mu,
            rwf_sigma=self.rwf_sigma,
            output_projection=output_projection,
        )
        return self._output_dim

    def init(self, rng) -> Dict:
        assert self._module is not None, "PirateNet not configured — add to a ModelBase first"
        dummy = jnp.ones((1, self._input_dim))
        return self._module.init(rng, dummy)

    def apply(self, x: jnp.ndarray, params: Dict = None, params_dict=None) -> jnp.ndarray:
        return self._module.apply(params, x)

    def __repr__(self) -> str:
        out_act = f", output_activation={self.output_activation!r}" if self.output_activation else ""
        return (
            f"PirateNet(input_dim={self._input_dim}, hidden_dim={self.hidden_dim}, "
            f"n_blocks={self.n_blocks}, output_dim={self._output_dim}{out_act})"
        )


__all__ = ["PirateNet"]
