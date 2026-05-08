"""
Composable PirateNet layer for use inside :class:`~pinns.network.Network`.
"""

from __future__ import annotations

import jax.numpy as jnp
from typing import Dict, Optional

from pinns.base_models.piratenet import PirateNetModule


class PirateNet:
    """
    PirateNet (Physics-Informed Residual AdapTivE Net) layer for use inside
    :class:`~pinns.network.Network`.

    Users specify only the hidden configuration; ``input_dim`` and
    ``output_dim`` are injected by the Network.

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
    """

    def __init__(
        self,
        hidden_dim: int,
        n_blocks: int = 3,
        activation: str = "tanh",
        rwf_mu: float = 0.5,
        rwf_sigma: float = 0.1,
    ):
        self.hidden_dim = hidden_dim
        self.n_blocks   = n_blocks
        self.activation = activation
        self.rwf_mu     = rwf_mu
        self.rwf_sigma  = rwf_sigma
        self._module: Optional[PirateNetModule] = None
        self._input_dim: Optional[int] = None
        self._output_dim: Optional[int] = None

    def _configure(self, network, input_dim: int) -> int:
        self._input_dim  = input_dim
        self._output_dim = network.output_dim
        self._module = PirateNetModule(
            input_dim=input_dim,
            output_dim=network.output_dim,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            activation=self.activation,
            rwf_mu=self.rwf_mu,
            rwf_sigma=self.rwf_sigma,
        )
        return network.output_dim

    def init(self, rng) -> Dict:
        assert self._module is not None, "PirateNet not configured — add to a Network first"
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
