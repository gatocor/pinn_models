"""
Composable FNN layer for use inside :class:`~pinns.network.Network`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Sequence

from pinns.base_models.fnn import FNNModule, WFFNNModule


class FNN:
    """
    Fully-connected layer for use inside :class:`~pinns.network.Network`.

    Users specify only the *hidden* widths; ``input_dim`` and ``output_dim``
    are injected by the Network when ``net.add(FNN([...]))`` is called.

    Parameters
    ----------
    hidden_dims : sequence of int
        Hidden layer widths.
    activation : str
        Hidden-layer activation (default ``'tanh'``).
    output_activation : str, optional
        Optional final activation name.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        activation: str = "tanh",
        output_activation: Optional[str] = None,
    ):
        self.hidden_dims      = list(hidden_dims)
        self.activation       = activation
        self.output_activation = output_activation
        self._module: Optional[FNNModule] = None
        self._layer_sizes: Optional[List[int]] = None

    def _configure(self, network, input_dim: int) -> int:
        output_dim = network.output_dim
        self._layer_sizes = [input_dim] + self.hidden_dims + [output_dim]
        self._module = FNNModule(
            layer_sizes=self._layer_sizes,
            activation=self.activation,
            output_activation=self.output_activation,
        )
        return output_dim

    def init(self, rng) -> Dict:
        assert self._module is not None, "FNN not configured — add to a Network first"
        dummy = jnp.ones((1, self._layer_sizes[0]))
        return self._module.init(rng, dummy)

    def apply(self, params: Dict, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        return self._module.apply(params, x)

    def __repr__(self) -> str:
        return f"FNN(layer_sizes={self._layer_sizes})"


class WFFNN:
    """
    Weight-Factorised FNN layer for use inside :class:`~pinns.network.Network`.

    Parameters
    ----------
    hidden_dims : sequence of int
        Hidden layer widths.
    activation : str
        Hidden-layer activation (default ``'tanh'``).
    rwf_mu, rwf_sigma : float
        Parameters for random weight factorisation initialisation.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        rwf_mu: float = 0.5,
        rwf_sigma: float = 0.1,
    ):
        self.hidden_dims      = list(hidden_dims)
        self.activation       = activation
        self.output_activation = output_activation
        self.rwf_mu           = rwf_mu
        self.rwf_sigma        = rwf_sigma
        self._module: Optional[WFFNNModule] = None
        self._layer_sizes: Optional[List[int]] = None

    def _configure(self, network, input_dim: int) -> int:
        output_dim = network.output_dim
        self._layer_sizes = [input_dim] + self.hidden_dims + [output_dim]
        self._module = WFFNNModule(
            layer_sizes=self._layer_sizes,
            activation=self.activation,
            output_activation=self.output_activation,
            rwf_mu=self.rwf_mu,
            rwf_sigma=self.rwf_sigma,
        )
        return output_dim

    def init(self, rng) -> Dict:
        assert self._module is not None, "WFFNN not configured — add to a Network first"
        dummy = jnp.ones((1, self._layer_sizes[0]))
        return self._module.init(rng, dummy)

    def apply(self, params: Dict, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        return self._module.apply(params, x)

    def __repr__(self) -> str:
        return f"WFFNN(layer_sizes={self._layer_sizes})"


__all__ = ["FNN", "WFFNN"]
