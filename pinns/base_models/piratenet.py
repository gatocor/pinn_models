"""
PirateNet - Physics-Informed Residual AdapTivE Network for JAX-based PINNs.

Based on Wang et al. "PirateNets: Physics-informed Deep Learning with
Residual Adaptive Networks".
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Callable, Optional, Dict

from ._common import get_activation, DenseRWF


class PirateNetBlock(nn.Module):
    """
    Single residual block for PirateNet with Random Weight Factorization.

    Each block applies three RWF-dense layers with gate-mixing::

        f   = act(RWF(x))
        z1  = f * U + (1-f) * V
        g   = act(RWF(z1))
        z2  = g * U + (1-g) * V
        h   = act(RWF(z2))
        out = alpha * h + (1 - alpha) * x    # alpha init to 0
    """
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
        return DenseRWF(self.output_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name="output")(h)


class PirateNet:
    """
    Physics-Informed Residual AdapTivE Network (PirateNet) for JAX PINNs.

    Uses adaptive residual connections with trainable alpha initialised to 0,
    so the network starts as a near-linear map. Combines Random Weight
    Factorization (RWF) throughout.

    Args:
        input_dim:          Input dimension (before any feature encoding).
        output_dim:         Output dimension.
        hidden_dim:         Width of all hidden layers.
        n_blocks:           Number of residual blocks (default 3).
        activation:         Activation function name (default tanh).
        normalize_input:    Normalise inputs to [-1, 1] (default True).
        unnormalize_output: Reverse-scale outputs (default True).
        input_transform:    Optional (x, params_dict) -> x callable.
        output_transform:   Optional (x_orig, y, params_dict) -> y callable.
        feature_encoding:   Optional feature encoder applied after normalisation.
        rwf_mu:             Mean for RWF s initialisation (default 0.5).
        rwf_sigma:          Std  for RWF s initialisation (default 0.1).

    Example::

        net = PirateNet(input_dim=2, output_dim=1, hidden_dim=64)
        params = net.init(jax.random.PRNGKey(0))
        y = net.apply(params, x)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_blocks: int = 3,
        activation: str = "tanh",
        normalize_input: bool = True,
        unnormalize_output: bool = True,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        feature_encoding: Optional[Callable] = None,
        rwf_mu: float = 0.5,
        rwf_sigma: float = 0.1,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.activation = activation
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.feature_encoding = feature_encoding
        self.rwf_mu = rwf_mu
        self.rwf_sigma = rwf_sigma

        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None
        self.layer_sizes = [input_dim, hidden_dim, output_dim]

        self._module = PirateNetModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            activation=activation,
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma,
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
        return np.array(self.forward(jnp.array(x_np), params_dict))

    def to(self, device=None, dtype=None, seed=0):
        if device is None:
            device = jax.devices()[0].platform
        if dtype is None:
            dtype = jnp.float32
        self.device = device
        self.dtype = dtype
        if not hasattr(self, "params") or self.params is None:
            dummy = jnp.ones((1, self.input_dim), dtype=dtype)
            if self.feature_encoding is not None:
                dummy = self.feature_encoding(dummy, None)
            self.params = self._module.init(jax.random.PRNGKey(seed), dummy)
        return self


__all__ = ["PirateNet", "PirateNetBlock", "PirateNetModule"]
