"""
Fully-connected Neural Network (FNN) and Weight-Factorised variant (WFFNN)
for JAX-based Physics-Informed Neural Networks.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Sequence, Callable, Optional, Dict

from ._common import get_activation, DenseRWF


# ---------------------------------------------------------------------------
# Internal Flax modules
# ---------------------------------------------------------------------------


class FNNModule(nn.Module):
    """Internal Flax module for FNN (no normalisation)."""

    layer_sizes: Sequence[int]
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


class WFFNNModule(nn.Module):
    """Internal Flax module for WFFNN (Weight-Factorised FNN)."""

    layer_sizes: Sequence[int]
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


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


class FNN:
    """
    Fully-connected Neural Network for JAX PINNs.

    Trainer-compatible wrapper with optional input/output normalisation,
    feature encoding, and hard-constraint transforms.

    Args:
        layer_sizes:        Network architecture [input, hidden..., output].
                            If using ``feature_encoding``, ``layer_sizes[0]``
                            should match the encoding's ``output_dim``.
        activation:         Hidden-layer activation name (default ``'tanh'``).
        output_activation:  Optional output activation name.
        normalize_input:    Normalise inputs to [-1, 1] using trainer-set bounds.
        unnormalize_output: Reverse-scale outputs using trainer-set bounds.
        input_transform:    ``(x, params_dict) -> x`` callable applied before
                            normalisation (e.g. symmetry transform).
        output_transform:   ``(x_orig, y, params_dict) -> y`` callable applied
                            after the network (e.g. hard-BC clamp).
        feature_encoding:   Optional feature encoder applied after normalisation.
        output_range:       Optional ``(ymin, ymax)`` pair (or list of pairs).

    Example::

        net = FNN([2, 64, 64, 1], activation='tanh')
        params = net.init(jax.random.PRNGKey(0))
        y = net.apply(params, x)
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = 'tanh',
        output_activation: Optional[str] = None,
        normalize_input: bool = True,
        unnormalize_output: bool = True,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        feature_encoding: Optional[Callable] = None,
        output_range=None,
    ):
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.feature_encoding = feature_encoding

        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None

        self._module = FNNModule(
            layer_sizes=layer_sizes,
            activation=activation,
            output_activation=output_activation,
        )

        if output_range is not None:
            if (isinstance(output_range, (list, tuple))
                    and not isinstance(output_range[0], (list, tuple))):
                n_out = layer_sizes[-1]
                ymin = np.full(n_out, float(output_range[0]))
                ymax = np.full(n_out, float(output_range[1]))
            else:
                ymin = np.array([r[0] for r in output_range], dtype=float)
                ymax = np.array([r[1] for r in output_range], dtype=float)
            self.set_output_range(ymin, ymax)

    def set_input_range(self, xmin: np.ndarray, xmax: np.ndarray):
        self.input_min = jnp.array(xmin)
        self.input_max = jnp.array(xmax)

    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        self.output_min = jnp.array(ymin)
        self.output_max = jnp.array(ymax)

    def init(self, rng: jax.random.PRNGKey, dummy_input: jnp.ndarray = None) -> Dict:
        if dummy_input is None:
            dummy_input = jnp.ones((1, self.layer_sizes[0]))
        return self._module.init(rng, dummy_input)

    def apply(
        self,
        params: Dict,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
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

    def forward(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        return self.apply(self.params, x, params_dict)

    def predict(self, x_np: np.ndarray, params_dict: Optional[Dict] = None) -> np.ndarray:
        return np.array(self.forward(jnp.array(x_np), params_dict))

    def to(self, device: str = None, dtype=None, seed: int = 0) -> 'FNN':
        if device is None:
            device = jax.devices()[0].platform
        if dtype is None:
            dtype = jnp.float32
        self.device = device
        self.dtype = dtype
        if not hasattr(self, 'params') or self.params is None:
            dummy = jnp.ones((1, self.layer_sizes[0]), dtype=dtype)
            self.params = self._module.init(jax.random.PRNGKey(seed), dummy)
        return self


class WFFNN:
    """
    Weight-Factorised FNN for JAX PINNs.

    Replaces every Dense layer with :class:`~._common.DenseRWF` where
    W = diag(exp(s)) * V. Recommended: ``rwf_mu=0.5``, ``rwf_sigma=0.1``.

    Args:
        layer_sizes:        Network architecture [input, hidden..., output].
        activation:         Hidden-layer activation name (default ``'tanh'``).
        output_activation:  Optional output activation name.
        normalize_input:    Normalise inputs to [-1, 1].
        unnormalize_output: Reverse-scale outputs.
        input_transform:    Optional symmetry transform.
        output_transform:   Optional hard-constraint transform.
        feature_encoding:   Optional feature encoder.
        rwf_mu:             Mean for s initialisation (default 0.5).
        rwf_sigma:          Std  for s initialisation (default 0.1).

    Example::

        net = WFFNN([2, 64, 64, 1], rwf_mu=0.5, rwf_sigma=0.1)
        params = net.init(jax.random.PRNGKey(0))
        y = net.apply(params, x)
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = 'tanh',
        output_activation: Optional[str] = None,
        normalize_input: bool = True,
        unnormalize_output: bool = True,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        feature_encoding: Optional[Callable] = None,
        rwf_mu: float = 0.5,
        rwf_sigma: float = 0.1,
    ):
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
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

        self._module = WFFNNModule(
            layer_sizes=layer_sizes,
            activation=activation,
            output_activation=output_activation,
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma,
        )

    def set_input_range(self, xmin: np.ndarray, xmax: np.ndarray):
        self.input_min = jnp.array(xmin)
        self.input_max = jnp.array(xmax)

    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        self.output_min = jnp.array(ymin)
        self.output_max = jnp.array(ymax)

    def init(self, rng: jax.random.PRNGKey, dummy_input: jnp.ndarray = None) -> Dict:
        if dummy_input is None:
            dummy_input = jnp.ones((1, self.layer_sizes[0]))
        return self._module.init(rng, dummy_input)

    def apply(
        self,
        params: Dict,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
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

    def forward(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        return self.apply(self.params, x, params_dict)

    def predict(self, x_np: np.ndarray, params_dict: Optional[Dict] = None) -> np.ndarray:
        return np.array(self.forward(jnp.array(x_np), params_dict))

    def to(self, device: str = None, dtype=None, seed: int = 0) -> 'WFFNN':
        if device is None:
            device = jax.devices()[0].platform
        if dtype is None:
            dtype = jnp.float32
        self.device = device
        self.dtype = dtype
        if not hasattr(self, 'params') or self.params is None:
            dummy = jnp.ones((1, self.layer_sizes[0]), dtype=dtype)
            self.params = self._module.init(jax.random.PRNGKey(seed), dummy)
        return self


__all__ = ["FNN", "WFFNN", "FNNModule", "WFFNNModule"]
