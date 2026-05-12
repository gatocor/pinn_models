"""
Random Fourier Feature encoding for Physics-Informed Neural Networks.

Based on Tancik et al. "Fourier Features Let Networks Learn High Frequency
Functions in Low Dimensional Domains" (NeurIPS 2020).
"""

import jax
import jax.numpy as jnp
from typing import Optional

class RandomFourierFeatures:
    """
    Random Fourier Feature encoding to mitigate spectral bias in PINNs.

    The encoding maps x to::

        [cos(2 * pi * B @ x_enc), sin(2 * pi * B @ x_enc)]

    where B ~ N(0, sigma^2) is a fixed random projection matrix.

    Like all layers, domain/input information is injected automatically when
    added to a :class:`~pinns.models.model_base.ModelBase` — no need to pass
    ``domain`` or ``input_dim``::

        net.add(RandomFourierFeatures(n_features=64, sigma=5.0, encode_time=False))

    Parameters
    ----------
    n_features : int
        Number of Fourier features; output width is ``2 * n_features``.
    sigma : float
        Standard deviation for the random projection matrix B.
    seed : int
        Random seed for reproducible B.
    include_input : bool
        If True, concatenate the encoded input coordinates before the
        cos/sin block.
    encode_time : bool or None
        How to handle a time column when the domain has a time axis:

        * ``True``  — include time in the Fourier encoding.
        * ``False`` — encode only spatial coords; raw ``t`` is appended.
        * ``None``  (default) — raise an error if the domain has time,
          forcing an explicit choice.
    """

    def __init__(
        self,
        n_features: int = 64,
        sigma: float = 1.0,
        seed: int = 0,
        include_input: bool = False,
        encode_time: Optional[bool] = None,
    ):
        self.n_features    = n_features
        self.sigma         = sigma
        self.seed          = seed
        self.include_input = include_input
        self.encode_time   = encode_time

        # Set by _configure:
        self.B:                  Optional[jnp.ndarray] = None
        self.output_dim:         Optional[int]         = None
        self._spatial_dims:      Optional[int]         = None
        self._has_time:          Optional[bool]        = None
        self._encode_time:       Optional[bool]        = None
        self._fourier_input_dim: Optional[int]         = None
        self._n_context:         Optional[int]         = None

    # ── ModelBase composable protocol ──────────────────────────────────────── #

    def _configure(self, network, input_dim: int) -> int:
        """Called by ModelBase.add(). Derives everything from network.domain."""
        domain    = network.domain
        t_interval = getattr(domain, "t_interval", None)
        has_time  = t_interval is not None

        if has_time and self.encode_time is None:
            raise ValueError(
                "RandomFourierFeatures: the domain has a time dimension — "
                "you must explicitly set encode_time=True or encode_time=False."
            )

        encode_time       = bool(self.encode_time) if self.encode_time is not None else False
        spatial_dims      = domain._spatial_dims
        fourier_input_dim = spatial_dims + (1 if (has_time and encode_time) else 0)
        n_context         = network.n_context

        key    = jax.random.PRNGKey(self.seed)
        self.B = jax.random.normal(key, (self.n_features, fourier_input_dim)) * self.sigma

        fourier_out = 2 * self.n_features + (fourier_input_dim if self.include_input else 0)
        raw_t_col   = 1 if (has_time and not encode_time) else 0
        out_dim     = fourier_out + raw_t_col + n_context

        self._spatial_dims      = spatial_dims
        self._has_time          = has_time
        self._encode_time       = encode_time
        self._fourier_input_dim = fourier_input_dim
        self._n_context         = n_context
        self.output_dim         = out_dim
        return out_dim

    def init(self, rng) -> dict:
        """No trainable parameters."""
        return {}

    def apply(self, params: dict, x, params_dict=None):
        """ModelBase-protocol forward pass."""
        return self._forward(x, params_dict)

    # ── Forward pass ───────────────────────────────────────────────────────── #

    def _forward(self, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        assert self.B is not None, "RandomFourierFeatures not configured — add to a ModelBase first"

        if self._has_time and not self._encode_time:
            x_enc = x[:, :self._spatial_dims]
            t_col = x[:, self._spatial_dims : self._spatial_dims + 1]
        else:
            x_enc = x[:, :self._fourier_input_dim]
            t_col = None

        Bx       = x_enc @ self.B.T
        features = jnp.concatenate(
            [jnp.cos(2 * jnp.pi * Bx), jnp.sin(2 * jnp.pi * Bx)], axis=-1
        )
        if self.include_input:
            features = jnp.concatenate([x_enc, features], axis=-1)
        if t_col is not None:
            features = jnp.concatenate([features, t_col], axis=-1)
        if self._n_context and self._n_context > 0:
            ctx      = x[:, -self._n_context:]
            features = jnp.concatenate([features, ctx], axis=-1)
        return features

    def __call__(self, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        return self._forward(x, params_dict)

    def transform(self, x: jnp.ndarray, params_dict=None) -> jnp.ndarray:
        """Alias for forward pass."""
        return self._forward(x, params_dict)

    def __repr__(self) -> str:
        if self.B is None:
            return (f"RandomFourierFeatures(n_features={self.n_features}, "
                    f"sigma={self.sigma}, encode_time={self.encode_time})")
        inc = ", include_input=True" if self.include_input else ""
        et  = f", encode_time={self._encode_time}" if self._has_time else ""
        ctx = f", n_context={self._n_context}" if self._n_context else ""
        return (
            f"RandomFourierFeatures(input_dim={self._fourier_input_dim}, "
            f"n_features={self.n_features}, sigma={self.sigma}{inc}{et}{ctx})"
        )


# Backward-compatible alias
FourierFeatures = RandomFourierFeatures

__all__ = ["RandomFourierFeatures", "FourierFeatures"]
