"""
Random Fourier Feature encoding for Physics-Informed Neural Networks.

Based on Tancik et al. "Fourier Features Let Networks Learn High Frequency
Functions in Low Dimensional Domains" (NeurIPS 2020).
"""

import jax
import jax.numpy as jnp
from typing import Optional, Dict

class FourierFeatures:
    """
    Random Fourier Feature encoding to mitigate spectral bias in PINNs.

    The encoding maps x to::

        [cos(2 * pi * B @ x_enc), sin(2 * pi * B @ x_enc)]

    where B ~ N(0, sigma^2) is a fixed random projection matrix and x_enc is
    either the full input or only the spatial part (see *encode_time* below).

    Can be constructed in two ways:

    **Explicit** — pass ``input_dim`` directly::

        ff = FourierFeatures(input_dim=2, n_features=64, sigma=5.0)

    **Domain-aware** — pass a ``domain`` object; spatial dimensionality is
    inferred automatically.  If the domain has a time axis you **must** also
    specify ``encode_time``::

        # spatial-only encoding, time appended as a raw column
        ff = FourierFeatures(domain=domain, n_features=32, encode_time=False)
        # full space-time Fourier encoding
        ff = FourierFeatures(domain=domain, n_features=32, encode_time=True)

    Parameters
    ----------
    input_dim : int, optional
        Total coordinate dimension (required when *domain* is not given).
    n_features : int
        Number of Fourier features; base output width is ``2 * n_features``.
    sigma : float
        Standard deviation for B.
    seed : int
        Random seed for reproducible B.
    include_input : bool
        If True, concatenate the *encoded* input coordinates before the
        cos/sin block.
    domain : optional
        Domain object exposing ``_spatial_dims`` and optionally ``t_interval``.
        Mutually exclusive with *input_dim*.
    encode_time : bool, optional
        Only used when *domain* is provided *and* the domain has a time axis.
        - ``True``  : time is included in the Fourier encoding.
        - ``False`` : Fourier encoding is spatial-only; raw ``t`` is appended
          as a final output column.
        Raises ``ValueError`` if the domain has time and this is not set.
    n_context : int
        Number of extra context columns appended to the input *after* the
        coordinate (+ optional time) columns (e.g. solution field U at the
        previous time step).  These columns are passed through unchanged and
        appended at the end of the output.
        ``output_dim`` increases by ``n_context``.

    Attributes
    ----------
    B : jnp.ndarray, shape (n_features, fourier_input_dim)
        Fixed projection matrix.
    output_dim : int
        Width of the encoded output.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        n_features: int = 64,
        sigma: float = 1.0,
        seed: int = 0,
        include_input: bool = False,
        domain=None,
        encode_time: Optional[bool] = None,
        n_context: int = 0,
    ):
        if domain is not None and input_dim is not None:
            raise ValueError(
                "FourierFeatures: provide either 'input_dim' or 'domain', not both."
            )
        if domain is not None:
            t_interval = getattr(domain, "t_interval", None)
            self.has_time = t_interval is not None
            self.spatial_dims = domain._spatial_dims
            if self.has_time and encode_time is None:
                raise ValueError(
                    "FourierFeatures: the domain has a time dimension — "
                    "you must explicitly set encode_time=True or encode_time=False."
                )
            self.encode_time = bool(encode_time) if encode_time is not None else False
            # Dimension actually fed into the Fourier projection:
            fourier_input_dim = self.spatial_dims + (1 if (self.has_time and self.encode_time) else 0)
        else:
            if input_dim is None:
                raise ValueError(
                    "FourierFeatures: provide either 'input_dim' or 'domain'."
                )
            self.has_time = False
            self.spatial_dims = input_dim
            self.encode_time = False
            fourier_input_dim = input_dim

        self.input_dim = fourier_input_dim  # kept for backward compat / repr
        self._fourier_input_dim = fourier_input_dim
        self.n_features = n_features
        self.sigma = sigma
        self.include_input = include_input

        key = jax.random.PRNGKey(seed)
        self.B = jax.random.normal(key, (n_features, fourier_input_dim)) * sigma

        self.n_context = n_context
        fourier_out = 2 * n_features + (fourier_input_dim if include_input else 0)
        # When time is present but NOT Fourier-encoded, append a raw t column.
        self.output_dim = fourier_out + (1 if (self.has_time and not self.encode_time) else 0) + n_context

    def __call__(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """Encode *x*. Returns array of shape (batch, output_dim).

        The first ``spatial_dims`` columns of *x* are the spatial coordinates;
        if the domain has time the next column is ``t``; any remaining
        ``n_context`` columns are context fields that are passed through
        unchanged and appended at the end of the output.
        """
        coord_end = self.spatial_dims + (1 if (self.has_time and not self.encode_time) else
                                          (1 if self.has_time else 0))
        if self.has_time and not self.encode_time:
            # Fourier encode only spatial coordinates
            x_enc = x[:, : self.spatial_dims]
            t_col = x[:, self.spatial_dims : self.spatial_dims + 1]
        else:
            coord_end = self._fourier_input_dim
            x_enc = x[:, :coord_end]
            t_col = None

        Bx = x_enc @ self.B.T
        features = jnp.concatenate(
            [jnp.cos(2 * jnp.pi * Bx), jnp.sin(2 * jnp.pi * Bx)], axis=-1
        )
        if self.include_input:
            features = jnp.concatenate([x_enc, features], axis=-1)
        if t_col is not None:
            features = jnp.concatenate([features, t_col], axis=-1)
        if self.n_context > 0:
            ctx = x[:, -self.n_context:]
            features = jnp.concatenate([features, ctx], axis=-1)
        return features

    def transform(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """Alias for ``__call__``."""
        return self.__call__(x, params_dict)

    # ── ModelBase composable protocol ──────────────────────────────────────── #

    def _configure(self, network, input_dim: int) -> int:
        """Called by ModelBase.add().  FourierFeatures is pre-built in __init__,
        so this just validates and returns output_dim."""
        return self.output_dim

    def init(self, rng) -> dict:
        """No trainable parameters."""
        return {}

    def apply(self, params: dict, x, params_dict=None):
        """ModelBase-protocol alias: apply(params, x) → forward pass."""
        return self.__call__(x, params_dict)

    def __repr__(self) -> str:
        inc = ", include_input=True" if self.include_input else ""
        if self.has_time:
            et = f", encode_time={self.encode_time}"
        else:
            et = ""
        ctx = f", n_context={self.n_context}" if self.n_context > 0 else ""
        return (
            f"FourierFeatures(input_dim={self._fourier_input_dim}, "
            f"n_features={self.n_features}, sigma={self.sigma}{inc}{et}{ctx})"
        )


__all__ = ["FourierFeatures"]
