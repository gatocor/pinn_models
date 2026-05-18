"""
Random Fourier Feature encoding for Physics-Informed Neural Networks.

Based on Tancik et al. "Fourier Features Let Networks Learn High Frequency
Functions in Low Dimensional Domains" (NeurIPS 2020).
"""

import jax
import jax.numpy as jnp
from typing import List, Optional, Sequence


class RandomFourierFeatures:
    """
    Random Fourier Feature encoding to mitigate spectral bias in PINNs.

    The encoding maps the input columns to::

        [cos(B @ x_enc), sin(B @ x_enc), x_skip]

    where ``x_enc`` are all non-context columns **except** those listed in
    ``skip_transform``, ``x_skip`` are those columns passed through unchanged,
    and ``B ~ N(0, sigma²)`` is a random projection matrix (fixed or trainable).

    Like all layers, domain/input information is injected automatically when
    added to a :class:`~pinns.models.model_base.ModelBase`::

        # encode all incoming columns (e.g. after PeriodicEmbedding)
        net.add(RandomFourierFeatures(n_features=256, sigma=1.0))

        # keep column 1 (e.g. raw t) unencoded after a spatial embedding
        net.add(RandomFourierFeatures(n_features=256, sigma=1.0, skip_transform=[1]))

    Parameters
    ----------
    n_features : int
        Number of Fourier features; output width is ``2 * n_features``
        (plus any skipped / context columns).
    sigma : float
        Standard deviation for the random projection matrix B.
    seed : int
        Random seed for reproducible B initialisation.
    include_input : bool
        If ``True``, also prepend the encoded columns before the cos/sin block.
    skip_transform : sequence of int
        0-based column indices (into the layer's input, *before* context
        columns) that are **not** Fourier-encoded and are instead appended
        unchanged at the end of the output.  All other non-context columns
        are encoded together.
    adaptive : bool
        If ``True`` (default) the projection matrix ``B`` becomes a **trainable**
        parameter updated by the optimizer (like jaxpi's ``FourierEmbs``).
        If ``False`` ``B`` is fixed at initialisation.
    """

    def __init__(
        self,
        n_features: int = 64,
        sigma: float = 1.0,
        seed: int = 0,
        include_input: bool = False,
        skip_transform: Sequence[int] = (),
        adaptive: bool = True,
    ):
        self.n_features     = n_features
        self.sigma          = sigma
        self.seed           = seed
        self.include_input  = include_input
        self.skip_transform = list(skip_transform)
        self.adaptive       = adaptive

        # Set by _configure:
        self.B:                  Optional[jnp.ndarray] = None
        self.output_dim:         Optional[int]         = None
        self._encode_cols:       Optional[List[int]]   = None
        self._skip_cols:         Optional[List[int]]   = None
        self._fourier_input_dim: Optional[int]         = None
        self._n_context:         Optional[int]         = None

    # ── ModelBase composable protocol ──────────────────────────────────────── #

    def _configure(self, network, input_dim: int) -> int:
        """Called by ModelBase.add(). Resolves column splits from input_dim."""
        n_context     = network.n_context
        n_non_context = input_dim - n_context

        skip_set    = set(self.skip_transform)
        encode_cols = [i for i in range(n_non_context) if i not in skip_set]
        skip_cols   = [i for i in range(n_non_context) if i in skip_set]

        if not encode_cols:
            raise ValueError(
                "RandomFourierFeatures: all non-context columns are in skip_transform; "
                "nothing left to encode."
            )

        fourier_input_dim = len(encode_cols)

        key    = jax.random.PRNGKey(self.seed)
        self.B = jax.random.normal(key, (self.n_features, fourier_input_dim)) * self.sigma

        fourier_out = 2 * self.n_features + (fourier_input_dim if self.include_input else 0)
        out_dim     = fourier_out + len(skip_cols) + n_context

        self._encode_cols       = encode_cols
        self._skip_cols         = skip_cols
        self._fourier_input_dim = fourier_input_dim
        self._n_context         = n_context
        self.output_dim         = out_dim
        return out_dim

    def init(self, rng) -> dict:
        """Returns trainable params: {'B': array} if adaptive, else {}."""
        if self.adaptive:
            return {"B": self.B}
        return {}

    def apply(self, params: dict, x, params_dict=None):
        """ModelBase-protocol forward pass."""
        B = params["B"] if self.adaptive else None
        return self._forward(x, params_dict, B=B)

    # ── Forward pass ───────────────────────────────────────────────────────── #

    def _forward(self, x: jnp.ndarray, params_dict=None, B=None) -> jnp.ndarray:
        assert self.B is not None, "RandomFourierFeatures not configured — add to a ModelBase first"
        B = B if B is not None else self.B

        enc_idx = jnp.array(self._encode_cols)
        x_enc   = x[:, enc_idx]

        Bx       = x_enc @ B.T
        features = jnp.concatenate([jnp.cos(Bx), jnp.sin(Bx)], axis=-1)
        if self.include_input:
            features = jnp.concatenate([x_enc, features], axis=-1)
        if self._skip_cols:
            skip_idx = jnp.array(self._skip_cols)
            features = jnp.concatenate([features, x[:, skip_idx]], axis=-1)
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
            skip = f", skip_transform={self.skip_transform}" if self.skip_transform else ""
            return f"RandomFourierFeatures(n_features={self.n_features}, sigma={self.sigma}{skip})"
        skip = f", skip_transform={self._skip_cols}" if self._skip_cols else ""
        ctx  = f", n_context={self._n_context}" if self._n_context else ""
        adp  = ", adaptive=True" if self.adaptive else ""
        return (
            f"RandomFourierFeatures(fourier_input_dim={self._fourier_input_dim}, "
            f"n_features={self.n_features}, sigma={self.sigma}{skip}{ctx}{adp})"
        )


# Backward-compatible alias
FourierFeatures = RandomFourierFeatures

__all__ = ["RandomFourierFeatures", "FourierFeatures"]
