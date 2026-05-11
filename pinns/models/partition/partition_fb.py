"""Finite-Basis PINN (FB-PINN) spatial strategy."""

from __future__ import annotations

from typing import Optional
import numpy as np
import jax.numpy as jnp

__all__ = ["PartitionFB"]


class PartitionFB:
    """Finite-Basis PINN (FB-PINN) strategy.

    Each network covers one subdomain and its output is weighted by a smooth
    **window function** that peaks at the subdomain centre and decays to zero
    at the boundary, with the width of the transition zone controlled by
    *overlap*.  The global solution is the sum of all windowed predictions
    (partition of unity).

    Subdomain bounds are read automatically from the domain during
    :meth:`setup`, or can be specified explicitly via *xmin* / *xmax* when a
    single network covers only part of a larger domain (multi-network
    decomposition).

    Parameters
    ----------
    overlap : float
        Controls the width of the tanh transition zone as a fraction of each
        subdomain's spatial extent (0 = no smooth overlap; 0.5 = default).
    continuity_weight : float
        Weight on the interface-continuity penalty used by the Trainer.
        Default ``1.0``.
    xmin, xmax : array-like or None
        Explicit spatial bounds of this network's subdomain.  When ``None``
        (default), bounds are taken from the domain passed to :meth:`setup`.

    Example::

        # Single network for the whole domain:
        net = ModelBase(domain, output_dim=1, spatial=PartitionFB(overlap=0.3))

        # Multi-network decomposition (each network gets its own bounds):
        net_left  = ModelBase(domain, output_dim=1,
                            spatial=PartitionFB(overlap=0.3, xmin=[0], xmax=[0.5]))
        net_right = ModelBase(domain, output_dim=1,
                            spatial=PartitionFB(overlap=0.3, xmin=[0.5], xmax=[1.0]))
    """

    def __init__(
        self,
        overlap: float = 0.5,
        continuity_weight: float = 1.0,
        xmin=None,
        xmax=None,
    ):
        if not (0.0 <= overlap < 1.0):
            raise ValueError("PartitionFB: overlap must be in [0, 1).")
        if continuity_weight < 0.0:
            raise ValueError("PartitionFB: continuity_weight must be >= 0.")
        self.overlap = float(overlap)
        self.continuity_weight = float(continuity_weight)
        self._xmin: Optional[np.ndarray] = (
            np.asarray(xmin, dtype=np.float64) if xmin is not None else None
        )
        self._xmax: Optional[np.ndarray] = (
            np.asarray(xmax, dtype=np.float64) if xmax is not None else None
        )

    def setup(self, domain) -> None:
        """Read spatial bounds from *domain* (if not already set explicitly)."""
        if self._xmin is not None:
            return  # user provided explicit bounds
        n_s = domain._spatial_dims
        self._xmin = np.asarray(domain.xmin[:n_s], dtype=np.float64)
        self._xmax = np.asarray(domain.xmax[:n_s], dtype=np.float64)

    # ------------------------------------------------------------------
    # Window function
    # ------------------------------------------------------------------

    def _window(self, x_spatial):
        """Smooth bump window: product of tanh ramps over each spatial dim.

        ``w(x) = prod_d  tanh((x_d - xmin_d)/σ_d) * tanh((xmax_d - x_d)/σ_d)``

        where ``σ_d = max(overlap * (xmax_d - xmin_d), 1e-8)``.
        The result is in ``(0, 1]`` (maximum 1 at the domain centre).
        Shape: ``(batch, 1)``.
        """
        xmin = jnp.array(self._xmin, dtype=x_spatial.dtype)
        xmax = jnp.array(self._xmax, dtype=x_spatial.dtype)
        sigma = jnp.maximum(
            self.overlap * (xmax - xmin),
            jnp.full_like(xmax, 1e-8),
        )
        w = jnp.prod(
            jnp.tanh((x_spatial - xmin) / sigma)
            * jnp.tanh((xmax - x_spatial) / sigma),
            axis=-1,
            keepdims=True,
        )
        return w

    def predict(self, apply_fn, params, x, params_dict=None):
        """Forward pass: sequential apply weighted by the subdomain window.

        Parameters
        ----------
        apply_fn :
            The network's sequential forward pass
            (``params, x, params_dict → jnp.ndarray``).
        params : dict
            ModelBase parameter dict.
        x : jnp.ndarray  shape ``(batch, n_dims)``
            Collocation points (spatial + optional time).
        params_dict : dict or None
            Optional auxiliary dict forwarded to every layer.

        Returns
        -------
        jnp.ndarray  shape ``(batch, output_dim)``
            ModelBase output multiplied element-wise by the window function
            evaluated on the spatial coordinates.
        """
        if self._xmin is None:
            raise RuntimeError(
                "PartitionFB.predict() called before setup(). "
                "Either call network._setup(domain) or use problem.set_model(network)."
            )
        n_s = self._xmin.shape[0]
        x_spatial = x[:, :n_s]
        y = apply_fn(params, x, params_dict)
        w = self._window(x_spatial)
        return w * y

    def __repr__(self) -> str:
        parts = [f"overlap={self.overlap}", f"continuity_weight={self.continuity_weight}"]
        if self._xmin is not None:
            parts.append(f"xmin={self._xmin.tolist()}, xmax={self._xmax.tolist()}")
        return f"PartitionFB({', '.join(parts)})"
