"""Extended PINN (X-PINN) spatial strategy and interface-loss helper."""

from __future__ import annotations

from typing import Optional
import numpy as np
import jax.numpy as jnp

__all__ = ["PartitionX", "register_interface_loss"]


class PartitionX:
    """Extended PINN (X-PINN) strategy.

    Decomposes the domain into **non-overlapping** subdomains.  Each network
    covers exactly one subdomain; its output is **hard-masked to zero** for
    points that lie outside those bounds.  Interface residuals (continuity of
    the solution and optional normal-flux) are added to the loss by the
    Trainer.

    Subdomain bounds are read automatically from the domain during
    :meth:`setup`, or can be specified explicitly via *xmin* / *xmax*.

    Parameters
    ----------
    interface_weight : float
        Weight on the interface-continuity residuals.  Default ``1.0``.
    flux_weight : float
        Weight on the normal-flux-continuity residuals.  Default ``0.0``.
    xmin, xmax : array-like or None
        Explicit spatial bounds of this network's subdomain.  When ``None``
        (default), bounds are taken from the domain passed to :meth:`setup`.

    Example::

        net_left  = ModelBase(domain, output_dim=1,
                            spatial=PartitionX(interface_weight=10.0,
                                             xmin=[0], xmax=[0.5]))
        net_right = ModelBase(domain, output_dim=1,
                            spatial=PartitionX(interface_weight=10.0,
                                             xmin=[0.5], xmax=[1.0]))
    """

    def __init__(
        self,
        interface_weight: float = 1.0,
        flux_weight: float = 0.0,
        xmin=None,
        xmax=None,
    ):
        if interface_weight < 0.0:
            raise ValueError("PartitionX: interface_weight must be >= 0.")
        if flux_weight < 0.0:
            raise ValueError("PartitionX: flux_weight must be >= 0.")
        self.interface_weight = float(interface_weight)
        self.flux_weight = float(flux_weight)
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

    def predict(self, apply_fn, params, x, params_dict=None):
        """Forward pass: output zeroed for points outside the subdomain.

        Points inside the subdomain bounds receive the normal network
        prediction; points outside receive ``0.0``.  This lets the global
        loss accumulate contributions only from the responsible network at
        each point.

        Parameters
        ----------
        apply_fn :
            The network's sequential forward pass.
        params : dict
            ModelBase parameter dict.
        x : jnp.ndarray  shape ``(batch, n_dims)``
            Collocation points (spatial + optional time).
        params_dict : dict or None
            Optional auxiliary dict forwarded to every layer.

        Returns
        -------
        jnp.ndarray  shape ``(batch, output_dim)``
            ModelBase output, zeroed where ``x`` lies outside ``[xmin, xmax]``.
        """
        if self._xmin is None:
            raise RuntimeError(
                "PartitionX.predict() called before setup(). "
                "Either call network._setup(domain) or use problem.set_model(network)."
            )
        n_s = self._xmin.shape[0]
        x_spatial = x[:, :n_s]
        xmin = jnp.array(self._xmin, dtype=x_spatial.dtype)
        xmax = jnp.array(self._xmax, dtype=x_spatial.dtype)
        inside = jnp.all(
            (x_spatial >= xmin) & (x_spatial <= xmax),
            axis=-1,
            keepdims=True,
        )  # (batch, 1) bool
        y = apply_fn(params, x, params_dict)
        return jnp.where(inside, y, jnp.zeros_like(y))

    def __repr__(self) -> str:
        parts = [
            f"interface_weight={self.interface_weight}",
            f"flux_weight={self.flux_weight}",
        ]
        if self._xmin is not None:
            parts.append(f"xmin={self._xmin.tolist()}, xmax={self._xmax.tolist()}")
        return f"PartitionX({', '.join(parts)})"


def register_interface_loss(
    net_a,
    net_b,
    x_interface=None,
    *,
    strategy_a: 'PartitionX',
    strategy_b: Optional['PartitionX'] = None,
    name: str = "interface",
    weight: Optional[float] = None,
) -> None:
    """
    Register an X-PINN interface-continuity loss on two
    :class:`~pinns.models.model_base.ModelBase` instances.

    Adds a :class:`~pinns.models.model_base.NetworkLoss` to **both** networks so that
    each one tries to match the other's prediction at the interface.  The
    gradient flows only through the *owning* network's parameters;
    the other network's output is treated as a fixed target via
    :func:`jax.lax.stop_gradient` (alternating-optimisation style).

    Parameters
    ----------
    net_a, net_b : ModelBase
        The two networks sharing an interface.
    x_interface : array-like, shape ``(n_pts, n_dims)``
        Collocation points on the shared interface.
        When ``None``, uses the PDE collocation batch.
    strategy_a : PartitionX
        Strategy that owns *net_a*'s subdomain.
    strategy_b : PartitionX or None
        Strategy that owns *net_b*'s subdomain.  Defaults to *strategy_a*.
    name : str
        Name prefix for the two loss terms (suffixed with ``'_a'`` and ``'_b'``).
    weight : float or None
        Loss weight; defaults to ``strategy_a.interface_weight``.
    """
    from pinns.models.model_base import NetworkLoss

    import jax
    import jax.numpy as _jnp

    if not isinstance(strategy_a, PartitionX):
        raise TypeError(
            "register_interface_loss: strategy_a must be a PartitionX instance."
        )
    _strat_b = strategy_b if strategy_b is not None else strategy_a
    if not isinstance(_strat_b, PartitionX):
        raise TypeError(
            "register_interface_loss: strategy_b must be a PartitionX instance."
        )

    _weight = float(weight) if weight is not None else strategy_a.interface_weight
    _x = _jnp.asarray(x_interface, dtype=_jnp.float32) if x_interface is not None else None

    _seq_a = net_a._sequential_apply
    _seq_b = net_b._sequential_apply

    def _fn_a(params_a, x):
        u_a = strategy_a.predict(_seq_a, params_a, x)
        u_b = jax.lax.stop_gradient(
            _strat_b.predict(_seq_b, net_b.params, x)
        )
        return _jnp.mean((u_a - u_b) ** 2)

    def _fn_b(params_b, x):
        u_a = jax.lax.stop_gradient(
            strategy_a.predict(_seq_a, net_a.params, x)
        )
        u_b = _strat_b.predict(_seq_b, params_b, x)
        return _jnp.mean((u_a - u_b) ** 2)

    net_a.add_network_loss(NetworkLoss(name=f"{name}_a", fn=_fn_a, weight=_weight, x=_x))
    net_b.add_network_loss(NetworkLoss(name=f"{name}_b", fn=_fn_b, weight=_weight, x=_x))
