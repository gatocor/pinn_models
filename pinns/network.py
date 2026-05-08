"""
Composable sequential Network for JAX PINNs.

All layer classes live in :mod:`pinns.layers`.  This module contains only
the :class:`Network` orchestrator.

Usage::

    from pinns.network import Network
    from pinns.layers import Normalize, Denormalize, FNN, GNNFeatures, ResNet
    from pinns.strategies import StrategyFB, StrategyStep

    net = Network(domain, output_dim=1)
    net.add(Normalize())
    net.add(GNNFeatures(hidden_dim=64, n_context=1))
    net.add(FNN([128, 128]))
    net.add(Denormalize())

    params = net.init(jax.random.PRNGKey(0))
    y = net.apply(params, x)

Layer protocol
--------------
Any object with the following three methods can be added to a Network:

``_configure(network, input_dim) -> int``
    Called by :meth:`Network.add`.  Receives the Network and the current data
    width; should finalise lazy init and return the new data width.

``init(rng) -> dict``
    Return a JAX pytree of trainable parameters (return ``{}`` if none).

``apply(params, x, params_dict=None) -> jnp.ndarray``
    Forward pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

# Re-export the common layers so ``from pinns.network import …`` still works.
from .layers.normalize import Normalize, Denormalize
from .layers.fnn       import FNN, WFFNN
from .layers.resnet    import ResNet
from .layers.piratenet import PirateNet
from .layers.lifting   import Lifting
from .strategies import (
    StrategyUnique, StrategyFB, StrategyX, StrategyStep,
    _SPATIAL_STRATEGIES, _TEMPORAL_STRATEGIES,
)


class NetworkLoss:
    """
    An architecture-driven loss term that belongs to a :class:`Network`,
    independent of the physical problem.

    The loss function ``fn(params, x) -> scalar`` is a JAX-differentiable
    callable that receives the **owning network's** current parameter dict
    and a batch of collocation points.  It must return a scalar loss value.

    Parameters
    ----------
    name : str
        Display name used in training logs and weight dictionaries.
    fn : callable
        ``fn(params, x) -> scalar`` — JAX-traceable loss function.
        ``params`` is the owning network's parameter dict;
        ``x`` is a ``jnp.ndarray`` of shape ``(batch, n_dims)``.
    weight : float
        Scalar multiplier applied to this loss before adding to the total.
        Default ``1.0``.
    x : array-like or None
        Pre-stored collocation points passed to ``fn`` every iteration.
        When ``None`` (default) the trainer uses the PDE collocation batch
        (``train_data['pde']``).

    Examples
    --------
    Hard-code a regularisation term::

        import jax.numpy as jnp
        loss = NetworkLoss(
            name='reg',
            fn=lambda params, x: jnp.mean(params['FNN_0']['kernel'] ** 2),
            weight=1e-4,
        )
        net.add_network_loss(loss)

    X-PINN interface continuity (use :func:`~pinns.strategies.register_interface_loss`
    instead of constructing this manually)::

        from pinns.strategies import register_interface_loss
        register_interface_loss(net_left, net_right, x_interface)
    """

    def __init__(
        self,
        name: str,
        fn: Callable,
        weight: float = 1.0,
        x=None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("NetworkLoss: name must be a non-empty string.")
        if not callable(fn):
            raise TypeError("NetworkLoss: fn must be callable.")
        if weight < 0.0:
            raise ValueError("NetworkLoss: weight must be >= 0.")
        self.name = name
        self.fn = fn
        self.weight = float(weight)
        self.x = jnp.asarray(x, dtype=jnp.float32) if x is not None else None

    def __repr__(self) -> str:
        x_info = f", x.shape={self.x.shape}" if self.x is not None else ""
        return f"NetworkLoss(name={self.name!r}, weight={self.weight}{x_info})"


class Network:
    """
    Composable sequential network for JAX PINNs.

    Parameters
    ----------
    domain :
        A ``DomainMesh`` (or cubic domain) object.  Passed to each layer that
        needs it (normalisation bounds, mesh vertices, etc.).
    output_dim : int
        Number of outputs of the network (injected into ``FNN`` as the final
        layer width).
    n_context : int
        Number of trailing context columns in the raw input tensor
        (e.g. ``u_t`` from a previous time step).  These are forwarded
        through normalisation and mesh-encoder layers.
    output_range : (ymin, ymax) or list of pairs, optional
        Physical output range used by ``Denormalize``.
    context_range : list of (min, max), optional
        Shorthand — forwarded to the first ``Normalize`` layer if the user
        does not set it manually.
    spatial : spatial strategy, optional
        One of :class:`~pinns.strategies.StrategyUnique` (default),
        :class:`~pinns.strategies.StrategyFB`, or
        :class:`~pinns.strategies.StrategyX`.
    temporal : :class:`~pinns.strategies.StrategyStep` or None
        Discrete time-stepping strategy.  ``None`` (default) means steady /
        continuous-time.

    Example::

        net = Network(domain, output_dim=1, n_context=1,
                      output_range=(0.0, 1.0),
                      context_range=[(0.0, 1.0)])
        net.add(Normalize())
        net.add(GNNFeatures(hidden_dim=64))
        net.add(FNN([128, 128]))
        net.add(Denormalize())

        params = net.init(jax.random.PRNGKey(0))
        y      = net.apply(params, x_with_context)

        # With strategies:
        net = Network(domain, output_dim=1,
                      spatial=StrategyFB(overlap=0.3),
                      temporal=StrategyStep())
    """

    def __init__(
        self,
        domain,
        output_dim: int,
        n_context: int = 0,
        output_range=None,
        context_range: Optional[List[Tuple[float, float]]] = None,
        spatial=None,
        temporal=None,
    ):
        # ── strategies --------------------------------------------------------
        if spatial is None:
            spatial = StrategyUnique()
        if not isinstance(spatial, _SPATIAL_STRATEGIES):
            raise TypeError(
                f"spatial must be StrategyUnique, StrategyFB, or StrategyX, "
                f"got {type(spatial).__name__!r}."
            )
        if temporal is not None and not isinstance(temporal, _TEMPORAL_STRATEGIES):
            raise TypeError(
                f"temporal must be a StrategyStep instance or None, "
                f"got {type(temporal).__name__!r}."
            )
        self.spatial: _SPATIAL_STRATEGIES = spatial
        self.temporal: Optional[StrategyStep] = temporal

        self.domain = domain
        self.output_dim = output_dim
        self.n_context = n_context
        self.output_range = output_range
        self.context_range = context_range

        self._layers: List[Any] = []
        self._layer_names: List[str] = []
        self._coord_transforms: List[Callable] = []
        self._network_losses: List[NetworkLoss] = []

        # Compute initial input_dim from domain
        spatial_dims = domain._spatial_dims
        t_interval = getattr(domain, "t_interval", None)
        self._current_dim = spatial_dims + (1 if t_interval is not None else 0) + n_context

        # Wire strategies to the domain immediately (domain is always present).
        self._setup(domain)

    # ── internal helpers ──────────────────────────────────────────────── #

    def _register_coord_transform(self, fn: Callable):
        """Called by Normalize to register its numpy-level transform."""
        self._coord_transforms.append(fn)

    def _apply_coord_transforms(self, coords: np.ndarray) -> np.ndarray:
        """Apply all accumulated coordinate transforms (numpy, for mesh build)."""
        for fn in self._coord_transforms:
            coords = fn(coords)
        return coords

    # ── public API ────────────────────────────────────────────────────── #

    def add(self, layer) -> "Network":
        """
        Add a layer to the network.

        Calls ``layer._configure(self, current_dim)`` which finalises any lazy
        initialisation and returns the new dimension.
        """
        # Inject context_range into Normalize if user did not set it
        if isinstance(layer, Normalize) and layer.context_range is None:
            layer.context_range = self.context_range

        new_dim = layer._configure(self, self._current_dim)
        self._current_dim = new_dim

        name = f"{type(layer).__name__}_{len(self._layers)}"
        self._layers.append(layer)
        self._layer_names.append(name)
        return self

    def init(self, rng: "jax.random.PRNGKey") -> Dict:
        """Initialise all trainable layers and return a merged params dict."""
        params = {}
        for name, layer in zip(self._layer_names, self._layers):
            rng, sub = jax.random.split(rng)
            sub_params = layer.init(sub)
            if sub_params:  # skip empty dicts (Normalize, Denormalize)
                params[name] = sub_params
        return params

    def _sequential_apply(
        self,
        params: Dict,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """Run layers in sequence without any strategy wrapping.

        Injects ``_x_orig`` into *params_dict* before the first layer so that
        :class:`~pinns.layers.Lifting` (and any other layer needing original
        coordinates) can access them regardless of how many transforms have
        been applied upstream.
        """
        if params_dict is None:
            params_dict = {}
        params_dict = dict(params_dict)   # shallow copy — don't mutate caller's dict
        params_dict.setdefault('_x_orig', x)  # original coords before Normalize etc.
        for name, layer in zip(self._layer_names, self._layers):
            sub_params = params.get(name, {})
            x = layer.apply(sub_params, x, params_dict)
        return x

    def apply(
        self,
        params: Dict,
        x: jnp.ndarray,
        params_dict: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """Forward pass through all layers, dispatched via the active strategy.

        The active strategy is :attr:`temporal` if set, otherwise
        :attr:`spatial`.  Each strategy's :meth:`~pinns.strategies.StrategyUnique.predict`
        method may wrap the sequential forward pass — e.g.
        :class:`~pinns.strategies.StrategyFB` multiplies the output by a smooth
        window function and :class:`~pinns.strategies.StrategyX` hard-masks
        output to zero outside the network's subdomain.
        """
        strategy = self.temporal if self.temporal is not None else self.spatial
        return strategy.predict(self._sequential_apply, params, x, params_dict)

    # ── network-level extra losses ─────────────────────────────────────── #

    def add_network_loss(self, loss: 'NetworkLoss') -> 'Network':
        """
        Register an architecture-driven loss term on this network.

        The term will be evaluated by the Trainer at every iteration and added
        to the total loss (multiplied by ``loss.weight``).

        Parameters
        ----------
        loss : NetworkLoss
            A :class:`NetworkLoss` instance describing the name, function,
            weight, and optional pre-stored collocation points.

        Returns
        -------
        Network
            ``self`` for method chaining.
        """
        if not isinstance(loss, NetworkLoss):
            raise TypeError(
                f"add_network_loss expects a NetworkLoss instance, "
                f"got {type(loss).__name__!r}."
            )
        self._network_losses.append(loss)
        return self

    @property
    def network_losses(self) -> List['NetworkLoss']:
        """All registered :class:`NetworkLoss` terms (read-only copy)."""
        return list(self._network_losses)

    # ── strategy helpers (mirrors Model API for ProblemStrong) ────────── #

    @property
    def is_stepping(self) -> bool:
        """``True`` if a :class:`~pinns.strategies.StrategyStep` is attached."""
        return self.temporal is not None

    @property
    def stepper(self) -> Optional[StrategyStep]:
        """The :class:`~pinns.strategies.StrategyStep`, or ``None``."""
        return self.temporal



    def _setup(self, domain) -> None:
        """Wire spatial and temporal strategies to *domain*."""
        self.spatial.setup(domain)
        if self.temporal is not None:
            self.temporal.setup(domain)

    def __repr__(self) -> str:
        parts = [f"output_dim={self.output_dim}"]
        if self.n_context:
            parts.append(f"n_context={self.n_context}")
        if not isinstance(self.spatial, StrategyUnique):
            parts.append(f"spatial={self.spatial!r}")
        if self.temporal is not None:
            parts.append(f"temporal={self.temporal!r}")
        lines = [f"Network({', '.join(parts)})"]
        for name, layer in zip(self._layer_names, self._layers):
            lines.append(f"  [{name}] {layer}")
        return "\n".join(lines)


__all__ = [
    "Network",
    "NetworkLoss",
    # re-exports from pinns.layers
    "Normalize", "Denormalize",
    "FNN", "WFFNN",
    "ResNet",
    "PirateNet",
    "Lifting",
]
