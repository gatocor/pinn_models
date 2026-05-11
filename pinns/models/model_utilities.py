"""
Factory function for building standard PINN architectures.

:func:`create_model` assembles::

    Normalize  →  [Features]  →  Core  →  [Denormalize]  →  [Lifting …]

and optionally wraps the result with :class:`~pinns.models.model_partitioned.ModelPartitioned`
(``partition=PartitionFB(…)`` or ``partition=PartitionX(…)``) and/or
:class:`~pinns.models.model_stepper.ModelStepper` (``stepper=True``).

Return types
------------
* No *partition*, no *stepper*  → :class:`~pinns.models.model_base.ModelBase`
* *partition* given, no *stepper* → :class:`~pinns.models.model_partitioned.ModelPartitioned`
* *stepper* only               → :class:`~pinns.models.model_stepper.ModelStepper` wrapping
  a :class:`~pinns.models.model_base.ModelBase`
* Both *partition* and *stepper* → :class:`~pinns.models.model_stepper.ModelStepper` wrapping
  a :class:`~pinns.models.model_partitioned.ModelPartitioned`
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from .model_base import ModelBase
from .layers.normalize import Normalize, Denormalize
from .layers.fnn import FNN
from .model_partitioned import ModelPartitioned
from .model_stepper import ModelStepper
from .partition import PartitionFB, PartitionX, _SPATIAL_STRATEGIES
from .stepping  import StepperStep, StepperDt, _TEMPORAL_STRATEGIES


def create_model(
    domain,
    output_dim: int,
    *,
    hidden_dims: Sequence[int] = (64, 64, 64),
    core=None,
    features=None,
    activation: str = "tanh",
    normalize: bool = True,
    denormalize: bool = False,
    output_range=None,
    n_context: int = 0,
    context_range: Optional[List[Tuple[float, float]]] = None,
    partition: Optional[Union[PartitionFB, PartitionX]] = None,
    partition_time: bool = True,
    stepper: bool = False,
    stepper_strategy: Optional[StepperDt] = None,
) -> Union[ModelBase, ModelPartitioned, ModelStepper]:
    """
    Build a standard PINN architecture and optionally wrap it for
    domain decomposition and/or autoregressive time-stepping.

    Architecture assembled for the inner :class:`~pinns.models.model_base.ModelBase`::

        Normalize  →  [features]  →  Core  →  [Denormalize]

    Hard boundary constraints can be added to the returned object via
    :meth:`~pinns.models.model_base.ModelBase.add_constraint`.

    Parameters
    ----------
    domain :
        A domain object (``DomainMesh`` or ``DomainCubic``).
    output_dim : int
        Number of scalar outputs.
    hidden_dims : sequence of int
        Hidden layer widths for the default ``FNN`` core.
        Ignored when *core* is given.  Default ``(64, 64, 64)``.
    core : layer or None
        Explicit core layer (e.g. ``PirateNet(…)``, ``ResNet(…)``).
        When ``None`` an ``FNN(hidden_dims, activation=activation)`` is used.
    features : layer or None
        Optional feature-extraction layer inserted before the core.
        Default ``None``.
    activation : str
        Activation for the default ``FNN`` core.  Default ``'tanh'``.
    normalize : bool
        Prepend a :class:`~pinns.models.layers.Normalize` layer.  Default ``True``.
    denormalize : bool
        Append a :class:`~pinns.models.layers.Denormalize` layer.  Requires
        *output_range*.  Default ``False``.
    output_range : (ymin, ymax) or list of pairs, optional
        Physical output range used by ``Denormalize``.
    n_context : int
        Number of trailing context columns in the input.  Automatically set
        to *output_dim* when ``stepper=True``.  Default ``0``.
    context_range : list of ``(min, max)`` pairs, optional
        Physical range of the context columns for normalisation.
        Forwarded to the first ``Normalize`` layer.  When ``stepper=True``
        this is also passed to :class:`~pinns.models.model_stepper.ModelStepper`.
    partition : ``PartitionFB`` | ``PartitionX`` | None
        When given, wraps the :class:`~pinns.models.model_base.ModelBase` with a
        :class:`~pinns.models.model_partitioned.ModelPartitioned` using this strategy
        as the subdomain prototype.  Cannot be combined with *spatial*.
    partition_time : bool
        Forwarded to :class:`~pinns.models.model_partitioned.ModelPartitioned`.
        Default ``True``.
    stepper : bool
        When ``True``, wraps the result with
        :class:`~pinns.models.model_stepper.ModelStepper`.  Sets ``n_context`` to
        *output_dim* automatically and passes *context_range* to it.
        Default ``False``.

    Returns
    -------
    ModelBase | ModelPartitioned | ModelStepper

    Raises
    ------
    ValueError
        If ``denormalize=True`` without *output_range*.

    Examples
    --------
    Minimal — single network with defaults::

        m = create_model(domain, output_dim=1)
        params = m.init(jax.random.PRNGKey(0))
        y = m.apply(params, x)

    FB-PINN with spatial domain decomposition::

        from pinns.models.partition import PartitionFB
        m = create_model(domain, output_dim=1,
                         partition=PartitionFB(overlap=0.3))
        # m is a ModelPartitioned

    Autoregressive stepper::

        m = create_model(domain, output_dim=1,
                         context_range=[(0.0, 1.0)],
                         stepper=True)
        # m is a ModelStepper
        traj = m.rollout(params, x_spatial, t_values, u0)

    Partitioned + stepper::

        m = create_model(domain, output_dim=2,
                         partition=PartitionFB(overlap=0.3),
                         context_range=[(-1., 1.), (-1., 1.)],
                         stepper=True)
        # m is a ModelStepper wrapping a ModelPartitioned
    """
    # ── Validate ──────────────────────────────────────────────────────

    if denormalize and output_range is None:
        raise ValueError(
            "create_model: denormalize=True requires output_range to be specified."
        )
    if stepper:
        # ModelStepper requires n_context == output_dim.
        n_context = output_dim

    # ── Build the inner ModelBase ───────────────────────────────────────
    base = ModelBase(
        domain,
        output_dim=output_dim,
        n_context=n_context,
        output_range=output_range if denormalize else None,
        context_range=context_range,
    )

    if normalize:
        base.add(Normalize())

    if features is not None:
        base.add(features)

    if core is not None:
        base.add(core)
    else:
        base.add(FNN(list(hidden_dims), activation=activation))

    if denormalize:
        base.add(Denormalize())

    # ── Optionally wrap with ModelPartitioned ──────────────────────────
    if partition is not None:
        result: Union[ModelBase, ModelPartitioned] = ModelPartitioned(
            base, partition, partition_time=partition_time
        )
    else:
        result = base

    # ── Optionally wrap with ModelStepper ──────────────────────────────
    if stepper:
        return ModelStepper(result, context_range=context_range,
                            strategy=stepper_strategy)

    return result

__all__ = ["create_model"]
