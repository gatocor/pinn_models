"""Optimizer registry for PINN trainers.

Available optimizers
--------------------
``"adam"``      – :class:`AdamOptimizer`
``"adamw"``     – :class:`AdamWOptimizer`
``"sgd"``       – :class:`SGDOptimizer`
``"rmsprop"``   – :class:`RMSPropOptimizer`
``"lion"``      – :class:`LionOptimizer`
``"lbfgs"``     – :class:`LBFGSOptimizer`
``"soap"``      – :class:`SOAPOptimizer`
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .base import BaseOptimizer
from .lbfgs import LBFGSOptimizer
from .optax_optimizers import (
    AdamOptimizer,
    AdamWOptimizer,
    LionOptimizer,
    RMSPropOptimizer,
    SGDOptimizer,
)
from .soap import SOAPOptimizer

__all__ = [
    "BaseOptimizer",
    "AdamOptimizer",
    "AdamWOptimizer",
    "SGDOptimizer",
    "RMSPropOptimizer",
    "LionOptimizer",
    "LBFGSOptimizer",
    "SOAPOptimizer",
    "OPTIMIZER_REGISTRY",
    "get_optimizer_class",
    "build_optimizer",
]

#: Mapping from name string to optimizer class.
OPTIMIZER_REGISTRY: Dict[str, Type[BaseOptimizer]] = {
    cls.name: cls
    for cls in (
        AdamOptimizer,
        AdamWOptimizer,
        SGDOptimizer,
        RMSPropOptimizer,
        LionOptimizer,
        LBFGSOptimizer,
        SOAPOptimizer,
    )
}


def get_optimizer_class(name: str) -> Type[BaseOptimizer]:
    """Return the optimizer class registered under *name*.

    Raises
    ------
    ValueError
        If *name* is not in :data:`OPTIMIZER_REGISTRY`.
    """
    try:
        return OPTIMIZER_REGISTRY[name.lower()]
    except KeyError:
        known = ", ".join(sorted(OPTIMIZER_REGISTRY))
        raise ValueError(
            f"Unknown optimizer '{name}'.  Available: {known}"
        )


def build_optimizer(
    name: str,
    learning_rate: float = 1e-3,
    *,
    grad_clip: Optional[float] = None,
    lr_scheduler: bool = False,
    **kwargs: Any,
):
    """Convenience: instantiate the optimizer class for *name* and call :py:meth:`~BaseOptimizer.build`.

    Parameters
    ----------
    name:
        Optimizer identifier string (e.g. ``"adam"``, ``"lbfgs"``).
    learning_rate:
        Initial learning rate (default 1e-3).
    grad_clip:
        Global-norm gradient clipping threshold, or *None*.
    lr_scheduler:
        Pass *True* if a learning-rate scheduler will be used.
    **kwargs:
        Extra keyword arguments forwarded to the optimizer *constructor*.
    """
    cls = get_optimizer_class(name)
    instance = cls(learning_rate, **kwargs)
    return instance.build(
        grad_clip=grad_clip,
        lr_scheduler=lr_scheduler,
    )
