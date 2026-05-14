"""Base class for PINN optimizers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseOptimizer(ABC):
    """Abstract base for optimizer wrappers.

    Subclass by:

    1. Declaring the class-level ``name`` string.
    2. Overriding ``__init__`` to accept and store optimizer-specific
       hyperparameters (call ``super().__init__()`` first).
    3. Overriding ``build`` to construct and return the underlying optimizer
       object (optax ``GradientTransformation``, jaxopt solver, …).

    Example usage::

        opt = AdamOptimizer(1e-3, b1=0.9, b2=0.99)
        tx  = opt.build(grad_clip=1.0, lr_scheduler=True)

    Or via the registry shorthand::

        tx = build_optimizer("adam", 1e-3, b1=0.9)
    """

    #: Short string identifier used in ``compile(optimizer=...)``.
    name: str = ""

    def __init__(self, learning_rate: float = 1e-3, **kwargs: Any) -> None:
        """Store learning rate and any extra keyword arguments.

        Subclasses should call ``super().__init__(learning_rate, **remaining_kwargs)``.
        """
        self.learning_rate = learning_rate
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.__init__() received unexpected "
                f"keyword arguments: {', '.join(kwargs)}"
            )

    @abstractmethod
    def build(
        self,
        *,
        grad_clip: Optional[float] = None,
        lr_scheduler: bool = False,
    ):
        """Return the constructed optimizer object.

        Parameters
        ----------
        grad_clip:
            If set, prepend ``optax.clip_by_global_norm(grad_clip)`` to the
            optimizer chain (ignored by non-optax optimizers).
        lr_scheduler:
            Whether a learning-rate scheduler will be attached.  When *True*
            optax-based optimizers are wrapped with ``optax.inject_hyperparams``
            so the learning rate can be mutated at runtime.
        """
        raise NotImplementedError

    def train_loop(
        self,
        trainer,
        epochs: int,
        print_each: int,
        show_plots: bool,
        save_plots,
        params_dict,
        weights,
    ) -> bool:
        """Optional custom training loop.

        Override in subclasses that require a loop different from the standard
        Adam/optax gradient-descent path (e.g. L-BFGS).

        Returns
        -------
        bool
            ``True`` if the loop was handled (trainer should return immediately);
            ``False`` to fall through to the standard loop.
        """
        return False
