"""SOAP optimizer wrapper (soap_jax)."""
from __future__ import annotations

from typing import Optional

from .base import BaseOptimizer


class SOAPOptimizer(BaseOptimizer):
    """SOAP (Second-Order Adaptive Preconditioner) optimizer via soap_jax.

    Parameters
    ----------
    b1:
        Adam-style beta1 (default 0.95).
    b2:
        Adam-style beta2 (default 0.95).
    shampoo_beta:
        Shampoo-style beta; set to -1 to reuse ``b2`` (default -1).
    eps:
        Numerical stability constant (default 1e-8).
    weight_decay:
        L2 regularisation coefficient (default 0.0).
    precondition_frequency:
        How often to update the preconditioner (default 10).
    max_precond_dim:
        Maximum parameter dimension for which the preconditioner is computed
        (default 10000).
    precondition_1d:
        Whether to precondition 1-D parameter tensors (default False).
    """

    name = "soap"

    def __init__(
        self,
        learning_rate: float = 1e-3,
        b1: float = 0.95,
        b2: float = 0.95,
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        precondition_1d: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(learning_rate, **kwargs)
        self.b1 = b1
        self.b2 = b2
        self.shampoo_beta = shampoo_beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.precondition_frequency = precondition_frequency
        self.max_precond_dim = max_precond_dim
        self.precondition_1d = precondition_1d

    def build(
        self,
        *,
        grad_clip: Optional[float] = None,
        lr_scheduler: bool = False,
    ):
        """Build and return the SOAP optimizer."""
        try:
            from soap_jax import soap as soap_optimizer
        except ImportError as exc:
            raise ImportError(
                "SOAP requires soap_jax.  Install with:\n"
                "  pip install git+https://github.com/haydn-jones/SOAP_JAX"
            ) from exc

        opt = soap_optimizer(
            learning_rate=self.learning_rate,
            b1=self.b1,
            b2=self.b2,
            shampoo_beta=self.shampoo_beta,
            eps=self.eps,
            weight_decay=self.weight_decay,
            precondition_frequency=self.precondition_frequency,
            max_precond_dim=self.max_precond_dim,
            precondition_1d=self.precondition_1d,
        )

        if grad_clip is not None:
            import optax
            return optax.chain(optax.clip_by_global_norm(grad_clip), opt)
        return opt
