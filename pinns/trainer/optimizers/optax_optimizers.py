"""Optax-based optimizer wrappers (Adam, AdamW, SGD, RMSProp, Lion)."""
from __future__ import annotations

from typing import Optional

import optax

from .base import BaseOptimizer


def _wrap_optax(opt, grad_clip: Optional[float]) -> optax.GradientTransformation:
    """Optionally prepend gradient clipping."""
    if grad_clip is not None:
        return optax.chain(optax.clip_by_global_norm(grad_clip), opt)
    return opt


class AdamOptimizer(BaseOptimizer):
    """Adam optimizer (optax).

    Parameters
    ----------
    learning_rate:
        Initial learning rate (default 1e-3).
    b1:
        Exponential decay for the 1st moment (default 0.9).
    b2:
        Exponential decay for the 2nd moment (default 0.999).
    eps:
        Numerical stability constant (default 1e-8).
    """

    name = "adam"

    def __init__(
        self,
        learning_rate: float = 1e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__(learning_rate, **kwargs)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def build(
        self,
        *,
        grad_clip: Optional[float] = None,
        lr_scheduler: bool = False,
    ) -> optax.GradientTransformation:
        if lr_scheduler:
            opt = optax.inject_hyperparams(optax.adam)(
                learning_rate=self.learning_rate, b1=self.b1, b2=self.b2, eps=self.eps
            )
        else:
            opt = optax.adam(self.learning_rate, b1=self.b1, b2=self.b2, eps=self.eps)
        return _wrap_optax(opt, grad_clip)


class AdamWOptimizer(BaseOptimizer):
    """AdamW optimizer with weight decay (optax).

    Parameters
    ----------
    learning_rate:
        Initial learning rate (default 1e-3).
    b1:
        Exponential decay for the 1st moment (default 0.9).
    b2:
        Exponential decay for the 2nd moment (default 0.999).
    eps:
        Numerical stability constant (default 1e-8).
    weight_decay:
        L2 regularisation coefficient (default 1e-4).
    """

    name = "adamw"

    def __init__(
        self,
        learning_rate: float = 1e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__(learning_rate, **kwargs)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay

    def build(
        self,
        *,
        grad_clip: Optional[float] = None,
        lr_scheduler: bool = False,
    ) -> optax.GradientTransformation:
        if lr_scheduler:
            opt = optax.inject_hyperparams(optax.adamw)(
                learning_rate=self.learning_rate,
                b1=self.b1, b2=self.b2, eps=self.eps,
                weight_decay=self.weight_decay,
            )
        else:
            opt = optax.adamw(
                self.learning_rate,
                b1=self.b1, b2=self.b2, eps=self.eps,
                weight_decay=self.weight_decay,
            )
        return _wrap_optax(opt, grad_clip)


class SGDOptimizer(BaseOptimizer):
    """Stochastic Gradient Descent with optional momentum (optax).

    Parameters
    ----------
    learning_rate:
        Initial learning rate (default 1e-3).
    momentum:
        Momentum coefficient (default 0.0 → vanilla SGD).
    nesterov:
        Use Nesterov momentum (default False).
    """

    name = "sgd"

    def __init__(
        self,
        learning_rate: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(learning_rate, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov

    def build(
        self,
        *,
        grad_clip: Optional[float] = None,
        lr_scheduler: bool = False,
    ) -> optax.GradientTransformation:
        if lr_scheduler:
            opt = optax.inject_hyperparams(optax.sgd)(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        else:
            opt = optax.sgd(self.learning_rate, momentum=self.momentum, nesterov=self.nesterov)
        return _wrap_optax(opt, grad_clip)


class RMSPropOptimizer(BaseOptimizer):
    """RMSProp optimizer (optax).

    Parameters
    ----------
    learning_rate:
        Initial learning rate (default 1e-3).
    decay:
        Discounting factor for the history (default 0.9).
    eps:
        Numerical stability constant (default 1e-8).
    momentum:
        Momentum coefficient (default 0.0).
    nesterov:
        Use Nesterov momentum (default False).
    centered:
        Normalise by an estimate of the variance (default False).
    """

    name = "rmsprop"

    def __init__(
        self,
        learning_rate: float = 1e-3,
        decay: float = 0.9,
        eps: float = 1e-8,
        momentum: float = 0.0,
        nesterov: bool = False,
        centered: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(learning_rate, **kwargs)
        self.decay = decay
        self.eps = eps
        self.momentum = momentum
        self.nesterov = nesterov
        self.centered = centered

    def build(
        self,
        *,
        grad_clip: Optional[float] = None,
        lr_scheduler: bool = False,
    ) -> optax.GradientTransformation:
        if lr_scheduler:
            opt = optax.inject_hyperparams(optax.rmsprop)(
                learning_rate=self.learning_rate,
                decay=self.decay, eps=self.eps,
                momentum=self.momentum,
                nesterov=self.nesterov,
                centered=self.centered,
            )
        else:
            opt = optax.rmsprop(
                self.learning_rate,
                decay=self.decay, eps=self.eps,
                momentum=self.momentum,
                nesterov=self.nesterov,
                centered=self.centered,
            )
        return _wrap_optax(opt, grad_clip)


class LionOptimizer(BaseOptimizer):
    """Lion optimizer — memory-efficient alternative to Adam (optax).

    Parameters
    ----------
    b1:
        Decay for the update EMA (default 0.9).
    b2:
        Decay for the gradient EMA (default 0.99).
    weight_decay:
        L2 regularisation (default 0.0).
    """

    name = "lion"

    def __init__(
        self,
        b1: float = 0.9,
        b2: float = 0.99,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay

    def build(
        self,
        learning_rate: float,
        *,
        grad_clip: Optional[float] = None,
        lr_scheduler: bool = False,
    ) -> optax.GradientTransformation:
        if lr_scheduler:
            opt = optax.inject_hyperparams(optax.lion)(
                learning_rate=learning_rate,
                b1=self.b1, b2=self.b2,
                weight_decay=self.weight_decay,
            )
        else:
            opt = optax.lion(
                learning_rate,
                b1=self.b1, b2=self.b2,
                weight_decay=self.weight_decay,
            )
        return _wrap_optax(opt, grad_clip)
