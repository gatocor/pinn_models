"""
pinns/integrators/stepsize_controller.py — Solver-agnostic step-size controllers.

Controllers are pure Python/JAX objects.  They do not know what ODE solver
is being used — they only see a scalar ``err_norm`` (RMS error / tolerance)
and the current ``dt``, and they return a proposed next ``dt`` and a boolean
``accept`` flag.

Usage pattern inside a JAX-traced adaptive loop::

    controller = PIDController(rtol=1e-5, atol=1e-7)
    ctrl_state = controller.init()

    # inside lax.scan body:
    dt_new, accept, ctrl_state = controller.step(err_norm, dt, ctrl_state)

Carry structure
---------------
``ctrl_state`` is a plain JAX scalar (float32) holding the *previous accepted*
error norm, used by the PI formula.  It starts at 1.0 (neutral, no prior step).
The ``ConstantStepController`` ignores it entirely.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import jax.numpy as jnp

__all__ = ["StepsizeController", "PIDController", "ConstantStepController"]


class StepsizeController(ABC):
    """Abstract base class for step-size controllers.

    All controllers must be JAX-traceable: ``step()`` is called inside
    ``jax.lax.scan`` and all operations must work on JAX arrays.
    """

    @abstractmethod
    def init(self):
        """Return the initial controller state (a JAX scalar)."""
        ...

    @abstractmethod
    def step(self, err_norm, dt, ctrl_state):
        """Compute the next step size and whether the current step is accepted.

        Args:
            err_norm:   Scalar JAX float — RMS error normalised by
                        ``rtol·|u| + atol`` (so accept iff err_norm ≤ 1.0).
            dt:         Current step size (JAX scalar).
            ctrl_state: Controller carry state from the previous call.

        Returns:
            Tuple ``(dt_new, accept, ctrl_state_new)`` where:
            - ``dt_new``       — proposed next step size (JAX scalar)
            - ``accept``       — bool JAX scalar, True iff the current step
                                 should be accepted
            - ``ctrl_state_new`` — updated carry state
        """
        ...


class PIDController(StepsizeController):
    """Standard PI(α, β) step-size controller (Gustafsson 1991).

    Uses the formula::

        scale  = safety · err_norm^{-α} · err_prev^{β}
        dt_new = clamp(dt · scale, fac_min·dt, fac_max·dt)

    with the classical choice α = 0.4/p, β = 0.3/p where p is the
    integrator order.  On rejection a simple I-controller (β=0) with
    ``fac_min`` scaling is used.

    Args:
        rtol:     Relative tolerance (used externally to build err_norm).
        atol:     Absolute tolerance (used externally to build err_norm).
        order:    Integrator order (default 2 for ETD2RK).
        safety:   Safety factor multiplying the scale (default 0.9).
        fac_min:  Minimum per-step scale factor (default 0.1).
        fac_max:  Maximum per-step scale factor (default 5.0).
        dt_min:   Hard lower bound on dt (default 1e-12).
        dt_max:   Hard upper bound on dt (default inf).
    """

    def __init__(
        self,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        order: int = 2,
        safety: float = 0.9,
        fac_min: float = 0.1,
        fac_max: float = 5.0,
        dt_min: float = 1e-12,
        dt_max: float = float("inf"),
    ):
        self.rtol    = float(rtol)
        self.atol    = float(atol)
        self.order   = int(order)
        self.safety  = float(safety)
        self.fac_min = float(fac_min)
        self.fac_max = float(fac_max)
        self.dt_min  = float(dt_min)
        self.dt_max  = float(dt_max)
        self._alpha  = 0.4 / order
        self._beta   = 0.3 / order

    def init(self):
        """Initial controller state: previous error norm = 1.0 (neutral)."""
        return jnp.float32(1.0)

    def step(self, err_norm, dt, ctrl_state):
        """PI step-size proposal.

        Returns:
            (dt_new, accept, err_norm_new)  where ``err_norm_new`` is the
            new carry state (only updated on accepted steps).
        """
        err_norm = jnp.maximum(err_norm, jnp.float32(1e-10))
        accept   = err_norm <= jnp.float32(1.0)

        # PI formula (accepted) or aggressive shrink (rejected)
        scale_acc = (self.safety
                     * err_norm ** (-self._alpha)
                     * ctrl_state ** self._beta)
        scale_rej = self.safety * self.fac_min
        scale     = jnp.where(accept, scale_acc, scale_rej)
        scale     = jnp.clip(scale, self.fac_min, self.fac_max)

        dt_new = jnp.clip(dt * scale,
                          jnp.float32(self.dt_min),
                          jnp.float32(self.dt_max))

        # Update carry only on accepted steps
        ctrl_state_new = jnp.where(accept, err_norm, ctrl_state)

        return dt_new, accept, ctrl_state_new

    def __repr__(self) -> str:
        return (
            f"PIDController(rtol={self.rtol}, atol={self.atol}, "
            f"order={self.order}, safety={self.safety})"
        )


class ConstantStepController(StepsizeController):
    """Trivial controller that always accepts and keeps dt fixed.

    Useful for fixed-step methods that still go through the adaptive
    scan infrastructure (e.g. to record observations uniformly).
    """

    def init(self):
        return jnp.float32(1.0)

    def step(self, err_norm, dt, ctrl_state):
        """Always accept, never change dt."""
        accept = jnp.bool_(True)
        return dt, accept, ctrl_state

    def __repr__(self) -> str:
        return "ConstantStepController()"
