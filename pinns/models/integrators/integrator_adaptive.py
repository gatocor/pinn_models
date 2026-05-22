"""
pinns/integrators/integrator_adaptive.py — Generic adaptive-step wrapper.

:class:`AdaptiveIntegrator` wraps **any** fixed-step integrator that
implements ``_one_step`` (e.g. :class:`IntegratorETD2RK`,
:class:`IntegratorIMEX`) and drives it with a
:class:`~pinns.models.integrators.StepsizeController`.

Usage::

    from pinns.models.integrators import AdaptiveIntegrator, IntegratorETD2RK, PIDController

    ctrl = PIDController(rtol=1e-5, atol=1e-7, order=2)
    ig   = AdaptiveIntegrator(
        integrator = IntegratorETD2RK(dt=1e-3),
        controller = ctrl,
        dt0        = 1e-3,
        max_steps  = 4000,
    )
    out = ig.solve(problem, inferred_params={"alpha": jnp.float32(0.5)})
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .integrator_base import Integrator
from .stepsize_controller import StepsizeController, PIDController

__all__ = ["AdaptiveIntegrator"]


class AdaptiveIntegrator(Integrator):
    """Generic adaptive-step wrapper for any fixed-step integrator.

    Takes an existing integrator that implements
    :meth:`~pinns.models.integrators.Integrator._one_step` and drives it with a
    :class:`~pinns.models.integrators.StepsizeController`.

    The wrapped integrator's original ``solve()`` (fixed-step scan) is never
    called — only its ``_one_step()`` is used.  The adaptive loop itself lives
    in :meth:`~pinns.models.integrators.Integrator._adaptive_solve` (shared base-class
    implementation).

    Args:
        integrator: Any :class:`~pinns.models.integrators.Integrator` subclass that
                    implements ``_one_step``.
        controller: Step-size controller.  Defaults to
                    ``PIDController(rtol=1e-4, atol=1e-6, order=2)``.
        dt0:        Initial step size.
        max_steps:  Maximum number of attempted steps (accepted + rejected).
        checkpoint: Wrap ``_one_step`` with ``jax.remat`` to trade
                    recomputation for peak memory during reverse-mode AD.
        rtol, atol: Convenience shortcuts — used only when *controller* is
                    ``None``.

    Example::

        from pinns.models.integrators import (
            AdaptiveIntegrator, IntegratorETD2RK, IntegratorIMEX, PIDController,
        )

        ctrl = PIDController(rtol=1e-5, atol=1e-7, order=2)

        ig_etd = AdaptiveIntegrator(IntegratorETD2RK(dt=1e-3), ctrl, dt0=1e-3)
        ig_imex = AdaptiveIntegrator(IntegratorIMEX(dt=1e-3),  ctrl, dt0=1e-3)
    """

    def __init__(
        self,
        integrator: Integrator,
        controller: Optional[StepsizeController] = None,
        dt0: float = 1e-3,
        max_steps: int = 2000,
        checkpoint: bool = False,
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ):
        self._integrator = integrator
        if controller is not None:
            self.controller = controller
        else:
            self.controller = PIDController(rtol=rtol, atol=atol, order=2)
        self.dt0        = float(dt0)
        self.max_steps  = int(max_steps)
        self.checkpoint = bool(checkpoint)

    @property
    def dt(self) -> float:
        """Alias for ``dt0``, required by :meth:`_get_obs_times`."""
        return self.dt0

    # ── Delegate _one_step to the wrapped integrator ───────────────────── #

    def _one_step(self, problem, state_hat, t, dt, L, params):
        """Delegate to the wrapped integrator's ``_one_step``."""
        return self._integrator._one_step(problem, state_hat, t, dt, L, params)

    # ── solve ──────────────────────────────────────────────────────────── #

    def solve(
        self,
        problem,
        inferred_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Adaptive-step forward solve using the wrapped integrator's step.

        Args:
            problem:         A fully-configured :class:`~pinns.models.ModelSpectralSolver`.
            inferred_params: Dict of JAX-differentiable parameter values.

        Returns:
            Dict ``{state_name: array(n_obs, *shape)}`` in physical space.
        """
        return self._adaptive_solve(
            problem         = problem,
            inferred_params = inferred_params,
            controller      = self.controller,
            dt0             = self.dt0,
            max_steps       = self.max_steps,
            checkpoint      = self.checkpoint,
        )

    def __repr__(self) -> str:
        return (
            f"AdaptiveIntegrator("
            f"integrator={self._integrator!r}, "
            f"controller={self.controller!r}, "
            f"dt0={self.dt0}, "
            f"max_steps={self.max_steps}, "
            f"checkpoint={self.checkpoint})"
        )
