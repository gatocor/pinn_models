"""
pinns/integrators/__init__.py — Spectral time integrators.

Available integrators:

* :class:`IntegratorETD2RK`  — Exponential Time Differencing 2nd-order RK
  (Cox-Matthews / Hochbruck-Ostermann).  Exact linear treatment — ideal for
  stiff spectral PDEs with diffuse linear parts.

* :class:`IntegratorRK4`     — Classical 4th-order Runge-Kutta.
  Treats the full RHS (L+N) explicitly.  Simple, but requires small ``dt``
  for stiff problems.

* :class:`IntegratorIMEX`    — First-order implicit-explicit (IMEX Euler).
  Linear part is implicit (pointwise division in spectral space), nonlinear
  part is explicit.  Unconditionally stable for the linear part.

* :class:`AdaptiveIntegrator` — Generic adaptive-step wrapper driven by a
  :class:`StepsizeController`.  Works with any integrator that implements
  ``_one_step`` (e.g. ``IntegratorETD2RK``, ``IntegratorIMEX``).  Uses
  ``jax.lax.scan`` so ``jax.grad`` works out of the box.

Available step-size controllers:

* :class:`PIDController`       — PI(α,β) Gustafsson controller.
* :class:`ConstantStepController` — always accepts, never changes ``dt``.

All integrators share the same ``solve(problem, inferred_params)`` interface
and are JAX-JIT / ``jax.grad`` compatible via ``jax.lax.scan``.
"""

from .integrator_base import Integrator
from .integrator_etd2rk import IntegratorETD2RK
from .integrator_adaptive import AdaptiveIntegrator
from .integrator_rk4 import IntegratorRK4
from .integrator_rk45 import IntegratorRK45
from .integrator_imex import IntegratorIMEX
from .integrator_diffrax import IntegratorDiffrax
from .integrator_euler import IntegratorEuler
from .integrator_dopri5 import IntegratorDopri5
from .integrator_tsit5 import IntegratorTsit5
from .stepsize_controller import StepsizeController, PIDController, ConstantStepController

__all__ = [
    "Integrator",
    "IntegratorETD2RK",
    "AdaptiveIntegrator",
    "IntegratorRK4",
    "IntegratorRK45",
    "IntegratorDiffrax",
    "IntegratorEuler",
    "IntegratorDopri5",
    "IntegratorTsit5",
    "IntegratorIMEX",
    "StepsizeController",
    "PIDController",
    "ConstantStepController",
]
