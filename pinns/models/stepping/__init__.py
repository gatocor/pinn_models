"""
Time-stepping strategies for Physics-Informed Neural Networks.

Each strategy lives in its own module:

* :mod:`~pinns.models.stepping.stepper_dt`   — :class:`StepperDt`
* :mod:`~pinns.models.stepping.stepper_step` — :class:`StepperStep`
"""

from .stepper_dt   import StepperDt
from .stepper_step import StepperStep

# Tuple of all valid temporal strategy types — used for isinstance checks.
_TEMPORAL_STRATEGIES = (StepperStep,)

__all__ = [
    "StepperDt",
    "StepperStep",
    "_TEMPORAL_STRATEGIES",
]
