"""
Trainer package for PINN training.
"""

from .trainer import Trainer
from ._plotting import TrainPlotter
from .schedulers import (
    Scheduler, is_notebook,
    SchedulerExponentialDecay, SchedulerReduceLROnPlateau,
    SchedulerResample, SchedulerAdaptiveResample,
    SchedulerCurriculum, SchedulerLagrange,
    SchedulerGradNorm, SchedulerCausal,
    SchedulerWarmupDecay, SchedulerNTK,
)
from .optimizers import (
    BaseOptimizer,
    AdamOptimizer, AdamWOptimizer, SGDOptimizer, RMSPropOptimizer, LionOptimizer,
    LBFGSOptimizer, SOAPOptimizer,
    OPTIMIZER_REGISTRY, get_optimizer_class, build_optimizer,
)

__all__ = [
    "Trainer",
    "TrainPlotter",
    "Scheduler", "is_notebook",
    "SchedulerExponentialDecay", "SchedulerReduceLROnPlateau",
    "SchedulerResample", "SchedulerAdaptiveResample",
    "SchedulerCurriculum", "SchedulerLagrange",
    "SchedulerGradNorm", "SchedulerCausal",
    "SchedulerWarmupDecay", "SchedulerNTK",
    "BaseOptimizer",
    "AdamOptimizer", "AdamWOptimizer", "SGDOptimizer", "RMSPropOptimizer", "LionOptimizer",
    "LBFGSOptimizer", "SOAPOptimizer",
    "OPTIMIZER_REGISTRY", "get_optimizer_class", "build_optimizer",
]
