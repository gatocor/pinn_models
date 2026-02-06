"""
PINNS - Physics-Informed Neural Networks

A PyTorch-based library for Physics-Informed Neural Networks (PINN) 
and Finite Basis PINN (FBPINN).
"""

__version__ = "0.1.0"

from .domain import DomainCubic, DomainCubicPartition, SubdomainInfo, bump
from .networks import FNN, FBPINN
from .problem import Problem
from .functional import derivative, gradient, laplacian, divergence
from .trainer import Trainer

__all__ = [
    # Version
    "__version__",
    # Domain
    "DomainCubic",
    "DomainCubicPartition",
    "SubdomainInfo",
    "bump",
    # Networks
    "FNN",
    "FBPINN",
    # Problems
    "Problem",
    # Functional
    "derivative",
    "gradient",
    "laplacian",
    "divergence",
    # Training
    "Trainer",
]