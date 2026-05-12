"""
PINNS - Physics-Informed Neural Networks (JAX/Flax/Optax)
"""

__version__ = "0.1.0"

# Domain
from .domain import DomainCubic, DomainMesh, SubdomainInfo, bump

# Residual terms (boundary conditions + interior / initial / data-point terms)
from .terms import (
    # BC classes
    TermDirichletBC, TermNeumannBC, TermRobinBC,
    TermCustomBC,
    TermPoints,
    TermPeriodicBC,
    TermCollection,
    TermOps,
    # Interior / initial / data-point PDE terms
    TermInner, TermInitial,
)

# Strategies
from .models.partition import PartitionFB, PartitionX, register_interface_loss
from .models.stepping import StepperStep, StepperDt

# Problems
from .problems.problem_strong import ProblemStrong
from .problems.problem_weak import ProblemWeak

# Utilities
from . import meshes
from .models import layers
from .models import model_base
from .models.model_base import ModelBase, NetworkLoss
from .models.model_utilities import create_model
from .models.model_partitioned import ModelPartitioned
from .models.model_stepper import ModelStepper

# Networks / layers
from .models.layers import (
    FNN, WFFNN, RandomFourierFeatures, FourierFeatures, DenseRWF,
    PirateNet, ResNet,
)
from .models.layers.gnn import GNNFeatures
from .models.layers.laplacian import LaplacianFeatures

# Functional (derivatives)
from .functional import derivative, gradient, laplacian, divergence

# Training
from .trainer import Trainer, LRScheduler, ExponentialDecay, ReduceLROnPlateau

__all__ = [
    "__version__",
    "meshes",
    "DomainCubic", "DomainMesh", "SubdomainInfo", "bump",
    # BC classes
    "TermDirichletBC", "TermNeumannBC", "TermRobinBC",
    "TermCustomBC",
    "TermPoints",
    "TermPeriodicBC",
    "TermCollection",
    "TermOps",
    # Interior / initial / data-point PDE terms
    "TermInner", "TermInitial",
    "FNN", "WFFNN", "PirateNet", "ResNet", "RandomFourierFeatures", "FourierFeatures", "DenseRWF",
    "GNNFeatures", "LaplacianFeatures",
    "ModelBase", "NetworkLoss", "create_model", "ModelPartitioned", "ModelStepper",
    "PartitionFB", "PartitionX", "StepperStep", "StepperDt", "register_interface_loss",
    "ProblemStrong", "ProblemWeak",
    "derivative", "gradient", "laplacian", "divergence",
    "Trainer",
    "LRScheduler", "ExponentialDecay", "ReduceLROnPlateau",
]
