"""
PINNS - Physics-Informed Neural Networks (JAX/Flax/Optax)
"""

__version__ = "0.1.0"

# Domain
from .domain import DomainCubic, DomainMesh, SubdomainInfo, bump

# Boundary conditions
from .boundary import DirichletBC, NeumannBC, BoundaryConditions

# Strategies
from .models.partition import PartitionFB, PartitionX, register_interface_loss
from .models.stepping import StepperStep, StepperDt

# Problems
from .problems.problem_strong import Problem, ProblemStrong
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
    FNN, WFFNN, FourierFeatures, DenseRWF,
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
    "DirichletBC", "NeumannBC", "BoundaryConditions",
    "FNN", "WFFNN", "PirateNet", "ResNet", "FourierFeatures", "DenseRWF",
    "GNNFeatures", "LaplacianFeatures",
    "ModelBase", "NetworkLoss", "create_model", "ModelPartitioned", "ModelStepper",
    "PartitionFB", "PartitionX", "StepperStep", "StepperDt", "register_interface_loss",
    "Problem", "ProblemStrong", "ProblemWeak",
    "derivative", "gradient", "laplacian", "divergence",
    "Trainer",
    "LRScheduler", "ExponentialDecay", "ReduceLROnPlateau",
]
