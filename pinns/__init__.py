"""
PINNS - Physics-Informed Neural Networks (JAX/Flax/Optax)
"""

__version__ = "0.1.0"

# Domain
from .domain import DomainCubic, DomainMesh, bump

# Residual terms (boundary conditions + interior / initial / data-point terms)
from .problems.terms import (
    # BC classes
    TermDirichletBC, TermNeumannBC, TermRobinBC,
    TermCustomBC,
    TermPeriodicBC,
    TermCollection,
    # Interior / initial PDE terms
    TermInner, TermInitial,
)

# Dataset (fixed observation / measurement data)
from .dataset import Dataset, TermPoints

# Strategies
from .models.partition import PartitionFB, PartitionX, register_interface_loss
from .models.stepping import StepperStep, StepperDt

# Problems
from .problems.problem_strong import ProblemStrong
from .problems.problem_weak import ProblemWeak

# Integrators
from .models.integrators import (
    Integrator,
    IntegratorETD2RK,
    AdaptiveIntegrator,
    IntegratorRK4,
    IntegratorRK45,
    IntegratorIMEX,
    IntegratorDiffrax,
    IntegratorEuler,
    IntegratorDopri5,
    IntegratorTsit5,
    StepsizeController,
    PIDController,
    ConstantStepController,
)

# Utilities
from . import meshes
from .models import layers
from .models import model_base
from .models.model_base import ModelBase, NetworkLoss
from .models.model_utilities import create_model
from .models.model_partitioned import ModelPartitioned
from .models.model_stepper import ModelStepper
from .models.model_solver import ModelSpectralSolver
from .models.model_fem_solver import ModelFEMSolver
from .models.nonlinear import NonlinearSolverBase, NonlinearSolverNone, NonlinearSolverNewton, NonlinearSolverPicard
from .models.linear import LinearSolverBase, LinearSolverGMRES, LinearSolverDirect

# Networks / layers
from .models.layers import (
    FNN, WFFNN, RandomFourierFeatures, FourierFeatures, DenseRWF,
    PirateNet, ResNet, PeriodicEmbedding,
    Normalize, Denormalize,
)
from .models.layers.gnn import GNNFeatures
from .models.layers.laplacian import LaplacianFeatures

# Functional (derivatives)
from .functional import derivative, gradient, laplacian, divergence

# Training
from .trainer import (Trainer, TrainPlotter,
                      Scheduler, is_notebook,
                      SchedulerExponentialDecay, SchedulerReduceLROnPlateau,
                      SchedulerResample, SchedulerAdaptiveResample,
                      SchedulerCurriculum, SchedulerLagrange,
                      SchedulerGradNorm, SchedulerCausal,
                      SchedulerWarmupDecay, SchedulerNTK,
                      SchedulerPartition, MaskedState, make_masked_optimizer,
                      BaseOptimizer,
                      AdamOptimizer, AdamWOptimizer, SGDOptimizer, RMSPropOptimizer,
                      LionOptimizer, LBFGSOptimizer, SOAPOptimizer)

__all__ = [
    "__version__",
    "meshes",
    "DomainCubic", "DomainMesh", "bump",
    # BC classes
    "TermDirichletBC", "TermNeumannBC", "TermRobinBC",
    "TermCustomBC",
    "Dataset", "TermPoints",
    "TermPeriodicBC",
    "TermCollection",
    # Interior / initial / data-point PDE terms
    "TermInner", "TermInitial",
    "FNN", "WFFNN", "PirateNet", "ResNet", "RandomFourierFeatures", "FourierFeatures", "DenseRWF", "PeriodicEmbedding",
    "Normalize", "Denormalize",
    "GNNFeatures", "LaplacianFeatures",
    "ModelBase", "NetworkLoss", "create_model", "ModelPartitioned", "ModelStepper", "ModelSpectralSolver",
    "PartitionFB", "PartitionX", "StepperStep", "StepperDt", "register_interface_loss",
    "NonlinearSolverBase", "NonlinearSolverNone", "NonlinearSolverNewton", "NonlinearSolverPicard",
    "LinearSolverBase", "LinearSolverGMRES", "LinearSolverDirect",
    "ProblemStrong", "ProblemWeak",
    "Integrator", "IntegratorETD2RK", "AdaptiveIntegrator",
    "IntegratorRK4", "IntegratorRK45", "IntegratorIMEX", "IntegratorDiffrax",
    "IntegratorEuler", "IntegratorDopri5", "IntegratorTsit5",
    "derivative", "gradient", "laplacian", "divergence",
    "Trainer", "TrainPlotter",
    "Scheduler", "is_notebook",
    "SchedulerExponentialDecay", "SchedulerReduceLROnPlateau",
    "SchedulerResample", "SchedulerAdaptiveResample",
    "SchedulerCurriculum", "SchedulerLagrange",
    "SchedulerGradNorm", "SchedulerCausal",
    "SchedulerWarmupDecay", "SchedulerNTK",
    "SchedulerPartition", "MaskedState", "make_masked_optimizer",
    "BaseOptimizer",
    "AdamOptimizer", "AdamWOptimizer", "SGDOptimizer", "RMSPropOptimizer",
    "LionOptimizer", "LBFGSOptimizer", "SOAPOptimizer",
]
