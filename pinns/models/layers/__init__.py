"""
pinns.models.layers — All composable ModelBase layers.

Includes:
  - Normalisation:   Normalize, Denormalize
  - Feature encoders: RandomFourierFeatures, GNNFeatures, LaplacianFeatures, AlphaTransform
  - ModelBase models:  FNN, WFFNN, ResNet, PirateNet
"""

from .normalize import Normalize, Denormalize

from ._common  import DenseRWF
from .fnn      import FNN
from .wffnn    import WFFNN
from .resnet   import ResNet
from .piratenet import PirateNet

from .fourier   import RandomFourierFeatures, FourierFeatures
from .periodic  import PeriodicEmbedding
from .gnn       import GNNFeatures
from .laplacian import LaplacianFeatures, AlphaTransform

from .lifting import Lifting, CustomLifting

__all__ = [
    # normalisation
    "Normalize",
    "Denormalize",
    # building block
    "DenseRWF",
    # network architectures
    "FNN",
    "WFFNN",
    "ResNet",
    "PirateNet",
    # feature encoders
    "RandomFourierFeatures",
    "FourierFeatures",
    "PeriodicEmbedding",
    "GNNFeatures",
    "LaplacianFeatures",
    "AlphaTransform",
    # full mesh-based networks
    # hard enforcement
    "Lifting",
    "CustomLifting",
]
