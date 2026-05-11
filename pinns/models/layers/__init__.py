"""
pinns.models.layers — All composable ModelBase layers.

Includes:
  - Normalisation:   Normalize, Denormalize
  - Feature encoders: FourierFeatures, GNNFeatures, LaplacianFeatures, AlphaTransform
  - ModelBase models:  FNN, WFFNN, ResNet, PirateNet
"""

from .normalize import Normalize, Denormalize

from ._common  import DenseRWF
from .fnn      import FNN
from .wffnn    import WFFNN
from .resnet   import ResNet
from .piratenet import PirateNet

from .fourier   import FourierFeatures
from .gnn       import GNNFeatures
from .laplacian import LaplacianFeatures, AlphaTransform

from .lifting import Lifting

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
    "FourierFeatures",
    "GNNFeatures",
    "LaplacianFeatures",
    "AlphaTransform",
    # full mesh-based networks
    # hard enforcement
    "Lifting",
]
