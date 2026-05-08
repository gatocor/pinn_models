"""
pinns.layers — All composable Network layers.

Includes:
  - Normalisation:   Normalize, Denormalize
  - Feature encoders: FourierFeatures, GNNFeatures, LaplacianFeatures, AlphaTransform
  - Network models:  FNN, WFFNN, ResNet, PirateNet
"""

from .normalize import Normalize, Denormalize

from .fnn      import FNN, WFFNN
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
    # hard enforcement
    "Lifting",
]
