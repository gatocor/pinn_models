"""
Base network models for PINNs.

Provides the three core neural-network architectures:

* :class:`FNN`       – Fully-connected Neural Network
* :class:`ResNet`    – Residual Network with pre-activation blocks
* :class:`PirateNet` – Physics-Informed Residual AdapTivE Network

All three share the same high-level API (``init``, ``apply``, ``predict``,
``set_input_range``, ``set_output_range``, ``to``), so they are
interchangeable as the *network* argument to :class:`~pinns.model.Model`.

Usage::

    from pinns.base_models import FNN, ResNet, PirateNet

    net = FNN([2, 64, 64, 1], activation='tanh')
    net = ResNet(input_dim=2, output_dim=1, hidden_dim=64, n_blocks=4)
    net = PirateNet(input_dim=2, output_dim=1, hidden_dim=64, n_blocks=3)

Feature encoders / input transforms live in :mod:`pinns.transformers`::

    from pinns.transformers import FourierFeatures, GNNFeatures, LaplacianFeatures
"""

from .fnn import FNN
from .resnet import ResNet
from .piratenet import PirateNet

__all__ = ["FNN", "ResNet", "PirateNet"]
