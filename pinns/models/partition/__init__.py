"""
Spatial domain-decomposition strategies for Physics-Informed Neural Networks.

Each strategy lives in its own module:

* :mod:`~pinns.models.partition.partition_fb` — :class:`PartitionFB`
* :mod:`~pinns.models.partition.partition_x`  — :class:`PartitionX`, :func:`register_interface_loss`
"""

from .partition_fb import PartitionFB
from .partition_x  import PartitionX, register_interface_loss

# Tuple used for isinstance checks (excludes StrategyUnique — that's not a partition strategy).
_SPATIAL_STRATEGIES = (PartitionFB, PartitionX)

__all__ = [
    "PartitionFB",
    "PartitionX",
    "register_interface_loss",
    "_SPATIAL_STRATEGIES",
]
