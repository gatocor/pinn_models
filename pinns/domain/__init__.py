"""
pinns.domain — Domain definitions for PINNs.

Two domain types are provided:
* :class:`DomainCubic` — rectangular (hyper-cubic) domain, in
  ``pinns.domain.domain_cubic``.
* :class:`DomainMesh` — mesh-based domain, in
  ``pinns.domain.domain_mesh``.

All public names are re-exported here so existing code that does
``from pinns.domain import DomainCubic, DomainMesh, …`` continues to work.
"""

from .domain_cubic import (
    SubdomainInfo,
    DomainCubic,
    Stepper,
    bump,
    bump_vectorized,
    sample_unit_hypercube,
    transform_samples,
    SamplingMethod,
)
from .domain_mesh import DomainMesh

__all__ = [
    "SubdomainInfo",
    "DomainCubic",
    "DomainMesh",
    "Stepper",
    "bump",
    "bump_vectorized",
    "sample_unit_hypercube",
    "transform_samples",
    "SamplingMethod",
]
