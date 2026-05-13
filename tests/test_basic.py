"""Basic tests for the pinns package — current API only."""

import numpy as np
import pytest


def test_import():
    """Package imports cleanly and exposes key symbols."""
    import pinns
    assert hasattr(pinns, '__version__')
    assert hasattr(pinns, 'FNN')
    assert hasattr(pinns, 'ModelBase')
    assert hasattr(pinns, 'DomainCubic')
    assert hasattr(pinns, 'DomainMesh')
    assert hasattr(pinns, 'TermDirichletBC')
    assert hasattr(pinns, 'TermNeumannBC')
    assert hasattr(pinns, 'TermCollection')
    assert hasattr(pinns, 'ProblemStrong')
    assert hasattr(pinns, 'ProblemWeak')


def test_domain_cubic():
    """DomainCubic creation and interior sampling."""
    from pinns import DomainCubic

    domain = DomainCubic(space=[(0.0, 1.0), (0.0, 1.0)])
    points = domain.sample_interior(500)
    assert points.shape == (500, 2)
    assert (points >= 0.0).all()
    assert (points <= 1.0).all()


def test_vanilla_network():
    """ModelBase + FNN layer creation and forward pass."""
    from pinns import ModelBase, FNN, DomainCubic
    import jax
    import jax.numpy as jnp

    domain = DomainCubic(space=[(0.0, 1.0), (0.0, 1.0)])
    net = ModelBase(domain, output_dim=1)
    net.add(FNN([32, 32], activation='tanh'))

    rng = jax.random.PRNGKey(0)
    params = net.init(rng)
    x = jnp.ones((100, 2))
    y = net.apply(params, x, {})

    assert y.shape == (100, 1)


def test_boundary_conditions():
    """TermDirichletBC / TermNeumannBC creation and TermCollection."""
    from pinns import TermDirichletBC, TermNeumannBC, TermCollection

    bc1 = TermDirichletBC(region='xmin', value=0.0, component=0, name='left')
    bc2 = TermNeumannBC(region='xmax', value=1.0, component=0, name='right')

    bcs = TermCollection()
    bcs.add(bc1)
    bcs.add(bc2)

    assert len(bcs) == 2
    assert bc1.region == 'xmin'
    assert bc1.value == 0.0
    assert bc2.region == 'xmax'
    assert bc2.name == 'right'


def test_term_points():
    """TermPoints with observation data (lives in pinns.dataset)."""
    from pinns import TermPoints

    x_obs = np.random.rand(50, 2).astype(np.float32)
    u_obs = np.zeros(50, dtype=np.float32)

    term = TermPoints(inputs=x_obs, outputs=u_obs, components=0, name='obs')
    assert term.inputs.shape == (50, 2)
    assert term.outputs.shape == (50, 1)
    assert term.components == [0]


def test_term_collection_repr():
    """TermCollection __repr__ works."""
    from pinns import TermDirichletBC, TermCollection

    col = TermCollection()
    col.add(TermDirichletBC(region='xmin', value=0.0))
    col.add(TermDirichletBC(region='xmax', value=1.0))
    r = repr(col)
    assert 'TermCollection' in r
    assert 'TermDirichletBC' in r
