"""Tests for DomainCubic — construction, sampling, region registration, and utilities."""

import numpy as np
import pytest
from pinns.domain import DomainCubic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _in_box(pts, lo, hi):
    """Return True when every point in *pts* lies in [lo, hi]."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return bool(np.all(pts >= lo - 1e-10) and np.all(pts <= hi + 1e-10))


# ===========================================================================
# Construction
# ===========================================================================

class TestConstruction:
    def test_plain_1d(self):
        d = DomainCubic([(0, 1)])
        assert d._spatial_dims == 1
        assert d.n_dims == 1
        assert d.grid_positions is None
        np.testing.assert_allclose(d.xmin, [0.0])
        np.testing.assert_allclose(d.xmax, [1.0])

    def test_plain_2d(self):
        d = DomainCubic([(0, 1), (-1, 2)])
        assert d._spatial_dims == 2
        np.testing.assert_allclose(d.xmin, [0.0, -1.0])
        np.testing.assert_allclose(d.xmax, [1.0,  2.0])

    def test_plain_3d(self):
        d = DomainCubic([(0, 1), (0, 1), (0, 1)])
        assert d._spatial_dims == 3

    def test_partitioned_1d(self):
        d = DomainCubic([[0, 0.5, 1]])
        assert d._spatial_dims == 1
        assert d.n_subdomains == 2
        assert d.n_subdomains_per_dim == [2]

    def test_partitioned_2d(self):
        d = DomainCubic([[0, 0.5, 1], [0, 0.5, 1]])
        assert d.n_subdomains == 4
        assert d.n_subdomains_per_dim == [2, 2]

    def test_partitioned_3d(self):
        d = DomainCubic([[0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1]])
        assert d.n_subdomains == 8

    def test_with_continuous_time(self):
        d = DomainCubic([(0, 1)], time=[0, 1])
        assert d.has_time
        assert not d.is_time_partitioned
        assert d.n_dims == 2
        assert d._t_min == 0.0
        assert d._t_max == 1.0
        assert d.time_grid_positions is None

    def test_with_partitioned_time(self):
        d = DomainCubic([[0, 0.5, 1]], time=[0, 0.5, 1])
        assert d.has_time
        assert d.is_time_partitioned
        assert d.n_time_subdomains == 2
        assert len(d.time_grid_positions) == 3

    def test_invalid_empty_space(self):
        with pytest.raises(ValueError):
            DomainCubic([])

    def test_invalid_mixed_spec(self):
        with pytest.raises(ValueError):
            DomainCubic([(0, 1), [0, 0.5, 1]])

    def test_invalid_min_ge_max(self):
        with pytest.raises(ValueError):
            DomainCubic([(1, 0)])

    def test_invalid_non_increasing_breakpoints(self):
        with pytest.raises(ValueError):
            DomainCubic([[1, 0.5, 0]])

    def test_invalid_time_non_increasing(self):
        with pytest.raises(ValueError):
            DomainCubic([(0, 1)], time=[1, 0])

    def test_invalid_time_single_value(self):
        with pytest.raises(ValueError):
            DomainCubic([(0, 1)], time=[0.5])


# ===========================================================================
# Properties
# ===========================================================================

class TestProperties:
    def test_bounds(self):
        d = DomainCubic([(0, 2), (1, 3)])
        lo, hi = d.bounds
        np.testing.assert_allclose(lo, [0.0, 1.0])
        np.testing.assert_allclose(hi, [2.0, 3.0])

    def test_extents(self):
        d = DomainCubic([(0, 2), (1, 4)])
        np.testing.assert_allclose(d.extents, [2.0, 3.0])

    def test_volume(self):
        d = DomainCubic([(0, 2), (0, 3)])
        assert pytest.approx(d.volume) == 6.0

    def test_len_partitioned(self):
        d = DomainCubic([[0, 0.5, 1], [0, 0.5, 1]])
        assert len(d) == 4

    def test_len_plain_raises(self):
        d = DomainCubic([(0, 1)])
        with pytest.raises(AttributeError):
            len(d)

    def test_has_time_false(self):
        assert not DomainCubic([(0, 1)]).has_time

    def test_has_time_true(self):
        assert DomainCubic([(0, 1)], time=[0, 1]).has_time

    def test_is_time_partitioned_false(self):
        assert not DomainCubic([(0, 1)], time=[0, 1]).is_time_partitioned

    def test_is_time_partitioned_true(self):
        assert DomainCubic([(0, 1)], time=[0, 0.5, 1]).is_time_partitioned


# ===========================================================================
# contains
# ===========================================================================

class TestContains:
    d = DomainCubic([(0, 1), (0, 1)])

    def test_inside(self):
        assert self.d.contains(np.array([0.5, 0.5]))

    def test_on_boundary(self):
        assert self.d.contains(np.array([0.0, 1.0]))

    def test_outside(self):
        assert not self.d.contains(np.array([1.5, 0.5]))

    def test_batch(self):
        pts = np.array([[0.5, 0.5], [2.0, 0.5], [0.1, 0.1]])
        result = self.d.contains(pts)
        np.testing.assert_array_equal(result, [True, False, True])


# ===========================================================================
# Partition utilities
# ===========================================================================

class TestPartitionUtils:
    d = DomainCubic([[0, 0.5, 1], [0, 0.5, 1]])

    def test_subdomain_centers_shape(self):
        c = self.d.get_subdomain_centers()
        assert c.shape == (4, 2)

    def test_subdomain_centers_values(self):
        c = self.d.get_subdomain_centers()
        expected = {(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)}
        got = {tuple(row) for row in c}
        assert got == expected

    def test_subdomain_bounds_shape(self):
        lo, hi = self.d.get_subdomain_bounds()
        assert lo.shape == (4, 2)
        assert hi.shape == (4, 2)

    def test_get_multi_index(self):
        # 2×2 grid: flat 0→(0,0), 1→(0,1), 2→(1,0), 3→(1,1)
        assert self.d.get_multi_index(0) == (0, 0)
        assert self.d.get_multi_index(1) == (0, 1)
        assert self.d.get_multi_index(2) == (1, 0)
        assert self.d.get_multi_index(3) == (1, 1)

    def test_get_internal_boundary_positions(self):
        pos = self.d.get_internal_boundary_positions(0)
        np.testing.assert_allclose(pos, [0.5])

    def test_subdomains_property(self):
        subs = self.d.subdomains
        assert len(subs) == 4
        for s in subs:
            assert hasattr(s, 'xmin')
            assert hasattr(s, 'xmax')

    def test_to_numpy(self):
        arr = self.d.to_numpy()
        assert arr.dtype == np.float32
        assert arr.shape == (4, 2)

    def test_require_partition_raises_on_plain(self):
        plain = DomainCubic([(0, 1)])
        with pytest.raises(AttributeError):
            plain.get_subdomain_centers()


# ===========================================================================
# sample_interior — basic shapes and bounds
# ===========================================================================

class TestSampleInterior:
    d2 = DomainCubic([(0, 1), (0, 1)])
    dp = DomainCubic([[0, 0.5, 1], [0, 0.5, 1]])
    dt = DomainCubic([[0, 0.5, 1], [0, 0.5, 1]], time=[0, 0.5, 1])

    def test_plain_shape(self):
        pts = self.d2.sample_interior(100, rng=RNG)
        assert pts.shape == (100, 2)

    def test_plain_bounds(self):
        pts = self.d2.sample_interior(500, rng=RNG)
        assert _in_box(pts, [0, 0], [1, 1])

    def test_region_none(self):
        pts = self.dp.sample_interior(100, region=None, rng=RNG)
        assert pts.shape == (100, 2)

    def test_region_all(self):
        pts = self.dp.sample_interior(100, region='all', rng=RNG)
        assert pts.shape == (100, 2)

    def test_region_subdomains(self):
        pts = self.dp.sample_interior(100, region='subdomains', rng=RNG)
        assert pts.shape == (100, 2)

    def test_region_subdomains_requires_partition(self):
        with pytest.raises(ValueError):
            self.d2.sample_interior(10, region='subdomains', rng=RNG)

    def test_region_tuple_spatial(self):
        pts = self.dp.sample_interior(100, region=(0, 0), rng=RNG)
        assert pts.shape == (100, 2)
        assert _in_box(pts, [0, 0], [0.5, 0.5])

    def test_region_tuple_with_time(self):
        pts = self.dt.sample_interior(100, region=(0, 0, 0), rng=RNG)
        assert pts.shape == (100, 3)
        # spatial
        assert _in_box(pts[:, :2], [0, 0], [0.5, 0.5])
        # time: first partition [0, 0.5]
        assert _in_box(pts[:, 2:], [0], [0.5])

    def test_region_tuple_wrong_length(self):
        with pytest.raises(ValueError):
            self.dp.sample_interior(10, region=(0,), rng=RNG)

    def test_region_tuple_out_of_range(self):
        with pytest.raises(IndexError):
            self.dp.sample_interior(10, region=(5, 0), rng=RNG)

    def test_region_named(self):
        d = DomainCubic([(0, 1), (0, 1)])
        d.add_inner([(0.2, 0.8), (0.2, 0.8)], name='centre')
        pts = d.sample_interior(100, region='centre', rng=RNG)
        assert pts.shape == (100, 2)
        assert _in_box(pts, [0.2, 0.2], [0.8, 0.8])

    def test_region_unknown_raises(self):
        with pytest.raises(KeyError):
            self.d2.sample_interior(10, region='nosuchregion', rng=RNG)

    def test_region_list_equal_split(self):
        d = DomainCubic([(0, 2), (0, 2)])
        d.add_inner([(0, 1), (0, 1)], name='a')
        d.add_inner([(1, 2), (1, 2)], name='b')
        pts = d.sample_interior(100, region=['a', 'b'], size='equal', rng=RNG)
        assert pts.shape == (100, 2)

    def test_region_list_explicit_weights(self):
        d = DomainCubic([(0, 2), (0, 2)])
        d.add_inner([(0, 1), (0, 1)], name='a')
        d.add_inner([(1, 2), (1, 2)], name='b')
        pts = d.sample_interior(100, region=['a', 'b'], size=[0.7, 0.3], rng=RNG)
        assert pts.shape == (100, 2)

    def test_with_time_appends_column(self):
        d = DomainCubic([(0, 1)], time=[0, 1])
        pts = d.sample_interior(100, rng=RNG)
        assert pts.shape == (100, 2)
        assert _in_box(pts[:, 1:], [0], [1])

    def test_mode_per_partition(self):
        pts = self.dp.sample_interior(100, mode='per_partition', rng=RNG)
        assert pts.shape == (100, 2)


# ===========================================================================
# sample_boundary — basic shapes and bounds
# ===========================================================================

class TestSampleBoundary:
    d2 = DomainCubic([(0, 1), (0, 1)])
    dp = DomainCubic([[0, 0.5, 1], [0, 0.5, 1]])
    dt = DomainCubic([[0, 0.5, 1], [0, 0.5, 1]], time=[0, 1])

    def test_default_all_faces(self):
        pts = self.d2.sample_boundary(100, rng=RNG)
        assert pts.shape == (100, 2)

    def test_region_all(self):
        pts = self.d2.sample_boundary(100, region='all', rng=RNG)
        assert pts.shape == (100, 2)

    def test_region_none_equals_all(self):
        pts = self.d2.sample_boundary(100, region=None, rng=RNG)
        assert pts.shape == (100, 2)

    def test_all_points_on_boundary(self):
        pts = self.d2.sample_boundary(200, rng=RNG)
        # each point must touch at least one face
        on_face = (
            np.isclose(pts[:, 0], 0) | np.isclose(pts[:, 0], 1) |
            np.isclose(pts[:, 1], 0) | np.isclose(pts[:, 1], 1)
        )
        assert on_face.all()

    def test_region_subdomains(self):
        pts = self.dp.sample_boundary(100, region='subdomains', rng=RNG)
        assert pts.shape == (100, 2)

    def test_region_subdomains_requires_partition(self):
        with pytest.raises(ValueError):
            self.d2.sample_boundary(10, region='subdomains', rng=RNG)

    def test_named_region(self):
        d = DomainCubic([(0, 1), (0, 1)])
        d.add_boundary(['min', (0, 1)], name='x_left')
        pts = d.sample_boundary(100, region='x_left', rng=RNG)
        assert pts.shape == (100, 2)
        np.testing.assert_allclose(pts[:, 0], 0.0)

    def test_named_region_x_max(self):
        d = DomainCubic([(0, 1), (0, 1)])
        d.add_boundary(['max', (0, 1)], name='x_right')
        pts = d.sample_boundary(100, region='x_right', rng=RNG)
        np.testing.assert_allclose(pts[:, 0], 1.0)

    def test_named_region_with_time(self):
        d = DomainCubic([(0, 1), (0, 1)], time=[0, 1])
        d.add_boundary(['min', (0, 1)], name='left')
        pts = d.sample_boundary(100, region='left', rng=RNG)
        assert pts.shape == (100, 3)
        np.testing.assert_allclose(pts[:, 0], 0.0)
        assert _in_box(pts[:, 2:], [0], [1])

    def test_region_list(self):
        d = DomainCubic([(0, 1), (0, 1)])
        d.add_boundary(['min', (0, 1)], name='left')
        d.add_boundary(['max', (0, 1)], name='right')
        pts = d.sample_boundary(100, region=['left', 'right'], rng=RNG)
        assert pts.shape == (100, 2)

    def test_unknown_region_raises(self):
        with pytest.raises(KeyError):
            self.d2.sample_boundary(10, region='nosuchregion', rng=RNG)

    def test_all_with_time_appends_t(self):
        pts = self.dt.sample_boundary(100, rng=RNG)
        assert pts.shape == (100, 3)


# ===========================================================================
# add_inner
# ===========================================================================

class TestAddInner:
    def test_basic(self):
        d = DomainCubic([(0, 1), (0, 1)])
        d.add_inner([(0.2, 0.8), (0.2, 0.8)], name='c')
        lo, hi = d._inner_regions['c']
        np.testing.assert_allclose(lo, [0.2, 0.2])
        np.testing.assert_allclose(hi, [0.8, 0.8])

    def test_auto_extends_time(self):
        d = DomainCubic([(0, 1), (0, 1)], time=[0, 1])
        d.add_inner([(0.1, 0.9), (0.1, 0.9)], name='c')
        lo, hi = d._inner_regions['c']
        assert len(lo) == 3
        assert lo[2] == 0.0
        assert hi[2] == 1.0

    def test_time_partition_index(self):
        d = DomainCubic([(0, 1)], time=[0, 0.5, 1])
        d.add_inner([(0.1, 0.9)], name='early', time=0)
        lo, hi = d._inner_regions['early']
        assert lo[-1] == pytest.approx(0.0)
        assert hi[-1] == pytest.approx(0.5)

    def test_time_explicit_range(self):
        d = DomainCubic([(0, 1)], time=[0, 1])
        d.add_inner([(0.1, 0.9)], name='mid', time=(0.2, 0.8))
        lo, hi = d._inner_regions['mid']
        assert lo[-1] == pytest.approx(0.2)
        assert hi[-1] == pytest.approx(0.8)

    def test_out_of_bounds_raises(self):
        d = DomainCubic([(0, 1)])
        with pytest.raises(ValueError):
            d.add_inner([(0.0, 2.0)], name='bad')

    def test_lo_ge_hi_raises(self):
        d = DomainCubic([(0, 1)])
        with pytest.raises(ValueError):
            d.add_inner([(0.8, 0.2)], name='bad')

    def test_time_index_out_of_range_raises(self):
        d = DomainCubic([(0, 1)], time=[0, 0.5, 1])
        with pytest.raises(IndexError):
            d.add_inner([(0.1, 0.9)], name='bad', time=5)

    def test_time_on_stationary_domain_raises(self):
        d = DomainCubic([(0, 1)])
        with pytest.raises(ValueError):
            d.add_inner([(0.1, 0.9)], name='bad', time=0)


# ===========================================================================
# add_boundary
# ===========================================================================

class TestAddBoundary:
    def test_min_face(self):
        d = DomainCubic([(0, 1), (0, 1)])
        d.add_boundary(['min', (0, 1)], name='left')
        reg = d._boundary_regions['left']
        assert reg['fixed_dim'] == 0
        assert reg['fixed_val'] == pytest.approx(0.0)

    def test_max_face(self):
        d = DomainCubic([(0, 1), (0, 1)])
        d.add_boundary(['max', (0, 1)], name='right')
        reg = d._boundary_regions['right']
        assert reg['fixed_val'] == pytest.approx(1.0)

    def test_restricted_face(self):
        d = DomainCubic([(0, 1), (0, 1), (0, 1)])
        d.add_boundary(['min', (0.2, 0.8), (0, 1)], name='left_mid')
        reg = d._boundary_regions['left_mid']
        assert reg['fixed_dim'] == 0
        assert reg['lo'][1] == pytest.approx(0.2)
        assert reg['hi'][1] == pytest.approx(0.8)

    def test_auto_extends_time(self):
        d = DomainCubic([(0, 1), (0, 1)], time=[0, 1])
        d.add_boundary(['min', (0, 1)], name='left')
        reg = d._boundary_regions['left']
        assert len(reg['lo']) == 3

    def test_time_partition_index(self):
        d = DomainCubic([(0, 1), (0, 1)], time=[0, 0.5, 1])
        d.add_boundary(['min', (0, 1)], name='left_t0', time=0)
        reg = d._boundary_regions['left_t0']
        assert reg['lo'][-1] == pytest.approx(0.0)
        assert reg['hi'][-1] == pytest.approx(0.5)

    def test_time_explicit_range(self):
        d = DomainCubic([(0, 1), (0, 1)], time=[0, 1])
        d.add_boundary(['min', (0, 1)], name='left_late', time=(0.5, 1.0))
        reg = d._boundary_regions['left_late']
        assert reg['lo'][-1] == pytest.approx(0.5)
        assert reg['hi'][-1] == pytest.approx(1.0)

    def test_no_face_selector_raises(self):
        d = DomainCubic([(0, 1), (0, 1)])
        with pytest.raises(ValueError):
            d.add_boundary([(0, 1), (0, 1)], name='bad')

    def test_two_face_selectors_raises(self):
        d = DomainCubic([(0, 1), (0, 1)])
        with pytest.raises(ValueError):
            d.add_boundary(['min', 'max'], name='bad')

    def test_range_out_of_bounds_raises(self):
        d = DomainCubic([(0, 1), (0, 1)])
        with pytest.raises(ValueError):
            d.add_boundary(['min', (0, 5)], name='bad')


# ===========================================================================
# _resolve_time_arg
# ===========================================================================

class TestResolveTimeArg:
    def test_tuple(self):
        d = DomainCubic([(0, 1)], time=[0, 1])
        t_lo, t_hi = d._resolve_time_arg((0.2, 0.8))
        assert t_lo == pytest.approx(0.2)
        assert t_hi == pytest.approx(0.8)

    def test_partition_index(self):
        d = DomainCubic([(0, 1)], time=[0, 0.5, 1])
        t_lo, t_hi = d._resolve_time_arg(1)
        assert t_lo == pytest.approx(0.5)
        assert t_hi == pytest.approx(1.0)

    def test_no_time_raises(self):
        d = DomainCubic([(0, 1)])
        with pytest.raises(ValueError):
            d._resolve_time_arg(0)

    def test_int_without_partition_raises(self):
        d = DomainCubic([(0, 1)], time=[0, 1])
        with pytest.raises(ValueError):
            d._resolve_time_arg(0)

    def test_out_of_bounds_tuple_raises(self):
        d = DomainCubic([(0, 1)], time=[0, 1])
        with pytest.raises(ValueError):
            d._resolve_time_arg((0.0, 5.0))


# ===========================================================================
# Sampling — count correctness
# ===========================================================================

class TestSamplingCounts:
    """Verify that the returned point count always matches n_points exactly."""

    @pytest.mark.parametrize("n", [1, 7, 100, 999])
    def test_interior_count(self, n):
        d = DomainCubic([(0, 1), (0, 1)])
        assert d.sample_interior(n, rng=RNG).shape[0] == n

    @pytest.mark.parametrize("n", [1, 7, 100, 999])
    def test_boundary_count(self, n):
        d = DomainCubic([(0, 1), (0, 1)])
        assert d.sample_boundary(n, rng=RNG).shape[0] == n

    @pytest.mark.parametrize("n", [1, 7, 100, 999])
    def test_subdomains_interior_count(self, n):
        d = DomainCubic([[0, 0.5, 1], [0, 0.5, 1]])
        assert d.sample_interior(n, region='subdomains', rng=RNG).shape[0] == n

    @pytest.mark.parametrize("n", [1, 7, 100, 999])
    def test_list_region_count(self, n):
        d = DomainCubic([(0, 2), (0, 2)])
        d.add_inner([(0, 1), (0, 1)], name='a')
        d.add_inner([(1, 2), (1, 2)], name='b')
        assert d.sample_interior(n, region=['a', 'b'], rng=RNG).shape[0] == n

    @pytest.mark.parametrize("n", [1, 7, 100, 999])
    def test_list_boundary_count(self, n):
        d = DomainCubic([(0, 1), (0, 1)])
        d.add_boundary(['min', (0, 1)], name='left')
        d.add_boundary(['max', (0, 1)], name='right')
        assert d.sample_boundary(n, region=['left', 'right'], rng=RNG).shape[0] == n


# ===========================================================================
# Sampling — spatial bounds correctness
# ===========================================================================

class TestSamplingBounds:
    def test_interior_in_domain(self):
        d = DomainCubic([(0, 1), (0, 2), (1, 3)])
        pts = d.sample_interior(500, rng=RNG)
        assert _in_box(pts, d.xmin, d.xmax)

    def test_boundary_in_domain(self):
        d = DomainCubic([(0, 1), (0, 1)])
        pts = d.sample_boundary(500, rng=RNG)
        assert _in_box(pts, d.xmin, d.xmax)

    def test_named_inner_in_region_box(self):
        d = DomainCubic([(0, 1), (0, 1)])
        d.add_inner([(0.3, 0.7), (0.3, 0.7)], name='mid')
        pts = d.sample_interior(500, region='mid', rng=RNG)
        assert _in_box(pts, [0.3, 0.3], [0.7, 0.7])

    def test_named_boundary_on_correct_face(self):
        d = DomainCubic([(0, 1), (0, 1)])
        d.add_boundary([(0, 1), 'min'], name='y_bottom')
        pts = d.sample_boundary(200, region='y_bottom', rng=RNG)
        np.testing.assert_allclose(pts[:, 1], 0.0)

    def test_tuple_subdomain_inside_cell(self):
        d = DomainCubic([[0, 0.5, 1], [0, 0.5, 1]])
        pts = d.sample_interior(200, region=(1, 1), rng=RNG)
        assert _in_box(pts, [0.5, 0.5], [1.0, 1.0])


# ===========================================================================
# __repr__
# ===========================================================================

class TestRepr:
    def test_plain(self):
        r = repr(DomainCubic([(0, 1), (0, 1)]))
        assert 'DomainCubic' in r

    def test_partitioned(self):
        r = repr(DomainCubic([[0, 0.5, 1], [0, 0.5, 1]]))
        assert 'n_subdomains_per_dim' in r

    def test_with_time(self):
        r = repr(DomainCubic([(0, 1)], time=[0, 1]))
        assert 'time' in r
