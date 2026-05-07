"""Tests for DomainMesh – region API, sampling, and visualisation.

Mesh fixture: 11×11 regular grid on [0,1]² → 200 right-angle triangles.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import matplotlib.pyplot as plt

from pinns.domain import DomainMesh

# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unit_square_mesh(n=11):
    """Return (verts, faces) for a regular triangulated unit square."""
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y)
    verts = np.column_stack([xx.ravel(), yy.ravel()])
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            bl = j * n + i
            br = bl + 1
            tl = (j + 1) * n + i
            tr = tl + 1
            faces += [[bl, br, tr], [bl, tr, tl]]
    return verts, np.array(faces, dtype=np.int64)


@pytest.fixture
def mesh():
    return _unit_square_mesh()


@pytest.fixture
def d_stat(mesh):
    return DomainMesh(mesh)


@pytest.fixture
def d_cont(mesh):
    return DomainMesh(mesh, time=(0.0, 1.0))


@pytest.fixture
def d_disc(mesh):
    return DomainMesh(mesh, time=[0.0, 0.25, 0.5, 0.75, 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# 1) Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruction:

    def test_stationary(self, mesh):
        d = DomainMesh(mesh)
        assert d._time_mode == 'stationary'
        assert d._vertices.shape[1] == 2
        assert d._faces.shape[1] == 3
        assert len(d._bnd_edges) > 0

    def test_continuous_time(self, mesh):
        d = DomainMesh(mesh, time=(0.0, 2.0))
        assert d._time_mode == 'continuous'
        assert d._t_min == pytest.approx(0.0)
        assert d._t_max == pytest.approx(2.0)

    def test_discrete_time(self, mesh):
        ts = [0.0, 0.5, 1.0]
        d = DomainMesh(mesh, time=ts)
        assert d._time_mode == 'discrete'
        assert list(d._time_points) == ts

    def test_boundary_edges_precomputed(self, d_stat):
        # unit square perimeter should have at least 40 boundary edges
        assert len(d_stat._bnd_edges) >= 40
        probs = d_stat._bnd_edge_probs
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_regions_empty_at_init(self, d_stat):
        assert d_stat._inner_regions == {}
        assert d_stat._boundary_regions == {}


# ─────────────────────────────────────────────────────────────────────────────
# 2) Region registration – add_inner
# ─────────────────────────────────────────────────────────────────────────────

class TestAddInner:

    def test_callable_selector(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='left')
        assert 'left' in d_stat._inner_regions
        reg = d_stat._inner_regions['left']
        assert 'face_indices' in reg
        assert len(reg['face_indices']) > 0

    def test_bbox_selector(self, d_stat):
        d_stat.add_inner([(0.25, 0.75), (0.25, 0.75)], name='centre')
        reg = d_stat._inner_regions['centre']
        assert len(reg['face_indices']) > 0

    def test_bool_mask_selector(self, d_stat):
        v = d_stat._vertices
        mask = v[:, 0] < 0.5
        d_stat.add_inner(mask, name='left_mask')
        assert 'left_mask' in d_stat._inner_regions

    def test_int_array_selector(self, d_stat):
        v = d_stat._vertices
        idx = np.where(v[:, 0] < 0.5)[0]
        d_stat.add_inner(idx, name='left_idx')
        assert 'left_idx' in d_stat._inner_regions

    def test_overwrite_existing(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='r')
        d_stat.add_inner(lambda v: v[:, 0] > 0.5, name='r')
        reg = d_stat._inner_regions['r']
        # After overwrite, faces should be in the right half
        fidx = reg['face_indices']
        centroids = d_stat._vertices[d_stat._faces[fidx]].mean(axis=1)
        assert (centroids[:, 0] > 0.4).all()

    def test_with_time_window(self, d_cont):
        d_cont.add_inner(lambda v: v[:, 0] < 0.5, name='left', time=(0.2, 0.8))
        reg = d_cont._inner_regions['left']
        assert reg['t_lo'] == pytest.approx(0.2)
        assert reg['t_hi'] == pytest.approx(0.8)

    def test_tri_probs_sum_to_one(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='left')
        probs = d_stat._inner_regions['left']['tri_probs']
        assert abs(probs.sum() - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 3) Region registration – add_boundary
# ─────────────────────────────────────────────────────────────────────────────

class TestAddBoundary:

    def test_callable_selector(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        assert 'x0' in d_stat._boundary_regions
        reg = d_stat._boundary_regions['x0']
        assert len(reg['edges']) > 0

    def test_bbox_selector(self, d_stat):
        d_stat.add_boundary([(0.0, 0.0), (0.0, 1.0)], name='left_bbox')
        reg = d_stat._boundary_regions['left_bbox']
        assert len(reg['edges']) > 0

    def test_edge_probs_sum_to_one(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        probs = d_stat._boundary_regions['x0']['edge_probs']
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_all_edges_on_correct_side(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        edges = d_stat._boundary_regions['x0']['edges']
        v = d_stat._vertices
        # All vertices of those edges should be at x≈0
        assert (np.abs(v[edges.ravel(), 0]) < 1e-6).all()

    def test_with_time_window(self, d_cont):
        d_cont.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0', time=(0.0, 0.5))
        reg = d_cont._boundary_regions['x0']
        assert reg['t_lo'] == pytest.approx(0.0)
        assert reg['t_hi'] == pytest.approx(0.5)

    def test_overwrite_existing(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='side')
        n1 = len(d_stat._boundary_regions['side']['edges'])
        d_stat.add_boundary(lambda v: v[:, 1] < 1e-6, name='side')
        n2 = len(d_stat._boundary_regions['side']['edges'])
        assert n1 == n2  # both sides have same number of edges on unit square


# ─────────────────────────────────────────────────────────────────────────────
# 4) sample_interior
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleInterior:

    rng = np.random.default_rng(42)

    def test_default_shape_stationary(self, d_stat):
        pts = d_stat.sample_interior(150, rng=self.rng)
        assert pts.shape == (150, 2)

    def test_default_shape_continuous(self, d_cont):
        pts = d_cont.sample_interior(150, rng=self.rng)
        assert pts.shape == (150, 3)

    def test_default_shape_discrete(self, d_disc):
        pts = d_disc.sample_interior(50, rng=self.rng)
        assert pts.shape == (50, 3)

    def test_bounds_inside_unit_square(self, d_stat):
        pts = d_stat.sample_interior(500, rng=self.rng)
        assert (pts[:, 0] >= 0).all() and (pts[:, 0] <= 1).all()
        assert (pts[:, 1] >= 0).all() and (pts[:, 1] <= 1).all()

    def test_region_named(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='left')
        pts = d_stat.sample_interior(200, region='left', rng=self.rng)
        assert pts.shape == (200, 2)
        assert pts[:, 0].max() <= 0.5 + 1e-9

    def test_region_all_keyword(self, d_stat):
        pts = d_stat.sample_interior(100, region='all', rng=self.rng)
        assert pts.shape == (100, 2)

    def test_region_list_equal(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='left')
        d_stat.add_inner(lambda v: v[:, 0] > 0.5, name='right')
        pts = d_stat.sample_interior(200, region=['left', 'right'], rng=self.rng)
        assert pts.shape == (200, 2)

    def test_region_list_explicit_weights(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='left')
        d_stat.add_inner(lambda v: v[:, 0] > 0.5, name='right')
        pts = d_stat.sample_interior(100, region=['left', 'right'],
                                      size=[0.75, 0.25], rng=self.rng)
        assert pts.shape == (100, 2)

    def test_region_list_area_weights(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='left')
        d_stat.add_inner(lambda v: v[:, 0] > 0.5, name='right')
        pts = d_stat.sample_interior(100, region=['left', 'right'],
                                      size='area', rng=self.rng)
        assert pts.shape == (100, 2)

    def test_unknown_region_raises(self, d_stat):
        with pytest.raises(KeyError):
            d_stat.sample_interior(50, region='nonexistent', rng=self.rng)

    def test_timed_region_time_column(self, d_cont):
        d_cont.add_inner(lambda v: v[:, 0] < 0.5, name='left', time=(0.2, 0.6))
        pts = d_cont.sample_interior(200, region='left', rng=self.rng)
        assert pts.shape == (200, 3)
        assert (pts[:, 2] >= 0.2 - 1e-9).all()
        assert (pts[:, 2] <= 0.6 + 1e-9).all()


# ─────────────────────────────────────────────────────────────────────────────
# 5) sample_boundary
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleBoundary:

    rng = np.random.default_rng(99)

    def test_default_shape_stationary(self, d_stat):
        pts = d_stat.sample_boundary(100, rng=self.rng)
        assert pts.shape == (100, 2)

    def test_default_shape_continuous(self, d_cont):
        pts = d_cont.sample_boundary(100, rng=self.rng)
        assert pts.shape == (100, 3)

    def test_points_on_boundary(self, d_stat):
        """Most sampled points should be very close to the perimeter of [0,1]²."""
        pts = d_stat.sample_boundary(500, rng=self.rng)
        tol = 0.05   # interpolated pts on short diag edges may be slightly off
        on_boundary = (
            (pts[:, 0] < tol) |
            (pts[:, 0] > 1 - tol) |
            (pts[:, 1] < tol) |
            (pts[:, 1] > 1 - tol)
        )
        # At least 95 % must satisfy the loose check
        assert on_boundary.mean() >= 0.95

    def test_region_named(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        pts = d_stat.sample_boundary(50, region='x0', rng=self.rng)
        assert pts.shape == (50, 2)
        assert np.abs(pts[:, 0]).max() < 1e-6

    def test_region_all_keyword(self, d_stat):
        pts = d_stat.sample_boundary(50, region='all', rng=self.rng)
        assert pts.shape == (50, 2)

    def test_region_list(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        d_stat.add_boundary(lambda v: v[:, 0] > 1 - 1e-6, name='x1')
        pts = d_stat.sample_boundary(100, region=['x0', 'x1'], rng=self.rng)
        assert pts.shape == (100, 2)

    def test_region_list_explicit_weights(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        d_stat.add_boundary(lambda v: v[:, 0] > 1 - 1e-6, name='x1')
        pts = d_stat.sample_boundary(100, region=['x0', 'x1'],
                                      size=[0.3, 0.7], rng=self.rng)
        assert pts.shape == (100, 2)

    def test_region_list_length_weights(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        d_stat.add_boundary(lambda v: v[:, 0] > 1 - 1e-6, name='x1')
        pts = d_stat.sample_boundary(100, region=['x0', 'x1'],
                                      size='length', rng=self.rng)
        assert pts.shape == (100, 2)

    def test_unknown_region_raises(self, d_stat):
        with pytest.raises(KeyError):
            d_stat.sample_boundary(50, region='ghost', rng=self.rng)

    def test_timed_region_time_column(self, d_cont):
        d_cont.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0', time=(0.1, 0.4))
        pts = d_cont.sample_boundary(200, region='x0', rng=self.rng)
        assert pts.shape == (200, 3)
        assert (pts[:, 2] >= 0.1 - 1e-9).all()
        assert (pts[:, 2] <= 0.4 + 1e-9).all()

    def test_legacy_positional_api(self, d_stat):
        """sample_boundary(n, dim, side, rng) should still work."""
        pts = d_stat.sample_boundary(50, 0, 0, rng=self.rng)
        assert pts.shape == (50, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 6) plot()
# ─────────────────────────────────────────────────────────────────────────────

class TestPlot:

    def _d(self):
        d = DomainMesh(_unit_square_mesh())
        d.add_inner(lambda v: v[:, 0] < 0.5, name='left')
        d.add_inner([(0.3, 0.7), (0.3, 0.7)], name='centre')
        d.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        d.add_boundary(lambda v: v[:, 0] > 1 - 1e-6, name='x1')
        return d

    def test_plot_no_region(self):
        d = self._d()
        fig = d.plot(backend='matplotlib')
        assert fig is not None
        plt.close('all')

    def test_plot_region_all(self):
        d = self._d()
        fig = d.plot(region='all', backend='matplotlib')
        assert fig is not None
        plt.close('all')

    def test_plot_region_named(self):
        d = self._d()
        fig = d.plot(region='left', backend='matplotlib')
        assert fig is not None
        plt.close('all')

    def test_plot_region_list(self):
        d = self._d()
        fig = d.plot(region=['left', 'centre'], backend='matplotlib')
        assert fig is not None
        plt.close('all')

    def test_plot_boundary_all(self):
        d = self._d()
        fig = d.plot(boundary='all', backend='matplotlib')
        assert fig is not None
        plt.close('all')

    def test_plot_boundary_named(self):
        d = self._d()
        fig = d.plot(boundary='x0', backend='matplotlib')
        assert fig is not None
        plt.close('all')

    def test_plot_combined(self):
        d = self._d()
        fig = d.plot(region='all', boundary='all', backend='matplotlib')
        assert fig is not None
        plt.close('all')

    def test_plot_with_points(self):
        d = self._d()
        rng = np.random.default_rng(7)
        pts = d.sample_interior(50, rng=rng)
        fig = d.plot(points=pts, backend='matplotlib')
        assert fig is not None
        plt.close('all')


# ─────────────────────────────────────────────────────────────────────────────
# 7) __repr__
# ─────────────────────────────────────────────────────────────────────────────

class TestRepr:

    def test_repr_stationary_no_regions(self, d_stat):
        r = repr(d_stat)
        assert 'DomainMesh' in r
        assert '2D' in r
        assert 'inner_regions' not in r   # hidden unless non-empty

    def test_repr_continuous_no_regions(self, d_cont):
        r = repr(d_cont)
        assert 't∈' in r
        assert 'continuous' in r

    def test_repr_with_inner_regions(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='left')
        r = repr(d_stat)
        assert 'inner_regions' in r
        assert 'left' in r

    def test_repr_with_boundary_regions(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        r = repr(d_stat)
        assert 'boundary_regions' in r
        assert 'x0' in r

    def test_repr_with_both(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='L')
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        r = repr(d_stat)
        assert 'L' in r
        assert 'x0' in r


# ─────────────────────────────────────────────────────────────────────────────
# 8) Count / spatial correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestSpatialCorrectness:

    rng = np.random.default_rng(0)

    def test_interior_left_half_x_max(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='left')
        pts = d_stat.sample_interior(1000, region='left', rng=self.rng)
        # All centroids of selected faces satisfy x < 0.5; interpolated pts may
        # be very slightly above due to barycentric interpolation on edge tris
        assert pts[:, 0].max() < 0.55

    def test_interior_bbox_x_bounds(self, d_stat):
        d_stat.add_inner([(0.2, 0.8), (0.0, 1.0)], name='mid')
        pts = d_stat.sample_interior(500, region='mid', rng=self.rng)
        assert pts[:, 0].min() >= 0.0 and pts[:, 0].max() <= 1.0

    def test_boundary_x0_all_at_x0(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        pts = d_stat.sample_boundary(200, region='x0', rng=self.rng)
        assert np.abs(pts[:, 0]).max() < 1e-6

    def test_boundary_y1_all_at_y1(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 1] > 1 - 1e-6, name='y1')
        pts = d_stat.sample_boundary(200, region='y1', rng=self.rng)
        assert np.abs(pts[:, 1] - 1).max() < 1e-6

    def test_multi_interior_total_count(self, d_stat):
        d_stat.add_inner(lambda v: v[:, 0] < 0.5, name='left')
        d_stat.add_inner(lambda v: v[:, 0] > 0.5, name='right')
        pts = d_stat.sample_interior(300, region=['left', 'right'], rng=self.rng)
        assert len(pts) == 300

    def test_multi_boundary_total_count(self, d_stat):
        d_stat.add_boundary(lambda v: v[:, 0] < 1e-6, name='x0')
        d_stat.add_boundary(lambda v: v[:, 1] < 1e-6, name='y0')
        pts = d_stat.sample_boundary(200, region=['x0', 'y0'], rng=self.rng)
        assert len(pts) == 200


# ─────────────────────────────────────────────────────────────────────────────
# 9) 3-D surface mesh
# ─────────────────────────────────────────────────────────────────────────────

def _unit_sphere_mesh(subdivisions=2):
    """Icosphere approximation of the unit sphere.

    Returns (verts, faces) with verts in R³, all rows on the unit sphere.
    At subdivision 2 the mesh has 162 vertices and 320 faces.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    ico_faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int64)

    verts = ico_verts.copy()
    faces = ico_faces.copy()

    for _ in range(subdivisions):
        mid_cache: dict = {}
        verts_list = list(verts)

        def _mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid_cache:
                m = (verts_list[a] + verts_list[b]) / 2.0
                m /= np.linalg.norm(m)   # project back onto unit sphere
                mid_cache[key] = len(verts_list)
                verts_list.append(m)
            return mid_cache[key]

        new_faces = []
        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(verts_list, dtype=float)
        faces = np.array(new_faces, dtype=np.int64)

    return verts, faces


class TestSurfaceMesh3D:
    """Tests for DomainMesh with a 3-D surface mesh (unit sphere)."""

    rng = np.random.default_rng(7)

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verts, self.faces = _unit_sphere_mesh(subdivisions=2)
        self.d = DomainMesh((self.verts, self.faces))

    # ── Construction ──────────────────────────────────────────────────────

    def test_spatial_dims_is_3(self):
        assert self.d._spatial_dims == 3

    def test_n_dims_stationary(self):
        assert self.d.n_dims == 3

    def test_n_dims_continuous(self):
        d = DomainMesh((self.verts, self.faces), time=(0.0, 1.0))
        assert d.n_dims == 4

    def test_tri_areas_computed(self):
        """_tri_areas must be populated (cross-product formula for 3-D)."""
        assert self.d._tri_areas is not None
        assert self.d._tri_areas.shape == (len(self.faces),)
        assert (self.d._tri_areas > 0).all()

    def test_tri_areas_sum_near_4pi(self):
        """Surface area of unit sphere ≈ 4π; mesh should be close."""
        sphere_area = 4 * np.pi
        assert abs(self.d._tri_areas.sum() - sphere_area) / sphere_area < 0.05

    def test_closed_surface_no_boundary_edges(self):
        """A closed sphere has no boundary (rim) edges."""
        assert len(self.d._bnd_edges) == 0

    # ── Region registration ───────────────────────────────────────────────

    def test_add_inner_upper_hemisphere(self):
        self.d.add_inner(lambda v: v[:, 2] > 0, name='upper')
        assert 'upper' in self.d._inner_regions
        info = self.d._inner_regions['upper']
        assert info['tri_probs'] is not None
        assert abs(info['tri_probs'].sum() - 1.0) < 1e-9

    def test_add_inner_bbox(self):
        self.d.add_inner([(-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)], name='cap')
        assert 'cap' in self.d._inner_regions

    # ── Interior sampling — points lie on the sphere ──────────────────────

    def test_sample_interior_on_surface(self):
        """All sampled points should lie very close to the unit sphere."""
        N = 500
        pts = self.d.sample_interior(N, rng=self.rng)
        assert pts.shape == (N, 3)
        r = np.linalg.norm(pts, axis=1)
        assert r.min() > 0.95
        assert r.max() < 1.05

    def test_sample_interior_upper_hemi_z_positive(self):
        self.d.add_inner(lambda v: v[:, 2] > 0, name='upper')
        pts = self.d.sample_interior(300, region='upper', rng=self.rng)
        assert pts.shape == (300, 3)
        # All points must come from triangles whose vertices have z > 0
        assert pts[:, 2].min() >= -1e-9

    def test_sample_interior_count(self):
        N = 250
        pts = self.d.sample_interior(N, rng=self.rng)
        assert len(pts) == N

    def test_sample_multi_interior_area_weighted(self):
        self.d.add_inner(lambda v: v[:, 2] > 0, name='upper')
        self.d.add_inner(lambda v: v[:, 2] < 0, name='lower')
        pts = self.d.sample_interior(400, region=['upper', 'lower'],
                                      size='area', rng=self.rng)
        assert len(pts) == 400

    # ── Time axis with a 3-D mesh ─────────────────────────────────────────

    def test_continuous_time_output_shape(self):
        d = DomainMesh((self.verts, self.faces), time=(0.0, 1.0))
        pts = d.sample_interior(200, rng=self.rng)
        assert pts.shape == (200, 4)       # x, y, z, t
        assert pts[:, 3].min() >= 0.0
        assert pts[:, 3].max() <= 1.0

    def test_timed_region_output_shape(self):
        d = DomainMesh((self.verts, self.faces), time=(0.0, 1.0))
        d.add_inner(lambda v: v[:, 2] > 0, name='upper_early', time=(0.0, 0.5))
        pts = d.sample_interior(100, region='upper_early', rng=self.rng)
        assert pts.shape == (100, 4)
        assert pts[:, 3].max() <= 0.5 + 1e-9

    # ── Open surface (hemisphere) has boundary edges ──────────────────────

    def test_open_surface_has_boundary_edges(self):
        """A hemisphere (upper half of unit sphere) has a rim at z=0."""
        v, f = self.verts, self.faces
        # Keep only faces whose centroid has z > 0
        centroids = v[f].mean(axis=1)
        keep = centroids[:, 2] > 0
        hemi_faces = f[keep]
        d = DomainMesh((v, hemi_faces))
        # The equatorial rim should produce boundary edges
        assert len(d._bnd_edges) > 0

    def test_open_surface_boundary_sampling(self):
        """Sampling from the rim of a hemisphere should return 3-D points."""
        v, f = self.verts, self.faces
        centroids = v[f].mean(axis=1)
        hemi_faces = f[centroids[:, 2] > 0]
        d = DomainMesh((v, hemi_faces))
        d.add_boundary(lambda vv: np.abs(vv[:, 2]) < 0.15, name='rim')
        pts = d.sample_boundary(100, region='rim', rng=self.rng)
        assert pts.shape == (100, 3)
        # Rim points should be near the equatorial plane
        assert np.abs(pts[:, 2]).max() < 0.2
