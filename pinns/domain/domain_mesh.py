import warnings
import numpy as np
from itertools import product
from typing import TYPE_CHECKING, Callable, Optional, Union, Literal, Tuple, List, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..problems.terms import TermDirichletBC, TermNeumannBC, TermRobinBC

class DomainMesh:
    """
    A spatial domain defined by a triangular mesh, with an
    optional time axis controlled by the ``time`` argument.

    The mesh provides vertex positions and face connectivity.  Interior
    sampling uses exact barycentric sampling, which works correctly for both
    **2-D flat meshes** (vertices in R²) and **3-D surface meshes** (vertices
    in R³, triangulated surface embedded in 3-D space).

    **Time modes** — selected automatically from the ``time`` argument:

    * ``time=None`` *(default)* — **stationary** domain.  ``n_dims`` equals
      ``spatial_dims``.
    * ``time=(t_min, t_max)`` — **continuous** time interval.  ``n_dims``
      equals ``spatial_dims + 1``.  ``ProblemWeak`` samples ``n_time_points``
      random levels per epoch.  Time-windows in BCs must be 2-element
      intervals.
    * ``time=array_or_list`` (≥ 2 values) — **discrete** time steps.
      ``domain.dt`` and ``domain.n_steps`` are set automatically.
      ``ProblemWeak`` uses BPTT rollout.  BC ``time_window`` defaults to all
      time steps; pass ``[t0]`` for an initial condition.

    Args:
        mesh: A mesh object with ``.vertices`` and ``.faces`` attributes
              (``trimesh.Trimesh``, ``pymesh.Mesh``, ``meshio`` mesh, or a
              ``(vertices, faces)`` tuple).
        time: Time specification — ``None``, a 2-tuple ``(t_min, t_max)``, or
              a 1-D array/list of time-point values.
        t_sampling_method: How to place time quadrature points (continuous
              mode only).  One of ``"uniform"`` *(default)*, ``"midpoint"``,
              ``"latin_hypercube"``, ``"sobol"``, ``"halton"``, or a callable
              ``(n, rng) -> ndarray`` in ``[0, 1]``.
        n_time_points: Default number of time levels per epoch (continuous
              mode only, default 10).

    Examples::

        # Stationary
        domain = DomainMesh(mesh)

        # Continuous time — BCs are TermMeshNodeBC objects appended to
        # domain.boundary_conditions directly or via the Problem class
        domain = DomainMesh(mesh, time=(0.0, 1.0))

        # Discrete time steps
        domain = DomainMesh(mesh, time=np.linspace(0, 1, 21))
    """

    @staticmethod
    def _extract_vertices_faces(mesh):
        """
        Extract vertices and triangular faces from heterogeneous mesh objects.

        Supported formats
        -----------------
        * pymesh / trimesh style:  ``mesh.vertices``, ``mesh.faces``
        * meshio style (pygmsh):   ``mesh.points``,   ``mesh.cells_dict["triangle"]``
        * plain tuple/dict:        ``(vertices, faces)``
        """
        # --- tuple / list shortcut ----------------------------------------
        if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
            return np.asarray(mesh[0], dtype=np.float64), np.asarray(mesh[1], dtype=np.int64)

        # --- meshio (pygmsh output) ----------------------------------------
        if hasattr(mesh, "points") and hasattr(mesh, "cells_dict"):
            verts = np.asarray(mesh.points, dtype=np.float64)
            # Drop the z-column when it is all zeros (2-D mesh embedded in R³)
            if verts.shape[1] == 3 and np.allclose(verts[:, 2], 0.0):
                verts = verts[:, :2]
            faces_raw = None
            for key in ("triangle", "triangle6"):
                if key in mesh.cells_dict:
                    faces_raw = mesh.cells_dict[key]
                    break
            if faces_raw is None:
                raise ValueError(
                    "meshio mesh has no 'triangle' cell block. "
                    "Make sure you requested a triangular surface mesh."
                )
            return verts, np.asarray(faces_raw, dtype=np.int64)

        # --- pymesh / trimesh style ----------------------------------------
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            return (np.asarray(mesh.vertices, dtype=np.float64),
                    np.asarray(mesh.faces,    dtype=np.int64))

        raise TypeError(
            f"Unrecognised mesh type {type(mesh)}.  "
            "Provide a pymesh, trimesh, meshio, or (vertices, faces) object."
        )

    def __init__(self, mesh, time=None, t_sampling_method="uniform",
                 n_time_points=10):
        vertices, faces = self._extract_vertices_faces(mesh)
        self._vertices = vertices
        self._spatial_dims = vertices.shape[1]
        self._faces = faces

        # trimesh needed.  For 3D meshes we still try trimesh as a fallback
        # for the _trimesh attribute (used by pyvista plotting).
        self._trimesh = None
        if self._spatial_dims == 3:
            try:
                import trimesh as _trimesh_mod
                if isinstance(mesh, _trimesh_mod.Trimesh):
                    self._trimesh = mesh
                else:
                    self._trimesh = _trimesh_mod.Trimesh(
                        vertices=vertices, faces=self._faces, process=False
                    )
            except ImportError:
                pass

        # Precompute triangle areas for weighted sampling.
        # Works for both 2-D flat meshes (signed-area formula) and 3-D surface
        # meshes embedded in R³ (cross-product magnitude / 2).
        A = vertices[self._faces[:, 0]]
        B = vertices[self._faces[:, 1]]
        C = vertices[self._faces[:, 2]]
        if self._spatial_dims == 2:
            cross = (B - A)[:, 0] * (C - A)[:, 1] - (C - A)[:, 0] * (B - A)[:, 1]
            self._tri_areas = 0.5 * np.abs(cross)          # (n_faces,)
        else:
            # 3-D surface: area = ||(B-A) × (C-A)|| / 2
            cross3 = np.cross(B - A, C - A)                # (n_faces, 3)
            self._tri_areas = 0.5 * np.linalg.norm(cross3, axis=1)  # (n_faces,)
        self._tri_probs = self._tri_areas / self._tri_areas.sum()

        sp_min = vertices.min(axis=0)
        sp_max = vertices.max(axis=0)

        self._t_sampling_method = t_sampling_method
        self.n_time_points = n_time_points

        # ── Interpret the `time` argument ────────────────────────────────
        # None                  → stationary
        # tuple/list of length 2 with numeric scalars → continuous interval
        # array/list with ≥ 2 values (or len>2)       → discrete time steps
        self._time_mode   = None   # 'stationary' | 'continuous' | 'discrete'
        self._time_points = None
        self.dt           = None
        self.n_steps      = None

        if time is None:
            self._time_mode = 'stationary'
            self._t_min = None
            self._t_max = None
            self.t_interval = None
            self.xmin = sp_min
            self.xmax = sp_max
        else:
            time_arr = np.asarray(time, dtype=float).ravel()
            if len(time_arr) < 2:
                raise ValueError(
                    "DomainMesh: 'time' must be None, a 2-tuple (t_min, t_max), "
                    "or an array/list of ≥ 2 time-point values."
                )
            # Distinguish: exactly 2 values that are meant as (t_min, t_max)
            # vs. a discrete array.  A plain 2-tuple always means continuous;
            # a list/array with exactly 2 elements could be either — we treat
            # a Python tuple of length 2 as continuous, everything else as
            # decides by whether the caller passed a sequence longer than 2.
            _is_tuple2 = isinstance(time, tuple) and len(time) == 2
            if _is_tuple2 or len(time_arr) == 2:
                # Check if caller actually passed a plain (t_min, t_max)
                # A tuple → continuous; list/array of length 2 → also continuous
                # (to get discrete with 2 steps, pass np.array([0, 0.5, 1]))
                if _is_tuple2 or (not isinstance(time, np.ndarray) and len(list(time)) == 2):
                    self._time_mode = 'continuous'
                    self._t_min = float(time_arr[0])
                    self._t_max = float(time_arr[1])
                else:
                    self._time_mode = 'discrete'
            else:
                self._time_mode = 'discrete'

            if self._time_mode == 'discrete':
                tp = time_arr
                self._time_points = list(tp)
                dts = tp[1:] - tp[:-1]
                if np.allclose(dts, dts[0]):
                    self.dt = float(dts[0])
                else:
                    self.dt = dts
                self.n_steps = len(tp) - 1
                self._t_min  = float(tp[0])
                self._t_max  = float(tp[-1])
                self._t_sampling_method = 'midpoint'

            self.t_interval = [self._t_min, self._t_max]
            self.xmin = np.append(sp_min, self._t_min)
            self.xmax = np.append(sp_max, self._t_max)

        self.n_dims = len(self.xmin)
        self.boundary_conditions: List = []

        # Precompute all unique mesh edges: (n_edges, 2) vertex index pairs.
        # Used by _resolve_select to let the user address BCs by edge index.
        _seen_edges: dict = {}
        _edges_list: list = []
        for _face in self._faces:
            for _j in range(3):
                _v0, _v1 = int(_face[_j]), int(_face[(_j + 1) % 3])
                _key = (min(_v0, _v1), max(_v0, _v1))
                if _key not in _seen_edges:
                    _seen_edges[_key] = len(_edges_list)
                    _edges_list.append([_v0, _v1])
        self._all_edges = (np.array(_edges_list, dtype=np.int64)
                           if _edges_list else np.empty((0, 2), dtype=np.int64))
        # canonical (min_v, max_v) -> edge_index  (used by helper methods)
        self._edge_lookup: dict = _seen_edges

        # Precompute boundary / interior node masks from edge counts.
        # A mesh edge shared by exactly one face is a boundary edge.
        _edge_face_count: dict = {}
        for _face in self._faces:
            for _j in range(3):
                _v0, _v1 = int(_face[_j]), int(_face[(_j + 1) % 3])
                _key = (min(_v0, _v1), max(_v0, _v1))
                _edge_face_count[_key] = _edge_face_count.get(_key, 0) + 1
        _bnd_mask = np.zeros(len(self._vertices), dtype=bool)
        for (_v0, _v1), _cnt in _edge_face_count.items():
            if _cnt == 1:
                _bnd_mask[_v0] = True
                _bnd_mask[_v1] = True
        self._boundary_node_mask: np.ndarray = _bnd_mask

        # Precompute boundary edge array and per-edge sampling weights.
        # A mesh edge is a boundary edge when both endpoints are boundary nodes.
        # This works for both 2-D flat meshes and 3-D surface meshes — in the
        # 3-D case "boundary" means edges on the open rim of the surface.
        if len(self._all_edges) > 0:
            _be_mask = np.array(
                [bool(_bnd_mask[int(e[0])]) and bool(_bnd_mask[int(e[1])])
                 for e in self._all_edges], dtype=bool)
            self._bnd_edges = self._all_edges[_be_mask]
        else:
            self._bnd_edges = np.empty((0, 2), dtype=np.int64)
        if len(self._bnd_edges) > 0:
            _bv0 = vertices[self._bnd_edges[:, 0]]
            _bv1 = vertices[self._bnd_edges[:, 1]]
            self._bnd_edge_lengths = np.linalg.norm(_bv1 - _bv0, axis=1)
            self._bnd_edge_probs   = self._bnd_edge_lengths / self._bnd_edge_lengths.sum()
        else:
            self._bnd_edge_lengths = None
            self._bnd_edge_probs   = None

        # Named sampling regions (fully independent from the BC system).
        self._inner_regions:    dict = {}
        self._boundary_regions: dict = {}
        self._periodic_regions: dict = {}

        # Partition (set via set_partition()).
        self._partition_labels:          np.ndarray | None = None
        self._partition_groups:          dict | None       = None
        self._partition_interface_edges: np.ndarray | None = None
        self._partition_interface_lengths: np.ndarray | None = None
        self._time_grid_positions:       np.ndarray | None = None
        self._n_time_partitions:         int                = 1

    # ------------------------------------------------------------------ #
    #  Partition                                                          #
    # ------------------------------------------------------------------ #

    def set_partition(self, space=None, time=None) -> None:
        """Define a spatial (and/or temporal) partition on the mesh.

        After calling this method the following *region* strings become
        available in :meth:`sample_interior` and :meth:`sample_boundary`:

        * ``'partition'`` — sample interior collocation points one per
          (spatial group × time interval) cell, weighted by area × Δt.
        * ``'partition_outer'`` — sample mesh boundary edges.
        * ``'partition_inner'`` — sample interface edges *between* spatial
          partition groups.
        * ``'partition'`` — outer + inner boundary combined (in :meth:`sample_boundary`).

        Args:
            space: How to assign each mesh triangle to a partition group.

                * **Callable** ``(centroids: ndarray) → int array`` — called
                  with triangle centroids of shape ``(n_faces, spatial_dims)``
                  and must return a 1-D integer array of length ``n_faces``
                  with a group label (any integer) per triangle.
                  Example: ``lambda c: (c[:, 0] > 0.5).astype(int)``
                * **int** — automatically split the mesh into this many
                  roughly equal-area groups using K-means clustering on
                  triangle centroids (requires *scikit-learn*).

            time: Time axis breakpoints (only meaningful when the domain has
                a time axis):

                * **int** — number of equal-width intervals; breakpoints are
                  generated with ``np.linspace(t_min, t_max, n+1)``.
                * **array-like** — explicit breakpoints (e.g. ``[0, 0.5, 1]``).

        Example::

            # Two spatial halves via callable + 4 equal time intervals
            domain.set_partition(
                space=lambda c: (c[:, 0] > 0.5).astype(int),
                time=4)

            # Automatic K-means split into 6 groups, 3 time intervals
            domain.set_partition(space=6, time=3)
        """
        # ── Spatial partition ────────────────────────────────────────────
        if space is not None:
            centroids = (self._vertices[self._faces[:, 0]] +
                         self._vertices[self._faces[:, 1]] +
                         self._vertices[self._faces[:, 2]]) / 3.0

            if isinstance(space, (int, np.integer)):
                n_groups = int(space)
                try:
                    from sklearn.cluster import KMeans
                    km = KMeans(n_clusters=n_groups, random_state=0, n_init='auto')
                    labels = km.fit_predict(centroids,
                                            sample_weight=self._tri_areas)
                except ImportError:
                    # Fallback: sort by first principal axis via centroid PCA
                    cen = centroids - centroids.mean(axis=0)
                    _, _, vt = np.linalg.svd(cen, full_matrices=False)
                    proj = cen @ vt[0]
                    order = np.argsort(proj)
                    labels = np.empty(len(self._faces), dtype=int)
                    # Area-weighted split into n_groups equal-area buckets
                    cumarea = np.cumsum(self._tri_areas[order])
                    total   = cumarea[-1]
                    edges_t = np.linspace(0, total, n_groups + 1)
                    for g in range(n_groups):
                        mask = (cumarea > edges_t[g]) & (cumarea <= edges_t[g + 1])
                        if g == 0:
                            mask |= (cumarea <= edges_t[1])
                        labels[order[mask]] = g
            else:
                labels = np.asarray(space(centroids), dtype=int).ravel()
                if len(labels) != len(self._faces):
                    raise ValueError(
                        f"set_partition: space callable returned {len(labels)} "
                        f"labels but the mesh has {len(self._faces)} triangles.")

            self._partition_labels = labels
            unique_labels = np.unique(labels)
            self._partition_groups = {}
            for lbl in unique_labels:
                face_idx = np.where(labels == lbl)[0]
                areas    = self._tri_areas[face_idx]
                self._partition_groups[int(lbl)] = {
                    'face_indices': face_idx,
                    'tri_probs':    areas / areas.sum(),
                    'total_area':   float(areas.sum()),
                }

            # ── Interface edges (shared between triangles of different groups) ──
            edge_to_faces: dict = {}
            for fi, face in enumerate(self._faces):
                for j in range(3):
                    v0 = int(face[j])
                    v1 = int(face[(j + 1) % 3])
                    key = (min(v0, v1), max(v0, v1))
                    edge_to_faces.setdefault(key, []).append(fi)
            iface = [list(key) for key, flist in edge_to_faces.items()
                     if len(flist) == 2 and labels[flist[0]] != labels[flist[1]]]
            if iface:
                self._partition_interface_edges = np.array(iface, dtype=np.int64)
                p0 = self._vertices[self._partition_interface_edges[:, 0]]
                p1 = self._vertices[self._partition_interface_edges[:, 1]]
                self._partition_interface_lengths = np.linalg.norm(p1 - p0, axis=1)
            else:
                self._partition_interface_edges   = np.empty((0, 2), dtype=np.int64)
                self._partition_interface_lengths = np.empty(0)

        # ── Time partition ───────────────────────────────────────────────
        if time is not None:
            if self._t_min is None:
                raise ValueError(
                    "set_partition: 'time' requires the domain to have a time "
                    "axis.  Construct DomainMesh with a time=(t_min, t_max) or "
                    "time-steps array.")
            if isinstance(time, (int, np.integer)):
                self._time_grid_positions = np.linspace(
                    self._t_min, self._t_max, int(time) + 1)
            else:
                self._time_grid_positions = np.asarray(time, dtype=float).ravel()
                if len(self._time_grid_positions) < 2:
                    raise ValueError(
                        "set_partition: 'time' must have at least 2 breakpoints.")
            self._n_time_partitions = len(self._time_grid_positions) - 1

    # ── Partition sampling helpers ──────────────────────────────────────

    def _sample_partition_interior(self, n_points: int, size, rng) -> np.ndarray:
        """Sample interior points across (spatial group × time interval) cells."""
        groups = self._partition_groups
        grp_keys = list(groups.keys())
        n_sp = len(grp_keys)

        # Time intervals
        if self._time_grid_positions is not None:
            t_breaks = self._time_grid_positions
            t_spans  = np.diff(t_breaks)
            n_t      = len(t_spans)
        else:
            t_spans  = np.array([1.0]) if self._t_min is None else np.array([self._t_max - self._t_min])
            n_t      = 1

        n_cells = n_sp * n_t

        # Compute weights per cell
        sp_areas = np.array([groups[k]['total_area'] for k in grp_keys])
        if size == 'equal':
            cell_weights = np.ones(n_cells, dtype=float)
        elif size == 'size':
            cell_weights = np.tile(sp_areas, n_t) * np.repeat(t_spans, n_sp)
        else:
            arr = np.asarray(size, dtype=float)
            if arr.size == n_cells:
                cell_weights = arr.ravel()
            elif arr.size == n_sp:
                cell_weights = np.tile(arr, n_t)
            else:
                raise ValueError(
                    f"set_partition interior: 'size' has {arr.size} elements "
                    f"but there are {n_cells} cells ({n_sp} spatial × {n_t} time).")
        cell_weights = cell_weights / cell_weights.sum()

        # Largest-remainder rounding
        raw     = cell_weights * n_points
        counts  = raw.astype(int)
        remains = raw - counts
        deficit = n_points - counts.sum()
        if deficit > 0:
            top = np.argsort(remains)[::-1][:deficit]
            counts[top] += 1

        parts = []
        for ci, cnt in enumerate(counts):
            if cnt == 0:
                continue
            sp_i = ci % n_sp
            t_i  = ci // n_sp
            grp  = groups[grp_keys[sp_i]]
            # spatial barycentric
            f    = self._faces[grp['face_indices']]
            v    = self._vertices
            sel  = rng.choice(len(f), cnt, p=grp['tri_probs'])
            A    = v[f[sel, 0]]
            B    = v[f[sel, 1]]
            C    = v[f[sel, 2]]
            r1   = rng.uniform(0.0, 1.0, cnt)
            r2   = rng.uniform(0.0, 1.0, cnt)
            swap = r1 + r2 > 1.0
            r1[swap] = 1.0 - r1[swap]
            r2[swap] = 1.0 - r2[swap]
            pts_sp = r1[:, None] * A + r2[:, None] * B + (1 - r1 - r2)[:, None] * C
            # time
            if self._t_min is not None:
                if self._time_grid_positions is not None:
                    t = rng.uniform(t_breaks[t_i], t_breaks[t_i + 1], (cnt, 1))
                else:
                    t = rng.uniform(self._t_min, self._t_max, (cnt, 1))
                parts.append(np.hstack([pts_sp, t]))
            else:
                parts.append(pts_sp)

        return np.vstack(parts) if parts else np.empty((0, self.n_dims))

    def _sample_partition_outer_boundary(self, n_points: int, size, rng) -> np.ndarray:
        """Sample points on the outer mesh boundary (existing boundary edges)."""
        if self._bnd_edge_probs is None or len(self._bnd_edges) == 0:
            raise ValueError(
                "region='partition_outer': the mesh has no detected boundary edges.")
        idx     = rng.choice(len(self._bnd_edges), n_points,
                              p=self._bnd_edge_probs)
        t_param = rng.uniform(0.0, 1.0, (n_points, 1))
        v0      = self._vertices[self._bnd_edges[idx, 0]]
        v1      = self._vertices[self._bnd_edges[idx, 1]]
        pts_sp  = v0 + t_param * (v1 - v0)
        if self._t_min is not None:
            return np.hstack([pts_sp,
                               rng.uniform(self._t_min, self._t_max, (n_points, 1))])
        return pts_sp

    def _sample_partition_inner_boundary(self, n_points: int, size, rng) -> np.ndarray:
        """Sample points on interface edges between partition groups."""
        if (self._partition_interface_edges is None or
                len(self._partition_interface_edges) == 0):
            raise ValueError(
                "region='partition_inner': no interface edges found. "
                "Did you call set_partition(space=...)?")
        lengths = self._partition_interface_lengths
        probs   = lengths / lengths.sum()
        idx     = rng.choice(len(self._partition_interface_edges), n_points, p=probs)
        t_param = rng.uniform(0.0, 1.0, (n_points, 1))
        v0      = self._vertices[self._partition_interface_edges[idx, 0]]
        v1      = self._vertices[self._partition_interface_edges[idx, 1]]
        pts_sp  = v0 + t_param * (v1 - v0)
        if self._t_min is not None:
            return np.hstack([pts_sp,
                               rng.uniform(self._t_min, self._t_max, (n_points, 1))])
        return pts_sp

    # ------------------------------------------------------------------ #
    #  Region registration                                                #
    # ------------------------------------------------------------------ #

    def _resolve_for_region(self, select) -> np.ndarray:
        """Resolve a region selector to node indices.

        Supports all forms accepted by :meth:`_resolve_node_select`, plus a
        **bounding-box** shortcut: a list of ``(lo, hi)`` tuples, one per
        spatial dimension.

        Examples::

            # Callable selector
            domain._resolve_for_region(lambda v: v[:, 0] < 0.5)

            # Bounding box [(x_lo, x_hi), (y_lo, y_hi)]
            domain._resolve_for_region([(0.0, 0.5), (0.0, 1.0)])
        """
        # Bounding-box shortcut: list/array of 2-element numeric tuples
        if (isinstance(select, (list, np.ndarray)) and len(select) > 0):
            first = select[0]
            if (isinstance(first, (tuple, list)) and len(first) == 2
                    and all(isinstance(x, (int, float, np.floating)) for x in first)):
                v = self._vertices
                mask = np.ones(len(v), dtype=bool)
                for dim_i, (lo, hi) in enumerate(select):
                    if dim_i >= v.shape[1]:
                        break
                    mask &= (v[:, dim_i] >= lo) & (v[:, dim_i] <= hi)
                return np.where(mask)[0].astype(np.intp)
        return self._resolve_node_select(select)

    def _resolve_time_region(self, time):
        """Parse the ``time`` argument used in :meth:`add_inner` /
        :meth:`add_boundary`.

        * ``None``        → ``(t_min, t_max)`` (full domain range, or both None
                            for a stationary domain).
        * ``(t_lo, t_hi)``  → explicit sub-interval (requires time axis).

        Raises:
            ValueError: On a stationary domain when *time* is not None, or
                when the requested window lies outside the domain range.
        """
        if time is None:
            return (self._t_min, self._t_max)
        if self._t_min is None:
            raise ValueError(
                "Cannot restrict a region's time window on a stationary "
                "domain (no time axis).  Pass time=None or add a time axis.")
        _t = np.asarray(time, dtype=float).ravel()
        if len(_t) != 2:
            raise ValueError(
                f"'time' for a region must be None or a 2-element (t_lo, t_hi) "
                f"tuple; got {time!r}.")
        t_lo, t_hi = float(_t[0]), float(_t[1])
        if t_lo > t_hi + 1e-12:
            raise ValueError(
                f"Region time window must have t_lo ≤ t_hi; got ({t_lo}, {t_hi}).")
        if t_lo < self._t_min - 1e-10 or t_hi > self._t_max + 1e-10:
            raise ValueError(
                f"Region time window ({t_lo}, {t_hi}) is outside the domain "
                f"time range [{self._t_min}, {self._t_max}].")
        return (t_lo, t_hi)

    @property
    def interior_node_mask(self) -> np.ndarray:
        """Boolean mask of shape ``(n_nodes,)`` — ``True`` for interior nodes."""
        return ~self._boundary_node_mask

    def add_inner(self, select, name: str, time=None, strict: bool = True) -> None:
        """Register a named **interior** sampling region.

        The region is defined by *select*, which identifies the mesh vertices
        that belong to it.  By default (*strict=True*) only triangles whose
        **all three** vertices are inside the selected set are included;
        with *strict=False* a triangle is included if **any** of its vertices
        is inside, which avoids gaps at region boundaries.

        Args:
            select: One of:

                * **Callable** ``(v: ndarray) → bool_mask`` — called with all
                  vertex positions ``(n_verts, spatial_dims)``.
                  Example: ``lambda v: v[:, 0] < 0.5``.
                * **Boolean array** of shape ``(n_verts,)`` — direct mask.
                * **1-D integer array** — explicit vertex indices.
                * **List of ``(lo, hi)`` tuples** — axis-aligned bounding box,
                  one tuple per spatial dimension.
                  Example: ``[(0.0, 0.5), (0.0, 1.0)]``.

            name: String label used in :meth:`sample_interior` as the
                ``region=`` key.
            time: Optional time restriction:

                * ``None`` — full domain time range (default).
                * ``(t_lo, t_hi)`` — restrict to this sub-interval.

            strict: If ``True`` (default) a triangle is included only when
                **all three** of its vertices are in the selected set.  If
                ``False`` a triangle is included when **at least one** vertex
                is in the selected set.  Use ``strict=False`` when adjacent
                regions share a boundary threshold (e.g. ``>= 1.0`` and
                ``<= 1.0``) so that straddling triangles are never dropped.

        Raises:
            ValueError: If no triangle matches the criterion.

        Example::

            domain.add_inner(lambda v: v[:, 0] < 0.5, name='left_half')
            domain.add_inner([(0.3, 0.7), (0.3, 0.7)], name='centre')
            domain.add_inner(lambda v: v[:, 0] < 0.5, name='left_t0',
                             time=(0.0, 0.5))
            # Non-strict: no gaps at shared threshold y = 1.0
            domain.add_inner(lambda v: v[:, 1] >= 1.0, name='upper', strict=False)
            domain.add_inner(lambda v: v[:, 1] <= 1.0, name='lower', strict=False)
        """
        node_idx = self._resolve_for_region(select)
        if len(node_idx) == 0:
            raise ValueError(f"Region '{name}': selector returned no vertices.")
        node_set = set(node_idx.tolist())
        if strict:
            face_mask = np.array(
                [int(f[0]) in node_set and int(f[1]) in node_set and
                 int(f[2]) in node_set
                 for f in self._faces], dtype=bool)
        else:
            face_mask = np.array(
                [int(f[0]) in node_set or int(f[1]) in node_set or
                 int(f[2]) in node_set
                 for f in self._faces], dtype=bool)
        face_idx = np.where(face_mask)[0]
        if len(face_idx) == 0:
            raise ValueError(
                f"Region '{name}': no triangle has "
                f"{'all three' if strict else 'any'} vertices inside "
                "the selected node set.  Broaden the selector or the bounding box.")
        # Per-region triangle sampling probabilities (2D and 3D surface).
        # _tri_areas is always populated in __init__ for both dimensionalities.
        reg_areas = self._tri_areas[face_idx]
        reg_probs = reg_areas / reg_areas.sum()
        t_lo, t_hi = self._resolve_time_region(time)
        self._inner_regions[name] = {
            'node_indices': node_idx,
            'face_indices': face_idx,
            'tri_probs':    reg_probs,
            't_lo':         t_lo,
            't_hi':         t_hi,
        }

    def add_boundary(self, select, name: str, time=None, strict: bool = True) -> None:
        """Register a named **boundary** sampling region.

        The region is defined by *select*, which identifies the mesh vertices
        on the boundary of interest.  By default (*strict=True*) only boundary
        **edges** whose **both** endpoints are in the selected set are included;
        with *strict=False* an edge is included if **either** endpoint is in the
        selected set.

        Args:
            select: Same forms as in :meth:`add_inner`.
            name: String label used in :meth:`sample_boundary` as the
                ``region=`` key.
            time: Optional time restriction — ``None`` or ``(t_lo, t_hi)``.
            strict: If ``True`` (default) both endpoints of a boundary edge
                must be in the selected set.  If ``False`` at least one
                endpoint must be in the selected set.  Useful when region
                boundaries are shared.

        Raises:
            ValueError: If no boundary edge is found in the selected set.

        Example::

            # Entire x = 0 face
            domain.add_boundary(lambda v: v[:, 0] < 1e-6, name='x_left')
            # y = 0 face, restricted to x ∈ [0.2, 0.8]
            domain.add_boundary(
                lambda v: (np.abs(v[:, 1]) < 1e-6) & (v[:, 0] > 0.2) & (v[:, 0] < 0.8),
                name='y_bottom_centre')
            # Bounding-box shorthand for the same as above
            domain.add_boundary([(0.0, 0.01), (0.2, 0.8)], name='y_bottom_centre')
        """
        node_idx = self._resolve_for_region(select)
        if len(node_idx) == 0:
            raise ValueError(f"Region '{name}': selector returned no vertices.")
        node_set   = set(node_idx.tolist())
        bnd_nodes  = self._boundary_node_mask
        if strict:
            edge_mask = np.array(
                [bool(bnd_nodes[int(e[0])]) and bool(bnd_nodes[int(e[1])])
                 and int(e[0]) in node_set and int(e[1]) in node_set
                 for e in self._all_edges], dtype=bool)
        else:
            edge_mask = np.array(
                [bool(bnd_nodes[int(e[0])]) and bool(bnd_nodes[int(e[1])])
                 and (int(e[0]) in node_set or int(e[1]) in node_set)
                 for e in self._all_edges], dtype=bool)
        edge_idx = np.where(edge_mask)[0]
        if len(edge_idx) == 0:
            raise ValueError(
                f"Region '{name}': no boundary edge found in the selected node "
                "set.  Make sure the selector covers actual mesh boundary vertices.")
        edges   = self._all_edges[edge_idx]
        v0      = self._vertices[edges[:, 0]]
        v1      = self._vertices[edges[:, 1]]
        lengths = np.linalg.norm(v1 - v0, axis=1)
        # Precompute outward normals for 2-D meshes only; 3-D surface meshes
        # don't have a meaningful in-plane edge normal via this method.
        normals = (self._infer_edge_outward_normals(edges)
                   if self._spatial_dims == 2 else None)
        t_lo, t_hi = self._resolve_time_region(time)
        self._boundary_regions[name] = {
            'node_indices': node_idx,
            'edge_indices': edge_idx,
            'edges':        edges,
            'edge_lengths': lengths,
            'edge_probs':   lengths / lengths.sum(),
            'normals':      normals,
            't_lo':         t_lo,
            't_hi':         t_hi,
        }

    def add_periodic(self, select_a, select_b, name: str) -> None:
        """Register a **periodic** boundary pairing between two boundary regions.

        Pairs nodes on *select_a* with their nearest neighbours on *select_b*
        using a KD-tree (after shifting *select_a* toward *select_b*).  The
        matched arrays are stored in ``domain._periodic_regions[name]`` and
        can be referenced by name from :meth:`~pinns.problems.BaseProblem.add_periodic`.

        Args:
            select_a: Boundary-node selector for the first boundary (same forms
                as in :meth:`add_boundary`).
            select_b: Boundary-node selector for the second boundary.
            name: String label stored in ``_periodic_regions``.

        Raises:
            UserWarning: If the largest pairing distance exceeds 10% of the
                mean shift distance (nodes not well matched).

        Example::

            domain.add_periodic(lambda v: np.abs(v[:, 0]) < 1e-6,
                                 lambda v: np.abs(v[:, 0] - 1.0) < 1e-6,
                                 name='per_x')
        """
        from scipy.spatial import cKDTree
        import warnings
        edges_a = self._all_edges[self._resolve_select(select_a)]
        edges_b = self._all_edges[self._resolve_select(select_b)]
        pts_a = self._vertices[np.unique(edges_a)]
        pts_b = self._vertices[np.unique(edges_b)]
        shift = pts_b.mean(axis=0) - pts_a.mean(axis=0)
        pts_a_shifted = pts_a + shift
        tree = cKDTree(pts_b)
        dists, idx = tree.query(pts_a_shifted, k=1)
        max_dist = dists.max()
        tol = np.linalg.norm(shift) * 0.1 + 1e-10
        if max_dist > tol:
            warnings.warn(
                f"add_periodic('{name}'): largest pairing distance is {max_dist:.4g}."
                " The two boundaries may not have matching node distributions.",
                UserWarning,
            )
        pts_b_matched = pts_b[idx]
        self._periodic_regions[name] = {
            'node_positions_a': pts_a.astype(np.float32),
            'node_positions_b': pts_b_matched.astype(np.float32),
        }

    # ------------------------------------------------------------------ #
    #  Private sampling helpers                                           #
    # ------------------------------------------------------------------ #

    def _sample_region_interior_all(self, n_points: int, rng) -> np.ndarray:
        """Sample *n_points* from the entire mesh interior."""
        pts_sp = self._sample_interior_spatial(n_points, rng)
        if self._t_min is not None:
            t = rng.uniform(self._t_min, self._t_max, (n_points, 1))
            return np.hstack([pts_sp, t])
        return pts_sp

    def _sample_region_interior(self, reg_info: dict,
                                 n_points: int, rng) -> np.ndarray:
        """Barycentric sampling over a registered interior region (2D/3D surface)."""
        face_idx  = reg_info['face_indices']
        tri_probs = reg_info['tri_probs']
        f         = self._faces[face_idx]
        v         = self._vertices
        tri_sel   = rng.choice(len(f), n_points, p=tri_probs)
        A = v[f[tri_sel, 0]]
        B = v[f[tri_sel, 1]]
        C = v[f[tri_sel, 2]]
        r1 = rng.uniform(0.0, 1.0, n_points)
        r2 = rng.uniform(0.0, 1.0, n_points)
        swap       = r1 + r2 > 1.0
        r1[swap]   = 1.0 - r1[swap]
        r2[swap]   = 1.0 - r2[swap]
        pts_sp = r1[:, None] * A + r2[:, None] * B + (1 - r1 - r2)[:, None] * C
        t_lo = reg_info.get('t_lo')
        t_hi = reg_info.get('t_hi')
        if t_lo is not None and t_hi is not None:
            return np.hstack([pts_sp, rng.uniform(t_lo, t_hi, (n_points, 1))])
        return pts_sp

    def _sample_multi_interior(self, regions: list, n_points: int,
                                size, rng) -> np.ndarray:
        """Sample *n_points* from a list of named interior regions."""
        n = len(regions)
        if size == 'equal':
            weights = np.ones(n, dtype=float) / n
        elif size == 'area':
            areas = np.array([
                float(self._tri_areas[self._inner_regions[nm]['face_indices']].sum())
                for nm in regions])
            weights = areas / areas.sum()
        else:
            weights = np.asarray(size, dtype=float)
            if len(weights) != n:
                raise ValueError(
                    f"'size' has {len(weights)} elements but region list has {n}.")
            weights = weights / weights.sum()
        counts        = (weights * n_points).astype(int)
        counts[-1]    = n_points - counts[:-1].sum()
        parts = []
        for nm, cnt in zip(regions, counts):
            if nm not in self._inner_regions:
                raise KeyError(
                    f"Unknown interior region '{nm}'. "
                    f"Registered: {list(self._inner_regions.keys())}")
            if cnt > 0:
                parts.append(self._sample_region_interior(
                    self._inner_regions[nm], cnt, rng))
        return np.vstack(parts) if parts else np.empty((0, self.n_dims))

    def _sample_boundary_all_edges(self, n_points: int, rng) -> np.ndarray:
        """Sample *n_points* uniformly along all mesh boundary edges (2D)."""
        if self._bnd_edge_probs is None or len(self._bnd_edges) == 0:
            return self._sample_region_interior_all(n_points, rng)
        idx      = rng.choice(len(self._bnd_edges), n_points,
                               p=self._bnd_edge_probs)
        t_param  = rng.uniform(0.0, 1.0, (n_points, 1))
        v0       = self._vertices[self._bnd_edges[idx, 0]]
        v1       = self._vertices[self._bnd_edges[idx, 1]]
        pts_sp   = v0 + t_param * (v1 - v0)
        if self._t_min is not None:
            return np.hstack([pts_sp, rng.uniform(self._t_min, self._t_max,
                                                   (n_points, 1))])
        return pts_sp

    def _sample_boundary_region(self, reg_info: dict,
                                 n_points: int, rng) -> np.ndarray:
        """Sample *n_points* from a registered boundary region."""
        edges  = reg_info['edges']
        probs  = reg_info['edge_probs']
        idx    = rng.choice(len(edges), n_points, p=probs)
        t_par  = rng.uniform(0.0, 1.0, (n_points, 1))
        v0     = self._vertices[edges[idx, 0]]
        v1     = self._vertices[edges[idx, 1]]
        pts_sp = v0 + t_par * (v1 - v0)
        t_lo = reg_info.get('t_lo')
        t_hi = reg_info.get('t_hi')
        if t_lo is not None and t_hi is not None:
            return np.hstack([pts_sp, rng.uniform(t_lo, t_hi, (n_points, 1))])
        return pts_sp

    def _sample_multi_boundary(self, regions: list, n_points: int,
                                size, rng) -> np.ndarray:
        """Sample *n_points* from a list of named boundary regions."""
        n = len(regions)
        if size == 'equal':
            weights = np.ones(n, dtype=float) / n
        elif size == 'length':
            lengths = np.array([
                float(self._boundary_regions[nm]['edge_lengths'].sum())
                for nm in regions])
            weights = lengths / lengths.sum()
        else:
            weights = np.asarray(size, dtype=float)
            if len(weights) != n:
                raise ValueError(
                    f"'size' has {len(weights)} elements but region list has {n}.")
            weights = weights / weights.sum()
        counts     = (weights * n_points).astype(int)
        counts[-1] = n_points - counts[:-1].sum()
        parts = []
        for nm, cnt in zip(regions, counts):
            if nm not in self._boundary_regions:
                raise KeyError(
                    f"Unknown boundary region '{nm}'. "
                    f"Registered: {list(self._boundary_regions.keys())}")
            if cnt > 0:
                parts.append(self._sample_boundary_region(
                    self._boundary_regions[nm], cnt, rng))
        return np.vstack(parts) if parts else np.empty((0, self.n_dims))

    def _sample_interior_spatial(self, n_points: int, rng) -> np.ndarray:
        """Sample *n_points* spatial points inside the mesh.

        Uses exact barycentric sampling for both 2-D flat meshes and 3-D
        surface meshes.  Points lie exactly on the triangulated surface in
        either case.
        """
        # ---- 2-D flat mesh OR 3-D surface mesh: barycentric sampling ------
        # This is exact and works regardless of spatial dimensionality because
        # we parameterise each triangle with two barycentric coordinates and
        # interpolate vertex positions in R^d.
        v = self._vertices
        f = self._faces
        tri_idx = rng.choice(len(f), n_points, p=self._tri_probs)
        A = v[f[tri_idx, 0]]
        B = v[f[tri_idx, 1]]
        C = v[f[tri_idx, 2]]
        r1 = rng.uniform(0.0, 1.0, n_points)
        r2 = rng.uniform(0.0, 1.0, n_points)
        mask = r1 + r2 > 1.0
        r1[mask] = 1.0 - r1[mask]
        r2[mask] = 1.0 - r2[mask]
        r3 = 1.0 - r1 - r2
        return r1[:, None] * A + r2[:, None] * B + r3[:, None] * C

    def _resolve_node_select(self, select) -> np.ndarray:
        """
        Return a 1-D integer array of **node indices** matching *select*.

        *select* can be one of:

        - **Callable** — called with ``self._vertices`` ``(n_verts, spatial_dims)``;
          must return a boolean mask of shape ``(n_verts,)``.
          Example: ``lambda v: v[:, 0] < 1e-6`` (x ≈ 0 plane).

        - **1-D integer array** — explicit node index list.
          Example: ``np.arange(100)``.

        - **Boolean array** of shape ``(n_verts,)`` — treated as a mask.

        - **2-D ``(n, 2)`` integer array** — interpreted as edge vertex-pair
          table (e.g. ``mesh.cells_dict["line"]``); the unique node indices
          contained in those edges are returned (backward-compatible usage
          for Neumann BCs sourced directly from mesh boundary cells).
        """
        if callable(select):
            mask = np.asarray(select(self._vertices), dtype=bool)
            return np.where(mask)[0].astype(np.intp)
        arr = np.asarray(select)
        if arr.dtype == bool:
            return np.where(arr)[0].astype(np.intp)
        if arr.ndim == 1:
            return arr.astype(np.intp)
        if arr.ndim == 2 and arr.shape[1] == 2:
            # Edge-pair table — extract unique node indices
            return np.unique(arr.ravel()).astype(np.intp)
        raise ValueError(
            "select must be a callable, a boolean mask, a 1-D integer array, "
            f"or a (n, 2) edge-pair array.  Got array with shape {arr.shape}."
        )

    # keep _resolve_select as an alias for internal / backward-compat callers
    def _resolve_select(self, select) -> np.ndarray:
        """Legacy alias — converts node selector to edge indices.

        Calls :meth:`_resolve_node_select` then returns indices of edges whose
        **both** endpoints are in the selected node set.
        """
        node_idx = self._resolve_node_select(select)
        return self.node_indices_to_edge_indices(node_idx)

    def edge_pairs_to_indices(self, edge_pairs: np.ndarray) -> np.ndarray:
        """
        Convert an array of ``(v0, v1)`` vertex-index pairs to edge indices.

        Looks up each pair in ``self._edge_lookup`` (the canonical
        ``(min_v, max_v) -> edge_index`` dict built at construction time).
        This is the most direct way to go from the ``"line"`` cells that Gmsh /
        meshio stores for physical boundaries to the edge arrays expected by
        :class:`~pinns.boundary.TermMeshNodeBC` (``edges``, ``edge_lengths``,
        ``edge_normals`` fields).

        Args:
            edge_pairs: ``(n, 2)`` integer array of vertex index pairs,
                        e.g. ``mesh.cells_dict["line"]`` for a physical group.

        Returns:
            1-D integer array of indices into ``self._all_edges``.
            Pairs not found in the mesh are silently skipped.

        Example::

            domain = DomainMesh(mesh)
            line_cells = mesh.cells_dict["line"]  # all boundary segments
            eidx = domain.edge_pairs_to_indices(line_cells)
            edges = domain._all_edges[eidx]
            # → pass edges to TermMeshNodeBC(edges=edges, ...)
        """
        indices = []
        for v0, v1 in edge_pairs:
            key = (min(int(v0), int(v1)), max(int(v0), int(v1)))
            idx = self._edge_lookup.get(key)
            if idx is not None:
                indices.append(idx)
        return np.array(indices, dtype=np.intp)

    def node_indices_to_edge_indices(self, node_indices: np.ndarray) -> np.ndarray:
        """
        Convert vertex indices to edge indices.

        Returns the indices (into ``self._all_edges``) of every mesh edge whose
        **both** endpoints are in *node_indices*.

        Args:
            node_indices: 1-D integer array of vertex indices (e.g. from a
                          physical-group node helper).

        Returns:
            1-D integer array of indices into ``self._all_edges``.
        """
        node_set = set(node_indices.tolist())
        v0_ok = np.array([int(e[0]) in node_set for e in self._all_edges])
        v1_ok = np.array([int(e[1]) in node_set for e in self._all_edges])
        return np.where(v0_ok & v1_ok)[0].astype(np.intp)

    def _infer_edge_outward_normals(self, edges: np.ndarray) -> np.ndarray:
        """
        Compute per-edge outward unit normals for a 2D boundary.

        For each edge the tangent is ``v1 − v0``; rotating 90° CCW gives a
        candidate normal ``(−dy, dx)`` that is then flipped to point away from
        the mesh centroid.

        Args:
            edges: ``(n_edges, 2)`` vertex index pairs (into ``self._vertices``).

        Returns:
            ``(n_edges, 2)`` outward unit normals, one per edge.
        """
        v0 = self._vertices[edges[:, 0]]   # (n_edges, 2)
        v1 = self._vertices[edges[:, 1]]
        tangents = v1 - v0                 # (n_edges, 2)

        # Rotate 90° CCW: (tx, ty) → (−ty, tx)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        norms   = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.where(norms > 0, norms, 1.0)

        # Orient away from mesh centroid
        centroid     = self._vertices.mean(axis=0)
        edge_centers = 0.5 * (v0 + v1)
        outward      = edge_centers - centroid
        flip         = (normals * outward).sum(axis=1) < 0
        normals[flip] *= -1
        return normals

    def get_boundary_normals(self, x: np.ndarray, region: str) -> np.ndarray:
        """Return outward unit normals at *x* for the given boundary *region*.

        For each query point the normal of the nearest boundary edge
        (by midpoint distance) is returned.  Normals are precomputed when
        :meth:`add_boundary` is called and are available for **2-D meshes**
        only; 3-D surface meshes do not have unambiguous in-plane normals.

        Args:
            x:      Query coordinates, shape ``(n, n_dims)``.
            region: Name of a boundary region registered with
                    :meth:`add_boundary`.

        Returns:
            ``np.ndarray`` of shape ``(n, 2)`` outward unit normals.

        Raises:
            KeyError:            If *region* has not been registered.
            NotImplementedError: If the mesh is 3-D (normals unavailable).
        """
        if region not in self._boundary_regions:
            raise KeyError(
                f"Boundary region '{region}' not registered. "
                f"Available: {list(self._boundary_regions.keys())}"
            )
        reg = self._boundary_regions[region]
        normals_arr = reg.get('normals')
        if normals_arr is None:
            raise NotImplementedError(
                f"Outward normals are not available for region '{region}' on a "
                "3-D DomainMesh.  Only 2-D meshes precompute edge normals."
            )
        # Build (and cache) a KD-tree over edge midpoints.
        if '_normal_tree' not in reg:
            from scipy.spatial import KDTree
            edges = reg['edges']
            midpoints = (
                self._vertices[edges[:, 0]] +
                self._vertices[edges[:, 1]]
            ) / 2.0
            reg['_normal_tree'] = KDTree(midpoints)
        tree = reg['_normal_tree']
        _, idx = tree.query(x[:, :normals_arr.shape[1]])
        return normals_arr[idx].astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Temporal-overlap validation                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tw_overlaps(tw_a, tw_b, t_min: float, t_max: float,
                     tol: float = 1e-10) -> bool:
        """
        Return ``True`` if two time windows share any common time.

        Rules
        -----
        * ``None`` stands for the full domain ``[t_min, t_max]``.
        * A 2-element list/tuple ``[a, b]`` is a closed interval.
        * A list with > 2 elements is a discrete set; each value is compared
          against the other window's discrete set or interval.
        """
        def to_interval(tw):
            """Reduce any time window to (lo, hi); discrete → (min, max)."""
            if tw is None:
                return (t_min, t_max)
            pts = [float(v) for v in tw]
            return (min(pts), max(pts))

        # Both discrete: exact-value intersection
        def is_discrete(tw):
            return tw is not None and len(tw) > 2

        if is_discrete(tw_a) and is_discrete(tw_b):
            # Round to tol grid to handle float noise
            grid = max(tol, 1e-14)
            ka = {round(float(v) / grid) for v in tw_a}
            kb = {round(float(v) / grid) for v in tw_b}
            return bool(ka & kb)

        if is_discrete(tw_a) or is_discrete(tw_b):
            # One discrete, one interval: check whether any discrete value
            # falls inside the interval
            disc, cont = (tw_a, tw_b) if is_discrete(tw_a) else (tw_b, tw_a)
            lo, hi = to_interval(cont)
            return any(lo - tol <= float(v) <= hi + tol for v in disc)

        # Both intervals (or None)
        # Two intervals overlap only if they share more than a single point.
        # Touching at exactly one endpoint ([0,0.5] and [0.5,1]) is NOT
        # an overlap — it is zero-measure and is allowed.
        a_lo, a_hi = to_interval(tw_a)
        b_lo, b_hi = to_interval(tw_b)
        return max(a_lo, b_lo) < min(a_hi, b_hi) - tol

    def _check_bc_time_overlap(self, new_bc) -> None:
        """
        Raise ``ValueError`` if *new_bc* conflicts in time with any already-
        registered BC on the same (bc_type, component, nodes) combination.

        A conflict exists when both BCs have at least one shared **node** *and*
        their **time windows overlap** — meaning the same node would be
        assigned two different conditions at the same time.
        """
        if not hasattr(new_bc, 'node_positions'):
            return

        t_min = self._t_min if self._t_min is not None else 0.0
        t_max = self._t_max if self._t_max is not None else 1.0

        for existing in self.boundary_conditions:
            if not hasattr(existing, 'node_positions'):
                continue
            if existing.bc_type != new_bc.bc_type:
                continue
            if existing.component != new_bc.component:
                continue

            # Check node-set intersection
            n_ex  = existing.node_indices
            n_new = new_bc.node_indices
            if n_ex is None or n_new is None:
                continue   # can't determine without index arrays
            common = np.intersect1d(n_ex, n_new)
            if len(common) == 0:
                continue

            # Check temporal overlap
            if self._tw_overlaps(
                existing.time_window, new_bc.time_window, t_min, t_max
            ):
                raise ValueError(
                    f"BC '{new_bc.name}' "
                    f"(type='{new_bc.bc_type}', component={new_bc.component}, "
                    f"time_window={new_bc.time_window}) overlaps in time with "
                    f"existing BC '{existing.name}' "
                    f"(time_window={existing.time_window}) on "
                    f"{len(common)} shared node(s).  "
                    "Adjust the time_window of one of the two conditions so "
                    "that they do not cover the same time simultaneously."
                )

    # ------------------------------------------------------------------ #
    #  Public sampling API (called by the trainer)                        #
    # ------------------------------------------------------------------ #

    def sample_interior(self, n_points: int, region=None, size='equal',
                        rng=None, **kwargs) -> np.ndarray:
        """Sample interior collocation points of shape ``(n_points, n_dims)``.

        Args:
            n_points: Number of points to return.
            region: Which region to sample from:

                * ``None`` or ``'all'`` — full mesh interior (default).
                * ``'name'`` — a named region registered with
                  :meth:`add_inner`.
                * ``['a', 'b', …]`` — multiple named regions; use *size*
                  to control the distribution of *n_points*.

            size: Distribution strategy when *region* is a list:

                * ``'equal'`` — equal split (default).
                * ``'area'`` — weight each region by its triangle area.
                * List of floats — explicit normalised weights.

            rng: NumPy random generator.

        Returns:
            ``(n_points, n_dims)`` array.
        """
        if rng is None:
            rng = np.random.default_rng()
        if region is None or region == 'all':
            return self._sample_region_interior_all(n_points, rng)
        if region == 'partition':
            if self._partition_groups is None:
                raise ValueError(
                    "region='partition' requires set_partition(space=...) to be "
                    "called first.")
            return self._sample_partition_interior(n_points, size, rng)
        if isinstance(region, str):
            if region not in self._inner_regions:
                raise KeyError(
                    f"Unknown interior region '{region}'. "
                    f"Registered: {list(self._inner_regions.keys())}")
            return self._sample_region_interior(self._inner_regions[region],
                                                n_points, rng)
        if isinstance(region, list):
            return self._sample_multi_interior(region, n_points, size, rng)
        raise ValueError(f"Invalid region: {region!r}")

    def sample_boundary(self, n_points: int, region=None, size='equal',
                        rng=None, **kwargs) -> np.ndarray:
        """Sample boundary collocation points of shape ``(n_points, n_dims)``.

        This is the **user-facing** method for obtaining spatial (and optional
        temporal) collocation points that lie on the mesh boundary.  It is
        region-aware and mirrors :meth:`sample_interior`.

        .. note::
            This method is **not** the same as :meth:`sample_boundary_bc`.
            Use this method when building custom training loops or exploring
            the domain.  The trainer uses :meth:`sample_boundary_bc` internally
            to draw points for specific registered BCs.

        Args:
            n_points: Number of points to return.
            region: Which boundary region to sample from:

                * ``None`` or ``'all'`` — all mesh boundary edges (default).
                * ``'name'`` — a named region registered with
                  :meth:`add_boundary`.
                * ``['a', 'b', …]`` — multiple named regions.

            size: Distribution strategy when *region* is a list:

                * ``'equal'`` — equal split (default).
                * ``'length'`` — weight each region by its total edge length.
                * List of floats — explicit normalised weights.

            rng: NumPy random generator.

        Returns:
            ``(n_points, n_dims)`` array.

        Note:
            The legacy positional call ``sample_boundary(n, dim, side, rng)``
            used by :class:`ProblemWeak` is still supported.
        """
        if rng is None:
            rng = np.random.default_rng()

        # ── Legacy positional API: sample_boundary(n, dim, side, rng) ───
        # Detected when region is an integer (old dim arg positional)
        # or when 'dim' is passed as a keyword.
        _dim  = kwargs.pop('dim',  None)
        _side = kwargs.pop('side', None)
        if isinstance(region, int) or _dim is not None:
            _d = region if isinstance(region, int) else _dim
            _s = size   if isinstance(region, int) else _side
            pts_sp = self._sample_interior_spatial(n_points, rng)
            if self._t_min is not None and _d == self._spatial_dims:
                t_val = self._t_min if _s == 0 else self._t_max
                return np.hstack([pts_sp, np.full((n_points, 1), t_val)])
            return self._sample_region_interior_all(n_points, rng)

        # ── New region-based API ─────────────────────────────────────────
        if region is None or region == 'all':
            return self._sample_boundary_all_edges(n_points, rng)
        if region == 'partition_outer':
            return self._sample_partition_outer_boundary(n_points, size, rng)
        if region == 'partition_inner':
            return self._sample_partition_inner_boundary(n_points, size, rng)
        if region == 'partition':
            # split n_points between outer and inner boundary by length
            outer_total = (self._bnd_edge_lengths.sum()
                           if self._bnd_edge_lengths is not None else 0.0)
            inner_total = (self._partition_interface_lengths.sum()
                           if self._partition_interface_lengths is not None
                              and len(self._partition_interface_lengths) > 0 else 0.0)
            total = outer_total + inner_total
            if total == 0:
                raise ValueError("region='boundary': no boundary or interface edges found.")
            n_outer = int(round(n_points * outer_total / total))
            n_inner = n_points - n_outer
            parts = []
            if n_outer > 0:
                parts.append(self._sample_partition_outer_boundary(n_outer, size, rng))
            if n_inner > 0:
                parts.append(self._sample_partition_inner_boundary(n_inner, size, rng))
            return np.vstack(parts)
        # ── periodic region: return stacked pair (side A then side B) ────
        if isinstance(region, str) and region in self._periodic_regions:
            _pr = self._periodic_regions[region]
            x_a = np.asarray(_pr['node_positions_a'], dtype=np.float32)
            x_b = np.asarray(_pr['node_positions_b'], dtype=np.float32)
            return np.concatenate([x_a, x_b], axis=0)

        if isinstance(region, str):
            if region not in self._boundary_regions:
                raise KeyError(
                    f"Unknown boundary region '{region}'. "
                    f"Registered: {list(self._boundary_regions.keys())}")
            return self._sample_boundary_region(self._boundary_regions[region],
                                                n_points, rng)
        if isinstance(region, list):
            return self._sample_multi_boundary(region, n_points, size, rng)
        raise ValueError(f"Invalid region: {region!r}")

    def sample_nodes(self, n: int, rng, node_pool=None, region=None) -> np.ndarray:
        """Sample *n* FEM nodes with uniformly-random times (for stochastic Galerkin).

        Returns an ``(n, 2)`` float32 array:

        * column 0 — node index (cast to ``float32``; recover with ``int32()``)
        * column 1 — time sampled from ``Uniform(t_min, t_max)``, or ``0.0`` for
          purely-spatial domains

        Parameters
        ----------
        n : int
            Number of (node, time) pairs to sample.
        rng :
            NumPy ``Generator`` (e.g. ``np.random.default_rng(…)``).
        node_pool : array-like of int, optional
            Explicit subset of node indices to sample from.  When combined with
            *region*, the pool is the intersection of *node_pool* and region nodes.
            If *None*, the pool is determined solely by *region*.
        region : str or None, optional
            Name of a registered inner region (see :meth:`add_inner`).  Only
            nodes that belong to that region's face set are eligible.  The
            region's time window ``(t_lo, t_hi)`` is also used when sampling
            times.  When combined with *node_pool*, the two are intersected.
        """
        if node_pool is not None:
            pool = np.asarray(node_pool, dtype=np.int32)
            if region is not None:
                if region not in self._inner_regions:
                    raise KeyError(
                        f"sample_nodes: unknown region '{region}'. "
                        f"Registered inner regions: {list(self._inner_regions.keys())}")
                reg_info = self._inner_regions[region]
                face_idx = reg_info['face_indices']
                reg_nodes = np.unique(self._faces[face_idx].ravel()).astype(np.int32)
                pool = np.intersect1d(pool, reg_nodes)
        elif region is not None:
            if region not in self._inner_regions:
                raise KeyError(
                    f"sample_nodes: unknown region '{region}'. "
                    f"Registered inner regions: {list(self._inner_regions.keys())}")
            reg_info = self._inner_regions[region]
            face_idx = reg_info['face_indices']
            pool = np.unique(self._faces[face_idx].ravel()).astype(np.int32)
        else:
            pool = np.arange(len(self._vertices), dtype=np.int32)

        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)
        replace = n > len(pool)
        chosen = rng.choice(pool, size=n, replace=replace).astype(np.float32)

        # Determine time range: respect region's time window if available
        if self._t_min is not None:
            if region is not None and region in self._inner_regions:
                reg_info = self._inner_regions[region]
                t_lo = reg_info.get('t_lo') if reg_info.get('t_lo') is not None else self._t_min
                t_hi = reg_info.get('t_hi') if reg_info.get('t_hi') is not None else self._t_max
            else:
                t_lo, t_hi = self._t_min, self._t_max
            times = rng.uniform(t_lo, t_hi, size=n).astype(np.float32)
        else:
            times = np.zeros(n, dtype=np.float32)
        return np.column_stack([chosen, times])  # (n, 2)

    def sample_boundary_bc(self, bc, n_points: int, rng=None) -> np.ndarray:
        """Sample *n_points* for a registered BC term.

        Geometry is looked up from ``domain._boundary_regions[bc.region]`` —
        the Term itself carries only the region name string.

        Returns ``(pts, edge_idx)`` where *edge_idx* indexes into the region's
        edge array (needed by the trainer to fetch per-edge normals for Neumann
        BCs).  For node-only regions (e.g. initial conditions) *edge_idx*
        contains node indices instead.

        A time coordinate is appended according to ``bc.time_window``.

        Args:
            bc: A BC Term with a ``region`` attribute.
            n_points: Number of collocation points to draw.
            rng: NumPy random generator.

        Returns:
            ``(pts, idx)`` where *pts* is ``(n_points, n_dims)`` and *idx*
            is a 1-D integer array of length *n_points*.
        """
        if rng is None:
            rng = np.random.default_rng()

        region = getattr(bc, 'region', None)
        if region == 'all':
            edges   = self._bnd_edges
            lengths = self._bnd_edge_lengths
            probs   = self._bnd_edge_probs
        elif region and region in self._boundary_regions:
            reg     = self._boundary_regions[region]
            edges   = reg.get('edges')
            lengths = reg.get('edge_lengths')
            probs   = reg.get('edge_probs')
        else:
            edges = lengths = probs = None

        if edges is not None:
            # ── Edge-based sampling: uniform along boundary edges ──────────
            idx     = rng.choice(len(edges), size=n_points, p=probs)
            t_param = rng.uniform(0.0, 1.0, (n_points, 1))
            v0      = self._vertices[edges[idx, 0]]
            v1      = self._vertices[edges[idx, 1]]
            pts_sp  = v0 + t_param * (v1 - v0)
        elif region and region in self._boundary_regions:
            # ── Node-only fallback (e.g. IC regions with no edges) ─────────
            ni      = self._boundary_regions[region]['node_indices']
            idx     = rng.integers(0, len(ni), n_points)
            pts_sp  = self._vertices[ni[idx]]
        elif hasattr(bc, 'node_positions'):
            # ── Legacy fallback for any BC that still carries node_positions ─
            n_nodes = len(bc.node_positions)
            idx     = rng.integers(0, n_nodes, n_points)
            pts_sp  = bc.node_positions[idx]
        else:
            raise ValueError(
                f"sample_boundary_bc: cannot sample BC with region={region!r}: "
                "no edges or node_indices found in domain._boundary_regions."
            )

        tw = getattr(bc, 'time_window', None)
        if self._t_min is None:
            return pts_sp, idx
        if tw is None:
            # BC has no explicit time_window: infer from term kind.
            # Initial-condition terms are sampled at t_min only; all other
            # BCs (dirichlet, neumann, …) are sampled over the full time range.
            kind = getattr(bc, 'kind', None)
            if kind == 'initial':
                tw = [self._t_min, self._t_min]
            else:
                tw = [self._t_min, self._t_max]
        pts_tw = [float(v) for v in tw]
        if len(pts_tw) == 0:
            return pts_sp, idx
        if len(pts_tw) == 2 and abs(pts_tw[0] - pts_tw[1]) < 1e-12:
            # degenerate interval — single fixed time (e.g. initial condition)
            t = np.full((n_points, 1), pts_tw[0])
        elif len(pts_tw) == 2:
            # continuous interval [a, b] — uniform sampling within window
            t = rng.uniform(pts_tw[0], pts_tw[1], (n_points, 1))
        else:
            # discrete list of time values — pick randomly
            chosen = rng.choice(pts_tw, size=n_points)
            t = chosen.reshape(-1, 1)
        return np.hstack([pts_sp, t]), idx

    def get_face_normal_direction(self, region: str):
        """Mesh boundary normals are always per-point; return ``None``.

        The trainer must use the per-point normals stored in the BC object
        (``bc._sampled_normals``) for mesh domains.
        """
        return None

    # ------------------------------------------------------------------ #
    #  Boundary-condition builders                                        #
    # ------------------------------------------------------------------ #

    def _default_time_window(self):
        """Return the default time_window for this domain's time mode."""
        if self._time_mode == 'continuous':
            return [self._t_min, self._t_max]
        if self._time_mode == 'discrete':
            return self._time_points
        return None

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bc_label(bc, color_idx: int) -> str:
        """Build a human-readable legend label for one TermMeshNodeBC."""
        label = bc.name or f"bc_{color_idx}"
        if bc.bc_type:
            label = f"[{bc.bc_type[0].upper()}] {label}"
        return label

    @staticmethod
    def _tw_label(tw) -> str:
        """Short string describing a time_window."""
        if tw is None:
            return "spatial"
        pts = [float(v) for v in tw]
        if len(pts) == 1 or (len(pts) == 2 and abs(pts[0] - pts[-1]) < 1e-12):
            return f"t = {pts[0]:.4g}"
        if len(pts) == 2:
            return f"t ∈ [{pts[0]:.4g}, {pts[1]:.4g}]"
        return f"{len(pts)} time steps"

    def _draw_bc_on_ax(self, ax, bc, color, label,
                       node_size: float, show_normals: bool = False) -> None:
        """Draw a single TermMeshNodeBC onto *ax*."""
        v = self._vertices
        if bc.node_indices is not None:
            pts = v[bc.node_indices]
        else:
            pts = bc.node_positions
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=node_size, c=color, zorder=3, label=label)
        if bc.bc_type == "neumann" and bc.edges is not None:
            for e in bc.edges:
                p0, p1 = v[e[0]], v[e[1]]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color=color, linewidth=2.5, zorder=4)

    def _draw_background(self, ax, show_mesh: bool) -> None:
        """Draw mesh triangulation edges."""
        import matplotlib.tri as mtri
        v, f = self._vertices, self._faces
        if show_mesh:
            tri = mtri.Triangulation(v[:, 0], v[:, 1], f)
            ax.triplot(tri, color="#cccccc", linewidth=0.5, zorder=1)

    @staticmethod
    def _is_jupyter() -> bool:
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    def plot(
        self,
        show_overlaps: bool = False,
        region=None,
        boundary=None,
        show_mesh: bool = True,
        node_size: float = 30.0,
        figsize=None,
        points=None,
        backend: str = 'auto',
    ):
        """
        Visualise the mesh with named regions and boundary conditions.

        Works for **2-D** and **3-D surface** meshes.
        For 3-D meshes, ``backend='pyvista'`` is recommended.

        Args:
            show_overlaps (bool): When ``True`` and the domain has a time axis,
                split into one subplot per time phase.  Default ``False``.
            region: Highlight named **interior** sampling regions:

                * ``None`` or ``'all'`` — highlight all registered inner
                  regions (default).
                * ``'name'`` — highlight that specific region.
                * ``['a', 'b', …]`` — highlight those regions.
                * ``'none'`` — don't highlight inner regions.

            boundary: Highlight named **boundary** sampling regions:

                * ``None`` — don't highlight boundary regions (default).
                * ``'all'`` — highlight all registered boundary regions.
                * ``'name'`` or ``['a', 'b', …]`` — specific regions.

                Registered boundary conditions are also drawn when this is
                not ``None``.
            show_mesh (bool): Draw triangulation edges (default ``True``).
            node_size (float): Scatter-point size for BC node markers (default 30).
            figsize (tuple | None): Figure size; auto-computed when ``None``.
            points (array-like | None): ``(N, D)`` array of collocation points
                to scatter on every panel.  Plotted in red.  Default ``None``.
            backend (str): ``'auto'``, ``'matplotlib'``, or ``'pyvista'``.

        Returns:
            Axes or array of Axes (matplotlib).

        Example::

            domain.plot()
            domain.plot(region='all', boundary='all')
            domain.plot(region='centre', boundary='x_left', points=pts)
        """
        _backend = backend
        if _backend == 'auto':
            _backend = 'pyvista' if self._is_jupyter() else 'matplotlib'

        if _backend == 'pyvista':
            return self._plot_pyvista(show_mesh=show_mesh,
                                       node_size=node_size)

        if self._spatial_dims not in (2, 3):
            raise NotImplementedError(
                "DomainMesh.plot() requires a 2-D or 3-D spatial mesh."
            )

        import matplotlib.pyplot as plt
        import matplotlib.collections as mcoll

        mesh_bcs = [bc for bc in self.boundary_conditions
                    if hasattr(bc, 'node_positions')]

        # ── Resolve which inner/boundary regions to highlight ─────────────
        # 'all'  → draw the entire mesh / boundary as one unified colour
        # list   → draw each named region in a distinct colour
        # name   → draw only that region
        # None   → draw nothing
        _inner_all = (region == 'all')
        _inner_partition = region in ('partition', 'all')
        if region is None or region == 'none':
            _inner_highlight = []
        elif region == 'all':
            # partition grid + all custom named regions
            _inner_highlight = list(self._inner_regions.keys())
        elif region in ('partition', 'subdomains'):
            # partition grid only, no custom named regions
            _inner_highlight = []
        elif region in ('custom', 'inner'):
            # custom named regions only, no partition grid
            _inner_highlight = list(self._inner_regions.keys())
        elif isinstance(region, str):
            _inner_highlight = [region]
        elif isinstance(region, list):
            _inner_highlight = region
        else:
            _inner_highlight = []

        _bnd_all = (boundary == 'all')
        _bnd_inner = boundary in ('partition_inner', 'partition')
        _bnd_outer = boundary in ('partition_outer', 'partition', 'all')
        if boundary is None:
            _bnd_highlight = []
        elif boundary == 'all':
            # partition boundaries + all custom named boundary regions
            _bnd_highlight = list(self._boundary_regions.keys())
        elif boundary in ('partition_inner', 'partition_outer', 'partition'):
            # partition boundaries only, no custom named boundary regions
            _bnd_highlight = []
        elif boundary in ('custom', 'inner'):
            # custom named boundary regions only
            _bnd_highlight = list(self._boundary_regions.keys())
        elif isinstance(boundary, str):
            _bnd_highlight = [boundary]
        elif isinstance(boundary, list):
            _bnd_highlight = boundary
        else:
            _bnd_highlight = []

        # ── Build time phases from BC breakpoints ──────────────────────────
        has_time = self._t_min is not None
        breakpoints: list = []

        if has_time and show_overlaps:
            breakpoints.extend([self._t_min, self._t_max])
            for bc in mesh_bcs:
                tw = bc.time_window
                if tw is not None:
                    for bv in tw:
                        breakpoints.append(float(bv))

        tol = 1e-10
        unique_bps: list = []
        for bv in sorted(set(round(b, 12) for b in breakpoints)):
            if not unique_bps or abs(bv - unique_bps[-1]) > tol:
                unique_bps.append(bv)

        if show_overlaps and len(unique_bps) >= 2:
            phases = [(unique_bps[i], unique_bps[i + 1])
                      for i in range(len(unique_bps) - 1)]
        else:
            phases = [None]

        # ── Colour palettes ────────────────────────────────────────────────
        _cyc  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bc_color: dict = {id(bc): _cyc[i % len(_cyc)]
                          for i, bc in enumerate(mesh_bcs)}
        # separate palettes for inner / boundary region highlights
        _inner_palette   = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd',
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        _bnd_palette     = ['#ff7f0e', '#17becf', '#e377c2', '#bcbd22',
                             '#9467bd', '#8c564b', '#7f7f7f', '#1f77b4']
        _inner_color  = {nm: _inner_palette[i % len(_inner_palette)]
                         for i, nm in enumerate(_inner_highlight)}
        _bnd_color    = {nm: _bnd_palette[i % len(_bnd_palette)]
                         for i, nm in enumerate(_bnd_highlight)}

        # ── Helper: does a BC's time_window cover a phase? ─────────────────
        def _bc_in_phase(bc, phase) -> bool:
            if phase is None:
                return True
            tw = bc.time_window
            p_lo, p_hi = phase
            p_mid = 0.5 * (p_lo + p_hi)
            if tw is None:
                return True
            pts_tw = [float(vv) for vv in tw]
            if len(pts_tw) > 2:
                return any(p_lo - tol <= vv <= p_hi + tol for vv in pts_tw)
            a, b = min(pts_tw), max(pts_tw)
            if abs(a - b) < tol:
                return p_lo - tol <= a <= p_hi + tol
            return a - tol <= p_mid <= b + tol

        n_panels = len(phases)
        panel_w, panel_h = 6.0, 5.5
        if figsize is None:
            _has_legend = (region is not None or boundary is not None)
            _legend_extra = 2.0 if _has_legend else 0.0
            figsize = (panel_w * n_panels + _legend_extra, panel_h)

        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = axes[0]   # (n_panels,)

        v, f = self._vertices, self._faces

        for ax, phase in zip(axes, phases):
            # Background: mesh triangulation
            self._draw_background(ax, show_mesh)

            # ── Highlight inner regions ────────────────────────────────────
            if _inner_partition and self._partition_groups is not None:
                # One colour per partition group
                _part_palette = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd',
                                  '#8c564b', '#e377c2', '#bcbd22', '#17becf',
                                  '#ff7f0e', '#7f7f7f']
                for gi, (lbl, grp) in enumerate(self._partition_groups.items()):
                    fidx  = grp['face_indices']
                    color = _part_palette[gi % len(_part_palette)]
                    tris  = v[f[fidx]]
                    poly  = mcoll.PolyCollection(
                        tris, facecolor=color, edgecolor='none', alpha=0.40,
                        zorder=2, label=f'[P] group {lbl}')
                    ax.add_collection(poly)
            elif _inner_all:
                # Fill all triangles as one unified region
                tris = v[f]
                poly = mcoll.PolyCollection(
                    tris, facecolor='#1f77b4', edgecolor='none', alpha=0.30,
                    zorder=2, label='interior')
                ax.add_collection(poly)
            else:
                for nm in _inner_highlight:
                    if nm not in self._inner_regions:
                        continue
                    reg    = self._inner_regions[nm]
                    fidx   = reg['face_indices']
                    color  = _inner_color[nm]
                    tris = v[f[fidx]]           # (n_sel, 3, 2)
                    poly = mcoll.PolyCollection(
                        tris, facecolor=color, edgecolor='none', alpha=0.35,
                        zorder=2, label=f'[I] {nm}')
                    ax.add_collection(poly)

            # ── Highlight boundary regions ─────────────────────────────────
            if _bnd_outer and self._bnd_edges is not None and len(self._bnd_edges) > 0:
                all_edges = self._bnd_edges
                segs = np.stack([v[all_edges[:, 0]], v[all_edges[:, 1]]], axis=1)
                lcol = mcoll.LineCollection(
                    segs, colors='#ff7f0e', linewidths=3.0, zorder=4,
                    label='outer boundary')
                ax.add_collection(lcol)
            elif _bnd_all and not _bnd_outer:
                # legacy 'all' without partition: draw all boundary edges
                all_edges = self._bnd_edges
                segs = np.stack([v[all_edges[:, 0]], v[all_edges[:, 1]]], axis=1)
                lcol = mcoll.LineCollection(
                    segs, colors='#ff7f0e', linewidths=3.0, zorder=4,
                    label='boundary')
                ax.add_collection(lcol)
            if _bnd_inner and self._partition_interface_edges is not None and len(self._partition_interface_edges) > 0:
                ie   = self._partition_interface_edges
                segs = np.stack([v[ie[:, 0]], v[ie[:, 1]]], axis=1)
                lcol = mcoll.LineCollection(
                    segs, colors='#d62728', linewidths=2.0, linestyles='--',
                    zorder=5, label='inner boundary')
                ax.add_collection(lcol)
            if not _bnd_outer and not _bnd_inner:
                for nm in _bnd_highlight:
                    if nm not in self._boundary_regions:
                        continue
                    reg   = self._boundary_regions[nm]
                    edges = reg['edges']      # (n_e, 2)
                    color = _bnd_color[nm]
                    segs  = np.stack([v[edges[:, 0]], v[edges[:, 1]]], axis=1)
                    lcol  = mcoll.LineCollection(
                        segs, colors=color, linewidths=3.0, zorder=4,
                        label=f'[B] {nm}')
                    ax.add_collection(lcol)

            # ── Boundary conditions ────────────────────────────────────────
            if boundary is not None:
                for bc in mesh_bcs:
                    if not _bc_in_phase(bc, phase):
                        continue
                    color = bc_color[id(bc)]
                    label = self._bc_label(
                        bc, list(bc_color.keys()).index(id(bc)))
                    self._draw_bc_on_ax(ax, bc, color, label, node_size)

            ax.set_aspect("equal")
            ax.autoscale_view()
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            if phase is None:
                ax.set_title("spatial", fontsize=10)
            else:
                p_lo, p_hi = phase
                if abs(p_lo - p_hi) < tol:
                    ax.set_title(f"t = {p_lo:.4g}", fontsize=10)
                else:
                    ax.set_title(f"t ∈ [{p_lo:.4g}, {p_hi:.4g}]", fontsize=10)

            handles, labels = ax.get_legend_handles_labels()
            visible = [(h, l) for h, l in zip(handles, labels)
                       if not l.startswith("_")]
            if visible:
                ax.legend(*zip(*visible), loc='upper left',
                          bbox_to_anchor=(1.01, 1), borderaxespad=0,
                          fontsize=8, framealpha=0.85)

        t_info = (f" × t∈[{self._t_min:.3g}, {self._t_max:.3g}]"
                  if self._t_min is not None else "")
        n_inner = len(self._inner_regions)
        n_bnd   = len(self._boundary_regions)
        region_info = (
            f", {n_inner} inner region{'s' if n_inner != 1 else ''}"
            f" + {n_bnd} boundary region{'s' if n_bnd != 1 else ''}"
            if n_inner or n_bnd else ""
        )
        fig.suptitle(
            f"{type(self).__name__}  |  {len(v)} nodes, {len(f)} triangles"
            f"  [{self._spatial_dims}D{t_info}]{region_info}",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()

        if points is not None:
            _pts = np.asarray(points)
            for ax in axes:
                ax.scatter(_pts[:, 0], _pts[:, 1],
                           s=8, color='tomato', alpha=0.5, zorder=5,
                           label='_points')
        return axes[0] if n_panels == 1 else axes

    def _plot_pyvista(
        self,
        show_mesh: bool = True,
        node_size: float = 30.0,
    ):
        """
        Interactive PyVista visualisation of the 2-D mesh.

        The triangulation is rendered as a flat surface (z = 0).  Any
        registered boundary conditions are shown as highlighted point clouds.
        """
        import pyvista as pv

        v = self._vertices          # (N, 2)
        f = self._faces             # (M, 3)  0-indexed triangles

        # Build pyvista PolyData: vertices as (N, 3) with z = 0
        verts_3d = np.column_stack([v, np.zeros(len(v), dtype=np.float32)])
        # pyvista face format: [3, i0, i1, i2, ...]
        faces_pv = np.column_stack(
            [np.full(len(f), 3, dtype=np.int_), f]
        ).ravel()
        mesh = pv.PolyData(verts_3d, faces_pv)

        pl = pv.Plotter(notebook=self._is_jupyter(), off_screen=True)

        edge_color = '#888888'
        face_color = '#e8e8e8'
        pl.add_mesh(mesh, color=face_color,
                    show_edges=show_mesh, edge_color=edge_color,
                    opacity=0.85, label='mesh')

        # BC point clouds (if any are registered on this domain)
        _colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                   '#911eb4', '#42d4f4', '#f032e6', '#bfef45']
        mesh_bcs = [bc for bc in self.boundary_conditions
                    if hasattr(bc, 'node_positions')]
        for i, bc in enumerate(mesh_bcs):
            color = _colors[i % len(_colors)]
            if bc.node_indices is not None:
                pts2 = v[bc.node_indices]
            else:
                pts2 = bc.node_positions
            pts3 = np.column_stack(
                [pts2, np.zeros(len(pts2), dtype=np.float32)]
            )
            cloud = pv.PolyData(pts3)
            label = bc.name or f'bc_{i}'
            pl.add_mesh(cloud, render_points_as_spheres=True,
                        point_size=node_size, color=color, label=label)

        pl.view_xy()
        pl.add_axes()
        if mesh_bcs:
            pl.add_legend(bcolor='white', border=True)
        return pl.show(jupyter_backend='trame' if self._is_jupyter() else None)

    def __repr__(self):
        n_bcs  = len(self.boundary_conditions)
        n_in   = len(self._inner_regions)
        n_bnd  = len(self._boundary_regions)
        sp = f"{self._spatial_dims}D"
        if self._time_mode == 'continuous':
            t_info = f" × t∈[{self._t_min}, {self._t_max}] continuous"
        elif self._time_mode == 'discrete':
            t_info = f" × t∈[{self._t_min}, {self._t_max}] discrete({self.n_steps} steps)"
        else:
            t_info = ""
        region_info = ""
        if n_in or n_bnd:
            region_info = (
                f", inner_regions={list(self._inner_regions.keys())}"
                f", boundary_regions={list(self._boundary_regions.keys())}"
            )
        return (f"DomainMesh({sp}{t_info}, "
                f"n_nodes={len(self._vertices)}, n_conditions={n_bcs}"
                f"{region_info})")


