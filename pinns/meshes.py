"""
pinns.meshes
============

Ready-to-use mesh geometries for ``DomainMesh``.  Every function returns

* ``verts`` — ``np.ndarray`` of shape ``(N, d)``, vertex coordinates
* ``faces`` — ``np.ndarray`` of shape ``(F, 3)``, triangle connectivity (int64)

2-D surface meshes (``d=2``) are generated with **pygmsh** when needed;
3-D surface meshes (``d=3``) are built with pure NumPy (no external dep).
Remote meshes are downloaded once and cached in ``~/.cache/pinns/meshes/``.

Available meshes
----------------
2-D (flat, vertices in R²):

* :func:`square`    — rectangle ``[0, x_max] × [0, y_max]``
* :func:`disk`      — filled circular disk
* :func:`annulus`   — annular ring
* :func:`l_shape`   — L-shaped domain
* :func:`u_shape`   — U-shaped / horseshoe domain

3-D surface (vertices in R³, triangulated surface):

* :func:`circle`        — 1-D circle boundary embedded as a flat annulus ring
* :func:`sphere`        — closed unit sphere surface (icosphere, no pygmsh)
* :func:`open_sphere`   — hemisphere (open surface with equatorial rim)
* :func:`cylinder`      — open cylindrical surface (lateral + two caps, pygmsh)

Remote public meshes (downloaded & cached):

* :func:`get_mesh` — download any mesh by name from the
  ``alecjacobson/common-3d-test-models`` repository.
  The full catalogue with descriptions and citations is in the function
  docstring.
"""

from __future__ import annotations

import os
import pathlib
import urllib.request
import numpy as np

__all__ = [
    # 2-D
    "u_shape",
    "horse_shoe",
    "square",
    "disk",
    "annulus",
    "l_shape",
    # 3-D surface
    "circle",
    "sphere",
    "open_sphere",
    "cylinder",
    # remote
    "get_mesh",
]


def _reindex(verts: np.ndarray, faces: np.ndarray):
    """Drop unused vertices and make the index space contiguous."""
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def _default_cache() -> pathlib.Path:
    base = pathlib.Path(os.environ.get("XDG_CACHE_HOME",
                                       pathlib.Path.home() / ".cache"))
    d = base / "pinns" / "meshes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_remote_mesh(url: str, cache_name: str,
                      cache_dir=None) -> tuple[np.ndarray, np.ndarray]:
    """Download *url* once, cache to disk, return ``(verts, faces)``.

    Uses **trimesh** when available for robust OBJ/STL/PLY parsing;
    falls back to a built-in minimal OBJ reader for plain text OBJ files.
    """
    cache_path = (pathlib.Path(cache_dir) if cache_dir else _default_cache()) / cache_name
    if not cache_path.exists():
        print(f"[pinns.meshes] Downloading {cache_name} …")
        try:
            urllib.request.urlretrieve(url, cache_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {cache_name} from:\n  {url}\n"
                f"Error: {exc}\n"
                "You can download the file manually and pass its path via "
                "the cache_dir argument or place it at:\n"
                f"  {cache_path}"
            ) from exc

    try:
        import trimesh
        mesh = trimesh.load(str(cache_path), force="mesh", process=False)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return verts, faces
    except ImportError:
        pass

    # Minimal OBJ fallback
    if str(cache_path).lower().endswith(".obj"):
        verts_list, faces_list = [], []
        with open(cache_path) as fh:
            for line in fh:
                tok = line.split()
                if not tok:
                    continue
                if tok[0] == "v":
                    verts_list.append([float(tok[1]), float(tok[2]), float(tok[3])])
                elif tok[0] == "f":
                    # Handle v, v/vt, v/vt/vn, v//vn
                    idx = [int(x.split("/")[0]) for x in tok[1:]]
                    if len(idx) == 3:
                        faces_list.append(idx)
                    elif len(idx) == 4:   # quad → two triangles
                        faces_list.append([idx[0], idx[1], idx[2]])
                        faces_list.append([idx[0], idx[2], idx[3]])
        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(faces_list, dtype=np.int64)
        # OBJ is 1-indexed
        if len(faces) > 0 and faces.min() == 1:
            faces -= 1
        return _reindex(verts, faces)

    raise RuntimeError(
        f"Cannot parse {cache_path}: install trimesh for full OBJ/STL/PLY support.\n"
        "  pip install trimesh"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal pygmsh helper
# ──────────────────────────────────────────────────────────────────────────────

def _gmsh_generate(geom, random_factor: float = 0.1, dim: int = 2):
    """Call ``geom.generate_mesh`` with optional node jitter.

    A non-zero ``Mesh.RandomFactor`` breaks the strict symmetry of the
    triangulation, producing more irregular, natural-looking meshes.
    Algorithm 5 (Delaunay) is used for consistent quality.
    """
    try:
        import gmsh
        gmsh.option.setNumber("Mesh.RandomFactor", random_factor)
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # 5 = Delaunay
    except Exception:
        pass
    return geom.generate_mesh(dim=dim, verbose=False)


# ──────────────────────────────────────────────────────────────────────────────
# U-shape
# ──────────────────────────────────────────────────────────────────────────────

def u_shape(
    x_max: float = 2.0,
    y_max: float = 2.0,
    notch_x: tuple[float, float] = (0.6, 1.4),
    notch_y: float = 0.8,
    mesh_size: float = 0.12,
    random_factor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """U-shaped (horseshoe) domain.

    A rectangle ``[0, x_max] × [0, y_max]`` with a rectangular notch cut from
    the top centre: ``(notch_x[0], notch_x[1]) × (notch_y, y_max)``.

    Parameters
    ----------
    x_max, y_max : float
        Outer bounding-box dimensions.
    notch_x : (float, float)
        Horizontal extent of the notch (left x, right x).
    notch_y : float
        Vertical start of the notch (notch extends upward to ``y_max``).
    mesh_size : float
        Target element size passed to gmsh.

    Returns
    -------
    verts : ndarray, shape (N, 2)
    faces : ndarray, shape (F, 3), dtype int64
    """
    try:
        import pygmsh
    except ImportError as e:
        raise ImportError(
            "pygmsh is required for mesh generation: pip install pygmsh"
        ) from e

    polygon = [
        [0.0,       0.0    ],
        [x_max,     0.0    ],
        [x_max,     y_max  ],
        [notch_x[1],y_max  ],
        [notch_x[1],notch_y],
        [notch_x[0],notch_y],
        [notch_x[0],y_max  ],
        [0.0,       y_max  ],
    ]

    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(polygon, mesh_size=mesh_size)
        mesh = _gmsh_generate(geom, random_factor)

    verts = mesh.points[:, :2].copy()
    faces = mesh.cells_dict["triangle"].astype(np.int64)
    return _reindex(verts, faces)

# ──────────────────────────────────────────────────────────────────────────────
# U-shape
# ──────────────────────────────────────────────────────────────────────────────

def trident(
    x_max: float = 1.0,
    y_max: float = 1.0,
    mesh_size: float = 0.03,
    random_factor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Trident.

    A rectangle ``[0, x_max] × [0, y_max]`` with a rectangular notch cut from
    the top centre: ``(notch_x[0], notch_x[1]) × (notch_y, y_max)``.

    Parameters
    ----------
    x_max, y_max : float
        Outer bounding-box dimensions.
    notch_x : (float, float)
        Horizontal extent of the notch (left x, right x).
    notch_y : float
        Vertical start of the notch (notch extends upward to ``y_max``).
    mesh_size : float
        Target element size passed to gmsh.

    Returns
    -------
    verts : ndarray, shape (N, 2)
    faces : ndarray, shape (F, 3), dtype int64
    """
    try:
        import pygmsh
    except ImportError as e:
        raise ImportError(
            "pygmsh is required for mesh generation: pip install pygmsh"
        ) from e

    polygon = [
        [0.0,        0.0    ],
        [x_max,      0.0    ],
        [x_max,      y_max  ],
        [x_max*10/12,y_max  ],
        [x_max*10/12,y_max*3/12],
        [x_max*9/12,y_max*3/12],
        [x_max*9/12,y_max],
        [x_max*8/12,y_max],
        [x_max*8/12,y_max*3/12],
        [x_max*7/12,y_max*3/12],
        [x_max*7/12,y_max],
        [x_max*5/12,y_max],
        [x_max*5/12,y_max*3/12],
        [x_max*4/12,y_max*3/12],
        [x_max*4/12,y_max],
        [x_max*3/12,y_max],
        [x_max*3/12,y_max*3/12],
        [x_max*2/12,y_max*3/12],
        [x_max*2/12,y_max],
        [0.0,       y_max  ],
    ]

    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(polygon, mesh_size=mesh_size)
        mesh = _gmsh_generate(geom, random_factor)

    verts = mesh.points[:, :2].copy()
    faces = mesh.cells_dict["triangle"].astype(np.int64)
    return _reindex(verts, faces)


def horse_shoe(
    width: float = 2.0,
    arm_width: float = 0.5,
    arm_height: float = 1.0,
    mesh_size: float = 0.12,
    random_factor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Horseshoe (magnet) domain: two concentric semicircles joined by straight arms.

    The bottom is a true semicircle (not rounded corners on a rectangle).
    The two arms have flat open tops.

    Cross-section schematic (opens upward)::

         ____         ____
        |    |       |    |   ← flat arm tips (open top)
        |    |       |    |
        |    |_______|    |
        |                 |
        |    inner semi   |   arm_width
        |   (         )   |
         \\   \\_____/   /
          \\             /
           \\___________/
                outer semicircle

    The coordinate origin is at the bottom of the outer semicircle.
    The shape spans ``x ∈ [0, width]``, ``y ∈ [0, outer_radius + arm_height]``.

    Parameters
    ----------
    width : float
        Total outer width (= 2 × outer_radius).
    arm_width : float
        Thickness of each arm.  Must satisfy ``arm_width < width / 2``.
    arm_height : float
        Straight portion height above the semicircle centre.
    mesh_size : float
        Target element size.
    random_factor : float
        Gmsh random perturbation (0 = regular, 0.1 = natural).

    Returns
    -------
    verts : ndarray, shape (N, 2)
    faces : ndarray, shape (F, 3), dtype int64
    """
    try:
        import pygmsh
    except ImportError as e:
        raise ImportError(
            "pygmsh is required for mesh generation: pip install pygmsh"
        ) from e

    R = width / 2.0          # outer radius
    r = R - arm_width        # inner radius
    cy = R                   # y of the semicircle centre (so bottom is at y=0)
    cx = R                   # x of the centre

    if arm_width <= 0 or arm_width >= R:
        raise ValueError("arm_width must satisfy 0 < arm_width < width/2")
    if arm_height <= 0:
        raise ValueError("arm_height must be positive")

    with pygmsh.geo.Geometry() as geom:
        ms = mesh_size

        def P(x, y):
            return geom.add_point([x, y, 0.0], mesh_size=ms)

        # ── Key points ───────────────────────────────────────────────────────
        # Outer arm tips (flat open top)
        p_tl  = P(0,      cy + arm_height)   # top-left  outer
        p_tr  = P(2*R,    cy + arm_height)   # top-right outer

        # Tangent points where arms meet the semicircles (at y = cy)
        p_ol  = P(0,      cy)                # outer left  tangent
        p_or  = P(2*R,    cy)                # outer right tangent
        p_ir  = P(cx + r, cy)                # inner right tangent
        p_il  = P(cx - r, cy)                # inner left  tangent

        # Midpoints split each 180° semicircle into two 90° arcs
        p_out_mid = P(cx,  cy - R)           # bottom of outer arc
        p_inn_mid = P(cx,  cy - r)           # bottom of inner arc

        # Arc centres (not meshed, no mesh_size needed but pygmsh needs a point)
        c = P(cx, cy)                        # shared centre

        # Notch (channel) top corners
        p_tr_notch = P(cx + r, cy + arm_height)
        p_tl_notch = P(cx - r, cy + arm_height)

        # ── Curves (CCW boundary) ─────────────────────────────────────────────
        curves = [
            # left arm: down from tip to outer tangent
            geom.add_line(p_tl,  p_ol),
            # outer semicircle (CW in std coords = correct CCW domain boundary)
            # split into two 90° arcs: left→bottom→right
            geom.add_circle_arc(p_ol,      c, p_out_mid),
            geom.add_circle_arc(p_out_mid, c, p_or),
            # right arm: up from outer tangent to tip
            geom.add_line(p_or,  p_tr),
            # right flat tip: inward from outer to notch
            geom.add_line(p_tr,  p_tr_notch),
            # right notch wall: down to inner tangent
            geom.add_line(p_tr_notch, p_ir),
            # inner semicircle (also CW, right→bottom→left — traces notch bottom)
            geom.add_circle_arc(p_ir,      c, p_inn_mid),
            geom.add_circle_arc(p_inn_mid, c, p_il),
            # left notch wall: up from inner tangent to notch tip
            geom.add_line(p_il, p_tl_notch),
            # left flat tip: outward back to outer tip
            geom.add_line(p_tl_notch, p_tl),
        ]

        cl = geom.add_curve_loop(curves)
        geom.add_plane_surface(cl)
        mesh = _gmsh_generate(geom, random_factor)

    verts = mesh.points[:, :2].copy()
    faces = mesh.cells_dict["triangle"].astype(np.int64)
    return _reindex(verts, faces)


# ──────────────────────────────────────────────────────────────────────────────
# Square / rectangle
# ──────────────────────────────────────────────────────────────────────────────

def square(
    x_max: float = 1.0,
    y_max: float = 1.0,
    mesh_size: float = 0.08,
    random_factor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular (or square) domain ``[0, x_max] × [0, y_max]``.

    Parameters
    ----------
    x_max, y_max : float
        Domain extents.
    mesh_size : float
        Target element size.

    Returns
    -------
    verts : ndarray, shape (N, 2)
    faces : ndarray, shape (F, 3), dtype int64
    """
    try:
        import pygmsh
    except ImportError as e:
        raise ImportError(
            "pygmsh is required for mesh generation: pip install pygmsh"
        ) from e

    polygon = [
        [0.0,   0.0  ],
        [x_max, 0.0  ],
        [x_max, y_max],
        [0.0,   y_max],
    ]

    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(polygon, mesh_size=mesh_size)
        mesh = _gmsh_generate(geom, random_factor)

    verts = mesh.points[:, :2].copy()
    faces = mesh.cells_dict["triangle"].astype(np.int64)
    return _reindex(verts, faces)


# ──────────────────────────────────────────────────────────────────────────────
# Disk
# ──────────────────────────────────────────────────────────────────────────────

def disk(
    radius: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    mesh_size: float = 0.08,
    random_factor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Circular disk of the given ``radius`` centred at ``center``.

    Parameters
    ----------
    radius : float
    center : (float, float)
    mesh_size : float
        Target element size.

    Returns
    -------
    verts : ndarray, shape (N, 2)
    faces : ndarray, shape (F, 3), dtype int64
    """
    try:
        import pygmsh
    except ImportError as e:
        raise ImportError(
            "pygmsh is required for mesh generation: pip install pygmsh"
        ) from e

    with pygmsh.occ.Geometry() as geom:
        geom.characteristic_length_max = mesh_size
        geom.add_disk(center, radius)
        mesh = _gmsh_generate(geom, random_factor)

    verts = mesh.points[:, :2].copy()
    faces = mesh.cells_dict["triangle"].astype(np.int64)
    return _reindex(verts, faces)


# ──────────────────────────────────────────────────────────────────────────────
# Annulus
# ──────────────────────────────────────────────────────────────────────────────

def annulus(
    inner_radius: float = 0.4,
    outer_radius: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    mesh_size: float = 0.08,
    random_factor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Annular ring: disk of ``outer_radius`` minus disk of ``inner_radius``.

    Parameters
    ----------
    inner_radius, outer_radius : float
    center : (float, float)
    mesh_size : float

    Returns
    -------
    verts : ndarray, shape (N, 2)
    faces : ndarray, shape (F, 3), dtype int64
    """
    try:
        import pygmsh
    except ImportError as e:
        raise ImportError(
            "pygmsh is required for mesh generation: pip install pygmsh"
        ) from e

    with pygmsh.occ.Geometry() as geom:
        outer = geom.add_disk(center, outer_radius)
        inner = geom.add_disk(center, inner_radius)
        geom.boolean_difference(outer, inner)
        geom.set_mesh_size_callback(lambda dim, tag, x, y, z, lc: mesh_size)
        mesh = _gmsh_generate(geom, random_factor)

    verts = mesh.points[:, :2].copy()
    faces = mesh.cells_dict["triangle"].astype(np.int64)
    return _reindex(verts, faces)


# ──────────────────────────────────────────────────────────────────────────────
# L-shape
# ──────────────────────────────────────────────────────────────────────────────

def l_shape(
    size: float = 1.0,
    mesh_size: float = 0.08,
    random_factor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """L-shaped domain: unit square minus the top-right quadrant.

    The domain covers ``[0, size] × [0, size]`` with the quadrant
    ``(size/2, size) × (size/2, size)`` removed.

    Parameters
    ----------
    size : float
        Side length of the enclosing square.
    mesh_size : float

    Returns
    -------
    verts : ndarray, shape (N, 2)
    faces : ndarray, shape (F, 3), dtype int64
    """
    try:
        import pygmsh
    except ImportError as e:
        raise ImportError(
            "pygmsh is required for mesh generation: pip install pygmsh"
        ) from e

    h = size / 2.0
    polygon = [
        [0.0,  0.0 ],
        [size, 0.0 ],
        [size, h   ],
        [h,    h   ],
        [h,    size],
        [0.0,  size],
    ]

    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(polygon, mesh_size=mesh_size)
        mesh = _gmsh_generate(geom, random_factor)

    verts = mesh.points[:, :2].copy()
    faces = mesh.cells_dict["triangle"].astype(np.int64)
    return _reindex(verts, faces)


# ──────────────────────────────────────────────────────────────────────────────
# 3-D surface: Circle (thin flat annulus)
# ──────────────────────────────────────────────────────────────────────────────

def circle(
    radius: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    mesh_size: float = 0.08,
    thickness: float = 0.02,
    random_factor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin annular region approximating the circle boundary ``|x| = radius``.

    Produces a 2-D mesh of the narrow ring
    ``(radius - thickness/2, radius + thickness/2)``, useful when the PDE
    lives on a circular curve rather than the disk interior.

    Requires **pygmsh**.

    Parameters
    ----------
    radius : float
        Circle radius.
    center : (float, float)
        Centre coordinates.
    mesh_size : float
        Target element size.
    thickness : float
        Radial width of the ring (default 0.02).

    Returns
    -------
    verts : ndarray, shape (N, 2)
    faces : ndarray, shape (F, 3), dtype int64
    """
    return annulus(
        inner_radius=radius - thickness / 2.0,
        outer_radius=radius + thickness / 2.0,
        center=center,
        mesh_size=mesh_size,
        random_factor=random_factor,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3-D surface: Sphere / hemisphere (pure NumPy — no external dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _icosphere(subdivisions: int, radius: float,
               center: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Build an icosphere via midpoint subdivision."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=float)
    raw /= np.linalg.norm(raw[0])
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int64)

    verts_list = list(raw)
    for _ in range(subdivisions):
        mid_cache: dict = {}
        snap = list(verts_list)

        def _mid(a: int, b: int, _snap=snap, _vl=verts_list,
                 _mc=mid_cache) -> int:
            key = (min(a, b), max(a, b))
            if key not in _mc:
                m = (_snap[a] + _snap[b]) / 2.0
                m = m / np.linalg.norm(m)
                _mc[key] = len(_vl)
                _vl.append(m)
            return _mc[key]

        new_faces: list = []
        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = np.array(new_faces, dtype=np.int64)

    cx, cy, cz = center
    v = np.array(verts_list, dtype=np.float64) * radius
    v += np.array([cx, cy, cz])
    return v, faces


def sphere(
    radius: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    subdivisions: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed unit-sphere surface mesh (icosphere).

    Vertices lie exactly on the sphere surface in R³.  No external dependency
    is required.  At ``subdivisions=3`` the mesh has 642 vertices and
    1280 faces; each extra subdivision quadruples the face count.

    Parameters
    ----------
    radius : float
        Sphere radius.
    center : (float, float, float)
        Sphere centre.
    subdivisions : int
        Icosphere refinement level (default 3).

    Returns
    -------
    verts : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3), dtype int64

    Notes
    -----
    A closed sphere has no boundary edges — ``DomainMesh._bnd_edges`` will be
    empty.  Use :func:`open_sphere` if you need a mesh with a rim boundary.
    """
    return _icosphere(subdivisions, radius, center)


def open_sphere(
    radius: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    subdivisions: int = 3,
    cut_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Hemisphere (open sphere surface) with a circular rim boundary.

    All faces whose centroid lies below ``center_z + cut_z`` are removed,
    leaving an open surface whose rim is a circle at that height.

    Parameters
    ----------
    radius : float
    center : (float, float, float)
    subdivisions : int
    cut_z : float
        Relative to centre.  Faces with centroid ``z - center_z < cut_z`` are
        discarded (default 0 → upper hemisphere).

    Returns
    -------
    verts : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3), dtype int64
    """
    verts, faces = _icosphere(subdivisions, radius, center)
    cz = center[2]
    centroids_z = verts[faces].mean(axis=1)[:, 2]
    keep = centroids_z - cz >= cut_z
    return _reindex(verts, faces[keep])


# ──────────────────────────────────────────────────────────────────────────────
# 3-D surface: Cylinder (pygmsh)
# ──────────────────────────────────────────────────────────────────────────────

def cylinder(
    radius: float = 1.0,
    height: float = 2.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mesh_size: float = 0.15,
    with_caps: bool = True,
    random_factor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Cylindrical surface mesh (lateral surface + optional end caps).

    Requires **pygmsh**.

    Parameters
    ----------
    radius : float
        Cylinder radius.
    height : float
        Cylinder height along the z-axis.
    center : (float, float, float)
        Centre of the bottom disk.
    mesh_size : float
        Target element size.
    with_caps : bool
        When ``True`` (default) both end disks are included (watertight closed
        surface).  When ``False`` only the lateral surface is kept (open
        cylinder, two circular rim boundaries).

    Returns
    -------
    verts : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3), dtype int64
    """
    try:
        import pygmsh
    except ImportError as e:
        raise ImportError(
            "pygmsh is required for cylinder generation: pip install pygmsh"
        ) from e

    cx, cy, cz = center
    with pygmsh.occ.Geometry() as geom:
        geom.add_cylinder(
            x0=[cx, cy, cz],
            axis=[0.0, 0.0, float(height)],
            radius=float(radius),
        )
        geom.set_mesh_size_callback(lambda dim, tag, x, y, z, lc: mesh_size)
        mesh = _gmsh_generate(geom, random_factor)

    verts = np.asarray(mesh.points, dtype=np.float64)
    tris = mesh.cells_dict.get("triangle")
    if tris is None:
        raise RuntimeError("pygmsh did not produce triangle cells for cylinder.")
    faces = np.asarray(tris, dtype=np.int64)

    if not with_caps:
        # Keep only lateral-surface triangles (whose normal has small z-component)
        e1 = verts[faces[:, 1]] - verts[faces[:, 0]]
        e2 = verts[faces[:, 2]] - verts[faces[:, 0]]
        normals = np.cross(e1, e2)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-15
        faces = faces[np.abs(normals[:, 2]) < 0.5]

    return _reindex(verts, faces)


# ──────────────────────────────────────────────────────────────────────────────
# Remote public meshes — generic catalogue + get_mesh()
# ──────────────────────────────────────────────────────────────────────────────

_JACOBSON_BASE = (
    "https://raw.githubusercontent.com/"
    "alecjacobson/common-3d-test-models/master/data/"
)

# Catalogue: name → (filename, approx_size, description, citation)
_MESH_CATALOGUE: dict[str, tuple[str, str, str, str]] = {
    # name              file                     size    description                           citation
    "armadillo":    ("armadillo.obj",         "~60 MB", "Stanford armadillo scan",
                     "Stanford 3D Scanning Repository, Krishnamurthy & Levoy 1996"),
    "beast":        ("beast.obj",             "~10 MB", "Autodesk beast character",
                     "Autodesk; Weber et al. 2007 (CGF)"),
    "beetle":       ("beetle.obj",            "~5 MB",  "Ivan Sutherland 1972 mesh",
                     "Ivan Sutherland 1972"),
    "cheburashka":  ("cheburashka.obj",        "~5 MB",  "Cheburashka character",
                     "Ilya Baran; Baran & Popović 2007 (SIGGRAPH)"),
    "cow":          ("cow.obj",               "~1 MB",  "Classic cow model",
                     "Viewpoint Animation Engineering / Sun Microsystems; "
                     "DeCarlo et al. 2003 (SIGGRAPH)"),
    "fandisk":      ("fandisk.obj",           "~2 MB",  "CAD fandisk (Pratt & Whitney)",
                     "Pratt & Whitney / Hughes Hoppe; Hoppe et al. 1994 (SIGGRAPH)"),
    "happy":        ("happy.obj",             "~60 MB", "Stanford Happy Buddha scan",
                     "Stanford 3D Scanning Repository, Curless & Levoy 1996 (SIGGRAPH)"),
    "horse":        ("horse.obj",             "~10 MB", "CyberWare horse scan",
                     "CyberWare; Georgia Tech Large Models Archive"),
    "max-planck":   ("max-planck.obj",        "~10 MB", "Max Planck Institute bust",
                     "MPI for Intelligent Systems; Princeton Shape Benchmark"),
    "ogre":         ("ogre.obj",              "~10 MB", "Ogre character",
                     "Kiaran Ritchie; Lipman et al. 2008 (SIGGRAPH)"),
    "rocker-arm":   ("rocker-arm.obj",        "~3 MB",  "INRIA rocker-arm CAD part",
                     "INRIA / VisionAIR repository"),
    "spot":         ("spot.obj",              "~1 MB",  "Keenan Crane spot-the-cow",
                     "Keenan Crane, https://www.cs.cmu.edu/~kmcrane/"),
    "bunny":        ("stanford-bunny.obj",    "~2 MB",  "Stanford Bunny scan",
                     "Stanford 3D Scanning Repository, Turk & Levoy 1994 (SIGGRAPH)"),
    "suzanne":      ("suzanne.obj",           "~1 MB",  "Blender Suzanne monkey head",
                     "Blender Foundation (open source)"),
    "teapot":       ("teapot.obj",            "~1 MB",  "Martin Newell teapot",
                     "Martin Newell 1975 (Utah teapot)"),
    "dragon":       ("xyzrgb_dragon.obj",     "~60 MB", "Stanford XYZ RGB Dragon scan",
                     "Stanford 3D Scanning Repository"),
    "alligator":    ("alligator.obj",         "~10 MB", "Alec Jacobson alligator",
                     "Alec Jacobson; Jacobson et al. 2011 (SIGGRAPH)"),
    "woody":        ("woody.obj",             "~2 MB",  "Woody / gingerbread man",
                     "Scott Schaefer; Schaefer et al. 2006 (SIGGRAPH)"),
    "lucy":         ("lucy.obj",              "~25 MB", "Stanford Lucy angel figurine",
                     "Stanford 3D Scanning Repository"),
    "bimba":        ("bimba.obj",             "~30 MB", "Bimba statue (3.7 M triangles)",
                     "Pierre Alliez & Marco Attene / VisionAIR"),
    "igea":         ("igea.obj",              "~10 MB", "Cyberware Igea head scan",
                     "Cyberware"),
    "homer":        ("homer.obj",             "~5 MB",  "Homer Simpson bust",
                     "Source unclear; distributed by Pierre Alliez"),
}


def get_mesh(
    name: str,
    cache_dir=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Download and return any mesh from the public catalogue by name.

    Meshes are fetched once from the
    ``alecjacobson/common-3d-test-models`` GitHub mirror and cached
    locally in ``~/.cache/pinns/meshes/``.

    Parameters
    ----------
    name : str
        Mesh name (case-insensitive).  See the catalogue table below.
    cache_dir : path-like or None
        Override the local cache directory.
        Defaults to ``~/.cache/pinns/meshes/``.

    Returns
    -------
    verts : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3), dtype int64

    Catalogue
    ---------
    All meshes are distributed from:
    https://github.com/alecjacobson/common-3d-test-models

    ================  ========  =================================  ===================================================
    name              size      description                        citation
    ================  ========  =================================  ===================================================
    alligator         ~10 MB    Alec Jacobson alligator            Jacobson et al. 2011 (SIGGRAPH)
    armadillo         ~60 MB    Stanford armadillo scan            Stanford 3D Scanning Repository, Krishnamurthy & Levoy 1996 (SIGGRAPH)
    beast             ~10 MB    Autodesk beast character           Autodesk; Weber et al. 2007 (CGF)
    beetle             ~5 MB    Ivan Sutherland 1972 mesh          Ivan Sutherland 1972
    bimba             ~30 MB    Bimba statue (3.7 M triangles)     Pierre Alliez & Marco Attene / VisionAIR
    bunny              ~2 MB    Stanford Bunny scan                Stanford 3D Scanning Repository, Turk & Levoy 1994 (SIGGRAPH)
    cheburashka        ~5 MB    Cheburashka character              Ilya Baran; Baran & Popović 2007 (SIGGRAPH)
    cow                ~1 MB    Classic cow model                  Viewpoint Animation Engineering / Sun Microsystems; DeCarlo et al. 2003 (SIGGRAPH)
    dragon            ~60 MB    Stanford XYZ RGB Dragon scan       Stanford 3D Scanning Repository
    fandisk            ~2 MB    CAD fandisk (Pratt & Whitney)      Pratt & Whitney / Hughes Hoppe; Hoppe et al. 1994 (SIGGRAPH)
    happy             ~60 MB    Stanford Happy Buddha scan         Stanford 3D Scanning Repository, Curless & Levoy 1996 (SIGGRAPH)
    homer              ~5 MB    Homer Simpson bust                 Unknown; distributed by Pierre Alliez
    horse             ~10 MB    CyberWare horse scan               CyberWare; Georgia Tech Large Models Archive
    igea              ~10 MB    Cyberware Igea head scan           Cyberware
    lucy              ~25 MB    Stanford Lucy angel figurine       Stanford 3D Scanning Repository
    max-planck        ~10 MB    Max Planck Institute bust          MPI for Intelligent Systems; Princeton Shape Benchmark
    ogre              ~10 MB    Ogre character                     Kiaran Ritchie; Lipman et al. 2008 (SIGGRAPH)
    rocker-arm         ~3 MB    INRIA rocker-arm CAD part          INRIA / VisionAIR repository
    spot               ~1 MB    Keenan Crane spot-the-cow          Keenan Crane, https://www.cs.cmu.edu/~kmcrane/
    suzanne            ~1 MB    Blender Suzanne monkey head        Blender Foundation (open source)
    teapot             ~1 MB    Martin Newell teapot               Martin Newell 1975 (Utah teapot)
    woody              ~2 MB    Woody / gingerbread man            Scott Schaefer; Schaefer et al. 2006 (SIGGRAPH)
    ================  ========  =================================  ===================================================

    Examples
    --------
    ::

        from pinns import meshes
        import numpy as np

        # Load the Stanford Bunny
        v, f = meshes.get_mesh('bunny')

        # Normalise to [-1, 1]
        v -= v.mean(axis=0)
        v /= np.abs(v).max()
    """
    entry = _MESH_CATALOGUE.get(name.lower())
    if entry is None:
        available = ", ".join(sorted(_MESH_CATALOGUE.keys()))
        raise ValueError(
            f"Unknown mesh '{name}'.  Available names:\n  {available}"
        )

    filename, _size, _description, _citation = entry
    return _load_remote_mesh(_JACOBSON_BASE + filename, filename, cache_dir)
