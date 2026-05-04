"""
pinns.meshes
============

Ready-to-use 2-D triangulated mesh geometries for quick testing.

Each function returns ``(verts, faces)`` where

* ``verts`` — ``np.ndarray`` of shape ``(N, 2)``, vertex coordinates (x, y)
* ``faces`` — ``np.ndarray`` of shape ``(F, 3)``, triangle connectivity (int64)

All meshes are generated with `pygmsh` and then re-indexed so that the vertex
index space is contiguous (i.e. ``np.unique(faces) == np.arange(len(verts))``).

Available meshes
----------------
* :func:`u_shape`   — U-shaped / horseshoe domain
* :func:`square`    — simple unit square (or arbitrary rectangle)
* :func:`disk`      — disk / circle
* :func:`annulus`   — annular ring (disk with a circular hole)
* :func:`l_shape`   — L-shaped domain (square minus a corner quadrant)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "u_shape",
    "square",
    "disk",
    "annulus",
    "l_shape",
]


def _reindex(verts: np.ndarray, faces: np.ndarray):
    """Drop unused vertices and make the index space contiguous."""
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


# ──────────────────────────────────────────────────────────────────────────────
# U-shape
# ──────────────────────────────────────────────────────────────────────────────

def u_shape(
    x_max: float = 2.0,
    y_max: float = 2.0,
    notch_x: tuple[float, float] = (0.6, 1.4),
    notch_y: float = 0.8,
    mesh_size: float = 0.12,
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
        mesh = geom.generate_mesh(dim=2, verbose=False)

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
        mesh = geom.generate_mesh(dim=2, verbose=False)

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

    with pygmsh.geo.Geometry() as geom:
        geom.add_disk(center, radius, mesh_size=mesh_size)
        mesh = geom.generate_mesh(dim=2, verbose=False)

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
        mesh = geom.generate_mesh(dim=2, verbose=False)

    verts = mesh.points[:, :2].copy()
    faces = mesh.cells_dict["triangle"].astype(np.int64)
    return _reindex(verts, faces)


# ──────────────────────────────────────────────────────────────────────────────
# L-shape
# ──────────────────────────────────────────────────────────────────────────────

def l_shape(
    size: float = 1.0,
    mesh_size: float = 0.08,
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
        mesh = geom.generate_mesh(dim=2, verbose=False)

    verts = mesh.points[:, :2].copy()
    faces = mesh.cells_dict["triangle"].astype(np.int64)
    return _reindex(verts, faces)
