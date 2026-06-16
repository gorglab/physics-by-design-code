"""
render_utils.py
===============
Lightweight, dependency-free 3D STL renderer for figures.

We do not want to pull in OpenGL/PyVista/Mayavi for a paper figure pipeline.
Instead we use matplotlib's ``Poly3DCollection`` together with a simple
Lambertian shading model based on the dot product between each triangle's
face normal and a single directional light.  The result is a solid 3D
look that reads as a "real" STL render, not a wireframe, while still being
deterministic and offline.

Usage:
    from render_utils import render_mesh
    render_mesh(ax, mesh, light=(0.4, -0.6, 0.8), base_color="#3a7ca5")
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = verts[faces]
    v01 = tris[:, 1] - tris[:, 0]
    v02 = tris[:, 2] - tris[:, 0]
    n = np.cross(v01, v02)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norms, 1e-12)


def _face_depths(verts: np.ndarray, faces: np.ndarray, view) -> np.ndarray:
    """Project face centroids onto view direction for painter's-algorithm sort."""
    view = np.asarray(view, dtype=float)
    view = view / np.linalg.norm(view)
    centroids = verts[faces].mean(axis=1)
    return centroids @ view


def render_mesh(ax, mesh, *,
                base_color: str = "#3a7ca5",
                light=(0.35, -0.55, 0.78),
                ambient: float = 0.30,
                edge_color=None,
                linewidth: float = 0.0,
                alpha: float = 1.0,
                view=(-0.3, -0.7, 0.6)) -> None:
    """Render a trimesh.Trimesh on a 3D matplotlib Axes with Lambertian shading.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
    mesh : trimesh.Trimesh
    base_color : matplotlib colour spec for the lit surface
    light : (3,) direction vector for the directional light
    ambient : ambient term in [0, 1] (1 = flat, no shading)
    edge_color : edge colour, or None for no edges
    linewidth : edge line width
    alpha : face alpha
    view : view direction used for painter's-algorithm sort
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if len(faces) == 0:
        return

    normals = _face_normals(verts, faces)
    L = np.asarray(light, dtype=float)
    L = L / np.linalg.norm(L)
    lambert = np.clip(np.abs(normals @ L), 0.0, 1.0)
    shade = ambient + (1.0 - ambient) * lambert

    base_rgb = np.array(to_rgb(base_color))
    face_rgb = np.clip(shade[:, None] * base_rgb[None, :], 0, 1)
    face_rgba = np.concatenate(
        [face_rgb, np.full((face_rgb.shape[0], 1), alpha)], axis=1,
    )

    order = np.argsort(_face_depths(verts, faces, view))
    tris = verts[faces[order]]
    face_rgba = face_rgba[order]

    pc = Poly3DCollection(tris, linewidths=linewidth, alpha=alpha)
    pc.set_facecolor(face_rgba)
    if edge_color is not None and linewidth > 0:
        pc.set_edgecolor(edge_color)
    else:
        pc.set_edgecolor("none")
    ax.add_collection3d(pc)

    # tidy axes for a clean isometric look
    b_min = verts.min(axis=0); b_max = verts.max(axis=0)
    pad = 0.02 * np.max(b_max - b_min)
    ax.set_xlim(b_min[0] - pad, b_max[0] + pad)
    ax.set_ylim(b_min[1] - pad, b_max[1] + pad)
    ax.set_zlim(b_min[2] - pad, b_max[2] + pad)
    ax.set_box_aspect(tuple((b_max - b_min) / (b_max - b_min).max()))
    ax.set_axis_off()
    # consistent viewpoint
    ax.view_init(elev=22, azim=-58)
