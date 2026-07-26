"""
schonhardt_particles.py
=======================

Physics by Design: Additive Manufacturing for Reproducible Science
Daniel N. Wilke
Gradient-Only Research Group (GorgLab) for Emerging Engineering Technology (EET)
Mathematical and Computational Applications (MDPI),
Special Issue "Advances in Computational and Applied Mechanics".

Licence: MIT. Free to use, copy, modify and redistribute, with attribution.
Provided "as is", without warranty of any kind, express or implied. The
author accepts no liability for any use of this code or its outputs.
Use at your own risk.

Generate Schoenhardt-twisted triangular prism STL meshes parameterised by a
single scalar twist angle in degrees.  Topology, height and base triangle
edge length are kept constant.  Twist angle = 0 deg recovers the convex
triangular prism; twist = 45 deg is the most non-convex variant studied in
Wilke et al. (2017, doi:10.1051/epjconf/201714006028).

Usage:

    python scripts/schonhardt_particles.py

Writes STL files to ``generated_stl/`` and a figure to
``figures/fig_schonhardt_family.png`` / ``.pdf``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
STL_DIR = ROOT / "generated_stl"
FIG_DIR = ROOT / "figures"
STL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def schonhardt(twist_deg: float, edge: float = 10.0, height: float = 10.0):
    """Build a Schoenhardt-style twisted triangular prism mesh.

    The top triangle is rotated by ``twist_deg`` about the central z axis
    relative to the bottom triangle.  Lateral faces are formed by connecting
    the rotated edges with non-planar quadrilaterals split into two
    triangles, producing the characteristic non-convex re-entrant facets.
    """
    th = np.deg2rad(twist_deg)
    # equilateral triangle vertices in the xy plane, centred on origin
    base_angles = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3])
    r = edge / np.sqrt(3.0)
    bot = np.stack([r * np.cos(base_angles), r * np.sin(base_angles),
                    np.zeros(3)], axis=1)
    top_angles = base_angles + th
    top = np.stack([r * np.cos(top_angles), r * np.sin(top_angles),
                    height * np.ones(3)], axis=1)
    verts = np.vstack([bot, top])  # 0..2 bottom, 3..5 top

    faces = []
    # bottom face (outward normal = -z), so order clockwise viewed from +z
    faces.append([0, 2, 1])
    # top face
    faces.append([3, 4, 5])
    # lateral: each pair of adjacent bottom vertices links to two top vertices
    # producing the Schoenhardt non-convex zig-zag
    for i in range(3):
        b0, b1 = i, (i + 1) % 3
        t0, t1 = 3 + i, 3 + (i + 1) % 3
        # Schoenhardt construction: cross the diagonals to create concavity
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    return mesh


def generate_family(twists=(0, 15, 30, 45)):
    records = []
    for tw in twists:
        m = schonhardt(twist_deg=tw)
        fname = f"schonhardt_twist{tw:02d}.stl"
        m.export(STL_DIR / fname)
        # Hull volume vs mesh volume gives a single-scalar non-convexity proxy
        try:
            hull = m.convex_hull
            nci = 1.0 - m.volume / hull.volume if hull.volume > 0 else 0.0
        except Exception:
            nci = float("nan")
        records.append((tw, float(m.volume), float(nci), fname))
        print(f"  twist={tw:2d}  V={m.volume:.2f} mm^3  NCI={nci:.3f}  -> {fname}")
    return records


def render_family(twists=(0, 15, 30, 45)):
    fig = plt.figure(figsize=(11, 3.2))
    for i, tw in enumerate(twists):
        m = schonhardt(twist_deg=tw)
        ax = fig.add_subplot(1, len(twists), i + 1, projection="3d")
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        tris = m.vertices[m.faces]
        pc = Poly3DCollection(tris, alpha=0.85, linewidth=0.4,
                              edgecolor="#222222")
        cmap = {0: "#7fb069", 15: "#9d7ab8", 30: "#e09f3e", 45: "#c1272d"}
        pc.set_facecolor(cmap.get(tw, "#888"))
        ax.add_collection3d(pc)
        b = 8
        ax.set_xlim(-b, b); ax.set_ylim(-b, b); ax.set_zlim(0, 12)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(f"twist = {tw}$^\\circ$", fontsize=10)
# No figure title: the manuscript caption carries the model description,
# and MDPI figures are captioned rather than titled.
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_schonhardt_family.png",
                dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_schonhardt_family.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_schonhardt_family.{png,pdf}")


if __name__ == "__main__":
    generate_family()
    render_family()
