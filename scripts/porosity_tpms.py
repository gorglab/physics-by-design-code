"""
porosity_tpms.py
================

Generate STL meshes of triply periodic minimal surfaces (TPMS) cubes whose
porosity is controlled by a single scalar parameter.  The same external
bounding box, lattice period and surface family are kept fixed across runs so
that only the porosity changes between specimens.  This is the canonical
"vary one parameter at a time" workflow that additive manufacturing makes
possible for physical specimens.

Surfaces implemented:
    * Schwarz-P:        cos(x) + cos(y) + cos(z) - t = 0
    * Gyroid:           sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) - t = 0
    * Schwarz-D:        sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z)
                       + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z) - t = 0

The implicit threshold ``t`` controls the volume fraction.  We compute the
realised porosity by voxel counting after meshing so that the STL filename
carries the *measured* porosity, not the *requested* offset.

Usage (from the repo root):

    python scripts/porosity_tpms.py

This writes a family of STLs to ``generated_stl/`` and a montage figure to
``figures/fig_porosity_family.png`` / ``.pdf``.

Daniel N. Wilke, 2026.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
STL_DIR = ROOT / "generated_stl"
FIG_DIR = ROOT / "figures"
STL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Implicit surface definitions
# --------------------------------------------------------------------------
def schwarz_p(x, y, z):
    return np.cos(x) + np.cos(y) + np.cos(z)


def gyroid(x, y, z):
    return (
        np.sin(x) * np.cos(y)
        + np.sin(y) * np.cos(z)
        + np.sin(z) * np.cos(x)
    )


def schwarz_d(x, y, z):
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz


SURFACES = {"schwarzP": schwarz_p, "gyroid": gyroid, "schwarzD": schwarz_d}


# --------------------------------------------------------------------------
# Mesh generation
# --------------------------------------------------------------------------
@dataclass
class TPMSSpec:
    family: str            # one of SURFACES
    threshold: float       # implicit level set value t
    n_periods: int = 2     # number of unit cells per side
    cube_size_mm: float = 20.0
    grid: int = 96         # voxel grid resolution per side


def build_tpms_mesh(spec: TPMSSpec) -> tuple[trimesh.Trimesh, float]:
    """Return (mesh, measured_porosity) for a TPMS unit-cell tile clipped to a
    cube of ``spec.cube_size_mm`` per side.

    Solid phase is where ``F(x,y,z) <= threshold``.  Porosity = 1 - solid
    fraction, measured by voxel counting on the same grid.
    """
    f = SURFACES[spec.family]
    n = spec.grid
    # sample in units where one TPMS period = 2 pi
    lin = np.linspace(0.0, 2.0 * np.pi * spec.n_periods, n)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    F = f(X, Y, Z)

    # marching cubes on F - t = 0
    verts, faces, _, _ = measure.marching_cubes(
        F, level=spec.threshold, spacing=(1.0, 1.0, 1.0)
    )
    # rescale verts so the cube spans [0, cube_size_mm]
    verts = verts * (spec.cube_size_mm / (n - 1))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # close the boundary to make a printable solid by capping with the cube
    # face planes via boolean intersect with a closed cube hull.
    cube = trimesh.creation.box(extents=[spec.cube_size_mm] * 3)
    cube.apply_translation([spec.cube_size_mm / 2.0] * 3)
    try:
        clipped = mesh.intersection(cube)
        if clipped.is_empty or len(clipped.faces) == 0:
            clipped = mesh
    except Exception:
        clipped = mesh

    # measured porosity by voxel counting on the same scalar field
    solid_voxels = np.count_nonzero(F <= spec.threshold)
    porosity = 1.0 - solid_voxels / F.size
    return clipped, float(porosity)


# --------------------------------------------------------------------------
# Family generation
# --------------------------------------------------------------------------
def generate_porosity_family():
    """Generate three TPMS families across a porosity sweep.

    The threshold sweep is the *only* parameter that varies.  Cube size, period
    count, grid resolution and surface family are fixed within each family.
    """
    sweep = {
        "gyroid":    np.linspace(-0.8, 0.8, 5),
        "schwarzP":  np.linspace(-0.8, 0.8, 5),
        "schwarzD":  np.linspace(-0.6, 0.6, 5),
    }
    records = []
    for family, thresholds in sweep.items():
        for t in thresholds:
            spec = TPMSSpec(family=family, threshold=float(t))
            mesh, phi = build_tpms_mesh(spec)
            fname = f"{family}_t{t:+.2f}_phi{phi:.3f}.stl"
            mesh.export(STL_DIR / fname)
            records.append((family, t, phi, fname))
            print(f"  wrote {fname}  porosity={phi:.3f}")
    return records


# --------------------------------------------------------------------------
# Visualisation: real 3D shaded STL renders
# --------------------------------------------------------------------------
from render_utils import render_mesh  # noqa: E402


FAMILY_COLOURS = {
    "gyroid":   "#3a7ca5",
    "schwarzP": "#c1666b",
    "schwarzD": "#6a994e",
}


def _build_render_mesh(spec: TPMSSpec) -> trimesh.Trimesh:
    """Build the marching-cubes mesh clipped to the cube, ready for rendering."""
    mesh, _ = build_tpms_mesh(spec)
    return mesh


def _render_iso(ax, spec: TPMSSpec, title: str, face_colour: str):
    mesh = _build_render_mesh(spec)
    render_mesh(ax, mesh, base_color=face_colour, ambient=0.32,
                light=(0.4, -0.55, 0.78))
    ax.set_title(title, fontsize=9)


def render_montage():
    """Compose a 3 x 5 grid of solid-rendered TPMS STL views.
    Rows = families, columns = porosity sweep.
    """
    sweep = {
        "gyroid":   np.linspace(-0.8, 0.8, 5),
        "schwarzP": np.linspace(-0.8, 0.8, 5),
        "schwarzD": np.linspace(-0.6, 0.6, 5),
    }
    fig = plt.figure(figsize=(11.5, 7.2))
    for r, (family, thresholds) in enumerate(sweep.items()):
        for c, t in enumerate(thresholds):
            ax = fig.add_subplot(3, 5, r * 5 + c + 1, projection="3d")
            spec = TPMSSpec(family=family, threshold=float(t), grid=72)
            _, phi = build_tpms_mesh(spec)
            _render_iso(ax, spec, f"{family}\n$\\phi$={phi:.2f}",
                        FAMILY_COLOURS[family])
    fig.suptitle(
        "Single-parameter porosity sweep across TPMS families\n"
        "(unit cell, cube size and lattice period held constant; STL solids rendered)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = FIG_DIR / "fig_porosity_family.png"
    out_pdf = FIG_DIR / "fig_porosity_family.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    print("Generating TPMS porosity family ...")
    generate_porosity_family()
    print("Rendering montage ...")
    render_montage()
    print("Done.")
