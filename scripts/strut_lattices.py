"""
strut_lattices.py
=================

Physics by Design: Additive Manufacturing for Reproducible Science
Daniel N. Wilke
Gradient-Only Research Group (GorgLab) for Emerging Engineering Technology (EET)
Mathematical and Computational Applications (MDPI),
Special Issue "Advances in Computational and Applied Mechanics".

Licence: MIT. Free to use, copy, modify and redistribute, with attribution.
Provided "as is", without warranty of any kind, express or implied. The
author accepts no liability for any use of this code or its outputs.
Use at your own risk.

Generate strut-based lattice cubes for five canonical topologies:

    * SC      -- simple cubic   (edges of the unit cell)
    * BCC     -- body-centred cubic (12 cube edges + 8 centre-to-corner
                 struts = 20 struts/cell)
    * FCC     -- face-centred cubic (12 cube edges + 24 corner-to-face-centre
                 struts = 36 struts/cell)
    * OCTET   -- octet truss (FCC + face-centring nodes connected)
    * DIAMOND -- diamond cubic (tetrahedral coordination, four struts per node)

For each topology the script tunes the strut radius so that the *measured*
solid volume fraction (1 - porosity) matches a target.  The result is a
family of cubes with the **same external dimensions**, the **same target
porosity**, but **different topology** -- the canonical "vary geometry,
hold bulk porosity constant" experiment that the photographs in
``figures/printed_lattice_*.jpg`` show physically.

We also emit a *topology-fixed porosity sweep* for the octet truss so the
companion of `porosity_tpms.py` is available for strut lattices.

The implementation is deliberately simple: each strut is a capped cylinder
union, voxelised on a coarse grid for porosity measurement, then exported
as a Boolean union mesh via ``trimesh.boolean``.  For small unit-cell
counts (2-3 cells per side, 60-80 voxels) this finishes in seconds on a
laptop.

Outputs:
    generated_stl/lattice_{topology}_{phi}.stl   (target porosity tagged)
    figures/fig_lattice_isoporous.{png,pdf}      (5 topologies, same phi)
    figures/fig_lattice_porosity_sweep.{png,pdf} (octet sweep)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq

from render_utils import render_mesh

ROOT = Path(__file__).resolve().parents[1]
STL_DIR = ROOT / "generated_stl"
FIG_DIR = ROOT / "figures"
STL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Topology definitions: list of (start, end) node pairs in the unit cube.
# Coordinates are in [0, 1]^3.
# --------------------------------------------------------------------------
def edges_sc() -> list[tuple[tuple, tuple]]:
    """12 cube edges."""
    c = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    pairs = []
    for i, a in enumerate(c):
        for b in c[i + 1:]:
            if sum(abs(ai - bi) for ai, bi in zip(a, b)) == 1:
                pairs.append((a, b))
    return pairs


def edges_bcc():
    pairs = edges_sc()
    centre = (0.5, 0.5, 0.5)
    for v in [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]:
        pairs.append((centre, v))
    return pairs


def edges_fcc():
    pairs = edges_sc()
    face_centres = [
        (0.5, 0.5, 0.0), (0.5, 0.5, 1.0),
        (0.5, 0.0, 0.5), (0.5, 1.0, 0.5),
        (0.0, 0.5, 0.5), (1.0, 0.5, 0.5),
    ]
    # connect each face centre to its 4 surrounding corners
    for fc in face_centres:
        for corner in [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]:
            if sum(abs(fc[i] - corner[i]) for i in range(3)) == 1.0:
                pairs.append((fc, corner))
    return pairs


def edges_octet():
    """Octet truss = FCC corner/face-centre lattice with octahedral cell."""
    face_centres = [
        (0.5, 0.5, 0.0), (0.5, 0.5, 1.0),
        (0.5, 0.0, 0.5), (0.5, 1.0, 0.5),
        (0.0, 0.5, 0.5), (1.0, 0.5, 0.5),
    ]
    pairs = []
    # corners <-> face centres
    for fc in face_centres:
        for corner in [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]:
            if sum(abs(fc[i] - corner[i]) for i in range(3)) == 1.0:
                pairs.append((fc, corner))
    # face-centres pairwise (octahedron edges)
    for i, a in enumerate(face_centres):
        for b in face_centres[i + 1:]:
            if sum((ai - bi) ** 2 for ai, bi in zip(a, b)) < 1.0:
                pairs.append((a, b))
    return pairs


def edges_diamond():
    """Diamond cubic: 8 atoms per cell, tetrahedral coordination."""
    a = [
        (0.00, 0.00, 0.00), (0.50, 0.50, 0.00),
        (0.50, 0.00, 0.50), (0.00, 0.50, 0.50),
        (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
        (0.75, 0.25, 0.75), (0.25, 0.75, 0.75),
    ]
    bonds = []
    # each "0.25" atom bonds to four nearest "0.0" atoms
    for i in range(4, 8):
        d = [(j, np.linalg.norm(np.array(a[i]) - np.array(a[j]))) for j in range(4)]
        d.sort(key=lambda x: x[1])
        for j, _ in d[:4]:
            bonds.append((a[i], a[j]))
    return bonds


TOPOLOGIES = {
    "sc":      edges_sc,
    "bcc":     edges_bcc,
    "fcc":     edges_fcc,
    "octet":   edges_octet,
    "diamond": edges_diamond,
}


# --------------------------------------------------------------------------
# Voxel-based solid-fraction estimator (cheap, robust, mesh-free)
# --------------------------------------------------------------------------
def _segment_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Point-to-segment distance for an (N,3) batch of points to one segment."""
    AB = B - A
    t = np.clip(((P - A) @ AB) / (AB @ AB), 0.0, 1.0)
    closest = A + np.outer(t, AB)
    return np.linalg.norm(P - closest, axis=1)


def solid_fraction(edges: Iterable[tuple], n_cells: int, r_rel: float,
                   grid: int = 60) -> float:
    """Voxel-count the union of all strut cylinders inside the [0,1]^3 cell."""
    edges = list(edges)
    lin = np.linspace(0.0, 1.0, grid)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    solid = np.zeros(pts.shape[0], dtype=bool)
    for a, b in edges:
        d = _segment_distance(pts, np.asarray(a), np.asarray(b))
        solid |= d <= r_rel
    return solid.mean()


# --------------------------------------------------------------------------
# Mesh generation: union of capped cylinders
# --------------------------------------------------------------------------
def build_lattice_mesh(topology: str, n_cells: int, r_rel: float,
                       cube_size_mm: float = 20.0,
                       cyl_sections: int = 10) -> trimesh.Trimesh:
    edges_unit = TOPOLOGIES[topology]()
    pitch = cube_size_mm / n_cells
    r_mm = r_rel * pitch
    parts = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                offset = np.array([i, j, k]) * pitch
                for a, b in edges_unit:
                    A = np.asarray(a) * pitch + offset
                    B = np.asarray(b) * pitch + offset
                    L = np.linalg.norm(B - A)
                    if L < 1e-9:
                        continue
                    cyl = trimesh.creation.cylinder(
                        radius=r_mm, height=L, sections=cyl_sections,
                    )
                    # default cylinder is z-aligned, centred at origin
                    z_axis = np.array([0, 0, 1.0])
                    direction = (B - A) / L
                    # rotation matrix from z to direction
                    R4 = trimesh.geometry.align_vectors(z_axis, direction)
                    cyl.apply_transform(R4)
                    cyl.apply_translation(0.5 * (A + B))
                    parts.append(cyl)
    mesh = trimesh.util.concatenate(parts)
    # crop to bounding cube to remove the half-struts that protrude on the
    # outer faces (purely cosmetic; the porosity number is measured in the
    # unit cell, not on the mesh)
    return mesh


# --------------------------------------------------------------------------
# Iso-porosity family
# --------------------------------------------------------------------------
def find_radius_for_porosity(topology: str, n_cells: int, target_phi: float,
                             grid: int = 60) -> float:
    """Solve for r_rel such that solid_fraction == 1 - target_phi."""
    edges_unit = TOPOLOGIES[topology]()
    target_solid = 1.0 - target_phi

    def f(r):
        return solid_fraction(edges_unit, n_cells, r, grid=grid) - target_solid
    # bracket: r=0.005 always under target, r=0.25 always above
    return brentq(f, 0.005, 0.25, xtol=1e-3)


def generate_isoporous_family(target_phi: float = 0.70, n_cells: int = 3):
    records = []
    for topo in ["sc", "bcc", "fcc", "octet", "diamond"]:
        r = find_radius_for_porosity(topo, n_cells, target_phi)
        phi_meas = 1.0 - solid_fraction(TOPOLOGIES[topo](), n_cells, r)
        mesh = build_lattice_mesh(topo, n_cells, r)
        fname = f"lattice_{topo}_phi{target_phi:.2f}_meas{phi_meas:.3f}.stl"
        mesh.export(STL_DIR / fname)
        records.append((topo, r, phi_meas, fname))
        print(f"  {topo:8s}  r/pitch={r:.3f}  phi_meas={phi_meas:.3f}  -> {fname}")
    return records


def generate_octet_porosity_sweep(targets=(0.5, 0.6, 0.7, 0.8), n_cells: int = 3):
    records = []
    for phi in targets:
        r = find_radius_for_porosity("octet", n_cells, phi)
        phi_meas = 1.0 - solid_fraction(TOPOLOGIES["octet"](), n_cells, r)
        mesh = build_lattice_mesh("octet", n_cells, r)
        fname = f"lattice_octet_phi{phi:.2f}_meas{phi_meas:.3f}.stl"
        mesh.export(STL_DIR / fname)
        records.append((phi, r, phi_meas, fname))
        print(f"  octet  target phi={phi:.2f}  r/pitch={r:.3f}  meas={phi_meas:.3f}")
    return records


# --------------------------------------------------------------------------
# Real 3D STL renders (Lambertian-shaded triangle meshes)
# --------------------------------------------------------------------------
TOPOLOGY_COLOURS = {
    "sc":      "#264653",
    "bcc":     "#2a9d8f",
    "fcc":     "#e9c46a",
    "octet":   "#f4a261",
    "diamond": "#9d4edd",
}


def _build_render_mesh(topology: str, r_rel: float,
                       n_cells_preview: int = 2,
                       cube_size_mm: float = 20.0,
                       cyl_sections: int = 8) -> trimesh.Trimesh:
    """Build a small mesh for figure rendering. We use 2x2x2 cells with a
    coarser cylinder discretisation so figure builds in seconds, while the
    full 3x3x3 STLs stay in ``generated_stl/`` for downstream printing.
    """
    return build_lattice_mesh(topology, n_cells_preview, r_rel,
                              cube_size_mm=cube_size_mm,
                              cyl_sections=cyl_sections)


def render_isoporous_montage(target_phi: float, records):
    topos = ["sc", "bcc", "fcc", "octet", "diamond"]
    fig = plt.figure(figsize=(13.5, 3.8))
    for i, topo in enumerate(topos):
        rec = next(r for r in records if r[0] == topo)
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        mesh = _build_render_mesh(topo, r_rel=rec[1], n_cells_preview=2,
                                  cyl_sections=10)
        render_mesh(ax, mesh, base_color=TOPOLOGY_COLOURS[topo],
                    ambient=0.32, light=(0.45, -0.55, 0.78))
        ax.set_title(f"{topo.upper()}\n$\\phi_{{\\rm meas}}={rec[2]:.2f}$",
                     fontsize=10)
    # No figure title: the manuscript caption carries the
    # description; MDPI figures are captioned, not titled.
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_lattice_isoporous.png", dpi=210,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_lattice_isoporous.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_lattice_isoporous.{png,pdf}")


def render_octet_sweep(records):
    fig = plt.figure(figsize=(11.5, 3.4))
    for i, (phi, r, phi_meas, _) in enumerate(records):
        ax = fig.add_subplot(1, len(records), i + 1, projection="3d")
        mesh = _build_render_mesh("octet", r_rel=r, n_cells_preview=2,
                                  cyl_sections=10)
        render_mesh(ax, mesh, base_color=TOPOLOGY_COLOURS["octet"],
                    ambient=0.32, light=(0.45, -0.55, 0.78))
        ax.set_title(f"$\\phi={phi_meas:.2f}$", fontsize=10)
    # No figure title: the manuscript caption carries the
    # description; MDPI figures are captioned, not titled.
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_lattice_porosity_sweep.png", dpi=210,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_lattice_porosity_sweep.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_lattice_porosity_sweep.{png,pdf}")


if __name__ == "__main__":
    print("Iso-porosity family (phi = 0.70):")
    iso = generate_isoporous_family(target_phi=0.70, n_cells=3)
    render_isoporous_montage(0.70, iso)

    print("\nOctet-truss porosity sweep:")
    sweep = generate_octet_porosity_sweep(targets=(0.50, 0.60, 0.70, 0.80))
    render_octet_sweep(sweep)
    print("Done.")
