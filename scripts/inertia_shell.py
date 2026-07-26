"""
inertia_shell.py
================

Physics by Design: Additive Manufacturing for Reproducible Science
Daniel N. Wilke
Gradient-Only Research Group (GorgLab) for Emerging Engineering Technology (EET)
Mathematical and Computational Applications (MDPI),
Special Issue "Advances in Computational and Applied Mechanics".

Licence: MIT. Free to use, copy, modify and redistribute, with attribution.
Provided "as is", without warranty of any kind, express or implied. The
author accepts no liability for any use of this code or its outputs.
Use at your own risk.

Demonstrate the second canonical "vary one thing at a time" sweep that AM
supports: keep the *outer* particle geometry fixed and change only the
*internal* mass distribution.  We model a hollow sphere whose printed shell
thickness moves an internal ballast inwards or outwards.  Outer radius and
total mass are held constant; only the principal moment of inertia changes.

This corresponds physically to FDM/SLA workflows where the slicer infill
density and infill-pattern offset are used to redistribute mass without
changing the exterior CAD surface.

Outputs:
    figures/fig_inertia_shell.png / .pdf
    generated_stl/shell_Router*_Rinner*.stl
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


def hollow_sphere_mesh(R_outer: float, R_inner: float, sections: int = 64):
    outer = trimesh.creation.icosphere(subdivisions=4, radius=R_outer)
    inner = trimesh.creation.icosphere(subdivisions=4, radius=R_inner)
    # invert inner to carve a cavity
    inner.invert()
    return trimesh.util.concatenate([outer, inner])


def moment_of_inertia_shell(R_outer, R_inner, mass):
    """Analytic I for a uniform thick spherical shell of given total mass."""
    num = R_outer ** 5 - R_inner ** 5
    den = R_outer ** 3 - R_inner ** 3
    return (2.0 / 5.0) * mass * num / den


def demo():
    R = 10.0  # mm, outer radius fixed
    M = 1.0   # arbitrary normalised mass
    inner_radii = np.linspace(0.0, 9.0, 7)  # 0 = solid, 9 mm = thin shell
    Is, mesh_files = [], []
    for Ri in inner_radii:
        I = moment_of_inertia_shell(R, Ri, M)
        Is.append(I)
        mesh = hollow_sphere_mesh(R, Ri)
        f = STL_DIR / f"shell_Router{R:.1f}_Rinner{Ri:.1f}.stl"
        mesh.export(f)
        mesh_files.append(f.name)
        print(f"  R_inner={Ri:4.1f}  I/MR^2={I/(M*R**2):.3f}  -> {f.name}")

    Is = np.array(Is)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(inner_radii / R, Is / (M * R ** 2), marker="o", lw=2,
            color="#264653")
    ax.set_xlabel(r"normalised inner radius $q = R_\mathrm{in}/R$")
    ax.set_ylabel(r"normalised principal moment $I_p / (M R^2)$")
    # No figure title: the manuscript caption carries the model description,
    # and MDPI figures are captioned rather than titled.
    ax.axhline(2 / 3, ls="--", color="#999",
               label=r"thin-shell limit $\frac{2}{3} M R^2$")
    ax.axhline(2 / 5, ls=":", color="#999",
               label=r"solid-sphere limit $\frac{2}{5} M R^2$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_inertia_shell.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_inertia_shell.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_inertia_shell.{png,pdf}")


if __name__ == "__main__":
    demo()
