# Physics by Design: Additive Manufacturing for Reproducible Science

**Author:** Daniel N. Wilke
School of Mechanical, Industrial and Aeronautical Engineering
University of the Witwatersrand, Johannesburg, South Africa
`daniel.wilke@wits.ac.za`

**Target:** *Mathematical and Computational Applications* (MDPI), Special Issue
*Advances in Computational and Applied Mechanics*
<https://www.mdpi.com/journal/mca/special_issues/4H3DF668ZO>
**Article type:** Perspective
**Template:** Official MDPI LaTeX template (ACS reference style)

## Repository layout

```
.
├── scripts/                    Reproducible figure & STL generators
│   ├── porosity_tpms.py        TPMS porosity sweep + STL family (Figure 4)
│   ├── schonhardt_particles.py Schönhardt twist sweep + STL family (Figure 2, Table 4)
│   ├── inertia_shell.py        Hollow-sphere shell-thickness sweep (Figure 3)
│   ├── strut_lattices.py       Iso-porous and porosity-swept strut lattices (Figures 6, 7, Table 5)
│   ├── heat_transfer_demo.py   1-D transient fin model (Figure 8)
│   └── render_utils.py         Lambertian software renderer used by the figure scripts
│
├── generated_stl/              STL outputs of the scripts (34 meshes)
└── figures/                    PDF + PNG + JPG figures used by main.tex
```

## How to rebuild everything from source

```bash
# 0. Reproducible Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Regenerate every figure and every STL from the scripts
python3 scripts/porosity_tpms.py
python3 scripts/schonhardt_particles.py
python3 scripts/inertia_shell.py
python3 scripts/strut_lattices.py
```

## Dependencies

- Python ≥ 3.10 with the libraries pinned in `requirements.txt`:
  `numpy`, `scipy`, `matplotlib`, `scikit-image`, `trimesh`, `pillow`,
  `numpy-stl`.

## License

- Scripts (`scripts/`) and Python code: MIT.
