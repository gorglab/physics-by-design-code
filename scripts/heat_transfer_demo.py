"""
heat_transfer_demo.py
=====================

Illustrative external-fin toy model with closed-cell porosity.

We solve the 1-D transient conduction-convection fin equation

    (rho c)_eff(phi) A_c dT/dt = k_eff(phi) A_c d2T/dx2
                                  - h_conv P_fin (T - T_inf)

with Voigt scaling of both the thermal conductivity and the volumetric heat
capacity:

    k_eff(phi)     = (1 - phi) k_solid
    (rho c)_eff    = (1 - phi) rho_solid c_solid

The convective coefficient h_conv and the external perimeter P_fin are held
fixed across phi: this is an EXTERNAL fin with closed-cell porosity, not a
through-flow TPMS heat sink. A real TPMS heat sink would also require
h_conv(phi), P_wet(phi), and the internal specific surface area S_v(phi).

This script is not a substitute for an experimental campaign; it is a
forward toy model used in the paper to illustrate how a porosity sweep maps
to a measurable temperature signal at the fin tip.

Outputs:
    figures/fig_heat_transfer.png / .pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def effective_conductivity(k_solid: float, phi: float, model: str = "parallel"):
    """Two simple effective-medium bounds for thermal conductivity."""
    if model == "parallel":
        return (1.0 - phi) * k_solid
    if model == "series":
        # avoid divide by zero for vacuum pores; assume k_air ~ 0.026 W/mK
        k_air = 0.026
        return 1.0 / ((1.0 - phi) / k_solid + phi / k_air)
    raise ValueError(model)


def effective_rho_c(rho_solid: float, c_solid: float, phi: float):
    """Voigt (parallel) effective volumetric heat capacity, fluid term = 0.

    (rho c)_eff(phi) = (1 - phi) rho_solid c_solid

    Justified for AlSi10Mg in air because (rho c)_air / (rho c)_solid
    is approximately 5e-4 -- negligible.
    """
    return (1.0 - phi) * rho_solid * c_solid


def solve_fin(phi: float, L=0.05, N=80, t_end=60.0, dt=0.05,
              T0=20.0, T_base=120.0,
              h_conv=15.0, P_fin=4 * 0.008, A_c=0.008 ** 2,
              rho_solid=2700.0, c_solid=900.0, k_solid=170.0):
    """Transient 1-D fin equation with Voigt-scaled effective properties.

    (rho c)_eff A_c dT/dt = k_eff A_c d2T/dx2 - h_conv P_fin (T - T_inf)

    Discretised with explicit Euler in time and centred differences in
    space. The insulated tip is enforced by setting T[N-1] = T[N-2].
    """
    k_eff = effective_conductivity(k_solid, phi, "parallel")
    rhoc_eff = effective_rho_c(rho_solid, c_solid, phi)
    dx = L / (N - 1)
    T = np.full(N, T0)
    # thermal diffusivity is phi-invariant under Voigt scaling
    alpha = k_eff / rhoc_eff if rhoc_eff > 0.0 else 0.0
    # convective time-scale per unit thermal mass (grows with phi)
    conv_coeff = h_conv * P_fin / (rhoc_eff * A_c)
    # stability
    if alpha > 0.0:
        dt = min(dt, 0.45 * dx ** 2 / alpha)
    nsteps = int(t_end / dt)
    times, tips = [], []
    for n in range(nsteps):
        Tn = T.copy()
        Tn[0] = T_base
        d2 = (T[2:] - 2 * T[1:-1] + T[:-2]) / dx ** 2
        Tn[1:-1] = T[1:-1] + dt * (alpha * d2 - conv_coeff * (T[1:-1] - T0))
        Tn[-1] = Tn[-2]   # insulated tip
        T = Tn
        if n % 50 == 0:
            times.append(n * dt)
            tips.append(T[-1])
    return np.array(times), np.array(tips)


def demo():
    porosities = [0.0, 0.2, 0.4, 0.6, 0.8]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cmap = plt.cm.viridis
    for i, phi in enumerate(porosities):
        t, tip = solve_fin(phi)
        ax.plot(t, tip, lw=2, color=cmap(i / max(1, len(porosities) - 1)),
                label=f"$\\phi$={phi:.1f}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fin-tip temperature (°C)")
    ax.set_title(
        "Illustrative external-fin toy model with closed-cell porosity\n"
        r"$(\rho c)_\mathrm{eff}=(1-\phi)\rho c$, $k_\mathrm{eff}=(1-\phi)k_\mathrm{solid}$, "
        r"$h_\mathrm{conv}$ and $P_\mathrm{fin}$ held constant",
        fontsize=9,
    )
    ax.legend(title="porosity $\\phi$", fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_heat_transfer.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_heat_transfer.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_heat_transfer.{png,pdf}")


if __name__ == "__main__":
    demo()
