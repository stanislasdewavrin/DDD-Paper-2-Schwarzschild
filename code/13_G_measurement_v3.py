"""
[v7-DEPRECATED CONVENTION]

This script uses the pre-v7 convention with an explicit flux cap F_MAX
and (in some scripts) a separate reserve floor R_MIN. As of Paper I v7,
F_MAX is no longer a free parameter: the flux is automatically bounded
by the reserve, and the only physical floor is R = 0 (one cannot drain
what is not there). Non-negativity is enforced by a per-node rate-limiter
beta_i, not by clipping.

The numerical results obtained here in the linear regime
(F_MAX never reached and R_MIN never reached) remain valid as
characterisations of the linearised DDD substrate. They should match,
to within Madelung-like discretisation corrections, the v7 results from:
  - paperII_gravity/code/G_measure_standalone.py  (linearised Jacobi)
  - paperI_foundations/code/v7_drainage_rule.py   (full rate-limited)

For new analyses, prefer the v7 scripts. This file is kept for
reproducibility of pre-v7 results referenced in earlier drafts.
"""

"""
G measurement v3 — Dirichlet boundary at center
=================================================

Approach: instead of trying to drain at the center (which saturates),
we IMPOSE a fixed deficit delta_0 at the center (Dirichlet boundary
condition) and let the flux equation diffuse it outward. The
stationary profile is the discrete Laplacian solution with point source,
which is 1/r in 3D.

This is the cleanest way to extract G_eff from the simulation.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)

ALPHA = 1.0       # diffusion coefficient
F_MAX = 1.0       # no flux clamp (linear regime)
DELTA_0 = 0.1     # imposed deficit at center
L = 64
N_ITER = 4000

print(f"Lattice {L}^3, ALPHA={ALPHA}, F_MAX={F_MAX}, DELTA_0={DELTA_0}, N_ITER={N_ITER}")

# Initialize
delta = np.zeros((L, L, L))
cx, cy, cz = L//2, L//2, L//2

# Iterate: at each step, set delta(center) = DELTA_0, then diffuse
print("\nIteration...")
for t in range(N_ITER):
    delta[cx, cy, cz] = DELTA_0
    # Diffuse
    new_delta = delta.copy()
    flux = np.zeros_like(delta)
    for ax in range(3):
        for shift in [-1, +1]:
            d_n = np.roll(delta, shift=shift, axis=ax)
            f = np.minimum(F_MAX, ALPHA * np.maximum(d_n - delta, 0.0))
            flux += f
    new_delta += 0.1 * flux  # damped update for stability
    new_delta[0, :, :] = 0; new_delta[-1, :, :] = 0
    new_delta[:, 0, :] = 0; new_delta[:, -1, :] = 0
    new_delta[:, :, 0] = 0; new_delta[:, :, -1] = 0
    delta = new_delta
    if t % 500 == 0:
        d5 = delta[cx+5, cy, cz]
        d10 = delta[cx+10, cy, cz]
        print(f"  t={t}: delta_5={d5:.6f}, delta_10={d10:.6f}")

# Final extraction
print(f"\nFinal: delta(0)={delta[cx,cy,cz]:.6f}, delta(5)={delta[cx+5,cy,cz]:.6f}")


# Radial profile
xs = np.arange(L) - cx
ys = np.arange(L) - cy
zs = np.arange(L) - cz
X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
r = np.sqrt(X**2 + Y**2 + Z**2)
r_bins = np.arange(0.5, 25, 0.25)
r_centers = (r_bins[:-1] + r_bins[1:]) / 2
profile = np.zeros(len(r_centers))
err = np.zeros(len(r_centers))
counts = np.zeros(len(r_centers))
for i, (rmin, rmax) in enumerate(zip(r_bins[:-1], r_bins[1:])):
    mask = (r >= rmin) & (r < rmax)
    n = mask.sum()
    counts[i] = n
    if n > 1:
        profile[i] = delta[mask].mean()
        err[i] = delta[mask].std() / np.sqrt(n)

# Fit Newton
def newton(r, A, B):
    return A / r + B

m = (r_centers > 2.5) & (r_centers < 18) & (counts > 5) & (profile > 0)
print(f"\nFitting {m.sum()} bins...")
if m.sum() > 5:
    e_fit = np.maximum(err[m], 1e-12)
    popt, pcov = curve_fit(newton, r_centers[m], profile[m], sigma=e_fit, absolute_sigma=True)
    A, B = popt
    A_err, B_err = np.sqrt(np.diag(pcov))
    print(f"\n  A = {A:.10f} +/- {A_err:.3e}")
    print(f"  Relative precision: {A_err/A:.3e}")

    # G_eff in lattice units
    M_eff = DELTA_0  # equivalent to imposed deficit charge
    G_eff = A / (2 * M_eff)
    G_eff_err = A_err / (2 * M_eff)
    G_rel = G_eff_err / G_eff
    print(f"\n  G_eff = A / (2 M_eff) = {G_eff:.10f}")
    print(f"  Relative precision G: {G_rel:.3e}")

    EXP = 2.2e-5
    print(f"\n  Experimental precision (CODATA): {EXP:.1e}")
    print(f"  Ratio sim/exp: {EXP/G_rel:.0f}x more precise" if G_rel < EXP else f"  {G_rel/EXP:.0f}x less precise")

    # Show a few sample profile points
    print(f"\n{'r':>6} {'delta':>14} {'A/r predicted':>16} {'residual':>12}")
    for i in range(0, 30, 3):
        if m[i]:
            pred = A/r_centers[i] + B
            res = profile[i] - pred
            print(f"{r_centers[i]:>6.2f} {profile[i]:>14.8f} {pred:>16.8f} {res:>+12.5e}")

    # Plot
    fig, ax = plt.subplots(figsize=(8,5))
    ax.errorbar(r_centers[m], profile[m], yerr=e_fit, fmt='o', ms=4, c='navy', label='simulation')
    rs = np.linspace(2, 18, 100)
    ax.plot(rs, newton(rs, A, B), 'r-', label=f'A/r + B, A={A:.5f}')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\delta(r)$')
    ax.set_title(f'Drainage profile, A precision = {A_err/A:.2e}')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIG / 'fig_G_v3.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    results = {
        "L": L, "N_ITER": N_ITER, "ALPHA": ALPHA,
        "DELTA_0": DELTA_0,
        "A": float(A), "A_err": float(A_err),
        "A_rel_precision": float(A_err/A),
        "G_eff": float(G_eff),
        "G_rel_precision": float(G_rel),
        "experimental_precision": EXP,
        "factor_better_than_exp": float(EXP/G_rel) if G_rel > 0 else None,
    }
    with open(DATA / "G_measurement_v3.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved fig_G_v3.png and G_measurement_v3.json")
else:
    print(f"Not enough bins to fit. Profile may not have developed.")
    print(f"Sample: {profile[:10]}")
