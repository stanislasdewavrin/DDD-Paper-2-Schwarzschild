"""
G measurement v4 - Discrete Poisson equation
==============================================

The DDD drainage equation in stationary regime, linearised:
    Laplacian(delta) = source(r)
For a point source at origin, the 3D solution is delta = A/r,
with A = M / (4 pi).

We solve the discrete Poisson equation by Jacobi iteration on a
cubic grid with Dirichlet BC at the boundary (delta=0 at infinity).
This is the cleanest way to extract G_eff with high numerical
precision in the simulation.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)

L = 64
N_ITER = 5000
M_source = 1.0  # point source mass

print(f"Lattice {L}^3, Jacobi iterations: {N_ITER}, point source M={M_source}")

cx, cy, cz = L//2, L//2, L//2
delta = np.zeros((L, L, L))
source = np.zeros((L, L, L))
source[cx, cy, cz] = M_source

# Jacobi iteration: delta_new[i] = (sum of 6 neighbours - source[i]) / 6
# This solves: 6 delta - sum_neigh = -source  <=>  Laplacian = source
print("\nJacobi iteration...")
for it in range(N_ITER):
    new_delta = (np.roll(delta, +1, 0) + np.roll(delta, -1, 0)
               + np.roll(delta, +1, 1) + np.roll(delta, -1, 1)
               + np.roll(delta, +1, 2) + np.roll(delta, -1, 2)
               + source) / 6.0
    # Dirichlet BC at all 6 boundary faces (delta=0 at infinity)
    new_delta[0, :, :] = 0; new_delta[-1, :, :] = 0
    new_delta[:, 0, :] = 0; new_delta[:, -1, :] = 0
    new_delta[:, :, 0] = 0; new_delta[:, :, -1] = 0
    if it % 5000 == 0:
        diff = np.max(np.abs(new_delta - delta))
        d5 = new_delta[cx+5, cy, cz]
        d10 = new_delta[cx+10, cy, cz]
        d20 = new_delta[cx+20, cy, cz]
        print(f"  it={it}: max change={diff:.2e}, delta(5)={d5:.6f}, "
              f"delta(10)={d10:.6f}, delta(20)={d20:.6f}")
    delta = new_delta
print(f"\nDone. Center: delta(0)={delta[cx,cy,cz]:.4f}")


# Radial profile
xs = np.arange(L) - cx
ys = np.arange(L) - cy
zs = np.arange(L) - cz
X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
r = np.sqrt(X**2 + Y**2 + Z**2)
r_bins = np.arange(0.5, 30, 0.25)
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


# Newton fit
def newton(r, A, B):
    return A / r + B

mask_fit = (r_centers > 3) & (r_centers < 25) & (counts > 5) & (profile > 0)
print(f"\nFitting {mask_fit.sum()} bins...")
e_fit = np.maximum(err[mask_fit], 1e-15)
popt, pcov = curve_fit(newton, r_centers[mask_fit], profile[mask_fit],
                        sigma=e_fit, absolute_sigma=True)
A, B = popt
A_err, B_err = np.sqrt(np.diag(pcov))
print(f"\n  A = {A:.12f} +/- {A_err:.3e}")
print(f"  Relative precision: {A_err/A:.3e}")
print(f"  B = {B:.6e} +/- {B_err:.3e}")

# Theoretical prediction: in 3D continuum, Laplacian(1/(4 pi r)) = -delta(r),
# so for source M at origin: delta = M / (4 pi r), i.e. A = M/(4 pi) = 0.0796
# In discrete cubic grid, the Green function differs slightly from 1/(4 pi r)
# at small r (Madelung-like correction). At large r, the continuum result
# applies.
A_continuum = M_source / (4 * np.pi)
print(f"\n  Theoretical A_continuum = M/(4 pi) = {A_continuum:.10f}")
print(f"  Ratio A_fit / A_continuum = {A / A_continuum:.6f}")

# Show profile vs theory
print(f"\n{'r':>6} {'delta_sim':>14} {'A_cont/r':>14} {'A_fit/r':>14} {'sim/cont':>10}")
for i in range(0, len(r_centers), 4):
    if mask_fit[i]:
        d_sim = profile[i]
        d_th = A_continuum / r_centers[i]
        d_fit = A / r_centers[i] + B
        print(f"{r_centers[i]:>6.2f} {d_sim:>14.8f} {d_th:>14.8f} {d_fit:>14.8f} {d_sim/d_th:>10.5f}")

# Convert to G_eff and compare to experimental precision
# In Newton form, delta = (G M) / r so A = G M
# Here A_fit gives "G_eff" times M_source
G_eff = A / M_source
G_eff_err = A_err / M_source
G_rel = G_eff_err / G_eff
print(f"\n  G_eff (lattice units) = A / M = {G_eff:.10f}")
print(f"  Relative precision: {G_rel:.3e}")

EXP = 2.2e-5
ratio = EXP / G_rel
print(f"\n  Experimental precision: {EXP:.1e}")
print(f"  Simulation precision:    {G_rel:.1e}")
if G_rel < EXP:
    print(f"  -> Simulation is {ratio:.0f}x MORE PRECISE than experiment")
else:
    print(f"  -> Simulation is {1/ratio:.1f}x LESS precise than experiment")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.errorbar(r_centers[mask_fit], profile[mask_fit], yerr=e_fit,
            fmt='o', ms=3, c='navy', label='simulation')
rs = np.linspace(2.5, 25, 100)
ax.plot(rs, newton(rs, A, B), 'r-', lw=1.5, label=f'fit $A/r$, $A={A:.5f}$')
ax.plot(rs, A_continuum/rs, 'g--', lw=1, label=f'continuum $1/(4\\pi r)$')
ax.set_xlabel('r (lattice units)')
ax.set_ylabel(r'$\delta(r)$')
ax.set_title(f'Discrete Poisson, A precision = {A_err/A:.2e}')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.loglog(r_centers[mask_fit], profile[mask_fit], 'o', ms=3, c='navy')
ax.loglog(rs, A/rs, 'r-', label='1/r fit')
ax.set_xlabel('r')
ax.set_ylabel(r'$\delta(r)$')
ax.set_title('log-log: 1/r scaling')
ax.legend()
ax.grid(alpha=0.3, which='both')

fig.suptitle(f'G measurement on DDD lattice (Poisson equation)', y=1.02)
fig.tight_layout()
fig.savefig(FIG / 'fig_G_poisson.png', dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved fig_G_poisson.png")

# Save
results = {
    "L": L, "N_ITER": N_ITER, "M_source": M_source,
    "A_fit": float(A), "A_err": float(A_err),
    "A_continuum_prediction": float(A_continuum),
    "ratio_fit_continuum": float(A / A_continuum),
    "G_eff": float(G_eff), "G_rel_precision": float(G_rel),
    "experimental_precision": EXP,
    "ratio_sim_vs_exp": float(EXP/G_rel) if G_rel > 0 else None,
    "verdict": (f"Simulation precision {G_rel:.1e}, "
                f"{'BETTER' if G_rel<EXP else 'worse'} than experiment ({EXP:.1e}) "
                f"by factor {EXP/G_rel:.1f}" if G_rel>0 else "FAIL"),
}
with open(DATA / "G_poisson.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved G_poisson.json")
