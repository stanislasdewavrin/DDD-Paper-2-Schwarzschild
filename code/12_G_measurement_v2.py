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
Mesure haute precision de G_eff dans DDD - version 2
======================================================

V2: parametres ajustes pour que le profil 1/r emerge
proprement, avec convergence verifiable.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)

# Parametres ajustes pour que le profil emerge
KAPPA  = 0.05       # drainage (plus fort)
ALPHA  = 1.0        # diffusion (plus rapide)
F_MAX  = 0.5        # cap du flux (large)
R_0    = 1.0
R_MIN  = 0.001      # plus profond pour eviter saturation precoce
E_RATE = 1.0        # excitation maintenue au centre

L = 48
N_TICKS = 3000
SAMPLE_INTERVAL = 1000

print("=" * 70)
print("Mesure haute precision de G_eff - v2")
print("=" * 70)
print(f"Reseau {L}^3, kappa={KAPPA}, alpha={ALPHA}, F_max={F_MAX}")
print(f"R_0={R_0}, R_min={R_MIN}, E maintenu au centre")

R = np.full((L, L, L), R_0)
cx, cy, cz = L//2, L//2, L//2

def step(R):
    """One DDD tick."""
    chi = np.clip((R - R_MIN) / (R_0 - R_MIN), 0.0, 1.0)
    delta = R_0 - R

    # Drainage at center only (point source)
    D = np.zeros_like(R)
    D[cx, cy, cz] = KAPPA * E_RATE * chi[cx, cy, cz]**2

    # Flux propagation (vectorized)
    flux_in  = np.zeros_like(R)
    flux_out = np.zeros_like(R)
    for ax in range(3):
        for shift in [-1, +1]:
            d_n = np.roll(delta, shift=shift, axis=ax)
            f_out = np.minimum(F_MAX, ALPHA * np.maximum(delta - d_n, 0.0))
            f_in  = np.minimum(F_MAX, ALPHA * np.maximum(d_n - delta, 0.0))
            flux_out += f_out
            flux_in  += f_in

    R_new = R - D + flux_in - flux_out
    return np.clip(R_new, R_MIN, R_0)


print(f"\nIteration {N_TICKS} ticks...")
profile_history = []
for t in range(N_TICKS + 1):
    if t % SAMPLE_INTERVAL == 0:
        delta_max = R_0 - R[cx, cy, cz]
        delta_at5 = R_0 - R[cx+5, cy, cz]
        delta_at10 = R_0 - R[cx+10, cy, cz]
        delta_at20 = R_0 - R[cx+20, cy, cz] if cx+20 < L else 0
        print(f"  tick {t:5d}: delta_0={delta_max:.5f}, "
              f"delta_5={delta_at5:.5f}, delta_10={delta_at10:.5f}, "
              f"delta_20={delta_at20:.5f}")
    if t < N_TICKS:
        R = step(R)


# Extraction du profil radial
print("\nExtraction profil radial...")
delta = R_0 - R
xs = np.arange(L) - cx
ys = np.arange(L) - cy
zs = np.arange(L) - cz
X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
r = np.sqrt(X**2 + Y**2 + Z**2)

# Fine binning
r_bins = np.arange(0.5, 25, 0.25)
r_centers = (r_bins[:-1] + r_bins[1:]) / 2
profile = np.zeros(len(r_centers))
profile_err = np.zeros(len(r_centers))
counts = np.zeros(len(r_centers))
for i, (rmin, rmax) in enumerate(zip(r_bins[:-1], r_bins[1:])):
    mask = (r >= rmin) & (r < rmax)
    n = mask.sum()
    counts[i] = n
    if n > 0:
        profile[i] = delta[mask].mean()
        profile_err[i] = delta[mask].std() / np.sqrt(n) if n > 1 else 0

print(f"\n{'r':>6} {'count':>6} {'delta':>14}")
for i in range(0, min(30, len(r_centers)), 2):
    if counts[i] > 0:
        print(f"{r_centers[i]:>6.2f} {int(counts[i]):>6d} {profile[i]:>14.6e}")


# Fit Newton 1/r
def newton(r, A, B):
    return A / r + B

# Use bins r in [3, 18]
mask = (r_centers > 2.5) & (r_centers < 18.0) & (counts > 5)
r_fit = r_centers[mask]
d_fit = profile[mask]
e_fit = np.maximum(profile_err[mask], 1e-15)  # avoid zero errors

print(f"\nFit Newton on {mask.sum()} bins from r={r_fit.min():.2f} to r={r_fit.max():.2f}")
popt, pcov = curve_fit(newton, r_fit, d_fit, sigma=e_fit, absolute_sigma=True)
A_fit, B_fit = popt
A_err, B_err = np.sqrt(np.diag(pcov))

print(f"\n  A = {A_fit:.10f} +/- {A_err:.3e}")
print(f"  Relative precision A: {A_err/A_fit:.3e}")
print(f"  B = {B_fit:.6e} +/- {B_err:.3e}")


# Compute residuals
residuals = d_fit - newton(r_fit, A_fit, B_fit)
chi2 = np.sum((residuals / e_fit)**2) / len(r_fit)
print(f"  Chi2/dof = {chi2:.3e}")

# Convert to G_eff
M_eff = N_TICKS * KAPPA * E_RATE  # accumulated drainage
c_eff = F_MAX
G_eff = A_fit * c_eff**2 / (2 * M_eff)
G_eff_err = A_err * c_eff**2 / (2 * M_eff)
G_eff_rel = G_eff_err / G_eff
print(f"\n  M_eff = {M_eff}")
print(f"  c_eff = {c_eff}")
print(f"  G_eff = A * c^2 / (2 M) = {G_eff:.10f}")
print(f"  Relative precision G: {G_eff_rel:.3e}")


# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear plot
ax = axes[0]
ax.errorbar(r_fit, d_fit, yerr=e_fit, fmt='o', c='navy', ms=4, label='simulation')
r_smooth = np.linspace(r_fit.min(), r_fit.max(), 100)
ax.plot(r_smooth, newton(r_smooth, A_fit, B_fit), 'r-', lw=1.5,
        label=f'fit: A/r + B,  A={A_fit:.5f}')
ax.set_xlabel('r (lattice units)')
ax.set_ylabel(r'$\delta(r)$')
ax.set_title(f'Newton fit, precision A: {A_err/A_fit:.2e}')
ax.legend()
ax.grid(alpha=0.3)

# Log-log plot
ax = axes[1]
mask_pos = d_fit > 0
ax.loglog(r_fit[mask_pos], d_fit[mask_pos], 'o', c='navy', ms=4, label='simulation')
ax.loglog(r_smooth, newton(r_smooth, A_fit, 0)*0.95, 'r--', label='1/r')
ax.set_xlabel('r')
ax.set_ylabel(r'$\delta(r)$')
ax.set_title('log-log: scaling 1/r')
ax.legend()
ax.grid(alpha=0.3, which='both')

fig.suptitle(f'DDD drainage profile - measurement of G_eff', y=1.02)
fig.tight_layout()
fig.savefig(FIG / 'fig_G_measurement_v2.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_G_measurement_v2.png', dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: fig_G_measurement_v2.png")


# ============================================================
# Comparison with experimental precision
# ============================================================
EXP_PRECISION = 2.2e-5  # CODATA 2018

print("\n" + "=" * 70)
print("VERDICT: Precision intra-simulation vs experimental")
print("=" * 70)
print(f"""
Simulation (cette mesure):
    G_eff = {G_eff:.10f}
    Precision relative = {G_eff_rel:.3e}

Experimental (CODATA 2018):
    Precision relative = {EXP_PRECISION:.1e}

Ratio simulation/experiment = {EXP_PRECISION/G_eff_rel:.1f}x
""")

if G_eff_rel < EXP_PRECISION:
    factor = EXP_PRECISION / G_eff_rel
    print(f"  -> La SIMULATION DDD est {factor:.0f}x plus precise!")
    print(f"     Cela confirme que la simulation peut servir d'oracle")
    print(f"     de precision pour G, conditionnel a la validation de DDD.")
else:
    factor = G_eff_rel / EXP_PRECISION
    print(f"  -> La simulation est {factor:.1f}x moins precise.")
    print(f"     Pour ameliorer: augmenter L (1/L^2), N_ticks (1/N), ou")
    print(f"     utiliser methode de Greens function/multipole.")

# Save
results = {
    "L_lattice":     L,
    "N_ticks":       N_TICKS,
    "kappa":         KAPPA,
    "alpha":         ALPHA,
    "F_max":         F_MAX,
    "A_fit":         float(A_fit),
    "A_err":         float(A_err),
    "A_rel_precision": float(A_err/A_fit),
    "G_eff":         float(G_eff),
    "G_eff_rel_precision": float(G_eff_rel),
    "chi2_per_dof":  float(chi2),
    "experimental_precision": EXP_PRECISION,
    "ratio_sim_vs_exp": float(EXP_PRECISION / G_eff_rel) if G_eff_rel > 0 else None,
}
with open(DATA / "G_measurement_v2.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {DATA / 'G_measurement_v2.json'}")
