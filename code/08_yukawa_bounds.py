"""
Paper II — Yukawa bounds from Eöt-Wash precision experiments
==============================================================

The non-linear feedback paper.tex sect 4 predicts a Yukawa correction
to Newtonian gravity for beta < 0:

    V(r) = -(G m1 m2 / r) * [1 + alpha_Y * exp(-r/lambda)]
    with lambda = sqrt(alpha / |beta|)

Eöt-Wash precision tests (Adelberger 2009) constrain alpha_Y as a
function of lambda. We compute the allowed range of beta/alpha given
these constraints.

Outputs:
    data/yukawa_bounds.json
    figures/fig07_yukawa_bounds.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"


# Adelberger 2009 Eöt-Wash bounds (approximate, |alpha_Y| 95% CL)
# data points from the published exclusion plot, schematic
lambda_bounds_m = np.array([
    1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 100, 1e3, 1e4, 1e5
])
alpha_Y_max = np.array([
    1e2, 1e0, 1e-1, 1e-2, 5e-3, 5e-4, 1e-3, 1e-2, 1e-1, 1.0, 10
])

# In DDD, alpha_Y ~ 1 (full gravitational strength of Yukawa correction).
# Eöt-Wash forbids alpha_Y > alpha_Y_max(lambda) at each lambda.
# Therefore lambda must lie OUTSIDE the constrained range, i.e. either
# lambda < ~10 mu m (below experimental sensitivity) OR lambda > ~10^5 m.

# In lattice units, lambda_sim = sqrt(alpha / |beta|) lattice spacings.
# Physical lambda = lambda_sim * d_min, with d_min = Planck length.
ell_P = 1.616e-35  # m

# Forbidden region: 10 mu m < lambda < 10^5 m corresponds to
# 6.2e29 < lambda_sim < 6.2e39 lattice units
print("Yukawa bounds for DDD framework")
print("=" * 60)
print(f"Eöt-Wash constrains physical Yukawa scale lambda for alpha_Y ~ 1.")
print(f"Forbidden range (in physical units): ~10 mu m to ~10^5 m")
print()
print(f"In lattice units (d_min = l_P = {ell_P:.2e} m):")
print(f"  lambda_sim = lambda_phys / l_P")
print(f"  Forbidden: {1e-5/ell_P:.2e} < lambda_sim < {1e5/ell_P:.2e}")
print()
print(f"Constraint on (alpha / |beta|)_lattice:")
print(f"  Allowed: alpha/|beta| < ({1e-5/ell_P:.2e})^2 = {(1e-5/ell_P)**2:.2e}")
print(f"        OR alpha/|beta| > ({1e5/ell_P:.2e})^2 = {(1e5/ell_P)**2:.2e}")
print()
# In typical DDD parameters (kappa=0.015, alpha=0.15), if beta were
# similar magnitude (say |beta| = 0.01), alpha/|beta| = 15.
# This corresponds to lambda_sim = sqrt(15) ~ 4 lattice units = 4 * l_P
# = 6.4e-35 m -- well below the experimental range 10 mu m to 10^5 m.
# So our typical DDD parameters give lambda below sensitivity = OK.
beta_test = 0.01
alpha_lat = 0.15
lambda_sim = np.sqrt(alpha_lat / beta_test)
lambda_phys = lambda_sim * ell_P
print(f"Test case: beta_lat = {beta_test}, alpha_lat = {alpha_lat}")
print(f"  lambda_sim = {lambda_sim:.2f} lattice units")
print(f"  lambda_phys = {lambda_phys:.2e} m (well below 10 mu m)")
print(f"  -- Allowed by Eöt-Wash")

results = {
    "ell_planck_m":            ell_P,
    "forbidden_lambda_phys":   [1e-5, 1e5],
    "allowed_alpha_over_beta": "alpha/beta < " + f"{(1e-5/ell_P)**2:.2e}" + " or > " + f"{(1e5/ell_P)**2:.2e}",
    "test_case":               {"beta": beta_test, "alpha": alpha_lat,
                                "lambda_sim": float(lambda_sim),
                                "lambda_phys_m": float(lambda_phys),
                                "status":   "allowed"},
    "lambda_bounds_m":         lambda_bounds_m.tolist(),
    "alpha_Y_max":             alpha_Y_max.tolist(),
}
with open(DATA_DIR / "yukawa_bounds.json", "w") as f:
    json.dump(results, f, indent=2)

# Figure
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.loglog(lambda_bounds_m, alpha_Y_max, "ko-", markersize=8,
          label="Eöt-Wash 95\\% CL (Adelberger 2009)")
ax.fill_between(lambda_bounds_m, alpha_Y_max, 1e3,
                color="red", alpha=0.2, label="excluded")
ax.axhline(1.0, color="blue", lw=1.5, ls="--",
           label="DDD prediction $\\alpha_Y \\sim 1$")
# Mark the allowed regions for DDD: lambda << 10 µm or lambda >> 1e5 m
ax.axvspan(1e-9, 1e-5, alpha=0.15, color="green", label="DDD allowed (sub-µm)")
ax.axvspan(1e5, 1e8, alpha=0.15, color="green")
ax.set_xlabel(r"Yukawa range $\lambda$ (m)")
ax.set_ylabel(r"Yukawa coupling $|\alpha_Y|$")
ax.set_title("Yukawa-correction constraints on DDD\n"
             "DDD predicts $\\alpha_Y \\sim 1$; allowed only for $\\lambda$ outside experimental range")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(1e-4, 1e3)
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout()
fig_path = FIG_DIR / "fig07_yukawa_bounds.pdf"
fig.savefig(fig_path, bbox_inches="tight")
fig.savefig(str(fig_path).replace(".pdf",".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {fig_path}")
