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
Paper II — Test 1: Newtonian profile delta(r) ~ A/r from local drainage
=========================================================================

Iterates the local rule on a 41^3 cubic 6-connected lattice with a
constant central source for 2000 ticks, measures the stationary radial
profile of the deficit, and fits a power law on r in [2, 15].

Outputs:
    data/radial_profile.json
    figures/fig01_radial_profile.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

R0      = 1.0
R_MIN   = 0.005
KAPPA   = 0.015
ALPHA   = 0.15
F_MAX   = 0.02
N_SIDE  = 41
N_TICKS = 2000
SEED    = 2024
np.random.seed(SEED)

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def step(R, E):
    chi = np.clip((R - R_MIN) / (R0 - R_MIN), 0.0, 1.0)
    delta = R0 - R
    drain = KAPPA * E * chi**2

    flux_out = np.zeros_like(R)
    flux_in  = np.zeros_like(R)
    shifts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for dx, dy, dz in shifts:
        shifted = np.roll(delta, shift=(dx, dy, dz), axis=(0, 1, 2))
        f_out = np.minimum(F_MAX, ALPHA * np.maximum(delta - shifted, 0.0))
        f_in  = np.minimum(F_MAX, ALPHA * np.maximum(shifted - delta, 0.0))
        flux_out += f_out
        flux_in  += f_in
    R_next = R - drain - flux_in + flux_out
    R_clip = np.minimum(R0, np.maximum(R_MIN, R_next))
    return R_clip


def radial_profile(field):
    """Compute radial average of field on a cubic grid centered at the centre."""
    n = field.shape[0]
    cx = cy = cz = n // 2
    coords = np.indices(field.shape) - np.array([[cx], [cy], [cz]]).reshape(3, 1, 1, 1)
    r = np.sqrt(np.sum(coords**2, axis=0))
    r_int = np.round(r).astype(int)
    r_max = r_int.max()
    profile = np.zeros(r_max + 1)
    counts  = np.zeros(r_max + 1)
    for r_val in range(r_max + 1):
        mask = (r_int == r_val)
        if mask.any():
            profile[r_val] = field[mask].mean()
            counts[r_val] = mask.sum()
    return profile, counts


def run(E0):
    R = np.full((N_SIDE,) * 3, R0, dtype=np.float64)
    E = np.zeros_like(R)
    centre = (N_SIDE // 2,) * 3
    E[centre] = E0
    for t in range(N_TICKS):
        R = step(R, E)
    delta = R0 - R
    return delta


if __name__ == "__main__":
    print("Paper II — Test 1: radial profile delta(r) ~ A/r")
    print("=" * 60)
    results = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for E0 in [0.2, 0.5, 1.0, 2.0]:
        delta = run(E0)
        prof, _ = radial_profile(delta)
        # Calibrate A at r=3
        A = float(prof[3] * 3.0)
        rs = np.arange(1, 16)
        delta_sim = prof[rs]
        delta_th = A / rs
        deviations = (delta_sim - delta_th) / delta_th * 100.0
        # Power-law fit on r in [2, 15]
        log_r = np.log(rs[1:])
        log_d = np.log(delta_sim[1:] + 1e-30)
        slope, intercept = np.polyfit(log_r, log_d, 1)
        results[f"E0={E0}"] = {
            "A_calibrated":   A,
            "exponent":       float(slope),
            "delta_sim_per_r": [float(x) for x in delta_sim],
            "delta_th_per_r":  [float(x) for x in delta_th],
            "deviation_pct":   [float(x) for x in deviations],
        }
        print(f"E0={E0}: A={A:.4f}, exponent={slope:.4f}")
        for r, dsim, dth, dev in zip(rs, delta_sim, delta_th, deviations):
            print(f"  r={r:>2}: sim={dsim:.6f}, th={dth:.6f}, dev={dev:+.2f}%")
        ax[0].loglog(rs, delta_sim, "o-", label=f"$E_0={E0}$")
        ax[1].plot(rs, deviations, "o-", label=f"$E_0={E0}$")

    rs_th = np.linspace(1, 15, 100)
    A_ref = results["E0=1.0"]["A_calibrated"]
    ax[0].loglog(rs_th, A_ref / rs_th, "k--", lw=1, label=r"$A/r$ (calibrated)")
    ax[0].set_xlabel(r"$r$")
    ax[0].set_ylabel(r"$\delta(r)$")
    ax[0].set_title("Radial profile (log-log)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_xlabel(r"$r$")
    ax[1].set_ylabel(r"$(\delta_{\rm sim} - A/r)/(A/r)$  (%)")
    ax[1].set_title(r"Deviation from analytic $A/r$")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = DATA_DIR / "radial_profile.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    fig_path = FIG_DIR / "fig01_radial_profile.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    print(f"Saved: {fig_path}")
