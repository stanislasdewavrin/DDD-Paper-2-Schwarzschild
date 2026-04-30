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
Paper II — Test 4: Non-linear self-consistent regime with feedback beta*delta
==============================================================================

Tests the user's ontological hypothesis: each drained point in the
lattice contributes additional drainage proportional to its own deficit.
The local rule becomes:

    drain_i = kappa * E_i * chi_i^2 + beta * delta_i

where the new term beta*delta_i implements "deficit feeds back on itself"
(every cell with non-zero deficit acts as an additional source of drain
at rate beta*delta).

Three candidate profiles are tested:
    - Newtonian (linear regime):   delta(r) = A/r
    - Yukawa (linear feedback):    delta(r) = A * exp(-kappa*r) / r
    - Schwarzschild (full GR):     delta(r) = A/(r - A)

Outputs:
    data/feedback_results.json
    figures/fig05_feedback.pdf
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
N_TICKS = 1500
SEED    = 2024
np.random.seed(SEED)

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def step(R, E, beta):
    """One tick with self-feedback beta*delta added to the drain."""
    chi = np.clip((R - R_MIN) / (R0 - R_MIN), 0.0, 1.0)
    delta = R0 - R
    # Extended drain: original + feedback proportional to local deficit
    drain = KAPPA * E * chi**2 + beta * delta

    flux_out = np.zeros_like(R)
    flux_in  = np.zeros_like(R)
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        shifted = np.roll(delta, shift=(dx,dy,dz), axis=(0,1,2))
        flux_out += np.minimum(F_MAX, ALPHA * np.maximum(delta - shifted, 0.0))
        flux_in  += np.minimum(F_MAX, ALPHA * np.maximum(shifted - delta, 0.0))
    return np.minimum(R0, np.maximum(R_MIN, R - drain - flux_in + flux_out))


def radial_profile(field):
    n = field.shape[0]
    cx = cy = cz = n // 2
    coords = np.indices(field.shape) - np.array([[cx],[cy],[cz]]).reshape(3,1,1,1)
    r = np.sqrt(np.sum(coords**2, axis=0))
    r_int = np.round(r).astype(int)
    r_max = r_int.max()
    profile = np.zeros(r_max+1)
    for r_val in range(r_max+1):
        mask = (r_int == r_val)
        if mask.any():
            profile[r_val] = field[mask].mean()
    return profile


def run(beta, E0=1.0):
    R = np.full((N_SIDE,)*3, R0)
    E = np.zeros_like(R)
    centre = (N_SIDE//2,)*3
    E[centre] = E0
    for _ in range(N_TICKS):
        R = step(R, E, beta)
    return R0 - R   # delta


def fit_profiles(rs, delta_obs):
    """Fit observed profile to Newtonian, Yukawa, Schwarzschild forms."""
    from scipy.optimize import curve_fit

    def newton(r, A): return A / r
    def yukawa(r, A, kappa): return A * np.exp(-kappa*r) / r
    def schwarz(r, A): return A / np.maximum(r - A, 0.05)

    fits = {}
    try:
        popt, _ = curve_fit(newton, rs, delta_obs, p0=[delta_obs[0]*rs[0]])
        residual = np.sum((newton(rs, *popt) - delta_obs)**2)
        fits["newton"] = {"A": float(popt[0]), "rss": float(residual)}
    except Exception as e:
        fits["newton"] = {"error": str(e)}

    try:
        popt, _ = curve_fit(yukawa, rs, delta_obs,
                            p0=[delta_obs[0]*rs[0], 0.1],
                            bounds=([0, -1.0], [10*delta_obs[0]*rs[0], 1.0]))
        residual = np.sum((yukawa(rs, *popt) - delta_obs)**2)
        fits["yukawa"] = {"A": float(popt[0]), "kappa": float(popt[1]),
                          "rss": float(residual)}
    except Exception as e:
        fits["yukawa"] = {"error": str(e)}

    try:
        popt, _ = curve_fit(schwarz, rs, delta_obs, p0=[delta_obs[0]*rs[0]])
        residual = np.sum((schwarz(rs, *popt) - delta_obs)**2)
        fits["schwarz"] = {"A": float(popt[0]), "rss": float(residual)}
    except Exception as e:
        fits["schwarz"] = {"error": str(e)}

    return fits


if __name__ == "__main__":
    print("Paper II — Test 4: non-linear feedback beta*delta")
    print("=" * 60)

    rs_eval = np.arange(2, 16)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    results = {}

    betas = [0.0, 0.001, 0.005, 0.010, -0.005]
    colors = ["C0", "C1", "C2", "C3", "C4"]

    for beta, color in zip(betas, colors):
        delta = run(beta)
        prof = radial_profile(delta)
        delta_obs = prof[rs_eval]
        # Avoid zero or negative values for the fit
        valid = delta_obs > 1e-6
        if valid.sum() >= 4:
            fits = fit_profiles(rs_eval[valid].astype(float), delta_obs[valid])
        else:
            fits = {}
        results[f"beta={beta}"] = {
            "rs":     rs_eval.tolist(),
            "delta":  [float(x) for x in delta_obs],
            "fits":   fits,
        }
        # Power-law exponent fit
        if valid.sum() >= 3:
            logs = np.polyfit(np.log(rs_eval[valid].astype(float)),
                              np.log(delta_obs[valid]), 1)
            results[f"beta={beta}"]["exponent"] = float(logs[0])
        ax[0].loglog(rs_eval, np.maximum(delta_obs, 1e-9), "o-",
                     color=color, label=fr"$\beta={beta}$")

        # Print summary
        e = results[f"beta={beta}"].get("exponent", None)
        print(f"\nbeta={beta:.4f}:")
        print(f"  exponent = {e}")
        if "newton" in fits and "A" in fits["newton"]:
            print(f"  Newton:        A={fits['newton']['A']:.5f}, RSS={fits['newton']['rss']:.2e}")
        if "yukawa" in fits and "A" in fits["yukawa"]:
            print(f"  Yukawa:        A={fits['yukawa']['A']:.5f}, "
                  f"kappa={fits['yukawa']['kappa']:.5f}, RSS={fits['yukawa']['rss']:.2e}")
        if "schwarz" in fits and "A" in fits["schwarz"]:
            print(f"  Schwarzschild: A={fits['schwarz']['A']:.5f}, RSS={fits['schwarz']['rss']:.2e}")

    # Theoretical curves at A = 0.007 (rough scale)
    rs_th = np.linspace(2, 15, 100)
    A_ref = 0.007
    ax[0].loglog(rs_th, A_ref / rs_th, "k--", lw=1.5, label="Newton $A/r$")
    ax[0].loglog(rs_th, A_ref / np.maximum(rs_th - A_ref, 0.1), "k:", lw=1.5,
                 label="Schwarzschild $A/(r-A)$")
    ax[0].set_xlabel(r"$r$")
    ax[0].set_ylabel(r"$\delta(r)$")
    ax[0].set_title(r"Profile under self-feedback drain $+ \beta \delta$")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].grid(True, alpha=0.3, which="both")

    # Right panel: ratio sim / Newton for each beta (highlights deviation)
    for beta, color in zip(betas, colors):
        delta_obs = np.array(results[f"beta={beta}"]["delta"])
        valid = delta_obs > 1e-6
        if valid.sum() == 0:
            continue
        if "newton" in results[f"beta={beta}"]["fits"]:
            A = results[f"beta={beta}"]["fits"]["newton"].get("A", None)
            if A is not None:
                ax[1].plot(rs_eval[valid], delta_obs[valid] / (A/rs_eval[valid]),
                           "o-", color=color, label=fr"$\beta={beta}$")
    ax[1].axhline(1.0, color="grey", lw=0.5, ls="--")
    ax[1].set_xlabel(r"$r$")
    ax[1].set_ylabel(r"$\delta_{\rm sim}(r) / (A/r)$")
    ax[1].set_title("Ratio of simulated profile to Newtonian (calibrated A)")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = FIG_DIR / "fig05_feedback.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf",".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")

    out_path = DATA_DIR / "feedback_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")
