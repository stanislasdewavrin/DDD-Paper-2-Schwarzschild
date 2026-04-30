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
Paper II — Test 5: Quadratic non-linear feedback beta*delta^2
================================================================

Tests the conjecture that the auto-trapped energy circulation pattern
folds back on itself with a "cross-section" proportional to its own
intensity, giving a drain contribution proportional to delta^2:

    drain_i = kappa * E_i * chi_i^2 + beta * delta_i^2

The hypothesis to falsify or validate: does this quadratic non-linearity
yield the Schwarzschild profile delta(r) = A/(r-A) preferentially over
Yukawa or Newton?

Outputs:
    data/quadratic_results.json
    figures/fig06_quadratic.pdf
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


def step(R, E, beta):
    """One tick with QUADRATIC self-feedback beta*delta^2."""
    chi = np.clip((R - R_MIN) / (R0 - R_MIN), 0.0, 1.0)
    delta = R0 - R
    drain = KAPPA * E * chi**2 + beta * delta**2

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
    return R0 - R


def fit_profiles(rs, delta_obs):
    from scipy.optimize import curve_fit

    def newton(r, A): return A / r
    def yukawa(r, A, mu): return A * np.exp(-mu*r) / r
    def schwarz(r, A): return A / np.maximum(r - A, 0.05)

    fits = {}
    A_init = delta_obs[0]*rs[0]
    try:
        popt, _ = curve_fit(newton, rs, delta_obs, p0=[A_init])
        fits["newton"] = {"A": float(popt[0]),
                          "rss": float(np.sum((newton(rs,*popt)-delta_obs)**2))}
    except Exception as e:
        fits["newton"] = {"error": str(e)}
    try:
        popt, _ = curve_fit(yukawa, rs, delta_obs,
                            p0=[A_init, 0.0], bounds=([0,-1],[10*A_init,1]))
        fits["yukawa"] = {"A": float(popt[0]), "mu": float(popt[1]),
                          "rss": float(np.sum((yukawa(rs,*popt)-delta_obs)**2))}
    except Exception as e:
        fits["yukawa"] = {"error": str(e)}
    try:
        popt, _ = curve_fit(schwarz, rs, delta_obs, p0=[A_init],
                            bounds=([0],[min(rs)*0.99]))
        fits["schwarz"] = {"A": float(popt[0]),
                           "rss": float(np.sum((schwarz(rs,*popt)-delta_obs)**2))}
    except Exception as e:
        fits["schwarz"] = {"error": str(e)}
    return fits


if __name__ == "__main__":
    print("Paper II — Test 5: QUADRATIC feedback beta*delta^2")
    print("=" * 60)
    rs_eval = np.arange(2, 16).astype(float)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    results = {}

    # Run several beta — quadratic is much weaker so use larger absolute beta
    betas = [0.0, 0.5, 1.0, 5.0, 20.0, -1.0]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    for beta, color in zip(betas, colors):
        delta = run(beta)
        prof = radial_profile(delta)
        delta_obs = prof[rs_eval.astype(int)]
        valid = delta_obs > 1e-7
        if valid.sum() >= 4:
            fits = fit_profiles(rs_eval[valid], delta_obs[valid])
        else:
            fits = {}
        # Power-law exponent
        if valid.sum() >= 3:
            slope, _ = np.polyfit(np.log(rs_eval[valid]), np.log(delta_obs[valid]), 1)
        else:
            slope = None
        results[f"beta={beta}"] = {
            "rs":       rs_eval.tolist(),
            "delta":    [float(x) for x in delta_obs],
            "exponent": float(slope) if slope is not None else None,
            "fits":     fits,
        }
        print(f"\nbeta={beta:>6.2f}:")
        print(f"  exponent = {slope}")
        for name in ["newton", "yukawa", "schwarz"]:
            if name in fits and "rss" in fits[name]:
                if name == "yukawa":
                    print(f"  {name:<12}: A={fits[name]['A']:.5f}, "
                          f"mu={fits[name]['mu']:+.5f}, RSS={fits[name]['rss']:.2e}")
                else:
                    print(f"  {name:<12}: A={fits[name]['A']:.5f}, "
                          f"RSS={fits[name]['rss']:.2e}")
        # Identify best fit by RSS
        best = min((n for n in fits if "rss" in fits[n]),
                   key=lambda n: fits[n]["rss"], default=None)
        print(f"  Best fit: {best}")
        ax[0].loglog(rs_eval, np.maximum(delta_obs, 1e-9), "o-",
                     color=color, label=fr"$\beta={beta}$")

    rs_th = np.linspace(2, 15, 200)
    A_ref = results["beta=0.0"]["fits"].get("newton", {}).get("A", 0.007)
    ax[0].loglog(rs_th, A_ref / rs_th, "k--", lw=1.5, label="Newton $A/r$")
    ax[0].loglog(rs_th, A_ref / np.maximum(rs_th - A_ref, 0.1), "k:", lw=1.5,
                 label="Schwarzschild $A/(r-A)$")
    ax[0].set_xlabel(r"$r$")
    ax[0].set_ylabel(r"$\delta(r)$")
    ax[0].set_title(r"Profile under quadratic feedback $+\beta\delta^2$")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].grid(True, alpha=0.3, which="both")

    # Right panel: which form fits best at each beta
    ax[1].set_title("Best fit quality (lower RSS = better)")
    fit_names = ["newton", "yukawa", "schwarz"]
    fit_colors = {"newton": "blue", "yukawa": "orange", "schwarz": "red"}
    width = 0.25
    xs = np.arange(len(betas))
    for k, name in enumerate(fit_names):
        rsss = []
        for beta in betas:
            f = results[f"beta={beta}"]["fits"]
            rsss.append(np.log10(max(f.get(name,{}).get("rss",1e10), 1e-12)))
        ax[1].bar(xs + (k-1)*width, rsss, width, label=name, color=fit_colors[name])
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([fr"$\beta={b}$" for b in betas], rotation=30)
    ax[1].set_ylabel(r"$\log_{10}(\rm RSS)$")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path = FIG_DIR / "fig06_quadratic.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf",".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")

    out_path = DATA_DIR / "quadratic_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    # Summary verdict
    print("\n" + "=" * 60)
    print("VERDICT: which form fits best as a function of beta?")
    print("=" * 60)
    for beta in betas:
        fits = results[f"beta={beta}"]["fits"]
        with_rss = {n: fits[n]["rss"] for n in ["newton","yukawa","schwarz"]
                    if "rss" in fits.get(n,{})}
        if with_rss:
            best = min(with_rss, key=with_rss.get)
            ratio = (with_rss.get("schwarz", 1e10) /
                     min(with_rss.get("newton", 1e10), with_rss.get("yukawa", 1e10)))
            print(f"  beta={beta:>5.1f}: best={best:<10} | RSS_schwarz/RSS_min = {ratio:.3f}")
