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
Paper II — Test 6: Path B — Geometric self-coupling via alpha_eff(chi)
========================================================================

Tests the conjecture that the lattice itself "deforms" under the deficit:
the diffusion coefficient is reduced in drained regions according to

    alpha_eff(i,j) = alpha * chi_avg^p,    chi_avg = (chi_i + chi_j) / 2

This implements "proper distances stretch in drained regions", analogous
to GR's spatial curvature. Two values of p tested:
    p = 0.5   (alpha_eff = alpha * sqrt(chi)) — user's specific suggestion
    p = 1.0   (alpha_eff = alpha * chi)        — natural proper-distance scaling

Analytic predictions (continuum stationary):
    delta(r) = 1 - (1 - (p+1)*A/r)^(1/(p+1))
    horizon at r_h = (p+1)*A

For p=1, this gives delta = 1 - sqrt(1 - 2A/r), so chi^2 = 1 - 2A/r,
i.e. exactly Schwarzschild g_tt structure with horizon at r=2A.

Outputs:
    data/pathB_results.json
    figures/fig07_pathB.pdf
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


def chi_field(R):
    return np.clip((R - R_MIN) / (R0 - R_MIN), 0.0, 1.0)


def step_pathB(R, E, p):
    """One tick with diffusion modified by chi^p."""
    chi = chi_field(R)
    delta = R0 - R
    drain = KAPPA * E * chi**2  # Original drain; no extra term

    flux_out = np.zeros_like(R)
    flux_in  = np.zeros_like(R)
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        chi_shifted = np.roll(chi, shift=(dx,dy,dz), axis=(0,1,2))
        delta_shifted = np.roll(delta, shift=(dx,dy,dz), axis=(0,1,2))
        # Effective alpha at link = alpha * (chi_avg)^p
        chi_avg = 0.5 * (chi + chi_shifted)
        alpha_eff = ALPHA * np.power(np.maximum(chi_avg, 1e-12), p)
        flux_out += np.minimum(F_MAX, alpha_eff * np.maximum(delta - delta_shifted, 0.0))
        flux_in  += np.minimum(F_MAX, alpha_eff * np.maximum(delta_shifted - delta, 0.0))
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


def run(p, E0=50.0):
    """Strong-field test: large E0 to push chi -> 0 near centre."""
    R = np.full((N_SIDE,)*3, R0)
    E = np.zeros_like(R)
    centre = (N_SIDE//2,)*3
    E[centre] = E0
    for _ in range(N_TICKS):
        R = step_pathB(R, E, p)
    return R0 - R, R


def fit_profiles(rs, delta_obs):
    from scipy.optimize import curve_fit

    def newton(r, A): return A / r
    def yukawa(r, A, mu): return A * np.exp(-mu*r) / r
    def schwarz(r, A): return A / np.maximum(r - A, 0.05)
    def pathB1(r, A): return 1.0 - np.sqrt(np.maximum(1 - 2*A/r, 1e-9))
    def pathB12(r, A): return 1.0 - np.power(np.maximum(1 - 1.5*A/r, 1e-9), 2.0/3.0)

    fits = {}
    A_init = max(delta_obs[0]*rs[0], 1e-4)
    for name, fn, p0, bounds in [
        ("newton", newton, [A_init], None),
        ("yukawa", yukawa, [A_init, 0.0], ([0,-1],[10*A_init,1])),
        ("schwarz", schwarz, [A_init], ([0],[min(rs)*0.99])),
        ("pathB1",  pathB1,  [A_init], ([0],[min(rs)*0.49])),  # horizon at 2A
        ("pathB12", pathB12, [A_init], ([0],[min(rs)*0.66])),  # horizon at 3A/2
    ]:
        try:
            if bounds:
                popt, _ = curve_fit(fn, rs, delta_obs, p0=p0, bounds=bounds)
            else:
                popt, _ = curve_fit(fn, rs, delta_obs, p0=p0)
            rss = float(np.sum((fn(rs,*popt)-delta_obs)**2))
            fits[name] = {"params": [float(x) for x in popt], "rss": rss}
        except Exception as e:
            fits[name] = {"error": str(e)}
    return fits


if __name__ == "__main__":
    print("Paper II — Test 6: Path B (alpha_eff = alpha * chi^p)")
    print("=" * 60)
    rs_eval = np.arange(2, 16).astype(float)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    results = {}

    # Test cases: p values
    cases = [(0.0, "Newton (baseline)"),
             (0.5, "p=1/2 (sqrt chi, user)"),
             (1.0, "p=1 (chi, Schwarzschild target)"),
             (2.0, "p=2 (chi^2, more aggressive)")]
    colors = ["C0", "C1", "C2", "C3"]

    for (p, label), color in zip(cases, colors):
        delta, R = run(p)
        prof = radial_profile(delta)
        delta_obs = prof[rs_eval.astype(int)]
        valid = delta_obs > 1e-7
        if valid.sum() >= 4:
            fits = fit_profiles(rs_eval[valid], delta_obs[valid])
        else:
            fits = {}
        if valid.sum() >= 3:
            slope, _ = np.polyfit(np.log(rs_eval[valid]), np.log(delta_obs[valid]), 1)
        else:
            slope = None
        results[f"p={p}"] = {
            "label":    label,
            "p":        p,
            "rs":       rs_eval.tolist(),
            "delta":    [float(x) for x in delta_obs],
            "exponent": float(slope) if slope is not None else None,
            "fits":     fits,
        }
        print(f"\np = {p} ({label}):")
        print(f"  exponent = {slope}")
        for name in ["newton", "yukawa", "schwarz", "pathB1", "pathB12"]:
            if name in fits and "rss" in fits[name]:
                pp = fits[name]["params"]
                print(f"  {name:<10}: params={pp}, RSS={fits[name]['rss']:.3e}")
        with_rss = {n: fits[n]["rss"] for n in fits if "rss" in fits[n]}
        if with_rss:
            best = min(with_rss, key=with_rss.get)
            print(f"  >>> BEST FIT: {best}")
        ax[0].loglog(rs_eval, np.maximum(delta_obs, 1e-9), "o-",
                     color=color, label=fr"{label}")

    # Theoretical curves
    rs_th = np.linspace(2, 15, 200)
    A_ref = 0.007
    ax[0].loglog(rs_th, A_ref / rs_th, "k--", lw=1.0, label="Newton $A/r$")
    ax[0].loglog(rs_th, A_ref / np.maximum(rs_th - A_ref, 0.05),
                 "k:", lw=1.0, label=r"Schwarzschild $A/(r-A)$")
    ax[0].loglog(rs_th, 1 - np.sqrt(np.maximum(1 - 2*A_ref/rs_th, 1e-9)),
                 "g-.", lw=1.0, label=r"PathB p=1: $1-\sqrt{1-2A/r}$")
    ax[0].set_xlabel(r"$r$")
    ax[0].set_ylabel(r"$\delta(r)$")
    ax[0].set_title(r"Path B: $\alpha_{\rm eff} = \alpha\,\chi^p$")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].grid(True, alpha=0.3, which="both")

    # Right: best-fit RSS for each model
    fit_names = ["newton", "yukawa", "schwarz", "pathB1", "pathB12"]
    fit_colors = {"newton":"steelblue", "yukawa":"orange", "schwarz":"crimson",
                  "pathB1":"green", "pathB12":"purple"}
    width = 0.16
    xs = np.arange(len(cases))
    for k, name in enumerate(fit_names):
        rsss = []
        for (p, _), _ in zip(cases, colors):
            f = results[f"p={p}"]["fits"]
            rsss.append(np.log10(max(f.get(name,{}).get("rss",1e10), 1e-12)))
        ax[1].bar(xs + (k-2)*width, rsss, width, label=name, color=fit_colors[name])
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"p={p}" for p,_ in cases])
    ax[1].set_ylabel(r"$\log_{10}(\rm RSS)$")
    ax[1].set_title("Best fit comparison")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path = FIG_DIR / "fig07_pathB.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf",".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")

    out_path = DATA_DIR / "pathB_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    print("\n" + "=" * 60)
    print("VERDICT — at each p, which form fits the simulated profile best?")
    print("=" * 60)
    for (p, label), _ in zip(cases, colors):
        with_rss = {n: results[f"p={p}"]["fits"][n]["rss"]
                    for n in fit_names
                    if "rss" in results[f"p={p}"]["fits"].get(n,{})}
        if with_rss:
            best = min(with_rss, key=with_rss.get)
            print(f"  p={p} ({label}): best={best:<10} (RSS={with_rss[best]:.2e})")
