"""
Paper II — Test 3: Photon deflection by geometric ray tracing
===============================================================

Integrates the eikonal equation
    dk/ds = grad(ln n),   n(r) = 1/sqrt(chi(r)),   chi(r) = 1 - A/r
on the analytic radial profile (using the simulation result that the
profile is A/r in the stationary regime, validated in Test 1).

Compares the simulated geometric deflection with the prediction
Delta_theta_geom = A/b. Notes that the full GR deflection is 2A/b,
and that the missing factor 2 corresponds to the Shapiro delay which
is not captured by geometric ray tracing.

Outputs:
    data/photon_deflection.json
    figures/fig03_photon_deflection.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DS = 0.1
X_START = -500.0
X_END   = +500.0
CHI_MIN = 1e-3

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def grad_log_n(x, y, A, identification="full_GR"):
    """grad(ln n) for two photon-speed identifications:

    identification = "scalar":   n = 1/sqrt(chi)   (Paper II initial -> alpha=1, half GR)
    identification = "full_GR":  n = 1/chi         (corrected -> alpha=2, full GR)
    """
    r = np.sqrt(x*x + y*y)
    if r < 1e-6:
        return np.array([0.0, 0.0])
    chi = max(CHI_MIN, 1.0 - A / r)
    # grad chi = A * (x/r^3, y/r^3)
    if identification == "scalar":
        # ln n = -0.5 ln chi -> grad ln n = -0.5/chi * grad chi
        factor = -0.5 / chi * (A / r**3)
    else:  # full_GR
        # ln n = -ln chi -> grad ln n = -1/chi * grad chi
        factor = -1.0 / chi * (A / r**3)
    return np.array([factor * x, factor * y])


def trace_ray(A, b, identification="full_GR"):
    """Trace one ray with impact parameter b; return total deflection (rad)."""
    x = X_START
    y = b
    kx = 1.0
    ky = 0.0
    n_steps = 0
    while x < X_END and n_steps < 20000:
        g = grad_log_n(x, y, A, identification)
        kx += g[0] * DS
        ky += g[1] * DS
        norm = np.sqrt(kx*kx + ky*ky)
        kx /= norm
        ky /= norm
        x += kx * DS
        y += ky * DS
        n_steps += 1
    deflection = float(np.arctan2(ky, kx) - np.arctan2(0.0, 1.0))
    return abs(deflection), n_steps


if __name__ == "__main__":
    print("Paper II — Test 3: photon deflection by geometric ray tracing")
    print("=" * 60)

    cases = []
    print("=== n = 1/chi (full GR identification, alpha = 2) ===")
    print(f"{'A':>6} {'b':>5} {'sim':>10} {'A/b':>10} {'2A/b':>10} {'dev2A/b':>9} {'ticks':>8}")
    for A in [0.010, 0.050, 0.100, 0.200, 0.500]:
        for b in [5.0, 10.0, 20.0, 50.0]:
            theta, ticks = trace_ray(A, b, identification="full_GR")
            theta_geo = A / b
            theta_full = 2 * A / b
            dev_to_full_GR = abs(theta - theta_full) / theta_full * 100.0
            cases.append({
                "A":            float(A),
                "b":            float(b),
                "theta_sim":    theta,
                "theta_geo":    theta_geo,
                "theta_full_GR": theta_full,
                "deviation_pct_vs_full_GR": dev_to_full_GR,
                "ticks":        ticks,
            })
            print(f"{A:>6.3f} {b:>5.1f} {theta:>10.5f} {theta_geo:>10.5f} {theta_full:>10.5f} {dev_to_full_GR:>9.2f} {ticks:>8}")

    # Power-law fit at A=0.05
    bs = np.array([c["b"] for c in cases if c["A"] == 0.05])
    ths = np.array([c["theta_sim"] for c in cases if c["A"] == 0.05])
    log_b = np.log(bs)
    log_t = np.log(ths)
    slope, intercept = np.polyfit(log_b, log_t, 1)
    print(f"\nPower-law fit at A=0.05: theta ~ b^{slope:.4f} (analytic: -1.0)")

    # Additional fit on a broader range with more b values
    bs_extra = [2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 50.0]
    th_extra = []
    for b in bs_extra:
        theta, _ = trace_ray(0.05, b)
        th_extra.append(theta)
    log_b_ex = np.log(np.array(bs_extra))
    log_t_ex = np.log(np.array(th_extra))
    slope_ex, _ = np.polyfit(log_b_ex, log_t_ex, 1)
    print(f"Power-law fit on extended range: theta ~ b^{slope_ex:.4f}")

    out = {"cases": cases, "fit_at_A0.05": {"slope": float(slope), "slope_extended": float(slope_ex)}}
    out_path = DATA_DIR / "photon_deflection.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # Left: deflection vs b at A=0.05, log-log
    ax[0].loglog(bs_extra, th_extra, "ro", markersize=8, label="DDD ray-tracing")
    bs_th = np.linspace(2, 50, 100)
    ax[0].loglog(bs_th, 0.05 / bs_th,     "k--", lw=1.5, label=r"$A/b$ (geometric)")
    ax[0].loglog(bs_th, 2 * 0.05 / bs_th, "b:",  lw=1.5, label=r"$2A/b$ (full GR)")
    ax[0].set_xlabel(r"$b$")
    ax[0].set_ylabel(r"$\Delta\theta$")
    ax[0].set_title(f"$A=0.05$: $\\theta \\sim b^{{{slope_ex:.3f}}}$")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3, which="both")

    # Right: deviation from A/b for several A
    A_values = sorted({c["A"] for c in cases})
    for A in A_values:
        bs_A    = [c["b"] for c in cases if c["A"] == A]
        devs_A  = [c["deviation_pct"] for c in cases if c["A"] == A]
        ax[1].semilogy(bs_A, np.maximum(devs_A, 1e-3), "o-", label=f"$A={A:.3f}$")
    ax[1].set_xlabel(r"$b$")
    ax[1].set_ylabel(r"$|\theta_{\rm sim} - A/b|/(A/b)$  (%)")
    ax[1].set_title("Deviation from geometric prediction")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig_path = FIG_DIR / "fig03_photon_deflection.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
