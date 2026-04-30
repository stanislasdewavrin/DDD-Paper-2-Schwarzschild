"""
Paper II — Test 2: Gravitational and kinematic time dilation
==============================================================

Three checks:

(2a) Pure gravitational dilation: at rest (phi=0) on chi(r) = 1 - A/r,
     dtau/dt = sqrt(chi). Identical to GR weak field by construction
     of the calibration A = 2GM/c^2.

(2b) Pure kinematic dilation (chi=1): dtau/dt = cos(phi) = sqrt(1-beta^2),
     identical to SR.

(2c) Combined: DDD product form vs GR additive expansion. Tabulate
     the deviation across (beta, r) grid for A=0.20.

Outputs:
    data/time_dilation.json
    figures/fig02_time_dilation.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def chi_of_r(r, A):
    return 1.0 - A / r


def dtau_DDD(beta, r, A):
    """Product form: sqrt(chi) * sqrt(1-beta^2)"""
    chi = chi_of_r(r, A)
    return np.sqrt(chi) * np.sqrt(1.0 - beta**2)


def dtau_GR_additive(beta, r, A):
    """GR weak-field expansion: sqrt(1 - A/r - beta^2)"""
    return np.sqrt(np.maximum(0.0, 1.0 - A / r - beta**2))


def test_2a():
    """Gravitational only."""
    rows = []
    for A in [0.10, 0.30, 0.60]:
        for r in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
            chi = chi_of_r(r, A)
            if chi <= 0:
                continue
            dtau_trame = float(np.sqrt(chi))
            dtau_gr    = float(np.sqrt(chi))  # by calibration: equal
            rows.append({
                "A":         A,
                "r":         r,
                "chi":       float(chi),
                "dtau_DDD":  dtau_trame,
                "dtau_GR":   dtau_gr,
                "deviation": float(abs(dtau_trame - dtau_gr)),
            })
    return rows


def test_2b():
    """Kinematic only at chi=1."""
    rows = []
    for beta in [0.0, 0.1, 0.3, 0.5, 0.7, 0.866, 0.9, 0.99]:
        dtau_trame = float(np.cos(np.arcsin(beta)))
        dtau_sr    = float(np.sqrt(1.0 - beta**2))
        rows.append({
            "beta":      beta,
            "dtau_DDD":  dtau_trame,
            "dtau_SR":   dtau_sr,
            "deviation": float(abs(dtau_trame - dtau_sr)),
        })
    return rows


def test_2c():
    """Combined: product vs additive. A = 0.20."""
    A = 0.20
    rows = []
    for beta in [0.0, 0.3, 0.5, 0.7, 0.9]:
        for r in [5.0, 10.0]:
            dtau_p = dtau_DDD(beta, r, A)
            dtau_a = dtau_GR_additive(beta, r, A)
            dev = abs(dtau_p - dtau_a) / max(abs(dtau_a), 1e-12) * 100.0
            rows.append({
                "beta":          beta,
                "r":             r,
                "chi":           float(chi_of_r(r, A)),
                "dtau_DDD":      float(dtau_p),
                "dtau_GR_add":   float(dtau_a),
                "deviation_pct": float(dev),
            })
    return rows


if __name__ == "__main__":
    print("Paper II — Test 2: time dilation")
    print("=" * 60)

    rows_2a = test_2a()
    print("\n(2a) Gravitational only — exact by calibration:")
    for r in rows_2a[:5]:
        print(f"  A={r['A']}, r={r['r']}: dtau_DDD={r['dtau_DDD']:.6f}, "
              f"dtau_GR={r['dtau_GR']:.6f}, dev={r['deviation']:.2e}")

    rows_2b = test_2b()
    print("\n(2b) Kinematic only at chi=1:")
    for r in rows_2b[:5]:
        print(f"  beta={r['beta']}: dtau_DDD={r['dtau_DDD']:.6f}, "
              f"dtau_SR={r['dtau_SR']:.6f}, dev={r['deviation']:.2e}")

    rows_2c = test_2c()
    print("\n(2c) Combined — DDD product vs GR additive (A=0.20):")
    print(f"{'beta':>5} {'r':>5} {'chi':>7} {'DDD':>10} {'GR':>10} {'dev (%)':>10}")
    for r in rows_2c:
        print(f"{r['beta']:>5.2f} {r['r']:>5.1f} {r['chi']:>7.4f} "
              f"{r['dtau_DDD']:>10.5f} {r['dtau_GR_add']:>10.5f} "
              f"{r['deviation_pct']:>10.2f}")

    out_path = DATA_DIR / "time_dilation.json"
    with open(out_path, "w") as f:
        json.dump({"2a_grav": rows_2a, "2b_kin": rows_2b, "2c_combined": rows_2c}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # Left: dtau vs r at beta=0 for several A
    rs = np.linspace(1.0, 20.0, 200)
    for A in [0.1, 0.3, 0.6]:
        ax[0].plot(rs, np.sqrt(chi_of_r(rs, A)), "-", label=f"$A={A}$")
    ax[0].set_xlabel(r"$r$")
    ax[0].set_ylabel(r"$d\tau/dt = \sqrt{\chi(r)}$")
    ax[0].set_title("Gravitational time dilation (DDD = GR by calibration)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Right: deviation product - additive vs beta at r=5, A=0.2
    A = 0.2
    betas = np.linspace(0.0, 0.99, 200)
    devs = []
    for r in [5.0, 10.0, 20.0]:
        d = []
        for beta in betas:
            p = dtau_DDD(beta, r, A)
            a = dtau_GR_additive(beta, r, A)
            d.append((p - a) / max(a, 1e-12) * 100.0)
        ax[1].plot(betas, d, "-", label=f"$r={r}$, $A=0.2$")
    ax[1].set_xlabel(r"$\beta$")
    ax[1].set_ylabel("DDD - GR$_\\text{add}$  (%)")
    ax[1].set_title("Distinguishing prediction: product vs additive")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = FIG_DIR / "fig02_time_dilation.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
