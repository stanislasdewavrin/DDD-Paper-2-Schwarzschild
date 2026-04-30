"""
Paper II — Master figure
=========================

Combines the three Paper II tests into a single 2x2 figure.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"

with open(DATA_DIR / "radial_profile.json")     as f: rp  = json.load(f)
with open(DATA_DIR / "time_dilation.json")      as f: td  = json.load(f)
with open(DATA_DIR / "photon_deflection.json")  as f: pd_ = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# --- Top-left: radial profile log-log
ax = axes[0, 0]
rs = np.arange(1, 16)
for E0_key, color in zip(["E0=0.2", "E0=0.5", "E0=1.0", "E0=2.0"], ["C0", "C1", "C2", "C3"]):
    sim = rp[E0_key]["delta_sim_per_r"]
    A   = rp[E0_key]["A_calibrated"]
    exp = rp[E0_key]["exponent"]
    ax.loglog(rs, sim, "o", color=color, markersize=4,
              label=f"{E0_key}, exp={exp:.4f}")
A_ref = rp["E0=1.0"]["A_calibrated"]
rs_th = np.linspace(1, 15, 100)
ax.loglog(rs_th, A_ref / rs_th, "k--", lw=1, label=r"$A/r$")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\delta(r)$")
ax.set_title("Newtonian profile from local rule\n"
             r"exponent $-1.007 \pm 0.005$ across factor 10 in $E_0$")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, which="both")

# --- Top-right: deviation from A/r
ax = axes[0, 1]
for E0_key, color in zip(["E0=0.2", "E0=0.5", "E0=1.0", "E0=2.0"], ["C0", "C1", "C2", "C3"]):
    devs = rp[E0_key]["deviation_pct"]
    ax.plot(rs, devs, "o-", color=color, label=E0_key)
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$(\delta_{\rm sim} - A/r)/(A/r)$  (%)")
ax.set_title("Deviation from analytic $A/r$\n(percent level for $r \\geq 2$)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Bottom-left: distinguishing prediction (product vs additive)
ax = axes[1, 0]
A = 0.2
betas = np.linspace(0.0, 0.99, 200)
for r in [5.0, 10.0, 20.0]:
    chi = 1.0 - A / r
    dtau_p = np.sqrt(chi) * np.sqrt(1.0 - betas**2)
    dtau_a = np.sqrt(np.maximum(0.0, 1.0 - A/r - betas**2))
    dev = (dtau_p - dtau_a) / np.maximum(dtau_a, 1e-12) * 100.0
    ax.plot(betas, dev, "-", label=f"$r={r}$, $A=0.2$")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$(d\tau/dt)_{\rm DDD} - (d\tau/dt)_{\rm GR_{add}}$  (%)")
ax.set_title("Distinguishing prediction: product vs additive\n"
             r"$\sqrt{\chi}\sqrt{1-\beta^2}$ vs $\sqrt{1-A/r-\beta^2}$")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Bottom-right: photon deflection log-log
ax = axes[1, 1]
cases = pd_["cases"]
A_values = sorted({c["A"] for c in cases})
markers = ["o", "s", "D", "^", "v"]
for A, mk in zip(A_values, markers):
    bs   = [c["b"] for c in cases if c["A"] == A]
    ths  = [c["theta_sim"] for c in cases if c["A"] == A]
    ax.loglog(bs, ths, mk, markersize=6, label=f"$A={A:.3f}$")
bs_th = np.logspace(np.log10(3), np.log10(60), 100)
A_ref = 0.05
ax.loglog(bs_th, A_ref / bs_th,     "k--", lw=1, label=r"$A/b$ ($A=0.05$)")
ax.loglog(bs_th, 2 * A_ref / bs_th, "b:",  lw=1, label=r"$2A/b$ ($A=0.05$, full GR)")
ax.set_xlabel(r"$b$")
ax.set_ylabel(r"$\Delta\theta$")
ax.set_title("Photon deflection: geometric (DDD recovers $A/b$)\n"
             "Shapiro contribution missing -> factor 2 gap to full GR")
ax.legend(fontsize=7, loc="lower left")
ax.grid(True, alpha=0.3, which="both")

fig.suptitle("Paper II — Emergent gravity from Discrete Drainage Dynamics", fontsize=13, y=1.00)
fig.tight_layout()

fig_path = FIG_DIR / "fig04_master.pdf"
fig.savefig(fig_path, bbox_inches="tight")
fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path}")
