"""
Generate the three figures for Paper II v7.2:

  fig1_reserve_bandwidth_proper_time.{pdf,png}
  fig2_radial_deficit_bandwidth.{pdf,png}
  fig3_ddd_vs_schwarzschild.{pdf,png}

Pure matplotlib + numpy. Reads simulation data from
data/bandwidth_clock_L24_E0.3.json.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)


# ----------------------------------------------------------------
# Figure 1: Reserve -> Bandwidth -> Proper time
# ----------------------------------------------------------------
def fig1_reserve_to_proper_time():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    R0 = 1.0

    # Panel A: tanks at different levels
    ax = axes[0]
    for k, R in enumerate([1.0, 0.6, 0.2]):
        x = 0.3 + k * 1.0
        ax.add_patch(mpatches.Rectangle((x, 0.0), 0.6, 1.0,
                                         fill=False, edgecolor='black', lw=1.5))
        ax.add_patch(mpatches.Rectangle((x, 0.0), 0.6, R,
                                         facecolor='steelblue', edgecolor='none', alpha=0.6))
        ax.text(x + 0.3, R + 0.06, f"$R_i = {R:.1f}$",
                ha='center', va='bottom', fontsize=9)
    ax.set_xlim(-0.2, 3.6)
    ax.set_ylim(-0.15, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) Local reserve $R_i$", fontsize=10)

    # Panel B: bandwidth alpha R as linear function of R
    ax = axes[1]
    R = np.linspace(0, 1.0, 100)
    ax.plot(R, R, lw=2, color='steelblue', label=r'$\alpha R_i$ (max throughput)')
    ax.set_xlabel(r'$R_i / R_0$')
    ax.set_ylabel('local bandwidth (rest units)')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    ax.set_title("(b) Bandwidth $\\propto R_i$", fontsize=10)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.text(0.05, 1.02, "rest", color='gray', fontsize=8)

    # Panel C: proper time fraction = R/R_0
    ax = axes[2]
    ax.plot(R, R, lw=2, color='darkred', label=r'$d\tau/dt = R_i/R_0$')
    ax.set_xlabel(r'$R_i / R_0$')
    ax.set_ylabel(r'$d\tau / dt$')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    ax.set_title("(c) Bandwidth $=$ proper-time rate", fontsize=10)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    # annotate
    ax.annotate("at rest:\n$d\\tau/dt=1$", xy=(1.0, 1.0), xytext=(0.6, 0.3),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate("drained:\n$d\\tau/dt < 1$", xy=(0.4, 0.4), xytext=(0.05, 0.7),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))

    fig.suptitle("Reserve $\\to$ Bandwidth $\\to$ Proper time",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_reserve_bandwidth_proper_time.pdf", bbox_inches='tight')
    fig.savefig(FIG / "fig1_reserve_bandwidth_proper_time.png", dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig1")


# ----------------------------------------------------------------
# Figure 2: Radial deficit and bandwidth profile
# ----------------------------------------------------------------
def fig2_radial_profile():
    json_path = DATA / "bandwidth_clock_L24_E0.3.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        return
    with open(json_path) as f:
        d = json.load(f)
    r = np.array(d["r_centers"])
    deficit = np.array(d["delta_over_R0"])
    bandwidth = np.array(d["R_over_R0"])
    M = d["M_eff"]
    A_continuum = M / (4 * np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: deficit
    ax = axes[0]
    ax.plot(r, deficit, 'o', ms=5, color='navy', label='simulation')
    rs = np.linspace(r.min(), r.max(), 100)
    ax.plot(rs, A_continuum / rs, '-', lw=1.5, color='darkred',
            label=r'continuum $M/(4\pi r)$')
    ax.set_xlabel('$r$ (lattice units)')
    ax.set_ylabel(r'$\delta(r) / R_0$')
    ax.set_title('(a) Radial deficit profile', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: bandwidth
    ax = axes[1]
    ax.plot(r, bandwidth, 'o', ms=5, color='navy', label='simulation')
    ax.plot(rs, 1 - A_continuum / rs, '-', lw=1.5, color='darkred',
            label=r'$1 - M/(4\pi r)$')
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('$r$ (lattice units)')
    ax.set_ylabel(r'$d\tau / dt = R(r)/R_0$')
    ax.set_title('(b) Bandwidth = proper-time rate', fontsize=10)
    ax.set_ylim(0.99, 1.005)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)

    fig.suptitle('Radial deficit and bandwidth profile, '
                 + r'$L=24$, $E_0=0.3$', y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_radial_deficit_bandwidth.pdf", bbox_inches='tight')
    fig.savefig(FIG / "fig2_radial_deficit_bandwidth.png", dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig2")


# ----------------------------------------------------------------
# Figure 3: DDD linear vs Schwarzschild weak-field
# ----------------------------------------------------------------
def fig3_ddd_vs_schwarzschild():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    x = np.linspace(0, 0.45, 200)
    psi_ddd = 1 - x
    # Schwarzschild factor 1/sqrt(g_00) = sqrt(1 - 2x), valid for x < 0.5
    psi_schw = np.sqrt(np.maximum(1 - 2*x, 0))

    # Left: psi vs x on full range
    ax = axes[0]
    ax.plot(x, psi_ddd, '-', lw=2, color='steelblue',
            label=r'DDD linear: $1 - x$')
    ax.plot(x, psi_schw, '-', lw=2, color='darkred',
            label=r'Schwarzschild: $\sqrt{1 - 2x}$')
    ax.axhline(0, color='gray', ls=':', lw=0.6)
    ax.set_xlabel(r'$x = GM/(c^2 r)$')
    ax.set_ylabel(r'$d\tau / dt$')
    ax.set_title('(a) Clock factors, full range', fontsize=10)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0, 1.05)

    # Right: relative deviation log-log
    ax = axes[1]
    eps = 1e-8
    x_log = np.logspace(-10, -0.5, 400)
    psi_ddd_log = 1 - x_log
    psi_schw_log = np.sqrt(np.maximum(1 - 2*x_log, eps))
    delta = np.abs(psi_ddd_log - psi_schw_log) / psi_schw_log

    ax.loglog(x_log, delta, '-', lw=2, color='black', label=r'$|\Delta| / \psi_\mathrm{Schw}$')
    # Reference: 1/2 x leading-order
    ax.loglog(x_log, 0.5 * x_log, '--', lw=1.2, color='gray',
              label=r'leading order $\frac{1}{2}x$')

    # Annotate physical regimes
    ann = [
        (7e-10, 'Earth surface'),
        (2e-6, 'solar limb'),
        (0.5, 'Schwarzschild radius'),
    ]
    for x_pt, label in ann:
        if x_pt < x_log.max():
            d_pt = 0.5 * x_pt
            ax.plot([x_pt], [d_pt], 'o', ms=6, color='darkred')
            ax.annotate(label, xy=(x_pt, d_pt), xytext=(x_pt * 2, d_pt * 0.3),
                        fontsize=8, arrowprops=dict(arrowstyle='->',
                                                    color='darkred', lw=0.6))

    ax.set_xlabel(r'$x = GM/(c^2 r)$')
    ax.set_ylabel(r'$|d\tau/dt_\mathrm{DDD} - d\tau/dt_\mathrm{Schw}| / d\tau/dt_\mathrm{Schw}$')
    ax.set_title('(b) Relative deviation (log--log)', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, which='both')

    fig.suptitle("DDD linear clock factor vs.\\ Schwarzschild",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ddd_vs_schwarzschild.pdf", bbox_inches='tight')
    fig.savefig(FIG / "fig3_ddd_vs_schwarzschild.png", dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig3")


def main():
    fig1_reserve_to_proper_time()
    fig2_radial_profile()
    fig3_ddd_vs_schwarzschild()


if __name__ == "__main__":
    main()
