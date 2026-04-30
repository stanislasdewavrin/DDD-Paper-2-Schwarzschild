"""Iterative cascade test: linear / single throttle / double throttle vs GR.
Single throttle (source-side R/R_0) predicts Schwarzschild form analytically.
"""
import json, time, sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)


def directional_flux_sums(R, alpha, R0, throttle):
    O = np.zeros_like(R); S = np.zeros_like(R)
    for axis in range(3):
        for shift in (-1, +1):
            Rn = np.roll(R, shift, axis=axis)
            if throttle == 'single':   gate_out = R / R0
            elif throttle == 'double': gate_out = (R * Rn) / (R0 * R0)
            else:                       gate_out = 1.0
            out = alpha * gate_out * np.maximum(R - Rn, 0.0)
            Rn2 = np.roll(R, -shift, axis=axis)
            if throttle == 'single':   gate_in = Rn2 / R0
            elif throttle == 'double': gate_in = (R * Rn2) / (R0 * R0)
            else:                       gate_in = 1.0
            inc = alpha * gate_in * np.maximum(Rn2 - R, 0.0)
            O += out; S += inc
    return O, S


def run(L, n_ticks, tau, alpha, kappa, R0, E0, throttle):
    cx = L // 2
    R = np.full((L, L, L), R0, dtype=np.float64)
    E = np.zeros_like(R); E[cx, cx, cx] = E0
    t0 = time.time()
    for it in range(n_ticks):
        O, S = directional_flux_sums(R, alpha, R0, throttle)
        A = R + tau * S
        D = tau * O + tau * kappa * E
        beta = np.where(D > 1e-15, np.minimum(1.0, A / np.maximum(D, 1e-15)), 1.0)
        R = R + tau * S - beta * tau * O - beta * tau * kappa * E
        R = np.maximum(R, 0.0)
        R[0, :, :] = R0; R[-1, :, :] = R0
        R[:, 0, :] = R0; R[:, -1, :] = R0
        R[:, :, 0] = R0; R[:, :, -1] = R0
    return R, time.time() - t0


def radial_profile(R, R0, L):
    cx = L // 2
    xs = np.arange(L) - cx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_bins = np.arange(0.5, L * 0.4, 0.5)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    delta = R0 - R
    prof = np.zeros(len(r_centers))
    for i in range(len(r_centers)):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if mask.sum() > 1:
            prof[i] = float(delta[mask].mean())
    return r_centers, prof


def main():
    L = 40; n_ticks = 5000; tau = 0.05
    alpha = 1.0; kappa = 1.0; R0 = 1.0; E0 = 1.5

    print(f"L={L} n_ticks={n_ticks} E0={E0}")
    print("Linear...")
    R_lin, t1 = run(L, n_ticks, tau, alpha, kappa, R0, E0, throttle=None)
    print(f"  done {t1:.1f}s, min(R)={R_lin.min():.4f}")
    print("Single throttle...")
    R_s, t2 = run(L, n_ticks, tau, alpha, kappa, R0, E0, throttle='single')
    print(f"  done {t2:.1f}s, min(R)={R_s.min():.4f}")
    print("Double throttle...")
    R_d, t3 = run(L, n_ticks, tau, alpha, kappa, R0, E0, throttle='double')
    print(f"  done {t3:.1f}s, min(R)={R_d.min():.4f}")

    r, prof_lin = radial_profile(R_lin, R0, L)
    _, prof_s = radial_profile(R_s, R0, L)
    _, prof_d = radial_profile(R_d, R0, L)

    M_eff = kappa * E0 / alpha
    x_gr = M_eff / (4 * np.pi * r) / R0
    delta_gr = R0 * (1 - np.sqrt(np.maximum(1 - 2 * x_gr, 1e-12)))

    print()
    print("    r          x       linear     single     double         GR  single/GR")
    for i in range(0, len(r), 2):
        if r[i] > L * 0.35 or x_gr[i] > 0.49:
            continue
        if delta_gr[i] < 1e-9:
            continue
        ratio = prof_s[i] / delta_gr[i]
        print(f"{r[i]:6.2f}  {x_gr[i]:9.4e}  {prof_lin[i]/R0:9.4e}  "
              f"{prof_s[i]/R0:9.4e}  {prof_d[i]/R0:9.4e}  "
              f"{delta_gr[i]/R0:9.4e}  {ratio:8.4f}")

    out = {"L": L, "n_ticks": n_ticks, "E0": E0, "M_eff": M_eff,
           "r": r.tolist(),
           "delta_linear": (prof_lin / R0).tolist(),
           "delta_single": (prof_s / R0).tolist(),
           "delta_double": (prof_d / R0).tolist(),
           "delta_gr": (delta_gr / R0).tolist(),
           "x": x_gr.tolist()}
    with open(DATA / "iterative_cascade_L32.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved data.")

    try:
        import matplotlib.pyplot as plt
        mask = (x_gr < 0.49) & (r > 1.5) & (r < L * 0.35)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.plot(r[mask], prof_lin[mask] / R0, 'o-', ms=4, label='linear v7')
        ax.plot(r[mask], prof_s[mask] / R0, 's-', ms=4, label='single throttle (cascade)')
        ax.plot(r[mask], prof_d[mask] / R0, '^-', ms=4, label='double throttle')
        ax.plot(r[mask], delta_gr[mask] / R0, 'k--', lw=1.5, label=r'GR: $1-\sqrt{1-2x}$')
        ax.set_xlabel('r'); ax.set_ylabel(r'$\delta(r)/R_0$')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.set_title('Deficit profile, log-log')

        ax = axes[1]
        # Plot ratio (single - linear) / (GR - linear_continuum)
        # The "linear continuum" prediction is just x. Lattice deviates.
        # We compare RATIO of throttled extras to GR extra.
        gr_extra = delta_gr / R0 - x_gr
        sing_extra = prof_s / R0 - prof_lin / R0
        doub_extra = prof_d / R0 - prof_lin / R0
        m2 = mask & (gr_extra > 1e-6)
        ax.plot(x_gr[m2], sing_extra[m2] / gr_extra[m2], 's-', ms=5,
                label='single / GR-extra')
        ax.plot(x_gr[m2], doub_extra[m2] / gr_extra[m2], '^-', ms=5,
                label='double / GR-extra')
        ax.axhline(1.0, color='k', ls='--', lw=1, label='exact GR match')
        ax.axhline(2.0, color='gray', ls=':', lw=1, label='analytic double = 2')
        ax.set_xscale('log')
        ax.set_xlabel('x'); ax.set_ylabel('throttled-extra / GR-extra')
        ax.set_title('Coefficient of second-order correction')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(FIG / "fig4_iterative_cascade.png", dpi=160, bbox_inches='tight')
        fig.savefig(FIG / "fig4_iterative_cascade.pdf", bbox_inches='tight')
        plt.close(fig)
        print("Saved figure.")
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
