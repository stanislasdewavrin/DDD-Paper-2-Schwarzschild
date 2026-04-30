"""
Standalone G measurement script for DDD substrate (numpy-only)
================================================================

Solves the discrete Poisson equation -Laplacian(delta) = M * point
source on a cubic lattice with point source at center and Dirichlet
boundary, by Jacobi relaxation. Extracts the Newton coefficient A
via a 1/r fit on the radial profile.

RELATION TO PAPER I v7 RULE:

  This script computes the *linear-regime steady state* of the v7
  rate-limited rule (Paper I, Sec. 6.1). When every beta_i = 1, the
  v7 rule reduces to explicit Euler integration of
      dR/dt = alpha * Delta R  -  kappa * E
  whose stationary solution satisfies the discrete Poisson equation
      -alpha * Delta delta_i  =  kappa * E_i,    delta_i = R_0 - R_i
  i.e. -Laplacian(delta) = M * source with M = kappa * E_0 / alpha.
  Jacobi iteration converges to that steady state directly, in a
  fraction of the time of the explicit-time-step rule. For the full
  rate-limited rule (including non-linear saturation when beta < 1
  near a strong source), see paperI_foundations/code/v7_drainage_rule.py.

Pure numpy. No scipy required.

USAGE:
    python G_measure_standalone.py --L 64 --n-iter 5000 --output result.txt

For high precision:
    L=64,  n-iter=5000   ~10s,  expect 10^-5 precision
    L=128, n-iter=50000  ~5min, expect 10^-7 precision
    L=256, n-iter=200000 ~2h,   expect 10^-9 precision
"""
import argparse
import json
import time
import sys
import numpy as np


def weighted_linear_fit(x, y, w):
    """Weighted least-squares fit y = A*x + B.
    Returns (A, B, A_err, B_err)."""
    Sw = np.sum(w)
    Swx = np.sum(w * x)
    Swy = np.sum(w * y)
    Swxx = np.sum(w * x * x)
    Swxy = np.sum(w * x * y)
    det = Sw * Swxx - Swx * Swx
    A = (Sw * Swxy - Swx * Swy) / det
    B = (Swxx * Swy - Swx * Swxy) / det
    A_err = np.sqrt(Sw / det)
    B_err = np.sqrt(Swxx / det)
    return A, B, A_err, B_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--n-iter", type=int, default=10000)
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    L = args.L
    N_ITER = args.n_iter
    M_source = args.mass

    out = []
    def log(msg):
        print(msg)
        out.append(msg)

    log("=" * 72)
    log("DDD G measurement - standalone Poisson solver (numpy only)")
    log("=" * 72)
    log("Lattice size:        L = " + str(L) + "^3 = " + str(L**3) + " nodes")
    log("Iterations:          N_ITER = " + str(N_ITER))
    log("Point source mass:   M = " + str(M_source))
    log("Method: Jacobi iteration with zero Dirichlet BC at boundary")
    log("")

    cx, cy, cz = L // 2, L // 2, L // 2
    delta = np.zeros((L, L, L), dtype=np.float64)

    log("--- Iterations ---")
    t0 = time.time()
    log_every = max(1, N_ITER // 20)
    for it in range(N_ITER):
        s = np.zeros_like(delta)
        s[1:, :, :]  += delta[:-1, :, :]
        s[:-1, :, :] += delta[1:, :, :]
        s[:, 1:, :]  += delta[:, :-1, :]
        s[:, :-1, :] += delta[:, 1:, :]
        s[:, :, 1:]  += delta[:, :, :-1]
        s[:, :, :-1] += delta[:, :, 1:]
        s[cx, cy, cz] += M_source
        new_delta = s / 6.0
        new_delta[0, :, :] = 0; new_delta[-1, :, :] = 0
        new_delta[:, 0, :] = 0; new_delta[:, -1, :] = 0
        new_delta[:, :, 0] = 0; new_delta[:, :, -1] = 0

        if it % log_every == 0 or it == N_ITER - 1:
            diff = np.max(np.abs(new_delta - delta))
            d_center = float(new_delta[cx, cy, cz])
            d_5 = float(new_delta[cx + 5, cy, cz]) if cx + 5 < L else 0.0
            d_10 = float(new_delta[cx + 10, cy, cz]) if cx + 10 < L else 0.0
            d_20 = float(new_delta[cx + 20, cy, cz]) if cx + 20 < L else 0.0
            elapsed = time.time() - t0
            line = "  it={0:7d}  dchg={1:.3e}  d0={2:.6f}  d5={3:.6f}  d10={4:.6f}  d20={5:.6f}  ({6:.1f}s)"
            log(line.format(it, diff, d_center, d_5, d_10, d_20, elapsed))
        delta = new_delta
    elapsed_total = time.time() - t0
    log("")
    log("Total iteration time: " + str(round(elapsed_total, 1)) + " s")
    log("")

    # Radial profile
    log("--- Radial profile extraction ---")
    xs = np.arange(L) - cx
    ys = np.arange(L) - cy
    zs = np.arange(L) - cz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)

    r_max = L * 0.4
    r_bins = np.arange(0.5, r_max, 0.25)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    profile = np.zeros(len(r_centers))
    err = np.zeros(len(r_centers))
    counts = np.zeros(len(r_centers), dtype=int)
    for i in range(len(r_centers)):
        rmin = r_bins[i]
        rmax2 = r_bins[i + 1]
        mask = (r >= rmin) & (r < rmax2)
        n = int(mask.sum())
        counts[i] = n
        if n > 1:
            profile[i] = float(delta[mask].mean())
            err[i] = float(delta[mask].std() / np.sqrt(n))

    # Fit Newton: delta(r) = A/r + B  =>  delta = A*(1/r) + B (linear in 1/r)
    r_min_fit = 3.0
    r_max_fit = 0.30 * L
    mask_fit = (r_centers > r_min_fit) & (r_centers < r_max_fit) & (counts > 5) & (profile > 0)
    n_bins = int(mask_fit.sum())
    if n_bins < 5:
        log("ERROR: insufficient bins for fit. Try larger L or more iterations.")
        return 1

    log("  Fit range: r in [" + str(r_min_fit) + ", " + str(round(r_max_fit, 1)) + "], " + str(n_bins) + " bins")

    x_fit = 1.0 / r_centers[mask_fit]   # transform to linear: y = A*x + B
    y_fit = profile[mask_fit]
    e_fit = np.maximum(err[mask_fit], 1e-15)
    w_fit = 1.0 / (e_fit * e_fit)

    A, B, A_err, B_err = weighted_linear_fit(x_fit, y_fit, w_fit)
    A_continuum = M_source / (4.0 * np.pi)

    log("")
    log("  A (fit)            = " + repr(float(A)))
    log("  A (statistical err)= " + repr(float(A_err)))
    log("  A precision        = " + repr(float(A_err / abs(A))))
    log("  B (fit)            = " + repr(float(B)))
    log("  A_continuum (4 pi) = " + repr(float(A_continuum)))
    log("  Ratio A/A_cont     = " + repr(float(A / A_continuum)))
    bias_pct = (A - A_continuum) / A_continuum * 100
    log("  Bias from continuum= " + str(round(bias_pct, 4)) + " %")

    log("")
    log("  Profile sample (sim vs continuum 1/(4 pi r)):")
    log("  r       count  delta_sim          A_cont/r           sim/cont")
    indices_to_show = np.linspace(0, len(r_centers) - 1, 12).astype(int)
    for i in indices_to_show:
        if mask_fit[i]:
            d_sim = float(profile[i])
            d_th = A_continuum / r_centers[i]
            ratio_v = d_sim / d_th
            log("  " + str(round(r_centers[i], 2)).rjust(6) + " "
                + str(counts[i]).rjust(6) + " "
                + repr(d_sim).ljust(20) + " "
                + repr(float(d_th)).ljust(20) + " "
                + str(round(ratio_v, 6)))

    G_eff = A / M_source
    G_rel = A_err / abs(A)
    EXP_PRECISION = 2.2e-5

    log("")
    log("--- Verdict ---")
    log("  G_eff in lattice units   = " + repr(float(G_eff)))
    log("  Statistical precision    = " + repr(float(G_rel)))
    log("  Experimental (CODATA)    = " + str(EXP_PRECISION))
    if G_rel < EXP_PRECISION:
        ratio_v = EXP_PRECISION / G_rel
        log("  >>> Simulation is " + str(round(ratio_v, 1)) + "x MORE PRECISE than experiment <<<")
    else:
        ratio_v = G_rel / EXP_PRECISION
        log("  Simulation is " + str(round(ratio_v, 1)) + "x less precise.")
        log("  Increase L (precision ~ 1/L^2) or n-iter for more.")

    if args.output is None:
        outpath = "G_measurement_L" + str(L) + "_N" + str(N_ITER) + ".txt"
    else:
        outpath = args.output

    summary = {
        "L": L, "N_ITER": N_ITER, "M_source": M_source,
        "A_fit": float(A), "A_err": float(A_err),
        "A_relative_precision": float(G_rel),
        "A_continuum": float(A_continuum),
        "ratio_fit_continuum": float(A / A_continuum),
        "bias_pct": float(bias_pct),
        "G_eff_lattice_units": float(G_eff),
        "experimental_precision_CODATA": EXP_PRECISION,
        "elapsed_seconds": float(elapsed_total),
    }
    if G_rel > 0:
        summary["factor_better_than_experiment"] = float(EXP_PRECISION / G_rel)

    with open(outpath, "w") as fout:
        fout.write("\n".join(out))
        fout.write("\n\n--- JSON summary ---\n")
        fout.write(json.dumps(summary, indent=2))

    log("")
    log("Results saved to: " + outpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
    if G_rel > 0:
        summary["factor_better_than_experiment"] = float(EXP_PRECISION / G_rel)

    with open(outpath, "w") as fout:
        fout.write("\n".join(out))
        fout.write("\n\n--- JSON summary ---\n")
        fout.write(json.dumps(summary, indent=2))

    log("")
    log("Results saved to: " + outpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
