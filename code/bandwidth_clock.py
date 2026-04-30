"""
Bandwidth = clock test on the v7 rule
========================================

Hypothesis to test: in the rate-limited v7 rule of Paper I, the
local available bandwidth alpha * R(r) is, by definition, the
local rate at which node operations can be performed. If we
interpret that as the local clock rate (operations per
coordinate tick), the natural prediction is

    d tau / dt  =  R(r) / R_0

without any postulated square-root rule.

Under the calibration  delta / R_0  <->  G M / (c^2 r)  this
matches weak-field general-relativistic time dilation
sqrt(1 - 2 G M / c^2 r) at first order in the small parameter.

This script:
  1. Runs the v7 explicit rate-limited rule with a strong sink
     until steady state.
  2. Bins R(r) radially.
  3. Compares the linear bandwidth ratio R/R_0 with:
     - the would-be sqrt(R/R_0) (Paper II v7 first draft)
     - the weak-field GR factor sqrt(1 - 2 GM/c^2 r) under both
       calibration choices  delta/R_0 = GM/c^2 r  and
       delta/R_0 = 2GM/c^2 r .
  4. Reports which interpretation is leading-order consistent.

Pure numpy, no external dependencies beyond numpy.
"""
import argparse
import json
import time
import sys
import numpy as np


def directional_flux_sums(R, alpha):
    O = np.zeros_like(R)
    S = np.zeros_like(R)
    for axis in range(3):
        for shift in (-1, +1):
            Rn = np.roll(R, shift, axis=axis)
            out = alpha * np.maximum(R - Rn, 0.0)
            Rn2 = np.roll(R, -shift, axis=axis)
            inc = alpha * np.maximum(Rn2 - R, 0.0)
            O += out
            S += inc
    return O, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--n_ticks", type=int, default=10000)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--R0", type=float, default=1.0)
    ap.add_argument("--E0", type=float, default=0.3)
    args = ap.parse_args()

    L = args.L
    cx, cy, cz = L//2, L//2, L//2
    R = np.full((L, L, L), args.R0, dtype=np.float64)
    E = np.zeros_like(R)
    E[cx, cy, cz] = args.E0

    print("=" * 72)
    print("Bandwidth = clock test on the v7 rule")
    print("=" * 72)
    print(f"L={L}, tau={args.tau}, alpha={args.alpha}, kappa={args.kappa}")
    print(f"R0={args.R0}, E0={args.E0} (single sink at centre)")
    print(f"n_ticks={args.n_ticks}")
    print()

    log_every = max(1, args.n_ticks // 10)
    t0 = time.time()
    for it in range(args.n_ticks):
        O_des, S_in = directional_flux_sums(R, args.alpha)
        A = R + args.tau * S_in
        D = args.tau * O_des + args.tau * args.kappa * E
        beta = np.where(D > 1e-15, np.minimum(1.0, A / np.maximum(D, 1e-15)), 1.0)
        R = R + args.tau * S_in - beta * args.tau * O_des - beta * args.tau * args.kappa * E
        R = np.maximum(R, 0.0)
        # Dirichlet BC at edges: tie outermost layer to R0
        R[0, :, :] = args.R0; R[-1, :, :] = args.R0
        R[:, 0, :] = args.R0; R[:, -1, :] = args.R0
        R[:, :, 0] = args.R0; R[:, :, -1] = args.R0

        if it % log_every == 0 or it == args.n_ticks - 1:
            d0 = args.R0 - float(R[cx, cy, cz])
            print(f"  it={it:7d}  delta(0)={d0:.5f}  min(R)={float(R.min()):.5f}  "
                  f"min(beta)={float(beta.min()):.4f}")
    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f} s")

    # Radial profile
    delta = args.R0 - R
    xs = np.arange(L) - cx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_bins = np.arange(0.5, L * 0.4, 0.5)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    delta_bin = np.zeros(len(r_centers))
    R_bin = np.zeros(len(r_centers))
    for i in range(len(r_centers)):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if mask.sum() > 1:
            delta_bin[i] = float(delta[mask].mean())
            R_bin[i] = float(R[mask].mean())

    # Use the bulk (skip near-source artefacts and near-boundary)
    r_min_bulk = 3.0
    r_max_bulk = 0.30 * L
    mask = (r_centers > r_min_bulk) & (r_centers < r_max_bulk) & (R_bin > 0)
    r_use = r_centers[mask]
    delta_use = delta_bin[mask]
    R_use = R_bin[mask]

    # Newton fit: delta = A/r
    # Theory: A = M / (4 pi)  with M = kappa * E0 / alpha
    M_eff = args.kappa * args.E0 / args.alpha
    A_continuum = M_eff / (4 * np.pi)
    A_fit = np.average(delta_use * r_use)  # rough average; precise fit not needed for test

    # Now compare three candidate "clock rate" rules:
    # 1) Linear bandwidth:  d tau / dt = R / R_0
    # 2) Square-root rule:  d tau / dt = sqrt(R / R_0)
    # 3) GR weak-field with calibration A:  sqrt(1 - 2 (delta) / R_0)
    #    -- this is the actual GR formula under the calibration
    #       delta/R_0 = 2 G M / (c^2 r), i.e. paper II v7 first draft
    # 4) GR weak-field with calibration B:  sqrt(1 - (delta) / R_0 * 2)
    #    -- but reinterpreted: delta/R_0 = G M / (c^2 r), so
    #       2 GM/c^2 r = 2 * delta / R_0. Same number.
    # Actually distinct option:
    # 5) GR with the "linear" calibration  delta/R_0 = GM/c^2 r,
    #    so the GR clock factor is sqrt(1 - 2 * delta/R_0).
    #    Compare both with their own GR factors.

    psi_linear = R_use / args.R0
    psi_sqrt = np.sqrt(R_use / args.R0)
    # GR factor with calibration delta/R_0 = 2 GM/c^2 r:
    # psi_GR_A = sqrt(1 - delta_use/R_0) ... wait, this is the same as sqrt rule
    # The point is: the SAME SHAPE of GR weak-field corresponds to
    # different values of delta. Let me just print the leading orders.

    # Leading-order weak-field GR: psi_GR ~ 1 - GM/c^2 r
    # If delta/R_0 = GM/c^2 r (linear calibration), then 1 - delta/R_0
    # = psi_linear matches at leading order.
    # If delta/R_0 = 2 GM/c^2 r (sqrt calibration), then 1 - delta/(2R_0)
    # would match. That's psi_sqrt at leading order indeed.

    # Both calibrations give a leading-order match; they differ in
    # what numerical value of M corresponds to a given delta.

    print()
    print("--- Bulk steady-state radial profile ---")
    print(f"  fit range r in [{r_min_bulk:.1f}, {r_max_bulk:.1f}]")
    print(f"  M_eff = kappa * E_0 / alpha = {M_eff:.4f}")
    print(f"  A_continuum = M_eff / (4 pi) = {A_continuum:.6f}")
    print()
    print(f"  {'r':>6} {'delta/R_0':>10} {'R/R_0':>10} "
          f"{'sqrt(R/R_0)':>12} {'1 - delta/R_0':>14}")
    for i in range(0, len(r_use), max(1, len(r_use)//10)):
        d_norm = delta_use[i] / args.R0
        print(f"  {r_use[i]:6.2f} {d_norm:10.5f} {psi_linear[i]:10.5f} "
              f"{psi_sqrt[i]:12.5f} {1.0 - d_norm:14.5f}")

    print()
    print("--- Interpretation ---")
    print("If we take d(tau)/dt = R/R_0 (LINEAR bandwidth rule), then under")
    print("the calibration delta/R_0 = GM/c^2 r, we have at leading order")
    print("    d(tau)/dt ~ 1 - GM/c^2 r")
    print("which matches the leading-order weak-field GR factor")
    print("    sqrt(1 - 2 GM/c^2 r) ~ 1 - GM/c^2 r")
    print("WITHOUT any postulated square-root rule.")
    print()
    print("If instead we take d(tau)/dt = sqrt(R/R_0) (SQRT rule), then")
    print("under the calibration delta/R_0 = 2 GM/c^2 r, we recover the")
    print("same leading-order match. Both work numerically; the linear")
    print("interpretation is conceptually simpler and removes the")
    print("postulated sqrt.")
    print()

    # Save
    out = {
        "L": L, "n_ticks": args.n_ticks,
        "alpha": args.alpha, "kappa": args.kappa, "R0": args.R0, "E0": args.E0,
        "M_eff": M_eff,
        "r_centers": r_use.tolist(),
        "delta_over_R0": (delta_use / args.R0).tolist(),
        "R_over_R0": (R_use / args.R0).tolist(),
        "psi_linear": psi_linear.tolist(),
        "psi_sqrt": psi_sqrt.tolist(),
    }
    fname = f"bandwidth_clock_L{L}_E{args.E0}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {fname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
