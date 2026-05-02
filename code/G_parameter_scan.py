"""
Parameter scan for Paper II.b — Newton's G structural form.

Verifies the predicted scaling of the throttled-mobility steady-state
parameter beta = (kappa * E_0) / (2 * pi * alpha * R_0) by scanning
each parameter independently and measuring beta from the simulation.

The single-throttle rule of Paper II is used:
    F_{i->j} = alpha * (R_i / R_0) * (R_i - R_j)_+

For each parameter combination, the simulation runs to steady state and
beta is extracted from the inner-shell relation
    R(r=3)/R_0 = sqrt(1 - beta/3)   =>   beta = 3 * (1 - (R(3)/R_0)^2).

Pure numpy. Run locally; total time about 2-3 minutes for the full scan.
"""
import json
import time
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)


def directional_flux_sums(R, alpha, R0):
    """Single-throttle flux: F_ij = alpha (R_i/R_0)(R_i - R_j)_+."""
    O = np.zeros_like(R); S = np.zeros_like(R)
    for axis in range(3):
        for shift in (-1, +1):
            Rn = np.roll(R, shift, axis=axis)
            gate_out = R / R0
            out = alpha * gate_out * np.maximum(R - Rn, 0.0)
            Rn2 = np.roll(R, -shift, axis=axis)
            gate_in = Rn2 / R0
            inc = alpha * gate_in * np.maximum(Rn2 - R, 0.0)
            O += out; S += inc
    return O, S


def run_one(L, n_ticks, tau, alpha, kappa, R0, E0):
    cx = L // 2
    R = np.full((L, L, L), R0, dtype=np.float64)
    E = np.zeros_like(R); E[cx, cx, cx] = E0
    for it in range(n_ticks):
        O, S = directional_flux_sums(R, alpha, R0)
        A = R + tau * S
        D = tau * O + tau * kappa * E
        beta_lim = np.where(D > 1e-15,
                            np.minimum(1.0, A / np.maximum(D, 1e-15)),
                            1.0)
        R = R + tau * S - beta_lim * tau * O - beta_lim * tau * kappa * E
        R = np.maximum(R, 0.0)
        R[0, :, :] = R0; R[-1, :, :] = R0
        R[:, 0, :] = R0; R[:, -1, :] = R0
        R[:, :, 0] = R0; R[:, :, -1] = R0
    # Extract beta from inner shell at r = 3 along x-axis
    R_at3 = float(R[cx + 3, cx, cx])
    u = R_at3 / R0
    beta_meas = 3.0 * (1.0 - u * u)
    return beta_meas, R_at3


def scan_one(label, vary, defaults, L=24, n_ticks=4000, tau=0.05):
    print(f"\n=== Scan over {label} ===")
    results = []
    for v in vary:
        params = dict(defaults)
        params[label] = v
        t0 = time.time()
        beta_meas, R_at3 = run_one(L, n_ticks, tau, **params)
        elapsed = time.time() - t0
        # Predicted beta from formula
        beta_pred = (params['kappa'] * params['E0']) / (
            2 * np.pi * params['alpha'] * params['R0'])
        ratio = beta_meas / beta_pred if beta_pred > 0 else float('nan')
        print(f"  {label}={v:.4f}  beta_pred={beta_pred:.5f}  "
              f"beta_meas={beta_meas:.5f}  ratio={ratio:.4f}  "
              f"({elapsed:.1f}s)")
        results.append({
            label: v, 'beta_pred': beta_pred,
            'beta_meas': beta_meas, 'ratio': ratio,
            'elapsed': elapsed,
        })
    return results


def fit_exponent(values, betas):
    """Fit power law beta = C * v^p."""
    v = np.array(values, dtype=np.float64)
    b = np.array(betas, dtype=np.float64)
    mask = (v > 0) & (b > 0)
    if mask.sum() < 2:
        return None, None
    logv = np.log(v[mask]); logb = np.log(b[mask])
    p, logC = np.polyfit(logv, logb, 1)
    return p, np.exp(logC)


def main():
    defaults = dict(alpha=1.0, kappa=1.0, R0=1.0, E0=0.3)
    L = 24
    n_ticks = 4000
    tau = 0.05

    print("=" * 72)
    print("Parameter scan for Paper II.b (Newton's G structural form)")
    print("=" * 72)
    print(f"L = {L}, n_ticks = {n_ticks}, tau = {tau}")
    print(f"Defaults: {defaults}")

    all_results = {}

    # Scan kappa
    kappa_values = [0.5, 0.75, 1.0, 1.5, 2.0]
    all_results['kappa'] = scan_one('kappa', kappa_values, defaults, L, n_ticks, tau)

    # Scan alpha
    alpha_values = [0.5, 0.75, 1.0, 1.5, 2.0]
    all_results['alpha'] = scan_one('alpha', alpha_values, defaults, L, n_ticks, tau)

    # Scan R0
    R0_values = [0.7, 0.85, 1.0, 1.15, 1.3]
    all_results['R0'] = scan_one('R0', R0_values, defaults, L, n_ticks, tau)

    # Scan E0
    E0_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_results['E0'] = scan_one('E0', E0_values, defaults, L, n_ticks, tau)

    # Fit exponents
    print("\n" + "=" * 72)
    print("Fitted power-law exponents (compare to predicted)")
    print("=" * 72)
    for label, expected in [('kappa', +1.0), ('alpha', -1.0),
                             ('R0', -1.0), ('E0', +1.0)]:
        rows = all_results[label]
        v = [r[label] for r in rows]
        b = [r['beta_meas'] for r in rows]
        p, C = fit_exponent(v, b)
        print(f"  {label:<8s}: predicted = {expected:+.2f}, "
              f"measured = {p:+.4f}, ratio_to_pred = {p/expected:.4f}")

    # Save
    with open(DATA / "G_parameter_scan.json", "w") as f:
        json.dump({
            'L': L, 'n_ticks': n_ticks, 'tau': tau,
            'defaults': defaults,
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved: data/G_parameter_scan.json")


if __name__ == "__main__":
    main()
