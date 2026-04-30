# Content relocated from old Paper II during v7 refocus

The original Paper II ("Emergent Gravity from Discrete Drainage Dynamics:
The Newtonian Profile, Time Dilation, and Photon Deflection from a Single
Local Rule") was rewritten as Paper II v7 with a narrower, more
defensible scope: **Clock-Rate Suppression and Lorentz-Like Kinematics
in Discrete Drainage Dynamics**.

The following content was removed from Paper II and must be ported into
its destination paper. This file is the audit trail.

---

## 1. Newtonian profile derivation (1/r law)

**Old location:** Paper II, Section 3 ("The Newtonian Profile from the
Discrete Gauss Law")

**Old content:**
- Analytic derivation of $\delta(r) = A/r$ from the discrete Gauss
  law on a graph of effective dimension 3.
- Numerical verification with measured exponent $-1.019 \pm 0.001$.
- Mass--coupling identification $A = M / (4\pi)$.

**New location:** Paper I v7 (already there).

**Status:** ✅ Fully covered in Paper I v7, Sec. 5 (rule), Sec. 6.1
(derivation of the Poisson limit), Sec. 6.2 (numerical check with
$L \in \{64, 128, 160\}$). Paper II v7 cites Paper I and does not
re-derive.

---

## 2. Photon deflection $\Delta\theta = 4GM/c^2 b$

**Old location:** Paper II, Section 5 ("Photon Deflection")

**Old content:**
- Geometric ray-tracing through the simulated $\chi(r)$ profile.
- Identification $c_{\rm eff} = c\,\chi$ for the photon's coordinate
  speed, encoding both temporal and spatial dilation.
- Numerical recovery of the full GR deflection $4GM/c^2 b$ to
  sub-percent accuracy in $b \in [2, 50]$, with measured log--log
  exponent $-1.012$.
- Comment on the absence of a separately added "Shapiro contribution".
- Files: `code/03_photon_deflection.py`, `data/photon_deflection.json`,
  `figures/fig03_photon_deflection.{pdf,png}`.

**New destination:** Paper IV --- *Predictions and tests of the
gravitational sector*.

**Status:** ⏳ To be ported when Paper IV is written. The numerical
data and figures are in place; only the rewritten text and v7
conventions ($\chi \to R/R_0$ as a clock-rate factor, not a metric
factor) are needed.

**Reframing required for Paper IV:** the original framing claimed
"recovery of the full GR result"; under the v7 grammar this should
be a *calibrated weak-field consequence* of the rule, not a
derivation of GR. The agreement at $b \gg A$ is a *check*, not a
*proof*.

---

## 3. Self-consistent feedback and falsifiable Yukawa modification

**Old location:** Paper II, Section 6 ("Self-Consistent Treatment of
the Deficit Field"), 5 subsections.

**Old content:**
- Quadratic feedback term in the rule: deficit acts as its own
  secondary source.
- Continuum self-consistent equation.
- Numerical verification.
- Comparison with the Schwarzschild profile.
- Predicted Yukawa modification with length scale
  $\lambda_{\rm phys} = \sqrt{\alpha/|\beta|}\,\ell_P$.
- Eöt-Wash bound: $\lambda_{\rm phys}$ must lie outside
  $[10\,\mu\text{m}, 10^5\,\text{m}]$ (calibrated, not derived).
- Files: `code/05_nonlinear_feedback.py`,
  `code/06_quadratic_feedback.py`, `code/08_yukawa_bounds.py`,
  associated data and figures.

**New destination:** Paper IV --- *Predictions and tests of the
gravitational sector*. This is exactly the kind of "extension-level
prediction" Paper IV is meant to host.

**Status:** ⏳ To be ported when Paper IV is written. The
calibrated--vs--derived distinction must be made explicit
(currently the framing implies derivation).

---

## 4. Ontological interpretation of $G$ via Floquet two-step structure

**Old location:** Paper II, Section 8 ("Ontological interpretation of
$G$ in the Weyl-Floquet substrate")

**Old content:**
- $\kappa = 1 / \tau_{\rm N\oe ud}$ (matter-update half-tick duration).
- $J = 1 / \tau_{\rm Lien}$ (gauge-update half-tick duration).
- $\kappa / J$ as the Floquet structural ratio, $\approx 0.10$ at
  hadronic scale.
- Connection between $G$ and $\alpha_{\rm EM}$ through the same
  temporal asymmetry.
- Argument for $\dot G / G = 0$ from topological invariance of the
  Weyl-pair separation.
- Two-scale picture (Planckian vs hadronic).

**New destinations:**
- Paper VI --- *Electromagnetism from the oriented sector*: ratio
  $\kappa/J$ and the connection between $G$ and $\alpha_{\rm EM}$.
- Paper XVIII --- *Weyl substrate identification*: topological
  invariance argument for $\dot G / G = 0$, two-scale picture.

**Status:** ⏳ To be ported in the v7 rewrites of Papers VI and
XVIII. The "Pas Nœud / Pas Lien" terminology must be replaced by
"Node Update / Link Update" everywhere, consistent with Paper I v7.

**Note:** under the v7 convention, the half-tick durations
$\tau_{\rm node}$ and $\tau_{\rm link}$ are not the structural
parameters; the rate-limiter $\beta_i$ is. Whether the
$\kappa = 1/\tau_{\rm node}$ identification still makes sense in v7
needs to be re-examined when porting to Paper VI / XVIII.

---

## 5. Kinematic clock-rate suppression (motion-induced time dilation)

**Old location:** Paper II v7 first draft (paper_v7.tex), Section 5
("Clock-rate suppression by motion") and Section 7 ("The combined
factor: product, not sum").

**Old content:**
- Postulated kinematic clock-rate factor $\psi_{\rm kin}(v) =
  \sqrt{1 - v^2/c_{\rm eff}^2}$ for a localised excitation
  translating at coordinate velocity $v$ across the substrate.
- Postulated quadratic motion budget cost $f_{\rm motion}(v) =
  v^2/c_{\rm eff}^2$.
- Postulated square-root budget--clock relation
  $d\tau/dt = \sqrt{\rho}$ (Sec. 6 of v7 first draft, "The
  square-root rule, under scrutiny") to recover the SR Lorentz
  factor at leading order.
- Combined product form $\psi(R, v) = \sqrt{R/R_0} \cdot
  \sqrt{1 - v^2/c_{\rm eff}^2}$ vs additive Schwarzschild
  weak-field expansion.
- Distinguishing second-order cross term, table of regime of
  distinguishability.

**New destination:** Paper III --- *discrete-time response,
inertial effects, and kinematic clock-rate suppression of moving
configurations*.

**Status:** ⏳ To be derived (not just postulated) in Paper III.
The simulation should track a translating Gaussian excitation in
the v7 rule and measure how its participation in the substrate's
flux activity scales with $v$. Whether the resulting clock-rate
factor takes the Lorentz form, and through what mechanism, is
left to that paper.

**Reframing required for Paper III:**
- The square-root rule that was used to make the kinematic factor
  match SR at leading order is dropped from Paper II v7.1
  (bandwidth = clock; linear dependence on reserve fraction). For
  the kinematic case in Paper III, a different mechanism must be
  identified --- the discrete causal cone, or the budget cost of
  maintaining a coherent translating excitation, or both --- and
  the resulting functional form derived rather than postulated.
- The product-vs-additive prediction follows automatically once
  the two factors are derived independently. Whether the rule
  predicts the product form or some other combination should be
  verified by simulation, not assumed.

---

## 6. Distinguishing prediction (product vs additive time-dilation)

**Old location:** original Paper II, Section 4 ("Gravitational and
Kinematic Time Dilation"), Subsection "The product form for joint
motion".

**Old content:**
- $(d\tau/dt)_{\rm DDD} = \sqrt{1 - 2GM/c^2 r} \cdot \sqrt{1 - \beta^2}$
- vs $(d\tau/dt)_{\rm GR} \approx \sqrt{1 - 2GM/c^2 r - \beta^2}$
- Agree to first order, differ at second order.
- Quantitative deviation: $0.21\%$ at $\beta = 0.30, r = 5A$ to
  $10.27\%$ at $\beta = 0.90, r = 5A$.

**New destination:** Paper III, together with the kinematic factor
on which it depends (item 5 above).

**Status:** ⏳ Removed from Paper II v7.1, which now restricts
itself to the gravitational sector and the bandwidth--clock
identification. The product-form distinguishing prediction is
deferred to Paper III, conditional on Paper III deriving the
kinematic factor from the rule rather than postulating it.

---

## Files preserved in `code/` and `data/` for re-use

The Python scripts and JSON datasets associated with the relocated
content remain in `paperII_gravity/code/` and `paperII_gravity/data/`
and will be referenced by the destination papers (or moved to their
folders when those are written).

| File | Original purpose | Destination paper |
|------|------------------|-------------------|
| `03_photon_deflection.py`, `data/photon_deflection.json` | Ray-tracing photon deflection | Paper IV |
| `05_nonlinear_feedback.py`, `data/feedback_results.json` | Self-consistent quadratic feedback | Paper IV |
| `06_quadratic_feedback.py`, `data/quadratic_results.json` | Quadratic feedback continuum | Paper IV |
| `07_path_B_diffusion.py`, `data/pathB_results.json` | Diffusion path | Paper IV |
| `08_yukawa_bounds.py`, `data/yukawa_bounds.json` | Eöt-Wash bound on $\lambda_{\rm phys}$ | Paper IV |
| `02_time_dilation.py`, `data/time_dilation.json` | Time-dilation simulation | Paper II v7 (kept) |

The pre-v7 scripts were marked with the `[v7-DEPRECATED CONVENTION]`
header during the Paper I v7 cleanup and remain syntactically valid
for reference.

---

## 7. Single-throttle as the prescription supporting cascade propagation

**Origin of this note:** discussion during the rewrite of Paper II
into its current form ("Cascade-Throttled Drainage and the Emergence
of the Schwarzschild Time-Dilation Factor"). The single-throttle
modification was selected for the gravitational time-dilation
derivation. A natural follow-up question is whether the same
prescription is the one that supports propagating excitations and
inertia in dynamical regimes.

**Old content (none in original Paper II series; this is a new
observation):**

The flux prescription $F_{i \to j} = \alpha (R_i/R_0)(R_i - R_j)_+$
(single throttle, source-side only) has two properties relevant to
dynamical settings:

1. **Cascade preservation.** A node with reduced reserve still
   accepts flux from full neighbours without any additional
   throttling on the receiver side. The cascade therefore
   propagates through low-$R$ regions without freezing. Compare
   with the double-throttle prescription, where the cascade
   collapses near the sink (verified numerically in Paper II,
   Sec. 6.4): the double throttle creates an artificial horizon
   at the first shell beyond the sink because the link to the
   first-shell neighbour, which has low bandwidth, is itself
   suppressed quadratically.

2. **Directional asymmetry between full and empty.** The rule
   already had this structural property at the level of Paper I
   (no-deadlock, the full fills the void). The single-throttle
   reinforces it: a full node retains its full pushing capacity
   regardless of whether its neighbours are drained, and a
   drained node retains its full receiving capacity regardless
   of whether its neighbours are full. The "stuff flows from
   full to empty" sense of the rule is preserved at all reserve
   levels.

**Conjecture (to be tested in Paper III):**

In dynamical regimes (translating excitations, propagating
disturbances), the single-throttle is the prescription that allows
a localised excitation to maintain its identity across the
substrate, while the double-throttle would dissipate it
quadratically. Whether the asymmetry of the single-throttle is the
structural support that enables effective inertia and kinematic
clock-rate suppression of moving configurations is open.

**Status:** ⏳ To be tested numerically in Paper III, by running a
translating Gaussian excitation under linear, single-throttle, and
double-throttle prescriptions and measuring the persistence,
effective dispersion, and propagation speed of each.

**Note:** the rule of Paper I as written does not contain a
second-time-derivative structure; oscillation and inertia in the
strict (mechanical) sense require additional structure (likely
from the gauge sector $(T, I, \psi)$ of Paper I, deferred to
Paper V). The single-throttle is a necessary but not sufficient
condition for propagating dynamics.

**Destination:** Paper III, alongside the cosmogenetic scenario
based on the no-deadlock property.
