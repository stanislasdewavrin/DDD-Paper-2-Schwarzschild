# Paper II — Discrete Drainage Dynamics: Cascade-Throttled Drainage and the Emergence of the Schwarzschild Time-Dilation Factor

Manuscript and reproducible simulation code for **Paper II** of the DDD series:

> *Discrete Drainage Dynamics II: Cascade-Throttled Drainage and the
> Emergence of the Schwarzschild Time-Dilation Factor*

This paper specialises the local rule of Paper I to the source-side
bandwidth-throttled mobility prescription $\mu(R) = R/R_0$ and shows
that the steady-state field around a point sink takes the form
$R/R_0 = \sqrt{1 - \beta/r}$, which under calibration matches the
Schwarzschild time-dilation factor at all orders in the continuum
limit. We verify the result numerically on a cubic substrate.

## Contents

```
paperII_gravity/
├── README.md                       ← This file
├── RELOCATED_CONTENT.md            ← Audit trail of content moved to other papers
├── paper.tex                       ← LaTeX source
├── paper.pdf                       ← Compiled manuscript (10 pages)
├── references.bib                  ← BibTeX bibliography
├── .gitignore
├── code/
│   ├── iterative_cascade_test.py   Cascade test: linear / single / double throttle
│   ├── bandwidth_clock.py          Linear-baseline radial profile measurement
│   └── (legacy pre-v7 scripts marked DEPRECATED in their headers)
├── data/                           Numerical outputs (JSON)
│   ├── iterative_cascade_L32.json
│   ├── bandwidth_clock_L24_E0.3.json
│   └── (other data files referenced in the paper)
└── figures/                        Generated figures (PDF + PNG)
    ├── fig1_reserve_bandwidth_proper_time.pdf
    ├── fig2_radial_deficit_bandwidth.pdf
    ├── fig3_ddd_vs_schwarzschild.pdf
    └── fig4_iterative_cascade.pdf
```

## What this paper claims

Three claims (Section 1 of the paper):

1. **Bandwidth identification.** Within the rule of Paper I, the
   maximum directional flux a node can push is $\alpha R_i$. We
   identify the dimensionless ratio $R_i/R_0$ with the local rate of
   effective proper time, $d\tau/dt = R_i/R_0$. This is an
   operational hypothesis, not derived from the rule.

2. **Single-throttle modification.** We fix the mobility function
   left open in Paper I to the source-side bandwidth-throttled choice
   $\mu(R) = R/R_0$, giving $F_{i \to j} = \alpha (R_i/R_0)(R_i -
   R_j)_+$. The receiver imposes no additional throttle.

3. **Schwarzschild form, derived.** In the continuum limit, the
   steady-state field of the throttled rule satisfies
   $\nabla^2(R^2) = 0$ away from the source. The unique spherically
   symmetric solution with $R \to R_0$ at infinity is
   $R/R_0 = \sqrt{1 - \beta/r}$, exactly the Schwarzschild
   time-dilation factor under the calibration $\beta = 2GM/c^2$. The
   match holds at all orders, not just leading order.

## What this paper does not claim

- We do not derive the full Schwarzschild metric (only $g_{00}$).
- We do not derive the Einstein field equations.
- We do not derive Newton's $G$ or the speed of light $c$.
- We do not derive the bandwidth identification or the single-throttle
  prescription; both are physically motivated choices.
- We do not address kinematic clock-rate effects (deferred to Paper III).
- We do not show the rule is unique; other mobility prescriptions
  yield different non-linear regimes.

## Quick start

```bash
# Linear baseline radial profile (Table 1):
python code/bandwidth_clock.py --L 24 --n_ticks 3000 --E0 0.3

# Cascade test (Tables 2 and 3, Figure 4):
python code/iterative_cascade_test.py
```

The cascade test runs three flux prescriptions (no throttle,
single-throttle, double-throttle) and produces the radial deficit
profiles compared to the Schwarzschild target. The single-throttle
matches Schwarzschild to within lattice resolution; the
double-throttle collapses the cascade near the sink (a structural
pathology that distinguishes the two prescriptions).

## Build the manuscript

```bash
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

## Required LaTeX packages

Standard TeX Live: amsmath, amssymb, amsthm, graphicx, hyperref,
xcolor, booktabs, authblk, geometry, cite.

## Reproducibility

All scripts are pure-numpy with no scipy dependency. The cascade
simulation at $L = 32$, $E_0 = 0.5$ runs in about 30 seconds per
prescription. Larger lattices ($L = 64$ with $E_0 = 1.5$) take a
few minutes per prescription and reveal the second-order agreement
with Schwarzschild more clearly. All scripts are deterministic given
the lattice size, source strength, and iteration count.

## Relationship to Paper I

Paper I (`DDD-Paper-1-Foundations`) introduces the rule with the
mobility function $\mu(R)$ as a structural choice and studies the
constant-mobility baseline $\mu \equiv 1$, recovering Newton's $1/r$
at leading order. Paper II fixes $\mu(R) = R/R_0$ and derives the
Schwarzschild form. The two papers are not different theories: they
are different choices within the same framework.

## License

- Simulation code in `code/`: MIT.
- Manuscript text and figures: standard arXiv terms.

## Contact

Stanislas Dewavrin — Independent Researcher — `dewavrin.iphone@gmail.com`
